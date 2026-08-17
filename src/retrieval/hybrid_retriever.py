"""Hybrid retrieval combining BM25 and dense vector search with RRF fusion."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from config.settings import get_settings
from src.indexing.embedder import Embedder
from src.indexing.indexer import DocumentIndexer
from src.indexing.spec_loader import SOURCE_TYPE_HF_3GPP, is_primary_spec_chunk


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    rank: int
    source: str  # bm25 | vector | hybrid
    metadata: dict

    def citation(self) -> str:
        spec = self.metadata.get("spec_id", "unknown")
        release = self.metadata.get("release", "Rel-19")
        clause = self.metadata.get("clause", "")
        if clause:
            return f"TS {spec} ({release}), Clause {clause}"
        return f"TS {spec} ({release})"

    @property
    def is_primary_spec(self) -> bool:
        return is_primary_spec_chunk(self.metadata)


class HybridRetriever:
    """
    Hybrid retriever using BM25 + dense vector search fused via Reciprocal Rank Fusion (RRF).
    Applies priority boost to primary 3GPP specification chunks.
    """

    def __init__(self, indexer: DocumentIndexer | None = None):
        settings = get_settings()
        self.indexer = indexer or DocumentIndexer()
        if not self.indexer.chunks:
            self.indexer.load()
        self.embedder = Embedder()
        self.top_k = settings.top_k_retrieval
        self.rrf_k = settings.rrf_k
        self.spec_boost = settings.spec_source_boost

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        primary_only: bool = False,
    ) -> list[RetrievalResult]:
        k = top_k or self.top_k
        search_query = self._expand_query(query)
        bm25_results = self._bm25_search(search_query, k * 3)
        vector_results = self._vector_search(search_query, k * 3)
        fused = self._reciprocal_rank_fusion(bm25_results, vector_results, k * 2)

        if primary_only:
            fused = [r for r in fused if r.is_primary_spec]

        return fused[:k]

    @staticmethod
    def _expand_query(query: str) -> str:
        """Expand everyday language to 3GPP terms for better semantic retrieval."""
        additions: list[str] = []
        query_lower = query.lower()
        expansions = {
            r"\bphone\b": "UE user equipment mobile device",
            r"\bconnect(?:s|ion|ing)?\b": "connection establishment attach register camp RRC",
            r"\bnetwork\b": "cell RAN LTE NR 3GPP PLMN serving network selection",
            r"\btower\b": "gNB eNB base station cell",
            r"\binternet\b": "PDN session bearer data UPF",
            r"\bsim\b": "USIM IMSI subscription",
        }
        for pattern, extra in expansions.items():
            if re.search(pattern, query_lower):
                additions.append(extra)
        if additions:
            return f"{query} {' '.join(additions)}"
        return query

    def retrieve_primary_specs(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve only from primary 3GPP specification documents."""
        settings = get_settings()
        k = top_k or settings.top_k_spec_retrieval
        results = self.retrieve(query, top_k=k * 3, primary_only=False)

        spec_results = [r for r in results if r.is_primary_spec]
        spec_results = [
            r for r in spec_results
            if r.metadata.get("spec_id") not in ("unknown", "", None)
        ]
        kg_results = [r for r in results if not r.is_primary_spec]

        # Primary 3GPP specs first, KG supplement last
        return (spec_results + kg_results)[:k]

    def enrich_with_parent_context(
        self, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Add parent section context to child chunks for richer grounding."""
        enriched = []
        seen_ids: set[str] = set()

        for result in results:
            if result.chunk_id in seen_ids:
                continue
            seen_ids.add(result.chunk_id)

            parent_text = self.get_parent_context(result)
            if parent_text and not result.metadata.get("is_parent"):
                combined = RetrievalResult(
                    chunk_id=result.chunk_id,
                    text=f"{parent_text}\n\n---\n\n{result.text}",
                    score=result.score,
                    rank=result.rank,
                    source=result.source,
                    metadata={**result.metadata, "enriched": True},
                )
                enriched.append(combined)
            else:
                enriched.append(result)

        return enriched

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        tokens = DocumentIndexer._tokenize(query)
        scores = self.indexer.bm25.get_scores(tokens)
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in ranked if scores[idx] > 0]

    def _vector_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        query_emb = self.embedder.embed_query(query)
        similarities = self.indexer.embeddings @ query_emb
        ranked = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in ranked]

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[tuple[int, float]],
        vector_results: list[tuple[int, float]],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Fuse rankings using RRF with 3GPP spec source boost."""
        rrf_scores: dict[int, float] = {}
        sources: dict[int, set[str]] = {}

        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (self.rrf_k + rank + 1)
            sources.setdefault(idx, set()).add("bm25")

        for rank, (idx, _) in enumerate(vector_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (self.rrf_k + rank + 1)
            sources.setdefault(idx, set()).add("vector")

        # Boost primary 3GPP specification chunks
        for idx in list(rrf_scores.keys()):
            chunk = self.indexer.chunks[idx]
            if chunk.get("source_type") == SOURCE_TYPE_HF_3GPP or chunk.get("authority") == "primary":
                rrf_scores[idx] *= self.spec_boost

        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        results = []
        for rank, idx in enumerate(sorted_indices[:top_k]):
            chunk = self.indexer.chunks[idx]
            source = "hybrid" if len(sources[idx]) > 1 else list(sources[idx])[0]
            results.append(
                RetrievalResult(
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    score=rrf_scores[idx],
                    rank=rank + 1,
                    source=source,
                    metadata={
                        k: v
                        for k, v in chunk.items()
                        if k not in ("chunk_id", "text")
                    },
                )
            )
        return results

    def get_parent_context(self, child_result: RetrievalResult) -> str | None:
        """Fetch parent chunk text for a child chunk (parent-child enrichment)."""
        parent_id = child_result.metadata.get("parent_id")
        if not parent_id:
            return None
        for chunk in self.indexer.chunks:
            if chunk.get("chunk_id") == parent_id:
                return chunk["text"]
        return None
