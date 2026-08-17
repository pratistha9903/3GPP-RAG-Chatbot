"""Tests for zero-hallucination and 3GPP-primary pipeline."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indexing.chunker import HierarchicalChunker
from src.indexing.spec_loader import SOURCE_TYPE_HF_3GPP
from src.generation.extractive_generator import ExtractiveAnswerBuilder
from src.retrieval.hybrid_retriever import RetrievalResult
from src.retrieval.query_router import QueryIntent, SemanticQueryRouter
from src.verification.citation_verifier import CitationVerifier
from src.verification.grounding_verifier import GroundingVerifier


class TestHierarchicalChunker:
    def test_chunk_document_preserves_structure(self):
        text = """# TS 38.300

5 NG-RAN architecture

5.1 General

The NG-RAN node interfaces with the 5GC via the NG interface.

5.3.1 RRC connection establishment

The UE initiates the procedure when upper layers request establishment.
"""
        chunker = HierarchicalChunker()
        chunks = chunker.chunk_document(text, spec_id="38.300")
        assert len(chunks) > 0
        assert any("5.3.1" in c.clause or "5.3.1" in c.title_path for c in chunks)


class TestPrimary3GPPSource:
    def test_hf_corpus_loader(self):
        from src.indexing.hf_loader import HFCorpusLoader, SOURCE_TYPE_HF_3GPP
        loader = HFCorpusLoader()
        chunks = loader.load_primary_corpus(max_chunks=50)
        assert len(chunks) > 10
        assert all(c.metadata.get("source_type") == SOURCE_TYPE_HF_3GPP for c in chunks)
        assert all(c.metadata.get("dataset") == "GSMA/telecom-kg-rel19" for c in chunks)
        assert all(c.metadata.get("authority") == "primary" for c in chunks)


class TestQueryRouter:
    def test_definition_intent(self):
        router = SemanticQueryRouter()
        result = router.route("What is RRC connection establishment?")
        assert result.intent == QueryIntent.DEFINITION


class TestExtractiveGenerator:
    def test_builds_verbatim_answer(self):
        chunk = RetrievalResult(
            chunk_id="c1",
            text=(
                "[Rel-19 TS 38.300] | Section 5.3.1\n\n"
                "5.3.1 RRC connection establishment\n\n"
                "The UE initiates the procedure when upper layers request establishment "
                "of an RRC connection while the UE has NAS signalling to be sent."
            ),
            score=1.0,
            rank=1,
            source="hybrid",
            metadata={
                "spec_id": "38.300",
                "release": "Rel-19",
                "clause": "5.3.1",
                "source_type": SOURCE_TYPE_HF_3GPP,
                "authority": "primary",
            },
        )
        builder = ExtractiveAnswerBuilder()
        result = builder.build("What is RRC connection establishment?", [chunk])
        assert result.grounded
        assert "38.300" in result.text
        assert "UE initiates" in result.body or "UE initiates" in result.text
        assert result.confidence > 0
        assert "### Answer" in result.text
        assert "### References" in result.text

    def test_rejects_irrelevant_query(self):
        chunk = RetrievalResult(
            chunk_id="c1",
            text="[Rel-19 TS 38.300]\n\nUnrelated content about something else entirely.",
            score=0.01,
            rank=1,
            source="hybrid",
            metadata={"spec_id": "38.300", "release": "Rel-19", "source_type": SOURCE_TYPE_HF_3GPP},
        )
        builder = ExtractiveAnswerBuilder()
        result = builder.build("quantum physics black holes", [chunk])
        assert not result.grounded


class TestCitationVerifier:
    def test_strict_rejects_unknown_spec(self):
        verifier = CitationVerifier()
        answer = "Defined in [TS 99.999 (Rel-19), Clause 1.1]"
        result = verifier.verify(answer, available_citations=["TS 38.300 (Rel-19)"])
        assert result.hallucinated

    def test_accepts_retrieved_spec(self):
        verifier = CitationVerifier()
        answer = "Defined in [TS 38.300 (Rel-19), Clause 5.3.1]"
        result = verifier.verify(
            answer,
            available_citations=["TS 38.300 (Rel-19), Clause 5.3.1"],
            available_clauses=[("38.300", "5.3.1")],
        )
        assert not result.hallucinated
        assert len(result.valid_citations) == 1


class TestGroundingVerifier:
    def test_detects_ungrounded_claim(self):
        verifier = GroundingVerifier()
        answer = "The AMF supports quantum teleportation of subscriber data. [TS 23.501 (Rel-19)]"
        context = ["The AMF supports NAS signalling termination and registration management."]
        result = verifier.verify(answer, context, min_score=0.45)
        assert not result.grounded

    def test_accepts_grounded_quote(self):
        verifier = GroundingVerifier()
        answer = (
            "The AMF supports NAS signalling termination and registration management. "
            "[TS 23.501 (Rel-19), Clause 4.3.1]"
        )
        context = [
            "The AMF supports NAS signalling termination and registration management."
        ]
        result = verifier.verify(answer, context, min_score=0.45)
        assert result.grounded


class TestRRF:
    def test_rrf_fusion(self):
        from src.retrieval.hybrid_retriever import HybridRetriever

        class MockIndexer:
            chunks = [
                {"chunk_id": "c0", "text": "chunk 0", "spec_id": "38.300", "source_type": "hf_3gpp_spec"},
                {"chunk_id": "c1", "text": "chunk 1", "spec_id": "38.300", "source_type": "hf_3gpp_spec"},
                {"chunk_id": "c2", "text": "chunk 2", "spec_id": "38.331", "source_type": "kg_supplement"},
            ]

        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.rrf_k = 60
        retriever.spec_boost = 3.0
        retriever.indexer = MockIndexer()
        bm25 = [(0, 1.0), (1, 0.8), (2, 0.5)]
        vector = [(1, 0.9), (2, 0.7), (0, 0.6)]
        results = retriever._reciprocal_rank_fusion(bm25, vector, top_k=3)
        assert len(results) <= 3
        assert results[0].is_primary_spec  # 3GPP spec should rank first due to boost
