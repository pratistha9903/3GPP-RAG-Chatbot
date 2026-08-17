"""Build and persist the vector + BM25 index."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from config.settings import get_settings
from src.indexing.chunker import DocumentChunk, HierarchicalChunker
from src.indexing.embedder import Embedder


class DocumentIndexer:
    """Build hybrid retrieval index from chunks."""

    def __init__(self, index_dir: Path | None = None):
        settings = get_settings()
        self.index_dir = index_dir or settings.index_dir
        self.embedder = Embedder()
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None
        self.tokenized_corpus: list[list[str]] | None = None

    def build_from_chunks(self, chunks: list[DocumentChunk], batch_size: int = 64) -> None:
        """Index a list of DocumentChunk objects."""
        self.chunks = [c.to_dict() for c in chunks]
        texts = [c["text"] for c in self.chunks]

        print(f"Embedding {len(texts)} chunks...")
        self.embeddings = self.embedder.embed_texts(texts, batch_size=batch_size)

        print("Building BM25 index...")
        self.tokenized_corpus = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self._save()

    def build_from_kg_nodes(self, kg_loader, max_nodes: int | None = None) -> None:
        """Build index from knowledge graph node descriptions."""
        chunker = HierarchicalChunker()
        chunks: list[DocumentChunk] = []

        nodes = kg_loader.get_indexable_nodes(max_nodes=max_nodes)
        for node in tqdm(nodes, desc="Creating KG chunks"):
            chunk = chunker.chunk_from_kg_node(
                node_id=node["id"],
                description=node["description"],
                entity_type=node.get("entity_type", ""),
                chunk_id=node.get("chunk_id", ""),
                source_file=node.get("source_file", ""),
                release=node.get("release", "Rel-19"),
            )
            if chunk:
                chunks.append(chunk)

        print(f"Created {len(chunks)} chunks from KG nodes")
        self.build_from_chunks(chunks)

    def _save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder.save_embeddings(self.embeddings, self.index_dir / "embeddings.npy")
        self.embedder.save_metadata(self.chunks, self.index_dir / "chunks.jsonl")
        with open(self.index_dir / "bm25.pkl", "wb") as f:
            pickle.dump(
                {"bm25": self.bm25, "tokenized_corpus": self.tokenized_corpus}, f
            )
        meta = {
            "num_chunks": len(self.chunks),
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "embedding_model": self.embedder.model_name,
        }
        import json
        with open(self.index_dir / "index_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Index saved to {self.index_dir} ({len(self.chunks)} chunks)")

    def load(self) -> None:
        """Load persisted index from disk."""
        chunks_path = self.index_dir / "chunks.jsonl"
        embeddings_path = self.index_dir / "embeddings.npy"
        bm25_path = self.index_dir / "bm25.pkl"

        if not chunks_path.exists():
            raise FileNotFoundError(f"Index not found at {self.index_dir}. Run build_index.py first.")

        self.chunks = self.embedder.load_metadata(chunks_path)
        self.embeddings = self.embedder.load_embeddings(embeddings_path)
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.tokenized_corpus = data["tokenized_corpus"]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenizer preserving telecom acronyms."""
        import re
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+(?:[-_/][a-z0-9]+)*", text)
        return tokens

    @property
    def is_built(self) -> bool:
        return (self.index_dir / "chunks.jsonl").exists()
