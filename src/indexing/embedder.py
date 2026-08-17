"""Embedding generation for document chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import get_settings


class Embedder:
    """Wrapper around sentence-transformers for batch embedding."""

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(
        self, texts: Sequence[str], batch_size: int = 64, show_progress: bool = True
    ) -> np.ndarray:
        """Embed a list of texts, returning (N, dim) array."""
        if not texts:
            return np.array([]).reshape(0, self.dimension)
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.embed_texts([query], show_progress=False)[0]

    def save_embeddings(self, embeddings: np.ndarray, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), embeddings)

    def load_embeddings(self, path: Path) -> np.ndarray:
        return np.load(str(path))

    def save_metadata(self, chunks: list[dict], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    def load_metadata(self, path: Path) -> list[dict]:
        chunks = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks
