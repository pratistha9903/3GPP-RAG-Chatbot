"""Application configuration loaded from environment variables."""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieval
    top_k_retrieval: int = 10
    top_k_spec_retrieval: int = 8
    top_k_kg_triples: int = 5
    rrf_k: int = 60
    bm25_weight: float = 0.5
    vector_weight: float = 0.5
    spec_source_boost: float = 3.0  # Boost 3GPP spec chunks over KG in ranking

    # Zero-hallucination mode (extractive answers only)
    strict_zero_hallucination: bool = True
    require_3gpp_spec_evidence: bool = True
    min_con_confidence: float = 0.75
    min_grounding_score: float = 0.45
    min_spec_chunks: int = 1
    reject_on_invalid_citation: bool = True
    reject_on_ungrounded_claim: bool = True

    # Paths
    data_dir: Path = Path("./data")
    index_dir: Path = Path("./data/index")
    kg_path: Path = Path("./data/tkg/rel19_3gpp_telecom_kg.graphml")

    # HuggingFace – PRIMARY 3GPP source
    hf_dataset: str = "GSMA/telecom-kg-rel19"
    demo_hf_chunks: int | None = None  # None = full HF corpus

    # Knowledge base scope
    demo_mode: bool = False
    demo_kg_nodes: int = 5000
    index_kg_supplement: bool = True

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks" / "rel19_text_chunks.jsonl"

    @property
    def entities_path(self) -> Path:
        return self.data_dir / "mappings" / "entities.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
