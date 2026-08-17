"""Source type constants and KG supplement loader."""

from __future__ import annotations

from src.indexing.chunker import DocumentChunk, HierarchicalChunker

SOURCE_TYPE_HF_3GPP = "hf_3gpp_spec"
SOURCE_TYPE_KG = "kg_supplement"

PRIMARY_SOURCE_TYPES = {SOURCE_TYPE_HF_3GPP}


def is_primary_spec_chunk(metadata: dict) -> bool:
    """True if chunk comes from the Hugging Face 3GPP primary corpus."""
    return (
        metadata.get("source_type") in PRIMARY_SOURCE_TYPES
        or metadata.get("authority") == "primary"
    )


def load_kg_supplement_chunks(kg_loader, max_nodes: int | None = None) -> list[DocumentChunk]:
    """Create supplementary chunks from KG (secondary to HF primary corpus)."""
    chunker = HierarchicalChunker()
    chunks: list[DocumentChunk] = []

    for node in kg_loader.get_indexable_nodes(max_nodes=max_nodes):
        chunk = chunker.chunk_from_kg_node(
            node_id=node["id"],
            description=node["description"],
            entity_type=node.get("entity_type", ""),
            chunk_id=node.get("chunk_id", ""),
            source_file=node.get("source_file", ""),
            release=node.get("release", "Rel-19"),
        )
        if chunk:
            chunk.metadata["source_type"] = SOURCE_TYPE_KG
            chunk.metadata["authority"] = "supplement"
            chunk.metadata["dataset"] = "GSMA/telecom-kg-rel19"
            chunks.append(chunk)

    return chunks
