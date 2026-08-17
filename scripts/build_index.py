"""Build search index with Hugging Face GSMA/3GPP dataset as PRIMARY source."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from src.indexing.hf_loader import HFCorpusLoader
from src.indexing.indexer import DocumentIndexer
from src.indexing.spec_loader import load_kg_supplement_chunks
from src.knowledge_graph.loader import KnowledgeGraphLoader


def main():
    settings = get_settings()
    print("=" * 60)
    print("3GPP KG-RAG Index Builder")
    print("Primary source: GSMA/telecom-kg-rel19 (Hugging Face)")
    print("=" * 60)

    indexer = DocumentIndexer()
    all_chunks = []

    print("\n[1/2] Loading PRIMARY 3GPP corpus from Hugging Face...")
    hf_loader = HFCorpusLoader()
    max_hf = settings.demo_hf_chunks if settings.demo_mode else None
    hf_chunks = hf_loader.load_primary_corpus(max_chunks=max_hf)
    print(f"  HF 3GPP chunks: {len(hf_chunks)}")
    all_chunks.extend(hf_chunks)

    kg_chunks = []
    if settings.index_kg_supplement:
        print("\n[2/2] Indexing KG supplement for relationship queries...")
        kg_loader = KnowledgeGraphLoader()
        max_nodes = settings.demo_kg_nodes if settings.demo_mode else None
        try:
            kg_loader.load(max_nodes=max_nodes)
            kg_chunks = load_kg_supplement_chunks(kg_loader, max_nodes=max_nodes)
            print(f"  KG supplement chunks: {len(kg_chunks)}")
            all_chunks.extend(kg_chunks)
        except FileNotFoundError:
            print("  KG not found. Run: python scripts/download_data.py")

    if not all_chunks:
        print("ERROR: No chunks to index. Run: python scripts/download_data.py")
        sys.exit(1)

    print(f"\nIndex composition:")
    print(f"  HF 3GPP (primary):  {len(hf_chunks)}")
    print(f"  KG (supplement):    {len(kg_chunks)}")
    print(f"  Total:              {len(all_chunks)}")

    indexer.build_from_chunks(all_chunks)
    print("\nIndex build complete!")


if __name__ == "__main__":
    main()
