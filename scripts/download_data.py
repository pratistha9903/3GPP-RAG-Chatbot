#!/usr/bin/env python3
"""
Download the GSMA/telecom-kg-rel19 3GPP dataset from Hugging Face.

Primary knowledge source:
  https://huggingface.co/datasets/GSMA/telecom-kg-rel19

This is a publicly available mirror of official 3GPP Release 19 specifications,
maintained as part of the GSMA Open Telco Assets Initiative.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from src.indexing.hf_loader import HFCorpusLoader, HF_DATASET_ID


def main():
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("3GPP Primary Corpus Download (Hugging Face)")
    print("=" * 60)
    print(f"Dataset: {HF_DATASET_ID}")
    print("Description: GSMA mirror of official 3GPP Release 19 specifications")
    print(f"Output:    {settings.data_dir}")
    print()

    loader = HFCorpusLoader()
    kg_path, chunks_path = loader.ensure_downloaded()

    print(f"\nDownloaded files:")
    print(f"  Knowledge graph:  {kg_path}")
    print(f"  Chunk metadata:   {chunks_path}")

    from huggingface_hub import hf_hub_download
    entities_path = hf_hub_download(
        HF_DATASET_ID,
        "mappings/entities.json",
        repo_type="dataset",
        local_dir=str(settings.data_dir),
    )
    print(f"  Entity mappings:  {entities_path}")

    # Preview corpus size
    print("\nPreviewing reconstructable 3GPP text from dataset...")
    corpus = loader.load_primary_corpus(max_chunks=100)
    specs = sorted({c.spec_id for c in corpus if c.spec_id != "unknown"})
    print(f"  Sample chunks:    {len(corpus)}")
    print(f"  Sample specs:     {specs[:10]}{'...' if len(specs) > 10 else ''}")

    print("\nNext step: python scripts/build_index.py")
    print("Download complete!")


if __name__ == "__main__":
    main()
