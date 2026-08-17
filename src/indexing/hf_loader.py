"""
Load primary 3GPP corpus from GSMA/telecom-kg-rel19 on Hugging Face.

The dataset mirrors official 3GPP Release 19 specifications. Text is reconstructed
from KG node/edge descriptions (grounded to chunk IDs) and enriched with chunk
metadata from the dataset JSONL.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from config.settings import get_settings
from src.indexing.chunker import DocumentChunk, HierarchicalChunker

SOURCE_TYPE_HF_3GPP = "hf_3gpp_spec"
HF_DATASET_ID = "GSMA/telecom-kg-rel19"

NS = {"g": "http://graphml.graphdrawing.org/xmlns"}


class HFCorpusLoader:
    """
    Primary 3GPP knowledge loader from Hugging Face GSMA/telecom-kg-rel19.

    Reconstructs specification text chunks by aggregating KG node/edge descriptions
    that are grounded to the same chunk_id, with provenance from 3GPP source files.
    """

    NODE_KEYS = {
        "d0": "id",
        "d1": "entity_type",
        "d2": "description",
        "d3": "chunk_id",
        "d4": "file_path",
        "d6": "source_file",
        "d7": "release",
    }
    EDGE_KEYS = {
        "d10": "description",
        "d11": "predicate",
        "d12": "chunk_id",
        "d13": "file_path",
        "d16": "source_file",
        "d17": "release",
    }

    def __init__(self):
        self.settings = get_settings()

    def ensure_downloaded(self) -> tuple[Path, Path]:
        """Download KG and chunk metadata from Hugging Face if missing."""
        data_dir = self.settings.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        kg_path = self.settings.kg_path
        if not kg_path.exists():
            print(f"Downloading knowledge graph from {HF_DATASET_ID}...")
            kg_path = Path(
                hf_hub_download(
                    HF_DATASET_ID,
                    "tkg/rel19_3gpp_telecom_kg.graphml",
                    repo_type="dataset",
                    local_dir=str(data_dir),
                )
            )

        chunks_path = self.settings.chunks_path
        if not chunks_path.exists():
            print(f"Downloading chunk metadata from {HF_DATASET_ID}...")
            chunks_path = Path(
                hf_hub_download(
                    HF_DATASET_ID,
                    "chunks/rel19_text_chunks.jsonl",
                    repo_type="dataset",
                    local_dir=str(data_dir),
                )
            )

        return kg_path, chunks_path

    def load_primary_corpus(self, max_chunks: int | None = None) -> list[DocumentChunk]:
        """
        Build primary 3GPP document chunks from the Hugging Face dataset.
        """
        kg_path, chunks_path = self.ensure_downloaded()
        chunk_meta = self._load_chunk_metadata(chunks_path)
        chunk_texts = self._extract_texts_from_kg(kg_path)

        print(f"HF corpus: {len(chunk_texts)} chunk IDs with 3GPP text from KG grounding")

        chunker = HierarchicalChunker()
        documents: list[DocumentChunk] = []
        seen_text: set[str] = set()

        items = sorted(chunk_texts.items(), key=lambda x: len(x[1]), reverse=True)
        if max_chunks:
            items = items[:max_chunks]

        for chunk_id, segments in tqdm(items, desc="Building HF 3GPP chunks"):
            meta = chunk_meta.get(chunk_id, {})
            combined = self._dedupe_segments(segments)
            if len(combined.strip()) < 40:
                continue

            text_hash = combined[:200]
            if text_hash in seen_text:
                continue
            seen_text.add(text_hash)

            spec_id = self._resolve_spec_id(meta)
            release = meta.get("release", "Rel-19")
            source_file = meta.get("source_file") or meta.get("hf_source", "")
            ts = meta.get("ts", "")

            context = f"[{release} TS {spec_id}]"
            if ts:
                context += f" | Document {ts}"
            context += f" | HF chunk {chunk_id}"

            full_text = f"{context}\n\n{combined.strip()}"

            doc_chunk = DocumentChunk(
                chunk_id=f"hf-{chunk_id}",
                text=full_text,
                spec_id=spec_id,
                release=release,
                source_file=source_file,
                metadata={
                    "source_type": SOURCE_TYPE_HF_3GPP,
                    "authority": "primary",
                    "dataset": HF_DATASET_ID,
                    "chunk_id": chunk_id,
                    "ts": ts,
                    "series": meta.get("series", ""),
                    "hf_provenance": "GSMA/telecom-kg-rel19 (3GPP Rel-19 mirror)",
                },
            )
            documents.append(doc_chunk)

            # Also create hierarchical child chunks for long texts
            if len(combined) > 1200:
                sub_chunks = chunker.chunk_document(
                    text=combined,
                    spec_id=spec_id,
                    release=release,
                    source_file=source_file or f"hf://{HF_DATASET_ID}/{chunk_id}",
                )
                for sc in sub_chunks[1:]:  # skip duplicate parent
                    sc.chunk_id = f"hf-sub-{chunk_id}-{sc.chunk_id}"
                    sc.metadata.update({
                        "source_type": SOURCE_TYPE_HF_3GPP,
                        "authority": "primary",
                        "dataset": HF_DATASET_ID,
                        "parent_hf_chunk": chunk_id,
                        "hf_provenance": "GSMA/telecom-kg-rel19 (3GPP Rel-19 mirror)",
                    })
                    documents.append(sc)

        print(f"Created {len(documents)} primary chunks from Hugging Face 3GPP corpus")
        return documents

    def _load_chunk_metadata(self, chunks_path: Path) -> dict[str, dict]:
        """Load chunk ID → metadata mapping from HF JSONL."""
        meta: dict[str, dict] = {}
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                cid = obj.get("chunk_id", "")
                if cid:
                    meta[cid] = {
                        "release": obj.get("release", "Rel-19"),
                        "series": obj.get("series", ""),
                        "ts": obj.get("ts", ""),
                        "source_file": obj.get("source_file", ""),
                    }
        return meta

    def _extract_texts_from_kg(self, kg_path: Path) -> dict[str, list[str]]:
        """Aggregate node/edge descriptions by chunk_id from GraphML."""
        chunk_texts: dict[str, list[str]] = defaultdict(list)

        tree = ET.parse(kg_path)
        root = tree.getroot()

        for node in root.findall(".//g:node", NS):
            attrs = self._parse_elem(node, self.NODE_KEYS)
            desc = (attrs.get("description") or "").strip()
            cid = attrs.get("chunk_id", "")
            if cid and len(desc) >= 25:
                chunk_texts[cid].append(desc)
                if cid not in chunk_texts or not any(True for _ in []):
                    pass
                # enrich metadata later via jsonl

        for edge in root.findall(".//g:edge", NS):
            attrs = self._parse_elem(edge, self.EDGE_KEYS)
            desc = (attrs.get("description") or "").strip()
            cid = attrs.get("chunk_id", "")
            if cid and len(desc) >= 25:
                chunk_texts[cid].append(desc)

        return dict(chunk_texts)

    @staticmethod
    def _parse_elem(elem, key_map: dict) -> dict:
        result = {}
        for data in elem.findall("g:data", NS):
            key = data.get("key", "")
            if key in key_map:
                result[key_map[key]] = data.text or ""
        return result

    @staticmethod
    def _dedupe_segments(segments: list[str]) -> str:
        seen: set[str] = set()
        unique = []
        for seg in segments:
            norm = re.sub(r"\s+", " ", seg.strip())
            if norm and norm not in seen:
                seen.add(norm)
                unique.append(seg.strip())
        return "\n\n".join(unique)

    @staticmethod
    def _resolve_spec_id(meta: dict) -> str:
        """Derive TS number from HF chunk metadata."""
        ts = meta.get("ts", "")
        series = meta.get("series", "")
        source = meta.get("source_file", "")

        # ts format: 38331-j20 -> 38.331
        match = re.match(r"(\d{2})(\d{3})", ts.replace("-", "").replace("_", "")[:5])
        if match:
            return f"{match.group(1)}.{match.group(2)}"

        # source path: Rel-19/38_series/38331-j20/...
        match = re.search(r"/(\d{2})_series/(\d{5})", source.replace("\\", "/"))
        if match:
            num = match.group(2)
            return f"{num[:2]}.{num[2:]}"

        match = re.search(r"/(\d{5})-", source.replace("\\", "/"))
        if match:
            num = match.group(1)
            return f"{num[:2]}.{num[2:]}"

        if series and len(series) == 2:
            return f"{series}.xxx"

        return "unknown"
