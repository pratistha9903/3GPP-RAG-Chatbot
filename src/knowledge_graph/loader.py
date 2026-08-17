"""Load and query the GSMA telecom-kg-rel19 knowledge graph."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import networkx as nx
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from config.settings import get_settings


class KnowledgeGraphLoader:
    """Load the 3GPP Release 19 telecom knowledge graph from GraphML."""

    NS = {"g": "http://graphml.graphdrawing.org/xmlns"}

    # GraphML key id -> attribute name mapping (from GSMA dataset)
    NODE_ATTRS = {
        "d0": "id",
        "d1": "entity_type",
        "d2": "description",
        "d3": "chunk_id",
        "d4": "file_path",
        "d5": "created_at",
        "d6": "source_file",
        "d7": "release",
        "d8": "raw_entity_id",
    }
    EDGE_ATTRS = {
        "d9": "weight",
        "d10": "description",
        "d11": "predicate",
        "d12": "chunk_id",
        "d13": "file_path",
        "d14": "created_at",
        "d15": "conditions",
        "d16": "source_file",
        "d17": "release",
    }

    def __init__(self, kg_path: Path | None = None):
        settings = get_settings()
        self.kg_path = kg_path or settings.kg_path
        self.graph: nx.DiGraph = nx.DiGraph()
        self.node_index: dict[str, dict] = {}
        self.search_index: dict[str, set[str]] = {}  # term -> node_ids
        self.is_loaded = False

    def download(self) -> Path:
        """Download KG from HuggingFace if not present locally."""
        settings = get_settings()
        self.kg_path.parent.mkdir(parents=True, exist_ok=True)

        if self.kg_path.exists():
            return self.kg_path

        print(f"Downloading knowledge graph from {settings.hf_dataset}...")
        downloaded = hf_hub_download(
            settings.hf_dataset,
            "tkg/rel19_3gpp_telecom_kg.graphml",
            repo_type="dataset",
            local_dir=str(settings.data_dir),
        )
        return Path(downloaded)

    def load(self, max_nodes: int | None = None) -> None:
        """Parse GraphML and build in-memory graph + search index."""
        path = self.download() if not self.kg_path.exists() else self.kg_path
        if not path.exists():
            raise FileNotFoundError(f"Knowledge graph not found: {path}")

        print(f"Loading knowledge graph from {path}...")
        tree = ET.parse(path)
        root = tree.getroot()

        nodes = root.findall(".//g:node", self.NS)
        if max_nodes:
            nodes = nodes[:max_nodes]

        for node_elem in tqdm(nodes, desc="Loading nodes"):
            node_id = node_elem.get("id", "")
            attrs = self._parse_attrs(node_elem, self.NODE_ATTRS)
            attrs["id"] = node_id
            self.graph.add_node(node_id, **attrs)
            self.node_index[node_id] = attrs
            self._index_node(node_id, attrs)

        node_ids = {n.get("id") for n in nodes}
        edges = root.findall(".//g:edge", self.NS)
        for edge_elem in tqdm(edges, desc="Loading edges"):
            src = edge_elem.get("source", "")
            tgt = edge_elem.get("target", "")
            if src not in node_ids or tgt not in node_ids:
                continue
            attrs = self._parse_attrs(edge_elem, self.EDGE_ATTRS)
            attrs["source"] = src
            attrs["target"] = tgt
            self.graph.add_edge(src, tgt, **attrs)

        self.is_loaded = True
        print(f"Loaded {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def _parse_attrs(self, elem, attr_map: dict) -> dict:
        result = {}
        for data_elem in elem.findall("g:data", self.NS):
            key = data_elem.get("key", "")
            if key in attr_map:
                result[attr_map[key]] = data_elem.text or ""
        return result

    def _index_node(self, node_id: str, attrs: dict) -> None:
        terms = set()
        terms.add(node_id.lower())
        for field in ("description", "entity_type"):
            text = attrs.get(field, "")
            if text:
                for token in re.findall(r"[a-z0-9]+", text.lower()):
                    if len(token) > 2:
                        terms.add(token)
        for term in terms:
            self.search_index.setdefault(term, set()).add(node_id)

    def search_nodes(self, terms: list[str], top_k: int = 10) -> list[tuple[str, float]]:
        """Search nodes by term overlap scoring."""
        scores: dict[str, float] = {}
        for term in terms:
            term_lower = term.lower()
            if term_lower in self.search_index:
                for node_id in self.search_index[term_lower]:
                    scores[node_id] = scores.get(node_id, 0) + 1.0
            for indexed_term, node_ids in self.search_index.items():
                if term_lower in indexed_term or indexed_term in term_lower:
                    for node_id in node_ids:
                        scores[node_id] = scores.get(node_id, 0) + 0.5

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_node(self, node_id: str) -> dict | None:
        return self.node_index.get(node_id)

    def get_edges_for_node(self, node_id: str) -> list[dict]:
        edges = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            edges.append({
                "source": node_id,
                "target": target,
                "predicate": data.get("predicate", "related_to"),
                "description": data.get("description", ""),
                "chunk_id": data.get("chunk_id", ""),
                "source_file": data.get("source_file", ""),
                "release": data.get("release", "Rel-19"),
                "weight": float(data.get("weight", 1.0) or 1.0),
            })
        for source, _, data in self.graph.in_edges(node_id, data=True):
            edges.append({
                "source": source,
                "target": node_id,
                "predicate": data.get("predicate", "related_to"),
                "description": data.get("description", ""),
                "chunk_id": data.get("chunk_id", ""),
                "source_file": data.get("source_file", ""),
                "release": data.get("release", "Rel-19"),
                "weight": float(data.get("weight", 1.0) or 1.0),
            })
        return edges

    def get_indexable_nodes(self, max_nodes: int | None = None) -> list[dict]:
        """Return nodes with non-empty descriptions for indexing."""
        nodes = []
        for node_id, attrs in self.node_index.items():
            desc = attrs.get("description", "")
            if desc and len(desc.strip()) >= 20:
                nodes.append({
                    "id": node_id,
                    "description": desc,
                    "entity_type": attrs.get("entity_type", ""),
                    "chunk_id": attrs.get("chunk_id", ""),
                    "source_file": attrs.get("source_file", ""),
                    "release": attrs.get("release", "Rel-19"),
                })
            if max_nodes and len(nodes) >= max_nodes:
                break
        return nodes

    def verify_entity_exists(self, entity_name: str) -> bool:
        """Check if an entity exists in the knowledge graph."""
        name_lower = entity_name.lower()
        if entity_name in self.node_index:
            return True
        return name_lower in self.search_index

    def verify_clause_reference(self, spec_id: str, clause: str) -> bool:
        """Check if a spec/clause reference appears in the KG."""
        for attrs in self.node_index.values():
            sf = attrs.get("source_file", "")
            if spec_id.replace(".", "") in sf.replace(".", ""):
                return True
        return False

    @staticmethod
    def extract_spec_from_path(path: str) -> str:
        match = re.search(r"/(\d{5})-", path.replace("\\", "/"))
        if match:
            num = match.group(1)
            return f"{num[:2]}.{num[2:]}"
        match = re.search(r"(\d{2})\.(\d{3})", path)
        if match:
            return f"{match.group(1)}.{match.group(2)}"
        return "unknown"
