"""Ontology-aware retrieval from the 3GPP knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass

from src.knowledge_graph.loader import KnowledgeGraphLoader


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    description: str
    chunk_id: str
    source_file: str
    release: str
    weight: float = 1.0

    def to_text(self) -> str:
        return f"({self.subject}) --[{self.predicate}]--> ({self.object}): {self.description}"

    def citation(self) -> str:
        spec = KnowledgeGraphLoader.extract_spec_from_path(self.source_file)
        return f"TS {spec} ({self.release}) [KG Triple]"


class KGRetriever:
    """
    Ontology-aware retriever fetching atomic triples from the knowledge graph.
    Ensures schema-aligned, consistent facts for generation.
    """

    def __init__(self, kg_loader: KnowledgeGraphLoader | None = None):
        self.kg = kg_loader or KnowledgeGraphLoader()
        if not self.kg.is_loaded:
            self.kg.load()

    def retrieve_triples(
        self,
        query: str,
        keywords: list[str] | None = None,
        top_k: int = 10,
    ) -> list[Triple]:
        """Retrieve relevant KG triples for a query."""
        search_terms = keywords or self._extract_search_terms(query)
        if not search_terms:
            search_terms = query.lower().split()

        node_hits = self.kg.search_nodes(search_terms, top_k=top_k * 2)
        triples: list[Triple] = []
        seen: set[str] = set()

        for node_id, score in node_hits:
            for edge in self.kg.get_edges_for_node(node_id):
                triple_key = f"{edge['source']}|{edge['predicate']}|{edge['target']}"
                if triple_key in seen:
                    continue
                seen.add(triple_key)

                triples.append(
                    Triple(
                        subject=edge["source"],
                        predicate=edge["predicate"],
                        object=edge["target"],
                        description=edge.get("description", ""),
                        chunk_id=edge.get("chunk_id", ""),
                        source_file=edge.get("source_file", ""),
                        release=edge.get("release", "Rel-19"),
                        weight=score * edge.get("weight", 1.0),
                    )
                )

            node = self.kg.get_node(node_id)
            if node and node.get("description"):
                def_key = f"def|{node_id}"
                if def_key not in seen:
                    seen.add(def_key)
                    triples.append(
                        Triple(
                            subject=node_id,
                            predicate="defined_as",
                            object=node.get("entity_type", "Entity"),
                            description=node["description"],
                            chunk_id=node.get("chunk_id", ""),
                            source_file=node.get("source_file", ""),
                            release=node.get("release", "Rel-19"),
                            weight=score,
                        )
                    )

        triples.sort(key=lambda t: t.weight, reverse=True)
        return triples[:top_k]

    def retrieve_entities(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve matching KG entities with descriptions."""
        terms = self._extract_search_terms(query)
        hits = self.kg.search_nodes(terms, top_k=top_k)
        entities = []
        for node_id, score in hits:
            node = self.kg.get_node(node_id)
            if node:
                entities.append({**node, "score": score})
        return entities

    @staticmethod
    def _extract_search_terms(query: str) -> list[str]:
        import re
        stopwords = {"what", "is", "the", "a", "an", "how", "does", "do", "of", "in", "for", "and", "or"}
        terms = re.findall(r"[a-z0-9]+", query.lower())
        return [t for t in terms if t not in stopwords and len(t) > 2]
