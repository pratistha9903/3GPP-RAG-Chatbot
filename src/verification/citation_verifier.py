"""Post-generation citation verification – strict mode for zero hallucination."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from config.settings import get_settings
from src.knowledge_graph.loader import KnowledgeGraphLoader


@dataclass
class Citation:
    spec_id: str
    release: str = "Rel-19"
    clause: str = ""
    raw: str = ""


@dataclass
class CitationVerificationResult:
    valid_citations: list[Citation] = field(default_factory=list)
    invalid_citations: list[Citation] = field(default_factory=list)
    hallucinated: bool = False
    details: list[str] = field(default_factory=list)


class CitationVerifier:
    """
    Verify citations exist ONLY in retrieved 3GPP context.
    In strict mode, any unverifiable citation = hallucination.
    """

    CITATION_PATTERN = re.compile(
        r"\[TS\s+(?P<spec>\d+\.\d+)\s*(?:\((?P<release>[^)]+)\))?"
        r"(?:,\s*Clause\s*(?P<clause>[\d.]+))?\]",
        re.IGNORECASE,
    )

    def __init__(self, kg_loader: KnowledgeGraphLoader | None = None):
        self.kg = kg_loader
        self.settings = get_settings()

    def extract_citations(self, text: str) -> list[Citation]:
        citations = []
        for match in self.CITATION_PATTERN.finditer(text):
            citations.append(Citation(
                spec_id=match.group("spec"),
                release=match.group("release") or "Rel-19",
                clause=match.group("clause") or "",
                raw=match.group(0),
            ))
        return citations

    def verify(
        self,
        answer: str,
        available_citations: list[str] | None = None,
        available_clauses: list[tuple[str, str]] | None = None,
    ) -> CitationVerificationResult:
        """
        Verify citations against retrieved context only.
        available_clauses: list of (spec_id, clause) tuples from retrieved chunks.
        """
        citations = self.extract_citations(answer)
        result = CitationVerificationResult()

        available_specs: set[str] = set()
        available_clause_set: set[tuple[str, str]] = set()

        if available_citations:
            for cite in available_citations:
                spec_match = re.search(r"TS\s+(\d+\.\d+)", cite, re.IGNORECASE)
                if spec_match:
                    available_specs.add(spec_match.group(1))
                clause_match = re.search(r"Clause\s+([\d.]+)", cite, re.IGNORECASE)
                if spec_match and clause_match:
                    available_clause_set.add((spec_match.group(1), clause_match.group(1)))

        if available_clauses:
            available_clause_set.update(available_clauses)
            available_specs.update(s for s, _ in available_clauses)

        for citation in citations:
            is_valid = self._validate_citation_strict(
                citation, available_specs, available_clause_set
            )
            if is_valid:
                result.valid_citations.append(citation)
            else:
                result.invalid_citations.append(citation)
                result.details.append(
                    f"Hallucinated citation not in retrieved 3GPP context: {citation.raw}"
                )

        # In strict mode, answers with factual claims must have at least one valid citation
        if self.settings.strict_zero_hallucination and citations and not result.valid_citations:
            result.hallucinated = True
            result.details.append("No valid 3GPP citations found in answer")

        result.hallucinated = result.hallucinated or len(result.invalid_citations) > 0
        return result

    def _validate_citation_strict(
        self,
        citation: Citation,
        available_specs: set[str],
        available_clauses: set[tuple[str, str]],
    ) -> bool:
        """Citation is valid ONLY if it appears in retrieved context."""
        if not available_specs:
            return False

        if citation.spec_id not in available_specs:
            return False

        if citation.clause:
            # Clause must match a retrieved chunk or be a prefix match
            if (citation.spec_id, citation.clause) in available_clauses:
                return True
            # Allow if any retrieved clause starts with same prefix (section match)
            for spec, clause in available_clauses:
                if spec == citation.spec_id and (
                    clause == citation.clause
                    or clause.startswith(citation.clause)
                    or citation.clause.startswith(clause)
                ):
                    return True
            return False

        return True
