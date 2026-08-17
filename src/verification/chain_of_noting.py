"""Chain-of-Noting: reject answers when retrieved info is insufficient."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from config.settings import get_settings
from src.generation.llm_client import BaseLLM, get_llm
from src.generation.prompts import CHAIN_OF_NOTING_SYSTEM, CHAIN_OF_NOTING_USER
from src.indexing.spec_loader import is_primary_spec_chunk
from src.retrieval.hybrid_retriever import RetrievalResult
from src.retrieval.kg_retriever import Triple

# Map everyday terms to 3GPP/telecom vocabulary for indirect questions
QUERY_SYNONYMS: dict[str, set[str]] = {
    "phone": {"ue", "mobile", "device", "user", "equipment", "terminal"},
    "connect": {"connection", "establish", "attach", "register", "camp", "selection", "association"},
    "network": {"cell", "ran", "lte", "nr", "3gpp", "plmn", "serving", "gnb", "enb"},
    "internet": {"data", "pdn", "session", "bearer", "upf", "smf"},
    "sim": {"usim", "imsi", "subscription", "credential"},
    "tower": {"gnb", "enb", "base", "station", "cell"},
    "call": {"voice", "cs", "ims", "sip", "bearer"},
}

QUERY_STOP_WORDS = {
    "what", "is", "the", "a", "an", "how", "does", "do", "of", "in", "for",
    "and", "or", "to", "are", "between", "explain", "describe", "work", "works",
}


@dataclass
class NotingResult:
    can_answer: bool
    reasoning: str
    missing_information: list[str]
    confidence: float


class ChainOfNoting:
    """
    Chain-of-Noting (CoN) evaluator.
    In strict mode, requires primary 3GPP specification evidence.
    """

    def __init__(self, llm: BaseLLM | None = None):
        self.llm = llm or get_llm()
        self.settings = get_settings()

    def evaluate(
        self,
        query: str,
        chunks: list[RetrievalResult],
        triples: list[Triple],
    ) -> NotingResult:
        spec_chunks = [c for c in chunks if is_primary_spec_chunk(c.metadata)]

        if self.settings.require_3gpp_spec_evidence and not spec_chunks:
            return NotingResult(
                can_answer=False,
                reasoning="No primary 3GPP specification chunks retrieved.",
                missing_information=["Matching 3GPP specification text"],
                confidence=0.0,
            )

        if not spec_chunks and not triples:
            return NotingResult(
                can_answer=False,
                reasoning="No relevant context was retrieved from the knowledge base.",
                missing_information=["Matching 3GPP specification content"],
                confidence=0.0,
            )

        chunks_summary = self._summarize_chunks(spec_chunks or chunks)
        triples_summary = self._summarize_triples(triples)

        user_prompt = CHAIN_OF_NOTING_USER.format(
            query=query,
            num_chunks=len(spec_chunks or chunks),
            chunks_summary=chunks_summary,
            num_triples=len(triples),
            triples_summary=triples_summary,
        )

        response = self.llm.generate(CHAIN_OF_NOTING_SYSTEM, user_prompt, temperature=0.0)
        result = self._parse_response(response)

        retrieval_conf = self._retrieval_confidence(query, spec_chunks or chunks)
        if retrieval_conf >= 0.45 and (spec_chunks or chunks):
            result.can_answer = True
            result.confidence = max(result.confidence, retrieval_conf)
            if not result.reasoning:
                result.reasoning = "Retrieved 3GPP chunks contain relevant context for this query."

        # Strict: downgrade confidence if only KG supplement, no primary spec
        if self.settings.require_3gpp_spec_evidence and not spec_chunks:
            result.can_answer = False
            result.confidence = 0.0
            result.missing_information.append("Primary 3GPP specification text required")

        if result.confidence < self.settings.min_con_confidence:
            result.can_answer = False

        return result

    def _retrieval_confidence(
        self, query: str, chunks: list[RetrievalResult]
    ) -> float:
        """Rule-based confidence from retrieval relevance (supports indirect questions)."""
        if not chunks:
            return 0.0

        query_terms = set(re.findall(r"[a-z0-9]+", query.lower())) - QUERY_STOP_WORDS
        expanded_terms = set(query_terms)
        for term in query_terms:
            expanded_terms |= QUERY_SYNONYMS.get(term, set())

        overlaps: list[float] = []
        for chunk in chunks[:5]:
            text_terms = set(re.findall(r"[a-z0-9]+", chunk.text.lower()))
            if not expanded_terms:
                continue
            overlaps.append(len(expanded_terms & text_terms) / len(expanded_terms))

        term_overlap = max(overlaps) if overlaps else 0.0
        top_score = max(c.score for c in chunks)
        score_factor = min(top_score * 15, 1.0)

        return round(0.6 * term_overlap + 0.4 * score_factor, 2)

    def _summarize_chunks(self, chunks: list[RetrievalResult]) -> str:
        parts = []
        for i, c in enumerate(chunks[:5], 1):
            src = "PRIMARY 3GPP" if is_primary_spec_chunk(c.metadata) else "SUPPLEMENT"
            parts.append(f"{i}. [{src} {c.citation()}] {c.text[:200]}...")
        return "\n".join(parts) if parts else "None"

    def _summarize_triples(self, triples: list[Triple]) -> str:
        parts = [f"{i}. {t.to_text()[:150]}" for i, t in enumerate(triples[:5], 1)]
        return "\n".join(parts) if parts else "None"

    def _parse_response(self, response: str) -> NotingResult:
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return NotingResult(
                    can_answer=bool(data.get("can_answer", False)),
                    reasoning=data.get("reasoning", ""),
                    missing_information=data.get("missing_information", []),
                    confidence=float(data.get("confidence", 0.5)),
                )
        except (json.JSONDecodeError, ValueError):
            pass

        return NotingResult(
            can_answer=False,
            reasoning="Could not evaluate context sufficiency",
            missing_information=["CoN evaluation failed"],
            confidence=0.0,
        )
