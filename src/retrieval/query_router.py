"""Semantic query router to classify intent and route to appropriate index."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class QueryIntent(str, Enum):
    DEFINITION = "definition"       # "What is X?", "Define Y"
    PROCEDURE = "procedure"       # "How does X work?", "Steps for Y"
    PARAMETER = "parameter"       # "What is the value of X?", "Range of Y"
    COMPARISON = "comparison"     # "Difference between X and Y"
    REFERENCE = "reference"       # "Which spec covers X?", "TS for Y"
    RELATIONSHIP = "relationship" # "How is X related to Y?"
    GENERAL = "general"


@dataclass
class RouteDecision:
    intent: QueryIntent
    confidence: float
    index_hint: str  # text | kg | hybrid
    keywords: list[str]
    spec_filter: str | None = None


class SemanticQueryRouter:
    """
    Classify user query intent and route to the most relevant knowledge source.
    Prevents irrelevant context from poisoning the response.
    """

    INTENT_PATTERNS: dict[QueryIntent, list[str]] = {
        QueryIntent.DEFINITION: [
            r"\bwhat is\b", r"\bdefine\b", r"\bdefinition of\b",
            r"\bmeaning of\b", r"\bexplain\b", r"\bdescribe\b",
        ],
        QueryIntent.PROCEDURE: [
            r"\bhow (?:does|do|to)\b", r"\bprocedure\b", r"\bsteps?\b",
            r"\bprocess\b", r"\bflow\b", r"\bsequence\b", r"\binitiat",
        ],
        QueryIntent.PARAMETER: [
            r"\bvalue\b", r"\bparameter\b", r"\brange\b", r"\bthreshold\b",
            r"\btimer\b", r"\binterval\b", r"\bconfiguration\b",
        ],
        QueryIntent.COMPARISON: [
            r"\bdifference\b", r"\bcompare\b", r"\bvs\.?\b", r"\bversus\b",
            r"\bdistinction\b",
        ],
        QueryIntent.REFERENCE: [
            r"\bwhich spec\b", r"\bwhich ts\b", r"\bwhich document\b",
            r"\bwhere is\b.*\bspecif", r"\bts \d",
        ],
        QueryIntent.RELATIONSHIP: [
            r"\brelated to\b", r"\brelationship\b", r"\bconnect",
            r"\bassociated with\b", r"\bdepends on\b", r"\binteract",
        ],
    }

    TELECOM_ENTITIES = {
        "rrc", "nas", "pdcp", "rlc", "mac", "phy", "mib", "sib", "sib1",
        "amf", "smf", "upf", "gnb", "enb", "ue", "nr", "lte", "5gc", "epc",
        "handover", "paging", "registration", "authentication", "bearer",
        "qos", "slice", "nssai", "plmn", "tac", "ncgi", "pci", "arfcn",
        "bandwidth", "numerology", "beam", "mimo", "ca", "dc", "nsa", "sa",
    }

    SPEC_PATTERN = re.compile(r"(?:ts\s*)?(\d{2}\.\d{3})", re.IGNORECASE)

    def route(self, query: str) -> RouteDecision:
        query_lower = query.lower()
        intent, confidence = self._classify_intent(query_lower)
        keywords = self._extract_keywords(query_lower)
        spec_filter = self._extract_spec(query)

        index_hint = self._choose_index(intent, keywords)

        return RouteDecision(
            intent=intent,
            confidence=confidence,
            index_hint=index_hint,
            keywords=keywords,
            spec_filter=spec_filter,
        )

    def _classify_intent(self, query: str) -> tuple[QueryIntent, float]:
        scores: dict[QueryIntent, int] = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            count = sum(1 for p in patterns if re.search(p, query))
            if count:
                scores[intent] = count

        if not scores:
            return QueryIntent.GENERAL, 0.5

        best = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[best] / total
        return best, confidence

    def _extract_keywords(self, query: str) -> list[str]:
        words = set(re.findall(r"[a-z0-9]+", query))
        return [w for w in words if w in self.TELECOM_ENTITIES or len(w) > 3]

    def _extract_spec(self, query: str) -> str | None:
        match = self.SPEC_PATTERN.search(query)
        return match.group(1) if match else None

    def _choose_index(self, intent: QueryIntent, keywords: list[str]) -> str:
        if intent in (QueryIntent.RELATIONSHIP, QueryIntent.REFERENCE):
            return "kg"
        if intent == QueryIntent.DEFINITION and len(keywords) <= 2:
            return "kg"
        if intent in (QueryIntent.PROCEDURE, QueryIntent.PARAMETER):
            return "text"
        return "hybrid"
