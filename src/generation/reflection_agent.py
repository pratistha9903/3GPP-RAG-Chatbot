"""Reflection/verification agent for cross-validating generated answers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.generation.llm_client import BaseLLM, get_llm
from src.generation.prompts import REFLECTION_AGENT_SYSTEM, REFLECTION_AGENT_USER


@dataclass
class VerificationResult:
    verdict: str  # APPROVED | REJECTED | NEEDS_REVISION
    confidence: float
    issues: list[str]
    corrected_answer: str
    missing_info: bool


class ReflectionAgent:
    """
    Secondary agent that reviews the compliance agent's output
    against retrieved context and the knowledge graph.
    """

    def __init__(self, llm: BaseLLM | None = None):
        self.llm = llm or get_llm()

    def verify(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> VerificationResult:
        user_prompt = REFLECTION_AGENT_USER.format(
            query=query,
            context=context[:8000],
            answer=answer,
        )

        response = self.llm.generate(REFLECTION_AGENT_SYSTEM, user_prompt, temperature=0.0)
        return self._parse_response(response)

    def _parse_response(self, response: str) -> VerificationResult:
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return VerificationResult(
                    verdict=data.get("verdict", "NEEDS_REVISION"),
                    confidence=float(data.get("confidence", 0.5)),
                    issues=data.get("issues", []),
                    corrected_answer=data.get("corrected_answer", ""),
                    missing_info=bool(data.get("missing_info", False)),
                )
        except (json.JSONDecodeError, ValueError):
            pass

        return VerificationResult(
            verdict="NEEDS_REVISION",
            confidence=0.3,
            issues=["Failed to parse verification response"],
            corrected_answer="",
            missing_info=False,
        )
