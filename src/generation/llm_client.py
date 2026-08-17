"""Rule-based mock LLM for CoN and reflection agents (no external API)."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, system: str, user: str, temperature: float = 0.1) -> str:
        ...


class MockLLM(BaseLLM):
    """Rule-based mock LLM for verification agents without external API keys."""

    def generate(self, system: str, user: str, temperature: float = 0.1) -> str:
        if "Chain-of-Noting" in system or "can_answer" in system:
            return self._mock_con_response(user)
        if "verification agent" in system or "verdict" in system:
            return self._mock_reflection_response(user)
        return self._mock_compliance_response(user)

    def _mock_compliance_response(self, user: str) -> str:
        query_match = re.search(r"Question:\s*(.+?)(?:\n\nRetrieved|$)", user, re.DOTALL)
        query = query_match.group(1).strip() if query_match else "your question"

        chunks = re.findall(r"\[Source \d+\].*?(?=\[Source|\n\nKnowledge|$)", user, re.DOTALL)
        kg_triples = re.findall(r"\(.*?\) --\[.*?\]-->.*", user)

        if not chunks and not kg_triples:
            return (
                "I cannot answer this question based on the available 3GPP documentation.\n\n"
                "The retrieved context does not contain sufficient information to provide "
                "a grounded response."
            )

        answer_parts = [f"Based on the retrieved 3GPP documentation, here is the answer to: **{query}**\n"]

        for chunk in chunks[:3]:
            text_match = re.search(r"Text:\s*(.+?)(?:\nCitation:|$)", chunk, re.DOTALL)
            cite_match = re.search(r"Citation:\s*(.+)", chunk)
            if text_match:
                text = text_match.group(1).strip()[:500]
                cite = cite_match.group(1).strip() if cite_match else "TS unknown (Rel-19)"
                answer_parts.append(f"- {text} [{cite}]")

        if kg_triples:
            answer_parts.append("\n**Knowledge Graph Relations:**")
            for triple in kg_triples[:3]:
                answer_parts.append(f"- {triple.strip()}")

        answer_parts.append("\n**References:**")
        for chunk in chunks[:3]:
            cite_match = re.search(r"Citation:\s*(.+)", chunk)
            if cite_match:
                answer_parts.append(f"- {cite_match.group(1).strip()}")

        return "\n".join(answer_parts)

    def _mock_con_response(self, user: str) -> str:
        has_content = "Text:" in user or "--[" in user
        return json.dumps({
            "can_answer": has_content,
            "reasoning": (
                "Retrieved context contains relevant 3GPP specification excerpts."
                if has_content
                else "No relevant context was retrieved for this query."
            ),
            "missing_information": [] if has_content else ["No matching specification content found"],
            "confidence": 0.85 if has_content else 0.1,
        })

    def _mock_reflection_response(self, user: str) -> str:
        answer_match = re.search(r"Generated Answer:\s*(.+?)$", user, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else ""

        if "cannot answer" in answer.lower():
            return json.dumps({
                "verdict": "APPROVED",
                "confidence": 0.95,
                "issues": [],
                "corrected_answer": "",
                "missing_info": True,
            })

        has_citations = bool(re.search(r"\[TS\s+\d+\.\d+", answer))
        issues = []
        if not has_citations:
            issues.append("Answer lacks mandatory TS citations")

        return json.dumps({
            "verdict": "APPROVED" if has_citations else "NEEDS_REVISION",
            "confidence": 0.9 if has_citations else 0.4,
            "issues": issues,
            "corrected_answer": answer if has_citations else "",
            "missing_info": False,
        })


def get_llm() -> BaseLLM:
    return MockLLM()
