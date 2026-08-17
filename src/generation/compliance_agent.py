"""Compliance agent - generates grounded answers with mandatory citations."""

from __future__ import annotations

from dataclasses import dataclass

from src.generation.llm_client import BaseLLM, get_llm
from src.generation.prompts import COMPLIANCE_AGENT_SYSTEM, COMPLIANCE_AGENT_USER
from src.retrieval.hybrid_retriever import RetrievalResult
from src.retrieval.kg_retriever import Triple


@dataclass
class GeneratedAnswer:
    text: str
    citations: list[str]
    chunks_used: list[str]
    triples_used: list[str]


class ComplianceAgent:
    """
    Primary generation agent that produces answers strictly grounded
    in retrieved context with forced citations.
    """

    def __init__(self, llm: BaseLLM | None = None):
        self.llm = llm or get_llm()

    def generate(
        self,
        query: str,
        chunks: list[RetrievalResult],
        triples: list[Triple],
    ) -> GeneratedAnswer:
        chunks_context = self._format_chunks(chunks)
        kg_context = self._format_triples(triples)

        user_prompt = COMPLIANCE_AGENT_USER.format(
            query=query,
            chunks_context=chunks_context or "No text chunks retrieved.",
            kg_context=kg_context or "No knowledge graph triples retrieved.",
        )

        answer_text = self.llm.generate(COMPLIANCE_AGENT_SYSTEM, user_prompt)

        citations = self._extract_citations(answer_text)
        return GeneratedAnswer(
            text=answer_text,
            citations=citations,
            chunks_used=[c.chunk_id for c in chunks],
            triples_used=[t.to_text() for t in triples],
        )

    def _format_chunks(self, chunks: list[RetrievalResult]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Source {i}]\n"
                f"Text: {chunk.text[:1500]}\n"
                f"Citation: {chunk.citation()}\n"
                f"Relevance Score: {chunk.score:.4f}"
            )
        return "\n\n".join(parts)

    def _format_triples(self, triples: list[Triple]) -> str:
        parts = []
        for i, triple in enumerate(triples, 1):
            parts.append(
                f"[Triple {i}] {triple.to_text()}\n"
                f"Citation: {triple.citation()}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _extract_citations(text: str) -> list[str]:
        import re
        pattern = r"\[TS\s+\d+\.\d+[^\]]*\]"
        return list(set(re.findall(pattern, text)))
