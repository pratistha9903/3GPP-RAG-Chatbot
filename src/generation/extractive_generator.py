"""Extractive answer generation – zero hallucination by quoting only retrieved 3GPP text."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.retrieval.hybrid_retriever import RetrievalResult


@dataclass
class ExtractiveAnswer:
    text: str
    quotes: list[dict]
    citations: list[str]
    grounded: bool
    confidence: float = 0.0
    body: str = ""
    references_md: str = ""


class ExtractiveAnswerBuilder:
    """
    Builds answers exclusively from verbatim 3GPP specification excerpts.
    No LLM paraphrasing – eliminates generative hallucination at the source.
    """

    def build(
        self,
        query: str,
        spec_chunks: list[RetrievalResult],
        max_quotes: int = 3,
    ) -> ExtractiveAnswer:
        if not spec_chunks:
            return ExtractiveAnswer(
                text=(
                    "I cannot answer this question based on the available 3GPP documentation. "
                    "No matching specification excerpts were retrieved."
                ),
                quotes=[],
                citations=[],
                grounded=False,
            )

        query_terms = self._terms(query)
        scored: list[tuple[RetrievalResult, float, str]] = []

        for chunk in spec_chunks:
            excerpt = self._best_excerpt(chunk.text, query_terms)
            relevance = self._relevance_score(excerpt, query_terms)
            combined = relevance * 0.7 + chunk.score * 0.3
            scored.append((chunk, combined, excerpt))

        scored.sort(key=lambda x: x[1], reverse=True)
        selected = scored[:max_quotes]

        if not selected or selected[0][1] < 0.15:
            return ExtractiveAnswer(
                text=(
                    "I cannot answer this question based on the available 3GPP documentation. "
                    "Retrieved specification excerpts are not sufficiently relevant."
                ),
                quotes=[],
                citations=[],
                grounded=False,
                confidence=0.0,
            )

        quotes = []
        citations = []
        seen_excerpts: set[str] = set()
        excerpt_parts: list[str] = []

        for chunk, score, excerpt in selected:
            clean = excerpt.strip()
            norm = re.sub(r"\s+", " ", clean.lower())
            if norm in seen_excerpts or len(clean) < 30:
                continue
            seen_excerpts.add(norm)

            cite = chunk.citation()
            quotes.append({
                "text": clean,
                "citation": cite,
                "chunk_id": chunk.chunk_id,
                "score": score,
            })
            excerpt_parts.append(clean)
            if cite not in citations:
                citations.append(cite)

        if not excerpt_parts:
            return ExtractiveAnswer(
                text=(
                    "I cannot answer this question based on the available 3GPP documentation. "
                    "Retrieved specification excerpts are not sufficiently relevant."
                ),
                quotes=[],
                citations=[],
                grounded=False,
                confidence=0.0,
            )

        body = self._combine_excerpts(excerpt_parts)
        confidence = min(0.99, sum(q["score"] for q in quotes) / len(quotes))

        # Inline citation markers for the body
        cite_refs = ", ".join(
            f"**[{c.split('(')[0].strip()}]**" for c in citations[:3]
        )
        body_with_cites = f"{body} {cite_refs}" if cite_refs else body

        references_md = "\n".join(f"- {cite}" for cite in citations)
        full_text = (
            f"### Answer\n\n{body_with_cites}\n\n"
            f"### References\n\n{references_md}"
        )

        return ExtractiveAnswer(
            text=full_text,
            body=body_with_cites,
            references_md=references_md,
            quotes=quotes,
            citations=citations,
            grounded=True,
            confidence=confidence,
        )

    def _combine_excerpts(self, parts: list[str]) -> str:
        """Merge multiple excerpts into one coherent paragraph."""
        if len(parts) == 1:
            return parts[0]

        combined: list[str] = []
        for part in parts:
            sentences = re.split(r"(?<=[.!?])\s+", part.strip())
            for sent in sentences:
                sent = sent.strip()
                if not sent or len(sent) < 20:
                    continue
                norm = re.sub(r"\s+", " ", sent.lower())
                if any(norm in re.sub(r"\s+", " ", existing.lower()) for existing in combined):
                    continue
                if any(re.sub(r"\s+", " ", existing.lower()) in norm for existing in combined):
                    continue
                combined.append(sent)

        if not combined:
            return " ".join(parts)

        # Group into 1–2 flowing paragraphs
        mid = max(1, len(combined) // 2)
        para1 = " ".join(combined[:mid])
        para2 = " ".join(combined[mid:]) if len(combined) > mid else ""
        return f"{para1}\n\n{para2}".strip() if para2 else para1

    def _best_excerpt(self, text: str, query_terms: set[str], max_len: int = 600) -> str:
        """Select the most query-relevant paragraph from chunk text."""
        # Strip context prefix added by chunker
        body = re.sub(r"^\[Rel-19 TS[^\]]+\][^\n]*\n+", "", text).strip()
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if len(p.strip()) > 40]

        if not paragraphs:
            return body[:max_len]

        best = paragraphs[0]
        best_score = -1.0
        for para in paragraphs:
            score = self._relevance_score(para, query_terms)
            if score > best_score:
                best_score = score
                best = para

        if len(best) > max_len:
            best = best[: max_len - 3].rsplit(" ", 1)[0] + "..."
        return best

    @staticmethod
    def _terms(text: str) -> set[str]:
        stop = {
            "what", "is", "the", "a", "an", "how", "does", "do", "of", "in", "for",
            "and", "or", "to", "are", "between", "difference", "explain", "describe",
        }
        return {
            t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in stop and len(t) > 2
        }

    @staticmethod
    def _relevance_score(text: str, query_terms: set[str]) -> float:
        if not query_terms:
            return 0.0
        text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
        overlap = len(query_terms & text_terms)
        return overlap / len(query_terms)
