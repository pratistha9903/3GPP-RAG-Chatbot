"""Verify every claim in an answer is grounded in retrieved 3GPP context."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class GroundingResult:
    grounded: bool
    score: float
    ungrounded_sentences: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)


class GroundingVerifier:
    """
    Sentence-level grounding check against retrieved 3GPP context.
    Rejects answers containing claims not supported by source text.
    """

    SKIP_PATTERNS = re.compile(
        r"^(references|answer|based on|\*\*|^\d+\.\s*$|extracted verbatim)",
        re.IGNORECASE,
    )

    def verify(
        self,
        answer: str,
        context_texts: list[str],
        min_score: float = 0.45,
    ) -> GroundingResult:
        if not answer.strip() or not context_texts:
            return GroundingResult(
                grounded=False,
                score=0.0,
                details=["No answer or no context to verify against"],
            )

        corpus = " ".join(context_texts).lower()
        corpus_terms = set(re.findall(r"[a-z0-9]+", corpus))

        sentences = self._split_sentences(answer)
        if not sentences:
            return GroundingResult(grounded=True, score=1.0)

        grounded_count = 0
        ungrounded: list[str] = []

        for sentence in sentences:
            if self._should_skip(sentence):
                grounded_count += 1
                continue

            sent_terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            sent_terms -= {"ts", "rel", "clause", "references", "primary", "source"}

            if len(sent_terms) < 3:
                grounded_count += 1
                continue

            overlap = len(sent_terms & corpus_terms) / len(sent_terms)
            if overlap >= min_score:
                grounded_count += 1
            else:
                ungrounded.append(sentence)

        score = grounded_count / len(sentences) if sentences else 0.0
        grounded = len(ungrounded) == 0 and score >= min_score

        return GroundingResult(
            grounded=grounded,
            score=score,
            ungrounded_sentences=ungrounded,
            details=[
                f"Ungrounded claim: {s[:120]}..." if len(s) > 120 else f"Ungrounded claim: {s}"
                for s in ungrounded
            ],
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        # Remove citation tags for analysis but keep sentence structure
        cleaned = re.sub(r"\[TS[^\]]+\]", "", text)
        parts = re.split(r"(?<=[.!?])\s+|\n(?=\d+\.)", cleaned)
        return [p.strip() for p in parts if len(p.strip()) > 20]

    def _should_skip(self, sentence: str) -> bool:
        if self.SKIP_PATTERNS.search(sentence.strip()):
            return True
        if sentence.strip().startswith("**References"):
            return True
        if re.match(r"^\d+\.\s+[A-Z]{2,}", sentence):  # Likely a spec quote header
            return False
        return False
