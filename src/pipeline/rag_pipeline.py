"""End-to-end KG-RAG pipeline with strict zero-hallucination mode."""

from __future__ import annotations

from dataclasses import dataclass, field

from config.settings import get_settings
from src.generation.extractive_generator import ExtractiveAnswerBuilder
from src.generation.reflection_agent import ReflectionAgent, VerificationResult
from src.indexing.indexer import DocumentIndexer
from src.knowledge_graph.loader import KnowledgeGraphLoader
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from src.retrieval.kg_retriever import KGRetriever, Triple
from src.retrieval.query_router import RouteDecision, SemanticQueryRouter
from src.verification.chain_of_noting import ChainOfNoting, NotingResult
from src.verification.citation_verifier import CitationVerifier, CitationVerificationResult
from src.verification.grounding_verifier import GroundingVerifier, GroundingResult


@dataclass
class PipelineResponse:
    query: str
    answer: str
    route: RouteDecision | None = None
    retrieved_chunks: list[RetrievalResult] = field(default_factory=list)
    spec_chunks: list[RetrievalResult] = field(default_factory=list)
    retrieved_triples: list[Triple] = field(default_factory=list)
    noting_result: NotingResult | None = None
    verification_result: VerificationResult | None = None
    citation_result: CitationVerificationResult | None = None
    grounding_result: GroundingResult | None = None
    rejected: bool = False
    rejection_reason: str = ""
    generation_mode: str = "extractive"
    confidence_score: float = 0.0
    extractive_confidence: float = 0.0


class KGRAGPipeline:
    """
    Zero-hallucination KG-RAG pipeline:
    - Primary source: 3GPP specification documents
    - Supplement: Knowledge graph triples
    - Generation: Extractive (verbatim quotes only)
    - Verification: CoN + citation + grounding + reflection
    """

    UNKNOWN_RESPONSE = (
        "I cannot answer this question based on the available 3GPP documentation. "
        "The retrieved information is insufficient to provide a reliable, "
        "citation-backed response."
    )

    def __init__(
        self,
        indexer: DocumentIndexer | None = None,
        kg_loader: KnowledgeGraphLoader | None = None,
    ):
        self.settings = get_settings()
        self.indexer = indexer or DocumentIndexer()
        self.kg_loader = kg_loader or KnowledgeGraphLoader()

        self.router = SemanticQueryRouter()
        self.retriever: HybridRetriever | None = None
        self.kg_retriever: KGRetriever | None = None
        self.extractive = ExtractiveAnswerBuilder()
        self.reflection_agent = ReflectionAgent()
        self.con = ChainOfNoting()
        self.citation_verifier = CitationVerifier()
        self.grounding_verifier = GroundingVerifier()

        self._initialized = False

    def initialize(self) -> None:
        if not self.indexer.is_built:
            raise RuntimeError(
                "Search index not built. Run: python scripts/build_index.py"
            )
        self.indexer.load()

        if not self.kg_loader.is_loaded:
            max_nodes = self.settings.demo_kg_nodes if self.settings.demo_mode else None
            self.kg_loader.load(max_nodes=max_nodes)

        self.retriever = HybridRetriever(self.indexer)
        self.kg_retriever = KGRetriever(self.kg_loader)
        self.citation_verifier.kg = self.kg_loader
        self._initialized = True

    def query(self, user_query: str) -> PipelineResponse:
        if not self._initialized:
            self.initialize()

        response = PipelineResponse(query=user_query, answer="")

        route = self.router.route(user_query)
        response.route = route

        spec_chunks = self.retriever.retrieve_primary_specs(user_query)
        spec_chunks = self.retriever.enrich_with_parent_context(spec_chunks)
        spec_chunks = [c for c in spec_chunks if c.is_primary_spec]

        triples: list[Triple] = []
        if self.settings.index_kg_supplement:
            triples = self.kg_retriever.retrieve_triples(
                user_query,
                keywords=route.keywords,
                top_k=self.settings.top_k_kg_triples,
            )

        response.spec_chunks = spec_chunks
        response.retrieved_chunks = spec_chunks
        response.retrieved_triples = triples

        if self.settings.require_3gpp_spec_evidence and len(spec_chunks) < self.settings.min_spec_chunks:
            response.rejected = True
            response.rejection_reason = (
                "No primary 3GPP specification evidence found for this query."
            )
            response.answer = self.UNKNOWN_RESPONSE
            return response

        noting = self.con.evaluate(user_query, spec_chunks, triples)
        response.noting_result = noting

        if not noting.can_answer or noting.confidence < self.settings.min_con_confidence:
            response.rejected = True
            response.rejection_reason = noting.reasoning or "Insufficient 3GPP context (CoN rejected)"
            response.answer = self.UNKNOWN_RESPONSE
            return response

        context_texts = [c.text for c in spec_chunks]
        available_citations = [c.citation() for c in spec_chunks]
        available_clauses = [
            (c.metadata.get("spec_id", ""), c.metadata.get("clause", ""))
            for c in spec_chunks
            if c.metadata.get("clause")
        ]

        extracted = self.extractive.build(user_query, spec_chunks)
        response.generation_mode = "extractive"
        if not extracted.grounded:
            response.rejected = True
            response.rejection_reason = "Extractive generator found no grounded 3GPP excerpts"
            response.answer = extracted.text
            return response
        answer_text = extracted.text
        response.extractive_confidence = extracted.confidence

        citation_result = self.citation_verifier.verify(
            answer_text,
            available_citations=available_citations,
            available_clauses=available_clauses,
        )
        response.citation_result = citation_result

        if self.settings.reject_on_invalid_citation and citation_result.hallucinated:
            response.rejected = True
            response.rejection_reason = "; ".join(citation_result.details)
            response.answer = self.UNKNOWN_RESPONSE
            return response

        grounding = self.grounding_verifier.verify(
            answer_text,
            context_texts,
            min_score=self.settings.min_grounding_score,
        )
        response.grounding_result = grounding

        if self.settings.reject_on_ungrounded_claim and not grounding.grounded:
            response.rejected = True
            response.rejection_reason = "; ".join(grounding.details) or "Ungrounded claims detected"
            response.answer = self.UNKNOWN_RESPONSE
            return response

        context = self._build_verification_context(spec_chunks, triples)
        verification = self.reflection_agent.verify(user_query, answer_text, context)
        response.verification_result = verification

        if verification.verdict == "REJECTED":
            response.rejected = True
            response.rejection_reason = "; ".join(verification.issues)
            response.answer = self.UNKNOWN_RESPONSE
        elif verification.verdict == "NEEDS_REVISION" and verification.corrected_answer:
            re_cite = self.citation_verifier.verify(
                verification.corrected_answer,
                available_citations=available_citations,
                available_clauses=available_clauses,
            )
            if re_cite.hallucinated:
                response.rejected = True
                response.rejection_reason = "Reflection correction failed citation check"
                response.answer = self.UNKNOWN_RESPONSE
            else:
                response.answer = verification.corrected_answer
        else:
            response.answer = answer_text

        response.confidence_score = self._compute_confidence(response)
        return response

    def _compute_confidence(self, response: PipelineResponse) -> float:
        if response.rejected:
            return 0.0

        scores: list[float] = []

        if response.extractive_confidence:
            scores.append(response.extractive_confidence)
        if response.noting_result:
            scores.append(response.noting_result.confidence)
        if response.grounding_result:
            scores.append(response.grounding_result.score)
        if response.verification_result:
            scores.append(response.verification_result.confidence)
        if response.citation_result and response.citation_result.valid_citations:
            total = len(response.citation_result.valid_citations) + len(
                response.citation_result.invalid_citations
            )
            if total:
                scores.append(len(response.citation_result.valid_citations) / total)
        if response.spec_chunks:
            top = max(c.score for c in response.spec_chunks)
            scores.append(min(top * 10, 1.0))

        return round(sum(scores) / len(scores), 2) if scores else 0.0

    def _build_verification_context(
        self, chunks: list[RetrievalResult], triples: list[Triple]
    ) -> str:
        parts = []
        for c in chunks:
            parts.append(f"[PRIMARY 3GPP {c.citation()}] {c.text[:1000]}")
        for t in triples:
            parts.append(f"[KG SUPPLEMENT {t.citation()}] {t.to_text()}")
        return "\n\n".join(parts)
