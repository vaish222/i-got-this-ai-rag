from __future__ import annotations

import operator
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Annotated, Any, Protocol, TypedDict

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, START, StateGraph

from .baseline import REFUSAL_TEXT, generate_grounded_answer
from .metadata_retrieval import MetadataConstraints, MetadataQueryAnalyzer
from .query_transformation import LLMQueryTransformer
from .reranking import CandidateReranker
from .retrieval import lexical_tokens
from .settings import Settings


AGENTIC_GRAPH_VERSION = "phase9-langgraph-v1"
CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
CONCRETE_FACT_PATTERNS = (
    r"\b(?:adult|child|friend|relative|neighbor|mentee|coordinator|colleague)(?:_[a-z]+)*_\d{2}\b",
    r"\b20\d{2}-\d{2}-\d{2}\b",
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+20\d{2})?\b",
    r"\b\d{1,2}(?::\d{2})?\s*(?:AM|PM)\b",
    r"\b\d+\b",
)
EVIDENCE_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "at",
    "be",
    "do",
    "does",
    "for",
    "from",
    "have",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "we",
    "what",
    "when",
    "which",
    "who",
}
ANSWER_BOILERPLATE_WORDS = {
    "according",
    "based",
    "breakdown",
    "data",
    "here",
    "provided",
    "source",
    "sources",
    "status",
    "will",
    "you",
}
MINIMUM_CLAIM_TERM_COVERAGE = 1.0

RetrievedResults = list[tuple[Document, float]]


def _claim_text(line: str) -> str:
    without_bullet = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line)
    without_citations = CITATION_PATTERN.sub("", without_bullet)
    return without_citations.replace("**", "").strip()


def _informative_terms(text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            _normalized_fact(token)
            for token in lexical_tokens(text)
            if len(token) > 2
            and token not in EVIDENCE_STOP_WORDS
            and token not in ANSWER_BOILERPLATE_WORDS
        )
    )


def _concrete_facts(text: str) -> tuple[str, ...]:
    facts: list[str] = []
    for pattern in CONCRETE_FACT_PATTERNS:
        facts.extend(
            match.group(0)
            for match in re.finditer(pattern, text, re.IGNORECASE)
        )
    return tuple(dict.fromkeys(facts))


def _normalized_fact(text: str) -> str:
    normalized = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", text.casefold())
    return re.sub(r"\s+", " ", normalized).strip()


def _repair_ordinal_suffixes(text: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        value = int(match.group(1))
        suffix = (
            "th"
            if 10 <= value % 100 <= 20
            else {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
        )
        return f"{value}{suffix}"

    return re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", replacement, text, flags=re.IGNORECASE)


def _is_factual_claim(text: str) -> bool:
    if text.endswith(":") and not _concrete_facts(text):
        return False
    return bool(_concrete_facts(text)) or len(_informative_terms(text)) >= 2


class AgenticRAGState(TypedDict, total=False):
    original_query: str
    search_query: str
    intent: dict[str, Any]
    metadata_constraints: dict[str, list[str]]
    metadata_filters: dict[str, Any] | None
    retrieved_docs: RetrievedResults
    reranked_docs: RetrievedResults
    retrieval_attempts: int
    evidence_sufficient: bool
    evidence_grade: dict[str, Any]
    draft_answer: str
    answer: str
    citations: list[str]
    grounded: bool
    grounding_result: dict[str, Any]
    generated_queries: list[str]
    protected_query_terms: list[str]
    query_guard_repairs: list[dict[str, Any]]
    refusal_reason: str | None
    retry_reason: str | None
    total_latency_seconds: float
    node_trace: Annotated[list[dict[str, Any]], operator.add]
    query_history: Annotated[list[dict[str, Any]], operator.add]
    retrieval_history: Annotated[list[dict[str, Any]], operator.add]
    evidence_history: Annotated[list[dict[str, Any]], operator.add]


@dataclass(frozen=True)
class AgenticGraphConfig:
    metadata_filter_enabled: bool = True
    metadata_fallback_enabled: bool = True
    retry_query_rewriting_enabled: bool = True
    reranker_enabled: bool = False
    initial_candidate_k: int = 5
    retry_candidate_k: int = 5
    final_top_k: int = 5
    max_retrieval_attempts: int = 2
    minimum_evidence_term_coverage: float = 0.2

    def validate(self) -> None:
        if self.max_retrieval_attempts != 2:
            raise ValueError(
                "Phase 9 permits exactly one retry, so max_retrieval_attempts must be 2."
            )
        if self.initial_candidate_k < self.final_top_k:
            raise ValueError("initial_candidate_k cannot be smaller than final_top_k.")
        if self.retry_candidate_k < self.final_top_k:
            raise ValueError("retry_candidate_k cannot be smaller than final_top_k.")
        if not 0 < self.minimum_evidence_term_coverage <= 1:
            raise ValueError("minimum_evidence_term_coverage must be between 0 and 1.")


@dataclass(frozen=True)
class EvidenceGrade:
    sufficient: bool
    reason: str
    matched_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]
    term_coverage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "reason": self.reason,
            "matched_terms": list(self.matched_terms),
            "missing_terms": list(self.missing_terms),
            "term_coverage": self.term_coverage,
        }


class EvidenceGrader(Protocol):
    def grade(self, question: str, results: RetrievedResults) -> EvidenceGrade: ...


class LexicalEvidenceGrader:
    """Auditable evidence gate used before generation and retry routing."""

    def __init__(self, minimum_term_coverage: float = 0.2) -> None:
        if not 0 < minimum_term_coverage <= 1:
            raise ValueError("minimum_term_coverage must be between 0 and 1.")
        self.minimum_term_coverage = minimum_term_coverage

    def grade(self, question: str, results: RetrievedResults) -> EvidenceGrade:
        query_terms = tuple(
            dict.fromkeys(
                token
                for token in lexical_tokens(question)
                if len(token) > 2 and token not in EVIDENCE_STOP_WORDS
            )
        )
        context = " ".join(
            f"{document.metadata.get('document_title', '')} {document.page_content}"
            for document, _ in results
        )
        context_terms = set(lexical_tokens(context))
        matched = tuple(term for term in query_terms if term in context_terms)
        missing = tuple(term for term in query_terms if term not in context_terms)
        coverage = len(matched) / len(query_terms) if query_terms else 0.0
        sufficient = bool(results) and bool(matched) and coverage >= self.minimum_term_coverage
        reason = (
            "retrieved evidence covers enough informative query terms"
            if sufficient
            else "retrieved evidence has insufficient informative-term coverage"
        )
        return EvidenceGrade(
            sufficient=sufficient,
            reason=reason,
            matched_terms=matched,
            missing_terms=missing,
            term_coverage=round(coverage, 6),
        )


@dataclass(frozen=True)
class GroundingResult:
    grounded: bool
    reason: str
    citation_labels: tuple[str, ...]
    unsupported_facts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "grounded": self.grounded,
            "reason": self.reason,
            "citation_labels": list(self.citation_labels),
            "unsupported_facts": list(self.unsupported_facts),
        }


class GroundingVerifier(Protocol):
    def verify(self, answer: str, results: RetrievedResults) -> GroundingResult: ...


class CitationAttributor:
    """Attach citations only when a retrieved source supports the claim text."""

    def attribute(self, answer: str, results: RetrievedResults) -> str:
        if answer.strip() == REFUSAL_TEXT:
            return answer
        repaired_answer = _repair_ordinal_suffixes(answer)
        if CITATION_PATTERN.search(repaired_answer):
            return repaired_answer

        attributed_lines: list[str] = []
        for line in repaired_answer.splitlines():
            claim = _claim_text(line)
            if not claim or not _is_factual_claim(claim):
                attributed_lines.append(line)
                continue

            claim_terms = set(_informative_terms(claim))
            facts = _concrete_facts(claim)
            best_label: str | None = None
            best_score = 0.0
            for index, (document, _) in enumerate(results, start=1):
                source_text = document.page_content.casefold()
                normalized_source = _normalized_fact(source_text)
                if any(
                    _normalized_fact(fact) not in normalized_source for fact in facts
                ):
                    continue
                source_terms = {
                    _normalized_fact(token) for token in lexical_tokens(source_text)
                }
                overlap = len(claim_terms & source_terms)
                coverage = overlap / len(claim_terms) if claim_terms else 0.0
                if overlap and coverage > best_score:
                    best_label = f"S{index}"
                    best_score = coverage

            if best_label is None or best_score < MINIMUM_CLAIM_TERM_COVERAGE:
                attributed_lines.append(line)
            else:
                attributed_lines.append(f"{line.rstrip()} [{best_label}]")
        return "\n".join(attributed_lines)


class CitationGroundingVerifier:
    """Reject answers with unresolved citations or concrete facts absent from cited text."""

    def verify(self, answer: str, results: RetrievedResults) -> GroundingResult:
        if answer.strip() == REFUSAL_TEXT:
            return GroundingResult(
                False,
                "generator returned the explicit insufficient-information response",
                (),
                (),
            )

        citation_numbers = tuple(dict.fromkeys(CITATION_PATTERN.findall(answer)))
        labels = tuple(f"S{number}" for number in citation_numbers)
        valid_numbers = {
            str(index) for index in range(1, len(results) + 1)
        }
        invalid_labels = tuple(
            f"S{number}" for number in citation_numbers if number not in valid_numbers
        )
        if not labels:
            return GroundingResult(False, "generated answer has no citations", (), ())
        if invalid_labels:
            return GroundingResult(
                False,
                "generated answer contains unresolved citations",
                labels,
                invalid_labels,
            )

        unsupported: list[str] = []
        for line in answer.splitlines():
            claim = _claim_text(line)
            if not claim or not _is_factual_claim(claim):
                continue
            line_citations = tuple(dict.fromkeys(CITATION_PATTERN.findall(line)))
            if not line_citations:
                unsupported.append(claim)
                continue
            cited_text = " ".join(
                results[int(number) - 1][0].page_content
                for number in line_citations
            ).casefold()
            facts = _concrete_facts(claim)
            normalized_cited_text = _normalized_fact(cited_text)
            if any(
                _normalized_fact(fact) not in normalized_cited_text for fact in facts
            ):
                unsupported.extend(
                    fact
                    for fact in facts
                    if _normalized_fact(fact) not in normalized_cited_text
                )
                continue
            claim_terms = set(_informative_terms(claim))
            cited_terms = {
                _normalized_fact(token) for token in lexical_tokens(cited_text)
            }
            coverage = (
                len(claim_terms & cited_terms) / len(claim_terms)
                if claim_terms
                else 0.0
            )
            if coverage < MINIMUM_CLAIM_TERM_COVERAGE:
                unsupported.append(claim)
        unsupported_facts = tuple(dict.fromkeys(unsupported))
        if unsupported_facts:
            return GroundingResult(
                False,
                "generated answer contains unsupported or uncited claims",
                labels,
                unsupported_facts,
            )
        return GroundingResult(
            True,
            "citations resolve and concrete facts are supported",
            labels,
            (),
        )


class LangGraphRAG:
    def __init__(
        self,
        settings: Settings,
        vector_store: PineconeVectorStore,
        llm: Any,
        config: AgenticGraphConfig | None = None,
        metadata_analyzer: MetadataQueryAnalyzer | None = None,
        evidence_grader: EvidenceGrader | None = None,
        grounding_verifier: GroundingVerifier | None = None,
        reranker: CandidateReranker | None = None,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.llm = llm
        self.config = config or AgenticGraphConfig()
        self.config.validate()
        if self.config.reranker_enabled != (reranker is not None):
            raise ValueError("reranker_enabled must match whether a reranker is provided.")
        self.metadata_analyzer = metadata_analyzer or MetadataQueryAnalyzer(
            settings.reference_date
        )
        self.evidence_grader = evidence_grader or LexicalEvidenceGrader(
            self.config.minimum_evidence_term_coverage
        )
        self.grounding_verifier = grounding_verifier or CitationGroundingVerifier()
        self.citation_attributor = CitationAttributor()
        self.reranker = reranker
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(AgenticRAGState)
        builder.add_node("query_analysis", self._query_analysis)
        builder.add_node("query_rewriting", self._query_rewriting)
        builder.add_node("metadata_construction", self._metadata_construction)
        builder.add_node("retrieval", self._retrieval)
        builder.add_node("reranking", self._reranking)
        builder.add_node("evidence_grading", self._evidence_grading)
        builder.add_node("prepare_retry", self._prepare_retry)
        builder.add_node("generation", self._generation)
        builder.add_node("grounding_verification", self._grounding_verification)
        builder.add_node("refusal", self._refusal)

        builder.add_edge(START, "query_analysis")
        builder.add_edge("query_analysis", "query_rewriting")
        builder.add_edge("query_rewriting", "metadata_construction")
        builder.add_edge("metadata_construction", "retrieval")
        builder.add_edge("retrieval", "reranking")
        builder.add_edge("reranking", "evidence_grading")
        builder.add_conditional_edges(
            "evidence_grading",
            self._route_after_evidence,
            {
                "generate": "generation",
                "retry": "prepare_retry",
                "refuse": "refusal",
            },
        )
        builder.add_edge("prepare_retry", "query_rewriting")
        builder.add_edge("generation", "grounding_verification")
        builder.add_conditional_edges(
            "grounding_verification",
            self._route_after_grounding,
            {"complete": END, "refuse": "refusal"},
        )
        builder.add_edge("refusal", END)
        return builder.compile()

    def invoke(self, question: str) -> AgenticRAGState:
        started = perf_counter()
        state = self.graph.invoke(
            {
                "original_query": question,
                "search_query": question,
                "retrieval_attempts": 0,
                "node_trace": [],
                "query_history": [],
                "retrieval_history": [],
                "evidence_history": [],
            }
        )
        return {
            **state,
            "total_latency_seconds": round(perf_counter() - started, 6),
        }

    def _query_analysis(self, state: AgenticRAGState) -> dict[str, Any]:
        started = perf_counter()
        question = state["original_query"]
        constraints = self.metadata_analyzer.analyze(question)
        lowered = question.lower()
        if re.search(r"\b(?:when|schedule|calendar|coming up|what's next|this week)\b", lowered):
            intent_type = "schedule"
        elif re.search(
            r"\b(?:prepare|complete|responsibilit|deadline|due|rsvp|gift|bring)\b",
            lowered,
        ):
            intent_type = "obligation"
        else:
            intent_type = "knowledge"
        intent = {
            "type": intent_type,
            "structured_constraints": constraints.to_dict(),
        }
        return {
            "intent": intent,
            "metadata_constraints": constraints.to_dict(),
            "node_trace": [self._trace("query_analysis", started, intent=intent_type)],
        }

    def _query_rewriting(self, state: AgenticRAGState) -> dict[str, Any]:
        started = perf_counter()
        is_retry = state.get("retrieval_attempts", 0) > 0
        strategy = (
            "rewrite"
            if is_retry and self.config.retry_query_rewriting_enabled
            else "original"
        )
        transformer = LLMQueryTransformer(
            strategy=strategy,
            llm=self.llm if strategy == "rewrite" else None,
            reference_date=self.settings.reference_date,
            timezone=self.settings.timezone,
            generated_query_count=1 if strategy == "rewrite" else 0,
        )
        transformed = transformer.transform(state["original_query"])
        search_query = transformed.retrieval_queries[0]
        query_record = {
            "attempt": state.get("retrieval_attempts", 0) + 1,
            "strategy": strategy,
            "search_query": search_query,
            "generated_queries": list(transformed.generated_queries),
            "protected_query_terms": list(transformed.protected_terms),
            "guard_repairs": list(transformed.guard_repairs),
            "raw_output": transformed.raw_output,
        }
        return {
            "search_query": search_query,
            "generated_queries": list(transformed.generated_queries),
            "protected_query_terms": list(transformed.protected_terms),
            "query_guard_repairs": list(transformed.guard_repairs),
            "query_history": [query_record],
            "node_trace": [self._trace("query_rewriting", started, strategy=strategy)],
        }

    def _metadata_construction(self, state: AgenticRAGState) -> dict[str, Any]:
        started = perf_counter()
        constraints = MetadataConstraints(
            **{
                key: tuple(value)
                for key, value in state.get("metadata_constraints", {}).items()
            }
        )
        use_filter = (
            self.config.metadata_filter_enabled
            and state.get("retrieval_attempts", 0) == 0
        )
        metadata_filter = constraints.to_pinecone_filter() if use_filter else None
        return {
            "metadata_filters": metadata_filter,
            "node_trace": [
                self._trace(
                    "metadata_construction",
                    started,
                    filter_applied=metadata_filter is not None,
                )
            ],
        }

    def _retrieval(self, state: AgenticRAGState) -> dict[str, Any]:
        started = perf_counter()
        attempt = state.get("retrieval_attempts", 0) + 1
        candidate_k = (
            self.config.initial_candidate_k
            if attempt == 1
            else self.config.retry_candidate_k
        )
        query = state["search_query"]
        metadata_filter = state.get("metadata_filters")
        filtered_results: RetrievedResults = []
        fallback_results: RetrievedResults = []
        if metadata_filter is not None:
            filtered_results = self.vector_store.similarity_search_with_score(
                query,
                k=candidate_k,
                filter=metadata_filter,
            )
            if (
                self.config.metadata_fallback_enabled
                and len(filtered_results) < candidate_k
            ):
                fallback_results = self.vector_store.similarity_search_with_score(
                    query,
                    k=candidate_k,
                )
        else:
            fallback_results = self.vector_store.similarity_search_with_score(
                query,
                k=candidate_k,
            )
        results = self._combine_results(
            filtered_results,
            fallback_results,
            candidate_k,
            attempt,
        )
        latency = perf_counter() - started
        record = {
            "attempt": attempt,
            "search_query": query,
            "candidate_k": candidate_k,
            "metadata_filter": metadata_filter,
            "filtered_result_count": len(filtered_results),
            "fallback_result_count": max(0, len(results) - len(filtered_results)),
            "result_chunk_ids": [
                document.metadata.get("chunk_id") for document, _ in results
            ],
            "latency_seconds": round(latency, 6),
        }
        return {
            "retrieved_docs": results,
            "retrieval_attempts": attempt,
            "retrieval_history": [record],
            "node_trace": [
                self._trace("retrieval", started, attempt=attempt, result_count=len(results))
            ],
        }

    def _reranking(self, state: AgenticRAGState) -> dict[str, Any]:
        started = perf_counter()
        candidates = state.get("retrieved_docs", [])
        if self.reranker is None:
            results = candidates[: self.config.final_top_k]
        else:
            results = self.reranker.rerank(
                state["original_query"],
                candidates,
                self.config.final_top_k,
            )
        return {
            "reranked_docs": results,
            "node_trace": [
                self._trace(
                    "reranking",
                    started,
                    enabled=self.reranker is not None,
                    result_count=len(results),
                )
            ],
        }

    def _evidence_grading(self, state: AgenticRAGState) -> dict[str, Any]:
        started = perf_counter()
        grade = self.evidence_grader.grade(
            state["original_query"],
            state.get("reranked_docs", []),
        )
        grade_dict = grade.to_dict()
        grade_dict["attempt"] = state["retrieval_attempts"]
        return {
            "evidence_sufficient": grade.sufficient,
            "evidence_grade": grade_dict,
            "evidence_history": [grade_dict],
            "node_trace": [
                self._trace(
                    "evidence_grading",
                    started,
                    sufficient=grade.sufficient,
                    term_coverage=grade.term_coverage,
                )
            ],
        }

    def _prepare_retry(self, state: AgenticRAGState) -> dict[str, Any]:
        started = perf_counter()
        reason = str(state.get("evidence_grade", {}).get("reason", "weak evidence"))
        return {
            "retry_reason": reason,
            "metadata_filters": None,
            "node_trace": [self._trace("prepare_retry", started, reason=reason)],
        }

    def _generation(self, state: AgenticRAGState) -> dict[str, Any]:
        started = perf_counter()
        results = state.get("reranked_docs", [])
        draft_answer = generate_grounded_answer(
            self.settings,
            self.llm,
            state["original_query"],
            results,
        )
        answer = self.citation_attributor.attribute(draft_answer, results)
        citations = [f"S{number}" for number in dict.fromkeys(CITATION_PATTERN.findall(answer))]
        return {
            "draft_answer": draft_answer,
            "answer": answer,
            "citations": citations,
            "node_trace": [
                self._trace("generation", started, citation_count=len(citations))
            ],
        }

    def _grounding_verification(self, state: AgenticRAGState) -> dict[str, Any]:
        started = perf_counter()
        result = self.grounding_verifier.verify(
            state.get("answer", ""),
            state.get("reranked_docs", []),
        )
        return {
            "grounded": result.grounded,
            "grounding_result": result.to_dict(),
            "node_trace": [
                self._trace(
                    "grounding_verification",
                    started,
                    grounded=result.grounded,
                )
            ],
        }

    def _refusal(self, state: AgenticRAGState) -> dict[str, Any]:
        started = perf_counter()
        if not state.get("evidence_sufficient", False):
            reason = "insufficient evidence after the allowed retrieval attempts"
        else:
            reason = str(
                state.get("grounding_result", {}).get(
                    "reason",
                    "generated answer failed grounding verification",
                )
            )
        return {
            "answer": REFUSAL_TEXT,
            "citations": [],
            "grounded": False,
            "refusal_reason": reason,
            "node_trace": [self._trace("refusal", started, reason=reason)],
        }

    def _route_after_evidence(self, state: AgenticRAGState) -> str:
        if state.get("evidence_sufficient", False):
            return "generate"
        if state.get("retrieval_attempts", 0) < self.config.max_retrieval_attempts:
            return "retry"
        return "refuse"

    @staticmethod
    def _route_after_grounding(state: AgenticRAGState) -> str:
        return "complete" if state.get("grounded", False) else "refuse"

    @staticmethod
    def _combine_results(
        filtered_results: RetrievedResults,
        fallback_results: RetrievedResults,
        limit: int,
        attempt: int,
    ) -> RetrievedResults:
        combined: RetrievedResults = []
        seen: set[str] = set()
        for origin, results in (
            ("metadata_filtered", filtered_results),
            ("dense_fallback", fallback_results),
        ):
            for document, score in results:
                chunk_id = str(document.metadata.get("chunk_id", document.page_content))
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                combined.append(
                    (
                        Document(
                            page_content=document.page_content,
                            metadata={
                                **document.metadata,
                                "agentic_retrieval_components": {
                                    "attempt": attempt,
                                    "origin": origin,
                                },
                            },
                        ),
                        score,
                    )
                )
                if len(combined) == limit:
                    return combined
        return combined

    @staticmethod
    def _trace(node: str, started: float, **details: Any) -> dict[str, Any]:
        return {
            "node": node,
            "latency_seconds": round(perf_counter() - started, 6),
            **details,
        }
