from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore

from .baseline import generate_grounded_answer, message_text
from .settings import Settings


TRANSFORMER_VERSION = "phase8-llm-v1"
SUPPORTED_QUERY_STRATEGIES = ("original", "rewrite", "multi_query")
MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
MONTH_DATE_TERM_PATTERN = (
    rf"\b(?:{MONTH_NAMES})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?"
    rf"(?:,?\s+20\d{{2}})?\b"
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Convert the user's question into one concise retrieval-focused search query for a personal and family knowledge base.
Do not answer the question. Do not invent people, events, dates, times, statuses, or obligations.
Preserve every protected term exactly as written, including anonymous IDs, dates, event names, RSVP or gift status, and deadline language.
Add only useful document vocabulary such as preparation, requirements, items to bring, forms, deadlines, schedules, or action items.
Return only the rewritten query with no label, explanation, Markdown, or quotation marks.""",
        ),
        (
            "human",
            "Protected terms: {protected_terms}\nOriginal question: {question}",
        ),
    ]
)

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Create exactly two distinct retrieval-focused search queries for a personal and family knowledge base.
Do not answer the question. Do not invent people, events, dates, times, statuses, or obligations.
Each query must preserve every protected term exactly as written. Vary document vocabulary and retrieval angle while preserving the user's intent.
Return only a JSON array of two strings. Do not use Markdown or add an explanation.""",
        ),
        (
            "human",
            "Protected terms: {protected_terms}\nOriginal question: {question}",
        ),
    ]
)

PROTECTED_PATTERNS = (
    r"\b(?:adult|child|friend|relative|neighbor|mentee|coordinator|colleague)(?:_[a-z]+)*_\d{2}\b",
    r"\b20\d{2}-\d{2}-\d{2}\b",
    MONTH_DATE_TERM_PATTERN,
    r"\b\d{1,2}(?::\d{2})?\s*(?:AM|PM)\b",
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    r"\b(?:this week|this weekend|next week|next weekend|next summer|next year)\b",
    r"\b(?:field trip|picture day|neighborhood potluck|birthday party|course assignment|beginning of school|school events?|social events?|volunteer mentoring)\b",
    r"\b(?:school|students?|activities?|course|assignment|learning|volunteer(?:ing)?|mentor(?:ing)?|household|social|commitments?|invitations?|birthdays?|graduation|summer|everyone)\b",
    r"\b(?:RSVP|gift|gifts|deadline|deadlines|due|pending|completed|purchased|needed)\b",
)

INVENTABLE_FACT_PATTERNS = (
    r"\b(?:adult|child|friend|relative|neighbor|mentee|coordinator|colleague)(?:_[a-z]+)*_\d{2}\b",
    r"\b20\d{2}-\d{2}-\d{2}\b",
    MONTH_DATE_TERM_PATTERN,
    rf"\b(?:{MONTH_NAMES})\.?\s+20\d{{2}}\b",
    r"\b20\d{2}\b",
    r"\b\d{1,2}(?::\d{2})?\s*(?:AM|PM)\b",
)


def extract_protected_terms(question: str) -> tuple[str, ...]:
    matches: list[tuple[int, int, str]] = []
    for pattern in PROTECTED_PATTERNS:
        matches.extend(
            (match.start(), match.end(), match.group(0))
            for match in re.finditer(pattern, question, re.IGNORECASE)
        )
    terms: list[str] = []
    seen: set[str] = set()
    accepted_spans: list[tuple[int, int]] = []
    for start, end, term in sorted(matches, key=lambda item: (item[0], -(item[1] - item[0]))):
        normalized = term.casefold()
        if normalized in seen or any(
            accepted_start <= start and end <= accepted_end
            for accepted_start, accepted_end in accepted_spans
        ):
            continue
        seen.add(normalized)
        accepted_spans.append((start, end))
        terms.append(term)
    return tuple(terms)


def remove_invented_facts(query: str, original_question: str) -> tuple[str, tuple[str, ...]]:
    original_lowered = original_question.casefold()
    invented: list[str] = []
    for pattern in INVENTABLE_FACT_PATTERNS:
        for match in re.finditer(pattern, query, re.IGNORECASE):
            value = match.group(0)
            if value.casefold() not in original_lowered:
                invented.append(value)
    sanitized = query
    for value in sorted(set(invented), key=len, reverse=True):
        sanitized = re.sub(re.escape(value), " ", sanitized, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", sanitized).strip(), tuple(dict.fromkeys(invented))


def clean_generated_query(value: str) -> str:
    query = value.strip()
    query = re.sub(r"^```(?:json|text)?\s*", "", query, flags=re.IGNORECASE)
    query = re.sub(r"\s*```$", "", query)
    query = re.sub(r"^(?:rewritten\s+)?query\s*:\s*", "", query, flags=re.IGNORECASE)
    if len(query) >= 2 and query[0] == query[-1] and query[0] in {'"', "'"}:
        query = query[1:-1].strip()
    return re.sub(r"\s+", " ", query).strip()[:1000]


def parse_multi_query_output(raw_output: str) -> list[str]:
    fenced_blocks = re.findall(
        r"```(?:json)?\s*(.*?)\s*```",
        raw_output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for block in fenced_blocks:
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            return [
                query
                for query in (clean_generated_query(str(item)) for item in payload)
                if query
            ]

    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list):
        queries = [clean_generated_query(str(item)) for item in payload]
    else:
        queries = [
            clean_generated_query(re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line))
            for line in cleaned.splitlines()
            if line.strip()
        ]
    return [query for query in queries if query]


def missing_protected_terms(query: str, protected_terms: tuple[str, ...]) -> tuple[str, ...]:
    lowered = query.casefold()
    return tuple(term for term in protected_terms if term.casefold() not in lowered)


@dataclass(frozen=True)
class QueryTransformation:
    strategy: str
    original_query: str
    retrieval_queries: tuple[str, ...]
    generated_queries: tuple[str, ...]
    protected_terms: tuple[str, ...]
    guard_repairs: tuple[dict[str, Any], ...]
    raw_output: str | None


class LLMQueryTransformer:
    def __init__(
        self,
        strategy: str,
        llm: Any | None,
        reference_date: str,
        timezone: str,
        generated_query_count: int | None = None,
    ) -> None:
        if strategy not in SUPPORTED_QUERY_STRATEGIES:
            raise ValueError(f"Unsupported query strategy: {strategy}")
        if strategy != "original" and llm is None:
            raise ValueError("LLM query transformation requires an LLM.")
        expected_count = {"original": 0, "rewrite": 1, "multi_query": 2}[strategy]
        if generated_query_count is None:
            generated_query_count = expected_count
        if generated_query_count != expected_count:
            raise ValueError(
                f"Phase 8 strategy '{strategy}' requires {expected_count} generated queries."
            )
        self.strategy = strategy
        self.llm = llm
        self.reference_date = reference_date
        self.timezone = timezone
        self.generated_query_count = generated_query_count

    @property
    def enabled(self) -> bool:
        return self.strategy != "original"

    def transform(self, question: str) -> QueryTransformation:
        protected_terms = extract_protected_terms(question)
        if self.strategy == "original":
            return QueryTransformation(
                strategy=self.strategy,
                original_query=question,
                retrieval_queries=(question,),
                generated_queries=(),
                protected_terms=protected_terms,
                guard_repairs=(),
                raw_output=None,
            )

        prompt = REWRITE_PROMPT if self.strategy == "rewrite" else MULTI_QUERY_PROMPT
        prompt_value = prompt.invoke(
            {
                "question": question,
                "protected_terms": ", ".join(protected_terms) or "none",
                "reference_date": self.reference_date,
                "timezone": self.timezone,
            }
        )
        raw_output = message_text(self.llm.invoke(prompt_value).content)
        generated = (
            [clean_generated_query(raw_output)]
            if self.strategy == "rewrite"
            else parse_multi_query_output(raw_output)[: self.generated_query_count]
        )
        generated = [query for query in generated if query]
        repairs: list[dict[str, Any]] = []
        repaired_queries: list[str] = []
        for index, query in enumerate(generated, start=1):
            repaired, invented_terms = remove_invented_facts(query, question)
            if invented_terms:
                repairs.append(
                    {
                        "generated_query_index": index,
                        "reason": "invented_fact_terms",
                        "removed_terms": list(invented_terms),
                    }
                )
            missing = missing_protected_terms(repaired, protected_terms)
            if missing:
                repaired = f"{repaired} {' '.join(missing)}".strip()
                repairs.append(
                    {
                        "generated_query_index": index,
                        "reason": "missing_protected_terms",
                        "missing_terms": list(missing),
                    }
                )
            repaired_queries.append(repaired)

        if not repaired_queries:
            repaired_queries = [question]
            repairs.append(
                {
                    "generated_query_index": None,
                    "reason": "empty_model_output",
                    "missing_terms": list(protected_terms),
                }
            )
        if self.strategy == "multi_query":
            while len(repaired_queries) < self.generated_query_count:
                repaired_queries.append(question)
                repairs.append(
                    {
                        "generated_query_index": len(repaired_queries),
                        "reason": "missing_generated_query",
                        "missing_terms": [],
                    }
                )

        retrieval_queries = (
            tuple(repaired_queries[:1])
            if self.strategy == "rewrite"
            else tuple([question, *repaired_queries])
        )
        return QueryTransformation(
            strategy=self.strategy,
            original_query=question,
            retrieval_queries=retrieval_queries,
            generated_queries=tuple(generated),
            protected_terms=protected_terms,
            guard_repairs=tuple(repairs),
            raw_output=raw_output,
        )


class QueryTransformationRAG:
    def __init__(
        self,
        settings: Settings,
        vector_store: PineconeVectorStore,
        llm: Any,
        transformer: LLMQueryTransformer,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("RRF k must be positive.")
        self.settings = settings
        self.vector_store = vector_store
        self.llm = llm
        self.transformer = transformer
        self.rrf_k = rrf_k

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        return self.retrieve_with_trace(question)["results"]

    def retrieve_with_trace(self, question: str) -> dict[str, Any]:
        transformation_started = perf_counter()
        transformation = self.transformer.transform(question)
        transformation_latency = perf_counter() - transformation_started

        retrieval_started = perf_counter()
        result_sets = [
            self.vector_store.similarity_search_with_score(query, k=self.settings.top_k)
            for query in transformation.retrieval_queries
        ]
        results = (
            result_sets[0]
            if len(result_sets) == 1
            else self._fuse(transformation.retrieval_queries, result_sets)
        )
        retrieval_latency = perf_counter() - retrieval_started
        return {
            "results": results,
            "candidate_results": results,
            "candidate_retrieval_latency_seconds": retrieval_latency,
            "reranking_latency_seconds": 0.0,
            "reranking_enabled": False,
            "metadata_filter_enabled": False,
            "metadata_filter_applied": False,
            "query_transformation_enabled": self.transformer.enabled,
            "query_transformation_strategy": transformation.strategy,
            "query_transformation_version": TRANSFORMER_VERSION,
            "query_transformation_latency_seconds": transformation_latency,
            "original_query": transformation.original_query,
            "retrieval_query": transformation.retrieval_queries[0],
            "retrieval_queries": list(transformation.retrieval_queries),
            "generated_queries": list(transformation.generated_queries),
            "protected_query_terms": list(transformation.protected_terms),
            "query_guard_triggered": bool(transformation.guard_repairs),
            "query_guard_repairs": list(transformation.guard_repairs),
            "raw_query_transformation_output": transformation.raw_output,
            "multi_query_fusion": "rrf" if len(result_sets) > 1 else None,
            "query_count": len(transformation.retrieval_queries),
        }

    def _fuse(
        self,
        queries: tuple[str, ...],
        result_sets: list[list[tuple[Document, float]]],
    ) -> list[tuple[Document, float]]:
        documents: dict[str, Document] = {}
        fused_scores: Counter[str] = Counter()
        components: dict[str, list[dict[str, Any]]] = {}
        for query_index, (query, results) in enumerate(
            zip(queries, result_sets, strict=True),
            start=1,
        ):
            for rank, (document, dense_score) in enumerate(results, start=1):
                chunk_id = str(document.metadata["chunk_id"])
                documents.setdefault(chunk_id, document)
                fused_scores[chunk_id] += 1 / (self.rrf_k + rank)
                components.setdefault(chunk_id, []).append(
                    {
                        "query_index": query_index,
                        "query": query,
                        "rank": rank,
                        "dense_score": float(dense_score),
                    }
                )

        ranked_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id),
        )[: self.settings.top_k]
        return [
            (
                Document(
                    page_content=documents[chunk_id].page_content,
                    metadata={
                        **documents[chunk_id].metadata,
                        "query_transformation_components": {
                            "rrf_k": self.rrf_k,
                            "query_matches": components[chunk_id],
                        },
                    },
                ),
                float(fused_scores[chunk_id]),
            )
            for chunk_id in ranked_ids
        ]

    def generate(self, question: str, results: list[tuple[Document, float]]) -> str:
        return generate_grounded_answer(self.settings, self.llm, question, results)
