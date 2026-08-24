from __future__ import annotations

import re
from statistics import mean
from time import perf_counter
from typing import Any, Sequence

from langchain_core.documents import Document

from .baseline import REFUSAL_TEXT
from .answer_routing import ANSWER_ROUTING_CURRENT, detect_answer_scope
from .chat_models import (
    ChatModelConfigurationError,
    MissingChatModelAPIKeyError,
)
from .conversation import ConversationQueryRewriter, ConversationTurn
from .evaluation import (
    EvaluationDataset,
    expected_source_metrics,
    extract_citations,
    serialize_retrieval,
)
from .final_evaluation import (
    DeterministicFaithfulnessScorer,
    nearest_rank_percentile,
)
from .grounded_generation import GroundedGeneration
from .retrieval import lexical_tokens
from .user_interface import (
    ANONYMOUS_IDENTIFIER_PATTERN,
    CLARIFICATION_TEXT,
    AnswerView,
    answer_question,
    select_relevant_ui_results,
)


CURRENT_APP_EVALUATION_VERSION = "phase10-current-app-v3"
CURRENT_APP_EXPERIMENT_ID = "E804_phase10_scoped_answer_routing"
BULLET_PATTERN = re.compile(r"^\s*[-*+]\s+(.+?)\s*$")
CORRECTNESS_STOP_WORDS = {
    "a", "an", "and", "are", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "of", "on", "or", "the", "to", "with",
    "your", "we", "what", "which", "when", "should", "need",
}


class CapturingPipeline:
    """Record the retrieval used by the real Streamlit answer path."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.settings = pipeline.settings
        self.resources = pipeline.resources
        self.last_results: list[tuple[Document, float]] = []
        self.last_scoped_results: list[tuple[Document, float]] = []
        self.last_generation_trace: dict[str, Any] | None = None

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        self.last_results = self.pipeline.retrieve(question)
        return self.last_results

    def retrieve_scoped(
        self,
        question: str,
        scope: Any,
    ) -> list[tuple[Document, float]]:
        retrieve_scoped = getattr(self.pipeline, "retrieve_scoped", None)
        if not callable(retrieve_scoped):
            return []
        self.last_scoped_results = retrieve_scoped(question, scope)
        return self.last_scoped_results

    def generate(
        self,
        question: str,
        results: list[tuple[Document, float]],
    ) -> str | GroundedGeneration:
        generated = self.pipeline.generate(question, results)
        self.last_generation_trace = getattr(
            self.pipeline,
            "last_generation_trace",
            None,
        )
        return generated


def _confirmed_answer(answer: str) -> str:
    confirmed = answer.split("\nOptional suggestions", maxsplit=1)[0].strip()
    return re.sub(
        r"^Confirmed from your information\s*\n?",
        "",
        confirmed,
        count=1,
        flags=re.IGNORECASE,
    ).strip()


def answer_relevance_correctness(
    expected_answer: str,
    generated_answer: str,
    *,
    answerable: bool,
) -> float:
    if not answerable:
        return 1.0 if generated_answer.strip() == REFUSAL_TEXT else 0.0

    def terms(value: str) -> set[str]:
        return {
            term
            for term in lexical_tokens(value)
            if len(term) > 1
            and term not in CORRECTNESS_STOP_WORDS
            and not re.fullmatch(r"s\d+", term)
        }

    expected = terms(expected_answer)
    actual = terms(_confirmed_answer(generated_answer))
    if not expected or not actual:
        return 0.0
    overlap = len(expected & actual)
    precision = overlap / len(actual)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def classify_generation_error(exc: Exception) -> dict[str, str]:
    name = type(exc).__name__
    message = str(exc).strip() or name
    lowered = f"{name} {message}".casefold()
    if isinstance(exc, MissingChatModelAPIKeyError):
        error_type = "missing_api_key"
    elif isinstance(exc, ChatModelConfigurationError):
        error_type = "invalid_provider_configuration"
    elif "timeout" in lowered or "timed out" in lowered:
        error_type = "api_timeout"
    elif (
        "notfound" in lowered
        or "not found" in lowered
        or "unknown model" in lowered
        or "404" in lowered
    ):
        error_type = "unavailable_model"
    elif (
        "validationerror" in lowered
        or "jsondecode" in lowered
        or "invalid json" in lowered
        or "malformed" in lowered
        or "structured output" in lowered
        or "response_format" in lowered
        or "json schema" in lowered
    ):
        error_type = "malformed_model_response"
    else:
        error_type = "model_provider_error"
    safe_message = re.sub(
        r"(?i)(?:bearer\s+|api[_ -]?key[=:]\s*)[A-Za-z0-9._-]+",
        "[redacted credential]",
        message,
    )
    return {
        "type": error_type,
        "exception": name,
        "message": safe_message[:500],
    }


def _selected_results(
    pipeline: CapturingPipeline,
    response: AnswerView,
) -> list[tuple[Document, float]]:
    answer_scope = detect_answer_scope(response.retrieval_question)
    candidate_results = pipeline.last_scoped_results or pipeline.last_results
    return select_relevant_ui_results(
        response.retrieval_question,
        candidate_results,
        pipeline.settings.reference_date,
        answer_scope=answer_scope,
        answer_routing_mode=getattr(
            pipeline.settings,
            "answer_routing_mode",
            ANSWER_ROUTING_CURRENT,
        ),
    )


def evaluate_current_app(
    pipeline: Any,
    dataset: EvaluationDataset,
    *,
    continue_on_generation_error: bool = False,
) -> dict[str, Any]:
    captured = CapturingPipeline(pipeline)
    scorer = DeterministicFaithfulnessScorer()
    question_results: list[dict[str, Any]] = []

    for question in dataset.questions:
        captured.last_results = []
        captured.last_scoped_results = []
        captured.last_generation_trace = None
        started = perf_counter()
        generation_error: dict[str, str] | None = None
        try:
            response = answer_question(
                captured,
                str(question["question"]),
                reference_date=dataset.reference_date,
            )
        except Exception as exc:
            if not continue_on_generation_error:
                raise
            generation_error = classify_generation_error(exc)
            response = AnswerView(
                question=str(question["question"]),
                retrieval_question=str(question["question"]),
                answer="",
                sources=(),
                used_conversation_context=False,
            )
        total_latency = perf_counter() - started
        results = _selected_results(captured, response)
        raw_retrieved_chunks = serialize_retrieval(captured.last_results)
        retrieved_chunks = serialize_retrieval(results)
        expected_ids = [str(value) for value in question["expected_source_ids"]]
        source_ranks, best_rank, recall_at_5 = expected_source_metrics(
            expected_ids,
            raw_retrieved_chunks,
        )
        faithfulness = scorer.score(
            answerable=bool(question["answerable"]),
            answer=_confirmed_answer(response.answer),
            results=results,
        )
        correctness = answer_relevance_correctness(
            str(question["expected_answer"]),
            response.answer,
            answerable=bool(question["answerable"]),
        )
        citations = extract_citations(response.answer, retrieved_chunks)
        question_results.append(
            {
                "question_id": str(question["question_id"]),
                "question": str(question["question"]),
                "category": str(question["category"]),
                "answerable": bool(question["answerable"]),
                "expected_answer": str(question["expected_answer"]),
                "expected_source_ids": expected_ids,
                "retrieval_question": response.retrieval_question,
                "raw_retrieved_chunks": raw_retrieved_chunks,
                "retrieved_chunks": retrieved_chunks,
                "retrieved_source_ids": [
                    str(chunk.get("document_id"))
                    for chunk in raw_retrieved_chunks
                    if chunk.get("document_id") is not None
                ],
                "answer_context_source_ids": [
                    str(chunk.get("document_id"))
                    for chunk in retrieved_chunks
                    if chunk.get("document_id") is not None
                ],
                "expected_source_ranks": source_ranks,
                "expected_source_rank": best_rank,
                "recall_at_5": recall_at_5,
                "generated_answer": response.answer,
                "citation_labels": [item["label"] for item in citations],
                "citations": citations,
                "faithfulness": faithfulness,
                "answer_relevance_correctness": round(correctness, 6),
                "generation_trace": response.generation_trace
                or captured.last_generation_trace,
                "generation_error": generation_error,
                "total_latency_seconds": round(total_latency, 6),
            }
        )

    answerable = [item for item in question_results if item["recall_at_5"] is not None]
    unanswerable = [item for item in question_results if not item["answerable"]]
    expected_ranks = [
        rank
        for item in answerable
        for rank in item["expected_source_ranks"].values()
        if rank is not None
    ]
    latencies = [float(item["total_latency_seconds"]) for item in question_results]
    categories: dict[str, dict[str, Any]] = {}
    for category in sorted({str(item["category"]) for item in question_results}):
        items = [item for item in question_results if item["category"] == category]
        recall_items = [item for item in items if item["recall_at_5"] is not None]
        categories[category] = {
            "question_count": len(items),
            "recall_at_5": (
                mean(float(item["recall_at_5"]) for item in recall_items)
                if recall_items
                else None
            ),
            "faithfulness": mean(
                float(item["faithfulness"]["score"]) for item in items
            ),
            "answer_relevance_correctness": mean(
                float(item["answer_relevance_correctness"]) for item in items
            ),
            "mean_total_latency_seconds": mean(
                float(item["total_latency_seconds"]) for item in items
            ),
            "generation_failure_count": sum(
                item["generation_error"] is not None for item in items
            ),
        }

    return {
        "metrics": {
            "recall_at_5": mean(float(item["recall_at_5"]) for item in answerable),
            "faithfulness": mean(
                float(item["faithfulness"]["score"]) for item in question_results
            ),
            "answer_relevance_correctness": mean(
                float(item["answer_relevance_correctness"])
                for item in question_results
            ),
            "correct_refusal_rate": (
                mean(
                    1.0 if item["faithfulness"]["correct_refusal"] else 0.0
                    for item in unanswerable
                )
                if unanswerable
                else None
            ),
            "average_latency_seconds": mean(latencies),
            "p95_latency_seconds": nearest_rank_percentile(latencies, 0.95),
            "mean_expected_source_rank": (
                mean(expected_ranks) if expected_ranks else None
            ),
            "retrieval_failure_count": sum(
                float(item["recall_at_5"]) < 1.0 for item in answerable
            ),
            "generation_failure_count": sum(
                item["generation_error"] is not None for item in question_results
            ),
            "generation_success_rate": mean(
                1.0 if item["generation_error"] is None else 0.0
                for item in question_results
            ),
        },
        "category_summary": categories,
        "questions": question_results,
    }


def _duplicate_bullets(answer: str) -> tuple[str, ...]:
    bullets = [
        re.sub(r"\[S\d+\]", "", match.group(1)).casefold().strip()
        for line in answer.splitlines()
        if (match := BULLET_PATTERN.match(line))
    ]
    return tuple(sorted({item for item in bullets if bullets.count(item) > 1}))


def _base_display_failures(answer: str) -> list[str]:
    failures: list[str] = []
    if ANONYMOUS_IDENTIFIER_PATTERN.search(answer):
        failures.append("answer exposes an anonymous person identifier")
    duplicates = _duplicate_bullets(answer)
    if duplicates:
        failures.append("answer repeats bullet items: " + ", ".join(duplicates))
    return failures


def _case_result(
    case_id: str,
    question: str,
    response: AnswerView,
    failures: list[str],
    latency_seconds: float,
) -> dict[str, Any]:
    failures = [*_base_display_failures(response.answer), *failures]
    return {
        "case_id": case_id,
        "question": question,
        "passed": not failures,
        "failures": failures,
        "answer": response.answer,
        "source_labels": [source.label for source in response.sources],
        "latency_seconds": round(latency_seconds, 6),
    }


def evaluate_ui_regressions(pipeline: Any) -> dict[str, Any]:
    captured = CapturingPipeline(pipeline)
    results: list[dict[str, Any]] = []

    def ask(
        question: str,
        history: Sequence[ConversationTurn] = (),
    ) -> tuple[AnswerView, float]:
        rewriter = None
        if history:
            rewriter = ConversationQueryRewriter(
                llm=captured.resources.llm,
                reference_date=captured.settings.reference_date,
                timezone=captured.settings.timezone,
                memory_exchanges=3,
            )
        started = perf_counter()
        response = answer_question(
            captured,
            question,
            history=history,
            rewriter=rewriter,
            reference_date=captured.settings.reference_date,
        )
        return response, perf_counter() - started

    question = "What's coming up this week?"
    response, latency = ask(question)
    failures: list[str] = []
    if response.answer.count("Here’s what’s coming up this week:") != 1:
        failures.append("weekly summary does not have exactly one opening")
    if response.answer.count("neighborhood potluck RSVP deadline") != 1:
        failures.append("weekly summary duplicates or omits the potluck RSVP deadline")
    if "**Friday, August 21**" not in response.answer:
        failures.append("weekly summary is missing Friday, August 21")
    results.append(_case_result("UI001", question, response, failures, latency))

    question = "Which invitations still need an RSVP?"
    response, latency = ask(question)
    failures = []
    if "6 invitations still need a response" not in response.answer:
        failures.append("pending RSVP count is not six")
    if "picture day" in response.answer.casefold():
        failures.append("RSVP answer includes an unrelated picture-day item")
    results.append(_case_result("UI002", question, response, failures, latency))

    question = "Is there any volunteer work this week?"
    volunteer_response, latency = ask(question)
    failures = []
    if "7 volunteer commitments" not in volunteer_response.answer:
        failures.append("volunteer answer does not contain seven commitments")
    if "welcome table" not in volunteer_response.answer.casefold():
        failures.append("volunteer answer omits the welcome-table commitment")
    if "september 9" in volunteer_response.answer.casefold():
        failures.append("volunteer answer includes an out-of-week meeting")
    results.append(
        _case_result("UI003", question, volunteer_response, failures, latency)
    )

    question = "what's next?"
    response, latency = ask(question)
    failures = []
    if response.answer != CLARIFICATION_TEXT:
        failures.append("underspecified question does not ask for clarification")
    if response.sources:
        failures.append("clarification response incorrectly includes sources")
    results.append(_case_result("UI004", question, response, failures, latency))

    question = "Is there any other volunteer work?"
    history = (
        ConversationTurn(
            "Is there any volunteer work this week?",
            volunteer_response.answer,
        ),
    )
    response, latency = ask(question, history)
    failures = []
    if "additional volunteer work due this week" not in response.answer.casefold():
        failures.append("additive follow-up does not preserve volunteer/week scope")
    results.append(_case_result("UI005", question, response, failures, latency))

    question = "Which birthdays still need gifts?"
    birthday_response, latency = ask(question)
    failures = []
    if not birthday_response.sources:
        failures.append("birthday answer has no resolved source")
    if "gift" not in birthday_response.answer.casefold():
        failures.append("birthday answer does not discuss gifts")
    results.append(_case_result("UI006", question, birthday_response, failures, latency))

    question = "When is my next volunteer work planned?"
    history = (
        ConversationTurn(
            "Which birthdays still need gifts?",
            birthday_response.answer,
        ),
    )
    response, latency = ask(question, history)
    failures = []
    if "birthday" in response.answer.casefold() or "gift" in response.answer.casefold():
        failures.append("standalone volunteer question was mixed with birthday context")
    if not response.sources:
        failures.append("standalone volunteer answer has no resolved source")
    results.append(_case_result("UI007", question, response, failures, latency))

    question = "Plan my week."
    response, latency = ask(question)
    failures = []
    if response.answer.count("Here’s what’s coming up this week:") != 1:
        failures.append("weekly planning phrase does not use the weekly agenda")
    if "**Friday, August 21**" not in response.answer:
        failures.append("weekly plan is missing Friday, August 21")
    if "Monday, August 24" in response.answer or "September" in response.answer:
        failures.append("weekly plan includes an event outside the requested week")
    if _duplicate_bullets(response.answer):
        failures.append("weekly plan includes duplicate events")
    results.append(_case_result("UI008", question, response, failures, latency))

    question = "What is the meal plan for Sunday?"
    response, latency = ask(question)
    failures = []
    normalized_answer = response.answer.casefold()
    if "neighborhood potluck" not in normalized_answer:
        failures.append("Sunday meal plan omits the neighborhood potluck")
    if "family is bringing lemon bars" not in normalized_answer:
        failures.append("Sunday meal plan omits the preparation note")
    if "sheet-pan chicken" in normalized_answer or "robotics" in normalized_answer:
        failures.append("Sunday meal plan includes Saturday's meal row")
    if not response.sources:
        failures.append("Sunday meal plan has no resolved source")
    results.append(_case_result("UI009", question, response, failures, latency))

    passed = sum(bool(item["passed"]) for item in results)
    return {
        "case_count": len(results),
        "passed_count": passed,
        "failed_count": len(results) - passed,
        "pass_rate": passed / len(results),
        "cases": results,
    }


def historical_baseline_delta(
    metrics: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, float]:
    baseline = next(
        item for item in comparison["versions"] if item["version_id"] == "baseline_dense"
    )["metrics"]
    return {
        "recall_at_5": float(metrics["recall_at_5"]) - float(baseline["recall_at_5"]),
        "faithfulness": float(metrics["faithfulness"])
        - float(baseline["faithfulness"]),
        "average_latency_seconds": float(metrics["average_latency_seconds"])
        - float(baseline["average_latency_seconds"]),
    }
