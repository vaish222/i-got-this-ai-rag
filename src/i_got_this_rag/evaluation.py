from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Protocol

from langchain_core.documents import Document


CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
REQUIRED_QUESTION_FIELDS = {
    "question_id",
    "question",
    "expected_answer",
    "expected_source_ids",
    "expected_sources",
    "category",
    "answerable",
}


class RAGPipeline(Protocol):
    def retrieve(self, question: str) -> list[tuple[Document, float]]: ...

    def generate(self, question: str, results: list[tuple[Document, float]]) -> str: ...


@dataclass(frozen=True)
class EvaluationDataset:
    path: Path
    schema_version: str
    dataset_name: str
    reference_date: str
    timezone: str
    questions: tuple[dict[str, Any], ...]
    sha256: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_evaluation_dataset(path: Path) -> EvaluationDataset:
    path = path.resolve()
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes)
    questions = payload.get("questions")
    if not isinstance(questions, list) or len(questions) != 15:
        raise ValueError("The Phase 2 evaluation dataset must contain exactly 15 questions.")

    seen_ids: set[str] = set()
    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise ValueError(f"Evaluation question {index} must be a JSON object.")
        missing = REQUIRED_QUESTION_FIELDS - question.keys()
        if missing:
            raise ValueError(f"Evaluation question {index} is missing: {', '.join(sorted(missing))}.")
        question_id = str(question["question_id"])
        if question_id in seen_ids:
            raise ValueError(f"Duplicate evaluation question ID: {question_id}.")
        seen_ids.add(question_id)
        if not isinstance(question["expected_source_ids"], list):
            raise ValueError(f"{question_id}.expected_source_ids must be a list.")
        if bool(question["answerable"]) and not question["expected_source_ids"]:
            raise ValueError(f"Answerable question {question_id} must define at least one expected source.")

    return EvaluationDataset(
        path=path,
        schema_version=str(payload.get("schema_version", "unknown")),
        dataset_name=str(payload.get("dataset_name", path.stem)),
        reference_date=str(payload.get("reference_date", "")),
        timezone=str(payload.get("timezone", "")),
        questions=tuple(questions),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def serialize_retrieval(results: list[tuple[Document, float]]) -> list[dict[str, Any]]:
    retrieved_chunks: list[dict[str, Any]] = []
    for rank, (document, score) in enumerate(results, start=1):
        metadata = document.metadata
        serialized = {
            "rank": rank,
            "document_id": metadata.get("document_id"),
            "document_title": metadata.get("document_title"),
            "chunk_id": metadata.get("chunk_id"),
            "source_path": metadata.get("source_path"),
            "page_number": metadata.get("page_number"),
            "similarity_score": float(score),
        }
        if metadata.get("retrieval_components") is not None:
            serialized["retrieval_components"] = metadata["retrieval_components"]
        if metadata.get("reranking_components") is not None:
            serialized["reranking_components"] = metadata["reranking_components"]
        if metadata.get("metadata_retrieval_components") is not None:
            serialized["metadata_retrieval_components"] = metadata[
                "metadata_retrieval_components"
            ]
        if metadata.get("query_transformation_components") is not None:
            serialized["query_transformation_components"] = metadata[
                "query_transformation_components"
            ]
        retrieved_chunks.append(serialized)
    return retrieved_chunks


def expected_source_metrics(
    expected_source_ids: list[str],
    retrieved_chunks: list[dict[str, Any]],
    cutoff: int = 5,
) -> tuple[dict[str, int | None], int | None, float | None]:
    if not expected_source_ids:
        return {}, None, None

    ranks: dict[str, int | None] = {}
    for source_id in expected_source_ids:
        ranks[source_id] = next(
            (
                int(chunk["rank"])
                for chunk in retrieved_chunks
                if chunk.get("document_id") == source_id
            ),
            None,
        )
    found_ranks = [rank for rank in ranks.values() if rank is not None]
    best_rank = min(found_ranks) if found_ranks else None
    found_at_cutoff = sum(rank is not None and rank <= cutoff for rank in ranks.values())
    return ranks, best_rank, found_at_cutoff / len(expected_source_ids)


def extract_citations(answer: str, retrieved_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = list(dict.fromkeys(f"S{match}" for match in CITATION_PATTERN.findall(answer)))
    chunks_by_label = {f"S{chunk['rank']}": chunk for chunk in retrieved_chunks}
    citations: list[dict[str, Any]] = []
    for label in labels:
        chunk = chunks_by_label.get(label)
        citations.append(
            {
                "label": label,
                "retrieval_rank": chunk.get("rank") if chunk else None,
                "document_id": chunk.get("document_id") if chunk else None,
                "chunk_id": chunk.get("chunk_id") if chunk else None,
                "resolved": chunk is not None,
            }
        )
    return citations


def summarize_by_category(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    categories = sorted({str(result["category"]) for result in results})
    for category in categories:
        category_results = [result for result in results if result["category"] == category]
        scored_results = [result for result in category_results if result["recall_at_5"] is not None]
        expected_ranks = [
            rank
            for result in scored_results
            for rank in result["expected_source_ranks"].values()
            if rank is not None
        ]
        failures: list[dict[str, Any]] = []
        for result in scored_results:
            if result["recall_at_5"] >= 1:
                continue
            missing_ids = [
                source_id
                for source_id, rank in result["expected_source_ranks"].items()
                if rank is None or rank > 5
            ]
            failures.append(
                {
                    "question_id": result["question_id"],
                    "recall_at_5": result["recall_at_5"],
                    "missing_expected_source_ids": missing_ids,
                }
            )

        summaries[category] = {
            "question_count": len(category_results),
            "scored_question_count": len(scored_results),
            "recall_at_5": (
                mean(result["recall_at_5"] for result in scored_results)
                if scored_results
                else None
            ),
            "mean_expected_source_rank": mean(expected_ranks) if expected_ranks else None,
            "retrieval_failure_count": len(failures),
            "failures": failures,
            "mean_retrieval_latency_seconds": mean(
                result["retrieval_latency_seconds"] for result in category_results
            ),
            "mean_reranking_latency_seconds": mean(
                result.get("reranking_latency_seconds", 0.0) for result in category_results
            ),
            "metadata_filter_applied_question_count": sum(
                bool(result.get("metadata_filter_applied", False))
                for result in category_results
            ),
            "metadata_fallback_chunk_count": sum(
                int(result.get("metadata_fallback_result_count", 0))
                for result in category_results
            ),
            "mean_metadata_analysis_latency_seconds": mean(
                result.get("metadata_analysis_latency_seconds", 0.0)
                for result in category_results
            ),
            "query_transformation_enabled_question_count": sum(
                bool(result.get("query_transformation_enabled", False))
                for result in category_results
            ),
            "query_guard_triggered_question_count": sum(
                bool(result.get("query_guard_triggered", False))
                for result in category_results
            ),
            "mean_query_count": mean(
                int(result.get("query_count", 1)) for result in category_results
            ),
            "mean_query_transformation_latency_seconds": mean(
                result.get("query_transformation_latency_seconds", 0.0)
                for result in category_results
            ),
            "mean_llm_latency_seconds": mean(
                result["llm_latency_seconds"] for result in category_results
            ),
            "mean_total_latency_seconds": mean(
                result["total_latency_seconds"] for result in category_results
            ),
        }
    return summaries


def build_question_comparison(experiment_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not experiment_results:
        raise ValueError("At least one experiment result is required for comparison.")
    by_experiment = {
        result["experiment_id"]: {
            question["question_id"]: question for question in result["questions"]
        }
        for result in experiment_results
    }
    comparisons: list[dict[str, Any]] = []
    for question in experiment_results[0]["questions"]:
        comparisons.append(
            {
                "question_id": question["question_id"],
                "category": question["category"],
                "expected_source_ids": question["expected_source_ids"],
                "experiments": {
                    experiment_id: {
                        "recall_at_5": questions[question["question_id"]]["recall_at_5"],
                        "expected_source_rank": questions[question["question_id"]][
                            "expected_source_rank"
                        ],
                        "expected_source_ranks": questions[question["question_id"]][
                            "expected_source_ranks"
                        ],
                        "candidate_recall": questions[question["question_id"]].get(
                            "candidate_recall"
                        ),
                        "candidate_expected_source_ranks": questions[question["question_id"]].get(
                            "candidate_expected_source_ranks"
                        ),
                        "reranking_latency_seconds": questions[question["question_id"]].get(
                            "reranking_latency_seconds",
                            0.0,
                        ),
                        "metadata_filter_applied": questions[question["question_id"]].get(
                            "metadata_filter_applied",
                            False,
                        ),
                        "metadata_constraints": questions[question["question_id"]].get(
                            "metadata_constraints",
                            {},
                        ),
                        "metadata_filter": questions[question["question_id"]].get(
                            "metadata_filter"
                        ),
                        "metadata_filtered_result_count": questions[
                            question["question_id"]
                        ].get("metadata_filtered_result_count", 0),
                        "metadata_fallback_result_count": questions[
                            question["question_id"]
                        ].get("metadata_fallback_result_count", 0),
                        "query_transformation_strategy": questions[
                            question["question_id"]
                        ].get("query_transformation_strategy", "original"),
                        "retrieval_queries": questions[question["question_id"]].get(
                            "retrieval_queries",
                            [questions[question["question_id"]].get("question", "")],
                        ),
                        "generated_queries": questions[question["question_id"]].get(
                            "generated_queries",
                            [],
                        ),
                        "query_guard_triggered": questions[question["question_id"]].get(
                            "query_guard_triggered",
                            False,
                        ),
                        "query_transformation_latency_seconds": questions[
                            question["question_id"]
                        ].get("query_transformation_latency_seconds", 0.0),
                        "retrieval_latency_seconds": questions[question["question_id"]][
                            "retrieval_latency_seconds"
                        ],
                    }
                    for experiment_id, questions in by_experiment.items()
                },
            }
        )
    return comparisons


def diagnose_retrieval_and_reranking_failures(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    by_category: dict[str, dict[str, int]] = {}
    retrieval_source_failures = 0
    reranking_source_failures = 0
    retrieval_question_failures: set[str] = set()
    reranking_question_failures: set[str] = set()

    for result in results:
        if not result["expected_source_ids"]:
            continue
        retrieval_missing: list[str] = []
        reranking_missing: list[str] = []
        for source_id in result["expected_source_ids"]:
            final_rank = result["expected_source_ranks"].get(source_id)
            if final_rank is not None and final_rank <= 5:
                continue
            candidate_rank = result["candidate_expected_source_ranks"].get(source_id)
            if candidate_rank is None:
                retrieval_missing.append(source_id)
            else:
                reranking_missing.append(source_id)

        if not retrieval_missing and not reranking_missing:
            continue
        question_id = str(result["question_id"])
        category = str(result["category"])
        category_counts = by_category.setdefault(
            category,
            {"retrieval_source_failures": 0, "reranking_source_failures": 0},
        )
        if retrieval_missing:
            retrieval_question_failures.add(question_id)
            retrieval_source_failures += len(retrieval_missing)
            category_counts["retrieval_source_failures"] += len(retrieval_missing)
        if reranking_missing:
            reranking_question_failures.add(question_id)
            reranking_source_failures += len(reranking_missing)
            category_counts["reranking_source_failures"] += len(reranking_missing)
        failures.append(
            {
                "question_id": question_id,
                "category": category,
                "retrieval_failure_source_ids": retrieval_missing,
                "reranking_failure_source_ids": reranking_missing,
            }
        )

    return {
        "retrieval_failure_question_count": len(retrieval_question_failures),
        "reranking_failure_question_count": len(reranking_question_failures),
        "retrieval_failure_source_count": retrieval_source_failures,
        "reranking_failure_source_count": reranking_source_failures,
        "by_category": by_category,
        "failures": failures,
    }


class BaselineEvaluator:
    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    def evaluate_question(self, question: dict[str, Any]) -> dict[str, Any]:
        started = perf_counter()
        retrieval_started = perf_counter()
        retrieve_with_trace = getattr(self.pipeline, "retrieve_with_trace", None)
        if callable(retrieve_with_trace):
            trace = retrieve_with_trace(str(question["question"]))
            raw_results = trace["results"]
            candidate_results = trace["candidate_results"]
            candidate_retrieval_latency = float(trace["candidate_retrieval_latency_seconds"])
            reranking_latency = float(trace["reranking_latency_seconds"])
            reranking_enabled = bool(trace["reranking_enabled"])
            metadata_filter_enabled = bool(trace.get("metadata_filter_enabled", False))
            metadata_filter_applied = bool(trace.get("metadata_filter_applied", False))
            metadata_constraints = trace.get("metadata_constraints", {})
            metadata_filter = trace.get("metadata_filter")
            metadata_analysis_latency = float(
                trace.get("metadata_analysis_latency_seconds", 0.0)
            )
            metadata_filtered_result_count = int(
                trace.get("metadata_filtered_result_count", 0)
            )
            metadata_fallback_result_count = int(
                trace.get("metadata_fallback_result_count", 0)
            )
            retrieval_query = str(trace.get("retrieval_query", question["question"]))
            query_transformation_enabled = bool(
                trace.get("query_transformation_enabled", False)
            )
            query_transformation_strategy = str(
                trace.get("query_transformation_strategy", "original")
            )
            query_transformation_version = trace.get("query_transformation_version")
            query_transformation_latency = float(
                trace.get("query_transformation_latency_seconds", 0.0)
            )
            original_query = str(trace.get("original_query", question["question"]))
            retrieval_queries = list(trace.get("retrieval_queries", [retrieval_query]))
            generated_queries = list(trace.get("generated_queries", []))
            protected_query_terms = list(trace.get("protected_query_terms", []))
            query_guard_triggered = bool(trace.get("query_guard_triggered", False))
            query_guard_repairs = list(trace.get("query_guard_repairs", []))
            raw_query_transformation_output = trace.get(
                "raw_query_transformation_output"
            )
            multi_query_fusion = trace.get("multi_query_fusion")
            query_count = int(trace.get("query_count", len(retrieval_queries)))
        else:
            raw_results = self.pipeline.retrieve(str(question["question"]))
            candidate_results = raw_results
            candidate_retrieval_latency = perf_counter() - retrieval_started
            reranking_latency = 0.0
            reranking_enabled = False
            metadata_filter_enabled = False
            metadata_filter_applied = False
            metadata_constraints = {}
            metadata_filter = None
            metadata_analysis_latency = 0.0
            metadata_filtered_result_count = 0
            metadata_fallback_result_count = 0
            retrieval_query = str(question["question"])
            query_transformation_enabled = False
            query_transformation_strategy = "original"
            query_transformation_version = None
            query_transformation_latency = 0.0
            original_query = str(question["question"])
            retrieval_queries = [original_query]
            generated_queries = []
            protected_query_terms = []
            query_guard_triggered = False
            query_guard_repairs = []
            raw_query_transformation_output = None
            multi_query_fusion = None
            query_count = 1
        retrieval_latency = perf_counter() - retrieval_started

        retrieved_chunks = serialize_retrieval(raw_results)
        candidate_chunks = serialize_retrieval(candidate_results)
        generation_started = perf_counter()
        generated_answer = self.pipeline.generate(str(question["question"]), raw_results)
        generation_latency = perf_counter() - generation_started
        total_latency = perf_counter() - started

        expected_ids = [str(source_id) for source_id in question["expected_source_ids"]]
        source_ranks, best_rank, recall_at_5 = expected_source_metrics(expected_ids, retrieved_chunks)
        candidate_source_ranks, candidate_best_rank, candidate_recall = expected_source_metrics(
            expected_ids,
            candidate_chunks,
            cutoff=len(candidate_chunks),
        )
        citations = extract_citations(generated_answer, retrieved_chunks)

        return {
            "question_id": question["question_id"],
            "question": question["question"],
            "category": question["category"],
            "answerable": bool(question["answerable"]),
            "expected_answer": question["expected_answer"],
            "expected_source_ids": expected_ids,
            "expected_sources": question["expected_sources"],
            "candidate_chunks": candidate_chunks,
            "candidate_k": len(candidate_chunks),
            "candidate_expected_source_ranks": candidate_source_ranks,
            "candidate_expected_source_rank": candidate_best_rank,
            "candidate_recall": candidate_recall,
            "retrieved_chunks": retrieved_chunks,
            "expected_source_ranks": source_ranks,
            "expected_source_rank": best_rank,
            "recall_at_5": recall_at_5,
            "generated_answer": generated_answer,
            "citation_labels": [citation["label"] for citation in citations],
            "citations": citations,
            "reranking_enabled": reranking_enabled,
            "retrieval_query": retrieval_query,
            "metadata_filter_enabled": metadata_filter_enabled,
            "metadata_filter_applied": metadata_filter_applied,
            "metadata_constraints": metadata_constraints,
            "metadata_filter": metadata_filter,
            "metadata_filtered_result_count": metadata_filtered_result_count,
            "metadata_fallback_result_count": metadata_fallback_result_count,
            "metadata_analysis_latency_seconds": round(metadata_analysis_latency, 6),
            "query_transformation_enabled": query_transformation_enabled,
            "query_transformation_strategy": query_transformation_strategy,
            "query_transformation_version": query_transformation_version,
            "query_transformation_latency_seconds": round(
                query_transformation_latency,
                6,
            ),
            "original_query": original_query,
            "retrieval_queries": retrieval_queries,
            "generated_queries": generated_queries,
            "protected_query_terms": protected_query_terms,
            "query_guard_triggered": query_guard_triggered,
            "query_guard_repairs": query_guard_repairs,
            "raw_query_transformation_output": raw_query_transformation_output,
            "multi_query_fusion": multi_query_fusion,
            "query_count": query_count,
            "candidate_retrieval_latency_seconds": round(candidate_retrieval_latency, 6),
            "reranking_latency_seconds": round(reranking_latency, 6),
            "retrieval_latency_seconds": round(retrieval_latency, 6),
            "llm_latency_seconds": round(generation_latency, 6),
            "generation_latency_seconds": round(generation_latency, 6),
            "total_latency_seconds": round(total_latency, 6),
        }

    def run(self, dataset: EvaluationDataset, experiment_id: str) -> dict[str, Any]:
        started_at = utc_now()
        results = [self.evaluate_question(question) for question in dataset.questions]
        scored_results = [result for result in results if result["recall_at_5"] is not None]
        candidate_scored_results = [
            result for result in results if result["candidate_recall"] is not None
        ]
        expected_ranks = [
            rank
            for result in scored_results
            for rank in result["expected_source_ranks"].values()
            if rank is not None
        ]
        category_summary = summarize_by_category(results)
        return {
            "schema_version": "1.0",
            "experiment_id": experiment_id,
            "started_at": started_at,
            "completed_at": utc_now(),
            "evaluation_dataset": {
                "name": dataset.dataset_name,
                "schema_version": dataset.schema_version,
                "path": dataset.path.as_posix(),
                "sha256": dataset.sha256,
                "question_count": len(dataset.questions),
                "reference_date": dataset.reference_date,
                "timezone": dataset.timezone,
            },
            "summary": {
                "question_count": len(results),
                "recall_at_5": mean(result["recall_at_5"] for result in scored_results),
                "recall_at_5_question_count": len(scored_results),
                "mean_candidate_recall": mean(
                    result["candidate_recall"] for result in candidate_scored_results
                ),
                "mean_expected_source_rank": mean(expected_ranks) if expected_ranks else None,
                "retrieval_failure_count": sum(
                    summary["retrieval_failure_count"] for summary in category_summary.values()
                ),
                "mean_candidate_retrieval_latency_seconds": mean(
                    result["candidate_retrieval_latency_seconds"] for result in results
                ),
                "mean_reranking_latency_seconds": mean(
                    result["reranking_latency_seconds"] for result in results
                ),
                "metadata_filter_applied_question_count": sum(
                    result["metadata_filter_applied"] for result in results
                ),
                "metadata_fallback_chunk_count": sum(
                    result["metadata_fallback_result_count"] for result in results
                ),
                "mean_metadata_analysis_latency_seconds": mean(
                    result["metadata_analysis_latency_seconds"] for result in results
                ),
                "query_transformation_enabled_question_count": sum(
                    result["query_transformation_enabled"] for result in results
                ),
                "query_guard_triggered_question_count": sum(
                    result["query_guard_triggered"] for result in results
                ),
                "mean_query_count": mean(result["query_count"] for result in results),
                "mean_query_transformation_latency_seconds": mean(
                    result["query_transformation_latency_seconds"] for result in results
                ),
                "mean_retrieval_latency_seconds": mean(
                    result["retrieval_latency_seconds"] for result in results
                ),
                "mean_llm_latency_seconds": mean(result["llm_latency_seconds"] for result in results),
                "mean_generation_latency_seconds": mean(
                    result["generation_latency_seconds"] for result in results
                ),
                "mean_total_latency_seconds": mean(result["total_latency_seconds"] for result in results),
            },
            "category_summary": category_summary,
            "questions": results,
        }


def write_experiment(
    output_directory: Path,
    config: dict[str, Any],
    results: dict[str, Any],
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    config_bytes = (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode()
    results["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    config_path = output_directory / "config.json"
    results_path = output_directory / "results.json"
    config_path.write_bytes(config_bytes)
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return config_path, results_path
