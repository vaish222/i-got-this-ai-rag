from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import yaml
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from .agentic_rag import (
    CitationAttributor,
    CitationGroundingVerifier,
    LangGraphRAG,
)
from .baseline import REFUSAL_TEXT, DenseRAGResources, generate_grounded_answer
from .chunk_experiments import namespace_vector_count, wait_for_vector_count
from .evaluation import (
    EvaluationDataset,
    expected_source_metrics,
    extract_citations,
    serialize_retrieval,
    utc_now,
)
from .reranking import CandidateReranker
from .retrieval import Retriever
from .settings import Settings


FINAL_EVALUATION_VERSION = "phase10-final-v1"
FAITHFULNESS_METHOD = "phase10-deterministic-grounding-v1"
REQUIRED_VERSION_IDS = (
    "baseline_dense",
    "best_chunking",
    "best_embedding",
    "hybrid_retrieval",
    "hybrid_reranker",
    "metadata_aware",
    "query_rewriting",
    "langgraph_workflow",
)


@dataclass(frozen=True)
class FinalVersionSpec:
    version_id: str
    label: str
    mechanism: str
    source_results_path: Path | None
    runtime_experiment_id: str | None


@dataclass(frozen=True)
class FinalEvaluationConfig:
    experiment_id: str
    experiment_name: str
    runtime_namespace: str
    hybrid_candidate_k: int
    final_top_k: int
    rrf_k: int
    bm25_k1: float
    bm25_b: float
    faithfulness_method: str
    versions: tuple[FinalVersionSpec, ...]
    config_path: Path
    config_sha256: str


def load_final_evaluation_config(
    path: Path,
    project_root: Path,
) -> FinalEvaluationConfig:
    path = path.resolve()
    raw_bytes = path.read_bytes()
    payload = yaml.safe_load(raw_bytes) or {}
    if not isinstance(payload, dict):
        raise ValueError("Phase 10 configuration must be a YAML mapping.")
    required = {
        "experiment_id",
        "experiment_name",
        "runtime_namespace",
        "hybrid_candidate_k",
        "final_top_k",
        "rrf_k",
        "bm25_k1",
        "bm25_b",
        "faithfulness_method",
        "versions",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"{path} is missing: {', '.join(sorted(missing))}.")
    if not str(payload["runtime_namespace"]).startswith("phase10-"):
        raise ValueError("Phase 10 runtime namespace must start with 'phase10-'.")
    if str(payload["faithfulness_method"]) != FAITHFULNESS_METHOD:
        raise ValueError(f"faithfulness_method must be {FAITHFULNESS_METHOD}.")

    raw_versions = payload["versions"]
    if not isinstance(raw_versions, list):
        raise ValueError("Phase 10 versions must be a list.")
    versions: list[FinalVersionSpec] = []
    for raw in raw_versions:
        if not isinstance(raw, dict):
            raise ValueError("Each Phase 10 version must be a mapping.")
        source_value = raw.get("source_results_path")
        source_path = (
            (project_root / str(source_value)).resolve()
            if source_value is not None
            else None
        )
        runtime_id = raw.get("runtime_experiment_id")
        if (source_path is None) == (runtime_id is None):
            raise ValueError(
                "Each Phase 10 version must define exactly one source_results_path "
                "or runtime_experiment_id."
            )
        versions.append(
            FinalVersionSpec(
                version_id=str(raw["version_id"]),
                label=str(raw["label"]),
                mechanism=str(raw["mechanism"]),
                source_results_path=source_path,
                runtime_experiment_id=str(runtime_id) if runtime_id is not None else None,
            )
        )
    version_ids = tuple(version.version_id for version in versions)
    if version_ids != REQUIRED_VERSION_IDS:
        raise ValueError(
            "Phase 10 requires the exact ordered version matrix: "
            + ", ".join(REQUIRED_VERSION_IDS)
            + "."
        )
    final_top_k = int(payload["final_top_k"])
    candidate_k = int(payload["hybrid_candidate_k"])
    if final_top_k != 5:
        raise ValueError("Phase 10 final_top_k must remain 5.")
    if candidate_k < final_top_k:
        raise ValueError("hybrid_candidate_k cannot be smaller than final_top_k.")
    return FinalEvaluationConfig(
        experiment_id=str(payload["experiment_id"]),
        experiment_name=str(payload["experiment_name"]),
        runtime_namespace=str(payload["runtime_namespace"]),
        hybrid_candidate_k=candidate_k,
        final_top_k=final_top_k,
        rrf_k=int(payload["rrf_k"]),
        bm25_k1=float(payload["bm25_k1"]),
        bm25_b=float(payload["bm25_b"]),
        faithfulness_method=str(payload["faithfulness_method"]),
        versions=tuple(versions),
        config_path=path,
        config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def normalized_refusal(answer: str) -> bool:
    normalized = answer.strip().replace("’", "'").casefold()
    return normalized == REFUSAL_TEXT.casefold()


class DeterministicFaithfulnessScorer:
    """Binary full-answer grounding with correct refusal handling."""

    def __init__(self) -> None:
        self.attributor = CitationAttributor()
        self.verifier = CitationGroundingVerifier()

    def score(
        self,
        *,
        answerable: bool,
        answer: str,
        results: list[tuple[Document, float]],
    ) -> dict[str, Any]:
        refused = normalized_refusal(answer)
        if not answerable:
            faithful = refused
            return {
                "score": 1.0 if faithful else 0.0,
                "faithful": faithful,
                "correct_refusal": refused,
                "attributed_answer": answer,
                "reason": (
                    "correct explicit refusal for an unanswerable question"
                    if faithful
                    else "unanswerable question did not receive the explicit refusal"
                ),
                "grounding_result": None,
            }
        if refused:
            return {
                "score": 0.0,
                "faithful": False,
                "correct_refusal": False,
                "attributed_answer": answer,
                "reason": "answerable question received a refusal",
                "grounding_result": None,
            }

        attributed_answer = self.attributor.attribute(answer, results)
        grounding = self.verifier.verify(attributed_answer, results)
        return {
            "score": 1.0 if grounding.grounded else 0.0,
            "faithful": grounding.grounded,
            "correct_refusal": False,
            "attributed_answer": attributed_answer,
            "reason": grounding.reason,
            "grounding_result": grounding.to_dict(),
        }


def reconstruct_results(
    question_result: dict[str, Any],
    chunks_by_id: dict[str, Document],
) -> list[tuple[Document, float]]:
    results: list[tuple[Document, float]] = []
    for retrieved in question_result.get("retrieved_chunks", []):
        chunk_id = str(retrieved.get("chunk_id", ""))
        document = chunks_by_id.get(chunk_id)
        if document is None:
            raise ValueError(f"Cannot reconstruct retrieved chunk: {chunk_id}.")
        results.append((document, float(retrieved.get("similarity_score", 0.0))))
    return results


def nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Percentile requires at least one value.")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be between 0 and 1.")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def evaluate_version_artifact(
    spec: FinalVersionSpec,
    results_payload: dict[str, Any],
    dataset: EvaluationDataset,
    chunks_by_id: dict[str, Document],
) -> dict[str, Any]:
    recorded_dataset = results_payload.get("evaluation_dataset", {})
    recorded_sha256 = recorded_dataset.get("sha256")
    if recorded_sha256 is not None and recorded_sha256 != dataset.sha256:
        raise ValueError(
            f"{spec.version_id} was measured against a different evaluation dataset."
        )
    questions = results_payload.get("questions")
    if not isinstance(questions, list) or len(questions) != len(dataset.questions):
        raise ValueError(
            f"{spec.version_id} must contain all {len(dataset.questions)} questions."
        )
    expected_ids = [str(question["question_id"]) for question in dataset.questions]
    actual_ids = [str(question.get("question_id")) for question in questions]
    if actual_ids != expected_ids:
        raise ValueError(f"{spec.version_id} question order or IDs do not match the dataset.")

    scorer = DeterministicFaithfulnessScorer()
    evaluated_questions: list[dict[str, Any]] = []
    for question in questions:
        reconstructed = reconstruct_results(question, chunks_by_id)
        faithfulness = scorer.score(
            answerable=bool(question["answerable"]),
            answer=str(question["generated_answer"]),
            results=reconstructed,
        )
        evaluated_questions.append(
            {
                "question_id": question["question_id"],
                "category": question["category"],
                "answerable": bool(question["answerable"]),
                "recall_at_5": question["recall_at_5"],
                "expected_source_ranks": question["expected_source_ranks"],
                "generated_answer": question["generated_answer"],
                "retrieved_chunks": question["retrieved_chunks"],
                "total_latency_seconds": float(question["total_latency_seconds"]),
                "faithfulness": faithfulness,
            }
        )

    answerable = [q for q in evaluated_questions if q["recall_at_5"] is not None]
    unanswerable = [q for q in evaluated_questions if not q["answerable"]]
    latencies = [q["total_latency_seconds"] for q in evaluated_questions]
    category_summary: dict[str, dict[str, Any]] = {}
    for category in sorted({str(q["category"]) for q in evaluated_questions}):
        category_questions = [q for q in evaluated_questions if q["category"] == category]
        scored_recall = [q for q in category_questions if q["recall_at_5"] is not None]
        category_summary[category] = {
            "question_count": len(category_questions),
            "recall_at_5": (
                mean(float(q["recall_at_5"]) for q in scored_recall)
                if scored_recall
                else None
            ),
            "faithfulness": mean(
                float(q["faithfulness"]["score"]) for q in category_questions
            ),
            "mean_total_latency_seconds": mean(
                float(q["total_latency_seconds"]) for q in category_questions
            ),
        }

    source_summary = results_payload.get("summary", {})
    return {
        "version_id": spec.version_id,
        "label": spec.label,
        "mechanism": spec.mechanism,
        "source_experiment_id": results_payload.get("experiment_id"),
        "metrics": {
            "recall_at_5": mean(float(q["recall_at_5"]) for q in answerable),
            "faithfulness": mean(
                float(q["faithfulness"]["score"]) for q in evaluated_questions
            ),
            "correct_refusal_rate": (
                mean(
                    1.0 if q["faithfulness"]["correct_refusal"] else 0.0
                    for q in unanswerable
                )
                if unanswerable
                else None
            ),
            "average_latency_seconds": mean(latencies),
            "p95_latency_seconds": nearest_rank_percentile(latencies, 0.95),
            "mean_expected_source_rank": source_summary.get(
                "mean_expected_source_rank"
            ),
            "retrieval_failure_count": source_summary.get("retrieval_failure_count"),
        },
        "category_summary": category_summary,
        "questions": evaluated_questions,
    }


class HybridCandidateRerankingRAG:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        reranker: CandidateReranker,
        llm: Any,
        candidate_k: int,
    ) -> None:
        if candidate_k < settings.top_k:
            raise ValueError("candidate_k cannot be smaller than final Top-K.")
        self.settings = settings
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.candidate_k = candidate_k

    def retrieve_with_trace(self, question: str) -> dict[str, Any]:
        retrieval_started = perf_counter()
        candidates = self.retriever.retrieve(question, self.candidate_k)
        candidate_latency = perf_counter() - retrieval_started
        reranking_started = perf_counter()
        results = self.reranker.rerank(question, candidates, self.settings.top_k)
        reranking_latency = perf_counter() - reranking_started
        return {
            "results": results,
            "candidate_results": candidates,
            "candidate_retrieval_latency_seconds": candidate_latency,
            "reranking_latency_seconds": reranking_latency,
            "reranking_enabled": True,
        }

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        return self.retrieve_with_trace(question)["results"]

    def generate(
        self,
        question: str,
        results: list[tuple[Document, float]],
    ) -> str:
        return generate_grounded_answer(self.settings, self.llm, question, results)


def rebuild_phase10_namespace(
    resources: DenseRAGResources,
    namespace: str,
    chunks: list[Document],
) -> tuple[PineconeVectorStore, dict[str, Any]]:
    if not namespace.startswith("phase10-"):
        raise ValueError("Refusing to rebuild a namespace outside the 'phase10-' prefix.")
    started = perf_counter()
    existing_count = namespace_vector_count(resources.pinecone_index, namespace)
    if existing_count:
        resources.pinecone_index.delete(delete_all=True, namespace=namespace)
        wait_for_vector_count(resources.pinecone_index, namespace, 0)
    vector_store = PineconeVectorStore(
        index=resources.pinecone_index,
        embedding=resources.embeddings,
        namespace=namespace,
    )
    point_ids = [
        str(uuid5(NAMESPACE_URL, f"{namespace}:{chunk.metadata['chunk_id']}"))
        for chunk in chunks
    ]
    vector_store.add_documents(documents=chunks, ids=point_ids)
    indexed_count = wait_for_vector_count(
        resources.pinecone_index,
        namespace,
        len(chunks),
    )
    return vector_store, {
        "namespace_rebuilt": True,
        "previous_vector_count": existing_count,
        "indexed_vector_count": indexed_count,
        "indexing_latency_seconds": round(perf_counter() - started, 6),
    }


def evaluate_langgraph(
    graph: LangGraphRAG,
    dataset: EvaluationDataset,
    experiment_id: str,
) -> dict[str, Any]:
    started_at = utc_now()
    question_results: list[dict[str, Any]] = []
    for question in dataset.questions:
        state = graph.invoke(str(question["question"]))
        results = state.get("reranked_docs", [])
        retrieved_chunks = serialize_retrieval(results)
        expected_ids = [str(value) for value in question["expected_source_ids"]]
        source_ranks, best_rank, recall = expected_source_metrics(
            expected_ids,
            retrieved_chunks,
        )
        answer = str(state.get("answer", REFUSAL_TEXT))
        citations = extract_citations(answer, retrieved_chunks)
        trace = list(state.get("node_trace", []))
        generation_latency = sum(
            float(node.get("latency_seconds", 0.0))
            for node in trace
            if node.get("node") == "generation"
        )
        total_latency = float(state.get("total_latency_seconds", 0.0))
        question_results.append(
            {
                "question_id": question["question_id"],
                "question": question["question"],
                "category": question["category"],
                "answerable": bool(question["answerable"]),
                "expected_answer": question["expected_answer"],
                "expected_source_ids": expected_ids,
                "expected_sources": question["expected_sources"],
                "retrieved_chunks": retrieved_chunks,
                "expected_source_ranks": source_ranks,
                "expected_source_rank": best_rank,
                "recall_at_5": recall,
                "generated_answer": answer,
                "citation_labels": [item["label"] for item in citations],
                "citations": citations,
                "retrieval_attempts": int(state.get("retrieval_attempts", 0)),
                "evidence_sufficient": bool(state.get("evidence_sufficient", False)),
                "grounded": bool(state.get("grounded", False)),
                "refusal_reason": state.get("refusal_reason"),
                "node_trace": trace,
                "query_history": state.get("query_history", []),
                "retrieval_history": state.get("retrieval_history", []),
                "evidence_history": state.get("evidence_history", []),
                "grounding_result": state.get("grounding_result"),
                "retrieval_latency_seconds": round(
                    max(0.0, total_latency - generation_latency),
                    6,
                ),
                "llm_latency_seconds": round(generation_latency, 6),
                "generation_latency_seconds": round(generation_latency, 6),
                "total_latency_seconds": round(total_latency, 6),
            }
        )

    answerable_results = [
        result for result in question_results if result["recall_at_5"] is not None
    ]
    found_ranks = [
        rank
        for result in answerable_results
        for rank in result["expected_source_ranks"].values()
        if rank is not None
    ]
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
        },
        "summary": {
            "question_count": len(question_results),
            "recall_at_5": mean(
                float(result["recall_at_5"]) for result in answerable_results
            ),
            "recall_at_5_question_count": len(answerable_results),
            "mean_expected_source_rank": mean(found_ranks) if found_ranks else None,
            "retrieval_failure_count": sum(
                float(result["recall_at_5"]) < 1 for result in answerable_results
            ),
            "mean_retrieval_latency_seconds": mean(
                result["retrieval_latency_seconds"] for result in question_results
            ),
            "mean_generation_latency_seconds": mean(
                result["generation_latency_seconds"] for result in question_results
            ),
            "mean_total_latency_seconds": mean(
                result["total_latency_seconds"] for result in question_results
            ),
        },
        "questions": question_results,
    }


def question_delta_ids(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    metric: str,
) -> tuple[list[str], list[str]]:
    baseline_by_id = {q["question_id"]: q for q in baseline["questions"]}
    improved: list[str] = []
    degraded: list[str] = []
    for question in candidate["questions"]:
        baseline_question = baseline_by_id[question["question_id"]]
        if metric == "recall_at_5":
            baseline_value = baseline_question[metric]
            candidate_value = question[metric]
            if baseline_value is None or candidate_value is None:
                continue
        else:
            baseline_value = baseline_question["faithfulness"]["score"]
            candidate_value = question["faithfulness"]["score"]
        if candidate_value > baseline_value:
            improved.append(str(question["question_id"]))
        elif candidate_value < baseline_value:
            degraded.append(str(question["question_id"]))
    return improved, degraded


def build_final_comparison(
    config: FinalEvaluationConfig,
    evaluated_versions: list[dict[str, Any]],
    dataset: EvaluationDataset,
    source_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    if [row["version_id"] for row in evaluated_versions] != list(REQUIRED_VERSION_IDS):
        raise ValueError("Evaluated versions do not match the required Phase 10 order.")
    baseline = evaluated_versions[0]
    baseline_metrics = baseline["metrics"]
    findings: list[dict[str, Any]] = []
    for version in evaluated_versions:
        metrics = version["metrics"]
        recall_improved, recall_degraded = question_delta_ids(
            baseline,
            version,
            "recall_at_5",
        )
        faith_improved, faith_degraded = question_delta_ids(
            baseline,
            version,
            "faithfulness",
        )
        recall_delta = metrics["recall_at_5"] - baseline_metrics["recall_at_5"]
        faith_delta = metrics["faithfulness"] - baseline_metrics["faithfulness"]
        latency_delta = (
            metrics["average_latency_seconds"]
            - baseline_metrics["average_latency_seconds"]
        )
        if (
            recall_delta >= 0
            and faith_delta >= 0
            and (recall_delta > 0 or faith_delta > 0)
        ):
            assessment = "quality gain with measured latency trade-off"
            value_assessment = "quality improvement demonstrated; inspect latency cost"
        elif recall_delta * faith_delta < 0:
            assessment = "mixed quality trade-off"
            value_assessment = (
                "not demonstrated because one quality metric improved while another regressed"
            )
        elif recall_delta <= 0 and faith_delta <= 0 and (
            recall_delta < 0 or faith_delta < 0
        ):
            assessment = "aggregate quality regression"
            value_assessment = "not demonstrated"
        elif latency_delta < 0:
            assessment = "same aggregate quality at lower measured latency"
            value_assessment = "operational improvement demonstrated"
        elif latency_delta > 0:
            assessment = "same aggregate quality at higher measured latency"
            value_assessment = "not demonstrated"
        else:
            assessment = "baseline-equivalent result"
            value_assessment = "baseline reference"
        findings.append(
            {
                "version_id": version["version_id"],
                "mechanism": version["mechanism"],
                "recall_at_5_delta_vs_baseline": round(recall_delta, 6),
                "faithfulness_delta_vs_baseline": round(faith_delta, 6),
                "average_latency_seconds_delta_vs_baseline": round(
                    latency_delta,
                    6,
                ),
                "recall_improved_question_ids": recall_improved,
                "recall_degraded_question_ids": recall_degraded,
                "faithfulness_improved_question_ids": faith_improved,
                "faithfulness_degraded_question_ids": faith_degraded,
                "assessment": assessment,
                "value_vs_baseline": value_assessment,
            }
        )

    best_recall = max(row["metrics"]["recall_at_5"] for row in evaluated_versions)
    best_faithfulness = max(
        row["metrics"]["faithfulness"] for row in evaluated_versions
    )
    fastest_latency = min(
        row["metrics"]["average_latency_seconds"] for row in evaluated_versions
    )
    target_results = {
        row["version_id"]: {
            "recall_at_5_at_least_0_80": row["metrics"]["recall_at_5"] >= 0.80,
            "faithfulness_at_least_0_90": row["metrics"]["faithfulness"] >= 0.90,
            "p95_latency_under_5_seconds": row["metrics"]["p95_latency_seconds"] < 5,
            "correct_refusal_rate_is_1": row["metrics"]["correct_refusal_rate"] == 1,
        }
        for row in evaluated_versions
    }
    target_qualified = [
        row
        for row in evaluated_versions
        if all(target_results[row["version_id"]].values())
    ]
    recommendation_pool = target_qualified or evaluated_versions
    selected = max(
        recommendation_pool,
        key=lambda row: (
            row["metrics"]["recall_at_5"],
            row["metrics"]["faithfulness"],
            -row["metrics"]["average_latency_seconds"],
        ),
    )
    selected_metrics = selected["metrics"]
    if target_qualified:
        rationale = (
            f"{selected['label']} meets all Phase 10 success targets and ranks highest "
            "by Recall@5, then faithfulness, then lower average latency among qualifying "
            "versions."
        )
    else:
        rationale = (
            f"No version meets every Phase 10 success target. {selected['label']} is the "
            "fallback selected by highest Recall@5, then faithfulness, then lower average "
            "latency."
        )
    return {
        "schema_version": "1.0",
        "phase": 10,
        "evaluation_version": FINAL_EVALUATION_VERSION,
        "experiment_id": config.experiment_id,
        "experiment_name": config.experiment_name,
        "completed_at": utc_now(),
        "evaluation_dataset": {
            "path": dataset.path.as_posix(),
            "sha256": dataset.sha256,
            "question_count": len(dataset.questions),
        },
        "faithfulness": {
            "method": FAITHFULNESS_METHOD,
            "definition": (
                "Answerable questions receive 1 only when every claim is attributable "
                "to retrieved evidence and passes grounding verification; unanswerable "
                "questions receive 1 only for the explicit refusal."
            ),
            "aggregation": "macro mean over all 15 questions",
        },
        "latency": {
            "definition": "mean end-to-end response latency over all 15 questions",
            "comparability_note": (
                "Historical phase artifacts and Phase 10 runtime artifacts were measured "
                "on the same project but not in one simultaneous benchmark window."
            ),
        },
        "best_recall_at_5": best_recall,
        "best_recall_version_ids": [
            row["version_id"]
            for row in evaluated_versions
            if row["metrics"]["recall_at_5"] == best_recall
        ],
        "best_faithfulness": best_faithfulness,
        "best_faithfulness_version_ids": [
            row["version_id"]
            for row in evaluated_versions
            if row["metrics"]["faithfulness"] == best_faithfulness
        ],
        "fastest_average_latency_seconds": fastest_latency,
        "fastest_version_ids": [
            row["version_id"]
            for row in evaluated_versions
            if row["metrics"]["average_latency_seconds"] == fastest_latency
        ],
        "success_targets": {
            "definitions": {
                "recall_at_5": ">= 0.80",
                "faithfulness": ">= 0.90",
                "p95_latency_seconds": "< 5",
                "correct_refusal_rate": "1.0",
            },
            "by_version": target_results,
        },
        "recommendation": {
            "selected_version_id": selected["version_id"],
            "selection_policy": (
                "Prefer versions meeting every success target; otherwise maximize "
                "Recall@5, then faithfulness, then minimize average latency."
            ),
            "rationale": rationale,
            "known_gap": (
                "Generation grounding remains the primary measured weakness; the selected "
                f"version's strict faithfulness is {selected_metrics['faithfulness']:.3f}."
            ),
        },
        "versions": evaluated_versions,
        "findings": findings,
        "source_artifacts": source_artifacts,
    }


def render_analysis_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Phase 10 Final Evaluation",
        "",
        "| Version | Recall@5 | Faithfulness | Avg. latency | p95 latency |",
        "|---|---:|---:|---:|---:|",
    ]
    for version in comparison["versions"]:
        metrics = version["metrics"]
        lines.append(
            f"| {version['label']} | {metrics['recall_at_5']:.3f} | "
            f"{metrics['faithfulness']:.3f} | "
            f"{metrics['average_latency_seconds']:.3f}s | "
            f"{metrics['p95_latency_seconds']:.3f}s |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"**Selected:** {comparison['recommendation']['selected_version_id']}.",
            "",
            comparison["recommendation"]["rationale"],
            "",
            comparison["recommendation"]["known_gap"],
            "",
            "",
            "## Why, regressions, and cost",
            "",
            "The explanations below combine the declared mechanism with measured "
            "question-level changes; causal language is intentionally avoided.",
            "",
        ]
    )
    labels = {row["version_id"]: row["label"] for row in comparison["versions"]}
    for finding in comparison["findings"]:
        lines.extend(
            [
                f"### {labels[finding['version_id']]}",
                "",
                finding["mechanism"],
                "",
                f"- Assessment: {finding['assessment']}.",
                f"- Value vs baseline: {finding['value_vs_baseline']}.",
                f"- Recall@5 delta vs baseline: "
                f"{finding['recall_at_5_delta_vs_baseline']:+.3f}.",
                f"- Faithfulness delta vs baseline: "
                f"{finding['faithfulness_delta_vs_baseline']:+.3f}.",
                f"- Average latency delta vs baseline: "
                f"{finding['average_latency_seconds_delta_vs_baseline']:+.3f}s.",
                f"- Recall improved questions: "
                f"{', '.join(finding['recall_improved_question_ids']) or 'none'}.",
                f"- Recall degraded questions: "
                f"{', '.join(finding['recall_degraded_question_ids']) or 'none'}.",
                f"- Faithfulness improved questions: "
                f"{', '.join(finding['faithfulness_improved_question_ids']) or 'none'}.",
                f"- Faithfulness degraded questions: "
                f"{', '.join(finding['faithfulness_degraded_question_ids']) or 'none'}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Measurement notes",
            "",
            f"- Faithfulness method: `{comparison['faithfulness']['method']}`.",
            f"- {comparison['faithfulness']['definition']}",
            f"- {comparison['latency']['comparability_note']}",
            "- Phase 10 does not implement the experiment dashboard or Streamlit UI.",
            "",
        ]
    )
    return "\n".join(lines)


def write_final_artifacts(
    output_root: Path,
    config_payload: dict[str, Any],
    comparison: dict[str, Any],
) -> tuple[Path, Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "config.json"
    comparison_path = output_root / "comparison.json"
    analysis_path = output_root / "analysis.md"
    config_path.write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    comparison["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    comparison_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    analysis_path.write_text(render_analysis_markdown(comparison), encoding="utf-8")
    return config_path, comparison_path, analysis_path
