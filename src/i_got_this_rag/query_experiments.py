from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import yaml
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from .baseline import DenseRAGResources
from .chunk_experiments import namespace_vector_count, wait_for_vector_count
from .query_transformation import SUPPORTED_QUERY_STRATEGIES, TRANSFORMER_VERSION


@dataclass(frozen=True)
class QueryExperiment:
    experiment_id: str
    experiment_name: str
    query_strategy: str
    query_transformer: str | None
    generated_query_count: int
    fusion: str | None
    rrf_k: int | None
    pinecone_namespace: str
    config_path: Path
    config_sha256: str


def load_query_experiments(config_directory: Path) -> tuple[QueryExperiment, ...]:
    experiments: list[QueryExperiment] = []
    for path in sorted(config_directory.resolve().glob("query_*.yaml")):
        raw_bytes = path.read_bytes()
        payload = yaml.safe_load(raw_bytes) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Query experiment config must be a mapping: {path}")
        required = {
            "experiment_id",
            "experiment_name",
            "query_strategy",
            "query_transformer",
            "generated_query_count",
            "fusion",
            "rrf_k",
            "pinecone_namespace",
        }
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"{path} is missing: {', '.join(sorted(missing))}.")
        transformer = payload["query_transformer"]
        fusion = payload["fusion"]
        rrf_k = payload["rrf_k"]
        experiment = QueryExperiment(
            experiment_id=str(payload["experiment_id"]),
            experiment_name=str(payload["experiment_name"]),
            query_strategy=str(payload["query_strategy"]),
            query_transformer=str(transformer) if transformer is not None else None,
            generated_query_count=int(payload["generated_query_count"]),
            fusion=str(fusion) if fusion is not None else None,
            rrf_k=int(rrf_k) if rrf_k is not None else None,
            pinecone_namespace=str(payload["pinecone_namespace"]),
            config_path=path,
            config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )
        if not experiment.pinecone_namespace.startswith("phase8-"):
            raise ValueError(f"{path}: Phase 8 namespaces must start with 'phase8-'.")
        experiments.append(experiment)

    by_strategy = {experiment.query_strategy: experiment for experiment in experiments}
    if tuple(sorted(by_strategy)) != tuple(sorted(SUPPORTED_QUERY_STRATEGIES)):
        raise ValueError(
            "Phase 8 requires exactly original, rewrite, and multi_query experiments."
        )
    original = by_strategy["original"]
    rewrite = by_strategy["rewrite"]
    multi_query = by_strategy["multi_query"]
    if (
        original.query_transformer is not None
        or original.generated_query_count != 0
        or original.fusion is not None
        or original.rrf_k is not None
    ):
        raise ValueError("The Phase 8 baseline must use only the original query.")
    if (
        rewrite.query_transformer != TRANSFORMER_VERSION
        or rewrite.generated_query_count != 1
        or rewrite.fusion is not None
        or rewrite.rrf_k is not None
    ):
        raise ValueError("The Phase 8 rewrite experiment must use one LLM rewrite.")
    if (
        multi_query.query_transformer != TRANSFORMER_VERSION
        or multi_query.generated_query_count != 2
        or multi_query.fusion != "rrf"
        or multi_query.rrf_k != 60
    ):
        raise ValueError(
            "The Phase 8 multi-query experiment must fuse the original plus two rewrites "
            "with RRF k=60."
        )
    if len(experiments) != 3 or len({item.experiment_id for item in experiments}) != 3:
        raise ValueError("Phase 8 requires exactly three unique experiment IDs.")
    if len({item.pinecone_namespace for item in experiments}) != 1:
        raise ValueError("All Phase 8 experiments must use the same controlled namespace.")
    return original, rewrite, multi_query


def rebuild_phase8_query_namespace(
    resources: DenseRAGResources,
    namespace: str,
    chunks: list[Document],
) -> tuple[PineconeVectorStore, dict[str, Any]]:
    if not namespace.startswith("phase8-"):
        raise ValueError("Refusing to rebuild a namespace outside the 'phase8-' prefix.")

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
    indexed_count = wait_for_vector_count(resources.pinecone_index, namespace, len(chunks))
    return vector_store, {
        "namespace_rebuilt": True,
        "previous_vector_count": existing_count,
        "indexed_vector_count": indexed_count,
        "indexing_latency_seconds": round(perf_counter() - started, 6),
    }


def _rank_penalty(question: dict[str, Any]) -> int:
    return sum(
        rank if rank is not None and rank <= 5 else 6
        for rank in question["expected_source_ranks"].values()
    )


def build_query_transformation_impact(
    baseline_results: dict[str, Any],
    transformed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = {
        question["question_id"]: question for question in baseline_results["questions"]
    }
    strategies: dict[str, Any] = {}
    for experiment_results in transformed_results:
        experiment_id = str(experiment_results["experiment_id"])
        counts = {"improved": 0, "degraded": 0, "unchanged": 0, "unscored": 0}
        questions: list[dict[str, Any]] = []
        quality_reduction_question_ids: list[str] = []
        recall_reduction_question_ids: list[str] = []
        for transformed in experiment_results["questions"]:
            question_id = str(transformed["question_id"])
            baseline = baseline_by_id[question_id]
            baseline_recall = baseline["recall_at_5"]
            transformed_recall = transformed["recall_at_5"]
            recall_reduced = False
            if baseline_recall is None or transformed_recall is None:
                outcome = "unscored"
            else:
                baseline_penalty = _rank_penalty(baseline)
                transformed_penalty = _rank_penalty(transformed)
                if (transformed_recall, -transformed_penalty) > (
                    baseline_recall,
                    -baseline_penalty,
                ):
                    outcome = "improved"
                elif (transformed_recall, -transformed_penalty) < (
                    baseline_recall,
                    -baseline_penalty,
                ):
                    outcome = "degraded"
                    quality_reduction_question_ids.append(question_id)
                    recall_reduced = transformed_recall < baseline_recall
                    if recall_reduced:
                        recall_reduction_question_ids.append(question_id)
                else:
                    outcome = "unchanged"
            counts[outcome] += 1
            questions.append(
                {
                    "question_id": question_id,
                    "category": transformed["category"],
                    "outcome": outcome,
                    "recall_reduced": recall_reduced,
                    "baseline_recall_at_5": baseline_recall,
                    "transformed_recall_at_5": transformed_recall,
                    "baseline_expected_source_ranks": baseline[
                        "expected_source_ranks"
                    ],
                    "transformed_expected_source_ranks": transformed[
                        "expected_source_ranks"
                    ],
                    "retrieval_queries": transformed.get("retrieval_queries", []),
                    "generated_queries": transformed.get("generated_queries", []),
                    "protected_query_terms": transformed.get(
                        "protected_query_terms", []
                    ),
                    "query_guard_triggered": transformed.get(
                        "query_guard_triggered", False
                    ),
                }
            )
        strategies[experiment_id] = {
            "query_transformation_strategy": experiment_results["questions"][0].get(
                "query_transformation_strategy"
            ),
            "outcome_counts": counts,
            "quality_reduction_question_ids": quality_reduction_question_ids,
            "recall_reduction_question_ids": recall_reduction_question_ids,
            "questions": questions,
        }
    return {"baseline_experiment_id": baseline_results["experiment_id"], "strategies": strategies}
