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
from .metadata_retrieval import ANALYZER_VERSION


@dataclass(frozen=True)
class MetadataExperiment:
    experiment_id: str
    experiment_name: str
    metadata_filter_enabled: bool
    metadata_analyzer: str | None
    fallback_to_unfiltered: bool
    pinecone_namespace: str
    config_path: Path
    config_sha256: str


def load_metadata_experiments(config_directory: Path) -> tuple[MetadataExperiment, ...]:
    experiments: list[MetadataExperiment] = []
    for path in sorted(config_directory.resolve().glob("metadata_*.yaml")):
        raw_bytes = path.read_bytes()
        payload = yaml.safe_load(raw_bytes) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Metadata experiment config must be a mapping: {path}")
        required = {
            "experiment_id",
            "experiment_name",
            "metadata_filter_enabled",
            "metadata_analyzer",
            "fallback_to_unfiltered",
            "pinecone_namespace",
        }
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"{path} is missing: {', '.join(sorted(missing))}.")
        analyzer = payload["metadata_analyzer"]
        experiment = MetadataExperiment(
            experiment_id=str(payload["experiment_id"]),
            experiment_name=str(payload["experiment_name"]),
            metadata_filter_enabled=bool(payload["metadata_filter_enabled"]),
            metadata_analyzer=str(analyzer) if analyzer is not None else None,
            fallback_to_unfiltered=bool(payload["fallback_to_unfiltered"]),
            pinecone_namespace=str(payload["pinecone_namespace"]),
            config_path=path,
            config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )
        if not experiment.pinecone_namespace.startswith("phase7-"):
            raise ValueError(f"{path}: Phase 7 namespaces must start with 'phase7-'.")
        experiments.append(experiment)

    if len(experiments) != 2:
        raise ValueError(
            "Phase 7 requires exactly two experiments: unfiltered and metadata-filtered."
        )
    unfiltered = next((item for item in experiments if not item.metadata_filter_enabled), None)
    filtered = next((item for item in experiments if item.metadata_filter_enabled), None)
    if unfiltered is None or unfiltered.metadata_analyzer is not None:
        raise ValueError("The Phase 7 unfiltered baseline must disable metadata analysis.")
    if filtered is None or filtered.metadata_analyzer != ANALYZER_VERSION:
        raise ValueError(
            f"The Phase 7 filtered experiment must use metadata_analyzer={ANALYZER_VERSION}."
        )
    if not filtered.fallback_to_unfiltered:
        raise ValueError(
            "The Phase 7 filtered experiment must preserve dense fallback retrieval."
        )
    if len({item.experiment_id for item in experiments}) != len(experiments):
        raise ValueError("Phase 7 experiment IDs must be unique.")
    if len({item.pinecone_namespace for item in experiments}) != 1:
        raise ValueError("Both Phase 7 experiments must use the same controlled namespace.")
    return unfiltered, filtered


def rebuild_phase7_metadata_namespace(
    resources: DenseRAGResources,
    namespace: str,
    chunks: list[Document],
) -> tuple[PineconeVectorStore, dict[str, Any]]:
    if not namespace.startswith("phase7-"):
        raise ValueError("Refusing to rebuild a namespace outside the 'phase7-' prefix.")

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
        "metadata_facet_version": ANALYZER_VERSION,
    }


def build_metadata_impact(
    unfiltered_results: dict[str, Any],
    filtered_results: dict[str, Any],
) -> dict[str, Any]:
    baseline_by_id = {
        question["question_id"]: question for question in unfiltered_results["questions"]
    }
    filtered_by_id = {
        question["question_id"]: question for question in filtered_results["questions"]
    }
    changes: list[dict[str, Any]] = []
    counts = {"improved": 0, "degraded": 0, "unchanged": 0, "unscored": 0}
    for question_id, baseline in baseline_by_id.items():
        filtered = filtered_by_id[question_id]
        baseline_recall = baseline["recall_at_5"]
        filtered_recall = filtered["recall_at_5"]
        if baseline_recall is None or filtered_recall is None:
            outcome = "unscored"
        else:
            baseline_penalty = sum(
                rank if rank is not None and rank <= 5 else 6
                for rank in baseline["expected_source_ranks"].values()
            )
            filtered_penalty = sum(
                rank if rank is not None and rank <= 5 else 6
                for rank in filtered["expected_source_ranks"].values()
            )
            if (filtered_recall, -filtered_penalty) > (baseline_recall, -baseline_penalty):
                outcome = "improved"
            elif (filtered_recall, -filtered_penalty) < (baseline_recall, -baseline_penalty):
                outcome = "degraded"
            else:
                outcome = "unchanged"
        counts[outcome] += 1
        changes.append(
            {
                "question_id": question_id,
                "category": baseline["category"],
                "outcome": outcome,
                "unfiltered_recall_at_5": baseline_recall,
                "filtered_recall_at_5": filtered_recall,
                "unfiltered_expected_source_ranks": baseline["expected_source_ranks"],
                "filtered_expected_source_ranks": filtered["expected_source_ranks"],
                "metadata_constraints": filtered.get("metadata_constraints", {}),
                "metadata_filter": filtered.get("metadata_filter"),
                "metadata_filtered_result_count": filtered.get(
                    "metadata_filtered_result_count", 0
                ),
                "metadata_fallback_result_count": filtered.get(
                    "metadata_fallback_result_count", 0
                ),
            }
        )
    return {"outcome_counts": counts, "questions": changes}
