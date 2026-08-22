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


@dataclass(frozen=True)
class RerankingExperiment:
    experiment_id: str
    experiment_name: str
    candidate_k: int
    reranker_enabled: bool
    reranker: str | None
    pinecone_namespace: str
    config_path: Path
    config_sha256: str


def load_reranking_experiments(config_directory: Path) -> tuple[RerankingExperiment, ...]:
    experiments: list[RerankingExperiment] = []
    for path in sorted(config_directory.resolve().glob("reranking_*.yaml")):
        raw_bytes = path.read_bytes()
        payload = yaml.safe_load(raw_bytes) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Reranking experiment config must be a mapping: {path}")
        required = {
            "experiment_id",
            "experiment_name",
            "candidate_k",
            "reranker_enabled",
            "reranker",
            "pinecone_namespace",
        }
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"{path} is missing: {', '.join(sorted(missing))}.")
        reranker_value = payload["reranker"]
        experiment = RerankingExperiment(
            experiment_id=str(payload["experiment_id"]),
            experiment_name=str(payload["experiment_name"]),
            candidate_k=int(payload["candidate_k"]),
            reranker_enabled=bool(payload["reranker_enabled"]),
            reranker=str(reranker_value) if reranker_value is not None else None,
            pinecone_namespace=str(payload["pinecone_namespace"]),
            config_path=path,
            config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )
        if not experiment.pinecone_namespace.startswith("phase6-"):
            raise ValueError(f"{path}: Phase 6 namespaces must start with 'phase6-'.")
        experiments.append(experiment)

    if len(experiments) != 2:
        raise ValueError("Phase 6 requires exactly two experiments: Top-5 and Top-20 plus reranking.")
    baseline = next((item for item in experiments if not item.reranker_enabled), None)
    reranked = next((item for item in experiments if item.reranker_enabled), None)
    if baseline is None or baseline.candidate_k != 5 or baseline.reranker is not None:
        raise ValueError("The Phase 6 baseline must retrieve Top-5 with reranking disabled.")
    if reranked is None or reranked.candidate_k != 20 or reranked.reranker != "bm25":
        raise ValueError("The Phase 6 reranked experiment must retrieve Top-20 and use BM25.")
    if len({experiment.experiment_id for experiment in experiments}) != len(experiments):
        raise ValueError("Phase 6 experiment IDs must be unique.")
    if len({experiment.pinecone_namespace for experiment in experiments}) != 1:
        raise ValueError("Both Phase 6 experiments must use the same controlled dense namespace.")
    return baseline, reranked


def rebuild_phase6_dense_namespace(
    resources: DenseRAGResources,
    namespace: str,
    chunks: list[Document],
) -> tuple[PineconeVectorStore, dict[str, Any]]:
    if not namespace.startswith("phase6-"):
        raise ValueError("Refusing to rebuild a namespace outside the 'phase6-' prefix.")

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
        str(uuid5(NAMESPACE_URL, f"{namespace}:{chunk.metadata['chunk_id']}")) for chunk in chunks
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

