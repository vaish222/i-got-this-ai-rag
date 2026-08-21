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


REQUIRED_RETRIEVAL_STRATEGIES = ("dense", "sparse", "hybrid")


@dataclass(frozen=True)
class RetrievalExperiment:
    experiment_id: str
    experiment_name: str
    retrieval_strategy: str
    pinecone_namespace: str
    config_path: Path
    config_sha256: str


def load_retrieval_experiments(config_directory: Path) -> tuple[RetrievalExperiment, ...]:
    experiments: list[RetrievalExperiment] = []
    for path in sorted(config_directory.resolve().glob("retrieval_*.yaml")):
        raw_bytes = path.read_bytes()
        payload = yaml.safe_load(raw_bytes) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Retrieval experiment config must be a mapping: {path}")
        required = {
            "experiment_id",
            "experiment_name",
            "retrieval_strategy",
            "pinecone_namespace",
        }
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"{path} is missing: {', '.join(sorted(missing))}.")
        experiment = RetrievalExperiment(
            experiment_id=str(payload["experiment_id"]),
            experiment_name=str(payload["experiment_name"]),
            retrieval_strategy=str(payload["retrieval_strategy"]),
            pinecone_namespace=str(payload["pinecone_namespace"]),
            config_path=path,
            config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )
        if not experiment.pinecone_namespace.startswith("phase5-"):
            raise ValueError(f"{path}: Phase 5 namespaces must start with 'phase5-'.")
        experiments.append(experiment)

    strategies = tuple(sorted(experiment.retrieval_strategy for experiment in experiments))
    if strategies != tuple(sorted(REQUIRED_RETRIEVAL_STRATEGIES)):
        raise ValueError(
            "Phase 5 requires exactly one experiment for each strategy: "
            + ", ".join(REQUIRED_RETRIEVAL_STRATEGIES)
            + "."
        )
    if len({experiment.experiment_id for experiment in experiments}) != len(experiments):
        raise ValueError("Phase 5 experiment IDs must be unique.")
    if len({experiment.pinecone_namespace for experiment in experiments}) != 1:
        raise ValueError("All Phase 5 strategies must reference the same controlled dense namespace.")
    return tuple(
        sorted(
            experiments,
            key=lambda experiment: REQUIRED_RETRIEVAL_STRATEGIES.index(
                experiment.retrieval_strategy
            ),
        )
    )


def rebuild_phase5_dense_namespace(
    resources: DenseRAGResources,
    namespace: str,
    chunks: list[Document],
) -> tuple[PineconeVectorStore, dict[str, Any]]:
    if not namespace.startswith("phase5-"):
        raise ValueError("Refusing to rebuild a namespace outside the 'phase5-' prefix.")

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

