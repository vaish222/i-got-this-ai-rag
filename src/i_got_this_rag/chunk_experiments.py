from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import yaml
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from .baseline import DenseRAGResources


REQUIRED_CHUNK_SIZES = (250, 500, 750, 1000)


@dataclass(frozen=True)
class ChunkExperiment:
    experiment_id: str
    experiment_name: str
    chunk_size: int
    chunk_overlap: int
    pinecone_namespace: str
    config_path: Path
    config_sha256: str


def load_chunk_experiments(
    config_directory: Path,
    baseline_namespace: str,
) -> tuple[ChunkExperiment, ...]:
    experiments: list[ChunkExperiment] = []
    for path in sorted(config_directory.resolve().glob("chunk_*.yaml")):
        raw_bytes = path.read_bytes()
        payload = yaml.safe_load(raw_bytes) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Chunk experiment config must be a mapping: {path}")
        required = {
            "experiment_id",
            "experiment_name",
            "chunk_size",
            "chunk_overlap",
            "pinecone_namespace",
        }
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"{path} is missing: {', '.join(sorted(missing))}.")
        experiment = ChunkExperiment(
            experiment_id=str(payload["experiment_id"]),
            experiment_name=str(payload["experiment_name"]),
            chunk_size=int(payload["chunk_size"]),
            chunk_overlap=int(payload["chunk_overlap"]),
            pinecone_namespace=str(payload["pinecone_namespace"]),
            config_path=path,
            config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )
        if experiment.chunk_size <= 0:
            raise ValueError(f"{path}: chunk_size must be positive.")
        if not 0 <= experiment.chunk_overlap < experiment.chunk_size:
            raise ValueError(f"{path}: chunk_overlap must be smaller than chunk_size.")
        if not experiment.pinecone_namespace.startswith("phase3-"):
            raise ValueError(f"{path}: Phase 3 namespaces must start with 'phase3-'.")
        if experiment.pinecone_namespace == baseline_namespace:
            raise ValueError(f"{path}: Phase 3 cannot overwrite the baseline namespace.")
        experiments.append(experiment)

    sizes = tuple(sorted(experiment.chunk_size for experiment in experiments))
    if sizes != REQUIRED_CHUNK_SIZES:
        raise ValueError(
            "Phase 3 requires exactly one experiment for each chunk size: "
            + ", ".join(str(size) for size in REQUIRED_CHUNK_SIZES)
            + "."
        )
    if len({experiment.experiment_id for experiment in experiments}) != len(experiments):
        raise ValueError("Phase 3 experiment IDs must be unique.")
    if len({experiment.pinecone_namespace for experiment in experiments}) != len(experiments):
        raise ValueError("Phase 3 Pinecone namespaces must be unique.")
    if len({experiment.chunk_overlap for experiment in experiments}) != 1:
        raise ValueError(
            "The Phase 3 configs must keep chunk overlap fixed so chunk size is the only content variable."
        )
    return tuple(sorted(experiments, key=lambda experiment: experiment.chunk_size))


def namespace_vector_count(pinecone_index: Any, namespace: str) -> int:
    stats = pinecone_index.describe_index_stats()
    namespace_stats = stats.namespaces.get(namespace)
    return int(namespace_stats.vector_count) if namespace_stats else 0


def wait_for_vector_count(
    pinecone_index: Any,
    namespace: str,
    expected_count: int,
    timeout_seconds: int = 120,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    while True:
        vector_count = namespace_vector_count(pinecone_index, namespace)
        if vector_count == expected_count:
            return vector_count
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Pinecone namespace '{namespace}' did not reach {expected_count} vectors "
                f"within {timeout_seconds} seconds (last count: {vector_count})."
            )
        time.sleep(2)


def rebuild_experiment_namespace(
    resources: DenseRAGResources,
    experiment: ChunkExperiment,
    chunks: list[Document],
) -> dict[str, Any]:
    if not experiment.pinecone_namespace.startswith("phase3-"):
        raise ValueError("Refusing to rebuild a namespace outside the 'phase3-' prefix.")

    started = perf_counter()
    existing_count = namespace_vector_count(resources.pinecone_index, experiment.pinecone_namespace)
    if existing_count:
        resources.pinecone_index.delete(delete_all=True, namespace=experiment.pinecone_namespace)
        wait_for_vector_count(resources.pinecone_index, experiment.pinecone_namespace, 0)

    vector_store = PineconeVectorStore(
        index=resources.pinecone_index,
        embedding=resources.embeddings,
        namespace=experiment.pinecone_namespace,
    )
    point_ids = [
        str(uuid5(NAMESPACE_URL, f"{experiment.pinecone_namespace}:{chunk.metadata['chunk_id']}"))
        for chunk in chunks
    ]
    vector_store.add_documents(documents=chunks, ids=point_ids)
    indexed_count = wait_for_vector_count(
        resources.pinecone_index,
        experiment.pinecone_namespace,
        len(chunks),
    )
    return {
        "namespace_rebuilt": True,
        "previous_vector_count": existing_count,
        "indexed_vector_count": indexed_count,
        "indexing_latency_seconds": round(perf_counter() - started, 6),
    }


def write_comparison(path: Path, comparison: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path

