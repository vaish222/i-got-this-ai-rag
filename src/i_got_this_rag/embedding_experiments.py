from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import yaml
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from .baseline import DenseRAGResources
from .chunk_experiments import namespace_vector_count, wait_for_vector_count
from .settings import Settings


REQUIRED_EMBEDDING_MODELS = ("all-minilm", "nomic-embed-text", "mxbai-embed-large")


@dataclass(frozen=True)
class EmbeddingExperiment:
    experiment_id: str
    experiment_name: str
    embedding_model: str
    pinecone_index_name: str
    pinecone_namespace: str
    config_path: Path
    config_sha256: str


def load_embedding_experiments(config_directory: Path) -> tuple[EmbeddingExperiment, ...]:
    experiments: list[EmbeddingExperiment] = []
    for path in sorted(config_directory.resolve().glob("embedding_*.yaml")):
        raw_bytes = path.read_bytes()
        payload = yaml.safe_load(raw_bytes) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Embedding experiment config must be a mapping: {path}")
        required = {
            "experiment_id",
            "experiment_name",
            "embedding_model",
            "pinecone_index_name",
            "pinecone_namespace",
        }
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"{path} is missing: {', '.join(sorted(missing))}.")
        experiment = EmbeddingExperiment(
            experiment_id=str(payload["experiment_id"]),
            experiment_name=str(payload["experiment_name"]),
            embedding_model=str(payload["embedding_model"]),
            pinecone_index_name=str(payload["pinecone_index_name"]),
            pinecone_namespace=str(payload["pinecone_namespace"]),
            config_path=path,
            config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,44}", experiment.pinecone_index_name):
            raise ValueError(f"{path}: invalid Pinecone index name.")
        if not experiment.pinecone_namespace.startswith("phase4-"):
            raise ValueError(f"{path}: Phase 4 namespaces must start with 'phase4-'.")
        experiments.append(experiment)

    models = tuple(sorted(experiment.embedding_model for experiment in experiments))
    if models != tuple(sorted(REQUIRED_EMBEDDING_MODELS)):
        raise ValueError(
            "Phase 4 requires exactly one experiment for each model: "
            + ", ".join(REQUIRED_EMBEDDING_MODELS)
            + "."
        )
    if len({experiment.experiment_id for experiment in experiments}) != len(experiments):
        raise ValueError("Phase 4 experiment IDs must be unique.")
    if len({experiment.pinecone_index_name for experiment in experiments}) != len(experiments):
        raise ValueError("Phase 4 uses one unique Pinecone index per embedding model.")
    return tuple(
        sorted(
            experiments,
            key=lambda experiment: REQUIRED_EMBEDDING_MODELS.index(experiment.embedding_model),
        )
    )


def validate_index_compatibility(
    index_description: Any,
    embedding_dimension: int,
    index_name: str,
) -> None:
    if int(index_description.dimension) != embedding_dimension:
        raise ValueError(
            f"Pinecone index '{index_name}' has dimension {index_description.dimension}, "
            f"but the configured embedding model produces {embedding_dimension}."
        )
    metric = str(index_description.metric).lower().split(".")[-1]
    if metric != "cosine":
        raise ValueError(
            f"Pinecone index '{index_name}' uses '{metric}', but Phase 4 requires cosine."
        )


def wait_for_index_ready(
    pinecone_client: Pinecone,
    index_name: str,
    timeout_seconds: int = 120,
) -> Any:
    deadline = time.monotonic() + timeout_seconds
    while True:
        description = pinecone_client.describe_index(index_name)
        status = description.status
        ready = status.get("ready", False) if isinstance(status, dict) else bool(status.ready)
        if ready:
            return description
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Pinecone index '{index_name}' was not ready within {timeout_seconds} seconds."
            )
        time.sleep(2)


def connect_embedding_resources(
    settings: Settings,
    experiment: EmbeddingExperiment,
    llm: Any,
) -> tuple[DenseRAGResources, dict[str, Any]]:
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("PINECONE_API_KEY is required to run Phase 4.")

    embeddings = OllamaEmbeddings(
        model=experiment.embedding_model,
        base_url=settings.ollama_base_url,
    )
    embedding_dimension = len(embeddings.embed_query("dimension probe"))
    pinecone_client = Pinecone(api_key=api_key)
    index_created = not pinecone_client.has_index(experiment.pinecone_index_name)
    if index_created:
        pinecone_client.create_index(
            name=experiment.pinecone_index_name,
            dimension=embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
    index_description = wait_for_index_ready(pinecone_client, experiment.pinecone_index_name)
    validate_index_compatibility(
        index_description,
        embedding_dimension,
        experiment.pinecone_index_name,
    )
    pinecone_index = pinecone_client.Index(experiment.pinecone_index_name)
    resources = DenseRAGResources(
        embeddings=embeddings,
        pinecone_client=pinecone_client,
        pinecone_index=pinecone_index,
        llm=llm,
    )
    return resources, {
        "embedding_dimension": embedding_dimension,
        "pinecone_index_created": index_created,
        "pinecone_metric": "cosine",
    }


def rebuild_embedding_namespace(
    resources: DenseRAGResources,
    experiment: EmbeddingExperiment,
    chunks: list[Document],
) -> dict[str, Any]:
    if not experiment.pinecone_namespace.startswith("phase4-"):
        raise ValueError("Refusing to rebuild a namespace outside the 'phase4-' prefix.")

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
        str(
            uuid5(
                NAMESPACE_URL,
                f"{experiment.embedding_model}:{experiment.pinecone_namespace}:"
                f"{chunk.metadata['chunk_id']}",
            )
        )
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

