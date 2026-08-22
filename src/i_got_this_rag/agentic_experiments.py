from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import yaml
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from .agentic_rag import AGENTIC_GRAPH_VERSION, AgenticGraphConfig
from .baseline import DenseRAGResources
from .chunk_experiments import namespace_vector_count, wait_for_vector_count


@dataclass(frozen=True)
class AgenticExperiment:
    experiment_id: str
    experiment_name: str
    pinecone_namespace: str
    graph_config: AgenticGraphConfig
    retry_query_strategy: str
    reranker: str | None
    config_path: Path
    config_sha256: str


def load_agentic_experiment(path: Path) -> AgenticExperiment:
    path = path.resolve()
    raw_bytes = path.read_bytes()
    payload = yaml.safe_load(raw_bytes) or {}
    if not isinstance(payload, dict):
        raise ValueError("Phase 9 configuration must be a YAML mapping.")
    required = {
        "experiment_id",
        "experiment_name",
        "pinecone_namespace",
        "graph_version",
        "metadata_filter_enabled",
        "metadata_fallback_enabled",
        "retry_query_rewriting_enabled",
        "retry_query_strategy",
        "reranker_enabled",
        "reranker",
        "initial_candidate_k",
        "retry_candidate_k",
        "final_top_k",
        "max_retrieval_attempts",
        "minimum_evidence_term_coverage",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"{path} is missing: {', '.join(sorted(missing))}.")
    if str(payload["graph_version"]) != AGENTIC_GRAPH_VERSION:
        raise ValueError(f"Phase 9 graph_version must be {AGENTIC_GRAPH_VERSION}.")
    namespace = str(payload["pinecone_namespace"])
    if not namespace.startswith("phase9-"):
        raise ValueError("Phase 9 namespaces must start with 'phase9-'.")

    graph_config = AgenticGraphConfig(
        metadata_filter_enabled=bool(payload["metadata_filter_enabled"]),
        metadata_fallback_enabled=bool(payload["metadata_fallback_enabled"]),
        retry_query_rewriting_enabled=bool(payload["retry_query_rewriting_enabled"]),
        reranker_enabled=bool(payload["reranker_enabled"]),
        initial_candidate_k=int(payload["initial_candidate_k"]),
        retry_candidate_k=int(payload["retry_candidate_k"]),
        final_top_k=int(payload["final_top_k"]),
        max_retrieval_attempts=int(payload["max_retrieval_attempts"]),
        minimum_evidence_term_coverage=float(
            payload["minimum_evidence_term_coverage"]
        ),
    )
    graph_config.validate()
    retry_strategy = str(payload["retry_query_strategy"])
    if retry_strategy != "rewrite":
        raise ValueError("Phase 9 retry_query_strategy must be 'rewrite'.")
    reranker_value = payload["reranker"]
    reranker = str(reranker_value) if reranker_value is not None else None
    if graph_config.reranker_enabled or reranker is not None:
        raise ValueError(
            "Phase 9 keeps the Phase 6 selected setting: reranking must remain disabled."
        )
    return AgenticExperiment(
        experiment_id=str(payload["experiment_id"]),
        experiment_name=str(payload["experiment_name"]),
        pinecone_namespace=namespace,
        graph_config=graph_config,
        retry_query_strategy=retry_strategy,
        reranker=reranker,
        config_path=path,
        config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def rebuild_phase9_agentic_namespace(
    resources: DenseRAGResources,
    namespace: str,
    chunks: list[Document],
) -> tuple[PineconeVectorStore, dict[str, Any]]:
    if not namespace.startswith("phase9-"):
        raise ValueError("Refusing to rebuild a namespace outside the 'phase9-' prefix.")

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


def write_agentic_run(path: Path, payload: dict[str, Any]) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
