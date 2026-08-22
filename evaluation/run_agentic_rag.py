from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.agentic_experiments import (  # noqa: E402
    load_agentic_experiment,
    rebuild_phase9_agentic_namespace,
    write_agentic_run,
)
from i_got_this_rag.agentic_rag import LangGraphRAG  # noqa: E402
from i_got_this_rag.baseline import DenseRAGResources  # noqa: E402
from i_got_this_rag.evaluation import serialize_retrieval, utc_now  # noqa: E402
from i_got_this_rag.ingestion import (  # noqa: E402
    chunk_documents,
    chunk_fingerprint,
    corpus_fingerprint,
    load_corpus,
)
from i_got_this_rag.metadata_retrieval import enrich_metadata_facets  # noqa: E402
from i_got_this_rag.settings import Settings  # noqa: E402


SELECTED_CHUNK_SIZE = 500
SELECTED_CHUNK_OVERLAP = 75
SELECTED_TOP_K = 5
SELECTED_EMBEDDING_MODEL = "embeddinggemma"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one question through the Phase 9 LangGraph RAG workflow."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "agentic_rag.yaml",
    )
    parser.add_argument(
        "--question",
        default="What do I need to take care of before Saturday?",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "phase9_agentic" / "run.json",
    )
    return parser.parse_args()


def serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in state.items()
        if key not in {"retrieved_docs", "reranked_docs"}
    } | {
        "retrieved_chunks": serialize_retrieval(state.get("retrieved_docs", [])),
        "final_chunks": serialize_retrieval(state.get("reranked_docs", [])),
    }


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    base_settings = Settings.from_environment(PROJECT_ROOT)
    if (base_settings.chunk_size, base_settings.chunk_overlap) != (
        SELECTED_CHUNK_SIZE,
        SELECTED_CHUNK_OVERLAP,
    ):
        raise ValueError(
            "Phase 9 must use the selected 500-token chunks with 75-token overlap."
        )
    if base_settings.top_k != SELECTED_TOP_K:
        raise ValueError("Phase 9 must keep final dense retrieval at Top-5.")
    if base_settings.embedding_model.removesuffix(":latest") != SELECTED_EMBEDDING_MODEL:
        raise ValueError("Phase 9 must keep embeddinggemma fixed.")

    experiment = load_agentic_experiment(args.config)
    if experiment.graph_config.final_top_k != base_settings.top_k:
        raise ValueError("Phase 9 graph final_top_k must match RAG_TOP_K.")
    settings = replace(
        base_settings,
        pinecone_namespace=experiment.pinecone_namespace,
    )
    settings.validate()

    corpus = corpus_fingerprint(settings.data_dir, PROJECT_ROOT)
    if corpus["document_count"] != 20:
        raise ValueError(
            f"Phase 9 requires the controlled 20-document corpus; found {corpus['document_count']}."
        )
    documents = load_corpus(settings.data_dir, PROJECT_ROOT)
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    chunk_set = chunk_fingerprint(chunks)
    enriched_chunks = enrich_metadata_facets(chunks, settings.reference_date)

    print("Connecting Ollama and rebuilding the guarded Phase 9 Pinecone namespace...")
    resources = DenseRAGResources.connect(settings)
    vector_store, indexing = rebuild_phase9_agentic_namespace(
        resources,
        settings.pinecone_namespace,
        enriched_chunks,
    )
    graph = LangGraphRAG(
        settings,
        vector_store,
        resources.llm,
        config=experiment.graph_config,
        reranker=None,
    )

    print(f"Running Phase 9 graph for: {args.question}")
    state = graph.invoke(args.question)
    payload = {
        "schema_version": "1.0",
        "phase": 9,
        "experiment_id": experiment.experiment_id,
        "experiment_name": experiment.experiment_name,
        "completed_at": utc_now(),
        "question": args.question,
        "graph_config": asdict(experiment.graph_config),
        "source_config": {
            "path": experiment.config_path.as_posix(),
            "sha256": experiment.config_sha256,
        },
        "controlled_configuration": {
            "embedding_model": settings.embedding_model,
            "chat_model": settings.chat_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
            "pinecone_index": settings.pinecone_index_name,
            "pinecone_namespace": settings.pinecone_namespace,
            "corpus": corpus,
            "chunk_set": chunk_set,
        },
        "indexing": indexing,
        "state": serialize_state(state),
    }
    output_path = write_agentic_run(args.output, payload)
    print(f"Answer: {state['answer']}")
    print(f"Attempts: {state['retrieval_attempts']}; grounded: {state['grounded']}")
    print(f"Trace: {output_path}")


if __name__ == "__main__":
    main()
