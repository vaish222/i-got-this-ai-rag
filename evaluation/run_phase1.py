from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.baseline import BaselineRAG, DenseRAGResources  # noqa: E402
from i_got_this_rag.chunk_experiments import (  # noqa: E402
    namespace_vector_count,
    wait_for_vector_count,
)
from i_got_this_rag.evaluation import (  # noqa: E402
    extract_citations,
    serialize_retrieval,
    utc_now,
)
from i_got_this_rag.ingestion import (  # noqa: E402
    chunk_documents,
    chunk_fingerprint,
    corpus_fingerprint,
    load_corpus,
)
from i_got_this_rag.settings import Settings  # noqa: E402


DEFAULT_QUESTION = "What do we need to bring to the neighborhood potluck?"
VectorStoreFactory = Callable[..., PineconeVectorStore]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build or reuse the Phase 1 dense namespace and run one grounded RAG question."
        )
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "phase1" / "run.json",
    )
    parser.add_argument(
        "--rebuild-namespace",
        action="store_true",
        help="Explicitly delete and rebuild only PINECONE_NAMESPACE before retrieval.",
    )
    return parser.parse_args()


def prepare_phase1_namespace(
    resources: DenseRAGResources,
    namespace: str,
    chunks: list[Document],
    *,
    rebuild_namespace: bool,
    vector_store_factory: VectorStoreFactory = PineconeVectorStore,
) -> tuple[PineconeVectorStore, dict[str, Any]]:
    if not namespace.strip():
        raise ValueError("Phase 1 namespace cannot be empty.")

    started = perf_counter()
    previous_count = namespace_vector_count(resources.pinecone_index, namespace)
    if rebuild_namespace and previous_count:
        resources.pinecone_index.delete(delete_all=True, namespace=namespace)
        wait_for_vector_count(resources.pinecone_index, namespace, 0)

    vector_store = vector_store_factory(
        index=resources.pinecone_index,
        embedding=resources.embeddings,
        namespace=namespace,
    )
    should_index = rebuild_namespace or previous_count == 0
    if should_index:
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
    else:
        indexed_count = previous_count

    action = (
        "rebuilt"
        if rebuild_namespace
        else "indexed_empty_namespace"
        if previous_count == 0
        else "reused_existing_namespace"
    )
    return vector_store, {
        "action": action,
        "namespace_rebuilt": rebuild_namespace,
        "previous_vector_count": previous_count,
        "indexed_vector_count": indexed_count,
        "indexing_latency_seconds": round(perf_counter() - started, 6),
    }


def write_phase1_run(path: Path, payload: dict[str, Any]) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    args = parse_args()
    question = args.question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    load_dotenv(PROJECT_ROOT / ".env", override=True)
    settings = Settings.from_environment(PROJECT_ROOT)
    corpus = corpus_fingerprint(settings.data_dir, PROJECT_ROOT)
    documents = load_corpus(settings.data_dir, PROJECT_ROOT)
    chunks = chunk_documents(
        documents,
        settings.chunk_size,
        settings.chunk_overlap,
    )
    chunk_set = chunk_fingerprint(chunks)

    print("Connecting to Ollama and Pinecone...")
    resources = DenseRAGResources.connect(
        settings,
        create_index=True,
    )
    vector_store, indexing = prepare_phase1_namespace(
        resources,
        settings.pinecone_namespace,
        chunks,
        rebuild_namespace=args.rebuild_namespace,
    )
    pipeline = BaselineRAG(
        settings,
        resources=resources,
        vector_store=vector_store,
    )

    retrieval_started = perf_counter()
    results = pipeline.retrieve(question)
    retrieval_latency = perf_counter() - retrieval_started
    generation_started = perf_counter()
    answer = pipeline.generate(question, results)
    generation_latency = perf_counter() - generation_started
    serialized_results = serialize_retrieval(results)

    payload = {
        "schema_version": "1.0",
        "phase": 1,
        "completed_at": utc_now(),
        "question": question,
        "answer": answer,
        "citations": extract_citations(answer, serialized_results),
        "retrieved_chunks": serialized_results,
        "configuration": settings.public_config(),
        "corpus": corpus,
        "chunk_set": chunk_set,
        "indexing": indexing,
        "latency": {
            "retrieval_seconds": round(retrieval_latency, 6),
            "generation_seconds": round(generation_latency, 6),
            "total_rag_seconds": round(retrieval_latency + generation_latency, 6),
        },
    }
    output_path = write_phase1_run(args.output, payload)
    print(f"Namespace action: {indexing['action']}")
    print(f"Retrieved chunks: {len(results)}")
    print(f"Answer: {answer}")
    print(f"Result: {output_path}")


if __name__ == "__main__":
    main()
