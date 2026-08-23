from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.agentic_experiments import load_agentic_experiment  # noqa: E402
from i_got_this_rag.agentic_rag import LangGraphRAG  # noqa: E402
from i_got_this_rag.baseline import DenseRAGResources  # noqa: E402
from i_got_this_rag.evaluation import (  # noqa: E402
    BaselineEvaluator,
    load_evaluation_dataset,
    utc_now,
    write_experiment,
)
from i_got_this_rag.final_evaluation import (  # noqa: E402
    FINAL_EVALUATION_VERSION,
    HybridCandidateRerankingRAG,
    build_final_comparison,
    evaluate_langgraph,
    evaluate_version_artifact,
    load_final_evaluation_config,
    rebuild_phase10_namespace,
    write_final_artifacts,
)
from i_got_this_rag.ingestion import (  # noqa: E402
    chunk_documents,
    chunk_fingerprint,
    corpus_fingerprint,
    load_corpus,
)
from i_got_this_rag.metadata_retrieval import enrich_metadata_facets  # noqa: E402
from i_got_this_rag.reranking import BM25CandidateReranker  # noqa: E402
from i_got_this_rag.retrieval import (  # noqa: E402
    BM25SparseRetriever,
    DenseRetriever,
    ReciprocalRankFusionRetriever,
)
from i_got_this_rag.settings import Settings  # noqa: E402


SELECTED_CHUNK_SIZE = 500
SELECTED_CHUNK_OVERLAP = 75
SELECTED_TOP_K = 5
SELECTED_EMBEDDING_MODEL = "embeddinggemma"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 10 final cross-version evaluation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "final_evaluation.yaml",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "questions.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "phase10_final",
    )
    return parser.parse_args()


def load_results_artifact(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Required measured artifact is missing: {path}. "
            "Run its phase experiment before Phase 10."
        )
    raw_bytes = path.read_bytes()
    return json.loads(raw_bytes), {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "kind": "historical_phase_artifact",
    }


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    config = load_final_evaluation_config(args.config, PROJECT_ROOT)
    dataset = load_evaluation_dataset(args.questions)
    settings = Settings.from_environment(PROJECT_ROOT)
    if (settings.chunk_size, settings.chunk_overlap) != (
        SELECTED_CHUNK_SIZE,
        SELECTED_CHUNK_OVERLAP,
    ):
        raise ValueError("Phase 10 runtime versions require selected 500/75 chunks.")
    if settings.top_k != SELECTED_TOP_K:
        raise ValueError("Phase 10 final output must remain Top-5.")
    if settings.embedding_model.removesuffix(":latest") != SELECTED_EMBEDDING_MODEL:
        raise ValueError("Phase 10 runtime versions require embeddinggemma.")

    corpus = corpus_fingerprint(settings.data_dir, PROJECT_ROOT)
    if corpus["document_count"] != 20:
        raise ValueError(
            f"Phase 10 requires the controlled 20-document corpus; "
            f"found {corpus['document_count']}."
        )
    documents = load_corpus(settings.data_dir, PROJECT_ROOT)
    chunks = chunk_documents(
        documents,
        SELECTED_CHUNK_SIZE,
        SELECTED_CHUNK_OVERLAP,
    )
    chunk_set = chunk_fingerprint(chunks)
    enriched_chunks = enrich_metadata_facets(chunks, settings.reference_date)
    chunks_by_id = {
        str(chunk.metadata["chunk_id"]): chunk for chunk in enriched_chunks
    }

    runtime_settings = replace(
        settings,
        pinecone_namespace=config.runtime_namespace,
    )
    runtime_settings.validate()
    print("Connecting Phase 10 runtime versions to Ollama and Pinecone...")
    resources = DenseRAGResources.connect(runtime_settings)
    vector_store, indexing = rebuild_phase10_namespace(
        resources,
        config.runtime_namespace,
        enriched_chunks,
    )

    output_root = args.output_root.resolve()
    runtime_root = output_root / "runtime"
    runtime_results: dict[str, tuple[dict[str, Any], Path]] = {}

    hybrid_spec = next(
        version for version in config.versions if version.version_id == "hybrid_reranker"
    )
    print(
        f"[{hybrid_spec.runtime_experiment_id}] Running hybrid Top-"
        f"{config.hybrid_candidate_k} + BM25 reranker over 15 questions..."
    )
    dense = DenseRetriever(vector_store)
    sparse = BM25SparseRetriever(
        enriched_chunks,
        k1=config.bm25_k1,
        b=config.bm25_b,
    )
    hybrid = ReciprocalRankFusionRetriever(dense, sparse, rrf_k=config.rrf_k)
    hybrid_pipeline = HybridCandidateRerankingRAG(
        runtime_settings,
        hybrid,
        BM25CandidateReranker(k1=config.bm25_k1, b=config.bm25_b),
        resources.llm,
        candidate_k=config.hybrid_candidate_k,
    )
    hybrid_results = BaselineEvaluator(hybrid_pipeline).run(
        dataset,
        str(hybrid_spec.runtime_experiment_id),
    )
    hybrid_config = {
        "schema_version": "1.0",
        "phase": 10,
        "experiment_id": hybrid_spec.runtime_experiment_id,
        "experiment_name": hybrid_spec.label,
        "retrieval_strategy": "hybrid_rrf",
        "candidate_k": config.hybrid_candidate_k,
        "reranker": "bm25",
        "final_top_k": config.final_top_k,
        "rrf_k": config.rrf_k,
        "bm25_k1": config.bm25_k1,
        "bm25_b": config.bm25_b,
        "embedding_model": runtime_settings.embedding_model,
        "chat_model": runtime_settings.chat_model,
        "chunk_size": runtime_settings.chunk_size,
        "chunk_overlap": runtime_settings.chunk_overlap,
        "pinecone_index": runtime_settings.pinecone_index_name,
        "pinecone_namespace": runtime_settings.pinecone_namespace,
        "corpus": corpus,
        "chunk_set": chunk_set,
        "indexing": indexing,
        "evaluation_dataset_sha256": dataset.sha256,
    }
    _, hybrid_results_path = write_experiment(
        runtime_root / str(hybrid_spec.runtime_experiment_id),
        hybrid_config,
        hybrid_results,
    )
    runtime_results[hybrid_spec.version_id] = (hybrid_results, hybrid_results_path)

    langgraph_spec = next(
        version
        for version in config.versions
        if version.version_id == "langgraph_workflow"
    )
    agentic_experiment = load_agentic_experiment(
        PROJECT_ROOT / "config" / "agentic_rag.yaml"
    )
    print(
        f"[{langgraph_spec.runtime_experiment_id}] Running LangGraph over all "
        f"{len(dataset.questions)} questions..."
    )
    graph = LangGraphRAG(
        runtime_settings,
        vector_store,
        resources.llm,
        config=agentic_experiment.graph_config,
        reranker=None,
    )
    langgraph_results = evaluate_langgraph(
        graph,
        dataset,
        str(langgraph_spec.runtime_experiment_id),
    )
    langgraph_config = {
        "schema_version": "1.0",
        "phase": 10,
        "experiment_id": langgraph_spec.runtime_experiment_id,
        "experiment_name": langgraph_spec.label,
        "graph_config": asdict(agentic_experiment.graph_config),
        "graph_source_config": {
            "path": agentic_experiment.config_path.as_posix(),
            "sha256": agentic_experiment.config_sha256,
        },
        "embedding_model": runtime_settings.embedding_model,
        "chat_model": runtime_settings.chat_model,
        "chunk_size": runtime_settings.chunk_size,
        "chunk_overlap": runtime_settings.chunk_overlap,
        "pinecone_index": runtime_settings.pinecone_index_name,
        "pinecone_namespace": runtime_settings.pinecone_namespace,
        "corpus": corpus,
        "chunk_set": chunk_set,
        "indexing": indexing,
        "evaluation_dataset_sha256": dataset.sha256,
    }
    _, langgraph_results_path = write_experiment(
        runtime_root / str(langgraph_spec.runtime_experiment_id),
        langgraph_config,
        langgraph_results,
    )
    runtime_results[langgraph_spec.version_id] = (
        langgraph_results,
        langgraph_results_path,
    )

    evaluated_versions: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, Any]] = []
    for version in config.versions:
        if version.source_results_path is not None:
            results_payload, source_record = load_results_artifact(
                version.source_results_path
            )
        else:
            results_payload, runtime_path = runtime_results[version.version_id]
            raw_bytes = runtime_path.read_bytes()
            source_record = {
                "path": runtime_path.as_posix(),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "kind": "phase10_runtime_artifact",
            }
        source_record["version_id"] = version.version_id
        source_artifacts.append(source_record)
        evaluated_versions.append(
            evaluate_version_artifact(
                version,
                results_payload,
                dataset,
                chunks_by_id,
            )
        )

    comparison = build_final_comparison(
        config,
        evaluated_versions,
        dataset,
        source_artifacts,
    )
    config_payload = {
        "schema_version": "1.0",
        "phase": 10,
        "evaluation_version": FINAL_EVALUATION_VERSION,
        "experiment_id": config.experiment_id,
        "experiment_name": config.experiment_name,
        "created_at": utc_now(),
        "source_config": {
            "path": config.config_path.as_posix(),
            "sha256": config.config_sha256,
        },
        "active_runtime_settings": runtime_settings.public_config(),
        "corpus": corpus,
        "chunk_set": chunk_set,
        "evaluation_dataset": {
            "path": dataset.path.as_posix(),
            "sha256": dataset.sha256,
            "question_count": len(dataset.questions),
        },
        "faithfulness_method": config.faithfulness_method,
        "versions": [
            {
                "version_id": version.version_id,
                "label": version.label,
                "mechanism": version.mechanism,
                "source_results_path": (
                    version.source_results_path.as_posix()
                    if version.source_results_path is not None
                    else None
                ),
                "runtime_experiment_id": version.runtime_experiment_id,
            }
            for version in config.versions
        ],
        "source_artifacts": source_artifacts,
    }
    config_path, comparison_path, analysis_path = write_final_artifacts(
        output_root,
        config_payload,
        comparison,
    )

    print("\nPhase 10 final comparison")
    for version in comparison["versions"]:
        metrics = version["metrics"]
        print(
            f"- {version['label']}: Recall@5={metrics['recall_at_5']:.3f}, "
            f"faithfulness={metrics['faithfulness']:.3f}, "
            f"avg latency={metrics['average_latency_seconds']:.3f}s"
        )
    print(f"Configuration: {config_path}")
    print(f"Comparison: {comparison_path}")
    print(f"Analysis: {analysis_path}")


if __name__ == "__main__":
    main()
