from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.baseline import DenseRAGResources  # noqa: E402
from i_got_this_rag.chunk_experiments import write_comparison  # noqa: E402
from i_got_this_rag.evaluation import (  # noqa: E402
    BaselineEvaluator,
    EvaluationDataset,
    build_question_comparison,
    load_evaluation_dataset,
    utc_now,
    write_experiment,
)
from i_got_this_rag.ingestion import (  # noqa: E402
    chunk_documents,
    chunk_fingerprint,
    corpus_fingerprint,
    load_corpus,
)
from i_got_this_rag.retrieval import (  # noqa: E402
    BM25SparseRetriever,
    ConfigurableRetrievalRAG,
    DenseRetriever,
    ReciprocalRankFusionRetriever,
    Retriever,
)
from i_got_this_rag.retrieval_experiments import (  # noqa: E402
    RetrievalExperiment,
    load_retrieval_experiments,
    rebuild_phase5_dense_namespace,
)
from i_got_this_rag.settings import Settings  # noqa: E402


SELECTED_CHUNK_SIZE = 500
SELECTED_CHUNK_OVERLAP = 75
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the controlled Phase 5 dense, sparse, and hybrid retrieval experiments."
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=PROJECT_ROOT / "config" / "retrieval_experiments",
        help="Directory containing the three retrieval_*.yaml experiment files.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "questions.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "phase5_retrieval",
    )
    return parser.parse_args()


def strategy_details(strategy: str, top_k: int) -> dict[str, Any]:
    if strategy == "dense":
        return {
            "score_type": "cosine_similarity",
            "dense_candidate_k": top_k,
        }
    if strategy == "sparse":
        return {
            "score_type": "bm25",
            "sparse_candidate_k": top_k,
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
        }
    if strategy == "hybrid":
        return {
            "score_type": "reciprocal_rank_fusion",
            "dense_candidate_k": top_k,
            "sparse_candidate_k": top_k,
            "final_top_k": top_k,
            "rrf_k": RRF_K,
        }
    raise ValueError(f"Unknown retrieval strategy: {strategy}")


def experiment_config(
    experiment: RetrievalExperiment,
    settings: Settings,
    dataset: EvaluationDataset,
    corpus: dict[str, Any],
    chunk_set: dict[str, Any],
    indexing: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": experiment.experiment_id,
        "experiment_name": experiment.experiment_name,
        "created_at": utc_now(),
        "pipeline_phase": 1,
        "evaluation_phase": 5,
        "vector_store": "pinecone",
        "pinecone_index": settings.pinecone_index_name,
        "pinecone_namespace": settings.pinecone_namespace,
        "pinecone_cloud": settings.pinecone_cloud,
        "pinecone_region": settings.pinecone_region,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "retrieval_strategy": experiment.retrieval_strategy,
        "retrieval_details": strategy_details(experiment.retrieval_strategy, settings.top_k),
        "candidate_k": settings.top_k,
        "top_k": settings.top_k,
        "reranker_enabled": False,
        "llm_model": settings.chat_model,
        "ollama_base_url": settings.ollama_base_url,
        "reference_date": settings.reference_date,
        "timezone": settings.timezone,
        "data_dir": settings.data_dir.as_posix(),
        "corpus": corpus,
        "chunk_set": chunk_set,
        "evaluation_dataset": {
            "name": dataset.dataset_name,
            "schema_version": dataset.schema_version,
            "path": dataset.path.as_posix(),
            "sha256": dataset.sha256,
            "question_count": len(dataset.questions),
        },
        "source_experiment_config": {
            "path": experiment.config_path.as_posix(),
            "sha256": experiment.config_sha256,
        },
        "indexing": indexing,
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
            "Phase 5 must use the selected 500-token chunks with 75-token overlap. "
            "Set RAG_CHUNK_SIZE=500 and RAG_CHUNK_OVERLAP=75."
        )

    dataset = load_evaluation_dataset(args.questions)
    experiments = load_retrieval_experiments(args.configs)
    dense_namespace = experiments[0].pinecone_namespace
    settings = replace(base_settings, pinecone_namespace=dense_namespace)
    settings.validate()

    corpus = corpus_fingerprint(settings.data_dir, PROJECT_ROOT)
    if corpus["document_count"] != 20:
        raise ValueError(
            f"Phase 5 requires the controlled 20-document corpus; found {corpus['document_count']}."
        )
    documents = load_corpus(settings.data_dir, PROJECT_ROOT)
    chunks = chunk_documents(documents, SELECTED_CHUNK_SIZE, SELECTED_CHUNK_OVERLAP)
    chunk_set = chunk_fingerprint(chunks)

    print("Connecting the fixed embedding model, LLM, and Pinecone index...")
    resources = DenseRAGResources.connect(settings)
    dense_vector_store, indexing = rebuild_phase5_dense_namespace(
        resources,
        dense_namespace,
        chunks,
    )
    indexing["source_document_count"] = corpus["document_count"]
    indexing["chunk_count"] = len(chunks)

    dense_retriever = DenseRetriever(dense_vector_store)
    sparse_retriever = BM25SparseRetriever(chunks, k1=BM25_K1, b=BM25_B)
    retrievers: dict[str, Retriever] = {
        "dense": dense_retriever,
        "sparse": sparse_retriever,
        "hybrid": ReciprocalRankFusionRetriever(
            dense_retriever,
            sparse_retriever,
            rrf_k=RRF_K,
        ),
    }

    output_root = args.output_root.resolve()
    completed_results: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for experiment in experiments:
        print(
            f"[{experiment.experiment_id}] Evaluating {experiment.retrieval_strategy} "
            f"retrieval over all {len(dataset.questions)} questions..."
        )
        pipeline = ConfigurableRetrievalRAG(
            settings,
            retrievers[experiment.retrieval_strategy],
            resources.llm,
        )
        results = BaselineEvaluator(pipeline).run(dataset, experiment.experiment_id)
        config = experiment_config(
            experiment,
            settings,
            dataset,
            corpus,
            chunk_set,
            indexing,
        )
        experiment_directory = output_root / experiment.experiment_id
        config_path, results_path = write_experiment(experiment_directory, config, results)
        completed_results.append(results)
        comparison_rows.append(
            {
                "experiment_id": experiment.experiment_id,
                "retrieval_strategy": experiment.retrieval_strategy,
                "retrieval_details": strategy_details(
                    experiment.retrieval_strategy,
                    settings.top_k,
                ),
                **results["summary"],
                "category_summary": results["category_summary"],
                "config_path": config_path.as_posix(),
                "results_path": results_path.as_posix(),
            }
        )
        print(f"[{experiment.experiment_id}] Recall@5: {results['summary']['recall_at_5']:.3f}")

    best_recall = max(row["recall_at_5"] for row in comparison_rows)
    comparison = {
        "schema_version": "1.0",
        "phase": 5,
        "experiment_suite": "controlled_retrieval_strategy_comparison",
        "completed_at": utc_now(),
        "varied_parameters": ["retrieval_strategy"],
        "controlled_variables": {
            "corpus_sha256": corpus["sha256"],
            "document_count": corpus["document_count"],
            "chunk_set_sha256": chunk_set["sha256"],
            "chunk_count": chunk_set["chunk_count"],
            "chunk_size": SELECTED_CHUNK_SIZE,
            "chunk_overlap": SELECTED_CHUNK_OVERLAP,
            "embedding_model": settings.embedding_model,
            "pinecone_index": settings.pinecone_index_name,
            "pinecone_namespace": settings.pinecone_namespace,
            "top_k": settings.top_k,
            "llm_model": settings.chat_model,
            "evaluation_dataset_sha256": dataset.sha256,
            "evaluation_question_count": len(dataset.questions),
        },
        "best_recall_at_5": best_recall,
        "best_recall_experiment_ids": [
            row["experiment_id"] for row in comparison_rows if row["recall_at_5"] == best_recall
        ],
        "experiments": comparison_rows,
        "question_comparison": build_question_comparison(completed_results),
    }
    comparison_path = write_comparison(output_root / "comparison.json", comparison)
    print(f"Comparison: {comparison_path}")


if __name__ == "__main__":
    main()

