from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.baseline import BaselineRAG, DenseRAGResources  # noqa: E402
from i_got_this_rag.chunk_experiments import (  # noqa: E402
    ChunkExperiment,
    load_chunk_experiments,
    rebuild_experiment_namespace,
    write_comparison,
)
from i_got_this_rag.evaluation import (  # noqa: E402
    BaselineEvaluator,
    EvaluationDataset,
    load_evaluation_dataset,
    utc_now,
    write_experiment,
)
from i_got_this_rag.ingestion import (  # noqa: E402
    chunk_documents,
    corpus_fingerprint,
    load_corpus,
)
from i_got_this_rag.settings import Settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the four controlled Phase 3 chunk-size experiments."
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=PROJECT_ROOT / "config" / "experiments",
        help="Directory containing the four chunk_*.yaml experiment files.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "questions.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "phase3_chunking",
    )
    return parser.parse_args()


def experiment_config(
    experiment: ChunkExperiment,
    settings: Settings,
    dataset: EvaluationDataset,
    corpus: dict[str, Any],
    indexing: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": experiment.experiment_id,
        "experiment_name": experiment.experiment_name,
        "created_at": utc_now(),
        "pipeline_phase": 1,
        "evaluation_phase": 3,
        "vector_store": "pinecone",
        "pinecone_index": settings.pinecone_index_name,
        "pinecone_namespace": settings.pinecone_namespace,
        "pinecone_cloud": settings.pinecone_cloud,
        "pinecone_region": settings.pinecone_region,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "retrieval_strategy": "dense",
        "candidate_k": settings.top_k,
        "top_k": settings.top_k,
        "reranker_enabled": False,
        "llm_model": settings.chat_model,
        "ollama_base_url": settings.ollama_base_url,
        "reference_date": settings.reference_date,
        "timezone": settings.timezone,
        "data_dir": settings.data_dir.as_posix(),
        "corpus": corpus,
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


def build_question_comparison(experiment_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_experiment = {
        result["experiment_id"]: {
            question["question_id"]: question for question in result["questions"]
        }
        for result in experiment_results
    }
    first_result = experiment_results[0]
    comparisons: list[dict[str, Any]] = []
    for question in first_result["questions"]:
        comparisons.append(
            {
                "question_id": question["question_id"],
                "category": question["category"],
                "expected_source_ids": question["expected_source_ids"],
                "experiments": {
                    experiment_id: {
                        "recall_at_5": questions[question["question_id"]]["recall_at_5"],
                        "expected_source_rank": questions[question["question_id"]][
                            "expected_source_rank"
                        ],
                        "expected_source_ranks": questions[question["question_id"]][
                            "expected_source_ranks"
                        ],
                        "retrieval_latency_seconds": questions[question["question_id"]][
                            "retrieval_latency_seconds"
                        ],
                    }
                    for experiment_id, questions in by_experiment.items()
                },
            }
        )
    return comparisons


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    base_settings = Settings.from_environment(PROJECT_ROOT)
    dataset = load_evaluation_dataset(args.questions)
    experiments = load_chunk_experiments(args.configs, base_settings.pinecone_namespace)
    corpus = corpus_fingerprint(base_settings.data_dir, PROJECT_ROOT)
    if corpus["document_count"] != 20:
        raise ValueError(
            f"Phase 3 requires the controlled 20-document corpus; found {corpus['document_count']}."
        )
    documents = load_corpus(base_settings.data_dir, PROJECT_ROOT)

    print("Connecting to the fixed Ollama and Pinecone resources...")
    resources = DenseRAGResources.connect(base_settings)
    output_root = args.output_root.resolve()
    completed_results: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for experiment in experiments:
        settings = replace(
            base_settings,
            chunk_size=experiment.chunk_size,
            chunk_overlap=experiment.chunk_overlap,
            pinecone_namespace=experiment.pinecone_namespace,
        )
        settings.validate()
        print(
            f"[{experiment.experiment_id}] Chunking at {settings.chunk_size} tokens "
            f"with {settings.chunk_overlap} overlap..."
        )
        chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
        indexing = rebuild_experiment_namespace(resources, experiment, chunks)
        indexing["source_document_count"] = corpus["document_count"]
        indexing["chunk_count"] = len(chunks)

        print(f"[{experiment.experiment_id}] Evaluating all {len(dataset.questions)} questions...")
        pipeline = BaselineRAG(settings, resources=resources)
        results = BaselineEvaluator(pipeline).run(dataset, experiment.experiment_id)
        config = experiment_config(experiment, settings, dataset, corpus, indexing)
        experiment_directory = output_root / experiment.experiment_id
        config_path, results_path = write_experiment(experiment_directory, config, results)
        completed_results.append(results)
        comparison_rows.append(
            {
                "experiment_id": experiment.experiment_id,
                "chunk_size": experiment.chunk_size,
                "chunk_overlap": experiment.chunk_overlap,
                "pinecone_namespace": experiment.pinecone_namespace,
                "chunk_count": len(chunks),
                "indexing_latency_seconds": indexing["indexing_latency_seconds"],
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
        "phase": 3,
        "experiment_suite": "controlled_chunk_size_comparison",
        "completed_at": utc_now(),
        "varied_parameters": ["chunk_size", "pinecone_namespace"],
        "overlap_policy": "Fixed at 75 tokens so chunk size is the only content variable.",
        "controlled_variables": {
            "corpus_sha256": corpus["sha256"],
            "document_count": corpus["document_count"],
            "embedding_model": base_settings.embedding_model,
            "pinecone_index": base_settings.pinecone_index_name,
            "retrieval_strategy": "dense",
            "top_k": base_settings.top_k,
            "llm_model": base_settings.chat_model,
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

