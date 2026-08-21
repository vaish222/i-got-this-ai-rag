from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_ollama import ChatOllama


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.baseline import (  # noqa: E402
    BaselineRAG,
    installed_ollama_models,
)
from i_got_this_rag.chunk_experiments import write_comparison  # noqa: E402
from i_got_this_rag.embedding_experiments import (  # noqa: E402
    EmbeddingExperiment,
    connect_embedding_resources,
    load_embedding_experiments,
    rebuild_embedding_namespace,
)
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
from i_got_this_rag.settings import Settings  # noqa: E402


SELECTED_CHUNK_SIZE = 500
SELECTED_CHUNK_OVERLAP = 75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the three controlled Phase 4 local embedding-model experiments."
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=PROJECT_ROOT / "config" / "embedding_experiments",
        help="Directory containing the three embedding_*.yaml experiment files.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "questions.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "phase4_embeddings",
    )
    return parser.parse_args()


def experiment_config(
    experiment: EmbeddingExperiment,
    settings: Settings,
    dataset: EvaluationDataset,
    corpus: dict[str, Any],
    chunk_set: dict[str, Any],
    index_info: dict[str, Any],
    indexing: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": experiment.experiment_id,
        "experiment_name": experiment.experiment_name,
        "created_at": utc_now(),
        "pipeline_phase": 1,
        "evaluation_phase": 4,
        "vector_store": "pinecone",
        "pinecone_index": settings.pinecone_index_name,
        "pinecone_namespace": settings.pinecone_namespace,
        "pinecone_cloud": settings.pinecone_cloud,
        "pinecone_region": settings.pinecone_region,
        "pinecone_metric": index_info["pinecone_metric"],
        "embedding_model": settings.embedding_model,
        "embedding_dimension": index_info["embedding_dimension"],
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
        "indexing": {**index_info, **indexing},
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
            "Phase 4 must use the selected 500-token chunks with 75-token overlap. "
            "Set RAG_CHUNK_SIZE=500 and RAG_CHUNK_OVERLAP=75."
        )

    dataset = load_evaluation_dataset(args.questions)
    experiments = load_embedding_experiments(args.configs)
    required_models = {
        *(experiment.embedding_model.removesuffix(":latest") for experiment in experiments),
        base_settings.chat_model.removesuffix(":latest"),
    }
    available_models = installed_ollama_models(base_settings.ollama_base_url)
    missing_models = required_models - available_models
    if missing_models:
        commands = "\n".join(f"ollama pull {model}" for model in sorted(missing_models))
        raise RuntimeError(f"Download the missing local model(s), then rerun Phase 4:\n{commands}")

    corpus = corpus_fingerprint(base_settings.data_dir, PROJECT_ROOT)
    if corpus["document_count"] != 20:
        raise ValueError(
            f"Phase 4 requires the controlled 20-document corpus; found {corpus['document_count']}."
        )
    documents = load_corpus(base_settings.data_dir, PROJECT_ROOT)
    chunks = chunk_documents(documents, SELECTED_CHUNK_SIZE, SELECTED_CHUNK_OVERLAP)
    chunk_set = chunk_fingerprint(chunks)
    shared_llm = ChatOllama(
        model=base_settings.chat_model,
        base_url=base_settings.ollama_base_url,
        temperature=0,
    )

    output_root = args.output_root.resolve()
    completed_results: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for experiment in experiments:
        settings = replace(
            base_settings,
            embedding_model=experiment.embedding_model,
            pinecone_index_name=experiment.pinecone_index_name,
            pinecone_namespace=experiment.pinecone_namespace,
        )
        settings.validate()
        print(
            f"[{experiment.experiment_id}] Connecting {experiment.embedding_model} "
            f"to {experiment.pinecone_index_name}..."
        )
        resources, index_info = connect_embedding_resources(settings, experiment, shared_llm)
        indexing = rebuild_embedding_namespace(resources, experiment, chunks)
        indexing["source_document_count"] = corpus["document_count"]
        indexing["chunk_count"] = len(chunks)

        print(f"[{experiment.experiment_id}] Evaluating all {len(dataset.questions)} questions...")
        pipeline = BaselineRAG(settings, resources=resources)
        results = BaselineEvaluator(pipeline).run(dataset, experiment.experiment_id)
        config = experiment_config(
            experiment,
            settings,
            dataset,
            corpus,
            chunk_set,
            index_info,
            indexing,
        )
        experiment_directory = output_root / experiment.experiment_id
        config_path, results_path = write_experiment(experiment_directory, config, results)
        completed_results.append(results)
        comparison_rows.append(
            {
                "experiment_id": experiment.experiment_id,
                "embedding_model": experiment.embedding_model,
                "embedding_dimension": index_info["embedding_dimension"],
                "pinecone_index": experiment.pinecone_index_name,
                "pinecone_namespace": experiment.pinecone_namespace,
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
        "phase": 4,
        "experiment_suite": "controlled_embedding_model_comparison",
        "completed_at": utc_now(),
        "varied_parameters": ["embedding_model", "pinecone_index", "embedding_dimension"],
        "controlled_variables": {
            "corpus_sha256": corpus["sha256"],
            "document_count": corpus["document_count"],
            "chunk_set_sha256": chunk_set["sha256"],
            "chunk_count": chunk_set["chunk_count"],
            "chunk_size": SELECTED_CHUNK_SIZE,
            "chunk_overlap": SELECTED_CHUNK_OVERLAP,
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

