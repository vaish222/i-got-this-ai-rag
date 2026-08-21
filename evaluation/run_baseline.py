from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.baseline import BaselineRAG  # noqa: E402
from i_got_this_rag.evaluation import (  # noqa: E402
    BaselineEvaluator,
    load_evaluation_dataset,
    utc_now,
    write_experiment,
)
from i_got_this_rag.settings import Settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Phase 1 dense Pinecone baseline against all 15 questions."
    )
    parser.add_argument("--experiment-id", default="E001_dense_baseline")
    parser.add_argument("--experiment-name", default="Phase 2 dense baseline")
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "questions.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    settings = Settings.from_environment(PROJECT_ROOT)
    dataset = load_evaluation_dataset(args.questions)

    config = {
        "schema_version": "1.0",
        "experiment_id": args.experiment_id,
        "experiment_name": args.experiment_name,
        "created_at": utc_now(),
        "pipeline_phase": 1,
        "evaluation_phase": 2,
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
        "evaluation_dataset": {
            "name": dataset.dataset_name,
            "schema_version": dataset.schema_version,
            "path": dataset.path.as_posix(),
            "sha256": dataset.sha256,
            "question_count": len(dataset.questions),
        },
    }

    print(f"Running {args.experiment_id} over {len(dataset.questions)} questions...")
    pipeline = BaselineRAG(settings)
    results = BaselineEvaluator(pipeline).run(dataset, args.experiment_id)
    output_directory = args.output_root.resolve() / args.experiment_id
    config_path, results_path = write_experiment(output_directory, config, results)
    print(f"Recall@5: {results['summary']['recall_at_5']:.3f}")
    print(f"Configuration: {config_path}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()

