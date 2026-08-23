from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.baseline import (  # noqa: E402
    PLAIN_LANGUAGE_ANSWER_STYLE,
    BaselineRAG,
    DenseRAGResources,
)
from i_got_this_rag.current_app_evaluation import evaluate_current_app  # noqa: E402
from i_got_this_rag.evaluation import load_evaluation_dataset, utc_now  # noqa: E402
from i_got_this_rag.grounded_generation import GENERATION_MODES  # noqa: E402
from i_got_this_rag.ingestion import (  # noqa: E402
    chunk_documents,
    chunk_fingerprint,
    corpus_fingerprint,
    load_corpus,
)
from i_got_this_rag.settings import Settings  # noqa: E402


QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
CONFIG_DIR = PROJECT_ROOT / "config" / "generation_experiments"
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results" / "generation_ablation"
CONFIG_NAMES = ("current.yaml", "strict_prompt.yaml", "strict_prompt_filter.yaml")
SOURCE_PATHS = (
    PROJECT_ROOT / "src" / "i_got_this_rag" / "baseline.py",
    PROJECT_ROOT / "src" / "i_got_this_rag" / "grounded_generation.py",
    PROJECT_ROOT / "src" / "i_got_this_rag" / "user_interface.py",
    PROJECT_ROOT / "src" / "i_got_this_rag" / "current_app_evaluation.py",
)


def file_record(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def load_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for name in CONFIG_NAMES:
        path = CONFIG_DIR / name
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Generation config must be a mapping: {path}")
        mode = str(payload.get("generation_mode", ""))
        if mode not in GENERATION_MODES:
            raise ValueError(f"Unsupported generation mode in {path}: {mode}")
        payload["config_file"] = file_record(path)
        configs.append(payload)
    return configs


def comparison_row(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": payload["experiment_id"],
        "label": payload["label"],
        "generation_mode": payload["generation_mode"],
        "metrics": payload["metrics"],
        "result_path": payload["result_path"],
    }


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    settings = Settings.from_environment(PROJECT_ROOT)
    dataset = load_evaluation_dataset(QUESTIONS_PATH)
    configs = load_configs()

    print("Connecting once to the unchanged dense Top-5 retrieval pipeline...")
    resources = DenseRAGResources.connect(settings)
    documents = load_corpus(settings.data_dir, PROJECT_ROOT)
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    source_code = [file_record(path) for path in SOURCE_PATHS]
    completed: list[dict[str, Any]] = []

    for config in configs:
        mode = str(config["generation_mode"])
        print(f"\n[{config['experiment_id']}] Evaluating {config['label']}...")
        pipeline = BaselineRAG(
            settings,
            resources=resources,
            answer_style=PLAIN_LANGUAGE_ANSWER_STYLE,
            generation_mode=mode,
        )
        evaluation = evaluate_current_app(pipeline, dataset)
        result_dir = OUTPUT_DIR / str(config["experiment_id"])
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / "results.json"
        saved_config = {
            **config,
            "fixed_retrieval": {
                "strategy": "dense",
                "top_k": settings.top_k,
                "index": settings.pinecone_index_name,
                "namespace": settings.pinecone_namespace,
                "embedding_model": settings.embedding_model,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
            },
        }
        (result_dir / "config.json").write_text(
            json.dumps(saved_config, indent=2) + "\n",
            encoding="utf-8",
        )
        payload = {
            "schema_version": "1.0",
            "experiment_suite": "generation_grounding_ablation",
            "experiment_id": config["experiment_id"],
            "label": config["label"],
            "generation_mode": mode,
            "completed_at": utc_now(),
            "changed_components": {
                "strict_grounding_prompt": bool(config["strict_grounding_prompt"]),
                "relevance_filtering": bool(config["relevance_filtering"]),
            },
            "fixed_components": saved_config["fixed_retrieval"],
            "evaluation_dataset": {
                "path": QUESTIONS_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": dataset.sha256,
                "question_count": len(dataset.questions),
            },
            "metric_methods": {
                "faithfulness": "phase10 deterministic citation grounding on confirmed facts",
                "answer_relevance_correctness": (
                    "deterministic token F1 between the confirmed answer and expected answer"
                ),
                "correct_refusal": f"exact match to the configured refusal for unanswerable questions",
            },
            "source_code": source_code,
            "active_runtime_settings": settings.public_config(),
            "corpus": corpus_fingerprint(settings.data_dir, PROJECT_ROOT),
            "chunk_set": chunk_fingerprint(chunks),
            "metrics": evaluation["metrics"],
            "category_summary": evaluation["category_summary"],
            "questions": evaluation["questions"],
            "result_path": result_path.relative_to(PROJECT_ROOT).as_posix(),
        }
        result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        completed.append(payload)
        metrics = payload["metrics"]
        print(
            f"Recall@5={metrics['recall_at_5']:.3f}; "
            f"faithfulness={metrics['faithfulness']:.3f}; "
            f"answer relevance/correctness="
            f"{metrics['answer_relevance_correctness']:.3f}; "
            f"correct refusal={metrics['correct_refusal_rate']:.3f}; "
            f"average={metrics['average_latency_seconds']:.3f}s; "
            f"p95={metrics['p95_latency_seconds']:.3f}s"
        )

    baseline_metrics = completed[0]["metrics"]
    comparison = {
        "schema_version": "1.0",
        "experiment_suite": "generation_grounding_ablation",
        "completed_at": utc_now(),
        "isolation_statement": (
            "Embeddings, Pinecone index and namespace, chunking, dense retrieval, "
            "Top-K, evaluation questions, and application routing were fixed. Only "
            "the generation prompt and generator-context filtering varied."
        ),
        "versions": [comparison_row(payload) for payload in completed],
        "deltas_vs_current": {
            payload["experiment_id"]: {
                metric: round(
                    float(payload["metrics"][metric]) - float(baseline_metrics[metric]),
                    6,
                )
                for metric in (
                    "recall_at_5",
                    "faithfulness",
                    "answer_relevance_correctness",
                    "correct_refusal_rate",
                    "average_latency_seconds",
                    "p95_latency_seconds",
                )
            }
            for payload in completed
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = OUTPUT_DIR / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nComparison: {comparison_path}")


if __name__ == "__main__":
    main()
