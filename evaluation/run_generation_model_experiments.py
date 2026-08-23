from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.baseline import BaselineRAG, DenseRAGResources  # noqa: E402
from i_got_this_rag.chat_models import get_chat_model  # noqa: E402
from i_got_this_rag.current_app_evaluation import (  # noqa: E402
    classify_generation_error,
    evaluate_current_app,
)
from i_got_this_rag.evaluation import (  # noqa: E402
    load_evaluation_dataset,
    serialize_retrieval,
    utc_now,
)
from i_got_this_rag.generation_model_experiments import (  # noqa: E402
    GenerationModelExperiment,
    build_model_comparison,
    load_generation_model_experiments,
)
from i_got_this_rag.grounded_generation import (  # noqa: E402
    GroundedGeneration,
    STRICT_GROUNDING_PROMPT,
    generate_strict_grounded_answer,
)
from i_got_this_rag.ingestion import (  # noqa: E402
    chunk_documents,
    chunk_fingerprint,
    corpus_fingerprint,
    load_corpus,
)
from i_got_this_rag.settings import Settings  # noqa: E402


QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
CONFIG_DIR = PROJECT_ROOT / "config" / "generation_model_experiments"
RESULTS_ROOT = PROJECT_ROOT / "evaluation" / "results"
COMPARISON_PATH = RESULTS_ROOT / "generation_model_comparison.json"
SOURCE_PATHS = (
    PROJECT_ROOT / "src" / "i_got_this_rag" / "chat_models.py",
    PROJECT_ROOT / "src" / "i_got_this_rag" / "grounded_generation.py",
    PROJECT_ROOT / "src" / "i_got_this_rag" / "generation_model_experiments.py",
    PROJECT_ROOT / "src" / "i_got_this_rag" / "current_app_evaluation.py",
    PROJECT_ROOT / "evaluation" / "run_generation_model_experiments.py",
)


class ErrorRunnable:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def invoke(self, _: Any) -> Any:
        raise self.error


class FailingChatModel:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def with_structured_output(self, *_: Any, **__: Any) -> ErrorRunnable:
        return ErrorRunnable(self.error)

    def invoke(self, _: Any) -> Any:
        raise self.error


class FixedContextStrictPipeline:
    """Mode B generator over one immutable retrieval cache."""

    def __init__(
        self,
        settings: Settings,
        llm: Any,
        retrieval_cache: dict[str, list[tuple[Document, float]]],
    ) -> None:
        self.settings = settings
        self.resources = SimpleNamespace(llm=llm)
        self.retrieval_cache = retrieval_cache
        self.last_generation_trace: dict[str, Any] | None = None

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        return list(self.retrieval_cache[question])

    def generate(
        self,
        question: str,
        results: list[tuple[Document, float]],
    ) -> GroundedGeneration:
        generated = generate_strict_grounded_answer(
            llm=self.resources.llm,
            question=question,
            results=results,
            reference_date=self.settings.reference_date,
            timezone=self.settings.timezone,
            filter_context=False,
        )
        self.last_generation_trace = generated.trace()
        return generated


def file_record(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def retrieval_cache_fingerprint(
    cache: dict[str, list[tuple[Document, float]]],
) -> dict[str, Any]:
    serialized = {
        question: serialize_retrieval(results)
        for question, results in cache.items()
    }
    canonical = json.dumps(serialized, sort_keys=True, separators=(",", ":"))
    return {
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "questions": serialized,
    }


def create_model(
    experiment: GenerationModelExperiment,
) -> tuple[Any, dict[str, str] | None]:
    try:
        return get_chat_model(experiment.chat_model_config()), None
    except Exception as exc:
        return FailingChatModel(exc), classify_generation_error(exc)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    settings = Settings.from_environment(PROJECT_ROOT)
    dataset = load_evaluation_dataset(QUESTIONS_PATH)
    experiments = load_generation_model_experiments(CONFIG_DIR)

    print("Connecting to the unchanged embedding and Pinecone retrieval resources...")
    retrieval_resources = DenseRAGResources.connect(settings)
    retrieval_pipeline = BaselineRAG(settings, resources=retrieval_resources)
    retrieval_cache = {
        str(question["question"]): retrieval_pipeline.retrieve(str(question["question"]))
        for question in dataset.questions
    }
    retrieval_snapshot = retrieval_cache_fingerprint(retrieval_cache)
    print(
        "Cached one immutable Top-5 context for each of the "
        f"{len(dataset.questions)} evaluation questions."
    )

    documents = load_corpus(settings.data_dir, PROJECT_ROOT)
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    source_code = [file_record(path) for path in SOURCE_PATHS]
    strict_prompt_sha256 = hashlib.sha256(
        STRICT_GROUNDING_PROMPT.pretty_repr().encode("utf-8")
    ).hexdigest()
    completed: list[dict[str, Any]] = []

    for experiment in experiments:
        public_model_config = experiment.public_config(PROJECT_ROOT)
        display_model = experiment.model or f"<{experiment.model_env or 'missing model'}>"
        print(
            f"\n[{experiment.experiment_id}] {experiment.provider} / {display_model}"
        )
        llm, configuration_error = create_model(experiment)
        pipeline = FixedContextStrictPipeline(
            settings,
            llm,
            retrieval_cache,
        )
        evaluation = evaluate_current_app(
            pipeline,
            dataset,
            continue_on_generation_error=True,
        )
        metrics = evaluation["metrics"]
        run_status = (
            "configuration_error"
            if configuration_error is not None
            else "partial_failure"
            if int(metrics["generation_failure_count"]) > 0
            else "complete"
        )
        result_dir = RESULTS_ROOT / experiment.experiment_id
        result_dir.mkdir(parents=True, exist_ok=True)
        resolved_config = {
            **public_model_config,
            "generation_mode": "strict_prompt",
            "relevance_filtering": False,
            "strict_prompt_sha256": strict_prompt_sha256,
            "fixed_retrieval": {
                "corpus_document_count": 20,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "embedding_model": settings.embedding_model,
                "pinecone_index": settings.pinecone_index_name,
                "pinecone_namespace": settings.pinecone_namespace,
                "strategy": "dense",
                "top_k": settings.top_k,
                "retrieval_cache_sha256": retrieval_snapshot["sha256"],
            },
        }
        (result_dir / "config.json").write_text(
            json.dumps(resolved_config, indent=2) + "\n",
            encoding="utf-8",
        )
        payload = {
            "schema_version": "1.0",
            "experiment_suite": "strict_prompt_generation_model_comparison",
            "experiment_id": experiment.experiment_id,
            "label": experiment.label,
            "completed_at": utc_now(),
            "run_status": run_status,
            "configuration_error": configuration_error,
            "active_model": public_model_config,
            "single_variable": "generation model",
            "fixed_generation": {
                "generation_mode": "strict_prompt",
                "relevance_filtering": False,
                "structured_output": True,
            },
            "fixed_retrieval": resolved_config["fixed_retrieval"],
            "evaluation_dataset": {
                "path": QUESTIONS_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": dataset.sha256,
                "question_count": len(dataset.questions),
            },
            "metric_methods": {
                "faithfulness": "phase10 deterministic citation grounding on confirmed facts",
                "answer_relevance_correctness": (
                    "deterministic token F1 between confirmed and expected answers"
                ),
                "correct_refusal": "exact standard refusal on unanswerable questions",
            },
            "source_code": source_code,
            "runtime_settings": settings.public_config(),
            "corpus": corpus_fingerprint(settings.data_dir, PROJECT_ROOT),
            "chunk_set": chunk_fingerprint(chunks),
            "retrieval_cache_sha256": retrieval_snapshot["sha256"],
            "metrics": metrics,
            "category_summary": evaluation["category_summary"],
            "questions": evaluation["questions"],
        }
        result_path = result_dir / "results.json"
        result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        payload["result_path"] = result_path.relative_to(PROJECT_ROOT).as_posix()
        completed.append(payload)
        print(
            f"status={run_status}; Recall@5={metrics['recall_at_5']:.3f}; "
            f"faithfulness={metrics['faithfulness']:.3f}; "
            f"relevance={metrics['answer_relevance_correctness']:.3f}; "
            f"refusal={metrics['correct_refusal_rate']:.3f}; "
            f"avg={metrics['average_latency_seconds']:.3f}s; "
            f"p95={metrics['p95_latency_seconds']:.3f}s; "
            f"generation failures={metrics['generation_failure_count']}"
        )

    ranking = build_model_comparison(completed)
    comparison = {
        "schema_version": "1.0",
        "experiment_suite": "strict_prompt_generation_model_comparison",
        "completed_at": utc_now(),
        "single_variable": "generation model",
        "fixed_components": {
            "corpus_document_count": 20,
            "retrieval_cache_sha256": retrieval_snapshot["sha256"],
            "strict_prompt": True,
            "strict_prompt_sha256": strict_prompt_sha256,
            "relevance_filtering": False,
            "evaluation_question_count": len(dataset.questions),
        },
        "versions": [
            {
                "experiment_id": result["experiment_id"],
                "label": result["label"],
                "provider": result["active_model"]["provider"],
                "model": result["active_model"]["model"],
                "run_status": result["run_status"],
                "configuration_error": result["configuration_error"],
                "metrics": result["metrics"],
                "result_path": result["result_path"],
            }
            for result in completed
        ],
        **ranking,
    }
    COMPARISON_PATH.write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nComparison: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
