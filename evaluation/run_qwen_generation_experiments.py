from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.baseline import BaselineRAG, DenseRAGResources  # noqa: E402
from i_got_this_rag.chat_models import get_chat_model  # noqa: E402
from i_got_this_rag.claim_faithfulness import (  # noqa: E402
    RetrievedContext,
    audit_question,
)
from i_got_this_rag.concise_generation import (  # noqa: E402
    CONCISE_RELEVANCE_PROMPT,
    GroundedGeneration,
    generate_qwen_experiment_answer,
)
from i_got_this_rag.current_app_evaluation import (  # noqa: E402
    classify_generation_error,
    evaluate_current_app,
)
from i_got_this_rag.evaluation import (  # noqa: E402
    load_evaluation_dataset,
    serialize_retrieval,
    utc_now,
)
from i_got_this_rag.final_evaluation import nearest_rank_percentile  # noqa: E402
from i_got_this_rag.grounded_generation import STRICT_GROUNDING_PROMPT  # noqa: E402
from i_got_this_rag.ingestion import (  # noqa: E402
    chunk_documents,
    chunk_fingerprint,
    corpus_fingerprint,
    load_corpus,
)
from i_got_this_rag.qwen_generation_experiments import (  # noqa: E402
    QwenGenerationExperiment,
    load_qwen_generation_experiments,
    mode_succeeds,
)
from i_got_this_rag.settings import Settings  # noqa: E402
from i_got_this_rag.user_interface import select_relevant_ui_results  # noqa: E402


QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
CONFIG_DIR = PROJECT_ROOT / "config" / "qwen_generation_experiments"
RESULTS_ROOT = PROJECT_ROOT / "evaluation" / "results"
COMPARISON_PATH = RESULTS_ROOT / "qwen_generation_comparison.json"
REPORT_PATH = RESULTS_ROOT / "qwen_generation_comparison.md"
SOURCE_PATHS = (
    PROJECT_ROOT / "src" / "i_got_this_rag" / "chat_models.py",
    PROJECT_ROOT / "src" / "i_got_this_rag" / "concise_generation.py",
    PROJECT_ROOT / "src" / "i_got_this_rag" / "qwen_generation_experiments.py",
    PROJECT_ROOT / "src" / "i_got_this_rag" / "claim_faithfulness.py",
    PROJECT_ROOT / "evaluation" / "run_qwen_generation_experiments.py",
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


class FixedRetrievalQwenPipeline:
    def __init__(
        self,
        settings: Settings,
        experiment: QwenGenerationExperiment,
        llm: Any,
        retrieval_cache: dict[str, list[tuple[Document, float]]],
    ) -> None:
        self.settings = settings
        self.experiment = experiment
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
        generated = generate_qwen_experiment_answer(
            llm=self.resources.llm,
            question=question,
            results=results,
            reference_date=self.settings.reference_date,
            timezone=self.settings.timezone,
            prompt_mode=self.experiment.prompt_mode,
            evidence_mode=self.experiment.evidence_mode,
            length_policy=self.experiment.length_policy,
        )
        self.last_generation_trace = generated.trace()
        return generated


def _file_record(path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _evidence_record(
    results: list[tuple[Document, float]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rank, (document, score) in enumerate(results, start=1):
        metadata = document.metadata
        records.append(
            {
                "source_id": f"S{rank}",
                "rank": rank,
                "document_id": str(metadata.get("document_id", "")),
                "document_title": str(metadata.get("document_title", "")),
                "chunk_id": str(metadata.get("chunk_id", "")),
                "domain": str(metadata.get("domain", "")),
                "source_path": str(metadata.get("source_path", "")),
                "similarity_score": float(score),
                "text": document.page_content,
            }
        )
    return records


def _retrieval_fingerprint(
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


def _claim_contexts(
    ui_results: list[tuple[Document, float]],
    trace: dict[str, Any] | None,
) -> tuple[RetrievedContext, ...]:
    included = (
        set(str(value) for value in trace.get("context_source_ids", []))
        if isinstance(trace, dict)
        else set()
    )
    contexts: list[RetrievedContext] = []
    for rank, (document, _) in enumerate(ui_results, start=1):
        source_id = f"S{rank}"
        if included and source_id not in included:
            continue
        metadata = document.metadata
        contexts.append(
            RetrievedContext(
                source_id=source_id,
                document_id=str(metadata.get("document_id", "")),
                chunk_id=str(metadata.get("chunk_id", "")),
                title=str(metadata.get("document_title", "")),
                domain=str(metadata.get("domain", "")),
                text=document.page_content,
            )
        )
    return tuple(contexts)


def _enrich_with_claims_and_usage(
    evaluation: dict[str, Any],
    retrieval_cache: dict[str, list[tuple[Document, float]]],
    reference_date: str,
    model: str,
) -> dict[str, Any]:
    total_claims = 0
    supported_claims = 0
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    total_tokens: list[int] = []
    generation_latencies: list[float] = []
    for question in evaluation["questions"]:
        text = str(question["question"])
        original = retrieval_cache[text]
        ui_results = select_relevant_ui_results(text, original, reference_date)
        trace = question.get("generation_trace")
        contexts = _claim_contexts(ui_results, trace if isinstance(trace, dict) else None)
        claim_audit = audit_question(question, contexts, model, reference_date)
        question["claim_analysis"] = claim_audit
        question["original_top5_evidence"] = _evidence_record(original)
        question["answer_path_evidence"] = _evidence_record(ui_results)
        total_claims += int(claim_audit["total_factual_claims"])
        supported_claims += int(claim_audit["supported_factual_claims"])

        usage = trace.get("token_usage") if isinstance(trace, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        input_tokens.append(int(usage.get("input_tokens", 0) or 0))
        output_tokens.append(int(usage.get("output_tokens", 0) or 0))
        total_tokens.append(int(usage.get("total_tokens", 0) or 0))
        generation_latencies.append(
            float(trace.get("generation_latency_seconds", 0.0) or 0.0)
            if isinstance(trace, dict)
            else 0.0
        )
        question["token_usage"] = {
            "input_tokens": input_tokens[-1],
            "output_tokens": output_tokens[-1],
            "total_tokens": total_tokens[-1],
        }
        question["generation_latency_seconds"] = generation_latencies[-1]

    metrics = evaluation["metrics"]
    metrics["claim_level_faithfulness"] = (
        supported_claims / total_claims if total_claims else None
    )
    metrics["total_factual_claims"] = total_claims
    metrics["supported_factual_claims"] = supported_claims
    metrics["unsupported_claims"] = total_claims - supported_claims
    metrics["average_claims_per_answer"] = total_claims / len(evaluation["questions"])
    metrics["average_input_tokens"] = mean(input_tokens)
    metrics["average_output_tokens"] = mean(output_tokens)
    metrics["average_total_tokens"] = mean(total_tokens)
    metrics["average_generation_latency_seconds"] = mean(generation_latencies)
    metrics["p95_generation_latency_seconds"] = nearest_rank_percentile(
        generation_latencies,
        0.95,
    )
    metrics["token_usage_available_count"] = sum(value > 0 for value in total_tokens)
    return evaluation


def _build_report(comparison: dict[str, Any]) -> str:
    lines = [
        "# Qwen concise-generation experiment",
        "",
        "One immutable Top-5 retrieval cache was reused by every mode.",
        "",
        "| Mode | Recall@5 | Claim Faithfulness | Relevance | Refusal | Claims/Answer | Output Tokens | Avg Latency | P95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for version in comparison["versions"]:
        metrics = version["metrics"]
        lines.append(
            f"| {version['label']} | {metrics['recall_at_5']:.3f} | "
            f"{metrics['claim_level_faithfulness']:.3f} | "
            f"{metrics['answer_relevance_correctness']:.3f} | "
            f"{metrics['correct_refusal_rate']:.3f} | "
            f"{metrics['average_claims_per_answer']:.3f} | "
            f"{metrics['average_output_tokens']:.1f} | "
            f"{metrics['average_latency_seconds']:.3f}s | "
            f"{metrics['p95_latency_seconds']:.3f}s |"
        )
    lines.extend(("", "## Success criteria", ""))
    for version in comparison["versions"]:
        status = "PASS" if version["meets_success_criteria"] else "NOT MET"
        lines.append(f"- **{version['label']}: {status}**")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the controlled Qwen E1/E2/E3 generation comparison."
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        help="Optional experiment IDs to rerun; other completed modes are reused.",
    )
    arguments = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    settings = Settings.from_environment(PROJECT_ROOT)
    dataset = load_evaluation_dataset(QUESTIONS_PATH)
    experiments = load_qwen_generation_experiments(CONFIG_DIR, PROJECT_ROOT)
    selected_ids = set(arguments.modes or (item.experiment_id for item in experiments))
    known_ids = {item.experiment_id for item in experiments}
    unknown_ids = selected_ids - known_ids
    if unknown_ids:
        raise ValueError("Unknown Qwen experiment IDs: " + ", ".join(sorted(unknown_ids)))

    print("Connecting once to the unchanged dense Top-5 retrieval pipeline...")
    resources = DenseRAGResources.connect(settings)
    retrieval_pipeline = BaselineRAG(settings, resources=resources)
    retrieval_cache = {
        str(question["question"]): retrieval_pipeline.retrieve(str(question["question"]))
        for question in dataset.questions
    }
    retrieval_snapshot = _retrieval_fingerprint(retrieval_cache)
    print(
        f"Cached the same retrieval results for {len(dataset.questions)} questions: "
        f"{retrieval_snapshot['sha256']}"
    )

    documents = load_corpus(settings.data_dir, PROJECT_ROOT)
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    strict_prompt_sha = hashlib.sha256(
        STRICT_GROUNDING_PROMPT.pretty_repr().encode("utf-8")
    ).hexdigest()
    concise_prompt_sha = hashlib.sha256(
        CONCISE_RELEVANCE_PROMPT.pretty_repr().encode("utf-8")
    ).hexdigest()
    source_code = [_file_record(path) for path in SOURCE_PATHS]
    completed: list[dict[str, Any]] = []

    for experiment in experiments:
        if experiment.experiment_id not in selected_ids:
            result_path = RESULTS_ROOT / experiment.experiment_id / "results.json"
            if not result_path.is_file():
                raise FileNotFoundError(
                    f"Cannot reuse {experiment.experiment_id}; {result_path} is missing."
                )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload.get("run_status") != "complete":
                raise ValueError(
                    f"Cannot reuse incomplete mode {experiment.experiment_id}."
                )
            if payload.get("retrieval_cache_sha256") != retrieval_snapshot["sha256"]:
                raise ValueError(
                    f"Cannot reuse {experiment.experiment_id}; retrieval fingerprint differs."
                )
            payload["result_path"] = result_path.relative_to(PROJECT_ROOT).as_posix()
            payload["meets_success_criteria"] = mode_succeeds(payload["metrics"])
            completed.append(payload)
            print(f"\n[{experiment.experiment_id}] reusing complete saved result")
            continue
        print(f"\n[{experiment.experiment_id}] {experiment.label}")
        configuration_error: dict[str, str] | None = None
        try:
            llm = get_chat_model(experiment.chat_config)
        except Exception as exc:
            configuration_error = classify_generation_error(exc)
            llm = FailingChatModel(exc)
        pipeline = FixedRetrievalQwenPipeline(
            settings,
            experiment,
            llm,
            retrieval_cache,
        )
        evaluation = evaluate_current_app(
            pipeline,
            dataset,
            continue_on_generation_error=True,
        )
        evaluation = _enrich_with_claims_and_usage(
            evaluation,
            retrieval_cache,
            dataset.reference_date,
            experiment.chat_config.model,
        )
        metrics = evaluation["metrics"]
        run_status = (
            "configuration_error"
            if configuration_error
            else "partial_failure"
            if int(metrics["generation_failure_count"]) > 0
            else "complete"
        )
        fixed_retrieval = {
            "corpus_document_count": 20,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "pinecone_index": settings.pinecone_index_name,
            "pinecone_namespace": settings.pinecone_namespace,
            "strategy": "dense",
            "top_k": settings.top_k,
            "retrieval_cache_sha256": retrieval_snapshot["sha256"],
        }
        public_config = experiment.public_config(PROJECT_ROOT)
        public_config["prompt_sha256"] = (
            strict_prompt_sha
            if experiment.prompt_mode == "current_strict"
            else concise_prompt_sha
        )
        public_config["fixed_retrieval"] = fixed_retrieval
        result_dir = RESULTS_ROOT / experiment.experiment_id
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "config.json").write_text(
            json.dumps(public_config, indent=2) + "\n",
            encoding="utf-8",
        )
        payload = {
            "schema_version": "1.0",
            "experiment_suite": "qwen_concise_relevance_comparison",
            "experiment_id": experiment.experiment_id,
            "label": experiment.label,
            "completed_at": utc_now(),
            "run_status": run_status,
            "configuration_error": configuration_error,
            "active_model": experiment.public_model_config,
            "changed_generation": {
                "prompt_mode": experiment.prompt_mode,
                "evidence_mode": experiment.evidence_mode,
                "length_policy": experiment.length_policy.to_dict(),
                "max_output_tokens": experiment.chat_config.max_output_tokens,
            },
            "fixed_retrieval": fixed_retrieval,
            "evaluation_dataset": {
                "path": QUESTIONS_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": dataset.sha256,
                "question_count": len(dataset.questions),
                "reference_date": dataset.reference_date,
            },
            "metric_methods": {
                "claim_level_faithfulness": "unchanged explicit claim support audit",
                "answer_relevance_correctness": "deterministic expected-answer token F1",
                "average_output_tokens": "provider-reported output tokens averaged over all 15 answers; no-call deterministic answers count as zero",
                "latency": "end-to-end Ask-path latency; generation-only latency also recorded",
            },
            "runtime_settings": settings.public_config(),
            "corpus": corpus_fingerprint(settings.data_dir, PROJECT_ROOT),
            "chunk_set": chunk_fingerprint(chunks),
            "source_code": source_code,
            "retrieval_cache_sha256": retrieval_snapshot["sha256"],
            "metrics": metrics,
            "category_summary": evaluation["category_summary"],
            "questions": evaluation["questions"],
        }
        result_path = result_dir / "results.json"
        result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        payload["result_path"] = result_path.relative_to(PROJECT_ROOT).as_posix()
        payload["meets_success_criteria"] = mode_succeeds(metrics)
        completed.append(payload)
        print(
            f"status={run_status}; recall={metrics['recall_at_5']:.3f}; "
            f"claim faith={metrics['claim_level_faithfulness']:.3f}; "
            f"relevance={metrics['answer_relevance_correctness']:.3f}; "
            f"refusal={metrics['correct_refusal_rate']:.3f}; "
            f"claims/answer={metrics['average_claims_per_answer']:.2f}; "
            f"output tokens={metrics['average_output_tokens']:.1f}; "
            f"avg={metrics['average_latency_seconds']:.3f}s; "
            f"p95={metrics['p95_latency_seconds']:.3f}s"
        )

    fingerprints = {item["retrieval_cache_sha256"] for item in completed}
    models = {item["active_model"]["model"] for item in completed}
    if len(fingerprints) != 1 or len(models) != 1:
        raise RuntimeError("Controlled experiment invariant failed after evaluation.")
    comparison = {
        "schema_version": "1.0",
        "experiment_suite": "qwen_concise_relevance_comparison",
        "completed_at": utc_now(),
        "question": (
            "Can selective evidence and shorter generation improve relevance and latency "
            "while preserving Qwen claim-level faithfulness?"
        ),
        "fixed_model": next(iter(models)),
        "fixed_retrieval_cache_sha256": next(iter(fingerprints)),
        "success_criteria": {
            "recall_at_5_minimum": 0.90,
            "claim_level_faithfulness_minimum": 0.95,
            "relevance_correctness_strictly_greater_than": 0.588,
            "correct_refusal_required": 1.0,
            "fewer_claims_than_E1": True,
            "lower_output_tokens_than_E1": True,
            "lower_average_and_p95_latency_than_E1": True,
        },
        "versions": [
            {
                "experiment_id": item["experiment_id"],
                "label": item["label"],
                "run_status": item["run_status"],
                "model": item["active_model"]["model"],
                "changed_generation": item["changed_generation"],
                "metrics": item["metrics"],
                "meets_quality_thresholds": item["meets_success_criteria"],
                "result_path": item["result_path"],
            }
            for item in completed
        ],
    }
    baseline_metrics = comparison["versions"][0]["metrics"]
    for version in comparison["versions"]:
        metrics = version["metrics"]
        version["meets_success_criteria"] = bool(
            mode_succeeds(metrics)
            and float(metrics["average_claims_per_answer"])
            < float(baseline_metrics["average_claims_per_answer"])
            and float(metrics["average_output_tokens"])
            < float(baseline_metrics["average_output_tokens"])
            and float(metrics["average_latency_seconds"])
            < float(baseline_metrics["average_latency_seconds"])
            and float(metrics["p95_latency_seconds"])
            < float(baseline_metrics["p95_latency_seconds"])
        )
    COMPARISON_PATH.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(_build_report(comparison), encoding="utf-8")
    print(f"\nComparison: {COMPARISON_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
