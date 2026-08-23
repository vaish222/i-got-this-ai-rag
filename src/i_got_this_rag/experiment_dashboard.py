from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentProfile:
    version_id: str
    embedding: str
    chunking: str
    retrieval: str
    reranking: str
    changed: str


@dataclass(frozen=True)
class ExperimentRow:
    version_id: str
    experiment: str
    version: str
    embedding: str
    chunking: str
    retrieval: str
    reranking: str
    recall_at_5: float
    faithfulness: float
    average_latency_seconds: float

    def table_record(self) -> dict[str, str | float]:
        return {
            "Experiment": self.experiment,
            "Version": self.version,
            "Embedding": self.embedding,
            "Chunk": self.chunking,
            "Retrieval": self.retrieval,
            "Rerank": self.reranking,
            "Recall@5": self.recall_at_5,
            "Faithfulness": self.faithfulness,
            "Avg. latency (s)": self.average_latency_seconds,
        }


@dataclass(frozen=True)
class ExperimentDetail:
    version_id: str
    experiment: str
    version: str
    changed: str
    stayed_constant: str
    improved: str
    became_worse: str
    why: str
    latency_cost: str
    worth_it: str


@dataclass(frozen=True)
class ExperimentDashboard:
    completed_at: str
    rows: tuple[ExperimentRow, ...]
    details: tuple[ExperimentDetail, ...]
    best_recall_at_5: float
    best_faithfulness: float
    fastest_average_latency_seconds: float
    recommendation_version_id: str
    recommendation_rationale: str

    def detail_for(self, version_id: str) -> ExperimentDetail:
        return next(detail for detail in self.details if detail.version_id == version_id)


@dataclass(frozen=True)
class CurrentAppBenchmark:
    completed_at: str
    experiment_id: str
    recall_at_5: float
    faithfulness: float
    correct_refusal_rate: float
    average_latency_seconds: float
    p95_latency_seconds: float
    recall_delta: float
    faithfulness_delta: float
    average_latency_delta_seconds: float
    regression_passed_count: int
    regression_case_count: int
    regression_pass_rate: float
    regression_cases: tuple[dict[str, Any], ...]

    def table_record(self) -> dict[str, str | float]:
        return {
            "System": "Current Streamlit app",
            "Recall@5": self.recall_at_5,
            "Faithfulness": self.faithfulness,
            "Correct refusal": self.correct_refusal_rate,
            "Avg. latency (s)": self.average_latency_seconds,
            "P95 latency (s)": self.p95_latency_seconds,
            "UI regressions": (
                f"{self.regression_passed_count}/{self.regression_case_count}"
            ),
        }


COMMON_CONSTANTS = (
    "20-document controlled corpus; 500/75 chunks; final Top-5 context; "
    "gemma3:1b generation; the same 15 evaluation questions and reference date."
)

EXPERIMENT_PROFILES = (
    ExperimentProfile(
        "baseline_dense",
        "embeddinggemma",
        "500 / 75",
        "Dense Top-5",
        "No",
        "Reference dense RAG configuration.",
    ),
    ExperimentProfile(
        "best_chunking",
        "embeddinggemma",
        "500 / 75",
        "Dense Top-5",
        "No",
        "Selected Phase 3 chunking result was measured as its own run.",
    ),
    ExperimentProfile(
        "best_embedding",
        "mxbai-embed-large",
        "500 / 75",
        "Dense Top-5",
        "No",
        "Replaced embeddinggemma with the strongest measured Phase 4 embedding.",
    ),
    ExperimentProfile(
        "hybrid_retrieval",
        "embeddinggemma",
        "500 / 75",
        "Dense + BM25 RRF",
        "No",
        "Fused dense and lexical BM25 rankings with reciprocal-rank fusion.",
    ),
    ExperimentProfile(
        "hybrid_reranker",
        "embeddinggemma",
        "500 / 75",
        "Hybrid Top-20",
        "BM25 → Top-5",
        "Expanded hybrid candidates to 20 and reranked them with BM25 to final Top-5.",
    ),
    ExperimentProfile(
        "metadata_aware",
        "embeddinggemma",
        "500 / 75",
        "Metadata + dense",
        "No",
        "Added deterministic metadata filters with unfiltered dense fallback.",
    ),
    ExperimentProfile(
        "query_rewriting",
        "embeddinggemma",
        "500 / 75",
        "Rewritten dense",
        "No",
        "Added one guarded local-LLM query rewrite before dense retrieval.",
    ),
    ExperimentProfile(
        "langgraph_workflow",
        "embeddinggemma",
        "500 / 75",
        "LangGraph dense",
        "Disabled",
        "Added evidence grading, one guarded retry, and grounding-based refusal.",
    ),
)

def _question_change(
    metric: str,
    delta: float,
    question_ids: list[str],
    direction: str,
) -> str:
    direction_matches_delta = (direction == "improved" and delta > 0) or (
        direction == "degraded" and delta < 0
    )
    if not direction_matches_delta and not question_ids:
        return f"No {metric} {direction}; aggregate delta {delta:+.3f}."
    aggregate = (
        f"aggregate {direction} by {abs(delta):.3f}"
        if direction_matches_delta
        else f"aggregate delta {delta:+.3f}"
    )
    questions = ", ".join(question_ids) if question_ids else "no individual questions"
    return f"{metric}: {aggregate}; {direction} on {questions}."


def _load_payload(path: Path) -> dict[str, Any]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("phase") != 10:
        raise ValueError("The experiment dashboard requires a Phase 10 comparison artifact.")
    if not isinstance(payload.get("versions"), list):
        raise ValueError("Phase 10 comparison is missing its version results.")
    if not isinstance(payload.get("findings"), list):
        raise ValueError("Phase 10 comparison is missing its experiment findings.")
    return payload


def load_experiment_dashboard(path: Path) -> ExperimentDashboard:
    payload = _load_payload(path)
    versions = {str(row["version_id"]): row for row in payload["versions"]}
    findings = {str(row["version_id"]): row for row in payload["findings"]}
    expected_ids = tuple(profile.version_id for profile in EXPERIMENT_PROFILES)
    if tuple(versions) != expected_ids or tuple(findings) != expected_ids:
        raise ValueError(
            "The experiment dashboard requires the exact eight-version Phase 10 matrix."
        )

    rows: list[ExperimentRow] = []
    details: list[ExperimentDetail] = []
    for profile in EXPERIMENT_PROFILES:
        version = versions[profile.version_id]
        finding = findings[profile.version_id]
        metrics = version["metrics"]
        experiment = str(version.get("source_experiment_id") or profile.version_id)
        rows.append(
            ExperimentRow(
                version_id=profile.version_id,
                experiment=experiment,
                version=str(version["label"]),
                embedding=profile.embedding,
                chunking=profile.chunking,
                retrieval=profile.retrieval,
                reranking=profile.reranking,
                recall_at_5=float(metrics["recall_at_5"]),
                faithfulness=float(metrics["faithfulness"]),
                average_latency_seconds=float(metrics["average_latency_seconds"]),
            )
        )
        improved = " ".join(
            (
                _question_change(
                    "Recall@5",
                    float(finding["recall_at_5_delta_vs_baseline"]),
                    list(finding["recall_improved_question_ids"]),
                    "improved",
                ),
                _question_change(
                    "Faithfulness",
                    float(finding["faithfulness_delta_vs_baseline"]),
                    list(finding["faithfulness_improved_question_ids"]),
                    "improved",
                ),
            )
        )
        became_worse = " ".join(
            (
                _question_change(
                    "Recall@5",
                    float(finding["recall_at_5_delta_vs_baseline"]),
                    list(finding["recall_degraded_question_ids"]),
                    "degraded",
                ),
                _question_change(
                    "Faithfulness",
                    float(finding["faithfulness_delta_vs_baseline"]),
                    list(finding["faithfulness_degraded_question_ids"]),
                    "degraded",
                ),
            )
        )
        latency_delta = float(finding["average_latency_seconds_delta_vs_baseline"])
        details.append(
            ExperimentDetail(
                version_id=profile.version_id,
                experiment=experiment,
                version=str(version["label"]),
                changed=profile.changed,
                stayed_constant=COMMON_CONSTANTS,
                improved=improved,
                became_worse=became_worse,
                why=(
                    f"{version['mechanism']} This describes the tested mechanism; "
                    "the measured comparison does not claim causation."
                ),
                latency_cost=(
                    f"{latency_delta:+.3f} seconds average versus baseline; "
                    f"{float(metrics['average_latency_seconds']):.3f} seconds absolute."
                ),
                worth_it=str(finding["value_vs_baseline"]),
            )
        )

    recommendation = payload["recommendation"]
    return ExperimentDashboard(
        completed_at=str(payload["completed_at"]),
        rows=tuple(rows),
        details=tuple(details),
        best_recall_at_5=float(payload["best_recall_at_5"]),
        best_faithfulness=float(payload["best_faithfulness"]),
        fastest_average_latency_seconds=float(
            payload["fastest_average_latency_seconds"]
        ),
        recommendation_version_id=str(recommendation["selected_version_id"]),
        recommendation_rationale=str(recommendation["rationale"]),
    )


def load_current_app_benchmark(path: Path) -> CurrentAppBenchmark:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if payload.get("phase") != 10:
        raise ValueError("The current-app benchmark requires a Phase 10 artifact.")
    if payload.get("evaluation_version") != "phase10-current-app-v1":
        raise ValueError("Unsupported current-app evaluation version.")
    metrics = payload.get("metrics")
    regressions = payload.get("ui_regressions")
    deltas = payload.get("delta_vs_historical_baseline")
    if not isinstance(metrics, dict) or not isinstance(regressions, dict):
        raise ValueError("Current-app metrics or UI regressions are missing.")
    if not isinstance(deltas, dict) or not isinstance(regressions.get("cases"), list):
        raise ValueError("Current-app comparison details are missing.")
    return CurrentAppBenchmark(
        completed_at=str(payload["completed_at"]),
        experiment_id=str(payload["experiment_id"]),
        recall_at_5=float(metrics["recall_at_5"]),
        faithfulness=float(metrics["faithfulness"]),
        correct_refusal_rate=float(metrics["correct_refusal_rate"]),
        average_latency_seconds=float(metrics["average_latency_seconds"]),
        p95_latency_seconds=float(metrics["p95_latency_seconds"]),
        recall_delta=float(deltas["recall_at_5"]),
        faithfulness_delta=float(deltas["faithfulness"]),
        average_latency_delta_seconds=float(deltas["average_latency_seconds"]),
        regression_passed_count=int(regressions["passed_count"]),
        regression_case_count=int(regressions["case_count"]),
        regression_pass_rate=float(regressions["pass_rate"]),
        regression_cases=tuple(regressions["cases"]),
    )
