from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.experiment_dashboard import (  # noqa: E402
    EXPERIMENT_PROFILES,
    load_claim_faithfulness_audit,
    load_current_app_benchmark,
    load_experiment_dashboard,
    load_generation_model_dashboard,
    load_qwen_generation_comparison,
)


def comparison_payload() -> dict:
    versions = []
    findings = []
    for index, profile in enumerate(EXPERIMENT_PROFILES):
        recall = 0.9 if index == 0 else 0.9 - index / 100
        faithfulness = 0.4 if index == 0 else 0.4 + index / 100
        latency = 1.0 + index / 10
        versions.append(
            {
                "version_id": profile.version_id,
                "label": profile.version_id.replace("_", " ").title(),
                "mechanism": f"Measured mechanism for {profile.version_id}.",
                "source_experiment_id": f"E{index + 1:03d}",
                "metrics": {
                    "recall_at_5": recall,
                    "faithfulness": faithfulness,
                    "average_latency_seconds": latency,
                },
            }
        )
        findings.append(
            {
                "version_id": profile.version_id,
                "recall_at_5_delta_vs_baseline": recall - 0.9,
                "faithfulness_delta_vs_baseline": faithfulness - 0.4,
                "average_latency_seconds_delta_vs_baseline": latency - 1.0,
                "recall_improved_question_ids": [],
                "recall_degraded_question_ids": ["Q001"] if index else [],
                "faithfulness_improved_question_ids": ["Q002"] if index else [],
                "faithfulness_degraded_question_ids": [],
                "value_vs_baseline": (
                    "baseline reference" if index == 0 else "not demonstrated"
                ),
            }
        )
    return {
        "phase": 10,
        "completed_at": "2026-08-22T12:00:00+00:00",
        "best_recall_at_5": 0.9,
        "best_faithfulness": 0.47,
        "fastest_average_latency_seconds": 1.0,
        "recommendation": {
            "selected_version_id": "baseline_dense",
            "rationale": "Best measured trade-off.",
        },
        "versions": versions,
        "findings": findings,
    }


class ExperimentDashboardTests(unittest.TestCase):

    def test_qwen_generation_comparison_requires_three_complete_metric_rows(self) -> None:
        metrics = {
            "recall_at_5": 0.9,
            "claim_level_faithfulness": 0.97,
            "answer_relevance_correctness": 0.6,
            "correct_refusal_rate": 1.0,
            "average_claims_per_answer": 4.0,
            "average_output_tokens": 100.0,
            "average_latency_seconds": 4.0,
            "p95_latency_seconds": 8.0,
        }
        payload = {
            "experiment_suite": "qwen_concise_relevance_comparison",
            "versions": [
                {"experiment_id": mode, "metrics": metrics}
                for mode in ("E1", "E2", "E3")
            ],
        }
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            path = Path(directory) / "comparison.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_qwen_generation_comparison(path)

        self.assertEqual(len(loaded["versions"]), 3)
    def write_payload(self, directory: str, payload: dict) -> Path:
        path = Path(directory) / "comparison.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_dashboard_loads_exact_measured_matrix_and_table_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dashboard = load_experiment_dashboard(
                self.write_payload(directory, comparison_payload())
            )

        self.assertEqual(len(dashboard.rows), 8)
        self.assertEqual(dashboard.rows[0].experiment, "E001")
        self.assertEqual(dashboard.rows[0].embedding, "embeddinggemma")
        self.assertEqual(dashboard.rows[2].embedding, "mxbai-embed-large")
        self.assertEqual(dashboard.rows[4].reranking, "BM25 → Top-5")
        self.assertEqual(dashboard.rows[0].table_record()["Recall@5"], 0.9)
        self.assertEqual(dashboard.recommendation_version_id, "baseline_dense")

    def test_experiment_detail_answers_every_prd_tradeoff_question(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dashboard = load_experiment_dashboard(
                self.write_payload(directory, comparison_payload())
            )

        detail = dashboard.detail_for("best_embedding")
        self.assertIn("Replaced embeddinggemma", detail.changed)
        self.assertIn("15 evaluation questions", detail.stayed_constant)
        self.assertIn("Q002", detail.improved)
        self.assertIn("Q001", detail.became_worse)
        self.assertIn("does not claim causation", detail.why)
        self.assertIn("+0.200 seconds", detail.latency_cost)
        self.assertEqual(detail.worth_it, "not demonstrated")

    def test_dashboard_rejects_incomplete_or_wrong_phase_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = comparison_payload()
            payload["versions"].pop()
            with self.assertRaisesRegex(ValueError, "exact eight-version"):
                load_experiment_dashboard(self.write_payload(directory, payload))

            payload = comparison_payload()
            payload["phase"] = 9
            with self.assertRaisesRegex(ValueError, "Phase 10"):
                load_experiment_dashboard(self.write_payload(directory, payload))

    def test_current_app_benchmark_loads_measured_metrics_and_regressions(self) -> None:
        payload = {
            "phase": 10,
            "evaluation_version": "phase10-current-app-v2",
            "experiment_id": "E803_phase10_current_app",
            "completed_at": "2026-08-23T06:00:00+00:00",
            "metrics": {
                "recall_at_5": 0.9,
                "faithfulness": 0.8,
                "correct_refusal_rate": 1.0,
                "average_latency_seconds": 1.2,
                "p95_latency_seconds": 2.5,
            },
            "delta_vs_historical_baseline": {
                "recall_at_5": 0.0,
                "faithfulness": 0.4,
                "average_latency_seconds": -0.25,
            },
            "ui_regressions": {
                "passed_count": 7,
                "case_count": 9,
                "pass_rate": 7 / 9,
                "cases": [
                    {
                        "case_id": "UI001",
                        "question": "What's next?",
                        "passed": True,
                        "failures": [],
                        "latency_seconds": 0.001,
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            benchmark = load_current_app_benchmark(path)

        self.assertEqual(benchmark.experiment_id, "E803_phase10_current_app")
        self.assertEqual(benchmark.faithfulness, 0.8)
        self.assertEqual(benchmark.regression_passed_count, 7)
        self.assertEqual(benchmark.regression_case_count, 9)
        self.assertEqual(benchmark.table_record()["UI regressions"], "7/9")

    def test_generation_model_dashboard_loads_metrics_and_separate_highlights(self) -> None:
        payload = {
            "experiment_suite": "strict_prompt_generation_model_comparison",
            "completed_at": "2026-08-23T12:00:00+00:00",
            "eligible_experiment_ids": ["D1", "D2"],
            "highlights": {
                "highest_faithfulness": ["D2"],
                "highest_relevance_correctness": ["D1"],
                "lowest_average_latency": ["D1"],
                "best_overall_balance": ["D2"],
            },
            "balance": {"method": "A multi-metric harmonic mean."},
            "versions": [
                {
                    "experiment_id": "D1",
                    "label": "Current model",
                    "provider": "ollama",
                    "model": "local-model",
                    "run_status": "complete",
                    "configuration_error": None,
                    "metrics": {
                        "recall_at_5": 0.9,
                        "faithfulness": 0.4,
                        "answer_relevance_correctness": 0.5,
                        "correct_refusal_rate": 1.0,
                        "average_latency_seconds": 1.0,
                        "p95_latency_seconds": 2.0,
                        "generation_failure_count": 0,
                    },
                },
                {
                    "experiment_id": "D2",
                    "label": "Hosted model",
                    "provider": "nebius",
                    "model": "configured-model",
                    "run_status": "complete",
                    "configuration_error": None,
                    "metrics": {
                        "recall_at_5": 0.9,
                        "faithfulness": 0.8,
                        "answer_relevance_correctness": 0.45,
                        "correct_refusal_rate": 1.0,
                        "average_latency_seconds": 1.4,
                        "p95_latency_seconds": 3.0,
                        "generation_failure_count": 0,
                    },
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            dashboard = load_generation_model_dashboard(path)

        self.assertEqual(dashboard.rows[0].table_record()["Relevance"], 0.5)
        self.assertEqual(dashboard.highest_faithfulness_ids, ("D2",))
        self.assertEqual(dashboard.lowest_latency_ids, ("D1",))
        self.assertIn("Hosted model", dashboard.labels_for(("D2",)))

    def test_generation_model_dashboard_rejects_wrong_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            path.write_text(json.dumps({"experiment_suite": "wrong"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                load_generation_model_dashboard(path)

    def test_claim_audit_loader_requires_saved_answer_reuse(self) -> None:
        payload = {
            "audit_type": "claim_level_faithfulness",
            "answers_regenerated": False,
            "model_summary": [],
            "models": [],
            "conclusion": {"code": "C"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_claim_faithfulness_audit(path)
            self.assertEqual(loaded["conclusion"]["code"], "C")

            payload["answers_regenerated"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reuse saved answers"):
                load_claim_faithfulness_audit(path)


if __name__ == "__main__":
    unittest.main()
