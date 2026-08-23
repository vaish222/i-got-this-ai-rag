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
    load_experiment_dashboard,
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


if __name__ == "__main__":
    unittest.main()
