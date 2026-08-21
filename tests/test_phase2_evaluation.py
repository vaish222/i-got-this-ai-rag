from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.evaluation import (  # noqa: E402
    BaselineEvaluator,
    EvaluationDataset,
    expected_source_metrics,
    extract_citations,
    load_evaluation_dataset,
    write_experiment,
)


class FakePipeline:
    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        return [
            (
                Document(
                    page_content="Distractor",
                    metadata={
                        "document_id": "other_001",
                        "document_title": "Other",
                        "chunk_id": "other_001::chunk_000",
                        "source_path": "data/other.md",
                    },
                ),
                0.91,
            ),
            (
                Document(
                    page_content="Expected evidence",
                    metadata={
                        "document_id": "expected_001",
                        "document_title": "Expected",
                        "chunk_id": "expected_001::chunk_000",
                        "source_path": "data/expected.md",
                    },
                ),
                0.82,
            ),
        ]

    def generate(self, question: str, results: list[tuple[Document, float]]) -> str:
        return "Grounded answer [S2]."


class Phase2EvaluationTests(unittest.TestCase):
    def test_repository_dataset_has_all_15_questions(self) -> None:
        dataset = load_evaluation_dataset(PROJECT_ROOT / "evaluation" / "questions.json")

        self.assertEqual(len(dataset.questions), 15)
        self.assertEqual(dataset.questions[0]["question_id"], "Q001")
        self.assertEqual(len(dataset.sha256), 64)

    def test_expected_source_rank_and_recall_at_5_use_unique_source_ids(self) -> None:
        chunks = [
            {"rank": 1, "document_id": "source_a"},
            {"rank": 2, "document_id": "source_a"},
            {"rank": 5, "document_id": "source_b"},
            {"rank": 6, "document_id": "source_c"},
        ]

        ranks, best_rank, recall = expected_source_metrics(
            ["source_a", "source_b", "source_c", "source_missing"], chunks
        )

        self.assertEqual(
            ranks,
            {"source_a": 1, "source_b": 5, "source_c": 6, "source_missing": None},
        )
        self.assertEqual(best_rank, 1)
        self.assertEqual(recall, 0.5)

    def test_citations_resolve_to_retrieved_chunks(self) -> None:
        chunks = [
            {"rank": 1, "document_id": "source_a", "chunk_id": "a::0"},
            {"rank": 2, "document_id": "source_b", "chunk_id": "b::0"},
        ]

        citations = extract_citations("Answer [S2][S1] and again [S2], plus [S9].", chunks)

        self.assertEqual([citation["label"] for citation in citations], ["S2", "S1", "S9"])
        self.assertEqual(citations[0]["document_id"], "source_b")
        self.assertFalse(citations[-1]["resolved"])

    def test_evaluator_records_required_phase_2_fields(self) -> None:
        question = {
            "question_id": "QTEST",
            "question": "What is expected?",
            "expected_answer": "Grounded answer.",
            "expected_source_ids": ["expected_001"],
            "expected_sources": ["Expected"],
            "category": "exact_lookup",
            "answerable": True,
        }

        result = BaselineEvaluator(FakePipeline()).evaluate_question(question)

        self.assertEqual(result["expected_source_rank"], 2)
        self.assertEqual(result["recall_at_5"], 1.0)
        self.assertEqual(result["citation_labels"], ["S2"])
        self.assertEqual(result["citations"][0]["chunk_id"], "expected_001::chunk_000")
        self.assertEqual(len(result["retrieved_chunks"]), 2)
        self.assertEqual(result["llm_latency_seconds"], result["generation_latency_seconds"])
        self.assertGreaterEqual(result["total_latency_seconds"], 0)

    def test_experiment_files_are_tied_by_config_hash(self) -> None:
        config = {"experiment_id": "E001", "top_k": 5}
        results = {"experiment_id": "E001", "summary": {"recall_at_5": 1.0}}
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path, results_path = write_experiment(
                Path(temporary_directory), config, results
            )
            config_bytes = config_path.read_bytes()
            saved_results = json.loads(results_path.read_text(encoding="utf-8"))

        self.assertEqual(saved_results["config_sha256"], hashlib.sha256(config_bytes).hexdigest())


if __name__ == "__main__":
    unittest.main()
