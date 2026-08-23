from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.baseline import REFUSAL_TEXT  # noqa: E402
from i_got_this_rag.current_app_evaluation import (  # noqa: E402
    _duplicate_bullets,
    evaluate_current_app,
)
from i_got_this_rag.evaluation import EvaluationDataset  # noqa: E402


class FakeCurrentAppPipeline:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(
            reference_date="2026-08-20",
            timezone="America/Los_Angeles",
        )
        self.resources = SimpleNamespace(llm=object())
        self.source = Document(
            page_content="The field trip form is due Friday.",
            metadata={
                "document_id": "school_001",
                "document_title": "School",
                "chunk_id": "school_001::chunk_000",
                "source_path": "data/school.md",
            },
        )

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        return [(self.source, 0.9)]

    def generate(
        self,
        question: str,
        results: list[tuple[Document, float]],
    ) -> str:
        if "graduation" in question.casefold():
            return REFUSAL_TEXT
        return "The field trip form is due Friday [S1]."


class CurrentAppEvaluationTests(unittest.TestCase):
    def test_current_app_scores_same_dataset_end_to_end(self) -> None:
        dataset = EvaluationDataset(
            path=PROJECT_ROOT / "evaluation" / "questions.json",
            schema_version="1.0",
            dataset_name="test",
            reference_date="2026-08-20",
            timezone="America/Los_Angeles",
            questions=(
                {
                    "question_id": "Q001",
                    "question": "When is the field trip form due?",
                    "category": "exact_lookup",
                    "answerable": True,
                    "expected_answer": "Friday",
                    "expected_source_ids": ["school_001"],
                    "expected_sources": ["School"],
                },
                {
                    "question_id": "Q002",
                    "question": "When is next year's graduation?",
                    "category": "unanswerable",
                    "answerable": False,
                    "expected_answer": REFUSAL_TEXT,
                    "expected_source_ids": [],
                    "expected_sources": [],
                },
            ),
            sha256="test-dataset",
        )

        result = evaluate_current_app(FakeCurrentAppPipeline(), dataset)

        self.assertEqual(result["metrics"]["recall_at_5"], 1.0)
        self.assertEqual(result["metrics"]["faithfulness"], 1.0)
        self.assertEqual(result["metrics"]["correct_refusal_rate"], 1.0)
        self.assertEqual(result["metrics"]["retrieval_failure_count"], 0)
        self.assertEqual(len(result["questions"]), 2)

    def test_duplicate_bullet_detection_ignores_citation_labels(self) -> None:
        answer = "- RSVP Friday [S1]\n- RSVP Friday [S2]\n- Potluck Sunday [S1]"

        self.assertEqual(_duplicate_bullets(answer), ("rsvp friday",))


if __name__ == "__main__":
    unittest.main()
