from __future__ import annotations

import sys
import unittest
from pathlib import Path

from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.baseline import REFUSAL_TEXT  # noqa: E402
from i_got_this_rag.user_interface import answer_question, normalize_question  # noqa: E402


class FakePipeline:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.questions: list[str] = []
        self.results = [
            (
                Document(
                    page_content="The field trip form is due Friday.",
                    metadata={
                        "document_id": "school_001",
                        "document_title": "Elementary School Newsletter",
                        "chunk_id": "school_001::chunk_000",
                        "source_path": "data/sample/school/newsletter.md",
                    },
                ),
                0.91,
            ),
            (
                Document(
                    page_content="The course assignment is due Sunday.",
                    metadata={
                        "document_id": "learning_001",
                        "document_title": "Course Schedule",
                        "chunk_id": "learning_001::chunk_000",
                        "source_path": "data/sample/learning/course.md",
                        "page_number": 2,
                    },
                ),
                0.84,
            ),
        ]

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        self.questions.append(question)
        return self.results

    def generate(
        self,
        question: str,
        results: list[tuple[Document, float]],
    ) -> str:
        self.questions.append(question)
        return self.answer


class StreamlitUserInterfaceTests(unittest.TestCase):
    def test_streamlit_app_renders_prd_v1_controls_without_connecting(self) -> None:
        app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=20)

        self.assertEqual(app.exception, [])
        self.assertEqual(app.title[0].value, "I GOT THIS")
        self.assertEqual(len(app.text_input), 1)
        self.assertEqual(
            app.text_input[0].placeholder,
            "What should I prepare for this week?",
        )
        self.assertEqual(len(app.button), 1)
        self.assertEqual(app.button[0].label, "Ask")
        self.assertEqual([tab.label for tab in app.tabs], ["Ask", "Experiments"])

        app.button[0].click().run(timeout=20)
        self.assertEqual(app.warning[0].value, "Enter a question before selecting Ask.")

    def test_empty_question_is_rejected_without_calling_pipeline(self) -> None:
        with self.assertRaisesRegex(ValueError, "Enter a question"):
            normalize_question("  \n  ")

    def test_answer_view_contains_only_cited_sources(self) -> None:
        pipeline = FakePipeline(
            "The field trip form is due Friday [S1]. The assignment is due Sunday [S2]."
        )

        response = answer_question(
            pipeline,
            "  What   should I prepare this week?  ",
        )

        self.assertEqual(response.question, "What should I prepare this week?")
        self.assertEqual(pipeline.questions, [response.question, response.question])
        self.assertEqual(
            [source.title for source in response.sources],
            ["Elementary School Newsletter", "Course Schedule"],
        )
        self.assertIsNone(response.sources[0].page_number)
        self.assertEqual(response.sources[1].page_number, 2)

    def test_supported_uncited_answer_is_attributed_before_display(self) -> None:
        response = answer_question(
            FakePipeline("The field trip form is due Friday."),
            "When is the field trip form due?",
        )

        self.assertEqual(response.answer, "The field trip form is due Friday. [S1]")
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.sources[0].label, "S1")
        self.assertEqual(response.sources[0].title, "Elementary School Newsletter")

    def test_unsupported_uncited_answer_does_not_receive_false_citation(self) -> None:
        response = answer_question(
            FakePipeline("The field trip form is due Monday."),
            "When is the field trip form due?",
        )

        self.assertNotIn("[S", response.answer)
        self.assertEqual(response.sources, ())

    def test_duplicate_and_unresolved_citations_are_not_displayed(self) -> None:
        pipeline = FakePipeline("The form is due Friday [S1][S1][S9].")

        response = answer_question(pipeline, "When is the form due?")

        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.sources[0].label, "S1")

    def test_refusal_has_no_sources(self) -> None:
        response = answer_question(
            FakePipeline(REFUSAL_TEXT),
            "What is not in the knowledge base?",
        )

        self.assertEqual(response.answer, REFUSAL_TEXT)
        self.assertEqual(response.sources, ())


if __name__ == "__main__":
    unittest.main()
