from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.answer_routing import (  # noqa: E402
    ANSWER_ROUTING_SCOPED,
    ANSWER_ROUTING_SCOPED_REQUERY,
    detect_answer_scope,
    filter_results_to_scope,
)
from i_got_this_rag.user_interface import answer_question  # noqa: E402


def document(
    *,
    document_id: str,
    domain: str,
    document_type: str,
    source_path: str,
    text: str,
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "document_id": document_id,
            "document_title": document_id.replace("_", " ").title(),
            "document_type": document_type,
            "domain": domain,
            "source_path": source_path,
            "chunk_id": f"{document_id}::chunk_000",
        },
    )


MEAL_DOCUMENT = document(
    document_id="household_001",
    domain="household",
    document_type="meal_plan",
    source_path="data/sample/household/meal_plan.md",
    text="""| Day | Dinner | Preparation note |
|---|---|---|
| Friday, Aug 21 | Leftover soup and bread | Early dinner before piano |
| Monday, Aug 24 | Chana masala, basmati rice, and cucumber raita | Soak chickpeas Sunday night or use two pantry cans |""",
)
SOCIAL_DOCUMENT = document(
    document_id="social_001",
    domain="social",
    document_type="invitation_tracker",
    source_path="data/sample/social/invitations.md",
    text="Event Sunday, August 23. RSVP by Monday, August 24. Bring lemon bars.",
)


class RoutingPipeline:
    def __init__(
        self,
        *,
        routing_mode: str,
        raw_results: list[tuple[Document, float]],
        scoped_results: list[tuple[Document, float]] | None = None,
        answer: str = "A generated answer [S1].",
    ) -> None:
        self.settings = SimpleNamespace(
            reference_date="2026-08-20",
            answer_routing_mode=routing_mode,
        )
        self.raw_results = raw_results
        self.scoped_results = scoped_results or []
        self.answer = answer
        self.generated_with: list[tuple[Document, float]] = []
        self.scoped_calls: list[object] = []

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        del question
        return self.raw_results

    def retrieve_scoped(
        self,
        question: str,
        scope: object,
    ) -> list[tuple[Document, float]]:
        del question
        self.scoped_calls.append(scope)
        return self.scoped_results

    def generate(
        self,
        question: str,
        results: list[tuple[Document, float]],
    ) -> str:
        del question
        self.generated_with = results
        return self.answer


class AnswerRoutingTests(unittest.TestCase):
    def test_meal_prep_synonym_gets_high_confidence_meal_scope(self) -> None:
        scope = detect_answer_scope("What does meal prep for Monday look like?")

        self.assertEqual(scope.intent, "meal_lookup")
        self.assertEqual(scope.domains, ("household",))
        self.assertEqual(scope.document_types, ("meal_plan",))
        self.assertEqual(
            scope.pinecone_filter(),
            {
                "$and": [
                    {"domain": {"$eq": "household"}},
                    {"document_type": {"$eq": "meal_plan"}},
                ]
            },
        )

    def test_scoped_mode_removes_cross_category_sources(self) -> None:
        scope = detect_answer_scope("What does meal prep for Monday look like?")

        selected = filter_results_to_scope(
            [(SOCIAL_DOCUMENT, 0.98), (MEAL_DOCUMENT, 0.91)],
            scope,
        )

        self.assertEqual([item.metadata["document_id"] for item, _ in selected], ["household_001"])

    def test_meal_prep_uses_exact_coming_monday_row_without_llm(self) -> None:
        pipeline = RoutingPipeline(
            routing_mode=ANSWER_ROUTING_SCOPED,
            raw_results=[(SOCIAL_DOCUMENT, 0.98), (MEAL_DOCUMENT, 0.91)],
        )

        response = answer_question(
            pipeline,
            "What is meal prep for Monday look like?",
        )

        self.assertEqual(pipeline.generated_with, [])
        self.assertIn("Monday, August 24", response.answer)
        self.assertIn("Chana masala, basmati rice, and cucumber raita", response.answer)
        self.assertIn("Soak chickpeas Sunday night or use two pantry cans", response.answer)
        self.assertNotIn("Leftover soup", response.answer)
        self.assertNotIn("RSVP", response.answer)
        self.assertEqual([source.source_path for source in response.sources], ["data/sample/household/meal_plan.md"])

    def test_scoped_requery_preserves_raw_results_but_uses_targeted_evidence(self) -> None:
        pipeline = RoutingPipeline(
            routing_mode=ANSWER_ROUTING_SCOPED_REQUERY,
            raw_results=[(SOCIAL_DOCUMENT, 0.98)],
            scoped_results=[(MEAL_DOCUMENT, 0.91)],
        )

        response = answer_question(
            pipeline,
            "What is meal prep for Monday look like?",
        )

        self.assertEqual(len(pipeline.scoped_calls), 1)
        self.assertIn("Chana masala", response.answer)
        routing = response.generation_trace["answer_routing"]
        self.assertEqual(routing["raw_document_ids"], ["social_001"])
        self.assertEqual(routing["answer_document_ids"], ["household_001"])
        self.assertTrue(routing["scoped_requery_used"])

    def test_activity_scope_excludes_volunteer_evidence_before_generation(self) -> None:
        activity = document(
            document_id="activities_001",
            domain="activities",
            document_type="activity_schedule",
            source_path="data/sample/activities/swimming_schedule.md",
            text="Swimming practice is Monday evening.",
        )
        volunteer = document(
            document_id="volunteer_001",
            domain="volunteer",
            document_type="volunteer_schedule",
            source_path="data/sample/volunteer/mentor_program.md",
            text="Mentor call is Monday evening.",
        )
        pipeline = RoutingPipeline(
            routing_mode=ANSWER_ROUTING_SCOPED,
            raw_results=[(volunteer, 0.99), (activity, 0.90)],
            answer="Swimming practice is Monday evening [S1].",
        )

        response = answer_question(
            pipeline,
            "What kids activities are on Monday?",
        )

        self.assertEqual(len(pipeline.generated_with), 1)
        self.assertEqual(
            pipeline.generated_with[0][0].metadata["document_id"],
            "activities_001",
        )
        self.assertEqual(len(response.sources), 1)
        self.assertNotIn("volunteer", response.sources[0].source_path)

    def test_pending_gifts_use_exact_tracker_rows_without_generation(self) -> None:
        gifts = document(
            document_id="social_002",
            domain="social",
            document_type="gift_tracker",
            source_path="data/sample/social/birthdays_and_gifts.md",
            text="""| Person | Celebration | Gift status | Next action |
|---|---|---|---|
| `friend_child_01` | Party Aug 22 | Needed | Buy and wrap a kit |
| `relative_01` | Birthday Aug 31 | Purchased | Mail the card |
| `friend_03` | Birthday Sep 8 | Idea saved | Confirm the group gift |""",
        )
        pipeline = RoutingPipeline(
            routing_mode=ANSWER_ROUTING_SCOPED,
            raw_results=[(SOCIAL_DOCUMENT, 0.99), (gifts, 0.90)],
        )

        response = answer_question(pipeline, "Which birthdays still need gifts?")

        self.assertEqual(pipeline.generated_with, [])
        self.assertIn("2 birthdays still need gift attention", response.answer)
        self.assertIn("Buy and wrap a kit", response.answer)
        self.assertIn("Confirm the group gift", response.answer)
        self.assertNotIn("Mail the card", response.answer)
        self.assertEqual(len(response.sources), 1)


if __name__ == "__main__":
    unittest.main()
