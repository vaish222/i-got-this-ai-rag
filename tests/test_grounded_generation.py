from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.grounded_generation import (  # noqa: E402
    REFUSAL_TEXT,
    GroundedAnswerItem,
    GroundedAnswerPayload,
    extract_question_constraints,
    filter_relevant_results,
    generate_strict_grounded_answer,
    narrow_results_to_question_constraints,
    resolve_relative_date_for_retrieval,
)


def document(domain: str, text: str, document_id: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "domain": domain,
            "document_id": document_id,
            "document_title": document_id,
            "chunk_id": f"{document_id}::chunk_000",
            "source_path": f"data/sample/{domain}/{document_id}.md",
        },
    )


class StructuredRunnable:
    def __init__(self, payload: GroundedAnswerPayload) -> None:
        self.payload = payload
        self.prompts: list[Any] = []

    def invoke(self, prompt: Any) -> GroundedAnswerPayload:
        self.prompts.append(prompt)
        return self.payload


class StructuredLLM:
    def __init__(self, payload: GroundedAnswerPayload) -> None:
        self.runnable = StructuredRunnable(payload)
        self.schema: type | None = None
        self.method: str | None = None

    def with_structured_output(
        self,
        schema: type,
        *,
        method: str,
    ) -> StructuredRunnable:
        self.schema = schema
        self.method = method
        return self.runnable


class ExplodingLLM:
    def with_structured_output(self, *_: Any, **__: Any) -> Any:
        raise AssertionError("The LLM must not run when no chunk passes filtering.")


class GroundedGenerationTests(unittest.TestCase):
    def test_extracts_person_date_domain_event_and_fact_mode(self) -> None:
        constraints = extract_question_constraints(
            "Are there kids' activities on Sunday?",
            "2026-08-20",
        )

        self.assertEqual(constraints.people, ("children",))
        self.assertIn("activities", constraints.domains)
        self.assertEqual(constraints.event_task_type, "activity")
        self.assertEqual(constraints.date_start.isoformat(), "2026-08-23")
        self.assertEqual(constraints.date_end, constraints.date_start)
        self.assertEqual(constraints.response_mode, "facts")

    def test_this_week_rolls_forward_from_sunday(self) -> None:
        constraints = extract_question_constraints(
            "What's coming up this week?",
            "2026-08-23",
        )

        self.assertEqual(constraints.date_start.isoformat(), "2026-08-24")
        self.assertEqual(constraints.date_end.isoformat(), "2026-08-30")

    def test_tomorrow_resolves_from_reference_date(self) -> None:
        constraints = extract_question_constraints(
            "What's for dinner tomorrow?",
            "2026-08-23",
        )

        self.assertEqual(constraints.date_start.isoformat(), "2026-08-24")
        self.assertEqual(constraints.date_end, constraints.date_start)
        self.assertEqual(constraints.date_phrase, "tomorrow")

    def test_tomorrow_resolves_consistently_across_domains(self) -> None:
        questions = (
            "What school events are tomorrow?",
            "What kids activities are tomorrow?",
            "What volunteer work is tomorrow?",
            "What course work is due tomorrow?",
            "What social events are tomorrow?",
            "What household tasks are tomorrow?",
            "What is on the family schedule tomorrow?",
        )

        for question in questions:
            with self.subTest(question=question):
                constraints = extract_question_constraints(question, "2026-08-23")
                self.assertEqual(constraints.date_start.isoformat(), "2026-08-24")
                self.assertEqual(constraints.date_end, constraints.date_start)

    def test_relative_date_is_added_only_to_retrieval_query(self) -> None:
        resolved = resolve_relative_date_for_retrieval(
            "What volunteer work is tommorrow?",
            "2026-08-23",
        )

        self.assertIn("Monday, August 24, 2026", resolved)
        self.assertIn("2026-08-24", resolved)
        self.assertEqual(
            resolve_relative_date_for_retrieval(
                "What volunteer work is Monday?",
                "2026-08-23",
            ),
            "What volunteer work is Monday?",
        )

    def test_relevance_filter_keeps_only_direct_kids_sunday_evidence(self) -> None:
        results = [
            (
                document(
                    "activities",
                    "Sunday, August 23: child_02 watercolor class at 10:00 AM.",
                    "activities_001",
                ),
                0.91,
            ),
            (
                document(
                    "volunteer",
                    "Sunday, August 23: adult_01 mentor call at 6:00 PM.",
                    "volunteer_001",
                ),
                0.89,
            ),
            (
                document(
                    "household",
                    "Sunday, August 23: HVAC filter replacement is due.",
                    "household_001",
                ),
                0.85,
            ),
        ]
        constraints = extract_question_constraints(
            "Are there kids' activities on Sunday?",
            "2026-08-20",
        )

        selected, decisions = filter_relevant_results(
            results,
            constraints,
            "2026-08-20",
        )

        self.assertEqual([rank for rank, _, _ in selected], [1])
        self.assertTrue(decisions[0].included)
        self.assertFalse(decisions[1].included)
        self.assertFalse(decisions[2].included)

    def test_exact_day_filter_drops_week_anchor_and_keeps_monday_volunteer_task(self) -> None:
        results = [
            (
                document(
                    "volunteer",
                    (
                        "Beginning the week of Monday, August 24, mentoring totals "
                        "five hours each week.\n\n"
                        "- Tuesday, 6:00 PM — mentor the AI team.\n"
                        "- Wednesday, 6:00 PM — mentor the web team."
                    ),
                    "volunteer_001",
                ),
                0.92,
            ),
            (
                document(
                    "volunteer",
                    (
                        "The 250-word newsletter draft is due Monday, "
                        "August 24 at noon."
                    ),
                    "volunteer_002",
                ),
                0.89,
            ),
        ]
        constraints = extract_question_constraints(
            "What is the volunteer work for Monday?",
            "2026-08-20",
        )

        narrowed = narrow_results_to_question_constraints(
            results,
            constraints,
            "2026-08-20",
        )

        self.assertEqual(
            [item.metadata["document_id"] for item, _ in narrowed],
            ["volunteer_002"],
        )
        self.assertIn("due Monday, August 24 at noon", narrowed[0][0].page_content)
        self.assertNotIn("Wednesday", narrowed[0][0].page_content)

    def test_exact_day_filter_works_for_tables_and_recurring_activity_rows(self) -> None:
        results = [
            (
                document(
                    "household",
                    (
                        "| Day | Dinner | Preparation note |\n"
                        "|---|---|---|\n"
                        "| Monday, Aug 24 | Chana masala | Soak chickpeas Sunday |\n"
                        "| Tuesday, Aug 25 | Hakka noodles | Chop vegetables |"
                    ),
                    "household_001",
                ),
                0.91,
            ),
            (
                document(
                    "activities",
                    (
                        "- Beginning Monday, August 24, child_01 has mathematics "
                        "class every Monday and Wednesday from 4:00–5:00 PM.\n"
                        "- Beginning Tuesday, August 25, child_01 has reading class "
                        "every Tuesday and Thursday from 4:00–5:00 PM."
                    ),
                    "activities_001",
                ),
                0.88,
            ),
        ]

        meal_constraints = extract_question_constraints(
            "What is the meal plan for Monday?",
            "2026-08-20",
        )
        meals = narrow_results_to_question_constraints(
            results,
            meal_constraints,
            "2026-08-20",
        )
        self.assertEqual(len(meals), 1)
        self.assertIn("Chana masala", meals[0][0].page_content)
        self.assertNotIn("Hakka noodles", meals[0][0].page_content)

        activity_constraints = extract_question_constraints(
            "What kids activities are on Wednesday?",
            "2026-08-20",
        )
        activities = narrow_results_to_question_constraints(
            results,
            activity_constraints,
            "2026-08-20",
        )
        self.assertEqual(len(activities), 1)
        self.assertIn("mathematics class", activities[0][0].page_content)
        self.assertNotIn("reading class", activities[0][0].page_content)

    def test_exact_day_filter_keeps_only_matching_invitation_record(self) -> None:
        results = [
            (
                document(
                    "social",
                    (
                        "## Neighborhood potluck\n"
                        "- Event: Sunday, August 23, 5:00 PM\n"
                        "- RSVP: pending\n\n"
                        "## October birthday\n"
                        "- Event: Sunday, October 4, 10:00 AM\n"
                        "- RSVP: pending\n"
                        "- RSVP deadline: Sunday, September 20"
                    ),
                    "social_001",
                ),
                0.93,
            )
        ]
        constraints = extract_question_constraints(
            "Which invitations need an RSVP on Sunday?",
            "2026-08-20",
        )

        narrowed = narrow_results_to_question_constraints(
            results,
            constraints,
            "2026-08-20",
        )

        self.assertEqual(len(narrowed), 1)
        self.assertIn("Neighborhood potluck", narrowed[0][0].page_content)
        self.assertNotIn("October birthday", narrowed[0][0].page_content)

    def test_empty_filtered_context_preserves_exact_refusal_without_llm_call(self) -> None:
        results = [
            (
                document(
                    "school",
                    "Picture day is August 26, 2026.",
                    "school_001",
                ),
                0.8,
            )
        ]

        generated = generate_strict_grounded_answer(
            llm=ExplodingLLM(),
            question="When is next year's graduation?",
            results=results,
            reference_date="2026-08-20",
            timezone="America/Los_Angeles",
            filter_context=True,
        )

        self.assertEqual(generated.answer, REFUSAL_TEXT)
        self.assertEqual(generated.items, ())
        self.assertEqual(generated.context_source_ids, ())

    def test_strict_prompt_mode_preserves_out_of_range_refusal_guard(self) -> None:
        results = [
            (
                document(
                    "social",
                    "The potluck is Sunday, August 23, 2026.",
                    "social_001",
                ),
                0.8,
            )
        ]

        generated = generate_strict_grounded_answer(
            llm=ExplodingLLM(),
            question="What social events are scheduled for next summer?",
            results=results,
            reference_date="2026-08-20",
            timezone="America/Los_Angeles",
            filter_context=False,
        )

        self.assertEqual(generated.answer, REFUSAL_TEXT)
        self.assertEqual(generated.context_source_ids, ())

    def test_structured_generation_preserves_original_source_rank(self) -> None:
        results = [
            (
                document("volunteer", "Saturday mentor call.", "volunteer_001"),
                0.92,
            ),
            (
                document(
                    "activities",
                    "Sunday, August 23: child_02 watercolor class at 10:00 AM.",
                    "activities_001",
                ),
                0.90,
            ),
        ]
        llm = StructuredLLM(
            GroundedAnswerPayload(
                items=[
                    GroundedAnswerItem(
                        title="Watercolor class",
                        date="2026-08-23",
                        time="10:00 AM",
                        category="activities",
                        person="your elementary-school child",
                        source_id="S2",
                        evidence="child_02 watercolor class at 10:00 AM",
                    )
                ],
                optional_suggestions=["Pack extra art supplies"],
            )
        )

        generated = generate_strict_grounded_answer(
            llm=llm,
            question="Are there kids' activities on Sunday?",
            results=results,
            reference_date="2026-08-20",
            timezone="America/Los_Angeles",
            filter_context=True,
        )

        self.assertEqual(generated.context_source_ids, ("S2",))
        self.assertIn("Watercolor class [S2]", generated.answer)
        self.assertNotIn("Pack extra art supplies", generated.answer)
        self.assertEqual(llm.schema, GroundedAnswerPayload)
        self.assertEqual(llm.method, "json_schema")

    def test_advice_is_separated_from_confirmed_items(self) -> None:
        llm = StructuredLLM(
            GroundedAnswerPayload(
                items=[
                    GroundedAnswerItem(
                        title="Course assignment is due",
                        date="2026-08-23",
                        source_id="S1",
                        evidence="assignment is due Sunday, August 23",
                    )
                ],
                optional_suggestions=["Block time before Sunday if that would help."],
            )
        )
        results = [
            (
                document(
                    "learning",
                    "The course assignment is due Sunday, August 23, 2026.",
                    "learning_001",
                ),
                0.9,
            )
        ]

        generated = generate_strict_grounded_answer(
            llm=llm,
            question="Help me plan my course work this week.",
            results=results,
            reference_date="2026-08-20",
            timezone="America/Los_Angeles",
            filter_context=False,
        )

        self.assertIn("Confirmed from your information", generated.answer)
        self.assertIn("Optional suggestions", generated.answer)
        self.assertIn("Block time before Sunday", generated.answer)

    def test_duplicate_structured_items_are_rendered_once(self) -> None:
        duplicate = GroundedAnswerItem(
            title="Bring the serving spatula",
            source_id="S1",
            evidence="Bring the serving spatula.",
        )
        llm = StructuredLLM(
            GroundedAnswerPayload(items=[duplicate, duplicate], optional_suggestions=[])
        )
        results = [
            (
                document(
                    "social",
                    "Bring the serving spatula to the potluck.",
                    "social_001",
                ),
                0.9,
            )
        ]

        generated = generate_strict_grounded_answer(
            llm=llm,
            question="What should we bring to the potluck?",
            results=results,
            reference_date="2026-08-20",
            timezone="America/Los_Angeles",
            filter_context=False,
        )

        self.assertEqual(len(generated.items), 1)
        self.assertEqual(generated.answer.count("serving spatula"), 1)


if __name__ == "__main__":
    unittest.main()
