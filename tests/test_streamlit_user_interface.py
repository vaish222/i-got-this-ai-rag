from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.baseline import (  # noqa: E402
    DEFAULT_ANSWER_STYLE,
    PLAIN_LANGUAGE_ANSWER_STYLE,
    REFUSAL_TEXT,
    generate_grounded_answer,
)
from i_got_this_rag.conversation import (  # noqa: E402
    ConversationQueryRewriter,
    ConversationRewrite,
    ConversationTurn,
)
from i_got_this_rag.user_interface import (  # noqa: E402
    CLARIFICATION_TEXT,
    AnswerView,
    SourceView,
    build_dated_meal_plan_answer,
    WEEKLY_AGENDA_EMPTY_TEXT,
    answer_question,
    build_weekly_agenda_answer,
    build_volunteer_week_answer,
    expand_cited_section_headings,
    filter_answer_to_current_week,
    format_answer_for_display,
    humanize_anonymous_identifiers,
    normalize_question,
    select_relevant_ui_results,
)


class FakePipeline:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.questions: list[str] = []
        self.generation_results: list[list[tuple[Document, float]]] = []
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
        self.generation_results.append(results)
        return self.answer


class FakeRewriter:
    def __init__(self, retrieval_question: str) -> None:
        self.retrieval_question = retrieval_question
        self.calls: list[tuple[str, tuple[ConversationTurn, ...]]] = []

    def rewrite(
        self,
        question: str,
        history: Sequence[ConversationTurn],
    ) -> ConversationRewrite:
        self.calls.append((question, tuple(history)))
        return ConversationRewrite(
            original_question=question,
            retrieval_question=self.retrieval_question,
            used_history=True,
            raw_output=self.retrieval_question,
            guard_repairs=(),
        )


class PromptCapturingLLM:
    def __init__(self, answer: str = "A plain answer [S1].") -> None:
        self.answer = answer
        self.prompt: object | None = None

    def invoke(self, prompt: object) -> SimpleNamespace:
        self.prompt = prompt
        return SimpleNamespace(content=self.answer)


class StreamlitUserInterfaceTests(unittest.TestCase):
    def test_plain_language_style_is_ui_specific_and_preserves_default(self) -> None:
        settings = SimpleNamespace(
            reference_date="2026-08-20",
            timezone="America/Los_Angeles",
        )
        results = [
            (
                Document(
                    page_content="friend_child_01 has a birthday on August 22.",
                    metadata={
                        "document_title": "Birthday Calendar",
                        "chunk_id": "birthday_001::chunk_000",
                        "source_path": "data/sample/family/birthdays.md",
                    },
                ),
                0.9,
            )
        ]
        plain_llm = PromptCapturingLLM()
        default_llm = PromptCapturingLLM()

        generate_grounded_answer(
            settings,
            plain_llm,
            "Which birthdays still need gifts?",
            results,
            answer_style=PLAIN_LANGUAGE_ANSWER_STYLE,
        )
        generate_grounded_answer(
            settings,
            default_llm,
            "Which birthdays still need gifts?",
            results,
            answer_style=DEFAULT_ANSWER_STYLE,
        )

        plain_prompt = plain_llm.prompt.to_messages()[0].content  # type: ignore[union-attr]
        default_prompt = default_llm.prompt.to_messages()[0].content  # type: ignore[union-attr]
        self.assertIn("plain, everyday language", plain_prompt)
        self.assertIn("Never mention provided data", plain_prompt)
        self.assertIn("friend_child_01", plain_prompt)
        self.assertIn("Never present a document title or section heading", plain_prompt)
        self.assertNotIn("plain, everyday language", default_prompt)

    def test_unknown_answer_style_is_rejected_before_model_call(self) -> None:
        llm = PromptCapturingLLM()

        with self.assertRaisesRegex(ValueError, "Unsupported answer style"):
            generate_grounded_answer(
                SimpleNamespace(reference_date="2026-08-20", timezone="UTC"),
                llm,
                "Question",
                [],
                answer_style="unknown",
            )

        self.assertIsNone(llm.prompt)

    def test_display_formatter_removes_data_preamble_and_humanizes_ids(self) -> None:
        answer = (
            "According to the data, the following items need attention:\n\n"
            "- `friend_family_02` dinner [S1]\n"
            "- `child_01` and `child_02` picture days [S2]"
        )

        formatted = format_answer_for_display(answer)

        self.assertTrue(formatted.startswith("The following items need attention:"))
        self.assertIn("your friends dinner [S1]", formatted)
        self.assertIn("your middle-school child", formatted)
        self.assertIn("your elementary-school child", formatted)
        self.assertNotIn("_0", formatted)
        self.assertNotIn("According to the data", formatted)

    def test_display_formatter_humanizes_spaced_and_possessive_ids(self) -> None:
        answer = (
            "Okay, here’s a breakdown based solely on the provided data:\n\n"
            "- Child 01's mathematics diagnostic\n"
            "- Adult 01’s practical AI class\n"
            "- Adult 02 at the welcome table"
        )

        formatted = format_answer_for_display(answer)

        self.assertIn("your middle-school child's mathematics diagnostic", formatted)
        self.assertIn("one adult in your household’s practical AI class", formatted)
        self.assertIn("another adult in your household at the welcome table", formatted)
        self.assertNotIn("Child 01", formatted)
        self.assertNotIn("Adult 01", formatted)
        self.assertNotIn("Adult 02", formatted)
        self.assertNotIn("provided data", formatted)

    def test_unknown_anonymous_id_is_still_rendered_as_a_readable_role(self) -> None:
        formatted = humanize_anonymous_identifiers(
            "Ask `neighbor_group_03` and `coordinator_04`."
        )

        self.assertEqual(formatted, "Ask your neighbors 3 and the coordinator 4.")

    def test_cited_section_heading_expands_to_its_concrete_commitments(self) -> None:
        pipeline = FakePipeline("- **Open commitments** [S4]")
        pipeline.results = [
            (
                Document(
                    page_content="unrelated",
                    metadata={
                        "document_id": f"unrelated_{index}",
                        "document_title": f"Unrelated {index}",
                        "chunk_id": f"unrelated_{index}::chunk_000",
                        "source_path": f"data/unrelated_{index}.md",
                    },
                ),
                0.9 - index / 10,
            )
            for index in range(1, 4)
        ]
        pipeline.results.append(
            (
                Document(
                    page_content="""# Social Commitments and Follow-ups

## Open commitments

- **Neighborhood potluck, Sunday, August 23:** `adult_01` promised to bake lemon bars; the RSVP is pending.
- **Dinner with `friend_family_02`, Saturday, August 29:** reply by Monday, August 24.
- **Coffee with `colleague_01`:** suggest two September dates by August 28.

## Completed follow-ups

- Confirmed birthday-party attendance.
""",
                    metadata={
                        "document_id": "social_003",
                        "document_title": "Social Commitments and Follow-ups",
                        "chunk_id": "social_003::chunk_000",
                        "source_path": "data/sample/social/social_commitments.md",
                    },
                ),
                0.55,
            )
        )

        response = answer_question(pipeline, "What do I still need to do?")

        self.assertNotIn("- **Open commitments**", response.answer)
        self.assertIn("Neighborhood potluck", response.answer)
        self.assertIn("Dinner with your friends", response.answer)
        self.assertIn("Coffee with your colleague", response.answer)
        self.assertNotIn("friend_family_02", response.answer)
        self.assertEqual(response.answer.count("[S4]"), 3)
        self.assertEqual(
            [source.title for source in response.sources],
            ["Social Commitments and Follow-ups"],
        )

    def test_uncited_or_unknown_heading_is_not_expanded(self) -> None:
        results = [
            (
                Document(
                    page_content="## Open commitments\n\n- Send the RSVP.",
                    metadata={},
                ),
                0.8,
            )
        ]

        self.assertEqual(
            expand_cited_section_headings("- Open commitments", results),
            "- Open commitments",
        )
        self.assertEqual(
            expand_cited_section_headings("- Unknown section [S1]", results),
            "- Unknown section [S1]",
        )

    def test_pending_rsvp_answer_excludes_unrelated_action_items(self) -> None:
        pipeline = FakePipeline(
            "According to the data, `friend_family_02`, `child_01`, and "
            "`child_02` need responses [S1][S2]."
        )
        pipeline.results = [
            (
                Document(
                    page_content="""# August Invitations and RSVP Tracker

## Neighborhood potluck

- Event: Sunday, August 23, **5:00–7:00 PM**
- RSVP: **pending**
- RSVP deadline: **Friday, August 21 at noon**

## `friend_family_02` dinner

- Event: Saturday, August 29, 6:00 PM
- RSVP: **pending**
- RSVP deadline: Monday, August 24
- Note: ask whether children are included before accepting

## `friend_child_01` birthday party

- Event: Saturday, August 22, 3:00–5:00 PM
- RSVP: **completed**
""",
                    metadata={
                        "document_id": "social_001",
                        "document_title": "August Invitations and RSVP Tracker",
                        "chunk_id": "social_001::chunk_000",
                        "source_path": "data/sample/social/invitations.md",
                    },
                ),
                0.95,
            ),
            (
                Document(
                    page_content="""# School Events

## Picture days

- `child_02` elementary picture day: outfit not selected
- `child_01` middle school picture day: no action yet
""",
                    metadata={
                        "document_id": "school_004",
                        "document_title": "School Events and Forms Tracker",
                        "chunk_id": "school_004::chunk_000",
                        "source_path": "data/sample/school/school_events.md",
                    },
                ),
                0.82,
            ),
        ]

        response = answer_question(
            pipeline,
            "Which invitations still need an RSVP?",
        )

        self.assertIn("2 invitations still need a response", response.answer)
        self.assertIn("Neighborhood potluck", response.answer)
        self.assertIn("Dinner with your friends", response.answer)
        self.assertIn("Friday, August 21 at noon", response.answer)
        self.assertIn("Monday, August 24", response.answer)
        self.assertNotIn("picture day", response.answer.casefold())
        self.assertNotIn("friend_family_02", response.answer)
        self.assertEqual(pipeline.questions, [response.question])
        self.assertEqual(
            [source.title for source in response.sources],
            ["August Invitations and RSVP Tracker"],
        )

    def test_pending_rsvp_question_refuses_without_explicit_pending_evidence(self) -> None:
        pipeline = FakePipeline("The picture days need responses [S1].")
        pipeline.results = [
            (
                Document(
                    page_content="Picture day outfit is not selected.",
                    metadata={
                        "document_id": "school_004",
                        "document_title": "School Events",
                        "chunk_id": "school_004::chunk_000",
                        "source_path": "data/sample/school/school_events.md",
                    },
                ),
                0.8,
            )
        ]

        response = answer_question(
            pipeline,
            "Which invitations still require an RSVP?",
        )

        self.assertEqual(response.answer, REFUSAL_TEXT)
        self.assertEqual(response.sources, ())
        self.assertEqual(pipeline.questions, [response.question])

    def test_completed_rsvp_question_keeps_the_normal_generation_path(self) -> None:
        pipeline = FakePipeline("The birthday-party RSVP is completed [S1].")

        response = answer_question(pipeline, "Which RSVPs are completed?")

        self.assertEqual(response.answer, "The birthday-party RSVP is completed [S1].")
        self.assertEqual(pipeline.questions, [response.question, response.question])

    def test_streamlit_theme_forces_readable_beans_colors(self) -> None:
        with (PROJECT_ROOT / ".streamlit" / "config.toml").open("rb") as config_file:
            theme = tomllib.load(config_file)["theme"]

        self.assertEqual(theme["base"], "light")
        self.assertEqual(theme["primaryColor"], "#1C2042")
        self.assertEqual(theme["backgroundColor"], "#FFEA8A")
        self.assertEqual(theme["secondaryBackgroundColor"], "#CBDBF2")
        self.assertEqual(theme["textColor"], "#1C2042")

    def test_streamlit_app_renders_personalized_chat_without_connecting(self) -> None:
        app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=20)

        self.assertEqual(app.exception, [])
        style = app.markdown[0].value.lower()
        for color in (
            "#1c2042",
            "#ffed8e",
            "#ffea8a",
            "#35aec0",
            "#efd6db",
            "#cbdbf2",
            "#badbe5",
            "#dc3f40",
            "#ff7618",
            "#dceeff",
            "#ebe3ff",
            "#ffe2d2",
            "#d5f2ec",
            "#dff2d8",
            "#ffdde2",
            "#ffe8c9",
            "#fcfbfa",
        ):
            self.assertIn(color, style)
        self.assertIn("::selection", style)
        self.assertIn("::-moz-selection", style)
        self.assertGreaterEqual(
            style.count("background: var(--igt-blue-strong)"),
            2,
        )
        self.assertIn("background-attachment: fixed", style)
        self.assertIn("var(--igt-hero-yellow) 0%", style)
        self.assertIn("background: var(--igt-hero-blue)", style)
        self.assertIn('[class*="st-key-suggestion_"]', style)
        self.assertIn('[class*="st-key-new_conversation"]', style)
        self.assertIn(
            '[aria-selected="true"][data-baseweb="tab"]',
            style,
        )
        self.assertIn('[data-testid="sttab"][data-selected]', style)
        self.assertIn(".react-aria-selectionindicator", style)
        self.assertIn("background: var(--igt-red)", style)
        self.assertIn("color: var(--igt-orange) !important", style)
        self.assertIn("background-color: var(--igt-orange) !important", style)
        self.assertIn('[class*="st-key-answer_category_school_"]', style)
        self.assertIn('[class*="st-key-answer_category_kids_"]', style)
        self.assertIn('[class*="st-key-answer_category_household_"]', style)
        self.assertIn('[class*="st-key-answer_category_learning_"]', style)
        self.assertIn('[class*="st-key-answer_category_volunteer_"]', style)
        self.assertIn('[class*="st-key-answer_category_social_"]', style)
        self.assertIn('[class*="st-key-answer_category_family_"]', style)
        self.assertIn('[class*="st-key-answer_daily_card_"]', style)
        self.assertIn('[class*="st-key-answer_event_row_"]', style)
        self.assertIn("border-left: 6px solid var(--igt-orange)", style)
        self.assertIn("font-size: 1.18rem", style)
        self.assertIn(".igt-category-chip", style)
        self.assertIn(".igt-action-chip", style)
        self.assertIn(".igt-time-chip", style)
        self.assertIn('[data-baseweb="tab"] *', style)
        self.assertIn('[data-testid="stchatinput"] textarea', style)
        self.assertTrue(
            any(
                "<h1>✨ I GOT THIS. What’s next?</h1>" in item.value
                and "<h4>24 hours." in item.value
                for item in app.markdown
            )
        )
        self.assertEqual(len(app.chat_message), 1)
        self.assertIn(
            "I can help connect schedules",
            app.chat_message[0].markdown[0].value,
        )
        self.assertEqual(len(app.chat_input), 1)
        self.assertEqual(
            app.chat_input[0].placeholder,
            "Ask what's next, what to prepare, or what you might be forgetting…",
        )
        self.assertEqual(
            {button.label for button in app.button},
            {
                "↻ New conversation",
                "⏭️ What's coming up this week?",
                "💌 Which invitations still need an RSVP?",
                "🎒 What should I prepare for this weekend?",
                "📅 Plan my week.",
            },
        )
        self.assertEqual([tab.label for tab in app.tabs], ["Ask", "Experiments"])

    def test_suggested_questions_remain_available_after_an_answer(self) -> None:
        previous_response = AnswerView(
            question="Which invitations still need an RSVP?",
            retrieval_question="Which invitations still need an RSVP?",
            answer="The neighborhood potluck still needs a response [S1].",
            sources=(),
            used_conversation_context=False,
        )
        app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=20)
        app.session_state["conversation"] = [previous_response]

        app.run(timeout=20)

        self.assertEqual(app.exception, [])
        self.assertEqual(
            {button.label for button in app.button},
            {
                "↻ New conversation",
                "⏭️ What's coming up this week?",
                "💌 Which invitations still need an RSVP?",
                "🎒 What should I prepare for this weekend?",
                "📅 Plan my week.",
            },
        )
        self.assertEqual(len(app.chat_message), 2)
        self.assertEqual(
            app.chat_message[0].markdown[0].value,
            previous_response.question,
        )
        rendered_answer = [item.value for item in app.chat_message[1].markdown]
        self.assertIn("**🎉 Social**", rendered_answer)
        self.assertIn(
            "The neighborhood potluck still needs a response.",
            rendered_answer,
        )
        self.assertNotIn("[S1]", "\n".join(rendered_answer))

    def test_answer_items_render_as_pastel_category_cards(self) -> None:
        response = AnswerView(
            question="What is coming up?",
            retrieval_question="What is coming up?",
            answer=(
                "Here is what needs attention:\n\n"
                "- The school field trip form is due Friday [S1].\n"
                "- The robotics workshop starts Saturday [S2].\n"
                "- Schedule the home repair [S3].\n"
                "- Submit the course assignment [S4].\n"
                "- Prepare for the mentor session [S5].\n"
                "- Send the dinner RSVP [S6].\n"
                "- Buy the family birthday gift [S7]."
            ),
            sources=tuple(
                SourceView(
                    label=f"S{index}",
                    title=category.title(),
                    source_path=f"data/sample/{category}/example.md",
                    page_number=None,
                )
                for index, category in enumerate(
                    (
                        "school",
                        "activities",
                        "household",
                        "learning",
                        "volunteer",
                        "social",
                        "family",
                    ),
                    start=1,
                )
            ),
            used_conversation_context=False,
        )
        app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=20)
        app.session_state["conversation"] = [response]

        app.run(timeout=20)

        self.assertEqual(app.exception, [])
        rendered = [item.value for item in app.chat_message[1].markdown]
        for label in (
            "**🏫 School**",
            "**👧 Kids activities**",
            "**🏠 Household**",
            "**📚 Learning**",
            "**🤝 Volunteer**",
            "**🎉 Social**",
            "**👨‍👩‍👧 Family**",
        ):
            self.assertIn(label, rendered)
        self.assertIn("Here is what needs attention:", rendered)
        self.assertIn("- The school field trip form is due Friday.", rendered)
        self.assertIn("- Buy the family birthday gift.", rendered)
        self.assertNotIn("[S", "\n".join(rendered))
        for index, category in enumerate(
            (
                "School",
                "Activities",
                "Household",
                "Learning",
                "Volunteer",
                "Social",
                "Family",
            ),
            start=1,
        ):
            self.assertIn(f"**Source {index}: {category}**", rendered)

    def test_dated_agenda_renders_one_day_card_with_compact_event_chips(
        self,
    ) -> None:
        response = AnswerView(
            question="Plan my week",
            retrieval_question="Plan my week",
            answer=(
                "Here’s what’s coming up this week:\n\n"
                "**Friday, August 21**\n"
                "- 8:30–9:15 AM — mathematics diagnostic for your "
                "middle-school child [S1]\n"
                "- Noon — neighborhood potluck RSVP deadline [S1]\n"
                "- 4:30–5:00 PM — piano lesson for your elementary-school "
                "child [S1]\n"
                "- 5:00 PM — emergency contact verification deadline for your "
                "middle-school child [S1]\n"
                "- 6:00 PM — mentor question-list comments due [S1]"
            ),
            sources=(
                SourceView(
                    label="S1",
                    title="Family Schedule",
                    source_path="data/sample/family/family_schedule.md",
                    page_number=None,
                ),
            ),
            used_conversation_context=False,
        )
        app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=20)
        app.session_state["conversation"] = [response]

        app.run(timeout=20)

        self.assertEqual(app.exception, [])
        rendered = [item.value for item in app.chat_message[1].markdown]
        joined = "\n".join(rendered)
        self.assertEqual(rendered.count("### Friday, August 21"), 1)
        self.assertEqual(joined.count("igt-chip-school"), 2)
        self.assertEqual(joined.count("igt-chip-social"), 1)
        self.assertEqual(joined.count("igt-chip-kids"), 1)
        self.assertEqual(joined.count("igt-chip-volunteer"), 1)
        self.assertIn("ACTION NEEDED", joined)
        self.assertIn("DEADLINE", joined)
        self.assertIn("igt-time-chip", joined)
        self.assertIn("8:30–9:15 AM", joined)
        self.assertIn("4:30–5:00 PM", joined)
        self.assertNotIn("**🏫 School**", rendered)

    def test_cited_household_source_wins_over_incidental_activity_word(
        self,
    ) -> None:
        response = AnswerView(
            question="What is the meal plan for Sunday?",
            retrieval_question="What is the meal plan for Sunday?",
            answer=(
                "**Sunday, August 23**\n"
                "- **Dinner:** Sheet-pan chicken and vegetables [S1]\n"
                "- **Preparation:** Prepare vegetables during robotics class [S1]"
            ),
            sources=(
                SourceView(
                    label="S1",
                    title="Family Meal Plan",
                    source_path="data/sample/household/meal_plan.md",
                    page_number=None,
                ),
            ),
            used_conversation_context=False,
        )
        app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=20)
        app.session_state["conversation"] = [response]

        app.run(timeout=20)

        self.assertEqual(app.exception, [])
        rendered = "\n".join(
            item.value for item in app.chat_message[1].markdown
        )
        self.assertEqual(rendered.count("igt-chip-household"), 2)
        self.assertNotIn("igt-chip-kids", rendered)

    def test_uncited_section_bullets_inherit_household_card_context(self) -> None:
        response = AnswerView(
            question="What should I prepare for this weekend?",
            retrieval_question="What should I prepare for this weekend?",
            answer=(
                "Here is what needs attention:\n\n"
                "**Saturday - August 22nd:** [S1]\n"
                "- **HVAC Service:** Check the system before the visit.\n"
                "- **Library Returns:** Put the due books in the car.\n"
                "- **Meal Prep:** Bake the lemon bars.\n\n"
                "**Sunday - August 23rd:** [S2]\n"
                "- **Family Potluck:** Pack the lemon bars and serving spatula."
            ),
            sources=(
                SourceView(
                    label="S1",
                    title="Home Tasks and Maintenance",
                    source_path="data/sample/household/home_tasks.md",
                    page_number=None,
                ),
                SourceView(
                    label="S2",
                    title="Children's Weekend Classes",
                    source_path="data/sample/activities/weekend_classes.md",
                    page_number=None,
                ),
            ),
            used_conversation_context=False,
        )
        app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=20)
        app.session_state["conversation"] = [response]

        app.run(timeout=20)

        self.assertEqual(app.exception, [])
        rendered = [item.value for item in app.chat_message[1].markdown]
        household_blocks = [
            value
            for value in rendered
            if "HVAC Service" in value or "Library Returns" in value
        ]
        self.assertEqual(len(household_blocks), 2)
        self.assertIn("HVAC Service", "\n".join(household_blocks))
        self.assertIn("Library Returns", "\n".join(household_blocks))
        self.assertIn("Meal Prep", "\n".join(rendered))
        self.assertTrue(
            any("igt-chip-household" in value and "🏠 Household" in value for value in rendered)
        )
        self.assertTrue(
            any("igt-chip-social" in value and "🎉 Social" in value for value in rendered)
        )
        self.assertIn("### Saturday - August 22nd", rendered)
        self.assertIn("### Sunday - August 23rd", rendered)
        self.assertNotIn("[S", "\n".join(rendered))
        self.assertIn(
            "**Source 1: Home Tasks and Maintenance**",
            rendered,
        )
        self.assertIn(
            "**Source 2: Children's Weekend Classes**",
            rendered,
        )

    def test_clarification_response_does_not_show_missing_source_notice(self) -> None:
        clarification = AnswerView(
            question="what's next?",
            retrieval_question="what's next?",
            answer=CLARIFICATION_TEXT,
            sources=(),
            used_conversation_context=False,
        )
        app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=20)
        app.session_state["conversation"] = [clarification]

        app.run(timeout=20)

        self.assertEqual(app.exception, [])
        self.assertIn(
            CLARIFICATION_TEXT,
            [item.value for item in app.chat_message[1].markdown],
        )
        self.assertNotIn(
            "No source could be safely attributed to this response.",
            [item.value for item in app.caption],
        )

    def test_empty_question_is_rejected_without_calling_pipeline(self) -> None:
        with self.assertRaisesRegex(ValueError, "Enter a question"):
            normalize_question("  \n  ")

    def test_underspecified_question_asks_for_clarification_without_retrieval(
        self,
    ) -> None:
        pipeline = FakePipeline(REFUSAL_TEXT)

        response = answer_question(pipeline, "what's next?")

        self.assertEqual(response.answer, CLARIFICATION_TEXT)
        self.assertIn("more specific", response.answer)
        self.assertIn("schedule", response.answer)
        self.assertEqual(response.sources, ())
        self.assertEqual(pipeline.questions, [])

    def test_specific_next_question_still_uses_retrieval(self) -> None:
        pipeline = FakePipeline("The field trip form is due Friday [S1].")

        response = answer_question(pipeline, "What's next for the field trip?")

        self.assertEqual(response.answer, "The field trip form is due Friday [S1].")
        self.assertEqual(pipeline.questions, [response.question, response.question])

    def test_answer_view_contains_only_cited_sources(self) -> None:
        pipeline = FakePipeline(
            "The field trip form is due Friday [S1]. The assignment is due Sunday [S2]."
        )

        response = answer_question(
            pipeline,
            "  What   should I prepare this week?  ",
        )

        self.assertEqual(response.question, "What should I prepare this week?")
        self.assertEqual(response.retrieval_question, response.question)
        self.assertFalse(response.used_conversation_context)
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

    def test_unsupported_uncited_answer_becomes_safe_refusal(self) -> None:
        response = answer_question(
            FakePipeline("The field trip form is due Monday."),
            "When is the field trip form due?",
        )

        self.assertEqual(response.answer, REFUSAL_TEXT)
        self.assertEqual(response.sources, ())

    def test_conversational_filler_becomes_safe_refusal(self) -> None:
        response = answer_question(
            FakePipeline("Okay, I understand. Let's proceed with that information."),
            "Is there any other volunteer work?",
        )

        self.assertEqual(response.answer, REFUSAL_TEXT)
        self.assertEqual(response.sources, ())

    def test_volunteer_week_scope_filters_domain_and_out_of_week_blocks(self) -> None:
        results = [
            (
                Document(
                    page_content=(
                        "Donate the clothing box during the September 5 collection."
                    ),
                    metadata={"document_id": "household_002", "domain": "household"},
                ),
                0.95,
            ),
            (
                Document(
                    page_content=(
                        "Comments are due Friday, August 21 at 6:00 PM.\n\n"
                        "The next group meeting is September 9 at 6:30 PM."
                    ),
                    metadata={"document_id": "volunteer_001", "domain": "volunteer"},
                ),
                0.85,
            ),
            (
                Document(
                    page_content=(
                        "The newsletter draft is due Monday, August 24 at noon.\n\n"
                        "Cover Sunday's neighborhood potluck welcome table."
                    ),
                    metadata={"document_id": "volunteer_002", "domain": "volunteer"},
                ),
                0.8,
            ),
        ]

        selected = select_relevant_ui_results(
            "What volunteer work is due this week?",
            results,
            "2026-08-20",
        )

        self.assertEqual(
            [document.metadata["document_id"] for document, _ in selected],
            ["volunteer_001", "volunteer_002"],
        )
        self.assertIn("August 21", selected[0][0].page_content)
        self.assertNotIn("September 9", selected[0][0].page_content)
        self.assertNotIn("August 24", selected[1][0].page_content)
        self.assertIn("Sunday's neighborhood potluck", selected[1][0].page_content)

    def test_weekly_agenda_is_deduplicated_and_uses_calendar_dates(self) -> None:
        results = [
            (
                Document(
                    page_content="""# Family Schedule

## Friday, August 21

- 8:30–9:15 AM — `child_01`: mathematics diagnostic
- Noon — neighborhood potluck RSVP deadline

## Saturday, August 22

- 3:00–5:00 PM — `child_02`: `friend_child_01` birthday party""",
                    metadata={
                        "document_id": "family_001",
                        "document_type": "family_calendar",
                        "document_title": "Family Schedule",
                        "chunk_id": "family_001::chunk_000",
                        "source_path": "data/sample/family/family_schedule.md",
                    },
                ),
                0.95,
            ),
            (
                Document(
                    page_content="""## Friday, August 21

- Noon — neighborhood potluck RSVP deadline

## Sunday, August 23

- 5:00–7:00 PM — family: neighborhood potluck; `adult_02` at welcome table at 4:50 PM""",
                    metadata={
                        "document_id": "family_001",
                        "document_type": "family_calendar",
                        "document_title": "Family Schedule",
                        "chunk_id": "family_001::chunk_001",
                        "source_path": "data/sample/family/family_schedule.md",
                    },
                ),
                0.9,
            ),
        ]

        answer = build_weekly_agenda_answer(results, "2026-08-20")

        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertEqual(answer.count("**Friday, August 21**"), 1)
        self.assertEqual(answer.count("potluck RSVP deadline"), 1)
        self.assertIn("Noon — neighborhood potluck RSVP deadline [S1]", answer)
        self.assertIn("**Sunday, August 23**", answer)
        self.assertIn("neighborhood potluck; adult_02 at welcome table", answer)

    def test_weekly_agenda_bypasses_free_form_generation(self) -> None:
        pipeline = FakePipeline(
            "This Week:\n- Child 01's diagnostic.\n\n"
            "This Week:\n- Adult 01’s class. [S1]"
        )
        pipeline.results = [
            (
                Document(
                    page_content="""# Family Schedule

## Thursday, August 20

- 7:00–8:15 PM — `adult_01`: Practical AI live class from home

## Friday, August 21

- Noon — neighborhood potluck RSVP deadline

## Saturday, August 22

- 3:00–5:00 PM — `child_02`: `friend_child_01` birthday party""",
                    metadata={
                        "document_id": "family_001",
                        "document_type": "family_calendar",
                        "document_title": "Family Schedule",
                        "chunk_id": "family_001::chunk_000",
                        "source_path": "data/sample/family/family_schedule.md",
                    },
                ),
                0.95,
            )
        ]

        response = answer_question(
            pipeline,
            "What's coming up this week?",
            reference_date="2026-08-20",
        )

        self.assertEqual(pipeline.questions, [response.question])
        self.assertEqual(response.answer.count("Here’s what’s coming up this week:"), 1)
        self.assertIn("one adult in your household", response.answer)
        self.assertNotIn("Adult 01", response.answer)
        self.assertIn("Friday, August 21", response.answer)
        self.assertIn(
            "your friend's child's birthday party for your elementary-school child",
            response.answer,
        )
        self.assertNotIn("friend_child_01", response.answer)
        self.assertEqual(len(response.sources), 1)

    def test_plan_my_week_uses_deterministic_weekly_agenda(self) -> None:
        pipeline = FakePipeline(
            "Monday, August 24 — invented school event. [S1]"
        )
        pipeline.results = [
            (
                Document(
                    page_content="""# Family Schedule

## Friday, August 21

- 4:30–5:00 PM — `child_02`: piano lesson

## Saturday, August 22

- 9:00–10:15 AM — `child_01`: swim practice; arrive 8:45 AM
- 1:00–2:30 PM — `child_01`: solar-rover workshop

## Monday, August 24

- 8:00 AM — event outside the requested week""",
                    metadata={
                        "document_id": "family_001",
                        "document_type": "family_calendar",
                        "document_title": "Family Schedule",
                        "chunk_id": "family_001::chunk_000",
                        "source_path": "data/sample/family/family_schedule.md",
                    },
                ),
                0.95,
            ),
            (
                Document(
                    page_content="""## Saturday, August 22

- 9:00–10:15 AM — `child_01`: swim practice; arrive 8:45 AM""",
                    metadata={
                        "document_id": "family_001",
                        "document_type": "family_calendar",
                        "document_title": "Family Schedule",
                        "chunk_id": "family_001::chunk_001",
                        "source_path": "data/sample/family/family_schedule.md",
                    },
                ),
                0.9,
            ),
        ]

        response = answer_question(
            pipeline,
            "Plan my week",
            reference_date="2026-08-20",
        )

        self.assertEqual(pipeline.questions, [response.question])
        self.assertEqual(pipeline.generation_results, [])
        self.assertEqual(response.answer.count("swim practice"), 1)
        self.assertIn("4:30–5:00 PM", response.answer)
        self.assertIn("9:00–10:15 AM", response.answer)
        self.assertIn("1:00–2:30 PM", response.answer)
        self.assertIn("your elementary-school child", response.answer)
        self.assertIn("your middle-school child", response.answer)
        self.assertNotIn("Monday, August 24", response.answer)
        self.assertNotIn("invented school event", response.answer)

    def test_weekly_plan_without_calendar_items_does_not_generate(self) -> None:
        pipeline = FakePipeline("Invented weekly plan. [S1]")

        response = answer_question(
            pipeline,
            "Help me organize the week",
            reference_date="2026-08-20",
        )

        self.assertEqual(pipeline.questions, [response.question])
        self.assertEqual(pipeline.generation_results, [])
        self.assertEqual(response.answer, WEEKLY_AGENDA_EMPTY_TEXT)
        self.assertEqual(response.sources, ())

    def test_sunday_meal_plan_uses_exact_table_row_without_generation(
        self,
    ) -> None:
        pipeline = FakePipeline(
            "Sunday, Aug 23 | Sheet-pan chicken | Prepare during robotics [S1]"
        )
        pipeline.results = [
            (
                Document(
                    page_content="""# Family Meal Plan

| Day | Dinner | Preparation note |
|---|---|---|
| Saturday, Aug 22 | Sheet-pan chicken | Prep vegetables during robotics class |
| Sunday, Aug 23 | Neighborhood potluck | Family is bringing lemon bars |""",
                    metadata={
                        "document_id": "household_001",
                        "document_type": "meal_plan",
                        "document_title": "Family Meal Plan",
                        "chunk_id": "household_001::chunk_000",
                        "source_path": "data/sample/household/meal_plan.md",
                    },
                ),
                0.97,
            ),
        ]

        response = answer_question(
            pipeline,
            "What is the meal plan for Sunday?",
            reference_date="2026-08-23",
        )

        self.assertEqual(pipeline.questions, [response.question])
        self.assertEqual(pipeline.generation_results, [])
        self.assertIn("Sunday, August 23", response.answer)
        self.assertIn("Neighborhood potluck", response.answer)
        self.assertIn("Family is bringing lemon bars", response.answer)
        self.assertNotIn("Sheet-pan chicken", response.answer)
        self.assertNotIn("robotics", response.answer)
        self.assertEqual(len(response.sources), 1)

    def test_dated_meal_plan_builder_returns_none_without_requested_row(
        self,
    ) -> None:
        results = [
            (
                Document(
                    page_content=(
                        "| Saturday, Aug 22 | Sheet-pan chicken | Prep vegetables |"
                    ),
                    metadata={
                        "document_type": "meal_plan",
                        "source_path": "data/sample/household/meal_plan.md",
                    },
                ),
                0.9,
            )
        ]

        answer = build_dated_meal_plan_answer(
            results,
            "What is the meal plan for Sunday?",
            "2026-08-23",
        )

        self.assertIsNone(answer)

    def test_out_of_week_answer_items_are_removed(self) -> None:
        answer = (
            "Volunteer work due this week:\n\n"
            "- Comments are due Friday, August 21 at 6:00 PM. [S1]\n"
            "- Donate clothing during the September 5 collection. [S2]"
        )

        filtered = filter_answer_to_current_week(
            answer,
            "What volunteer work is due this week?",
            "2026-08-20",
        )

        self.assertIn("August 21", filtered)
        self.assertNotIn("September 5", filtered)

    def test_volunteer_week_answer_is_deterministic_and_tracks_other_items(self) -> None:
        results = [
            (
                Document(
                    page_content="""The next mentoring session is **Saturday, August 22, from 11:00 AM–12:00 PM** by video call.

- Add comments to the question list by **Friday, August 21 at 6:00 PM**.
- Bring two examples of behavioral questions to Saturday's call.
- Enter a short session note by **Sunday, August 23 at 6:00 PM**.

The next group meeting is September 9 at 6:30 PM.""",
                    metadata={"document_id": "volunteer_001", "domain": "volunteer"},
                ),
                0.9,
            ),
            (
                Document(
                    page_content=(
                        "At Sunday's neighborhood potluck, `adult_02` will cover "
                        "the welcome table from **4:50–5:00 PM**."
                    ),
                    metadata={"document_id": "volunteer_002", "domain": "volunteer"},
                ),
                0.8,
            ),
        ]
        selected = select_relevant_ui_results(
            "What volunteer work is due this week?",
            results,
            "2026-08-20",
        )

        first = build_volunteer_week_answer(
            "What volunteer work is due this week?",
            selected,
            (),
            "2026-08-20",
        )
        second = build_volunteer_week_answer(
            "Is there any other volunteer work?",
            selected,
            (
                ConversationTurn(
                    "What volunteer work is due this week?",
                    format_answer_for_display(first),
                ),
            ),
            "2026-08-20",
        )

        self.assertIn("5 volunteer commitments", first)
        self.assertIn("August 22", first)
        self.assertIn("August 21", first)
        self.assertIn("August 23", first)
        self.assertIn("welcome table", first)
        self.assertNotIn("September 9", first)
        self.assertEqual(first.count("[S1]"), 4)
        self.assertEqual(first.count("[S2]"), 1)
        self.assertIn("any additional volunteer work", second)
        self.assertIn("[S1][S2]", second)

    def test_additive_volunteer_answer_lists_items_missing_from_previous_answer(self) -> None:
        results = [
            (
                Document(
                    page_content=(
                        "Add comments by Friday, August 21 at 6:00 PM.\n"
                        "Bring examples to Saturday's mentoring session."
                    ),
                    metadata={"document_id": "volunteer_001", "domain": "volunteer"},
                ),
                0.9,
            )
        ]

        answer = build_volunteer_week_answer(
            "Is there any other volunteer work?",
            results,
            (
                ConversationTurn(
                    "What volunteer work is due this week?",
                    "The September 5 clothing collection was listed.",
                ),
            ),
            "2026-08-20",
        )

        self.assertIn("2 other volunteer commitments", answer)
        self.assertIn("August 21", answer)
        self.assertIn("Saturday's mentoring session", answer)

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

    def test_follow_up_is_rewritten_before_retrieval_and_generation(self) -> None:
        pipeline = FakePipeline("Bring a side dish [S1].")
        history = (
            ConversationTurn(
                user_question="When is the neighborhood potluck?",
                assistant_answer="It is Sunday at 5 PM.",
            ),
        )
        rewriter = FakeRewriter(
            "What should we bring to the neighborhood potluck on Sunday at 5 PM?"
        )

        response = answer_question(
            pipeline,
            "What should we bring?",
            history=history,
            rewriter=rewriter,
        )

        self.assertEqual(response.question, "What should we bring?")
        self.assertEqual(response.retrieval_question, rewriter.retrieval_question)
        self.assertTrue(response.used_conversation_context)
        self.assertEqual(rewriter.calls, [(response.question, history)])
        self.assertEqual(
            pipeline.questions,
            [rewriter.retrieval_question, rewriter.retrieval_question],
        )

    def test_standalone_topic_switch_uses_current_question_end_to_end(self) -> None:
        pipeline = FakePipeline("The volunteer session is Saturday [S1].")
        llm = PromptCapturingLLM("Which birthdays still need gifts?")
        history = (
            ConversationTurn(
                user_question="Which birthdays still need gifts?",
                assistant_answer="Your friend's child still needs a gift.",
            ),
        )
        rewriter = ConversationQueryRewriter(
            llm=llm,
            reference_date="2026-08-20",
            timezone="America/Los_Angeles",
        )

        response = answer_question(
            pipeline,
            "When is my next volunteer work planned?",
            history=history,
            rewriter=rewriter,
        )

        self.assertEqual(response.question, "When is my next volunteer work planned?")
        self.assertEqual(response.retrieval_question, response.question)
        self.assertFalse(response.used_conversation_context)
        self.assertIsNone(llm.prompt)
        self.assertEqual(pipeline.questions, [response.question, response.question])

    def test_follow_up_history_requires_a_rewriter(self) -> None:
        pipeline = FakePipeline("unused")
        history = (ConversationTurn("What is due?", "The form is due Friday."),)

        with self.assertRaisesRegex(ValueError, "requires a follow-up query rewriter"):
            answer_question(pipeline, "What about that?", history=history)

        self.assertEqual(pipeline.questions, [])


if __name__ == "__main__":
    unittest.main()
