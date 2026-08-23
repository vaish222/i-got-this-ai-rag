from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.conversation import (  # noqa: E402
    ConversationQueryRewriter,
    ConversationTurn,
    format_conversation_history,
    recent_turns,
)


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[object] = []

    def invoke(self, prompt: object) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.response)


class ConversationTests(unittest.TestCase):
    def test_recent_turns_keeps_only_the_latest_exchanges(self) -> None:
        history = tuple(
            ConversationTurn(f"question {index}", f"answer {index}")
            for index in range(5)
        )

        selected = recent_turns(history, maximum=3)

        self.assertEqual(selected, history[-3:])

    def test_recent_turns_requires_positive_memory(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one exchange"):
            recent_turns((), maximum=0)

    def test_history_is_normalized_and_limited(self) -> None:
        rendered = format_conversation_history(
            (ConversationTurn("  what\n happened?  ", "x" * 2000),)
        )

        self.assertTrue(rendered.startswith("User: what happened?\nAssistant: "))
        self.assertEqual(len(rendered.split("Assistant: ", 1)[1]), 1500)

    def test_no_history_returns_original_without_calling_model(self) -> None:
        llm = FakeLLM("unused")
        rewriter = ConversationQueryRewriter(llm, "2025-08-21", "America/Los_Angeles")

        result = rewriter.rewrite("What is due Friday?", ())

        self.assertEqual(result.retrieval_question, "What is due Friday?")
        self.assertFalse(result.used_history)
        self.assertEqual(llm.prompts, [])

    def test_history_can_resolve_a_follow_up_reference(self) -> None:
        llm = FakeLLM(
            "What should we bring to the neighborhood potluck on August 23 at 5 PM?"
        )
        rewriter = ConversationQueryRewriter(llm, "2025-08-21", "America/Los_Angeles")
        history = (
            ConversationTurn(
                "When is the neighborhood potluck?",
                "The neighborhood potluck is August 23 at 5 PM.",
            ),
        )

        result = rewriter.rewrite("What should we bring?", history)

        self.assertEqual(
            result.retrieval_question,
            "What should we bring to the neighborhood potluck on August 23 at 5 PM?",
        )
        self.assertTrue(result.used_history)
        self.assertEqual(result.guard_repairs, ())
        self.assertEqual(len(llm.prompts), 1)

    def test_invented_facts_are_removed_from_a_follow_up(self) -> None:
        llm = FakeLLM("What should we bring to the potluck on August 30 at 7 PM?")
        rewriter = ConversationQueryRewriter(llm, "2025-08-21", "America/Los_Angeles")
        history = (
            ConversationTurn(
                "When is the potluck?",
                "The potluck is August 23 at 5 PM.",
            ),
        )

        result = rewriter.rewrite("What should we bring?", history)

        self.assertNotIn("August 30", result.retrieval_question)
        self.assertNotIn("7 PM", result.retrieval_question)
        self.assertEqual(result.guard_repairs[0]["reason"], "invented_fact_terms")

    def test_current_question_protected_terms_are_restored(self) -> None:
        llm = FakeLLM("What needs to be prepared for the event?")
        rewriter = ConversationQueryRewriter(llm, "2025-08-21", "America/Los_Angeles")
        history = (ConversationTurn("What is upcoming?", "A field trip is upcoming."),)

        result = rewriter.rewrite("What about child_01 on Friday?", history)

        self.assertIn("child_01", result.retrieval_question)
        self.assertIn("Friday", result.retrieval_question)
        self.assertEqual(
            result.guard_repairs[0]["reason"],
            "missing_current_question_terms",
        )

    def test_empty_model_output_falls_back_to_the_original_question(self) -> None:
        rewriter = ConversationQueryRewriter(
            FakeLLM("  "),
            "2025-08-21",
            "America/Los_Angeles",
        )

        result = rewriter.rewrite(
            "What about that?",
            (ConversationTurn("What is due?", "The form is due Friday."),),
        )

        self.assertEqual(result.retrieval_question, "What about that?")
        self.assertEqual(result.guard_repairs[-1]["reason"], "empty_model_output_fallback")


if __name__ == "__main__":
    unittest.main()
