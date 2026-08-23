from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.concise_generation import (  # noqa: E402
    EVIDENCE_MODE_ALL,
    EVIDENCE_MODE_RELEVANCE_FIRST,
    PROMPT_MODE_CONCISE,
    AnswerLengthPolicy,
    SelectedAnswerItem,
    SelectedAnswerPayload,
    classify_answer_intent,
    generate_qwen_experiment_answer,
    select_relevance_first_evidence,
)
from i_got_this_rag.grounded_generation import extract_question_constraints  # noqa: E402
from i_got_this_rag.qwen_generation_experiments import (  # noqa: E402
    load_qwen_generation_experiments,
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


class RawStructuredRunnable:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.prompts: list[Any] = []

    def invoke(self, prompt: Any) -> dict[str, Any]:
        self.prompts.append(prompt)
        return {
            "parsed": self.payload,
            "raw": SimpleNamespace(
                usage_metadata={
                    "input_tokens": 120,
                    "output_tokens": 40,
                    "total_tokens": 160,
                }
            ),
            "parsing_error": None,
        }


class RawStructuredLLM:
    def __init__(self, payload: Any) -> None:
        self.runnable = RawStructuredRunnable(payload)
        self.schema: type | None = None
        self.include_raw: bool | None = None

    def with_structured_output(
        self,
        schema: type,
        *,
        method: str,
        include_raw: bool,
    ) -> RawStructuredRunnable:
        self.schema = schema
        self.include_raw = include_raw
        self.method = method
        return self.runnable


class QwenGenerationExperimentTests(unittest.TestCase):
    def test_intent_policy_distinguishes_lookup_summary_and_planning(self) -> None:
        reference = "2026-08-20"
        self.assertEqual(
            classify_answer_intent(
                "Are there kids' activities Sunday?",
                extract_question_constraints("Are there kids' activities Sunday?", reference),
            ),
            "yes_no",
        )
        self.assertEqual(
            classify_answer_intent(
                "What's next for everyone this week?",
                extract_question_constraints("What's next for everyone this week?", reference),
            ),
            "cross_domain_summary",
        )
        self.assertEqual(
            classify_answer_intent(
                "Plan my week.",
                extract_question_constraints("Plan my week.", reference),
            ),
            "planning_request",
        )

    def test_relevance_first_keeps_kids_sunday_and_logs_every_decision(self) -> None:
        question = "Are there kids' activities on Sunday?"
        constraints = extract_question_constraints(question, "2026-08-20")
        results = [
            (
                document(
                    "activities",
                    "Sunday, August 23: child_02 watercolor class at 10:00 AM.",
                    "activities_001",
                ),
                0.9,
            ),
            (
                document(
                    "volunteer",
                    "Sunday, August 23: adult_01 mentor call at 11:00 AM.",
                    "volunteer_001",
                ),
                0.89,
            ),
            (
                document(
                    "household",
                    "Saturday, August 22: HVAC appointment.",
                    "household_001",
                ),
                0.88,
            ),
        ]

        selected, decisions = select_relevance_first_evidence(
            results,
            question=question,
            constraints=constraints,
            reference_date="2026-08-20",
            intent="yes_no",
        )

        self.assertEqual([rank for rank, _, _ in selected], [1])
        self.assertEqual(len(decisions), 3)
        self.assertTrue(decisions[0].included)
        self.assertFalse(decisions[1].included)

    def test_concise_generation_enforces_item_limit_and_tracks_usage(self) -> None:
        payload = SelectedAnswerPayload(
            items=[
                SelectedAnswerItem(
                    title=f"Relevant fact {index}",
                    source_id="S1",
                    evidence="Sunday activity",
                    relevance_reason="matches Sunday activity",
                )
                for index in range(1, 5)
            ]
        )
        llm = RawStructuredLLM(payload)
        generated = generate_qwen_experiment_answer(
            llm=llm,
            question="Which activity details are recorded?",
            results=[(document("activities", "Sunday activity at 10:00 AM.", "a1"), 0.9)],
            reference_date="2026-08-20",
            timezone="America/Los_Angeles",
            prompt_mode=PROMPT_MODE_CONCISE,
            evidence_mode=EVIDENCE_MODE_ALL,
            length_policy=AnswerLengthPolicy(exact_lookup=3),
        )

        self.assertEqual(len(generated.items), 3)
        self.assertNotIn("Okay", generated.answer)
        self.assertEqual(generated.token_usage["output_tokens"], 40)
        self.assertTrue(llm.include_raw)

    def test_three_configs_resolve_to_one_qwen_model(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            config_dir = Path(directory)
            for mode, prompt, evidence in (
                ("E1", "current_strict", "all"),
                ("E2", "concise_relevance", "all"),
                ("E3", "concise_relevance", "relevance_first"),
            ):
                (config_dir / f"{mode}.yaml").write_text(
                    "\n".join(
                        (
                            f"experiment_id: {mode}",
                            f"label: {mode}",
                            "provider: nebius",
                            "api_style: openai_compatible",
                            "model_env: TEST_QWEN_MODEL",
                            "base_url_default: https://example.test/v1/",
                            "api_key_env: TEST_QWEN_KEY",
                            f"prompt_mode: {prompt}",
                            f"evidence_mode: {evidence}",
                            "max_output_tokens: 900",
                        )
                    ),
                    encoding="utf-8",
                )
            with patch.dict(
                os.environ,
                {"TEST_QWEN_MODEL": "Qwen/test", "TEST_QWEN_KEY": "secret"},
                clear=False,
            ):
                experiments = load_qwen_generation_experiments(
                    config_dir,
                    PROJECT_ROOT,
                )

        self.assertEqual(len(experiments), 3)
        self.assertEqual({item.chat_config.model for item in experiments}, {"Qwen/test"})
        self.assertEqual(experiments[2].evidence_mode, EVIDENCE_MODE_RELEVANCE_FIRST)


if __name__ == "__main__":
    unittest.main()
