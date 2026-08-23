from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.claim_faithfulness import (  # noqa: E402
    ClaimCandidate,
    RetrievedContext,
    audit_claim,
    audit_question,
    determine_conclusion,
)
from i_got_this_rag.grounded_generation import REFUSAL_TEXT  # noqa: E402


def context(
    *,
    source_id: str = "S1",
    domain: str = "activities",
    text: str = "Saturday, August 22: 9:00–10:15 AM — `child_01`: swim practice.",
) -> RetrievedContext:
    return RetrievedContext(
        source_id=source_id,
        document_id="activities_001",
        chunk_id="activities_001::chunk_000",
        title="Family Schedule — August 17–23, 2026",
        domain=domain,
        text=text,
    )


def question_result(answer: str, trace: dict | None, automated: float = 0.0) -> dict:
    return {
        "question_id": "Q001",
        "question": "What time does Saturday's activity start?",
        "generated_answer": answer,
        "faithfulness": {"score": automated},
        "generation_trace": trace,
    }


class ClaimFaithfulnessTests(unittest.TestCase):
    def test_grounded_structured_paraphrase_receives_claim_credit(self) -> None:
        item = {
            "title": "Swim practice starts at 9:00 AM",
            "date": "2026-08-22",
            "time": "09:00",
            "category": "activities",
            "person": "child_01",
            "source_id": "S1",
            "evidence": "9:00–10:15 AM — `child_01`: swim practice.",
        }
        audited = audit_question(
            question_result("Swim practice starts at 9:00 AM [S1]", {"structured_items": [item]}),
            (context(),),
            "test-model",
            "2026-08-20",
        )

        self.assertEqual(audited["total_factual_claims"], 3)
        self.assertEqual(audited["claim_faithfulness"], 1.0)
        self.assertTrue(audited["claims"][0]["supported"])
        self.assertTrue(audited["evaluator_disagreement"])
        self.assertEqual(audited["retrieved_source_ids"], ["S1"])
        self.assertEqual(audited["retrieved_document_ids"], ["activities_001"])

    def test_unsupported_advice_is_categorized_not_excused_as_paraphrase(self) -> None:
        claim = ClaimCandidate("You should troubleshoot the HVAC system", ("S1",))
        audited = audit_claim(
            claim,
            (
                context(
                    domain="household",
                    text="HVAC service appointment is Saturday, August 22.",
                ),
            ),
            "What is scheduled Saturday?",
            "2026-08-20",
        )

        self.assertFalse(audited["supported"])
        self.assertEqual(audited["category"], "unsupported advice")

    def test_irrelevant_but_grounded_claim_is_not_a_faithfulness_failure(self) -> None:
        volunteer = context(
            domain="volunteer",
            text="Sunday, August 23: `adult_01` has a mentor video call at 11:00 AM.",
        )
        claim = ClaimCandidate("adult_01 has a mentor video call at 11:00 AM", ("S1",))
        audited = audit_claim(
            claim,
            (volunteer,),
            "What activities do the kids have Sunday?",
            "2026-08-20",
        )

        self.assertTrue(audited["supported"])
        self.assertFalse(audited["relevant_to_question"])
        self.assertEqual(audited["category"], "irrelevant but grounded information")

    def test_support_uses_all_retrieved_context_not_only_a_wrong_citation(self) -> None:
        claim = ClaimCandidate("Picture day is August 26 at 10:20 AM", ("S1",))
        audited = audit_claim(
            claim,
            (
                context(source_id="S1", domain="activities", text="Piano is Friday."),
                context(
                    source_id="S2",
                    domain="school",
                    text="Elementary picture day is August 26 at 10:20 AM.",
                ),
            ),
            "When is elementary picture day?",
            "2026-08-20",
        )

        self.assertTrue(audited["supported"])
        self.assertEqual(audited["supporting_source_ids"], ["S2"])

    def test_noon_and_twelve_oclock_are_equivalent(self) -> None:
        audited = audit_claim(
            ClaimCandidate("The RSVP is due at 12:00 PM", ("S1",)),
            (
                context(
                    domain="social",
                    text="The neighborhood potluck RSVP is due Friday at noon.",
                ),
            ),
            "When is the RSVP due?",
            "2026-08-20",
        )

        self.assertTrue(audited["supported"])

    def test_refusal_reports_no_factual_claims_instead_of_zero(self) -> None:
        audited = audit_question(
            question_result(REFUSAL_TEXT, None, automated=1.0),
            (context(),),
            "test-model",
            "2026-08-20",
        )

        self.assertTrue(audited["no_factual_claims"])
        self.assertIsNone(audited["claim_faithfulness"])
        self.assertFalse(audited["evaluator_disagreement"])

    def test_conclusion_can_distinguish_evaluator_and_generation_problems(self) -> None:
        conclusion = determine_conclusion(
            (
                {"evaluator_disagreement_count": 2, "unsupported_factual_claims": 0},
                {"evaluator_disagreement_count": 0, "unsupported_factual_claims": 1},
            )
        )
        self.assertEqual(conclusion["code"], "C")


if __name__ == "__main__":
    unittest.main()
