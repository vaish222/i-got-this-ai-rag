from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.agentic_experiments import (  # noqa: E402
    load_agentic_experiment,
    rebuild_phase9_agentic_namespace,
)
from i_got_this_rag.agentic_rag import (  # noqa: E402
    AgenticGraphConfig,
    CitationAttributor,
    CitationGroundingVerifier,
    EvidenceGrade,
    LangGraphRAG,
)
from i_got_this_rag.baseline import REFUSAL_TEXT  # noqa: E402
from i_got_this_rag.settings import Settings  # noqa: E402


def document(name: str, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={
            "chunk_id": f"{name}::chunk_000",
            "document_id": name,
            "document_title": name,
            "source_path": f"data/{name}.md",
        },
    )


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class QueueLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, prompt: Any) -> FakeResponse:
        self.calls += 1
        if not self.responses:
            raise AssertionError("Unexpected LLM invocation")
        return FakeResponse(self.responses.pop(0))


class FakeVectorStore:
    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def similarity_search_with_score(
        self,
        query: str,
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        self.calls.append({"query": query, "k": k, "filter": filter})
        return self.results[:k]


class SequenceEvidenceGrader:
    def __init__(self, sufficient_values: list[bool]) -> None:
        self.sufficient_values = list(sufficient_values)
        self.calls = 0

    def grade(
        self,
        question: str,
        results: list[tuple[Document, float]],
    ) -> EvidenceGrade:
        self.calls += 1
        sufficient = self.sufficient_values.pop(0)
        return EvidenceGrade(
            sufficient=sufficient,
            reason="enough" if sufficient else "weak",
            matched_terms=("birthday",) if sufficient else (),
            missing_terms=() if sufficient else ("birthday",),
            term_coverage=1.0 if sufficient else 0.0,
        )


class Phase9AgenticRAGTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = replace(Settings.from_environment(PROJECT_ROOT), top_k=5)
        self.source = document(
            "social_001",
            "The birthday party is Saturday, August 22 at 3:00 PM.",
        )

    def graph(
        self,
        llm: QueueLLM,
        grader: SequenceEvidenceGrader | None = None,
    ) -> tuple[LangGraphRAG, FakeVectorStore]:
        store = FakeVectorStore([(self.source, 0.9)])
        graph = LangGraphRAG(
            self.settings,
            store,  # type: ignore[arg-type]
            llm,
            config=AgenticGraphConfig(),
            evidence_grader=grader,
        )
        return graph, store

    def test_graph_contains_every_required_phase9_node(self) -> None:
        graph, _ = self.graph(QueueLLM([]))

        nodes = set(graph.graph.get_graph().nodes)

        self.assertTrue(
            {
                "query_analysis",
                "query_rewriting",
                "metadata_construction",
                "retrieval",
                "reranking",
                "evidence_grading",
                "generation",
                "grounding_verification",
            }.issubset(nodes)
        )

    def test_sufficient_evidence_generates_and_verifies_grounded_answer(self) -> None:
        graph, _ = self.graph(
            QueueLLM(["The birthday party is August 22 [S1]."]),
            SequenceEvidenceGrader([True]),
        )

        state = graph.invoke("When is the birthday party?")

        self.assertEqual(state["retrieval_attempts"], 1)
        self.assertTrue(state["evidence_sufficient"])
        self.assertTrue(state["grounded"])
        self.assertEqual(state["citations"], ["S1"])
        self.assertEqual(state["draft_answer"], state["answer"])
        self.assertNotEqual(state["answer"], REFUSAL_TEXT)
        self.assertNotIn("prepare_retry", [item["node"] for item in state["node_trace"]])

    def test_weak_evidence_retries_exactly_once_then_refuses(self) -> None:
        graph, store = self.graph(
            QueueLLM(["birthday party schedule retry"]),
            SequenceEvidenceGrader([False, False]),
        )

        state = graph.invoke("When is the birthday party?")

        self.assertEqual(state["retrieval_attempts"], 2)
        self.assertEqual(state["answer"], REFUSAL_TEXT)
        self.assertFalse(state["grounded"])
        self.assertEqual(len(state["evidence_history"]), 2)
        self.assertEqual(
            [item["strategy"] for item in state["query_history"]],
            ["original", "rewrite"],
        )
        retrieval_nodes = [
            item for item in state["node_trace"] if item["node"] == "retrieval"
        ]
        self.assertEqual(len(retrieval_nodes), 2)
        self.assertLessEqual(len(store.calls), 3)

    def test_retry_broadens_by_removing_metadata_filter(self) -> None:
        graph, _ = self.graph(
            QueueLLM(["birthday party schedule retry"]),
            SequenceEvidenceGrader([False, False]),
        )

        state = graph.invoke("When is the birthday party?")

        self.assertIsNotNone(state["retrieval_history"][0]["metadata_filter"])
        self.assertIsNone(state["retrieval_history"][1]["metadata_filter"])

    def test_retry_can_recover_then_generate(self) -> None:
        graph, _ = self.graph(
            QueueLLM(
                [
                    "birthday party schedule retry",
                    "The birthday party is August 22 [S1].",
                ]
            ),
            SequenceEvidenceGrader([False, True]),
        )

        state = graph.invoke("When is the birthday party?")

        self.assertEqual(state["retrieval_attempts"], 2)
        self.assertTrue(state["evidence_sufficient"])
        self.assertTrue(state["grounded"])
        self.assertEqual(state["answer"], "The birthday party is August 22 [S1].")

    def test_ungrounded_generated_answer_is_replaced_with_refusal(self) -> None:
        graph, _ = self.graph(
            QueueLLM(["The birthday party is August 30."]),
            SequenceEvidenceGrader([True]),
        )

        state = graph.invoke("When is the birthday party?")

        self.assertEqual(state["answer"], REFUSAL_TEXT)
        self.assertEqual(state["draft_answer"], "The birthday party is August 30.")
        self.assertFalse(state["grounded"])
        self.assertIn("no citations", state["refusal_reason"])

    def test_grounding_verifier_rejects_unsupported_concrete_fact(self) -> None:
        verifier = CitationGroundingVerifier()

        result = verifier.verify(
            "The birthday party is August 30 [S1].",
            [(self.source, 0.9)],
        )

        self.assertFalse(result.grounded)
        self.assertIn("August 30", result.unsupported_facts)

    def test_citation_attributor_repairs_supported_uncited_claim(self) -> None:
        attributor = CitationAttributor()

        answer = attributor.attribute(
            "The birthday party is Saturday, August 22nd at 3:00 PM.",
            [(self.source, 0.9)],
        )

        self.assertEqual(
            answer,
            "The birthday party is Saturday, August 22nd at 3:00 PM. [S1]",
        )
        self.assertTrue(
            CitationGroundingVerifier().verify(answer, [(self.source, 0.9)]).grounded
        )

    def test_citation_attributor_does_not_cite_unsupported_claim(self) -> None:
        answer = CitationAttributor().attribute(
            "The birthday party is August 30.",
            [(self.source, 0.9)],
        )

        self.assertEqual(answer, "The birthday party is August 30.")

    def test_citation_attributor_does_not_cite_unsupported_status(self) -> None:
        answer = CitationAttributor().attribute(
            "The birthday party is canceled on Saturday, August 22.",
            [(self.source, 0.9)],
        )

        self.assertEqual(
            answer,
            "The birthday party is canceled on Saturday, August 22.",
        )

    def test_citation_attributor_repairs_malformed_ordinal_suffix(self) -> None:
        answer = CitationAttributor().attribute(
            "The birthday party is August 22rd.",
            [(self.source, 0.9)],
        )

        self.assertEqual(answer, "The birthday party is August 22nd. [S1]")

    def test_citation_attributor_treats_status_as_a_structural_label(self) -> None:
        source = document(
            "social_001",
            "Neighborhood potluck\nRSVP: pending",
        )

        answer = CitationAttributor().attribute(
            "The neighborhood potluck RSVP status is pending.",
            [(source, 0.9)],
        )

        self.assertEqual(
            answer,
            "The neighborhood potluck RSVP status is pending. [S1]",
        )

    def test_generated_refusal_is_safe_but_not_marked_as_grounded(self) -> None:
        verifier = CitationGroundingVerifier()

        result = verifier.verify(REFUSAL_TEXT, [(self.source, 0.9)])

        self.assertFalse(result.grounded)
        self.assertIn("insufficient-information", result.reason)

    def test_phase9_config_keeps_one_retry_and_reranking_disabled(self) -> None:
        experiment = load_agentic_experiment(PROJECT_ROOT / "config" / "agentic_rag.yaml")

        self.assertEqual(experiment.graph_config.max_retrieval_attempts, 2)
        self.assertTrue(experiment.graph_config.metadata_filter_enabled)
        self.assertTrue(experiment.graph_config.retry_query_rewriting_enabled)
        self.assertFalse(experiment.graph_config.reranker_enabled)
        self.assertIsNone(experiment.reranker)

    def test_config_rejects_more_than_one_retry(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one retry"):
            AgenticGraphConfig(max_retrieval_attempts=3).validate()

    def test_namespace_rebuild_refuses_non_phase9_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "Refusing to rebuild"):
            rebuild_phase9_agentic_namespace(
                object(),  # type: ignore[arg-type]
                "baseline",
                [],
            )


if __name__ == "__main__":
    unittest.main()
