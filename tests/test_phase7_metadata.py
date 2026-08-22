from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.evaluation import BaselineEvaluator  # noqa: E402
from i_got_this_rag.metadata_experiments import (  # noqa: E402
    build_metadata_impact,
    load_metadata_experiments,
    rebuild_phase7_metadata_namespace,
)
from i_got_this_rag.metadata_retrieval import (  # noqa: E402
    MetadataAwareDenseRAG,
    MetadataQueryAnalyzer,
    enrich_metadata_facets,
)
from i_got_this_rag.settings import Settings  # noqa: E402


def document(index: int, **metadata: Any) -> Document:
    return Document(
        page_content=f"candidate {index}",
        metadata={
            "chunk_id": f"source_{index:03d}::chunk_000",
            "document_id": f"source_{index:03d}",
            "document_title": f"Source {index}",
            "source_path": f"data/source_{index:03d}.md",
            **metadata,
        },
    )


class FakeVectorStore:
    def __init__(
        self,
        filtered: list[tuple[Document, float]],
        unfiltered: list[tuple[Document, float]],
    ) -> None:
        self.filtered = filtered
        self.unfiltered = unfiltered
        self.calls: list[dict[str, Any]] = []

    def similarity_search_with_score(
        self,
        question: str,
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        self.calls.append({"question": question, "k": k, "filter": filter})
        return (self.filtered if filter is not None else self.unfiltered)[:k]


class TracedMetadataPipeline:
    def retrieve_with_trace(self, question: str) -> dict[str, Any]:
        results = [(document(1), 0.9)]
        return {
            "results": results,
            "candidate_results": results,
            "candidate_retrieval_latency_seconds": 0.03,
            "reranking_latency_seconds": 0.0,
            "reranking_enabled": False,
            "metadata_filter_enabled": True,
            "metadata_filter_applied": True,
            "metadata_constraints": {"domain": ["social"]},
            "metadata_filter": {"facet_domain": {"$eq": "social"}},
            "metadata_analysis_latency_seconds": 0.001,
            "metadata_filtered_result_count": 1,
            "metadata_fallback_result_count": 0,
            "retrieval_query": question,
        }

    def generate(self, question: str, results: list[tuple[Document, float]]) -> str:
        return "Answer [S1]."


class Phase7MetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = replace(Settings.from_environment(PROJECT_ROOT), top_k=5)
        self.analyzer = MetadataQueryAnalyzer("2026-08-20")

    def test_analyzer_builds_structured_filter_without_rewriting_question(self) -> None:
        constraints = self.analyzer.analyze(
            "Which birthday invitations still need an RSVP?"
        )

        self.assertEqual(constraints.domain, ("social",))
        self.assertEqual(constraints.document_type, ("invitation_tracker",))
        self.assertEqual(constraints.event_type, ("birthday",))
        self.assertEqual(constraints.rsvp_status, ("pending",))
        self.assertEqual(
            constraints.to_pinecone_filter(),
            {
                "$and": [
                    {"facet_domain": {"$eq": "social"}},
                    {"facet_document_type": {"$eq": "invitation_tracker"}},
                    {"facet_event_type__birthday": {"$eq": True}},
                    {"facet_rsvp_status__pending": {"$eq": True}},
                ]
            },
        )

    def test_analyzer_skips_domain_filter_for_broad_cross_domain_question(self) -> None:
        constraints = self.analyzer.analyze(
            "What do I need to prepare this weekend across school, activities, "
            "learning, volunteering, household, and social commitments?"
        )

        self.assertEqual(constraints.to_dict(), {})
        self.assertIsNone(constraints.to_pinecone_filter())

    def test_analyzer_extracts_exact_person_and_resolved_weekday(self) -> None:
        constraints = self.analyzer.analyze(
            "What school activity does child_02 have on Saturday?"
        )

        self.assertEqual(constraints.person, ("child_02",))
        self.assertEqual(constraints.event_date, ("2026-08-22",))

    def test_enrichment_adds_filter_safe_event_status_and_date_facets(self) -> None:
        chunk = Document(
            page_content=(
                "Birthday party on August 22, 2026. RSVP: pending. "
                "Gift status: Needed."
            ),
            metadata={
                "document_id": "social_001",
                "document_type": "invitation_tracker",
                "domain": "social",
                "person": ["child_02"],
                "related_person": "friend_child_01",
                "tags": ["birthdays", "rsvp"],
                "updated_at": "2026-08-20",
            },
        )

        metadata = enrich_metadata_facets([chunk], "2026-08-20")[0].metadata

        self.assertEqual(metadata["facet_domain"], "social")
        self.assertTrue(metadata["facet_person__child_02"])
        self.assertTrue(metadata["facet_event_type__birthday"])
        self.assertTrue(metadata["facet_event_date__2026_08_22"])
        self.assertTrue(metadata["facet_rsvp_status__pending"])
        self.assertTrue(metadata["facet_gift_status__needed"])

    def test_filtered_retrieval_preserves_query_and_fills_top5_with_dense_results(self) -> None:
        dense = [(document(index), 1 - index / 100) for index in range(1, 7)]
        filtered = [dense[2], dense[0]]
        store = FakeVectorStore(filtered, dense)
        pipeline = MetadataAwareDenseRAG(
            self.settings,
            store,  # type: ignore[arg-type]
            llm=object(),
            analyzer=self.analyzer,
            metadata_filter_enabled=True,
            fallback_to_unfiltered=True,
        )
        question = "Which invitations still need an RSVP?"

        trace = pipeline.retrieve_with_trace(question)

        self.assertEqual(len(trace["results"]), 5)
        self.assertEqual(trace["retrieval_query"], question)
        self.assertEqual(trace["metadata_filtered_result_count"], 2)
        self.assertEqual(trace["metadata_fallback_result_count"], 3)
        self.assertIsNotNone(store.calls[0]["filter"])
        self.assertIsNone(store.calls[1]["filter"])
        self.assertEqual(
            trace["results"][0][0].metadata["metadata_retrieval_components"]["origin"],
            "filtered",
        )

    def test_unfiltered_path_performs_one_plain_dense_search(self) -> None:
        dense = [(document(index), 1 - index / 100) for index in range(5)]
        store = FakeVectorStore([], dense)
        pipeline = MetadataAwareDenseRAG(
            self.settings,
            store,  # type: ignore[arg-type]
            llm=object(),
            analyzer=self.analyzer,
            metadata_filter_enabled=False,
        )

        trace = pipeline.retrieve_with_trace("Question")

        self.assertEqual(len(store.calls), 1)
        self.assertIsNone(store.calls[0]["filter"])
        self.assertFalse(trace["metadata_filter_applied"])
        self.assertEqual(trace["metadata_fallback_result_count"], 0)

    def test_evaluator_records_metadata_filter_trace(self) -> None:
        question = {
            "question_id": "QTEST",
            "question": "Question",
            "expected_answer": "Answer",
            "expected_source_ids": ["source_001"],
            "expected_sources": ["Source 1"],
            "category": "exact_lookup",
            "answerable": True,
        }

        result = BaselineEvaluator(TracedMetadataPipeline()).evaluate_question(question)

        self.assertEqual(result["retrieval_query"], question["question"])
        self.assertTrue(result["metadata_filter_applied"])
        self.assertEqual(result["metadata_constraints"], {"domain": ["social"]})
        self.assertEqual(result["metadata_filtered_result_count"], 1)
        self.assertEqual(result["metadata_analysis_latency_seconds"], 0.001)

    def test_configs_define_filtered_and_unfiltered_comparison(self) -> None:
        unfiltered, filtered = load_metadata_experiments(
            PROJECT_ROOT / "config" / "metadata_experiments"
        )

        self.assertFalse(unfiltered.metadata_filter_enabled)
        self.assertTrue(filtered.metadata_filter_enabled)
        self.assertTrue(filtered.fallback_to_unfiltered)
        self.assertEqual(unfiltered.pinecone_namespace, filtered.pinecone_namespace)

    def test_metadata_impact_reports_recall_improvement(self) -> None:
        baseline = {
            "questions": [
                {
                    "question_id": "Q001",
                    "category": "semantic",
                    "recall_at_5": 0.0,
                    "expected_source_ranks": {"source_001": None},
                }
            ]
        }
        filtered = {
            "questions": [
                {
                    "question_id": "Q001",
                    "category": "semantic",
                    "recall_at_5": 1.0,
                    "expected_source_ranks": {"source_001": 1},
                    "metadata_constraints": {"domain": ["social"]},
                }
            ]
        }

        impact = build_metadata_impact(baseline, filtered)

        self.assertEqual(impact["outcome_counts"]["improved"], 1)
        self.assertEqual(impact["questions"][0]["outcome"], "improved")

    def test_namespace_rebuild_refuses_non_phase7_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "Refusing to rebuild"):
            rebuild_phase7_metadata_namespace(
                object(),  # type: ignore[arg-type]
                "baseline",
                [],
            )


if __name__ == "__main__":
    unittest.main()
