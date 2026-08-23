from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.baseline import REFUSAL_TEXT  # noqa: E402
from i_got_this_rag.final_evaluation import (  # noqa: E402
    FAITHFULNESS_METHOD,
    REQUIRED_VERSION_IDS,
    DeterministicFaithfulnessScorer,
    FinalVersionSpec,
    HybridCandidateRerankingRAG,
    evaluate_version_artifact,
    load_final_evaluation_config,
    nearest_rank_percentile,
    rebuild_phase10_namespace,
)
from i_got_this_rag.evaluation import EvaluationDataset  # noqa: E402
from i_got_this_rag.settings import Settings  # noqa: E402


def document(name: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "chunk_id": f"{name}::chunk_000",
            "document_id": name,
            "document_title": name,
            "source_path": f"data/{name}.md",
        },
    )


class FakeRetriever:
    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results
        self.requested_k: list[int] = []

    def retrieve(self, question: str, k: int) -> list[tuple[Document, float]]:
        self.requested_k.append(k)
        return self.results[:k]


class ReversingReranker:
    def rerank(
        self,
        question: str,
        candidates: list[tuple[Document, float]],
        top_k: int,
    ) -> list[tuple[Document, float]]:
        return list(reversed(candidates))[:top_k]


class Phase10FinalEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = document(
            "social_001",
            "Neighborhood potluck\nRSVP: pending\nEvent: Sunday, August 23.",
        )

    def test_config_defines_exact_eight_version_matrix(self) -> None:
        config = load_final_evaluation_config(
            PROJECT_ROOT / "config" / "final_evaluation.yaml",
            PROJECT_ROOT,
        )

        self.assertEqual(
            tuple(version.version_id for version in config.versions),
            REQUIRED_VERSION_IDS,
        )
        self.assertEqual(config.faithfulness_method, FAITHFULNESS_METHOD)
        self.assertEqual(config.final_top_k, 5)
        self.assertEqual(config.hybrid_candidate_k, 20)
        self.assertTrue(config.runtime_namespace.startswith("phase10-"))

    def test_supported_answer_is_faithful_after_deterministic_attribution(self) -> None:
        result = DeterministicFaithfulnessScorer().score(
            answerable=True,
            answer="The neighborhood potluck RSVP is pending.",
            results=[(self.source, 0.9)],
        )

        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["faithful"])
        self.assertIn("[S1]", result["attributed_answer"])

    def test_unsupported_answer_is_not_faithful(self) -> None:
        result = DeterministicFaithfulnessScorer().score(
            answerable=True,
            answer="The neighborhood potluck RSVP is completed.",
            results=[(self.source, 0.9)],
        )

        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["faithful"])

    def test_refusal_scores_only_for_unanswerable_question(self) -> None:
        scorer = DeterministicFaithfulnessScorer()

        answerable = scorer.score(
            answerable=True,
            answer=REFUSAL_TEXT,
            results=[(self.source, 0.9)],
        )
        unanswerable = scorer.score(
            answerable=False,
            answer=REFUSAL_TEXT,
            results=[(self.source, 0.9)],
        )

        self.assertEqual(answerable["score"], 0.0)
        self.assertEqual(unanswerable["score"], 1.0)
        self.assertTrue(unanswerable["correct_refusal"])

    def test_hybrid_reranking_pipeline_records_candidates_and_final_top5(self) -> None:
        settings = replace(Settings.from_environment(PROJECT_ROOT), top_k=5)
        candidates = [
            (document(f"doc_{index:02d}", f"content {index}"), float(index))
            for index in range(20)
        ]
        retriever = FakeRetriever(candidates)
        pipeline = HybridCandidateRerankingRAG(
            settings,
            retriever,
            ReversingReranker(),
            object(),
            candidate_k=20,
        )

        trace = pipeline.retrieve_with_trace("question")

        self.assertEqual(retriever.requested_k, [20])
        self.assertEqual(len(trace["candidate_results"]), 20)
        self.assertEqual(len(trace["results"]), 5)
        self.assertTrue(trace["reranking_enabled"])
        self.assertEqual(
            trace["results"][0][0].metadata["document_id"],
            "doc_19",
        )

    def test_nearest_rank_p95_is_deterministic(self) -> None:
        self.assertEqual(nearest_rank_percentile([1.0, 4.0, 2.0, 3.0], 0.95), 4.0)

    def test_version_artifact_scores_recall_faithfulness_and_latency(self) -> None:
        dataset = EvaluationDataset(
            path=PROJECT_ROOT / "evaluation" / "questions.json",
            schema_version="1.0",
            dataset_name="test",
            reference_date="2026-08-20",
            timezone="America/Los_Angeles",
            questions=(
                {
                    "question_id": "Q001",
                    "question": "What is the status?",
                    "category": "exact_lookup",
                    "answerable": True,
                    "expected_answer": "pending",
                    "expected_source_ids": ["social_001"],
                    "expected_sources": ["source"],
                },
            ),
            sha256="dataset",
        )
        payload = {
            "experiment_id": "source",
            "summary": {
                "mean_expected_source_rank": 1.0,
                "retrieval_failure_count": 0,
            },
            "questions": [
                {
                    "question_id": "Q001",
                    "category": "exact_lookup",
                    "answerable": True,
                    "recall_at_5": 1.0,
                    "expected_source_ranks": {"social_001": 1},
                    "generated_answer": "The neighborhood potluck RSVP is pending.",
                    "retrieved_chunks": [
                        {
                            "chunk_id": "social_001::chunk_000",
                            "similarity_score": 0.9,
                        }
                    ],
                    "total_latency_seconds": 1.25,
                }
            ],
        }
        spec = FinalVersionSpec(
            version_id="baseline_dense",
            label="Baseline dense RAG",
            mechanism="dense",
            source_results_path=Path("source.json"),
            runtime_experiment_id=None,
        )

        evaluated = evaluate_version_artifact(
            spec,
            payload,
            dataset,
            {"social_001::chunk_000": self.source},
        )

        self.assertEqual(evaluated["metrics"]["recall_at_5"], 1.0)
        self.assertEqual(evaluated["metrics"]["faithfulness"], 1.0)
        self.assertEqual(evaluated["metrics"]["average_latency_seconds"], 1.25)

    def test_namespace_rebuild_refuses_non_phase10_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "Refusing to rebuild"):
            rebuild_phase10_namespace(
                object(),  # type: ignore[arg-type]
                "baseline",
                [],
            )


if __name__ == "__main__":
    unittest.main()
