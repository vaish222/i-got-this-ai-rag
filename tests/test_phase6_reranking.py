from __future__ import annotations

import sys
import unittest
from pathlib import Path

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.evaluation import (  # noqa: E402
    BaselineEvaluator,
    diagnose_retrieval_and_reranking_failures,
)
from i_got_this_rag.reranking import BM25CandidateReranker, DenseRerankingRAG  # noqa: E402
from i_got_this_rag.reranking_experiments import (  # noqa: E402
    load_reranking_experiments,
    rebuild_phase6_dense_namespace,
)
from i_got_this_rag.settings import Settings  # noqa: E402


def document(index: int, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={
            "chunk_id": f"source_{index:03d}::chunk_000",
            "document_id": f"source_{index:03d}",
            "document_title": f"Source {index}",
            "source_path": f"data/source_{index:03d}.md",
        },
    )


class FakeVectorStore:
    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results

    def similarity_search_with_score(self, question: str, k: int) -> list[tuple[Document, float]]:
        return self.results[:k]


class TracedFakePipeline:
    def __init__(self) -> None:
        self.candidates = [(document(index, f"candidate {index}"), 1 - index / 100) for index in range(20)]

    def retrieve_with_trace(self, question: str) -> dict[str, object]:
        return {
            "results": self.candidates[:5],
            "candidate_results": self.candidates,
            "candidate_retrieval_latency_seconds": 0.2,
            "reranking_latency_seconds": 0.01,
            "reranking_enabled": True,
        }

    def generate(self, question: str, results: list[tuple[Document, float]]) -> str:
        return "Answer [S1]."


class Phase6RerankingTests(unittest.TestCase):
    def test_configs_define_top5_and_top20_reranking_comparison(self) -> None:
        baseline, reranked = load_reranking_experiments(
            PROJECT_ROOT / "config" / "reranking_experiments"
        )

        self.assertEqual(baseline.candidate_k, 5)
        self.assertFalse(baseline.reranker_enabled)
        self.assertEqual(reranked.candidate_k, 20)
        self.assertTrue(reranked.reranker_enabled)
        self.assertEqual(reranked.reranker, "bm25")
        self.assertEqual(baseline.pinecone_namespace, reranked.pinecone_namespace)

    def test_bm25_reranker_promotes_lexically_relevant_candidate(self) -> None:
        candidates = [
            (document(1, "General family information"), 0.95),
            (document(2, "Course schedule and assignment"), 0.90),
            (document(3, "Neighborhood potluck RSVP August 24 lemon bars"), 0.85),
        ]

        results = BM25CandidateReranker().rerank(
            "Which potluck RSVP is due August 24?",
            candidates,
            top_k=2,
        )

        self.assertEqual(results[0][0].metadata["document_id"], "source_003")
        self.assertEqual(results[0][0].metadata["reranking_components"]["candidate_rank"], 3)
        self.assertGreater(results[0][1], results[1][1])

    def test_dense_reranking_pipeline_records_top20_trace_and_final_top5(self) -> None:
        candidates = [
            (document(index, "potluck" if index == 19 else f"candidate {index}"), 1 - index / 100)
            for index in range(20)
        ]
        settings = Settings.from_environment(PROJECT_ROOT)
        pipeline = DenseRerankingRAG(
            settings,
            FakeVectorStore(candidates),  # type: ignore[arg-type]
            llm=object(),
            candidate_k=20,
            reranker=BM25CandidateReranker(),
        )

        trace = pipeline.retrieve_with_trace("potluck")

        self.assertEqual(len(trace["candidate_results"]), 20)
        self.assertEqual(len(trace["results"]), 5)
        self.assertTrue(trace["reranking_enabled"])
        self.assertEqual(trace["results"][0][0].metadata["document_id"], "source_019")
        self.assertGreaterEqual(trace["reranking_latency_seconds"], 0)

    def test_evaluator_records_candidates_and_separate_reranking_latency(self) -> None:
        question = {
            "question_id": "QTEST",
            "question": "Question",
            "expected_answer": "Answer",
            "expected_source_ids": ["source_000"],
            "expected_sources": ["Source 0"],
            "category": "exact_lookup",
            "answerable": True,
        }

        result = BaselineEvaluator(TracedFakePipeline()).evaluate_question(question)

        self.assertEqual(result["candidate_k"], 20)
        self.assertEqual(len(result["retrieved_chunks"]), 5)
        self.assertEqual(result["candidate_expected_source_rank"], 1)
        self.assertEqual(result["candidate_recall"], 1.0)
        self.assertTrue(result["reranking_enabled"])
        self.assertEqual(result["candidate_retrieval_latency_seconds"], 0.2)
        self.assertEqual(result["reranking_latency_seconds"], 0.01)

    def test_failure_diagnosis_separates_retrieval_from_reranking(self) -> None:
        results = [
            {
                "question_id": "Q001",
                "category": "semantic",
                "expected_source_ids": ["missing_candidate", "lost_after_rerank"],
                "expected_source_ranks": {
                    "missing_candidate": None,
                    "lost_after_rerank": None,
                },
                "candidate_expected_source_ranks": {
                    "missing_candidate": None,
                    "lost_after_rerank": 10,
                },
            }
        ]

        diagnosis = diagnose_retrieval_and_reranking_failures(results)

        self.assertEqual(diagnosis["retrieval_failure_source_count"], 1)
        self.assertEqual(diagnosis["reranking_failure_source_count"], 1)
        self.assertEqual(
            diagnosis["failures"][0]["retrieval_failure_source_ids"],
            ["missing_candidate"],
        )
        self.assertEqual(
            diagnosis["failures"][0]["reranking_failure_source_ids"],
            ["lost_after_rerank"],
        )

    def test_namespace_rebuild_refuses_non_phase6_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "Refusing to rebuild"):
            rebuild_phase6_dense_namespace(object(), "baseline", [])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
