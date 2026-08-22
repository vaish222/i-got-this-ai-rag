from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.evaluation import BaselineEvaluator, serialize_retrieval  # noqa: E402
from i_got_this_rag.query_experiments import (  # noqa: E402
    build_query_transformation_impact,
    load_query_experiments,
    rebuild_phase8_query_namespace,
)
from i_got_this_rag.query_transformation import (  # noqa: E402
    LLMQueryTransformer,
    QueryTransformationRAG,
    extract_protected_terms,
    parse_multi_query_output,
)
from i_got_this_rag.settings import Settings  # noqa: E402


def document(name: str) -> Document:
    return Document(
        page_content=f"content {name}",
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


class FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def invoke(self, prompt: Any) -> FakeResponse:
        self.calls += 1
        return FakeResponse(self.content)


class FakeVectorStore:
    def __init__(self, results_by_query: dict[str, list[tuple[Document, float]]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[str] = []

    def similarity_search_with_score(
        self,
        query: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        self.calls.append(query)
        return self.results_by_query[query][:k]


class TracedQueryPipeline:
    def retrieve_with_trace(self, question: str) -> dict[str, Any]:
        results = [(document("source_001"), 0.9)]
        return {
            "results": results,
            "candidate_results": results,
            "candidate_retrieval_latency_seconds": 0.03,
            "reranking_latency_seconds": 0.0,
            "reranking_enabled": False,
            "query_transformation_enabled": True,
            "query_transformation_strategy": "rewrite",
            "query_transformation_version": "phase8-llm-v1",
            "query_transformation_latency_seconds": 0.2,
            "original_query": question,
            "retrieval_query": "retrieval query",
            "retrieval_queries": ["retrieval query"],
            "generated_queries": ["retrieval query"],
            "protected_query_terms": ["Saturday"],
            "query_guard_triggered": False,
            "query_guard_repairs": [],
            "raw_query_transformation_output": "retrieval query",
            "multi_query_fusion": None,
            "query_count": 1,
        }

    def generate(self, question: str, results: list[tuple[Document, float]]) -> str:
        return "Answer [S1]."


class Phase8QueryTransformationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = replace(Settings.from_environment(PROJECT_ROOT), top_k=5)

    def test_protected_terms_include_people_dates_times_events_and_status(self) -> None:
        terms = extract_protected_terms(
            "Is the RSVP for friend_child_01 due Saturday, August 22 at 3:00 PM "
            "for the birthday party?"
        )

        self.assertEqual(
            terms,
            (
                "RSVP",
                "friend_child_01",
                "due",
                "Saturday",
                "August 22",
                "3:00 PM",
                "birthday party",
            ),
        )

    def test_multi_query_parser_accepts_json_and_numbered_fallback(self) -> None:
        self.assertEqual(
            parse_multi_query_output('["first query", "second query"]'),
            ["first query", "second query"],
        )
        self.assertEqual(
            parse_multi_query_output("1. first query\n2. second query"),
            ["first query", "second query"],
        )
        self.assertEqual(
            parse_multi_query_output(
                '```json\n["first query", "second query"]\n```\n'
                '```json\n["extra query", "another query"]\n```'
            ),
            ["first query", "second query"],
        )

    def test_original_strategy_does_not_call_llm(self) -> None:
        transformer = LLMQueryTransformer(
            "original",
            llm=None,
            reference_date="2026-08-20",
            timezone="America/Los_Angeles",
            generated_query_count=0,
        )

        transformed = transformer.transform("Original question")

        self.assertEqual(transformed.retrieval_queries, ("Original question",))
        self.assertEqual(transformed.generated_queries, ())
        self.assertEqual(transformed.guard_repairs, ())

    def test_rewrite_guard_restores_missing_protected_terms(self) -> None:
        llm = FakeLLM("upcoming preparation requirements and action items")
        transformer = LLMQueryTransformer(
            "rewrite",
            llm=llm,
            reference_date="2026-08-20",
            timezone="America/Los_Angeles",
            generated_query_count=1,
        )

        transformed = transformer.transform("What is due before Saturday?")

        self.assertEqual(llm.calls, 1)
        self.assertIn("due", transformed.retrieval_queries[0])
        self.assertIn("Saturday", transformed.retrieval_queries[0])
        self.assertEqual(len(transformed.guard_repairs), 1)
        self.assertEqual(
            transformed.guard_repairs[0]["missing_terms"],
            ["due", "Saturday"],
        )

    def test_rewrite_guard_removes_invented_dates_and_preserves_domain_terms(self) -> None:
        transformer = LLMQueryTransformer(
            "rewrite",
            llm=FakeLLM("October 2027 preparation requirements"),
            reference_date="2026-08-20",
            timezone="America/Los_Angeles",
            generated_query_count=1,
        )

        transformed = transformer.transform("What school events should we prepare for?")

        retrieval_query = transformed.retrieval_queries[0]
        self.assertNotIn("October", retrieval_query)
        self.assertNotIn("2027", retrieval_query)
        self.assertIn("school events", retrieval_query)
        self.assertEqual(
            transformed.guard_repairs[0]["reason"],
            "invented_fact_terms",
        )

    def test_multi_query_keeps_original_plus_two_generated_queries(self) -> None:
        transformer = LLMQueryTransformer(
            "multi_query",
            llm=FakeLLM('["preparation requirements", "deadlines and items to bring"]'),
            reference_date="2026-08-20",
            timezone="America/Los_Angeles",
            generated_query_count=2,
        )

        transformed = transformer.transform("What should I prepare?")

        self.assertEqual(len(transformed.retrieval_queries), 3)
        self.assertEqual(transformed.retrieval_queries[0], "What should I prepare?")
        self.assertEqual(len(transformed.generated_queries), 2)

    def test_rewrite_pipeline_searches_only_rewritten_query(self) -> None:
        rewritten = "course assignment requirements deadlines"
        store = FakeVectorStore({rewritten: [(document("learning_001"), 0.9)]})
        transformer = LLMQueryTransformer(
            "rewrite",
            llm=FakeLLM(rewritten),
            reference_date="2026-08-20",
            timezone="America/Los_Angeles",
            generated_query_count=1,
        )
        pipeline = QueryTransformationRAG(
            self.settings,
            store,  # type: ignore[arg-type]
            llm=object(),
            transformer=transformer,
        )

        trace = pipeline.retrieve_with_trace("What should I complete for my course?")

        self.assertEqual(store.calls, [rewritten])
        self.assertEqual(trace["retrieval_query"], rewritten)
        self.assertEqual(trace["original_query"], "What should I complete for my course?")
        self.assertEqual(trace["query_count"], 1)

    def test_multi_query_pipeline_fuses_results_with_rrf(self) -> None:
        original = "question"
        first_query = "first retrieval angle"
        second_query = "second retrieval angle"
        shared = document("shared")
        store = FakeVectorStore(
            {
                original: [(document("a"), 0.9), (shared, 0.8)],
                first_query: [(shared, 0.95), (document("b"), 0.7)],
                second_query: [(shared, 0.93), (document("c"), 0.6)],
            }
        )
        transformer = LLMQueryTransformer(
            "multi_query",
            llm=FakeLLM(f'["{first_query}", "{second_query}"]'),
            reference_date="2026-08-20",
            timezone="America/Los_Angeles",
            generated_query_count=2,
        )
        pipeline = QueryTransformationRAG(
            self.settings,
            store,  # type: ignore[arg-type]
            llm=object(),
            transformer=transformer,
            rrf_k=60,
        )

        trace = pipeline.retrieve_with_trace(original)
        serialized = serialize_retrieval(trace["results"])

        self.assertEqual(trace["query_count"], 3)
        self.assertEqual(trace["multi_query_fusion"], "rrf")
        self.assertEqual(serialized[0]["document_id"], "shared")
        self.assertEqual(
            len(serialized[0]["query_transformation_components"]["query_matches"]),
            3,
        )

    def test_evaluator_records_query_transformation_trace(self) -> None:
        question = {
            "question_id": "QTEST",
            "question": "Question",
            "expected_answer": "Answer",
            "expected_source_ids": ["source_001"],
            "expected_sources": ["Source 1"],
            "category": "exact_lookup",
            "answerable": True,
        }

        result = BaselineEvaluator(TracedQueryPipeline()).evaluate_question(question)

        self.assertTrue(result["query_transformation_enabled"])
        self.assertEqual(result["query_transformation_strategy"], "rewrite")
        self.assertEqual(result["retrieval_queries"], ["retrieval query"])
        self.assertEqual(result["query_transformation_latency_seconds"], 0.2)

    def test_configs_define_original_rewrite_and_multi_query_matrix(self) -> None:
        original, rewrite, multi_query = load_query_experiments(
            PROJECT_ROOT / "config" / "query_experiments"
        )

        self.assertEqual(original.query_strategy, "original")
        self.assertEqual(rewrite.generated_query_count, 1)
        self.assertEqual(multi_query.generated_query_count, 2)
        self.assertEqual(multi_query.fusion, "rrf")
        self.assertEqual(original.pinecone_namespace, multi_query.pinecone_namespace)

    def test_impact_records_questions_where_rewriting_reduces_quality(self) -> None:
        baseline = {
            "experiment_id": "baseline",
            "questions": [
                {
                    "question_id": "Q001",
                    "category": "semantic",
                    "recall_at_5": 1.0,
                    "expected_source_ranks": {"source_001": 1},
                }
            ],
        }
        rewritten = {
            "experiment_id": "rewrite",
            "questions": [
                {
                    "question_id": "Q001",
                    "category": "semantic",
                    "recall_at_5": 0.0,
                    "expected_source_ranks": {"source_001": None},
                    "retrieval_queries": ["bad rewrite"],
                    "generated_queries": ["bad rewrite"],
                }
            ],
        }

        impact = build_query_transformation_impact(baseline, [rewritten])

        strategy = impact["strategies"]["rewrite"]
        self.assertEqual(strategy["outcome_counts"]["degraded"], 1)
        self.assertEqual(strategy["quality_reduction_question_ids"], ["Q001"])
        self.assertEqual(strategy["recall_reduction_question_ids"], ["Q001"])

    def test_namespace_rebuild_refuses_non_phase8_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "Refusing to rebuild"):
            rebuild_phase8_query_namespace(
                object(),  # type: ignore[arg-type]
                "baseline",
                [],
            )


if __name__ == "__main__":
    unittest.main()
