from __future__ import annotations

import sys
import unittest
from pathlib import Path

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.evaluation import serialize_retrieval  # noqa: E402
from i_got_this_rag.retrieval import (  # noqa: E402
    BM25SparseRetriever,
    ReciprocalRankFusionRetriever,
    lexical_tokens,
)
from i_got_this_rag.retrieval_experiments import (  # noqa: E402
    REQUIRED_RETRIEVAL_STRATEGIES,
    load_retrieval_experiments,
    rebuild_phase5_dense_namespace,
)


def document(chunk_id: str, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "document_id": chunk_id.split("::")[0],
            "document_title": chunk_id,
            "source_path": f"data/{chunk_id}.md",
        },
    )


class FixedRetriever:
    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results

    def retrieve(self, question: str, k: int) -> list[tuple[Document, float]]:
        return self.results[:k]


class Phase5RetrievalTests(unittest.TestCase):
    def test_configs_define_exact_strategy_matrix(self) -> None:
        experiments = load_retrieval_experiments(
            PROJECT_ROOT / "config" / "retrieval_experiments"
        )

        self.assertEqual(
            tuple(experiment.retrieval_strategy for experiment in experiments),
            REQUIRED_RETRIEVAL_STRATEGIES,
        )
        self.assertEqual(len({experiment.pinecone_namespace for experiment in experiments}), 1)
        self.assertTrue(experiments[0].pinecone_namespace.startswith("phase5-"))

    def test_lexical_tokenizer_preserves_identifiers_and_normalizes_case(self) -> None:
        self.assertEqual(
            lexical_tokens("RSVP for child_02 on August 24!"),
            ["rsvp", "for", "child_02", "on", "august", "24"],
        )

    def test_bm25_prioritizes_exact_names_dates_and_terms(self) -> None:
        target = document(
            "social_001::chunk_000",
            "RSVP for the neighborhood potluck by August 24. Bring lemon bars.",
        )
        distractor = document(
            "learning_001::chunk_000",
            "Complete the semantic retrieval course assignment this weekend.",
        )
        retriever = BM25SparseRetriever([distractor, target])

        results = retriever.retrieve("Which RSVP is due August 24?", k=2)

        self.assertEqual(results[0][0].metadata["chunk_id"], "social_001::chunk_000")
        self.assertGreater(results[0][1], results[1][1])

    def test_hybrid_rrf_combines_dense_and_sparse_ranks(self) -> None:
        first = document("source_a::chunk_000", "A")
        shared = document("source_b::chunk_000", "B")
        third = document("source_c::chunk_000", "C")
        hybrid = ReciprocalRankFusionRetriever(
            FixedRetriever([(first, 0.9), (shared, 0.8)]),
            FixedRetriever([(shared, 5.0), (third, 4.0)]),
            rrf_k=60,
        )

        results = hybrid.retrieve("question", k=3)
        serialized = serialize_retrieval(results)

        self.assertEqual(results[0][0].metadata["chunk_id"], "source_b::chunk_000")
        self.assertEqual(
            serialized[0]["retrieval_components"],
            {"dense_rank": 2, "sparse_rank": 1},
        )
        self.assertGreater(results[0][1], results[1][1])

    def test_namespace_rebuild_refuses_non_phase5_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "Refusing to rebuild"):
            rebuild_phase5_dense_namespace(object(), "baseline", [])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

