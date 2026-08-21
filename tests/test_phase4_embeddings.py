from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.embedding_experiments import (  # noqa: E402
    REQUIRED_EMBEDDING_MODELS,
    EmbeddingExperiment,
    load_embedding_experiments,
    rebuild_embedding_namespace,
    validate_index_compatibility,
)
from i_got_this_rag.evaluation import build_question_comparison  # noqa: E402
from i_got_this_rag.ingestion import (  # noqa: E402
    chunk_documents,
    chunk_fingerprint,
    load_corpus,
)


class Phase4EmbeddingTests(unittest.TestCase):
    def test_configs_define_exact_model_matrix_with_separate_indexes(self) -> None:
        experiments = load_embedding_experiments(
            PROJECT_ROOT / "config" / "embedding_experiments"
        )

        self.assertEqual(
            tuple(experiment.embedding_model for experiment in experiments),
            REQUIRED_EMBEDDING_MODELS,
        )
        self.assertEqual(len({experiment.pinecone_index_name for experiment in experiments}), 3)
        self.assertTrue(
            all(experiment.pinecone_namespace.startswith("phase4-") for experiment in experiments)
        )

    def test_selected_chunk_set_is_stable_and_shared(self) -> None:
        documents = load_corpus(PROJECT_ROOT / "data" / "sample", PROJECT_ROOT)
        first = chunk_fingerprint(chunk_documents(documents, 500, 75))
        second = chunk_fingerprint(chunk_documents(documents, 500, 75))

        self.assertEqual(first["chunk_count"], 20)
        self.assertEqual(first["sha256"], second["sha256"])

    def test_index_compatibility_rejects_dimension_or_metric_mismatch(self) -> None:
        compatible = SimpleNamespace(dimension=384, metric="cosine")
        validate_index_compatibility(compatible, 384, "compatible-index")

        with self.assertRaisesRegex(ValueError, "dimension"):
            validate_index_compatibility(compatible, 768, "wrong-dimension")
        with self.assertRaisesRegex(ValueError, "requires cosine"):
            validate_index_compatibility(
                SimpleNamespace(dimension=384, metric="dotproduct"),
                384,
                "wrong-metric",
            )

    def test_namespace_rebuild_refuses_non_phase4_target(self) -> None:
        unsafe_experiment = EmbeddingExperiment(
            experiment_id="unsafe",
            experiment_name="Unsafe",
            embedding_model="all-minilm",
            pinecone_index_name="unsafe-index",
            pinecone_namespace="baseline",
            config_path=Path("unsafe.yaml"),
            config_sha256="0" * 64,
        )

        with self.assertRaisesRegex(ValueError, "Refusing to rebuild"):
            rebuild_embedding_namespace(object(), unsafe_experiment, [])  # type: ignore[arg-type]

    def test_question_comparison_preserves_source_ranks_for_each_model(self) -> None:
        results = [
            {
                "experiment_id": "model_a",
                "questions": [
                    {
                        "question_id": "Q001",
                        "category": "exact_lookup",
                        "expected_source_ids": ["source_a"],
                        "recall_at_5": 1.0,
                        "expected_source_rank": 1,
                        "expected_source_ranks": {"source_a": 1},
                        "retrieval_latency_seconds": 0.1,
                    }
                ],
            },
            {
                "experiment_id": "model_b",
                "questions": [
                    {
                        "question_id": "Q001",
                        "category": "exact_lookup",
                        "expected_source_ids": ["source_a"],
                        "recall_at_5": 1.0,
                        "expected_source_rank": 3,
                        "expected_source_ranks": {"source_a": 3},
                        "retrieval_latency_seconds": 0.2,
                    }
                ],
            },
        ]

        comparison = build_question_comparison(results)

        self.assertEqual(comparison[0]["experiments"]["model_a"]["expected_source_rank"], 1)
        self.assertEqual(comparison[0]["experiments"]["model_b"]["expected_source_rank"], 3)


if __name__ == "__main__":
    unittest.main()

