from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.chunk_experiments import (  # noqa: E402
    REQUIRED_CHUNK_SIZES,
    ChunkExperiment,
    load_chunk_experiments,
    rebuild_experiment_namespace,
)
from i_got_this_rag.evaluation import summarize_by_category  # noqa: E402
from i_got_this_rag.ingestion import (  # noqa: E402
    chunk_documents,
    corpus_fingerprint,
    load_corpus,
)


class Phase3ChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = PROJECT_ROOT / "data" / "sample"
        cls.documents = load_corpus(cls.data_dir, PROJECT_ROOT)

    def test_configs_define_exact_controlled_matrix(self) -> None:
        experiments = load_chunk_experiments(
            PROJECT_ROOT / "config" / "experiments",
            baseline_namespace="baseline",
        )

        self.assertEqual(tuple(experiment.chunk_size for experiment in experiments), REQUIRED_CHUNK_SIZES)
        self.assertEqual({experiment.chunk_overlap for experiment in experiments}, {75})
        self.assertEqual(len({experiment.pinecone_namespace for experiment in experiments}), 4)
        self.assertTrue(
            all(experiment.pinecone_namespace.startswith("phase3-") for experiment in experiments)
        )

    def test_controlled_corpus_has_20_stable_source_files(self) -> None:
        fingerprint = corpus_fingerprint(self.data_dir, PROJECT_ROOT)

        self.assertEqual(fingerprint["document_count"], 20)
        self.assertEqual(len(fingerprint["files"]), 20)
        self.assertEqual(len(fingerprint["sha256"]), 64)

    def test_chunking_preserves_source_metadata_and_records_configuration(self) -> None:
        small_chunks = chunk_documents(self.documents, chunk_size=250, chunk_overlap=75)
        large_chunks = chunk_documents(self.documents, chunk_size=1000, chunk_overlap=75)

        self.assertGreaterEqual(len(small_chunks), len(large_chunks))
        self.assertTrue(all(chunk.metadata.get("document_id") for chunk in small_chunks))
        self.assertTrue(all(chunk.metadata.get("chunk_id") for chunk in small_chunks))
        self.assertTrue(all(chunk.metadata["chunk_size"] == 250 for chunk in small_chunks))
        self.assertTrue(all(chunk.metadata["chunk_overlap"] == 75 for chunk in small_chunks))

    def test_namespace_rebuild_refuses_non_phase3_target(self) -> None:
        unsafe_experiment = ChunkExperiment(
            experiment_id="unsafe",
            experiment_name="Unsafe",
            chunk_size=250,
            chunk_overlap=75,
            pinecone_namespace="baseline",
            config_path=Path("unsafe.yaml"),
            config_sha256="0" * 64,
        )

        with self.assertRaisesRegex(ValueError, "Refusing to rebuild"):
            rebuild_experiment_namespace(object(), unsafe_experiment, [])  # type: ignore[arg-type]

    def test_category_summary_identifies_retrieval_failure_patterns(self) -> None:
        base_latency = {
            "retrieval_latency_seconds": 0.1,
            "llm_latency_seconds": 0.2,
            "total_latency_seconds": 0.3,
        }
        results = [
            {
                "question_id": "Q001",
                "category": "exact_lookup",
                "recall_at_5": 1.0,
                "expected_source_ranks": {"source_a": 1},
                **base_latency,
            },
            {
                "question_id": "Q002",
                "category": "cross_domain",
                "recall_at_5": 0.5,
                "expected_source_ranks": {"source_a": 2, "source_b": None},
                **base_latency,
            },
            {
                "question_id": "Q003",
                "category": "unanswerable",
                "recall_at_5": None,
                "expected_source_ranks": {},
                **base_latency,
            },
        ]

        summary = summarize_by_category(results)

        self.assertEqual(summary["exact_lookup"]["retrieval_failure_count"], 0)
        self.assertEqual(summary["cross_domain"]["retrieval_failure_count"], 1)
        self.assertEqual(
            summary["cross_domain"]["failures"][0]["missing_expected_source_ids"],
            ["source_b"],
        )
        self.assertIsNone(summary["unanswerable"]["recall_at_5"])


if __name__ == "__main__":
    unittest.main()

