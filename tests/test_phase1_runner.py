from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.run_phase1 import (  # noqa: E402
    prepare_phase1_namespace,
    write_phase1_run,
)


class FakeIndex:
    def __init__(self, count: int) -> None:
        self.count = count
        self.delete_calls: list[dict[str, Any]] = []

    def describe_index_stats(self) -> SimpleNamespace:
        namespaces = (
            {"baseline": SimpleNamespace(vector_count=self.count)}
            if self.count
            else {}
        )
        return SimpleNamespace(namespaces=namespaces)

    def delete(self, **kwargs: Any) -> None:
        self.delete_calls.append(kwargs)
        self.count = 0


class FakeVectorStore:
    def __init__(self, *, index: FakeIndex, embedding: object, namespace: str) -> None:
        self.index = index
        self.embedding = embedding
        self.namespace = namespace
        self.added_ids: list[str] = []

    def add_documents(self, *, documents: list[Document], ids: list[str]) -> None:
        self.added_ids = ids
        self.index.count = len(documents)


def chunks() -> list[Document]:
    return [
        Document(
            page_content="first",
            metadata={"chunk_id": "doc_01::chunk_000", "document_id": "doc_01"},
        ),
        Document(
            page_content="second",
            metadata={"chunk_id": "doc_02::chunk_000", "document_id": "doc_02"},
        ),
    ]


class Phase1RunnerTests(unittest.TestCase):
    def resources(self, count: int) -> SimpleNamespace:
        return SimpleNamespace(pinecone_index=FakeIndex(count), embeddings=object())

    def test_existing_namespace_is_reused_by_default(self) -> None:
        resources = self.resources(2)

        store, indexing = prepare_phase1_namespace(
            resources,
            "baseline",
            chunks(),
            rebuild_namespace=False,
            vector_store_factory=FakeVectorStore,
        )

        self.assertEqual(indexing["action"], "reused_existing_namespace")
        self.assertEqual(indexing["indexed_vector_count"], 2)
        self.assertEqual(resources.pinecone_index.delete_calls, [])
        self.assertEqual(store.added_ids, [])

    def test_rebuild_deletes_only_configured_namespace_and_indexes_stable_ids(self) -> None:
        resources = self.resources(2)

        store, indexing = prepare_phase1_namespace(
            resources,
            "baseline",
            chunks(),
            rebuild_namespace=True,
            vector_store_factory=FakeVectorStore,
        )

        self.assertEqual(indexing["action"], "rebuilt")
        self.assertEqual(
            resources.pinecone_index.delete_calls,
            [{"delete_all": True, "namespace": "baseline"}],
        )
        self.assertEqual(len(store.added_ids), 2)
        self.assertEqual(len(set(store.added_ids)), 2)

    def test_empty_namespace_is_indexed_without_delete(self) -> None:
        resources = self.resources(0)

        store, indexing = prepare_phase1_namespace(
            resources,
            "baseline",
            chunks(),
            rebuild_namespace=False,
            vector_store_factory=FakeVectorStore,
        )

        self.assertEqual(indexing["action"], "indexed_empty_namespace")
        self.assertEqual(resources.pinecone_index.delete_calls, [])
        self.assertEqual(len(store.added_ids), 2)

    def test_result_writer_does_not_require_external_services(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            output = write_phase1_run(path, {"phase": 1, "answer": "grounded"})

            self.assertEqual(output, path.resolve())
            self.assertEqual(json.loads(path.read_text()), {"phase": 1, "answer": "grounded"})


if __name__ == "__main__":
    unittest.main()
