from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from .baseline import generate_grounded_answer
from .settings import Settings


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")


def lexical_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class Retriever(Protocol):
    def retrieve(self, question: str, k: int) -> list[tuple[Document, float]]: ...


class DenseRetriever:
    def __init__(self, vector_store: PineconeVectorStore) -> None:
        self.vector_store = vector_store

    def retrieve(self, question: str, k: int) -> list[tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(question, k=k)


class BM25SparseRetriever:
    def __init__(
        self,
        documents: list[Document],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("BM25 requires at least one document.")
        if k1 <= 0:
            raise ValueError("BM25 k1 must be positive.")
        if not 0 <= b <= 1:
            raise ValueError("BM25 b must be between 0 and 1.")
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(lexical_tokens(document.page_content)) for document in documents]
        self.document_lengths = [sum(frequencies.values()) for frequencies in self.term_frequencies]
        self.average_document_length = sum(self.document_lengths) / len(self.document_lengths)

        document_frequencies: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            document_frequencies.update(frequencies.keys())
        document_count = len(documents)
        self.inverse_document_frequencies = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def score(self, question: str) -> list[float]:
        query_terms = Counter(lexical_tokens(question))
        scores: list[float] = []
        for frequencies, document_length in zip(
            self.term_frequencies,
            self.document_lengths,
            strict=True,
        ):
            score = 0.0
            length_normalizer = 1 - self.b + self.b * (
                document_length / self.average_document_length
            )
            for term, query_frequency in query_terms.items():
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                numerator = term_frequency * (self.k1 + 1)
                denominator = term_frequency + self.k1 * length_normalizer
                score += (
                    self.inverse_document_frequencies.get(term, 0.0)
                    * numerator
                    / denominator
                    * query_frequency
                )
            scores.append(score)
        return scores

    def retrieve(self, question: str, k: int) -> list[tuple[Document, float]]:
        scores = self.score(question)
        ranked = sorted(
            zip(self.documents, scores, strict=True),
            key=lambda item: (-item[1], str(item[0].metadata.get("chunk_id", ""))),
        )
        return [(document, float(score)) for document, score in ranked[:k]]


class ReciprocalRankFusionRetriever:
    def __init__(
        self,
        dense_retriever: Retriever,
        sparse_retriever: Retriever,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("RRF k must be positive.")
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k

    def retrieve(self, question: str, k: int) -> list[tuple[Document, float]]:
        dense_results = self.dense_retriever.retrieve(question, k)
        sparse_results = self.sparse_retriever.retrieve(question, k)
        documents: dict[str, Document] = {}
        fused_scores: Counter[str] = Counter()
        component_ranks: dict[str, dict[str, int]] = {}

        for strategy, results in (("dense", dense_results), ("sparse", sparse_results)):
            for rank, (document, _) in enumerate(results, start=1):
                chunk_id = str(document.metadata["chunk_id"])
                documents.setdefault(chunk_id, document)
                fused_scores[chunk_id] += 1 / (self.rrf_k + rank)
                component_ranks.setdefault(chunk_id, {})[f"{strategy}_rank"] = rank

        ranked_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id),
        )[:k]
        fused_results: list[tuple[Document, float]] = []
        for chunk_id in ranked_ids:
            document = documents[chunk_id]
            fused_document = Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "retrieval_components": component_ranks[chunk_id],
                },
            )
            fused_results.append((fused_document, float(fused_scores[chunk_id])))
        return fused_results


class ConfigurableRetrievalRAG:
    def __init__(self, settings: Settings, retriever: Retriever, llm: object) -> None:
        self.settings = settings
        self.retriever = retriever
        self.llm = llm

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        return self.retriever.retrieve(question, self.settings.top_k)

    def generate(self, question: str, results: list[tuple[Document, float]]) -> str:
        return generate_grounded_answer(self.settings, self.llm, question, results)

