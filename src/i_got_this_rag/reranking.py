from __future__ import annotations

from time import perf_counter
from typing import Any, Protocol

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from .baseline import generate_grounded_answer
from .retrieval import BM25SparseRetriever
from .settings import Settings


class CandidateReranker(Protocol):
    def rerank(
        self,
        question: str,
        candidates: list[tuple[Document, float]],
        top_k: int,
    ) -> list[tuple[Document, float]]: ...


class BM25CandidateReranker:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def rerank(
        self,
        question: str,
        candidates: list[tuple[Document, float]],
        top_k: int,
    ) -> list[tuple[Document, float]]:
        if not candidates:
            return []
        documents = [document for document, _ in candidates]
        reranker = BM25SparseRetriever(documents, k1=self.k1, b=self.b)
        reranker_scores = reranker.score(question)
        ranked = sorted(
            enumerate(zip(candidates, reranker_scores, strict=True), start=1),
            key=lambda item: (-item[1][1], item[0]),
        )[:top_k]

        results: list[tuple[Document, float]] = []
        for candidate_rank, ((document, dense_score), reranker_score) in ranked:
            reranked_document = Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "reranking_components": {
                        "candidate_rank": candidate_rank,
                        "dense_score": float(dense_score),
                        "bm25_reranker_score": float(reranker_score),
                    },
                },
            )
            results.append((reranked_document, float(reranker_score)))
        return results


class DenseRerankingRAG:
    def __init__(
        self,
        settings: Settings,
        vector_store: PineconeVectorStore,
        llm: Any,
        candidate_k: int,
        reranker: CandidateReranker | None = None,
    ) -> None:
        if candidate_k < settings.top_k:
            raise ValueError("candidate_k cannot be smaller than the final Top-K.")
        self.settings = settings
        self.vector_store = vector_store
        self.llm = llm
        self.candidate_k = candidate_k
        self.reranker = reranker

    def retrieve_with_trace(self, question: str) -> dict[str, Any]:
        candidate_started = perf_counter()
        candidates = self.vector_store.similarity_search_with_score(
            question,
            k=self.candidate_k,
        )
        candidate_latency = perf_counter() - candidate_started

        reranking_latency = 0.0
        if self.reranker is None:
            results = candidates[: self.settings.top_k]
        else:
            reranking_started = perf_counter()
            results = self.reranker.rerank(question, candidates, self.settings.top_k)
            reranking_latency = perf_counter() - reranking_started
        return {
            "results": results,
            "candidate_results": candidates,
            "candidate_retrieval_latency_seconds": candidate_latency,
            "reranking_latency_seconds": reranking_latency,
            "reranking_enabled": self.reranker is not None,
        }

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        return self.retrieve_with_trace(question)["results"]

    def generate(self, question: str, results: list[tuple[Document, float]]) -> str:
        return generate_grounded_answer(self.settings, self.llm, question, results)

