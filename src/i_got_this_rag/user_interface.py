from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document

from .agentic_rag import CitationAttributor
from .evaluation import extract_citations, serialize_retrieval


class QuestionAnsweringPipeline(Protocol):
    def retrieve(self, question: str) -> list[tuple[Document, float]]: ...

    def generate(
        self,
        question: str,
        results: list[tuple[Document, float]],
    ) -> str: ...


@dataclass(frozen=True)
class SourceView:
    label: str
    title: str
    source_path: str
    page_number: int | None


@dataclass(frozen=True)
class AnswerView:
    question: str
    answer: str
    sources: tuple[SourceView, ...]


def normalize_question(question: str) -> str:
    normalized = " ".join(question.split())
    if not normalized:
        raise ValueError("Enter a question before selecting Ask.")
    return normalized


def answer_question(
    pipeline: QuestionAnsweringPipeline,
    question: str,
) -> AnswerView:
    normalized = normalize_question(question)
    results = pipeline.retrieve(normalized)
    generated_answer = pipeline.generate(normalized, results)
    answer = CitationAttributor().attribute(generated_answer, results)
    retrieved_chunks = serialize_retrieval(results)
    chunks_by_rank = {int(chunk["rank"]): chunk for chunk in retrieved_chunks}

    sources: list[SourceView] = []
    seen_chunks: set[str] = set()
    for citation in extract_citations(answer, retrieved_chunks):
        rank = citation.get("retrieval_rank")
        if rank is None:
            continue
        chunk = chunks_by_rank[int(rank)]
        chunk_key = str(chunk.get("chunk_id") or f"rank:{rank}")
        if chunk_key in seen_chunks:
            continue
        seen_chunks.add(chunk_key)
        page_number = chunk.get("page_number")
        sources.append(
            SourceView(
                label=str(citation["label"]),
                title=str(chunk.get("document_title") or "Untitled source"),
                source_path=str(chunk.get("source_path") or "Unknown source"),
                page_number=int(page_number) if page_number is not None else None,
            )
        )

    return AnswerView(
        question=normalized,
        answer=answer,
        sources=tuple(sources),
    )
