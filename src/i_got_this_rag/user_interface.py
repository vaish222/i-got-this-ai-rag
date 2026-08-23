from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from langchain_core.documents import Document

from .agentic_rag import CitationAttributor
from .baseline import REFUSAL_TEXT
from .conversation import (
    ConversationRewrite,
    ConversationTurn,
)
from .evaluation import extract_citations, serialize_retrieval


ANONYMOUS_IDENTIFIER_PATTERN = re.compile(
    r"`?(?P<identifier>"
    r"(?:adult|child|friend|relative|neighbor|mentee|coordinator|colleague)"
    r"(?:_[a-z]+)*_\d{2})`?",
    re.IGNORECASE,
)
IDENTIFIER_ALIASES = {
    "adult_01": "one adult in your household",
    "adult_02": "another adult in your household",
    "child_01": "your middle-school child",
    "child_02": "your elementary-school child",
    "friend_child_01": "your friend's child",
    "friend_family_02": "your friends",
    "relative_01": "your relative",
}
ROLE_ALIASES = {
    "adult": "an adult in your household",
    "child": "your child",
    "friend": "your friend",
    "friend_child": "your friend's child",
    "friend_family": "your friends",
    "relative": "your relative",
    "neighbor": "your neighbor",
    "neighbor_group": "your neighbors",
    "mentee": "your mentee",
    "coordinator": "the coordinator",
    "colleague": "your colleague",
}
EVENT_HEADING_ALIASES = {
    "friend_family_02 dinner": "Dinner with your friends",
    "friend_child_01 birthday party": "Your friend's child's birthday party",
}
DATA_PREAMBLE_PATTERN = re.compile(
    r"^\s*(?:(?:according to|based on)\s+"
    r"(?:the\s+)?(?:provided\s+|retrieved\s+)?"
    r"(?:data|context|sources|records),?\s*)",
    re.IGNORECASE,
)
PENDING_RSVP_PATTERN = re.compile(
    r"\brsvp\b[^\n]{0,80}\bpending\b",
    re.IGNORECASE,
)
RSVP_QUESTION_PATTERN = re.compile(r"\brsvp(?:s|ed|ing)?\b", re.IGNORECASE)
MARKDOWN_SECTION_PATTERN = re.compile(
    r"^##\s+(?P<heading>.+?)\s*$\n(?P<body>.*?)(?=^##\s|\Z)",
    re.MULTILINE | re.DOTALL,
)
BARE_SECTION_ITEM_PATTERN = re.compile(
    r"^(?P<indent>\s*)(?P<bullet>[-*+])\s+"
    r"(?P<title>.+?)\s*\[(?P<label>S(?P<rank>\d+))\]\s*$",
    re.IGNORECASE,
)
SOURCE_BULLET_PATTERN = re.compile(r"^\s*[-*+]\s+(.+?)\s*$")


class QuestionAnsweringPipeline(Protocol):
    def retrieve(self, question: str) -> list[tuple[Document, float]]: ...

    def generate(
        self,
        question: str,
        results: list[tuple[Document, float]],
    ) -> str: ...


class FollowUpQueryRewriter(Protocol):
    def rewrite(
        self,
        question: str,
        history: Sequence[ConversationTurn],
    ) -> ConversationRewrite: ...


@dataclass(frozen=True)
class SourceView:
    label: str
    title: str
    source_path: str
    page_number: int | None


@dataclass(frozen=True)
class AnswerView:
    question: str
    retrieval_question: str
    answer: str
    sources: tuple[SourceView, ...]
    used_conversation_context: bool


@dataclass(frozen=True)
class PendingRSVPItem:
    title: str
    event: str | None
    deadline: str | None
    note: str | None
    source_label: str


def humanize_anonymous_identifiers(answer: str) -> str:
    def replace_identifier(match: re.Match[str]) -> str:
        identifier = match.group("identifier").casefold()
        direct_alias = IDENTIFIER_ALIASES.get(identifier)
        if direct_alias is not None:
            return direct_alias
        role, raw_number = identifier.rsplit("_", 1)
        role_alias = ROLE_ALIASES.get(role, f"the {role.replace('_', ' ')}")
        return f"{role_alias} {int(raw_number)}"

    return ANONYMOUS_IDENTIFIER_PATTERN.sub(replace_identifier, answer)


def remove_data_preamble(answer: str) -> str:
    cleaned = DATA_PREAMBLE_PATTERN.sub("", answer, count=1)
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def format_answer_for_display(answer: str) -> str:
    if answer.strip() == REFUSAL_TEXT:
        return answer
    return humanize_anonymous_identifiers(remove_data_preamble(answer))


def _normalized_heading(value: str) -> str:
    cleaned = _clean_markdown_value(value)
    cleaned = re.sub(r"^\*+|\*+$", "", cleaned).strip()
    return " ".join(cleaned.casefold().split())


def _section_bullets(document: Document, heading: str) -> tuple[str, ...] | None:
    target = _normalized_heading(heading)
    for section in MARKDOWN_SECTION_PATTERN.finditer(document.page_content):
        if _normalized_heading(section.group("heading")) != target:
            continue
        bullets = tuple(
            match.group(1).strip()
            for line in section.group("body").splitlines()
            if (match := SOURCE_BULLET_PATTERN.match(line))
        )
        return bullets
    return None


def expand_cited_section_headings(
    answer: str,
    results: list[tuple[Document, float]],
) -> str:
    expanded_lines: list[str] = []
    changed = False
    for line in answer.splitlines():
        match = BARE_SECTION_ITEM_PATTERN.match(line)
        if match is None:
            expanded_lines.append(line)
            continue

        rank = int(match.group("rank"))
        if not 1 <= rank <= len(results):
            expanded_lines.append(line)
            continue
        bullets = _section_bullets(results[rank - 1][0], match.group("title"))
        if bullets is None:
            expanded_lines.append(line)
            continue

        changed = True
        for bullet_text in bullets:
            citation = (
                ""
                if re.search(r"\[S\d+\]", bullet_text, re.IGNORECASE)
                else f" [{match.group('label').upper()}]"
            )
            expanded_lines.append(
                f"{match.group('indent')}{match.group('bullet')} "
                f"{bullet_text}{citation}"
            )

    expanded = "\n".join(expanded_lines).strip()
    if changed and not expanded:
        return REFUSAL_TEXT
    return expanded


def is_pending_rsvp_question(question: str) -> bool:
    lowered = question.casefold()
    return bool(RSVP_QUESTION_PATTERN.search(question)) and any(
        marker in lowered
        for marker in ("still", "pending", "need", "require")
    )


def _clean_markdown_value(value: str) -> str:
    return re.sub(r"\*\*|`", "", value).strip().rstrip(".")


def _section_field(body: str, label: str) -> str | None:
    match = re.search(
        rf"^\s*-\s*{re.escape(label)}:\s*(.+?)\s*$",
        body,
        re.IGNORECASE | re.MULTILINE,
    )
    return _clean_markdown_value(match.group(1)) if match else None


def _event_title(raw_heading: str) -> str:
    cleaned = _clean_markdown_value(raw_heading)
    alias = EVENT_HEADING_ALIASES.get(cleaned.casefold())
    if alias is not None:
        return alias
    humanized = humanize_anonymous_identifiers(cleaned)
    return humanized[0].upper() + humanized[1:] if humanized else humanized


def pending_rsvp_items(
    results: list[tuple[Document, float]],
) -> tuple[PendingRSVPItem, ...]:
    items: list[PendingRSVPItem] = []
    seen_titles: set[str] = set()
    for rank, (document, _) in enumerate(results, start=1):
        for section in MARKDOWN_SECTION_PATTERN.finditer(document.page_content):
            body = section.group("body")
            if not PENDING_RSVP_PATTERN.search(body):
                continue
            title = _event_title(section.group("heading"))
            title_key = title.casefold()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            items.append(
                PendingRSVPItem(
                    title=title,
                    event=_section_field(body, "Event"),
                    deadline=_section_field(body, "RSVP deadline"),
                    note=_section_field(body, "Note"),
                    source_label=f"S{rank}",
                )
            )
    return tuple(items)


def build_pending_rsvp_answer(
    results: list[tuple[Document, float]],
) -> str:
    items = pending_rsvp_items(results)
    if not items:
        return REFUSAL_TEXT

    count = len(items)
    opening = (
        "One invitation still needs a response:"
        if count == 1
        else f"{count} invitations still need a response:"
    )
    lines = [opening, ""]
    for item in items:
        details: list[str] = []
        if item.event:
            details.append(item.event)
        if item.deadline:
            details.append(f"RSVP by {item.deadline}")
        if item.note:
            note = item.note[0].upper() + item.note[1:] if item.note else item.note
            details.append(note)
        description = ". ".join(details)
        if description:
            description = f" — {description}."
        lines.append(f"- **{item.title}**{description} [{item.source_label}]")
    return "\n".join(lines)


def normalize_question(question: str) -> str:
    normalized = " ".join(question.split())
    if not normalized:
        raise ValueError("Enter a question before selecting Ask.")
    return normalized


def answer_question(
    pipeline: QuestionAnsweringPipeline,
    question: str,
    history: Sequence[ConversationTurn] = (),
    rewriter: FollowUpQueryRewriter | None = None,
) -> AnswerView:
    normalized = normalize_question(question)
    if history:
        if rewriter is None:
            raise ValueError("Conversation history requires a follow-up query rewriter.")
        rewrite = rewriter.rewrite(normalized, history)
    else:
        rewrite = ConversationRewrite(
            original_question=normalized,
            retrieval_question=normalized,
            used_history=False,
            raw_output=None,
            guard_repairs=(),
        )
    results = pipeline.retrieve(rewrite.retrieval_question)
    generated_answer = (
        build_pending_rsvp_answer(results)
        if is_pending_rsvp_question(rewrite.retrieval_question)
        else pipeline.generate(rewrite.retrieval_question, results)
    )
    answer = CitationAttributor().attribute(generated_answer, results)
    answer = expand_cited_section_headings(answer, results)
    answer = format_answer_for_display(answer)
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
        retrieval_question=rewrite.retrieval_question,
        answer=answer,
        sources=tuple(sources),
        used_conversation_context=rewrite.used_history,
    )
