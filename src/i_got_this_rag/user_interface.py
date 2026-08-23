from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol, Sequence

from langchain_core.documents import Document

from .agentic_rag import CitationAttributor
from .baseline import REFUSAL_TEXT
from .conversation import (
    ADDITIVE_FOLLOW_UP_PATTERN,
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
VOLUNTEER_QUESTION_PATTERN = re.compile(
    r"\b(?:volunteer(?:ing)?|mentor(?:ing)?)\b",
    re.IGNORECASE,
)
THIS_WEEK_PATTERN = re.compile(r"\bthis week\b", re.IGNORECASE)
WEEKDAY_PATTERN = re.compile(
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    re.IGNORECASE,
)
VOLUNTEER_WEEK_ACTION_PATTERN = re.compile(
    r"\b(?:due|session|comments?|bring|enter|welcome table|cover|volunteer)\b",
    re.IGNORECASE,
)
ISO_DATE_PATTERN = re.compile(r"\b(?P<year>20\d{2})-(?P<month>\d{2})-(?P<day>\d{2})\b")
MONTH_DATE_PATTERN = re.compile(
    r"\b(?P<month>January|February|March|April|May|June|July|August|"
    r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|"
    r"Sep|Sept|Oct|Nov|Dec)\.?\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?"
    r"(?:,?\s+(?P<year>20\d{2}))?\b",
    re.IGNORECASE,
)
MONTH_NUMBERS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


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


def _reference_date(value: str | date | None) -> date | None:
    if isinstance(value, date):
        return value
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _explicit_dates(text: str, reference: date) -> tuple[date, ...]:
    found: list[date] = []
    occupied: list[tuple[int, int]] = []
    for match in ISO_DATE_PATTERN.finditer(text):
        try:
            found.append(
                date(
                    int(match.group("year")),
                    int(match.group("month")),
                    int(match.group("day")),
                )
            )
            occupied.append(match.span())
        except ValueError:
            continue

    for match in MONTH_DATE_PATTERN.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in occupied):
            continue
        month = MONTH_NUMBERS[match.group("month").casefold()]
        year = int(match.group("year") or reference.year)
        try:
            found.append(date(year, month, int(match.group("day"))))
        except ValueError:
            continue
    return tuple(dict.fromkeys(found))


def _week_bounds(reference: date) -> tuple[date, date]:
    start = reference - timedelta(days=reference.weekday())
    return start, start + timedelta(days=6)


def _filter_document_to_week(document: Document, reference: date) -> Document | None:
    start, end = _week_bounds(reference)
    blocks = re.split(r"\n\s*\n", document.page_content)
    kept_blocks: list[str] = []
    for block in blocks:
        dates = _explicit_dates(block, reference)
        if dates and not any(start <= value <= end for value in dates):
            continue
        if block.strip():
            kept_blocks.append(block.strip())
    if not kept_blocks:
        return None
    return Document(
        page_content="\n\n".join(kept_blocks),
        metadata=dict(document.metadata),
    )


def select_relevant_ui_results(
    question: str,
    results: list[tuple[Document, float]],
    reference_date: str | date | None = None,
) -> list[tuple[Document, float]]:
    selected = results
    if VOLUNTEER_QUESTION_PATTERN.search(question):
        volunteer_results = [
            (document, score)
            for document, score in results
            if str(document.metadata.get("domain", "")).casefold() == "volunteer"
            or str(document.metadata.get("document_id", "")).casefold().startswith(
                "volunteer_"
            )
        ]
        if volunteer_results:
            selected = volunteer_results

    reference = _reference_date(reference_date)
    if reference is None or not THIS_WEEK_PATTERN.search(question):
        return selected

    filtered: list[tuple[Document, float]] = []
    for document, score in selected:
        narrowed = _filter_document_to_week(document, reference)
        if narrowed is not None:
            filtered.append((narrowed, score))
    return filtered or selected


def filter_answer_to_current_week(
    answer: str,
    question: str,
    reference_date: str | date | None,
) -> str:
    reference = _reference_date(reference_date)
    if reference is None or not THIS_WEEK_PATTERN.search(question):
        return answer

    start, end = _week_bounds(reference)
    kept_lines: list[str] = []
    for line in answer.splitlines():
        dates = _explicit_dates(line, reference)
        if dates and not any(start <= value <= end for value in dates):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()


def is_volunteer_week_question(question: str) -> bool:
    return bool(
        VOLUNTEER_QUESTION_PATTERN.search(question)
        and THIS_WEEK_PATTERN.search(question)
    )


def _volunteer_week_items(
    results: list[tuple[Document, float]],
    reference_date: str | date | None,
) -> tuple[tuple[str, str], ...]:
    reference = _reference_date(reference_date)
    if reference is None:
        return ()
    start, end = _week_bounds(reference)
    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for rank, (document, _) in enumerate(results, start=1):
        domain = str(document.metadata.get("domain", "")).casefold()
        document_id = str(document.metadata.get("document_id", "")).casefold()
        if domain != "volunteer" and not document_id.startswith("volunteer_"):
            continue
        for raw_line in document.page_content.splitlines():
            line = raw_line.strip().removeprefix("-").strip()
            if not line or not VOLUNTEER_WEEK_ACTION_PATTERN.search(line):
                continue
            dates = _explicit_dates(line, reference)
            has_in_week_date = any(start <= value <= end for value in dates)
            if not has_in_week_date and not WEEKDAY_PATTERN.search(line):
                continue
            lowered = line.casefold()
            if "optional" in lowered or ("complete" in lowered and "pending" not in lowered):
                continue
            cleaned = _clean_markdown_value(line)
            normalized = re.sub(r"\W+", " ", cleaned.casefold()).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            items.append((cleaned, f"S{rank}"))
    return tuple(items)


def _display_normalized(value: str) -> str:
    readable = humanize_anonymous_identifiers(_clean_markdown_value(value))
    return re.sub(r"\W+", " ", readable.casefold()).strip()


def build_volunteer_week_answer(
    question: str,
    results: list[tuple[Document, float]],
    history: Sequence[ConversationTurn],
    reference_date: str | date | None,
) -> str:
    items = _volunteer_week_items(results, reference_date)
    if not items:
        return REFUSAL_TEXT

    is_additive = bool(ADDITIVE_FOLLOW_UP_PATTERN.search(question))
    if is_additive and history:
        previous_answer = _display_normalized(history[-1].assistant_answer)
        items = tuple(
            (item, label)
            for item, label in items
            if _display_normalized(item) not in previous_answer
        )
        if not items:
            labels = "".join(
                f"[S{rank}]" for rank in range(1, len(results) + 1)
            )
            return (
                "I couldn't find any additional volunteer work due this week "
                f"beyond what was already listed. {labels}"
            ).strip()

    label = "other volunteer commitments" if is_additive else "volunteer commitments"
    lines = [f"{len(items)} {label} are scheduled or due this week:"]
    lines.extend(f"- {item} [{source_label}]" for item, source_label in items)
    return "\n".join(lines)


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
    reference_date: str | date | None = None,
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
    if reference_date is None:
        reference_date = getattr(
            getattr(pipeline, "settings", None),
            "reference_date",
            None,
        )
    results = select_relevant_ui_results(
        rewrite.retrieval_question,
        results,
        reference_date,
    )
    if is_pending_rsvp_question(rewrite.retrieval_question):
        generated_answer = build_pending_rsvp_answer(results)
    elif is_volunteer_week_question(rewrite.retrieval_question):
        generated_answer = build_volunteer_week_answer(
            normalized,
            results,
            history,
            reference_date,
        )
    else:
        generated_answer = pipeline.generate(rewrite.retrieval_question, results)
    answer = CitationAttributor().attribute(generated_answer, results)
    answer = expand_cited_section_headings(answer, results)
    answer = filter_answer_to_current_week(
        answer,
        rewrite.retrieval_question,
        reference_date,
    )
    retrieved_chunks = serialize_retrieval(results)
    resolved_citations = [
        citation
        for citation in extract_citations(answer, retrieved_chunks)
        if citation.get("resolved")
    ]
    if not answer or (answer.strip() != REFUSAL_TEXT and not resolved_citations):
        answer = REFUSAL_TEXT
    answer = format_answer_for_display(answer)
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
