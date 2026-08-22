from __future__ import annotations

import re
from dataclasses import dataclass, fields
from datetime import date, datetime, timedelta
from time import perf_counter
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from .baseline import generate_grounded_answer
from .settings import Settings


ANALYZER_VERSION = "phase7-rules-v1"

MONTH_NUMBERS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
MONTH_PATTERN = "|".join(sorted(MONTH_NUMBERS, key=len, reverse=True))
MONTH_DATE_PATTERN = re.compile(
    rf"\b({MONTH_PATTERN})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(20\d{{2}}))?\b",
    re.IGNORECASE,
)
ISO_DATE_PATTERN = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")

DOMAIN_PATTERNS = {
    "activities": (
        r"\bactivit(?:y|ies)\b",
        r"\bswim(?:ming)?\b",
        r"\brobotics?\b",
        r"\bwatercolor\b",
        r"\bpiano\b",
    ),
    "family": (r"\bfamily\b",),
    "household": (
        r"\bhousehold\b",
        r"\bhome\b",
        r"\bhvac\b",
        r"\bmeal\b",
        r"\bgrocer(?:y|ies)\b",
    ),
    "learning": (r"\bcourse\b", r"\bcertificate\b", r"\bcorpus audit\b"),
    "school": (
        r"\bschool\b",
        r"\bstudents?\b",
        r"\bfield[- ]trip\b",
        r"\bpicture day\b",
        r"\bgraduation\b",
    ),
    "social": (
        r"\bsocial\b",
        r"\binvitations?\b",
        r"\brsvp\b",
        r"\bbirthdays?\b",
        r"\bgifts?\b",
        r"\bdinner\b",
    ),
    "volunteer": (r"\bvolunteer(?:ing)?\b", r"\bmentor(?:ing)?\b"),
}

EVENT_TYPE_PATTERNS = {
    "birthday": (r"\bbirthdays?\b", r"\bbirthday party\b"),
    "dinner": (r"\bdinner\b",),
    "field_trip": (r"\bfield[- ]trip\b",),
    "graduation": (r"\bgraduation\b",),
    "picture_day": (r"\bpicture day\b",),
    "potluck": (r"\bpotluck\b",),
}

DOCUMENT_TYPE_PATTERNS = {
    "calendar": (r"\bevents? calendar\b",),
    "course_schedule": (r"\bcourse schedule\b",),
    "family_calendar": (r"\bfamily calendar\b",),
    "gift_tracker": (r"\bgift tracker\b",),
    "invitation_tracker": (r"\binvitations?\b",),
    "newsletter": (r"\bnewsletter\b", r"\bbulletin\b"),
}

BROAD_SCOPE_PATTERNS = (
    r"\bacross\b",
    r"\beveryone\b",
    r"\ball domains?\b",
)


def _matches(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item).strip().lower() for item in values if str(item).strip()]


def _facet_key(facet: str, value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return f"facet_{facet}__{normalized}"


def _default_year(metadata: dict[str, Any], fallback: int) -> int:
    updated_at = str(metadata.get("updated_at", ""))
    match = re.match(r"(20\d{2})", updated_at)
    return int(match.group(1)) if match else fallback


def extract_dates(text: str, default_year: int) -> tuple[str, ...]:
    dates: set[str] = set()
    for match in ISO_DATE_PATTERN.finditer(text):
        try:
            dates.add(
                date(
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                ).isoformat()
            )
        except ValueError:
            continue
    for match in MONTH_DATE_PATTERN.finditer(text):
        month = MONTH_NUMBERS[match.group(1).lower().rstrip(".")]
        year = int(match.group(3)) if match.group(3) else default_year
        try:
            dates.add(date(year, month, int(match.group(2))).isoformat())
        except ValueError:
            continue
    return tuple(sorted(dates))


def _event_types(text: str, tags: list[str]) -> tuple[str, ...]:
    combined = f"{text}\n{' '.join(tags)}"
    return tuple(
        event_type
        for event_type, patterns in EVENT_TYPE_PATTERNS.items()
        if _matches(combined, patterns)
    )


def _rsvp_statuses(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    if "rsvp" not in lowered:
        return ()
    statuses: set[str] = set()
    if re.search(r"rsvp[^\n.]{0,80}\bpending\b|\bpending\b[^\n.]{0,80}rsvp", lowered):
        statuses.add("pending")
    if re.search(
        r"rsvp[^\n.]{0,80}\b(?:completed|finished|confirmed)\b|"
        r"\b(?:completed|finished|confirmed)\b[^\n.]{0,80}rsvp",
        lowered,
    ):
        statuses.add("completed")
    return tuple(sorted(statuses))


def _gift_statuses(text: str, document_type: str) -> tuple[str, ...]:
    lowered = text.lower()
    if "gift" not in lowered and document_type != "gift_tracker":
        return ()
    status_text = (
        lowered
        if document_type == "gift_tracker"
        else "\n".join(line for line in lowered.splitlines() if "gift" in line)
    )
    statuses: set[str] = set()
    if re.search(r"\bneeded\b|\bnot (?:been )?purchased\b", status_text):
        statuses.add("needed")
    if re.search(r"\bpurchased\b|\bdelivered\b", status_text):
        statuses.add("purchased")
    if re.search(r"\bidea saved\b", status_text):
        statuses.add("idea_saved")
    return tuple(sorted(statuses))


def _general_statuses(text: str) -> tuple[str, ...]:
    statuses: set[str] = set()
    lowered = text.lower()
    for status, pattern in {
        "completed": r"\b(?:complete|completed|finished)\b",
        "incomplete": r"\bincomplete\b",
        "pending": r"\bpending\b",
        "scheduled": r"\bscheduled\b|\bon calendar\b",
    }.items():
        if re.search(pattern, lowered):
            statuses.add(status)
    return tuple(sorted(statuses))


def enrich_metadata_facets(chunks: list[Document], reference_date: str) -> list[Document]:
    """Return copies with normalized, filter-safe Phase 7 metadata facets."""
    fallback_year = datetime.strptime(reference_date, "%Y-%m-%d").year
    enriched: list[Document] = []
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        domain = str(metadata.get("domain", "")).strip().lower()
        document_type = str(metadata.get("document_type", "")).strip().lower()
        people = sorted(
            set(_string_values(metadata.get("person")))
            | set(_string_values(metadata.get("related_person")))
        )
        tags = sorted(set(_string_values(metadata.get("tags"))))
        event_types = _event_types(chunk.page_content, tags)
        event_dates = extract_dates(
            chunk.page_content,
            _default_year(metadata, fallback_year),
        )
        statuses = _general_statuses(chunk.page_content)
        rsvp_statuses = _rsvp_statuses(chunk.page_content)
        gift_statuses = _gift_statuses(chunk.page_content, document_type)

        metadata.update(
            {
                "metadata_facet_version": ANALYZER_VERSION,
                "facet_domain": domain,
                "facet_document_type": document_type,
                "facet_people": people,
                "facet_tags": tags,
                "facet_event_types": list(event_types),
                "facet_event_dates": list(event_dates),
                "facet_statuses": list(statuses),
                "facet_rsvp_statuses": list(rsvp_statuses),
                "facet_gift_statuses": list(gift_statuses),
            }
        )
        for facet, values in {
            "person": people,
            "tag": tags,
            "event_type": event_types,
            "event_date": event_dates,
            "status": statuses,
            "rsvp_status": rsvp_statuses,
            "gift_status": gift_statuses,
        }.items():
            for value in values:
                metadata[_facet_key(facet, value)] = True
        enriched.append(Document(page_content=chunk.page_content, metadata=metadata))
    return enriched


@dataclass(frozen=True)
class MetadataConstraints:
    domain: tuple[str, ...] = ()
    person: tuple[str, ...] = ()
    document_type: tuple[str, ...] = ()
    event_type: tuple[str, ...] = ()
    event_date: tuple[str, ...] = ()
    status: tuple[str, ...] = ()
    rsvp_status: tuple[str, ...] = ()
    gift_status: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list[str]]:
        return {
            item.name: list(getattr(self, item.name))
            for item in fields(self)
            if getattr(self, item.name)
        }

    def to_pinecone_filter(self) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if self.domain:
            operator = "$eq" if len(self.domain) == 1 else "$in"
            value: str | list[str] = (
                self.domain[0] if len(self.domain) == 1 else list(self.domain)
            )
            clauses.append({"facet_domain": {operator: value}})
        if self.document_type:
            operator = "$eq" if len(self.document_type) == 1 else "$in"
            value = (
                self.document_type[0]
                if len(self.document_type) == 1
                else list(self.document_type)
            )
            clauses.append({"facet_document_type": {operator: value}})

        for facet in (
            "person",
            "event_type",
            "event_date",
            "status",
            "rsvp_status",
            "gift_status",
        ):
            values = getattr(self, facet)
            if not values:
                continue
            alternatives = [{_facet_key(facet, item): {"$eq": True}} for item in values]
            clauses.append(
                alternatives[0] if len(alternatives) == 1 else {"$or": alternatives}
            )

        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class MetadataQueryAnalyzer:
    """Deterministic constraint extraction; it never rewrites the semantic query."""

    def __init__(self, reference_date: str) -> None:
        self.reference_date = datetime.strptime(reference_date, "%Y-%m-%d").date()

    def analyze(self, question: str) -> MetadataConstraints:
        lowered = question.lower()
        broad_scope = _matches(lowered, BROAD_SCOPE_PATTERNS)
        domains = () if broad_scope else self._domains(lowered)
        people = tuple(
            sorted(set(re.findall(r"\b[a-z]+(?:_[a-z]+)*_\d{2}\b", lowered)))
        )
        document_types = tuple(
            document_type
            for document_type, patterns in DOCUMENT_TYPE_PATTERNS.items()
            if _matches(lowered, patterns)
        )
        event_types = tuple(
            event_type
            for event_type, patterns in EVENT_TYPE_PATTERNS.items()
            if _matches(lowered, patterns)
        )
        event_dates = self._query_dates(lowered)

        rsvp_status: tuple[str, ...] = ()
        if "rsvp" in lowered:
            if re.search(r"\b(?:pending|still need|need(?:s)?|require)\b", lowered):
                rsvp_status = ("pending",)
            elif re.search(r"\b(?:complete|completed|finished|confirmed)\b", lowered):
                rsvp_status = ("completed",)

        gift_status: tuple[str, ...] = ()
        if "gift" in lowered:
            if re.search(r"\b(?:needed|still need|need to buy|not purchased)\b", lowered):
                gift_status = ("needed",)
            elif re.search(r"\b(?:purchased|bought|delivered)\b", lowered):
                gift_status = ("purchased",)

        status: tuple[str, ...] = ()
        if not rsvp_status and not gift_status:
            for value in ("incomplete", "pending", "completed", "scheduled"):
                if re.search(rf"\b{value}\b", lowered):
                    status = (value,)
                    break

        return MetadataConstraints(
            domain=domains,
            person=people,
            document_type=document_types,
            event_type=event_types,
            event_date=event_dates,
            status=status,
            rsvp_status=rsvp_status,
            gift_status=gift_status,
        )

    def _domains(self, question: str) -> tuple[str, ...]:
        domains = {
            domain
            for domain, patterns in DOMAIN_PATTERNS.items()
            if _matches(question, patterns)
        }
        if "potluck" in question:
            domains.update({"household", "social"})
        return tuple(sorted(domains))

    def _query_dates(self, question: str) -> tuple[str, ...]:
        dates = set(extract_dates(question, self.reference_date.year))
        for weekday, target_weekday in {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }.items():
            if re.search(rf"\b{weekday}\b", question):
                offset = (target_weekday - self.reference_date.weekday()) % 7
                dates.add((self.reference_date + timedelta(days=offset)).isoformat())
        return tuple(sorted(dates))


class MetadataAwareDenseRAG:
    def __init__(
        self,
        settings: Settings,
        vector_store: PineconeVectorStore,
        llm: Any,
        analyzer: MetadataQueryAnalyzer,
        metadata_filter_enabled: bool,
        fallback_to_unfiltered: bool = True,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.llm = llm
        self.analyzer = analyzer
        self.metadata_filter_enabled = metadata_filter_enabled
        self.fallback_to_unfiltered = fallback_to_unfiltered

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        return self.retrieve_with_trace(question)["results"]

    def retrieve_with_trace(self, question: str) -> dict[str, Any]:
        analysis_started = perf_counter()
        constraints = (
            self.analyzer.analyze(question)
            if self.metadata_filter_enabled
            else MetadataConstraints()
        )
        metadata_filter = constraints.to_pinecone_filter()
        analysis_latency = perf_counter() - analysis_started

        retrieval_started = perf_counter()
        filtered_results: list[tuple[Document, float]] = []
        fallback_results: list[tuple[Document, float]] = []
        if metadata_filter is not None:
            filtered_results = self.vector_store.similarity_search_with_score(
                question,
                k=self.settings.top_k,
                filter=metadata_filter,
            )
            if self.fallback_to_unfiltered and len(filtered_results) < self.settings.top_k:
                fallback_results = self.vector_store.similarity_search_with_score(
                    question,
                    k=self.settings.top_k,
                )
        else:
            fallback_results = self.vector_store.similarity_search_with_score(
                question,
                k=self.settings.top_k,
            )

        combined = self._combine(filtered_results, fallback_results)
        retrieval_latency = perf_counter() - retrieval_started
        return {
            "results": combined,
            "candidate_results": combined,
            "candidate_retrieval_latency_seconds": retrieval_latency,
            "reranking_latency_seconds": 0.0,
            "reranking_enabled": False,
            "metadata_filter_enabled": self.metadata_filter_enabled,
            "metadata_filter_applied": metadata_filter is not None,
            "metadata_constraints": constraints.to_dict(),
            "metadata_filter": metadata_filter,
            "metadata_analysis_latency_seconds": analysis_latency,
            "metadata_filtered_result_count": len(filtered_results),
            "metadata_fallback_result_count": (
                max(0, len(combined) - len(filtered_results))
                if metadata_filter is not None
                else 0
            ),
            "retrieval_query": question,
        }

    def _combine(
        self,
        filtered_results: list[tuple[Document, float]],
        fallback_results: list[tuple[Document, float]],
    ) -> list[tuple[Document, float]]:
        combined: list[tuple[Document, float]] = []
        seen: set[str] = set()
        for origin, results in (
            ("filtered", filtered_results),
            ("dense_fallback", fallback_results),
        ):
            for document, score in results:
                chunk_id = str(document.metadata.get("chunk_id", document.page_content))
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                metadata = dict(document.metadata)
                metadata["metadata_retrieval_components"] = {"origin": origin}
                combined.append(
                    (Document(page_content=document.page_content, metadata=metadata), score)
                )
                if len(combined) == self.settings.top_k:
                    return combined
        return combined

    def generate(self, question: str, results: list[tuple[Document, float]]) -> str:
        return generate_grounded_answer(self.settings, self.llm, question, results)
