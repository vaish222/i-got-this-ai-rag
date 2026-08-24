from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from langchain_core.documents import Document


ANSWER_ROUTING_CURRENT = "current"
ANSWER_ROUTING_SCOPED = "scoped"
ANSWER_ROUTING_SCOPED_REQUERY = "scoped_requery"
ANSWER_ROUTING_MODES = {
    ANSWER_ROUTING_CURRENT,
    ANSWER_ROUTING_SCOPED,
    ANSWER_ROUTING_SCOPED_REQUERY,
}

ScopeConfidence = Literal["high", "none"]


@dataclass(frozen=True)
class AnswerScope:
    intent: str
    domains: tuple[str, ...] = ()
    document_types: tuple[str, ...] = ()
    confidence: ScopeConfidence = "none"
    matched_phrase: str | None = None

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence == "high"

    def pinecone_filter(self) -> dict[str, object] | None:
        clauses: list[dict[str, object]] = []
        if self.domains:
            operator = "$eq" if len(self.domains) == 1 else "$in"
            value: str | list[str] = (
                self.domains[0] if len(self.domains) == 1 else list(self.domains)
            )
            clauses.append({"domain": {operator: value}})
        if self.document_types:
            operator = "$eq" if len(self.document_types) == 1 else "$in"
            value = (
                self.document_types[0]
                if len(self.document_types) == 1
                else list(self.document_types)
            )
            clauses.append({"document_type": {operator: value}})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def trace(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "domains": list(self.domains),
            "document_types": list(self.document_types),
            "confidence": self.confidence,
            "matched_phrase": self.matched_phrase,
            "metadata_filter": self.pinecone_filter(),
        }


@dataclass(frozen=True)
class _ScopeRule:
    intent: str
    pattern: re.Pattern[str]
    domains: tuple[str, ...]
    document_types: tuple[str, ...] = ()


SCOPE_RULES = (
    _ScopeRule(
        intent="meal_lookup",
        pattern=re.compile(
            r"\b(?:meal\s+(?:plan|prep|preparation|schedule)|menu|"
            r"dinner\s+(?:plan|menu|for)|"
            r"what\s+is\s+(?:the\s+)?dinner\s+(?:plan\s+)?(?:for|on)|"
            r"what(?:'|’)s\s+for\s+(?:breakfast|lunch|dinner)|"
            r"what\s+(?:are\s+we|should\s+we)\s+eat)\b",
            re.IGNORECASE,
        ),
        domains=("household",),
        document_types=("meal_plan",),
    ),
    _ScopeRule(
        intent="invitation_lookup",
        pattern=re.compile(r"\b(?:invitations?|rsvp(?:s|ed|ing)?)\b", re.IGNORECASE),
        domains=("social",),
        document_types=("invitation_tracker",),
    ),
    _ScopeRule(
        intent="gift_lookup",
        pattern=re.compile(r"\bgifts?\b", re.IGNORECASE),
        domains=("social",),
        document_types=("gift_tracker",),
    ),
    _ScopeRule(
        intent="social_event_lookup",
        pattern=re.compile(r"\bbirthdays?\b", re.IGNORECASE),
        domains=("social",),
    ),
    _ScopeRule(
        intent="volunteer_lookup",
        pattern=re.compile(
            r"\b(?:volunteer(?:ing)?|mentor(?:ing|ship)?|donat(?:e|ion))\b",
            re.IGNORECASE,
        ),
        domains=("volunteer",),
    ),
    _ScopeRule(
        intent="kids_activity_lookup",
        pattern=re.compile(
            r"\b(?:kids?'?\s+activit(?:y|ies)|children'?s?\s+activit(?:y|ies)|"
            r"swim(?:ming)?|piano|robotics?|watercolor|singing|taekwondo|"
            r"(?:math|reading)\s+class)\b",
            re.IGNORECASE,
        ),
        domains=("activities",),
        document_types=("activity_schedule",),
    ),
    _ScopeRule(
        intent="school_lookup",
        pattern=re.compile(
            r"\b(?:school|field[- ]trip|picture\s+day|teacher|student\s+council)\b",
            re.IGNORECASE,
        ),
        domains=("school",),
    ),
    _ScopeRule(
        intent="learning_lookup",
        pattern=re.compile(
            r"\b(?:course\s+(?:assignment|schedule|project|test|requirements?)|"
            r"certificate\s+requirements?|week\s+[235]\s+assignment)\b",
            re.IGNORECASE,
        ),
        domains=("learning",),
    ),
    _ScopeRule(
        intent="household_task_lookup",
        pattern=re.compile(
            r"\b(?:home\s+(?:task|repair|maintenance)|household\s+(?:task|chore)|"
            r"hvac|library\s+returns?|chores?)\b",
            re.IGNORECASE,
        ),
        domains=("household",),
    ),
    _ScopeRule(
        intent="family_schedule_lookup",
        pattern=re.compile(
            r"\b(?:family\s+(?:schedule|calendar|vacation|trip)|"
            r"long[- ]weekend\s+(?:vacation|trip)|vacation\s+plans?)\b",
            re.IGNORECASE,
        ),
        domains=("family",),
    ),
)


def detect_answer_scope(question: str) -> AnswerScope:
    for rule in SCOPE_RULES:
        match = rule.pattern.search(question)
        if match is not None:
            return AnswerScope(
                intent=rule.intent,
                domains=rule.domains,
                document_types=rule.document_types,
                confidence="high",
                matched_phrase=match.group(0),
            )
    return AnswerScope(intent="general")


def _metadata_domain(document: Document) -> str:
    direct = str(document.metadata.get("domain", "")).strip().casefold()
    if direct:
        return direct
    source_path = str(document.metadata.get("source_path", "")).casefold()
    match = re.search(r"(?:^|/)data/sample/([^/]+)/", source_path)
    if match is not None:
        return match.group(1)
    document_id = str(document.metadata.get("document_id", "")).casefold()
    return document_id.split("_", 1)[0]


def _metadata_document_type(document: Document) -> str:
    direct = str(document.metadata.get("document_type", "")).strip().casefold()
    if direct:
        return direct
    source_path = str(document.metadata.get("source_path", "")).casefold()
    if source_path.endswith("/household/meal_plan.md"):
        return "meal_plan"
    if source_path.endswith("/social/invitations.md"):
        return "invitation_tracker"
    if source_path.endswith("/social/birthdays_and_gifts.md"):
        return "gift_tracker"
    if "/activities/" in source_path:
        return "activity_schedule"
    return ""


def document_matches_scope(document: Document, scope: AnswerScope) -> bool:
    if not scope.is_high_confidence:
        return True
    if scope.domains and _metadata_domain(document) not in scope.domains:
        return False
    if (
        scope.document_types
        and _metadata_document_type(document) not in scope.document_types
    ):
        return False
    return True


def filter_results_to_scope(
    results: list[tuple[Document, float]],
    scope: AnswerScope,
) -> list[tuple[Document, float]]:
    if not scope.is_high_confidence:
        return results
    return [
        (document, score)
        for document, score in results
        if document_matches_scope(document, scope)
    ]
