from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable

from .grounded_generation import REFUSAL_TEXT, extract_question_constraints
from .retrieval import lexical_tokens


CITATION_PATTERN = re.compile(r"\[S(\d+)\]", re.IGNORECASE)
DATE_PATTERN = re.compile(
    r"\b(?P<month>January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"\.?\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(?P<year>20\d{2}))?\b",
    re.IGNORECASE,
)
ISO_DATE_PATTERN = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
TIME_PATTERN = re.compile(r"\b(\d{1,2}):(\d{2})\s*(AM|PM)?\b", re.IGNORECASE)
MARKDOWN_PATTERN = re.compile(r"[`*_>#]")
TABLE_SEPARATOR_PATTERN = re.compile(r"^[\s|:-]+$")
ADVICE_PATTERN = re.compile(
    r"\b(should|recommend|consider|ensure|make sure|remember to|try to|prepare|bring)\b",
    re.IGNORECASE,
)
PRIORITY_PATTERN = re.compile(
    r"\b(priority|prioritize|most important|first|urgent|key task)\b",
    re.IGNORECASE,
)
STATUS_PATTERN = re.compile(
    r"\b(due|deadline|pending|completed|confirmed|cancelled|rsvp|still needs?)\b",
    re.IGNORECASE,
)
INFERENCE_PATTERN = re.compile(
    r"\b(therefore|because|likely|probably|this means|so that|which means)\b",
    re.IGNORECASE,
)
PEOPLE_PATTERN = re.compile(
    r"\b(?:adult|child|friend_?child|relative)_?\d+\b",
    re.IGNORECASE,
)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "will", "with", "your", "you", "one", "another", "am", "pm",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "date", "time", "person", "applies", "concerns", "occur", "occurs",
}
TOKEN_ALIASES = {
    "activities": "activity",
    "children": "child",
    "kids": "child",
    "starts": "start",
    "starting": "start",
    "scheduled": "schedule",
    "scheduling": "schedule",
    "deadlines": "deadline",
    "invitations": "invitation",
    "gifts": "gift",
    "classes": "class",
    "lessons": "lesson",
    "commitments": "commitment",
}
HUMAN_ROLE_REPLACEMENTS = {
    "your middle school child": "child_01",
    "your middle-school child": "child_01",
    "your elementary school child": "child_02",
    "your elementary-school child": "child_02",
    "one adult in your household": "adult_01",
    "another adult in your household": "adult_02",
    "your friend s child": "friend_child_01",
    "your friend's child": "friend_child_01",
}
DOMAIN_ALIASES = {
    "activity": "activities",
    "kids": "activities",
    "school": "school",
    "household": "household",
    "learning": "learning",
    "volunteer": "volunteer",
    "social": "social",
    "family": "family",
}


@dataclass(frozen=True)
class RetrievedContext:
    source_id: str
    document_id: str
    chunk_id: str
    title: str
    domain: str
    text: str

    @property
    def searchable_text(self) -> str:
        return f"{self.title}\n{self.text}"

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "domain": self.domain,
            "text": self.text,
        }


@dataclass(frozen=True)
class ClaimCandidate:
    text: str
    source_ids: tuple[str, ...]
    structured_item: dict[str, Any] | None = None
    field: str = "claim"


def _plain_text(value: str) -> str:
    value = CITATION_PATTERN.sub("", value)
    value = MARKDOWN_PATTERN.sub("", value)
    value = value.replace("–", "-").replace("—", "-").replace("’", "'")
    return " ".join(value.split()).strip(" -|:")


def _normalized(value: str) -> str:
    normalized = _plain_text(value).casefold()
    for source, target in HUMAN_ROLE_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _tokens(value: str) -> set[str]:
    return {
        TOKEN_ALIASES.get(token, token)
        for token in lexical_tokens(_normalized(value))
        if token not in STOP_WORDS and len(token) > 1 and not token.isdigit()
    }


def _date_facts(value: str) -> tuple[tuple[str, int, int | None], ...]:
    facts: list[tuple[str, int, int | None]] = []
    for match in DATE_PATTERN.finditer(value):
        month = match.group("month").casefold().rstrip(".")[:3]
        facts.append((month, int(match.group("day")), int(match.group("year")) if match.group("year") else None))
    for year, month, day in ISO_DATE_PATTERN.findall(value):
        month_name = (
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec",
        )[int(month) - 1]
        facts.append((month_name, int(day), int(year)))
    return tuple(dict.fromkeys(facts))


def _time_facts(value: str) -> tuple[tuple[int, int], ...]:
    facts: list[tuple[int, int]] = []
    for raw_hour, raw_minute, meridiem in TIME_PATTERN.findall(value):
        hour = int(raw_hour)
        if meridiem.casefold() == "pm" and hour < 12:
            hour += 12
        elif meridiem.casefold() == "am" and hour == 12:
            hour = 0
        facts.append((hour, int(raw_minute)))
    normalized = _normalized(value)
    if re.search(r"\bnoon\b", normalized):
        facts.append((12, 0))
    if re.search(r"\bmidnight\b", normalized):
        facts.append((0, 0))
    return tuple(dict.fromkeys(facts))


def _facts_supported(claim: str, source: str) -> tuple[bool, str | None]:
    source_dates = _date_facts(source)
    for month, day, year in _date_facts(claim):
        if not any(
            source_month == month
            and source_day == day
            and (
                year is None
                or source_year == year
                or (source_year is None and str(year) in source)
            )
            for source_month, source_day, source_year in source_dates
        ):
            return False, "date in the claim is not explicitly supported"
    source_times = set(_time_facts(source))
    for time_fact in _time_facts(claim):
        if time_fact not in source_times:
            return False, "time in the claim is not explicitly supported"
    return True, None


def _best_evidence(claim: str, source: str) -> str:
    segments = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+|\n+", source)
        if segment.strip()
    ]
    claim_tokens = _tokens(claim)
    if not segments:
        return source[:500]
    return max(
        segments,
        key=lambda segment: len(claim_tokens & _tokens(segment)),
    )[:500]


def _source_supports_claim(claim: str, source: str) -> tuple[bool, str]:
    facts_supported, fact_reason = _facts_supported(claim, source)
    if not facts_supported:
        return False, fact_reason or "concrete facts do not match"
    claim_tokens = _tokens(claim)
    source_tokens = _tokens(source)
    if not claim_tokens:
        return False, "claim has no auditable factual terms"
    coverage = len(claim_tokens & source_tokens) / len(claim_tokens)
    if coverage < 0.45:
        return False, f"only {coverage:.0%} of the claim's factual terms appear in the source"
    return True, f"concrete facts match and factual-term coverage is {coverage:.0%}"


def extract_claims(question_result: dict[str, Any]) -> tuple[ClaimCandidate, ...]:
    answer = str(question_result.get("generated_answer", "")).strip()
    if not answer or _normalized(answer) == _normalized(REFUSAL_TEXT):
        return ()
    trace = question_result.get("generation_trace")
    structured_items = trace.get("structured_items", []) if isinstance(trace, dict) else []
    if structured_items:
        atomic: list[ClaimCandidate] = []
        for item in structured_items:
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            source_ids = (str(item.get("source_id", "")).upper(),)
            atomic.append(ClaimCandidate(title, source_ids, item, "title"))
            item_date = str(item.get("date") or "").strip()
            if item_date and not _date_facts(title):
                atomic.append(
                    ClaimCandidate(
                        f"{title} occurs on {item_date}",
                        source_ids,
                        item,
                        "date",
                    )
                )
            item_time = str(item.get("time") or "").strip()
            if item_time and not _time_facts(title):
                atomic.append(
                    ClaimCandidate(
                        f"{title} occurs at {item_time}",
                        source_ids,
                        item,
                        "time",
                    )
                )
            person = str(item.get("person") or "").strip()
            if person:
                atomic.append(
                    ClaimCandidate(
                        f"{title} concerns {person}",
                        source_ids,
                        item,
                        "person",
                    )
                )
        return tuple(atomic)

    candidates: list[ClaimCandidate] = []
    table_headers: set[str] = set()
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line or TABLE_SEPARATOR_PATTERN.fullmatch(line):
            continue
        source_ids = tuple(f"S{value}" for value in CITATION_PATTERN.findall(line))
        plain = _plain_text(line)
        if not plain:
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = tuple(_plain_text(cell) for cell in line.strip("|").split("|"))
            if any(cell.casefold() in {"day", "date", "time", "dinner", "preparation note"} for cell in cells):
                table_headers.update(cell.casefold() for cell in cells)
                continue
            plain = "; ".join(cell for cell in cells if cell)
        elif line.startswith("#") or (line.startswith("**") and line.endswith("**")):
            continue
        elif re.match(r"^[-*+]\s+", line):
            plain = _plain_text(re.sub(r"^[-*+]\s+", "", line))
        elif not source_ids and not re.search(r"\d|\b(?:due|pending|scheduled|starts?|is|are|has|have)\b", plain, re.IGNORECASE):
            continue
        candidates.append(ClaimCandidate(plain, source_ids))

    unique: list[ClaimCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _normalized(candidate.text)
        if key and key not in seen:
            seen.add(key)
            unique.append(candidate)
    return tuple(unique)


def _claim_relevance(
    question: str,
    claim: ClaimCandidate,
    supporting_contexts: tuple[RetrievedContext, ...],
    reference_date: str,
) -> tuple[bool, str]:
    constraints = extract_question_constraints(question, reference_date)
    normalized_question = _normalized(question)
    broad_household_request = bool(
        re.search(
            r"\b(everyone|whole family|whole household|plan (?:my|our|the) week)\b",
            normalized_question,
        )
    )
    if constraints.domains and not broad_household_request:
        source_domains = {
            DOMAIN_ALIASES.get(context.domain.casefold(), context.domain.casefold())
            for context in supporting_contexts
            if context.domain
        }
        requested = {
            DOMAIN_ALIASES.get(domain.casefold(), domain.casefold())
            for domain in constraints.domains
        }
        if source_domains and source_domains.isdisjoint(requested):
            return False, "source is grounded but outside the requested domain/category"
    if constraints.people:
        claim_people = set(PEOPLE_PATTERN.findall(_normalized(claim.text)))
        if claim_people and claim_people.isdisjoint(set(constraints.people)):
            return False, "claim concerns a different person or group"
    return True, "claim satisfies the explicit question constraints"


def _unsupported_category(claim: str, reason: str) -> str:
    if "date" in reason or "time" in reason:
        return "incorrect date/time"
    if "person" in reason or "domain" in reason or "category" in reason:
        return "wrong person/category"
    if PRIORITY_PATTERN.search(claim):
        return "unsupported priority"
    if ADVICE_PATTERN.search(claim):
        return "unsupported advice"
    if STATUS_PATTERN.search(claim):
        return "unsupported status"
    if INFERENCE_PATTERN.search(claim):
        return "unsupported inference"
    if (
        "coverage" in reason
        or "terms" in reason
        or "evidence is not present" in reason
        or "no retrieved source" in reason
    ):
        return "invented detail"
    return "other"


def _person_supported(person: str, source: str) -> bool:
    normalized_person = _normalized(person)
    normalized_source = _normalized(source)
    if normalized_person in normalized_source:
        return True
    if normalized_person in {"children", "child"}:
        return bool(
            re.search(
                r"\b(?:child_?\d+|students?|children|child)\b",
                normalized_source,
            )
        )
    if normalized_person == "whole household":
        return "household" in normalized_source or "family" in normalized_source
    if normalized_person == "requesting adult":
        return bool(re.search(r"\badult_?\d+\b", normalized_source))
    return False


def audit_claim(
    claim: ClaimCandidate,
    contexts: tuple[RetrievedContext, ...],
    question: str,
    reference_date: str,
) -> dict[str, Any]:
    by_label = {context.source_id: context for context in contexts}
    cited_contexts = tuple(
        by_label[source_id]
        for source_id in claim.source_ids
        if source_id in by_label
    )
    # Faithfulness is support against the complete retrieved context, not citation
    # accuracy. Check cited chunks first for clearer evidence, then every other
    # retrieved chunk so a wrong citation label does not become a hallucination.
    candidate_contexts = cited_contexts + tuple(
        context for context in contexts if context not in cited_contexts
    )
    structured_evidence = ""
    if claim.structured_item:
        structured_evidence = str(claim.structured_item.get("evidence", "")).strip()

    supported_contexts: list[RetrievedContext] = []
    reasons: list[str] = []
    evidence = ""
    for context in candidate_contexts:
        support_text = context.searchable_text
        if structured_evidence:
            evidence_is_verbatim = (
                _normalized(structured_evidence) in _normalized(support_text)
            )
            item = claim.structured_item or {}
            person = str(item.get("person") or "").strip()
            if claim.field == "person" and person and not _person_supported(person, support_text):
                reasons.append(f"{context.source_id}: stated person is absent from the retrieved chunk")
                continue
            supported, reason = _source_supports_claim(claim.text, support_text)
            evidence = (
                structured_evidence[:500]
                if evidence_is_verbatim
                else _best_evidence(claim.text, support_text)
            )
            if supported and not evidence_is_verbatim:
                reason = (
                    f"{reason}; model-supplied evidence was not verbatim, but the "
                    "claim is independently explicit in the retrieved chunk"
                )
        else:
            supported, reason = _source_supports_claim(claim.text, support_text)
            evidence = _best_evidence(claim.text, support_text)
        if supported:
            supported_contexts.append(context)
            reasons.append(f"{context.source_id}: {reason}")
        else:
            reasons.append(f"{context.source_id}: {reason}")

    supported = bool(supported_contexts)
    relevant = True
    relevance_reason = "not evaluated because the claim is unsupported"
    category: str | None = None
    if supported:
        relevant, relevance_reason = _claim_relevance(
            question,
            claim,
            tuple(supported_contexts),
            reference_date,
        )
        if not relevant:
            category = "irrelevant but grounded information"
    else:
        category = _unsupported_category(claim.text, "; ".join(reasons))
    return {
        "claim": claim.text,
        "supported": supported,
        "supporting_source_ids": [context.source_id for context in supported_contexts],
        "supporting_document_ids": [context.document_id for context in supported_contexts],
        "supporting_evidence": evidence if supported else "",
        "reason": "; ".join(reasons) or "no retrieved source explicitly supports the claim",
        "relevant_to_question": relevant if supported else None,
        "relevance_reason": relevance_reason,
        "category": category,
    }


def audit_question(
    question_result: dict[str, Any],
    contexts: tuple[RetrievedContext, ...],
    model: str,
    reference_date: str,
) -> dict[str, Any]:
    claims = extract_claims(question_result)
    audited = [
        audit_claim(
            claim,
            contexts,
            str(question_result["question"]),
            reference_date,
        )
        for claim in claims
    ]
    supported_count = sum(bool(item["supported"]) for item in audited)
    total = len(audited)
    claim_score = supported_count / total if total else None
    automated = float(question_result["faithfulness"]["score"])
    disagreement = claim_score is not None and abs(automated - claim_score) >= 0.25
    return {
        "question_id": str(question_result["question_id"]),
        "question": str(question_result["question"]),
        "model": model,
        "retrieved_chunks": [context.to_dict() for context in contexts],
        "retrieved_source_ids": [context.source_id for context in contexts],
        "retrieved_document_ids": [context.document_id for context in contexts],
        "generated_answer": str(question_result["generated_answer"]),
        "automated_faithfulness": automated,
        "claims": audited,
        "total_factual_claims": total,
        "supported_factual_claims": supported_count,
        "unsupported_factual_claims": total - supported_count,
        "irrelevant_but_grounded_claims": sum(
            item["category"] == "irrelevant but grounded information"
            for item in audited
        ),
        "claim_faithfulness": claim_score,
        "no_factual_claims": total == 0,
        "evaluator_disagreement": disagreement,
        "disagreement_label": (
            "Evaluator disagreement — manual inspection recommended"
            if disagreement
            else None
        ),
    }


def summarize_model(
    experiment_result: dict[str, Any],
    audited_questions: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    questions = tuple(audited_questions)
    claims = tuple(claim for item in questions for claim in item["claims"])
    factual_claims = len(claims)
    supported = sum(bool(claim["supported"]) for claim in claims)
    unsupported = factual_claims - supported
    category_counts = {
        category: sum(claim.get("category") == category for claim in claims)
        for category in (
            "invented detail",
            "unsupported advice",
            "unsupported priority",
            "incorrect date/time",
            "wrong person/category",
            "unsupported status",
            "unsupported inference",
            "irrelevant but grounded information",
            "other",
        )
    }
    return {
        "experiment_id": str(experiment_result["experiment_id"]),
        "model": str(experiment_result["active_model"]["model"]),
        "existing_faithfulness": float(experiment_result["metrics"]["faithfulness"]),
        "claim_level_faithfulness": supported / factual_claims if factual_claims else None,
        "relevance_correctness": float(
            experiment_result["metrics"]["answer_relevance_correctness"]
        ),
        "total_factual_claims": factual_claims,
        "supported_factual_claims": supported,
        "unsupported_factual_claims": unsupported,
        "unsupported_claims_per_answer": unsupported / len(questions),
        "answers_with_no_factual_claims": sum(item["no_factual_claims"] for item in questions),
        "evaluator_disagreement_count": sum(item["evaluator_disagreement"] for item in questions),
        "category_counts": category_counts,
    }


def determine_conclusion(model_summaries: Iterable[dict[str, Any]]) -> dict[str, str]:
    summaries = tuple(model_summaries)
    evaluator_problem = any(item["evaluator_disagreement_count"] > 0 for item in summaries)
    generation_problem = any(item["unsupported_factual_claims"] > 0 for item in summaries)
    if evaluator_problem and generation_problem:
        return {
            "code": "C",
            "label": "Both",
            "reason": "The binary evaluator disagrees with supported claim-level findings, while the answers also contain genuinely unsupported claims.",
        }
    if evaluator_problem:
        return {
            "code": "B",
            "label": "Evaluator problem",
            "reason": "The audit finds grounded claims that the binary evaluator scores as unfaithful, without material unsupported claims.",
        }
    return {
        "code": "A",
        "label": "Generation problem",
        "reason": "The claim audit confirms unsupported factual claims and finds no material evaluator disagreement.",
    }
