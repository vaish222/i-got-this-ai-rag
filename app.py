from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.baseline import (  # noqa: E402
    PLAIN_LANGUAGE_ANSWER_STYLE,
    BaselineRAG,
)
from i_got_this_rag.conversation import (  # noqa: E402
    ConversationQueryRewriter,
    ConversationTurn,
)
from i_got_this_rag.experiment_dashboard import (  # noqa: E402
    CurrentAppBenchmark,
    ExperimentDashboard,
    GenerationModelDashboard,
    load_claim_faithfulness_audit,
    load_current_app_benchmark,
    load_experiment_dashboard,
    load_generation_model_dashboard,
    load_qwen_generation_comparison,
)
from i_got_this_rag.settings import Settings  # noqa: E402
from i_got_this_rag.user_interface import (  # noqa: E402
    CLARIFICATION_TEXT,
    AnswerView,
    answer_question,
)


COMPARISON_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "phase10_final"
    / "comparison.json"
)
CURRENT_APP_RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "phase10_current_app"
    / "results.json"
)
GENERATION_MODEL_COMPARISON_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "generation_model_comparison.json"
)
CLAIM_FAITHFULNESS_AUDIT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "claim_faithfulness_audit"
    / "results.json"
)
QWEN_GENERATION_COMPARISON_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "qwen_generation_comparison.json"
)
SUGGESTED_QUESTIONS = (
    ("⏭️", "What's coming up this week?"),
    ("💌", "Which invitations still need an RSVP?"),
    ("🎒", "What should I prepare for this weekend?"),
    ("📅", "Plan my week."),
)
PENDING_PROMPT_KEY = "pending_prompt"
ANSWER_CATEGORY_LABELS = {
    "school": "🏫 School",
    "kids": "👧 Kids activities",
    "household": "🏠 Household",
    "learning": "📚 Learning",
    "volunteer": "🤝 Volunteer",
    "social": "🎉 Social",
    "family": "👨‍👩‍👧 Family",
}
ANSWER_CATEGORY_KEYWORDS = (
    (
        "volunteer",
        (
            "volunteer",
            "mentor",
            "mentoring",
            "donate",
            "donation",
            "collection",
            "welcome table",
            "neighborhood association",
        ),
    ),
    (
        "kids",
        (
            "piano",
            "robotics",
            "singing",
            "taekwondo",
            "watercolor",
            "swim",
            "music lesson",
            "ensemble",
            "workshop",
            "kids activity",
            "children's activity",
        ),
    ),
    (
        "school",
        (
            "school",
            "teacher",
            "classroom",
            "field trip",
            "picture day",
            "diagnostic",
            "permission form",
        ),
    ),
    (
        "learning",
        (
            "practical ai",
            "course",
            "assignment",
            "certificate",
            "study",
            "learning",
        ),
    ),
    (
        "household",
        (
            "household",
            "home task",
            "maintenance",
            "repair",
            "hvac",
            "technician arrival",
            "groceries",
            "meal plan",
            "laundry",
            "trash",
        ),
    ),
    (
        "social",
        (
            "rsvp",
            "invitation",
            "potluck",
            "dinner with",
            "coffee with",
            "birthday party",
            "friend",
            "social",
        ),
    ),
    (
        "family",
        (
            "family",
            "birthday",
            "gift",
            "relative",
        ),
    ),
)
SOURCE_CATEGORY_PATTERN = re.compile(
    r"(?:^|/)(school|activities|household|learning|volunteer|social|family)(?:/|$)",
    re.IGNORECASE,
)
CITATION_LABEL_PATTERN = re.compile(r"\[(S\d+)\]", re.IGNORECASE)
DISPLAY_CITATION_PATTERN = re.compile(r"(?:\s*\[S\d+\])+", re.IGNORECASE)
LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*+] |\d+[.)] )")
DATE_HEADING_PATTERN = re.compile(
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
    r"\s*(?:[-–—,]\s*)?"
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|"
    r"Nov|Dec)\s+"
    r"\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?:?",
    re.IGNORECASE,
)
AGENDA_TIME_PREFIX_PATTERN = re.compile(
    r"^\s*[-*+]\s+(?P<time>"
    r"(?:(?:\d{1,2}(?::\d{2})?\s*(?:AM|PM)?\s*[–-]\s*)?"
    r"\d{1,2}(?::\d{2})?\s*(?:AM|PM)|Noon|Afternoon|Evening)"
    r")\s+—\s+(?P<body>.+)$",
    re.IGNORECASE,
)

load_dotenv(PROJECT_ROOT / ".env", override=True)


def display_name() -> str:
    return os.getenv("APP_USER_NAME", "").strip() or "there"


@st.cache_resource(show_spinner=False)
def connect_pipeline() -> BaselineRAG:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    settings = Settings.from_environment(PROJECT_ROOT)
    return BaselineRAG(settings, answer_style=PLAIN_LANGUAGE_ANSWER_STYLE)


def render_sources(response: AnswerView) -> None:
    if response.answer == CLARIFICATION_TEXT:
        return
    if not response.sources:
        st.caption("No source could be safely attributed to this response.")
        return

    with st.expander(f"📚 Sources ({len(response.sources)})"):
        for source_index, source in enumerate(response.sources, start=1):
            location = source.source_path
            if source.page_number is not None:
                location = f"{location} · page {source.page_number}"
            st.markdown(f"**Source {source_index}: {source.title}**")
            st.caption(location)


def _source_category(source_path: str) -> str | None:
    match = SOURCE_CATEGORY_PATTERN.search(source_path.replace("\\", "/"))
    if not match:
        return None
    category = match.group(1).casefold()
    return "kids" if category == "activities" else category


def _answer_block_category(
    text: str,
    source_categories: dict[str, str],
) -> str | None:
    cited_categories = {
        source_categories[label.upper()]
        for label in CITATION_LABEL_PATTERN.findall(text)
        if label.upper() in source_categories
    }
    if len(cited_categories) == 1:
        cited_category = next(iter(cited_categories))
        if cited_category != "family":
            return cited_category

    normalized = text.casefold()
    for category, keywords in ANSWER_CATEGORY_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            return category
    for label in CITATION_LABEL_PATTERN.findall(text):
        category = source_categories.get(label.upper())
        if category:
            return category
    return None


def _split_answer_blocks(answer: str) -> tuple[str, ...]:
    blocks: list[str] = []
    for paragraph in re.split(r"\n\s*\n", answer.strip()):
        current: list[str] = []
        for line in paragraph.splitlines():
            if LIST_ITEM_PATTERN.match(line) and current:
                blocks.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            blocks.append("\n".join(current).strip())
    return tuple(block for block in blocks if block)


def _display_answer_text(text: str) -> str:
    without_citations = DISPLAY_CITATION_PATTERN.sub("", text)
    cleaned_lines = (line.rstrip() for line in without_citations.splitlines())
    return "\n".join(cleaned_lines).strip()


def _is_date_heading(text: str) -> bool:
    visible = _display_answer_text(text).strip()
    visible = re.sub(r"^#{1,6}\s+", "", visible)
    visible = visible.strip("*_ ")
    return "\n" not in visible and bool(DATE_HEADING_PATTERN.fullmatch(visible))


def _date_heading_label(text: str) -> str:
    visible = _display_answer_text(text).strip()
    visible = re.sub(r"^#{1,6}\s+", "", visible)
    return visible.strip("*_ ").removesuffix(":")


def _agenda_item_parts(text: str) -> tuple[str | None, str | None, str]:
    visible = _display_answer_text(text).strip()
    match = AGENDA_TIME_PREFIX_PATTERN.match(visible)
    if match:
        time_label = match.group("time").strip()
        body = match.group("body").strip()
    else:
        time_label = None
        body = LIST_ITEM_PATTERN.sub("", visible, count=1).strip()

    normalized = body.casefold()
    if "rsvp" in normalized and ("deadline" in normalized or "due" in normalized):
        status = "ACTION NEEDED"
    elif "deadline" in normalized:
        status = "DEADLINE"
    elif re.search(r"\bdue\b", normalized):
        status = "DUE"
    else:
        status = None
    return time_label, status, body


def _render_day_entry(
    category: str | None,
    text: str,
    response_index: int,
    day_index: int,
    entry_index: int,
) -> None:
    time_label, status, body = _agenda_item_parts(text)
    with st.container(
        key=f"answer_event_row_{response_index}_{day_index}_{entry_index}",
    ):
        badges: list[str] = []
        if category is not None:
            category_label = html.escape(ANSWER_CATEGORY_LABELS[category])
            badges.append(
                f'<span class="igt-category-chip igt-chip-{category}">'
                f"{category_label}</span>"
            )
        if status is not None:
            badges.append(
                f'<span class="igt-action-chip">{html.escape(status)}</span>'
            )
        if time_label is not None:
            badges.append(
                f'<span class="igt-time-chip">{html.escape(time_label)}</span>'
            )
        if badges:
            st.markdown(
                f'<div class="igt-event-meta">{"".join(badges)}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(body)


def categorized_answer_blocks(
    response: AnswerView,
) -> tuple[tuple[str | None, str], ...]:
    source_categories = {
        source.label.upper(): category
        for source in response.sources
        if (category := _source_category(source.source_path)) is not None
    }
    categorized: list[tuple[str | None, str]] = []
    active_category: str | None = None
    within_dated_section = False
    for text in _split_answer_blocks(response.answer):
        explicit_category = _answer_block_category(text, source_categories)
        if _is_date_heading(text):
            within_dated_section = True
            active_category = explicit_category
            category = None
        elif explicit_category is not None:
            active_category = explicit_category
            category = explicit_category
        elif LIST_ITEM_PATTERN.match(text):
            category = active_category
        else:
            category = None
        if (
            categorized
            and not within_dated_section
            and category is not None
            and categorized[-1][0] == category
        ):
            previous_category, previous_text = categorized[-1]
            categorized[-1] = (previous_category, f"{previous_text}\n{text}")
        else:
            categorized.append((category, text))
    return tuple(categorized)


def render_answer_content(response: AnswerView, response_index: int) -> None:
    blocks = categorized_answer_blocks(response)
    has_date_headings = any(_is_date_heading(text) for _, text in blocks)
    if not has_date_headings and not any(category for category, _ in blocks):
        st.markdown(_display_answer_text(response.answer))
        return

    if has_date_headings:
        block_index = 0
        day_index = 0
        while block_index < len(blocks):
            category, text = blocks[block_index]
            visible_text = _display_answer_text(text)
            if not visible_text:
                block_index += 1
                continue
            if not _is_date_heading(visible_text):
                st.markdown(visible_text)
                block_index += 1
                continue

            day_entries: list[tuple[str | None, str]] = []
            next_index = block_index + 1
            while next_index < len(blocks):
                next_category, next_text = blocks[next_index]
                if _is_date_heading(next_text):
                    break
                if (
                    day_entries
                    and next_category is None
                    and not LIST_ITEM_PATTERN.match(next_text)
                ):
                    break
                day_entries.append((next_category, next_text))
                next_index += 1

            with st.container(
                key=f"answer_daily_card_{response_index}_{day_index}",
            ):
                st.markdown(f"### {_date_heading_label(visible_text)}")
                for entry_index, (entry_category, entry_text) in enumerate(
                    day_entries
                ):
                    _render_day_entry(
                        entry_category,
                        entry_text,
                        response_index,
                        day_index,
                        entry_index,
                    )
            day_index += 1
            block_index = next_index
        return

    for block_index, (category, text) in enumerate(blocks):
        visible_text = _display_answer_text(text)
        if not visible_text:
            continue
        if category is None:
            st.markdown(visible_text)
            continue
        with st.container(
            key=f"answer_category_{category}_{response_index}_{block_index}",
        ):
            st.markdown(f"**{ANSWER_CATEGORY_LABELS[category]}**")
            st.markdown(visible_text)


@st.cache_data(show_spinner=False)
def load_dashboard(path: str, modified_at_ns: int) -> ExperimentDashboard:
    del modified_at_ns
    return load_experiment_dashboard(Path(path))


@st.cache_data(show_spinner=False)
def load_current_app_results(path: str, modified_at_ns: int) -> CurrentAppBenchmark:
    del modified_at_ns
    return load_current_app_benchmark(Path(path))


@st.cache_data(show_spinner=False)
def load_generation_model_results(
    path: str,
    modified_at_ns: int,
) -> GenerationModelDashboard:
    del modified_at_ns
    return load_generation_model_dashboard(Path(path))


@st.cache_data(show_spinner=False)
def load_claim_audit_results(path: str, modified_at_ns: int) -> dict:
    del modified_at_ns
    return load_claim_faithfulness_audit(Path(path))


@st.cache_data(show_spinner=False)
def load_qwen_generation_results(path: str, modified_at_ns: int) -> dict:
    del modified_at_ns
    return load_qwen_generation_comparison(Path(path))


def render_generation_model_comparison() -> None:
    st.divider()
    st.subheader("Generation model comparison")
    st.caption(
        "Single-variable experiment: every model receives the same cached Top-5 "
        "evidence and the same strict grounding prompt. Relevance filtering is off."
    )
    if not GENERATION_MODEL_COMPARISON_PATH.is_file():
        st.info(
            "No generation-model comparison is available yet. Run "
            "`uv run python evaluation/run_generation_model_experiments.py` "
            "to generate it."
        )
        return
    try:
        comparison = load_generation_model_results(
            GENERATION_MODEL_COMPARISON_PATH.as_posix(),
            GENERATION_MODEL_COMPARISON_PATH.stat().st_mtime_ns,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"The generation-model comparison could not be loaded: {exc}")
        return

    st.dataframe(
        [row.table_record() for row in comparison.rows],
        hide_index=True,
        width="stretch",
        column_config={
            "Recall@5": st.column_config.NumberColumn(format="%.3f"),
            "Faithfulness": st.column_config.NumberColumn(format="%.3f"),
            "Relevance": st.column_config.NumberColumn(format="%.3f"),
            "Refusal": st.column_config.NumberColumn(format="%.3f"),
            "Avg. latency (s)": st.column_config.NumberColumn(format="%.3f"),
            "P95 latency (s)": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    if comparison.eligible_experiment_ids:
        faithfulness, relevance, latency, balance = st.columns(4)
        faithfulness.markdown("**Highest faithfulness**")
        faithfulness.write(
            comparison.labels_for(comparison.highest_faithfulness_ids)
        )
        relevance.markdown("**Highest relevance/correctness**")
        relevance.write(comparison.labels_for(comparison.highest_relevance_ids))
        latency.markdown("**Lowest average latency**")
        latency.write(comparison.labels_for(comparison.lowest_latency_ids))
        balance.markdown("**Best overall balance**")
        balance.write(comparison.labels_for(comparison.best_balance_ids))
        st.caption(
            "Overall balance is a multi-metric comparison, not an automatic model "
            f"recommendation. {comparison.balance_method}"
        )
    else:
        st.warning(
            "No model is eligible for highlights. A run must complete without "
            "generation failures and preserve correct refusal at 1.000."
        )

    failed_rows = [
        row for row in comparison.rows if row.run_status != "complete"
    ]
    if failed_rows:
        with st.expander("Model configuration and API failures"):
            for row in failed_rows:
                error = row.configuration_error or {}
                st.write(
                    f"**{row.label}:** {error.get('message', row.run_status)}"
                )
    st.caption(f"Model comparison completed: {comparison.completed_at}")


def render_claim_faithfulness_audit() -> None:
    st.divider()
    st.subheader("Claim-level faithfulness audit")
    st.caption(
        "Evaluation-only audit over saved answers and exact indexed chunk text. "
        "No answers or retrieval results were regenerated."
    )
    if not CLAIM_FAITHFULNESS_AUDIT_PATH.is_file():
        st.info(
            "No claim audit is available yet. Run "
            "`uv run python evaluation/run_claim_faithfulness_audit.py`."
        )
        return
    try:
        audit = load_claim_audit_results(
            CLAIM_FAITHFULNESS_AUDIT_PATH.as_posix(),
            CLAIM_FAITHFULNESS_AUDIT_PATH.stat().st_mtime_ns,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"The claim-level audit could not be loaded: {exc}")
        return

    st.dataframe(
        [
            {
                "Model": item["model"],
                "Existing faithfulness": item["existing_faithfulness"],
                "Claim faithfulness": item["claim_level_faithfulness"],
                "Relevance": item["relevance_correctness"],
                "Claims": item["total_factual_claims"],
                "Supported": item["supported_factual_claims"],
                "Unsupported": item["unsupported_factual_claims"],
                "Unsupported / answer": item["unsupported_claims_per_answer"],
                "Disagreements": item["evaluator_disagreement_count"],
            }
            for item in audit["model_summary"]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "Existing faithfulness": st.column_config.NumberColumn(format="%.3f"),
            "Claim faithfulness": st.column_config.NumberColumn(format="%.3f"),
            "Relevance": st.column_config.NumberColumn(format="%.3f"),
            "Unsupported / answer": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    conclusion = audit["conclusion"]
    st.warning(
        f"Conclusion {conclusion['code']} — {conclusion['label']}: "
        f"{conclusion['reason']}"
    )

    with st.expander("Unsupported-claim categories by model"):
        st.dataframe(
            [
                {"Model": item["model"], **item["category_counts"]}
                for item in audit["model_summary"]
            ],
            hide_index=True,
            width="stretch",
        )

    models = {item["model"]: item for item in audit["models"]}
    selected_model = st.selectbox(
        "Inspect claim evidence for a model",
        options=list(models),
        key="claim_audit_model",
    )
    for question in models[selected_model]["questions"]:
        claim_score = question["claim_faithfulness"]
        score_label = (
            "no factual claims" if claim_score is None else f"{claim_score:.3f}"
        )
        disagreement = (
            " · evaluator disagreement" if question["evaluator_disagreement"] else ""
        )
        with st.expander(
            f"{question['question_id']} · claim score {score_label}{disagreement}"
        ):
            st.markdown("**Question**")
            st.write(question["question"])
            st.markdown("**Generated answer**")
            st.code(question["generated_answer"], language="text")
            st.markdown("**Retrieved context**")
            for context in question["retrieved_chunks"]:
                st.markdown(
                    f"**{context['source_id']} — {context['title']}**  "
                    f"`{context['chunk_id']}`"
                )
                st.code(context["text"], language="text")
            if not question["claims"]:
                st.info("No factual claims: explicit refusal or empty answer.")
            for index, claim in enumerate(question["claims"], start=1):
                st.markdown(f"**Claim {index}: {claim['claim']}**")
                st.write(f"Supported: {'Yes' if claim['supported'] else 'No'}")
                st.write(
                    "Supporting sources: "
                    + (", ".join(claim["supporting_source_ids"]) or "None")
                )
                st.write(f"Evidence: {claim['supporting_evidence'] or 'None'}")
                st.write(f"Reason: {claim['reason']}")
                st.write(f"Relevance: {claim['relevance_reason']}")
                st.write(f"Category: {claim['category'] or 'supported and relevant'}")
            st.write(
                f"Automated faithfulness: {question['automated_faithfulness']:.3f}"
            )
            st.write(f"Claim-level faithfulness: {score_label}")
            if question["evaluator_disagreement"]:
                st.warning(
                    "Evaluator disagreement — manual inspection recommended"
                )
    st.caption(f"Claim audit completed: {audit['completed_at']}")


def render_qwen_generation_comparison() -> None:
    st.divider()
    st.subheader("Latest: Qwen answer-quality optimization")
    st.caption(
        "E1/E2/E3 use the same Qwen model, 15 questions, and immutable Top-5 "
        "retrieval cache. Only answer directness, length policy, and E3 evidence "
        "selection vary."
    )
    if not QWEN_GENERATION_COMPARISON_PATH.is_file():
        st.info(
            "No Qwen concise-generation comparison is available yet. Run "
            "`uv run python evaluation/run_qwen_generation_experiments.py`."
        )
        return
    try:
        comparison = load_qwen_generation_results(
            QWEN_GENERATION_COMPARISON_PATH.as_posix(),
            QWEN_GENERATION_COMPARISON_PATH.stat().st_mtime_ns,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"The Qwen generation comparison could not be loaded: {exc}")
        return

    versions = comparison["versions"]
    baseline = versions[0]["metrics"]
    highest_relevance = max(
        versions,
        key=lambda version: version["metrics"]["answer_relevance_correctness"],
    )
    highest_faithfulness = max(
        versions,
        key=lambda version: version["metrics"]["claim_level_faithfulness"],
    )
    lowest_output = min(
        versions,
        key=lambda version: version["metrics"]["average_output_tokens"],
    )
    fastest = min(
        versions,
        key=lambda version: version["metrics"]["average_latency_seconds"],
    )

    relevance, faithfulness, output, latency = st.columns(4)
    relevance.metric(
        "Highest relevance",
        f"{highest_relevance['metrics']['answer_relevance_correctness']:.3f}",
        highest_relevance["experiment_id"].split("_", 1)[0],
    )
    faithfulness.metric(
        "Highest claim faithfulness",
        f"{highest_faithfulness['metrics']['claim_level_faithfulness']:.3f}",
        highest_faithfulness["experiment_id"].split("_", 1)[0],
    )
    output.metric(
        "Lowest average output",
        f"{lowest_output['metrics']['average_output_tokens']:.1f} tokens",
        lowest_output["experiment_id"].split("_", 1)[0],
    )
    latency.metric(
        "Fastest average latency",
        f"{fastest['metrics']['average_latency_seconds']:.3f}s",
        fastest["experiment_id"].split("_", 1)[0],
    )

    st.dataframe(
        [
            {
                "Mode": version["label"],
                "Recall@5": version["metrics"]["recall_at_5"],
                "Claim faithfulness": version["metrics"]["claim_level_faithfulness"],
                "Relevance": version["metrics"]["answer_relevance_correctness"],
                "Refusal": version["metrics"]["correct_refusal_rate"],
                "Unsupported": version["metrics"]["unsupported_claims"],
                "Claims / answer": version["metrics"]["average_claims_per_answer"],
                "Output tokens": version["metrics"]["average_output_tokens"],
                "Avg. latency (s)": version["metrics"]["average_latency_seconds"],
                "P95 latency (s)": version["metrics"]["p95_latency_seconds"],
                "Success": "Yes" if version["meets_success_criteria"] else "No",
            }
            for version in versions
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "Recall@5": st.column_config.NumberColumn(format="%.3f"),
            "Claim faithfulness": st.column_config.NumberColumn(format="%.3f"),
            "Relevance": st.column_config.NumberColumn(format="%.3f"),
            "Refusal": st.column_config.NumberColumn(format="%.3f"),
            "Claims / answer": st.column_config.NumberColumn(format="%.3f"),
            "Output tokens": st.column_config.NumberColumn(format="%.1f"),
            "Avg. latency (s)": st.column_config.NumberColumn(format="%.3f"),
            "P95 latency (s)": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    def reduction(current: float, original: float) -> float:
        if original == 0:
            return 0.0
        return ((original - current) / original) * 100

    st.markdown("**Change from E1 (current strict prompt)**")
    st.dataframe(
        [
            {
                "Mode": version["experiment_id"].split("_", 1)[0],
                "Faithfulness change": (
                    version["metrics"]["claim_level_faithfulness"]
                    - baseline["claim_level_faithfulness"]
                ),
                "Relevance change": (
                    version["metrics"]["answer_relevance_correctness"]
                    - baseline["answer_relevance_correctness"]
                ),
                "Fewer claims": reduction(
                    version["metrics"]["average_claims_per_answer"],
                    baseline["average_claims_per_answer"],
                ),
                "Fewer output tokens": reduction(
                    version["metrics"]["average_output_tokens"],
                    baseline["average_output_tokens"],
                ),
                "Lower avg. latency": reduction(
                    version["metrics"]["average_latency_seconds"],
                    baseline["average_latency_seconds"],
                ),
                "Lower P95 latency": reduction(
                    version["metrics"]["p95_latency_seconds"],
                    baseline["p95_latency_seconds"],
                ),
            }
            for version in versions
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "Faithfulness change": st.column_config.NumberColumn(format="%+.3f"),
            "Relevance change": st.column_config.NumberColumn(format="%+.3f"),
            "Fewer claims": st.column_config.NumberColumn(format="%.1f%%"),
            "Fewer output tokens": st.column_config.NumberColumn(format="%.1f%%"),
            "Lower avg. latency": st.column_config.NumberColumn(format="%.1f%%"),
            "Lower P95 latency": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

    successful = [
        version["label"]
        for version in versions
        if version["meets_success_criteria"]
    ]
    if successful:
        st.success("Modes meeting every success criterion: " + ", ".join(successful))
        st.info(
            "E2 has the highest relevance and lowest average latency. E3 has the "
            "highest claim faithfulness, fewest unsupported claims, and lowest "
            "token use. Both preserve Recall@5 at 0.900 and correct refusal at "
            "1.000, so the preferred mode depends on the quality-versus-efficiency "
            "tradeoff rather than one metric alone."
        )
    else:
        st.warning(
            "No mode met every target simultaneously. Inspect the per-mode results "
            "before choosing a generation policy."
        )

    with st.expander("Mode definitions and token details"):
        st.markdown(
            "- **E1 — Current:** existing strict grounded prompt with the unchanged "
            "Top-5 evidence.\n"
            "- **E2 — Concise:** directness prompt and intent-based answer-length "
            "policy with the unchanged Top-5 evidence.\n"
            "- **E3 — Concise + selection:** E2 plus relevance-first evidence "
            "selection. Original Top-5 retrieval remains unchanged in evaluation "
            "logs."
        )
        st.dataframe(
            [
                {
                    "Mode": version["experiment_id"].split("_", 1)[0],
                    "Input tokens": version["metrics"].get("average_input_tokens"),
                    "Output tokens": version["metrics"]["average_output_tokens"],
                    "Total tokens": version["metrics"].get("average_total_tokens"),
                    "Generation latency (s)": version["metrics"].get(
                        "average_generation_latency_seconds"
                    ),
                    "Unsupported claims": version["metrics"].get(
                        "unsupported_claims"
                    ),
                }
                for version in versions
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "Input tokens": st.column_config.NumberColumn(format="%.1f"),
                "Output tokens": st.column_config.NumberColumn(format="%.1f"),
                "Total tokens": st.column_config.NumberColumn(format="%.1f"),
                "Generation latency (s)": st.column_config.NumberColumn(
                    format="%.3f"
                ),
            },
        )
        st.caption(
            "Token averages use provider-reported usage when available; deterministic "
            "refusals or answers may not invoke the model. Fixed model: "
            f"{comparison.get('fixed_model', 'not recorded')}. Retrieval cache: "
            f"{comparison.get('fixed_retrieval_cache_sha256', 'not recorded')}."
        )
    st.caption(f"Qwen comparison completed: {comparison['completed_at']}")


def render_experiment_dashboard() -> None:
    st.header("Experiment Dashboard")
    st.caption(
        "Controlled evaluations across the same 15-question dataset. The latest "
        "generation experiment is shown first."
    )

    render_qwen_generation_comparison()

    st.divider()
    st.subheader("Phase 10 retrieval and workflow comparison")
    if not COMPARISON_PATH.is_file():
        st.info(
            "No final comparison is available yet. Run "
            "`uv run python evaluation/run_final_evaluation.py` to generate it."
        )
        return

    try:
        dashboard = load_dashboard(
            COMPARISON_PATH.as_posix(),
            COMPARISON_PATH.stat().st_mtime_ns,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"The saved experiment comparison could not be loaded: {exc}")
        return

    recall_column, faithfulness_column, latency_column = st.columns(3)
    recall_column.metric("Best Recall@5", f"{dashboard.best_recall_at_5:.3f}")
    faithfulness_column.metric(
        "Best faithfulness",
        f"{dashboard.best_faithfulness:.3f}",
    )
    latency_column.metric(
        "Fastest average latency",
        f"{dashboard.fastest_average_latency_seconds:.3f}s",
    )

    st.dataframe(
        [row.table_record() for row in dashboard.rows],
        hide_index=True,
        width="stretch",
        column_config={
            "Recall@5": st.column_config.NumberColumn(format="%.3f"),
            "Faithfulness": st.column_config.NumberColumn(format="%.3f"),
            "Avg. latency (s)": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    labels = {
        row.version_id: f"{row.experiment} — {row.version}"
        for row in dashboard.rows
    }
    selected_id = st.selectbox(
        "Inspect an experiment",
        options=list(labels),
        format_func=labels.__getitem__,
    )
    detail = dashboard.detail_for(selected_id)
    st.subheader(detail.version)
    left, right = st.columns(2)
    with left:
        st.markdown("**What changed?**")
        st.write(detail.changed)
        st.markdown("**What stayed constant?**")
        st.write(detail.stayed_constant)
        st.markdown("**Why might the result differ?**")
        st.write(detail.why)
    with right:
        st.markdown("**What improved?**")
        st.write(detail.improved)
        st.markdown("**What became worse?**")
        st.write(detail.became_worse)
        st.markdown("**What latency did it add?**")
        st.write(detail.latency_cost)
        st.markdown("**Was it worth it?**")
        st.write(detail.worth_it)

    recommended_label = labels.get(
        dashboard.recommendation_version_id,
        dashboard.recommendation_version_id,
    )
    st.success(
        f"Phase 10 retrieval/workflow recommendation: {recommended_label}. "
        f"{dashboard.recommendation_rationale}"
    )
    st.caption(f"Comparison completed: {dashboard.completed_at}")

    render_generation_model_comparison()
    render_claim_faithfulness_audit()

    st.divider()
    st.subheader("Previous app end-to-end")
    st.caption(
        "Measured through the same answer path used by the Ask tab after the "
        "conversation, citation, deduplication, and plain-language corrections."
    )
    if not CURRENT_APP_RESULTS_PATH.is_file():
        st.info(
            "No current-app benchmark is available yet. Run "
            "`uv run python evaluation/run_current_app_evaluation.py` to generate it."
        )
        return

    try:
        current = load_current_app_results(
            CURRENT_APP_RESULTS_PATH.as_posix(),
            CURRENT_APP_RESULTS_PATH.stat().st_mtime_ns,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"The current-app benchmark could not be loaded: {exc}")
        return

    recall, faithfulness, refusal, latency = st.columns(4)
    recall.metric(
        "Current Recall@5",
        f"{current.recall_at_5:.3f}",
        f"{current.recall_delta:+.3f} vs historical baseline",
    )
    faithfulness.metric(
        "Current faithfulness",
        f"{current.faithfulness:.3f}",
        f"{current.faithfulness_delta:+.3f} vs historical baseline",
    )
    refusal.metric("Correct refusal", f"{current.correct_refusal_rate:.3f}")
    latency.metric(
        "Current average latency",
        f"{current.average_latency_seconds:.3f}s",
        f"{current.average_latency_delta_seconds:+.3f}s vs historical baseline",
        delta_color="inverse",
    )
    st.dataframe(
        [current.table_record()],
        hide_index=True,
        width="stretch",
        column_config={
            "Recall@5": st.column_config.NumberColumn(format="%.3f"),
            "Faithfulness": st.column_config.NumberColumn(format="%.3f"),
            "Correct refusal": st.column_config.NumberColumn(format="%.3f"),
            "Avg. latency (s)": st.column_config.NumberColumn(format="%.3f"),
            "P95 latency (s)": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    regression_summary = (
        f"Corrected-behavior checks: {current.regression_passed_count}/"
        f"{current.regression_case_count} passed."
    )
    if current.regression_pass_rate == 1.0:
        st.success(regression_summary)
    else:
        st.warning(regression_summary)
    with st.expander("Inspect corrected-behavior checks"):
        st.dataframe(
            [
                {
                    "Case": item["case_id"],
                    "Question": item["question"],
                    "Passed": bool(item["passed"]),
                    "Failures": "; ".join(item["failures"]) or "None",
                    "Latency (s)": float(item["latency_seconds"]),
                }
                for item in current.regression_cases
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "Latency (s)": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    st.caption(f"Current app measurement completed: {current.completed_at}")


def render_chat_response(response: AnswerView, response_index: int) -> None:
    with st.chat_message("assistant", avatar="✨"):
        render_answer_content(response, response_index)
        render_sources(response)


def queue_suggested_question(question: str) -> None:
    st.session_state[PENDING_PROMPT_KEY] = question


def render_question_answer() -> None:
    conversation = st.session_state.setdefault("conversation", [])
    toolbar_left, toolbar_right = st.columns([3, 1])
    with toolbar_left:
        st.markdown("#### Hi, Vaishali! 👋 Here’s your life, a little more organized")
        st.caption("🔒 Private by design · Answers grounded in your information")
    with toolbar_right:
        if st.button(
            "↻ New conversation",
            key="new_conversation",
            width="stretch",
        ):
            st.session_state["conversation"] = []
            st.session_state.pop(PENDING_PROMPT_KEY, None)
            st.rerun()

    if not conversation:
        with st.chat_message("assistant", avatar="✨"):
            st.markdown(
                "Hi! I can help connect schedules, deadlines, invitations, "
                "gifts, and family commitments. What would you like to figure out?"
            )

    st.markdown("##### WHAT DO YOU NEED RIGHT NOW?")
    suggestion_columns = st.columns(2)
    for index, (icon, question) in enumerate(SUGGESTED_QUESTIONS):
        with suggestion_columns[index % 2]:
            st.button(
                f"{icon} {question}",
                key=f"suggestion_{index}",
                width="stretch",
                on_click=queue_suggested_question,
                args=(question,),
            )

    for response_index, response in enumerate(conversation):
        if not isinstance(response, AnswerView):
            continue
        with st.chat_message("user", avatar="😊"):
            st.markdown(response.question)
        render_chat_response(response, response_index)

    typed_prompt = st.chat_input(
        "Ask what's next, what to prepare, or what you might be forgetting…"
    )
    suggested_prompt = st.session_state.pop(PENDING_PROMPT_KEY, None)
    prompt = suggested_prompt or typed_prompt
    if not prompt:
        return

    with st.chat_message("user", avatar="😊"):
        st.markdown(prompt)
    try:
        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("Connecting the dots..."):
                pipeline = connect_pipeline()
                history = tuple(
                    ConversationTurn(
                        user_question=response.question,
                        assistant_answer=response.answer,
                    )
                    for response in conversation[-3:]
                    if isinstance(response, AnswerView)
                )
                rewriter = ConversationQueryRewriter(
                    llm=pipeline.resources.llm,
                    reference_date=pipeline.settings.reference_date,
                    timezone=pipeline.settings.timezone,
                    memory_exchanges=3,
                )
                response = answer_question(
                    pipeline,
                    prompt,
                    history=history,
                    rewriter=rewriter,
                )
            render_answer_content(response, len(conversation))
            render_sources(response)
        conversation.append(response)
        st.rerun()
    except Exception as exc:  # Streamlit is the user-facing error boundary.
        st.error(str(exc))


def render_app() -> None:
    st.set_page_config(
        page_title="I Got This — What's Next?",
        page_icon="✓",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        :root {
            --igt-navy: #1c2042;
            --igt-yellow: #ffed8e;
            --igt-hero-yellow: #ffea8a;
            --igt-hero-blue: #35aec0;
            --igt-blush: #efd6db;
            --igt-blue: #cbdbf2;
            --igt-blue-strong: #badbe5;
            --igt-red: #dc3f40;
            --igt-orange: #ff7618;
            --igt-school: #dceeff;
            --igt-kids: #ebe3ff;
            --igt-household: #ffe2d2;
            --igt-learning: #d5f2ec;
            --igt-volunteer: #dff2d8;
            --igt-social: #ffdde2;
            --igt-family: #ffe8c9;
            --igt-paper: #fcfbfa;
        }
        ::selection {
            background: var(--igt-blue-strong);
            color: var(--igt-navy);
        }
        ::-moz-selection {
            background: var(--igt-blue-strong);
            color: var(--igt-navy);
        }
        .stApp {
            background: linear-gradient(
                120deg,
                var(--igt-hero-yellow) 0%,
                var(--igt-yellow) 62%,
                var(--igt-blush) 100%
            );
            background-attachment: fixed;
            color: var(--igt-navy);
            min-height: 100vh;
        }
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stWidgetLabel"] {
            color: var(--igt-navy);
        }
        .stApp [data-testid="stMarkdownContainer"] :is(
            p, li, strong, em, h1, h2, h3, h4, h5, h6
        ) {
            color: inherit;
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { max-width: 1180px; padding-top: 2rem; }
        .igt-hero {
            background: var(--igt-hero-blue);
            border-radius: 28px;
            box-shadow: 0 18px 45px rgba(28, 32, 66, .16);
            color: var(--igt-navy);
            margin-bottom: 1.25rem;
            overflow: hidden;
            padding: 1.8rem 2rem;
            position: relative;
        }
        .igt-hero::after {
            background: var(--igt-blue);
            border-radius: 999px;
            content: "";
            height: 150px;
            position: absolute;
            right: -25px;
            top: -55px;
            width: 150px;
        }
        .igt-brand {
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .18em;
            margin-bottom: .45rem;
            opacity: .9;
        }
        .igt-hero h1 { color: var(--igt-navy); font-size: 2.2rem; margin: 0; }
        .igt-hero p { font-size: 1.02rem; margin: .4rem 0 0; opacity: .9; }
        [data-testid="stChatMessage"] {
            backdrop-filter: blur(12px);
            background: rgba(252, 251, 250, .88);
            border: 1px solid rgba(28, 32, 66, .14);
            border-radius: 20px;
            box-shadow: 0 7px 22px rgba(28, 32, 66, .08);
            margin-bottom: .8rem;
            padding: .35rem .5rem;
        }
        [data-testid="stChatMessage"]:has(
            [class*="st-key-answer_category_"]
        ),
        [data-testid="stChatMessage"]:has(
            [class*="st-key-answer_daily_card_"]
        ) {
            backdrop-filter: none;
            background: transparent;
            border-color: transparent;
            box-shadow: none;
        }
        [class*="st-key-answer_category_"] {
            border: 1px solid rgba(28, 32, 66, .12);
            border-radius: 18px;
            box-shadow: 0 6px 18px rgba(28, 32, 66, .07);
            margin: .45rem 0;
            padding: .8rem 1rem .55rem;
        }
        [class*="st-key-answer_category_"]
        [data-testid="stMarkdownContainer"] p {
            color: var(--igt-navy);
        }
        [class*="st-key-answer_category_school_"] {
            background: var(--igt-school);
        }
        [class*="st-key-answer_category_kids_"] {
            background: var(--igt-kids);
        }
        [class*="st-key-answer_category_household_"] {
            background: var(--igt-household);
        }
        [class*="st-key-answer_category_learning_"] {
            background: var(--igt-learning);
        }
        [class*="st-key-answer_category_volunteer_"] {
            background: var(--igt-volunteer);
        }
        [class*="st-key-answer_category_social_"] {
            background: var(--igt-social);
        }
        [class*="st-key-answer_category_family_"] {
            background: var(--igt-family);
        }
        [class*="st-key-answer_daily_card_"] {
            background: rgba(252, 251, 250, .78);
            border: 1px solid rgba(28, 32, 66, .14);
            border-left: 6px solid var(--igt-orange);
            border-radius: 18px;
            box-shadow: 0 7px 20px rgba(28, 32, 66, .08);
            margin: .75rem 0;
            padding: .75rem 1rem .35rem;
        }
        [class*="st-key-answer_daily_card_"] h3 {
            color: var(--igt-navy);
            font-size: 1.18rem;
            font-weight: 850;
            letter-spacing: .01em;
            margin: 0 0 .3rem;
        }
        [class*="st-key-answer_event_row_"] {
            border-bottom: 1px solid rgba(28, 32, 66, .1);
            padding: .48rem 0 .38rem;
        }
        [class*="st-key-answer_event_row_"]:last-child {
            border-bottom: 0;
        }
        [class*="st-key-answer_event_row_"]
        [data-testid="stMarkdownContainer"] p {
            margin: .18rem 0;
        }
        .igt-event-meta {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: .35rem;
            margin-bottom: .18rem;
        }
        .igt-category-chip,
        .igt-action-chip,
        .igt-time-chip {
            border: 1px solid rgba(28, 32, 66, .12);
            border-radius: 999px;
            color: var(--igt-navy);
            display: inline-block;
            font-size: .76rem;
            font-weight: 800;
            line-height: 1;
            padding: .34rem .55rem;
        }
        .igt-chip-school { background: var(--igt-school); }
        .igt-chip-kids { background: var(--igt-kids); }
        .igt-chip-household { background: var(--igt-household); }
        .igt-chip-learning { background: var(--igt-learning); }
        .igt-chip-volunteer { background: var(--igt-volunteer); }
        .igt-chip-social { background: var(--igt-social); }
        .igt-chip-family { background: var(--igt-family); }
        .igt-action-chip {
            background: rgba(255, 118, 24, .16);
            border-color: rgba(255, 118, 24, .4);
            color: #8a3500;
        }
        .igt-time-chip {
            background: rgba(203, 219, 242, .7);
        }
        [data-testid="stChatInput"] {
            background: #ffffff;
            border: 2px solid var(--igt-blue-strong);
            border-radius: 18px;
            box-shadow: 0 8px 26px rgba(28, 32, 66, .10);
        }
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] textarea:focus {
            caret-color: var(--igt-navy);
            color: var(--igt-navy) !important;
            -webkit-text-fill-color: var(--igt-navy) !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: rgba(28, 32, 66, .62) !important;
            -webkit-text-fill-color: rgba(28, 32, 66, .62) !important;
        }
        .stButton > button {
            background: rgba(239, 214, 219, .42);
            border: 1px solid rgba(28, 32, 66, .2);
            border-radius: 14px;
            color: var(--igt-navy);
            font-weight: 650;
            transition: transform .15s ease, box-shadow .15s ease;
        }
        .stButton > button:hover {
            background: var(--igt-yellow);
            border-color: var(--igt-navy);
            color: var(--igt-navy);
            box-shadow: 0 7px 18px rgba(28, 32, 66, .14);
            transform: translateY(-1px);
        }
        [class*="st-key-suggestion_"] .stButton > button:hover,
        [class*="st-key-suggestion_"] .stButton > button:focus,
        [class*="st-key-suggestion_"] .stButton > button:active,
        [class*="st-key-new_conversation"] .stButton > button:hover,
        [class*="st-key-new_conversation"] .stButton > button:focus,
        [class*="st-key-new_conversation"] .stButton > button:active {
            background: var(--igt-red);
            border-color: var(--igt-navy);
            color: var(--igt-navy);
        }
        .stButton > button * { color: inherit !important; }
        [data-baseweb="tab-list"] { gap: .5rem; }
        [data-baseweb="tab"] {
            border-radius: 999px;
            color: var(--igt-navy) !important;
            font-weight: 700;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
        }
        [data-baseweb="tab"] * { color: inherit !important; }
        .stTabs [data-testid="stTab"][data-selected],
        [aria-selected="true"][data-baseweb="tab"] {
            background: transparent !important;
            color: var(--igt-orange) !important;
            -webkit-text-fill-color: var(--igt-orange) !important;
        }
        .stTabs [data-testid="stTab"][data-selected] *,
        [aria-selected="true"][data-baseweb="tab"] * {
            color: var(--igt-orange) !important;
            -webkit-text-fill-color: var(--igt-orange) !important;
        }
        .stTabs .react-aria-SelectionIndicator,
        [data-baseweb="tab-highlight"] {
            background-color: var(--igt-orange) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    name = html.escape(display_name())
    st.markdown(
        f"""
        <div class="igt-hero">
            <h1>✨ I GOT THIS. What’s next?</h1>
            <h4>24 hours. A hundred things to remember. Let’s make “What’s next?” the easy part.</h4>
            <p>School. Kids. Home. Learning. Volunteering. Social plans. Life. You keep living it. I Got This keeps track of it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    ask_tab, experiments_tab = st.tabs(["Ask", "Experiments"])
    with ask_tab:
        left, center, right = st.columns([1, 3, 1])
        del left, right
        with center:
            render_question_answer()
    with experiments_tab:
        render_experiment_dashboard()


render_app()
