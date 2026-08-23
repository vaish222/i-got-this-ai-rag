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
    load_current_app_benchmark,
    load_experiment_dashboard,
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
SUGGESTED_QUESTIONS = (
    ("📅", "What's coming up this week?"),
    ("💌", "Which invitations still need an RSVP?"),
    ("🎒", "What should I prepare for this weekend?"),
    ("🎁", "Which birthdays still need gifts?"),
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
        "kids",
        (
            "robotics",
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
LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*+] |\d+[.)] )")

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
        for source in response.sources:
            location = source.source_path
            if source.page_number is not None:
                location = f"{location} · page {source.page_number}"
            st.markdown(f"**[{source.label}] {source.title}**")
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


def categorized_answer_blocks(
    response: AnswerView,
) -> tuple[tuple[str | None, str], ...]:
    source_categories = {
        source.label.upper(): category
        for source in response.sources
        if (category := _source_category(source.source_path)) is not None
    }
    categorized: list[tuple[str | None, str]] = []
    for text in _split_answer_blocks(response.answer):
        category = _answer_block_category(text, source_categories)
        if categorized and category is not None and categorized[-1][0] == category:
            previous_category, previous_text = categorized[-1]
            categorized[-1] = (previous_category, f"{previous_text}\n{text}")
        else:
            categorized.append((category, text))
    return tuple(categorized)


def render_answer_content(response: AnswerView, response_index: int) -> None:
    blocks = categorized_answer_blocks(response)
    if not any(category for category, _ in blocks):
        st.markdown(response.answer)
        return

    for block_index, (category, text) in enumerate(blocks):
        if category is None:
            st.markdown(text)
            continue
        with st.container(
            key=f"answer_category_{category}_{response_index}_{block_index}",
        ):
            st.markdown(f"**{ANSWER_CATEGORY_LABELS[category]}**")
            st.markdown(text)


@st.cache_data(show_spinner=False)
def load_dashboard(path: str, modified_at_ns: int) -> ExperimentDashboard:
    del modified_at_ns
    return load_experiment_dashboard(Path(path))


@st.cache_data(show_spinner=False)
def load_current_app_results(path: str, modified_at_ns: int) -> CurrentAppBenchmark:
    del modified_at_ns
    return load_current_app_benchmark(Path(path))


def render_experiment_dashboard() -> None:
    st.header("Experiment Dashboard")
    st.caption(
        "Measured Phase 10 results across the same controlled evaluation dataset."
    )
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
        f"Recommended: {recommended_label}. "
        f"{dashboard.recommendation_rationale}"
    )
    st.caption(f"Comparison completed: {dashboard.completed_at}")

    st.divider()
    st.subheader("Current app end-to-end")
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

    st.markdown("##### Try asking")
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
            <h1>✨ I GOT THIS.</h1>
            <h4>24 hours. A hundred things to remember. Let’s make “What’s next?” the easy part.</h4>
            <p>School. Kids. Home. Learning. Volunteering. Social plans. One place to remember what matters, what’s coming, and what still needs your attention.</p>
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
