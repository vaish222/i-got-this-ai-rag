from __future__ import annotations

import html
import os
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
    ExperimentDashboard,
    load_experiment_dashboard,
)
from i_got_this_rag.settings import Settings  # noqa: E402
from i_got_this_rag.user_interface import AnswerView, answer_question  # noqa: E402


COMPARISON_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "phase10_final"
    / "comparison.json"
)
SUGGESTED_QUESTIONS = (
    ("📅", "What's coming up this week?"),
    ("💌", "Which invitations still need an RSVP?"),
    ("🎒", "What should I prepare for this weekend?"),
    ("🎁", "Which birthdays still need gifts?"),
)
PENDING_PROMPT_KEY = "pending_prompt"

load_dotenv(PROJECT_ROOT / ".env", override=True)


def display_name() -> str:
    return os.getenv("APP_USER_NAME", "").strip() or "there"


@st.cache_resource(show_spinner=False)
def connect_pipeline() -> BaselineRAG:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    settings = Settings.from_environment(PROJECT_ROOT)
    return BaselineRAG(settings, answer_style=PLAIN_LANGUAGE_ANSWER_STYLE)


def render_sources(response: AnswerView) -> None:
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


@st.cache_data(show_spinner=False)
def load_dashboard(path: str, modified_at_ns: int) -> ExperimentDashboard:
    del modified_at_ns
    return load_experiment_dashboard(Path(path))


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


def render_chat_response(response: AnswerView) -> None:
    with st.chat_message("assistant", avatar="✨"):
        st.markdown(response.answer)
        render_sources(response)


def queue_suggested_question(question: str) -> None:
    st.session_state[PENDING_PROMPT_KEY] = question


def render_question_answer() -> None:
    conversation = st.session_state.setdefault("conversation", [])
    toolbar_left, toolbar_right = st.columns([3, 1])
    with toolbar_left:
        st.markdown("#### Your family knowledge assistant")
        st.caption("Private session memory · grounded answers · safe citations")
    with toolbar_right:
        if st.button("↻ New conversation", width="stretch"):
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

    for response in conversation:
        if not isinstance(response, AnswerView):
            continue
        with st.chat_message("user", avatar="😊"):
            st.markdown(response.question)
        render_chat_response(response)

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
            st.markdown(response.answer)
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
            --igt-blush: #efd6db;
            --igt-blue: #cbdbf2;
            --igt-blue-strong: #badbe5;
            --igt-paper: #fcfbfa;
        }
        ::selection {
            background: var(--igt-yellow);
            color: var(--igt-navy);
        }
        ::-moz-selection {
            background: var(--igt-yellow);
            color: var(--igt-navy);
        }
        .stApp {
            background:
                radial-gradient(circle at 8% 8%, rgba(239, 214, 219, .72), transparent 24rem),
                radial-gradient(circle at 92% 4%, rgba(203, 219, 242, .72), transparent 28rem),
                linear-gradient(180deg, var(--igt-paper) 0%, #ffffff 100%);
            color: var(--igt-navy);
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
            background: linear-gradient(
                120deg,
                var(--igt-hero-yellow) 0%,
                var(--igt-yellow) 62%,
                var(--igt-blush) 100%
            );
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
        [aria-selected="true"][data-baseweb="tab"] {
            background: var(--igt-yellow);
            color: var(--igt-navy) !important;
        }
        [data-baseweb="tab-highlight"] { background: var(--igt-navy) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    name = html.escape(display_name())
    st.markdown(
        f"""
        <div class="igt-hero">
            <div class="igt-brand">✨ I GOT THIS</div>
            <h1>Hi, {name}! What’s next?</h1>
            <p>Your colorful command center for family plans, promises, and everything in between.</p>
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
