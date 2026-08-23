from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.baseline import BaselineRAG  # noqa: E402
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


@st.cache_resource(show_spinner=False)
def connect_pipeline() -> BaselineRAG:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    settings = Settings.from_environment(PROJECT_ROOT)
    return BaselineRAG(settings)


def render_sources(response: AnswerView) -> None:
    st.subheader("Sources")
    if not response.sources:
        st.caption("No sources were cited for this response.")
        return

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


def render_question_answer() -> None:
    st.write("Ask about your family knowledge.")

    with st.form("family_knowledge_question"):
        question = st.text_input(
            "Question",
            placeholder="What should I prepare for this week?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(
            "Ask",
            type="primary",
            width="stretch",
        )

    if submitted:
        if not question.strip():
            st.session_state.pop("answer_view", None)
            st.warning("Enter a question before selecting Ask.")
        else:
            try:
                with st.spinner("Searching your knowledge base..."):
                    response = answer_question(connect_pipeline(), question)
                st.session_state["answer_view"] = response
            except Exception as exc:  # Streamlit is the user-facing error boundary.
                st.session_state.pop("answer_view", None)
                st.error(str(exc))

    response = st.session_state.get("answer_view")
    if isinstance(response, AnswerView):
        st.divider()
        st.subheader("Answer")
        st.markdown(response.answer)
        render_sources(response)


def render_app() -> None:
    st.set_page_config(
        page_title="I Got This — What's Next?",
        page_icon="✓",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container { max-width: 1180px; padding-top: 3rem; }
        [data-testid="stForm"] { border: 0; padding: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("I GOT THIS")
    st.markdown("### What's next?")
    ask_tab, experiments_tab = st.tabs(["Ask", "Experiments"])
    with ask_tab:
        left, center, right = st.columns([1, 2, 1])
        del left, right
        with center:
            render_question_answer()
    with experiments_tab:
        render_experiment_dashboard()


render_app()
