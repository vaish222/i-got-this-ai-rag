from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.baseline import (  # noqa: E402
    PLAIN_LANGUAGE_ANSWER_STYLE,
    BaselineRAG,
)
from i_got_this_rag.current_app_evaluation import (  # noqa: E402
    CURRENT_APP_EVALUATION_VERSION,
    CURRENT_APP_EXPERIMENT_ID,
    evaluate_current_app,
    evaluate_ui_regressions,
    historical_baseline_delta,
)
from i_got_this_rag.evaluation import load_evaluation_dataset, utc_now  # noqa: E402
from i_got_this_rag.ingestion import (  # noqa: E402
    chunk_documents,
    chunk_fingerprint,
    corpus_fingerprint,
    load_corpus,
)
from i_got_this_rag.settings import Settings  # noqa: E402


QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
HISTORICAL_COMPARISON_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "phase10_final" / "comparison.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "phase10_current_app" / "results.json"
)
SOURCE_PATHS = (
    PROJECT_ROOT / "app.py",
    PROJECT_ROOT / "src" / "i_got_this_rag" / "user_interface.py",
    PROJECT_ROOT / "src" / "i_got_this_rag" / "conversation.py",
)


def file_record(path: Path) -> dict[str, str]:
    raw_bytes = path.read_bytes()
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    settings = Settings.from_environment(PROJECT_ROOT)
    dataset = load_evaluation_dataset(QUESTIONS_PATH)
    comparison = json.loads(HISTORICAL_COMPARISON_PATH.read_text(encoding="utf-8"))

    print("Connecting the current Streamlit dense pipeline...")
    pipeline = BaselineRAG(settings, answer_style=PLAIN_LANGUAGE_ANSWER_STYLE)
    print("Running the current app over the 15 Phase 10 questions...")
    evaluation = evaluate_current_app(pipeline, dataset)
    print("Running seven corrected-behavior regression scenarios...")
    regressions = evaluate_ui_regressions(pipeline)

    documents = load_corpus(settings.data_dir, PROJECT_ROOT)
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    metrics = evaluation["metrics"]
    payload = {
        "schema_version": "1.0",
        "phase": 10,
        "evaluation_version": CURRENT_APP_EVALUATION_VERSION,
        "experiment_id": CURRENT_APP_EXPERIMENT_ID,
        "experiment_name": "Current Streamlit app end-to-end evaluation",
        "completed_at": utc_now(),
        "mechanism": (
            "Selected dense Top-5 retrieval followed by the current Streamlit "
            "answer routing, deterministic handlers, citations, humanization, and "
            "conversation guards."
        ),
        "evaluation_dataset": {
            "path": QUESTIONS_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": dataset.sha256,
            "question_count": len(dataset.questions),
        },
        "historical_comparison": file_record(HISTORICAL_COMPARISON_PATH),
        "source_code": [file_record(path) for path in SOURCE_PATHS],
        "active_runtime_settings": settings.public_config(),
        "corpus": corpus_fingerprint(settings.data_dir, PROJECT_ROOT),
        "chunk_set": chunk_fingerprint(chunks),
        "metrics": metrics,
        "delta_vs_historical_baseline": historical_baseline_delta(metrics, comparison),
        "category_summary": evaluation["category_summary"],
        "questions": evaluation["questions"],
        "ui_regressions": regressions,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\nCurrent app end-to-end evaluation")
    print(f"- Recall@5: {metrics['recall_at_5']:.3f}")
    print(f"- Faithfulness: {metrics['faithfulness']:.3f}")
    print(f"- Correct refusal rate: {metrics['correct_refusal_rate']:.3f}")
    print(f"- Average latency: {metrics['average_latency_seconds']:.3f}s")
    print(f"- P95 latency: {metrics['p95_latency_seconds']:.3f}s")
    print(
        f"- UI regressions: {regressions['passed_count']}/"
        f"{regressions['case_count']} passed"
    )
    print(f"Results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
