from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.claim_faithfulness import (  # noqa: E402
    RetrievedContext,
    audit_question,
    determine_conclusion,
    summarize_model,
)
from i_got_this_rag.evaluation import utc_now  # noqa: E402
from i_got_this_rag.settings import Settings  # noqa: E402


EXPERIMENT_IDS = (
    "D1_current_model",
    "D2_nebius_model_1",
    "D3_nebius_model_2",
)
RESULTS_ROOT = PROJECT_ROOT / "evaluation" / "results"
OUTPUT_DIR = RESULTS_ROOT / "claim_faithfulness_audit"
OUTPUT_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = OUTPUT_DIR / "report.md"


def load_saved_results() -> tuple[dict[str, Any], ...]:
    results: list[dict[str, Any]] = []
    for experiment_id in EXPERIMENT_IDS:
        path = RESULTS_ROOT / experiment_id / "results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("run_status") != "complete":
            raise ValueError(
                f"{experiment_id} must be a complete saved run before auditing claims."
            )
        if len(payload.get("questions", [])) != 15:
            raise ValueError(f"{experiment_id} does not contain the same 15 questions.")
        payload["_artifact_path"] = path
        payload["_artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        results.append(payload)
    fingerprints = {str(item["retrieval_cache_sha256"]) for item in results}
    datasets = {str(item["evaluation_dataset"]["sha256"]) for item in results}
    question_ids = {
        tuple(str(question["question_id"]) for question in item["questions"])
        for item in results
    }
    if len(fingerprints) != 1 or len(datasets) != 1 or len(question_ids) != 1:
        raise ValueError("Saved model runs do not share one retrieval cache and dataset.")
    return tuple(results)


def fetch_indexed_chunks(settings: Settings) -> dict[str, dict[str, Any]]:
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("PINECONE_API_KEY is required to fetch saved chunk text.")
    index = Pinecone(api_key=api_key).Index(settings.pinecone_index_name)
    vector_ids: list[str] = []
    for page in index.list(namespace=settings.pinecone_namespace):
        vector_ids.extend(str(value) for value in page)
    if not vector_ids:
        raise ValueError("The configured Pinecone namespace contains no vectors.")
    fetched = index.fetch(ids=vector_ids, namespace=settings.pinecone_namespace)
    chunks: dict[str, dict[str, Any]] = {}
    for vector in fetched.vectors.values():
        metadata = dict(vector.metadata or {})
        chunk_id = str(metadata.get("chunk_id", ""))
        text = str(metadata.get("text", ""))
        if chunk_id and text:
            chunks[chunk_id] = metadata
    if not chunks:
        raise ValueError("Indexed vectors do not contain reusable chunk text.")
    return chunks


def question_contexts(
    question: dict[str, Any],
    indexed_chunks: dict[str, dict[str, Any]],
) -> tuple[RetrievedContext, ...]:
    contexts: list[RetrievedContext] = []
    for row in question["retrieved_chunks"]:
        chunk_id = str(row["chunk_id"])
        metadata = indexed_chunks.get(chunk_id)
        if metadata is None:
            raise ValueError(f"Indexed text is missing for retrieved chunk {chunk_id}.")
        contexts.append(
            RetrievedContext(
                source_id=f"S{int(row['rank'])}",
                document_id=str(row["document_id"]),
                chunk_id=chunk_id,
                title=str(row.get("document_title") or metadata.get("document_title", "")),
                domain=str(metadata.get("domain", "")),
                text=str(metadata["text"]),
            )
        )
    return tuple(contexts)


def _metric(value: float | None) -> str:
    return "No factual claims" if value is None else f"{value:.3f}"


def build_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Claim-level faithfulness audit",
        "",
        "This report reuses saved answers and exact indexed chunk text. It does not run retrieval or generation.",
        "",
        "| Model | Existing Faithfulness | Claim-Level Faithfulness | Relevance | Unsupported Claims / Answer |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in payload["model_summary"]:
        lines.append(
            f"| {summary['model']} | {summary['existing_faithfulness']:.3f} | "
            f"{_metric(summary['claim_level_faithfulness'])} | "
            f"{summary['relevance_correctness']:.3f} | "
            f"{summary['unsupported_claims_per_answer']:.3f} |"
        )
    lines.extend(
        (
            "",
            "## Conclusion",
            "",
            f"**{payload['conclusion']['code']} — {payload['conclusion']['label']}**",
            "",
            payload["conclusion"]["reason"],
            "",
            "## Unsupported-claim categories",
            "",
        )
    )
    categories = tuple(payload["model_summary"][0]["category_counts"])
    lines.extend(
        (
            "| Model | " + " | ".join(categories) + " |",
            "|---|" + "---:|" * len(categories),
        )
    )
    for summary in payload["model_summary"]:
        lines.append(
            f"| {summary['model']} | "
            + " | ".join(str(summary["category_counts"][category]) for category in categories)
            + " |"
        )

    lines.extend(("", "## Question-level audit", ""))
    for model in payload["models"]:
        lines.extend((f"### {model['model']}", ""))
        for item in model["questions"]:
            warning = (
                " — Evaluator disagreement — manual inspection recommended"
                if item["evaluator_disagreement"]
                else ""
            )
            lines.extend(
                (
                    f"<details><summary>{item['question_id']}: {item['question']}{warning}</summary>",
                    "",
                    "#### QUESTION",
                    "",
                    item["question"],
                    "",
                    "#### RETRIEVED CONTEXT",
                    "",
                )
            )
            for context in item["retrieved_chunks"]:
                lines.extend(
                    (
                        f"**{context['source_id']} — {context['title']} ({context['chunk_id']})**",
                        "",
                        "```text",
                        context["text"],
                        "```",
                        "",
                    )
                )
            lines.extend(
                (
                    "#### GENERATED ANSWER",
                    "",
                    "```text",
                    item["generated_answer"],
                    "```",
                    "",
                )
            )
            if not item["claims"]:
                lines.extend(("No factual claims (explicit refusal or empty answer).", ""))
            for index, claim in enumerate(item["claims"], start=1):
                lines.extend(
                    (
                        f"#### CLAIM {index}",
                        "",
                        f"**Text:** {claim['claim']}",
                        "",
                        f"**Supported:** {'Yes' if claim['supported'] else 'No'}",
                        "",
                        f"**Supporting sources:** {', '.join(claim['supporting_source_ids']) or 'None'}",
                        "",
                        f"**Evidence:** {claim['supporting_evidence'] or 'None'}",
                        "",
                        f"**Reason:** {claim['reason']}",
                        "",
                        f"**Relevance:** {claim['relevance_reason']}",
                        "",
                        f"**Category:** {claim['category'] or 'supported and relevant'}",
                        "",
                    )
                )
            lines.extend(
                (
                    f"**AUTOMATED FAITHFULNESS:** {item['automated_faithfulness']:.3f}",
                    "",
                    f"**CLAIM-LEVEL FAITHFULNESS:** {_metric(item['claim_faithfulness'])}",
                    "",
                    "</details>",
                    "",
                )
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    settings = Settings.from_environment(PROJECT_ROOT)
    saved_results = load_saved_results()
    print("Fetching exact indexed chunk text; saved answers will not be regenerated...")
    indexed_chunks = fetch_indexed_chunks(settings)
    models: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for experiment in saved_results:
        model = str(experiment["active_model"]["model"])
        questions = [
            audit_question(
                question,
                question_contexts(question, indexed_chunks),
                model,
                str(experiment["evaluation_dataset"].get("reference_date") or settings.reference_date),
            )
            for question in experiment["questions"]
        ]
        summary = summarize_model(experiment, questions)
        summaries.append(summary)
        models.append(
            {
                "experiment_id": experiment["experiment_id"],
                "model": model,
                "saved_result_path": experiment["_artifact_path"].relative_to(PROJECT_ROOT).as_posix(),
                "saved_result_sha256": experiment["_artifact_sha256"],
                "questions": questions,
            }
        )
        print(
            f"{model}: existing={summary['existing_faithfulness']:.3f}; "
            f"claim-level={_metric(summary['claim_level_faithfulness'])}; "
            f"unsupported={summary['unsupported_factual_claims']}; "
            f"disagreements={summary['evaluator_disagreement_count']}"
        )

    payload = {
        "schema_version": "1.0",
        "audit_type": "claim_level_faithfulness",
        "completed_at": utc_now(),
        "evaluation_only": True,
        "answers_regenerated": False,
        "retrieval_rerun": False,
        "retrieval_cache_sha256": saved_results[0]["retrieval_cache_sha256"],
        "evaluation_dataset": saved_results[0]["evaluation_dataset"],
        "scoring_method": {
            "claim_faithfulness": "supported factual claims / total factual claims",
            "no_factual_claims": "reported as null and excluded from the claim denominator",
            "support": "explicit retrieved text with matching dates, times, people, and factual-term coverage",
            "disagreement_threshold": 0.25,
        },
        "model_summary": summaries,
        "conclusion": determine_conclusion(summaries),
        "models": models,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(build_report(payload), encoding="utf-8")
    print(f"Results: {OUTPUT_PATH}")
    print(f"Detailed report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
