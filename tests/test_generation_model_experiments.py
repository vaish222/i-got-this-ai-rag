from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from i_got_this_rag.chat_models import (  # noqa: E402
    API_STYLE_OLLAMA,
    API_STYLE_OPENAI_COMPATIBLE,
    ChatModelConfig,
    ChatModelConfigurationError,
    MissingChatModelAPIKeyError,
    get_chat_model,
)
from i_got_this_rag.current_app_evaluation import (  # noqa: E402
    classify_generation_error,
    evaluate_current_app,
)
from i_got_this_rag.evaluation import EvaluationDataset  # noqa: E402
from i_got_this_rag.generation_model_experiments import (  # noqa: E402
    build_model_comparison,
    load_generation_model_experiments,
)
from i_got_this_rag.settings import Settings  # noqa: E402


class GenerationModelExperimentTests(unittest.TestCase):
    def test_factory_supports_ollama_and_openai_compatible_models(self) -> None:
        local = get_chat_model(
            ChatModelConfig(
                provider="ollama",
                api_style=API_STYLE_OLLAMA,
                model="local-model",
                base_url="http://localhost:11434",
            )
        )
        hosted = get_chat_model(
            ChatModelConfig(
                provider="nebius",
                api_style=API_STYLE_OPENAI_COMPATIBLE,
                model="configured-model",
                base_url="https://example.test/v1/",
                api_key="secret-value",
            )
        )

        self.assertIsInstance(local, ChatOllama)
        self.assertIsInstance(hosted, ChatOpenAI)

    def test_factory_rejects_missing_key_and_invalid_provider_style(self) -> None:
        with self.assertRaises(MissingChatModelAPIKeyError):
            get_chat_model(
                ChatModelConfig(
                    provider="nebius",
                    api_style=API_STYLE_OPENAI_COMPATIBLE,
                    model="configured-model",
                    base_url="https://example.test/v1/",
                )
            )
        with self.assertRaises(ChatModelConfigurationError):
            get_chat_model(
                ChatModelConfig(
                    provider="provider",
                    api_style="unknown",
                    model="configured-model",
                    base_url="https://example.test/v1/",
                )
            )

    def test_application_settings_use_generic_llm_environment_and_redact_key(self) -> None:
        environment = {
            "LLM_PROVIDER": "nebius",
            "LLM_API_STYLE": "openai_compatible",
            "LLM_MODEL": "configured-model",
            "LLM_API_KEY": "do-not-serialize",
            "LLM_BASE_URL": "https://example.test/v1/",
            "LLM_TIMEOUT_SECONDS": "12",
        }
        with patch.dict(os.environ, environment, clear=False):
            settings = Settings.from_environment(PROJECT_ROOT)

        self.assertEqual(settings.llm_provider, "nebius")
        self.assertEqual(settings.llm_model, "configured-model")
        self.assertEqual(settings.chat_model_config().timeout_seconds, 12)
        self.assertNotIn("llm_api_key", settings.public_config())
        self.assertTrue(settings.public_config()["llm_api_key_configured"])

    def test_yaml_configs_resolve_model_names_from_environment_without_leaking_key(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            config_dir = Path(directory)
            (config_dir / "D2.yaml").write_text(
                "\n".join(
                    (
                        "experiment_id: D2_test",
                        "label: Test hosted model",
                        "provider: nebius",
                        "api_style: openai_compatible",
                        "model_env: TEST_NEBIUS_MODEL",
                        "base_url_default: https://example.test/v1/",
                        "api_key_env: TEST_NEBIUS_KEY",
                    )
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "TEST_NEBIUS_MODEL": "configured-model",
                    "TEST_NEBIUS_KEY": "do-not-serialize",
                },
                clear=False,
            ):
                experiment = load_generation_model_experiments(config_dir)[0]
                public = experiment.public_config(PROJECT_ROOT)

        self.assertEqual(experiment.model, "configured-model")
        self.assertTrue(public["api_key_configured"])
        self.assertNotIn("do-not-serialize", repr(public))

    def test_comparison_separates_metrics_and_disqualifies_refusal_regressions(self) -> None:
        def result(
            experiment_id: str,
            faithfulness: float,
            relevance: float,
            latency: float,
            refusal: float = 1.0,
        ) -> dict:
            return {
                "experiment_id": experiment_id,
                "run_status": "complete",
                "metrics": {
                    "faithfulness": faithfulness,
                    "answer_relevance_correctness": relevance,
                    "correct_refusal_rate": refusal,
                    "average_latency_seconds": latency,
                    "p95_latency_seconds": latency * 2,
                    "generation_failure_count": 0,
                },
            }

        comparison = build_model_comparison(
            [
                result("D1", 0.5, 0.8, 1.0),
                result("D2", 0.9, 0.7, 1.5),
                result("D3", 1.0, 1.0, 0.5, refusal=0.0),
            ]
        )

        self.assertEqual(comparison["highlights"]["highest_faithfulness"], ["D2"])
        self.assertEqual(
            comparison["highlights"]["highest_relevance_correctness"], ["D1"]
        )
        self.assertEqual(comparison["highlights"]["lowest_average_latency"], ["D1"])
        self.assertNotIn("D3", comparison["eligible_experiment_ids"])

    def test_generation_failures_are_classified_and_evaluation_continues(self) -> None:
        class FailingPipeline:
            settings = SimpleNamespace(reference_date="2026-08-20")
            resources = SimpleNamespace(llm=None)

            def retrieve(self, question: str) -> list[tuple[Document, float]]:
                del question
                return [
                    (
                        Document(
                            page_content="Alpha is scheduled.",
                            metadata={
                                "document_id": "doc-1",
                                "document_title": "Alpha",
                                "chunk_id": "doc-1::chunk-0001",
                                "source_path": "alpha.md",
                            },
                        ),
                        0.9,
                    )
                ]

            def generate(self, question: str, results: object) -> str:
                del question, results
                raise TimeoutError("provider timed out")

        dataset = EvaluationDataset(
            path=Path("questions.json"),
            schema_version="1",
            dataset_name="test",
            reference_date="2026-08-20",
            timezone="America/Los_Angeles",
            questions=(
                {
                    "question_id": "Q1",
                    "question": "Tell me about alpha.",
                    "expected_answer": "Alpha is scheduled.",
                    "expected_source_ids": ["doc-1"],
                    "expected_sources": ["alpha.md"],
                    "category": "test",
                    "answerable": True,
                },
            ),
            sha256="test",
        )
        measured = evaluate_current_app(
            FailingPipeline(),
            dataset,
            continue_on_generation_error=True,
        )

        self.assertEqual(measured["metrics"]["generation_failure_count"], 1)
        self.assertEqual(
            measured["questions"][0]["generation_error"]["type"],
            "api_timeout",
        )
        self.assertEqual(measured["questions"][0]["retrieved_source_ids"], ["doc-1"])
        self.assertEqual(classify_generation_error(ValueError("malformed response"))["type"], "malformed_model_response")


if __name__ == "__main__":
    unittest.main()
