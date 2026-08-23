from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import harmonic_mean
from typing import Any

import yaml

from .chat_models import ChatModelConfig


@dataclass(frozen=True)
class GenerationModelExperiment:
    experiment_id: str
    label: str
    provider: str
    api_style: str
    model: str
    model_env: str | None
    base_url: str
    base_url_env: str | None
    api_key: str
    api_key_env: str | None
    timeout_seconds: float
    max_retries: int
    config_path: Path
    config_sha256: str

    def chat_model_config(self) -> ChatModelConfig:
        return ChatModelConfig(
            provider=self.provider,
            api_style=self.api_style,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            api_key_env=self.api_key_env,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
        )

    def public_config(self, project_root: Path) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "label": self.label,
            "provider": self.provider,
            "api_style": self.api_style,
            "model": self.model,
            "model_env": self.model_env,
            "base_url": self.base_url,
            "base_url_env": self.base_url_env,
            "api_key_env": self.api_key_env,
            "api_key_configured": bool(self.api_key.strip()),
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "config_path": self.config_path.relative_to(project_root).as_posix(),
            "config_sha256": self.config_sha256,
        }


def _environment_value(
    payload: dict[str, Any],
    value_name: str,
) -> tuple[str, str | None]:
    environment_name = payload.get(f"{value_name}_env")
    default = str(payload.get(f"{value_name}_default", "") or "")
    if environment_name is None:
        return default, None
    environment_name = str(environment_name).strip()
    return os.getenv(environment_name, default).strip(), environment_name


def load_generation_model_experiments(
    config_dir: Path,
) -> tuple[GenerationModelExperiment, ...]:
    experiments: list[GenerationModelExperiment] = []
    seen_ids: set[str] = set()
    for path in sorted(config_dir.glob("*.yaml")):
        raw = path.read_bytes()
        payload = yaml.safe_load(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"Model experiment config must be a mapping: {path}")
        experiment_id = str(payload.get("experiment_id", "")).strip()
        if not experiment_id:
            raise ValueError(f"Model experiment is missing experiment_id: {path}")
        if experiment_id in seen_ids:
            raise ValueError(f"Duplicate model experiment ID: {experiment_id}")
        seen_ids.add(experiment_id)
        model, model_env = _environment_value(payload, "model")
        base_url, base_url_env = _environment_value(payload, "base_url")
        api_key_env_value = payload.get("api_key_env")
        api_key_env = (
            str(api_key_env_value).strip()
            if api_key_env_value is not None
            else None
        )
        api_key = os.getenv(api_key_env, "").strip() if api_key_env else ""
        experiments.append(
            GenerationModelExperiment(
                experiment_id=experiment_id,
                label=str(payload.get("label", experiment_id)),
                provider=str(payload.get("provider", "")).strip(),
                api_style=str(payload.get("api_style", "")).strip(),
                model=model,
                model_env=model_env,
                base_url=base_url,
                base_url_env=base_url_env,
                api_key=api_key,
                api_key_env=api_key_env,
                timeout_seconds=float(payload.get("timeout_seconds", 30)),
                max_retries=int(payload.get("max_retries", 0)),
                config_path=path.resolve(),
                config_sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
    if not experiments:
        raise ValueError(f"No generation model configs found in {config_dir}.")
    return tuple(experiments)


def model_is_eligible(result: dict[str, Any]) -> bool:
    metrics = result["metrics"]
    return (
        result.get("run_status") == "complete"
        and int(metrics.get("generation_failure_count", 0)) == 0
        and float(metrics.get("correct_refusal_rate", 0.0)) == 1.0
    )


def build_model_comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [result for result in results if model_is_eligible(result)]
    if not eligible:
        highlights = {
            "highest_faithfulness": [],
            "highest_relevance_correctness": [],
            "lowest_average_latency": [],
            "best_overall_balance": [],
        }
        return {"eligible_experiment_ids": [], "highlights": highlights, "balance": {}}

    def winners(metric: str, *, minimize: bool = False) -> list[str]:
        values = [float(item["metrics"][metric]) for item in eligible]
        best = min(values) if minimize else max(values)
        return [
            str(item["experiment_id"])
            for item in eligible
            if float(item["metrics"][metric]) == best
        ]

    balance_scores: dict[str, float] = {}
    for item in eligible:
        metrics = item["metrics"]
        latency_component = min(
            1.0,
            5.0 / max(float(metrics["p95_latency_seconds"]), 0.001),
        )
        components = (
            max(float(metrics["faithfulness"]), 0.000001),
            max(float(metrics["answer_relevance_correctness"]), 0.000001),
            max(latency_component, 0.000001),
        )
        balance_scores[str(item["experiment_id"])] = harmonic_mean(components)
    best_balance = max(balance_scores.values())
    return {
        "eligible_experiment_ids": [str(item["experiment_id"]) for item in eligible],
        "highlights": {
            "highest_faithfulness": winners("faithfulness"),
            "highest_relevance_correctness": winners(
                "answer_relevance_correctness"
            ),
            "lowest_average_latency": winners(
                "average_latency_seconds",
                minimize=True,
            ),
            "best_overall_balance": [
                experiment_id
                for experiment_id, score in balance_scores.items()
                if score == best_balance
            ],
        },
        "balance": {
            "method": (
                "harmonic mean of faithfulness, relevance/correctness, and a P95 "
                "latency component capped at 1.0 (5 seconds / measured P95); only "
                "complete runs with correct refusal 1.0 are eligible"
            ),
            "scores": balance_scores,
        },
    }
