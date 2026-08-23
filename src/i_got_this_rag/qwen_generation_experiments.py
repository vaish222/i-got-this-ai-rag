from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .chat_models import ChatModelConfig
from .concise_generation import (
    EVIDENCE_MODES,
    PROMPT_MODES,
    AnswerLengthPolicy,
)
from .generation_model_experiments import load_generation_model_experiments


@dataclass(frozen=True)
class QwenGenerationExperiment:
    experiment_id: str
    label: str
    prompt_mode: str
    evidence_mode: str
    length_policy: AnswerLengthPolicy
    chat_config: ChatModelConfig
    public_model_config: dict[str, Any]
    config_path: Path
    config_sha256: str

    def public_config(self, project_root: Path) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "label": self.label,
            "prompt_mode": self.prompt_mode,
            "evidence_mode": self.evidence_mode,
            "length_policy": self.length_policy.to_dict(),
            "model": self.public_model_config,
            "config_path": self.config_path.relative_to(project_root).as_posix(),
            "config_sha256": self.config_sha256,
        }


def load_qwen_generation_experiments(
    config_dir: Path,
    project_root: Path,
) -> tuple[QwenGenerationExperiment, ...]:
    model_configs = load_generation_model_experiments(config_dir)
    experiments: list[QwenGenerationExperiment] = []
    for model_config in model_configs:
        payload = yaml.safe_load(model_config.config_path.read_text(encoding="utf-8"))
        expected_model = str(payload.get("expected_model", "")).strip()
        if expected_model and model_config.model != expected_model:
            raise ValueError(
                f"{model_config.experiment_id} must use {expected_model}; "
                f"resolved {model_config.model or '<empty>'}."
            )
        prompt_mode = str(payload.get("prompt_mode", "")).strip()
        evidence_mode = str(payload.get("evidence_mode", "")).strip()
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(
                f"Unsupported prompt_mode '{prompt_mode}' in {model_config.config_path}."
            )
        if evidence_mode not in EVIDENCE_MODES:
            raise ValueError(
                f"Unsupported evidence_mode '{evidence_mode}' in {model_config.config_path}."
            )
        raw_policy = payload.get("length_policy") or {}
        if not isinstance(raw_policy, dict):
            raise ValueError(f"length_policy must be a mapping: {model_config.config_path}")
        policy = AnswerLengthPolicy(
            exact_lookup=int(raw_policy.get("exact_lookup", 3)),
            yes_no=int(raw_policy.get("yes_no", 2)),
            schedule_lookup=int(raw_policy.get("schedule_lookup", 5)),
            cross_domain_summary=int(raw_policy.get("cross_domain_summary", 8)),
            planning_request=int(raw_policy.get("planning_request", 8)),
        )
        if any(value <= 0 for value in policy.to_dict().values()):
            raise ValueError(f"All answer-length limits must be positive: {model_config.config_path}")
        experiments.append(
            QwenGenerationExperiment(
                experiment_id=model_config.experiment_id,
                label=model_config.label,
                prompt_mode=prompt_mode,
                evidence_mode=evidence_mode,
                length_policy=policy,
                chat_config=model_config.chat_model_config(),
                public_model_config=model_config.public_config(project_root),
                config_path=model_config.config_path,
                config_sha256=model_config.config_sha256,
            )
        )
    models = {item.chat_config.model for item in experiments}
    providers = {item.chat_config.provider for item in experiments}
    if len(models) != 1 or len(providers) != 1:
        raise ValueError("E1/E2/E3 must use one identical generation model and provider.")
    if len(experiments) != 3:
        raise ValueError("The Qwen generation experiment requires exactly E1, E2, and E3.")
    return tuple(experiments)


def mode_succeeds(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics["recall_at_5"]) >= 0.90
        and float(metrics["claim_level_faithfulness"]) >= 0.95
        and float(metrics["answer_relevance_correctness"]) > 0.588
        and float(metrics["correct_refusal_rate"]) == 1.0
        and int(metrics.get("generation_failure_count", 0)) == 0
    )
