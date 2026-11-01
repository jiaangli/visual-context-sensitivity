"""Hugging Face generation policies for the project-native VLM evaluations."""

from __future__ import annotations

from typing import Any

import torch
from transformers import LogitsProcessor, LogitsProcessorList

# ``None`` means to pass no decoding kwargs and use the checkpoint's
# ``generation_config.json`` exactly as loaded by Transformers.
OFFICIAL_EVAL_GENERATION: dict[str, dict[str, float | bool] | None] = {
    "qwen2_5_vl": None,
    "qwen3_5": {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "gemma4": {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
    },
}

_GENERATION_CONFIG_KEYS = (
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
)


class GeneratedTokenPresencePenalty(LogitsProcessor):
    """Implement OpenAI/vLLM presence penalty for Transformers generation.

    Hugging Face Transformers does not expose a ``presence_penalty`` field.
    Qwen's recommended value is therefore applied by subtracting a fixed value
    once from each token that has already occurred in the generated completion.
    Prompt tokens are intentionally excluded.
    """

    def __init__(self, penalty: float, prompt_length: int) -> None:
        self.penalty = float(penalty)
        self.prompt_length = int(prompt_length)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        generated_ids = input_ids[:, self.prompt_length :]
        if generated_ids.numel() == 0 or self.penalty == 0.0:
            return scores

        appeared = torch.zeros_like(scores, dtype=torch.bool)
        appeared.scatter_(1, generated_ids, True)
        return scores - appeared.to(dtype=scores.dtype) * self.penalty


def resolve_eval_model_family(model: Any) -> str:
    """Resolve one of the supported model families from a Transformers model."""
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type in OFFICIAL_EVAL_GENERATION:
        return model_type

    identity = " ".join(
        (
            model_type,
            type(model).__name__,
            str(getattr(config, "name_or_path", "")),
            str(getattr(model, "name_or_path", "")),
        )
    ).lower()
    if "qwen2.5-vl" in identity or "qwen2_5_vl" in identity:
        return "qwen2_5_vl"
    if "qwen3.5" in identity or "qwen3_5" in identity:
        return "qwen3_5"
    if "gemma-4" in identity or "gemma4" in identity:
        return "gemma4"

    supported = ", ".join(sorted(OFFICIAL_EVAL_GENERATION))
    raise ValueError(
        f"No project eval generation policy for model_type={model_type!r}; supported families: {supported}"
    )


def build_eval_generation_kwargs(
    model: Any,
    input_ids: torch.Tensor,
) -> dict[str, Any]:
    """Build model-author decoding kwargs for one HF ``generate`` call."""
    family = resolve_eval_model_family(model)
    policy = OFFICIAL_EVAL_GENERATION[family]
    if policy is None:
        return {}

    kwargs: dict[str, Any] = dict(policy)
    presence_penalty = float(kwargs.pop("presence_penalty", 0.0))
    if presence_penalty != 0.0:
        if input_ids.ndim < 2:
            raise ValueError("presence_penalty requires batched input_ids")
        kwargs["logits_processor"] = LogitsProcessorList(
            [
                GeneratedTokenPresencePenalty(
                    presence_penalty,
                    prompt_length=int(input_ids.shape[-1]),
                )
            ]
        )
    return kwargs


def describe_eval_generation(model: Any) -> dict[str, Any]:
    """Return JSON-serializable generation provenance for result artifacts."""
    family = resolve_eval_model_family(model)
    policy = OFFICIAL_EVAL_GENERATION[family]
    if policy is not None:
        parameters = dict(policy)
        source = "model_author_recommendation"
    else:
        generation_config = getattr(model, "generation_config", None)
        if generation_config is None:
            raise ValueError("Qwen2.5-VL evaluation requires the checkpoint generation_config")
        config = generation_config.to_dict()
        parameters = {key: config[key] for key in _GENERATION_CONFIG_KEYS if config.get(key) is not None}
        source = "huggingface_checkpoint_generation_config"

    return {
        "backend": "huggingface_transformers",
        "model_family": family,
        "source": source,
        "parameters": parameters,
    }
