from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class StringStoppingCriteria(StoppingCriteria):
    """Stop generation when any stop string appears in the decoded output."""

    def __init__(self, stop_strings: list[str], tokenizer: AutoTokenizer, prompt_len: int):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Decode only the newly generated tokens for efficiency
        generated = self.tokenizer.decode(
            input_ids[0][self.prompt_len :], skip_special_tokens=True
        )
        return any(s in generated for s in self.stop_strings)


def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int,
    stop: list[str] | None,
) -> GenerationResult:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    do_sample = temperature is not None and temperature > 0.0

    use_base_model = os.environ.get("ORTHRUS_BASE_MODEL", "0") == "1"
    use_diffusion = (
        not use_base_model and os.environ.get("ORTHRUS_DIFFUSION", "1") != "0"
    )

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
    }
    if use_diffusion:
        generate_kwargs["use_diffusion_mode"] = True
    if do_sample:
        generate_kwargs["temperature"] = temperature
        if top_p is not None:
            generate_kwargs["top_p"] = top_p

    if stop:
        generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [StringStoppingCriteria(stop, tokenizer, prompt_len)]
        )

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)

    # Slice off the prompt tokens; decode only the completion
    new_ids = output_ids[0][prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)

    # Trim at stop strings if present (stopping criteria fires at token boundary,
    # may include a partial stop string)
    if stop:
        for s in stop:
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]

    return GenerationResult(
        text=text,
        prompt_tokens=prompt_len,
        completion_tokens=len(new_ids),
    )
