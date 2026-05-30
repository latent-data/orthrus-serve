from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from .settings import settings

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    # Number of times model.forward was called during generate(). In AR mode this
    # is ~completion_tokens (1 prefill + 1 per new token); in diffusion mode the
    # drafter proposes a block per forward, so completion_tokens / forward_count
    # = tokens-per-forward (TPF) is a direct measure of drafter accept rate. A
    # diffusion-labelled run with TPF ≈ 1.0 means the drafter isn't firing.
    forward_count: int


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
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    prompt_len = input_ids.shape[1]

    do_sample = temperature is not None and temperature > 0.0

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
    }
    if not settings.base_model:
        generate_kwargs["use_diffusion_mode"] = settings.diffusion_enabled
    if do_sample:
        generate_kwargs["temperature"] = temperature
        if top_p is not None:
            generate_kwargs["top_p"] = top_p

    if stop:
        generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [StringStoppingCriteria(stop, tokenizer, prompt_len)]
        )

    loggable = {k: v for k, v in generate_kwargs.items() if k != "stopping_criteria"}
    logger.debug("generate_kwargs %s", loggable)

    forward_count = 0

    def _count_forward(_module, _args, _kwargs):
        nonlocal forward_count
        forward_count += 1

    hook = model.register_forward_pre_hook(_count_forward, with_kwargs=True)
    try:
        with torch.inference_mode():
            output_ids = model.generate(input_ids=input_ids, **generate_kwargs)
    finally:
        hook.remove()

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
        forward_count=forward_count,
    )
