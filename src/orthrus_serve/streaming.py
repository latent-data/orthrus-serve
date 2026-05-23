from __future__ import annotations

import json
import logging
import secrets
import time
from collections.abc import AsyncGenerator
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

from .openai_schemas import ChatCompletionChunk, DeltaMessage, StreamChoice

logger = logging.getLogger(__name__)

TOOL_CALL_OPEN = "<tool_call>"


async def stream_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int,
    stop: list[str] | None,
    completion_id: str,
) -> AsyncGenerator[str, None]:
    """
    Yield SSE data lines for a streaming chat completion.

    If a <tool_call> token is detected mid-stream, the caller should have
    already decided to use the non-streaming path. This function is only
    called for requests without tools, or when tool_choice=none.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    do_sample = temperature is not None and temperature > 0.0
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "use_diffusion_mode": True,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        if top_p is not None:
            generate_kwargs["top_p"] = top_p

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs["streamer"] = streamer

    thread = Thread(
        target=_run_generate,
        args=(model, inputs, generate_kwargs),
        daemon=True,
    )
    thread.start()

    created = int(time.time())
    accumulated = ""
    stop_strings = stop or []

    # First chunk: role
    first_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        choices=[StreamChoice(delta=DeltaMessage(role="assistant"), finish_reason=None)],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    finished = False
    for token_text in streamer:
        if finished:
            break
        accumulated += token_text

        # Check stop strings
        stop_hit = False
        for s in stop_strings:
            idx = accumulated.find(s)
            if idx != -1:
                token_text = accumulated[:idx][len(accumulated) - len(token_text):]
                accumulated = accumulated[:idx]
                stop_hit = True
                break

        if token_text:
            chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                choices=[StreamChoice(delta=DeltaMessage(content=token_text), finish_reason=None)],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        if stop_hit:
            finished = True

    # Final chunk with finish_reason
    final_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


def _run_generate(model, inputs, generate_kwargs):
    with torch.inference_mode():
        model.generate(**inputs, **generate_kwargs)
