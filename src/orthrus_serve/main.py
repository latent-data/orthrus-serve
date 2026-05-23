from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .generation import generate
from .model import load_model_and_tokenizer
from .openai_schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ModelCard,
    ModelList,
    ResponseMessage,
    ToolCall,
    FunctionCall,
    Usage,
)
from .streaming import stream_completion
from .tool_parse import parse_tool_calls, strip_think_tags

DEBUG = os.environ.get("ORTHRUS_DEBUG", "0") == "1"
BASE_MODEL = os.environ.get("ORTHRUS_BASE_MODEL", "0") == "1"
SERVED_MODEL_ID = "qwen3-8b" if BASE_MODEL else "orthrus-qwen3-8b"

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":%(message)s}',
)
logger = logging.getLogger("orthrus_serve")

# Prometheus metrics
REQUEST_COUNTER = Counter("orthrus_requests_total", "Total requests")
TOOL_CALL_COUNTER = Counter("orthrus_tool_call_requests_total", "Requests that produced tool calls")
LATENCY_HIST = Histogram(
    "orthrus_request_duration_seconds",
    "Request duration",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120],
)
TTFT_HIST = Histogram(
    "orthrus_ttft_seconds",
    "Time to first token",
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5],
)

_model = None
_tokenizer = None
_ready = False
_request_semaphore: asyncio.Semaphore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _ready, _request_semaphore
    _request_semaphore = asyncio.Semaphore(1)
    loop = asyncio.get_event_loop()
    _model, _tokenizer = await loop.run_in_executor(None, load_model_and_tokenizer)
    _ready = True
    logger.info('"Model ready"')
    yield


app = FastAPI(title="orthrus-serve", lifespan=lifespan)


@app.get("/health")
async def health():
    if not _ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return ModelList(
        data=[
            ModelCard(
                id=SERVED_MODEL_ID,
                owned_by="latent-data",
                created=int(time.time()),
                max_model_len=40960,
            )
        ]
    )


@app.get("/metrics")
async def metrics():
    return StreamingResponse(
        iter([generate_latest()]),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    if not _ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    REQUEST_COUNTER.inc()
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    t_start = time.perf_counter()

    # Normalise stop to a list
    stop: list[str] = []
    if isinstance(request.stop, str):
        stop = [request.stop]
    elif isinstance(request.stop, list):
        stop = request.stop

    max_tokens = request.max_tokens if request.max_tokens is not None else 2048
    temperature = request.temperature
    top_p = request.top_p
    chat_template_kwargs = request.chat_template_kwargs or {}

    # Build messages as plain dicts for apply_chat_template
    messages = [m.model_dump(exclude_none=True) for m in request.messages]

    # Serialise tools for the template
    tools_raw = None
    if request.tools:
        tools_raw = [t.model_dump() for t in request.tools]

    has_tools = bool(tools_raw)

    prompt = _tokenizer.apply_chat_template(
        messages,
        tools=tools_raw,
        add_generation_prompt=True,
        tokenize=False,
        **chat_template_kwargs,
    )

    if DEBUG:
        logger.debug(json.dumps({"request_id": request_id, "event": "request", "body": request.model_dump()}))
        logger.debug(json.dumps({"request_id": request_id, "event": "prompt", "prompt": prompt}))

    # Plain streaming: no tools, client wants SSE
    if request.stream and not has_tools:
        return StreamingResponse(
            stream_completion(
                _model, _tokenizer, prompt, temperature, top_p, max_tokens, stop, request_id,
            ),
            media_type="text/event-stream",
        )

    # Buffered path: tools present (must see full output to parse tool calls),
    # or client did not request streaming.
    #
    # If the client requested streaming we also need to keep the connection alive
    # while generating — diffusion-mode generation can take 60-90s and HTTP
    # clients time out waiting for the first byte. We do this by returning an
    # SSE StreamingResponse that ticks keepalive comments while the GPU works,
    # then emits the actual result chunks when generation completes.
    if request.stream:
        return StreamingResponse(
            _buffered_sse(
                request_id, t_start, prompt, temperature, top_p, max_tokens, stop
            ),
            media_type="text/event-stream",
        )

    try:
        async with asyncio.timeout(300):
            async with _request_semaphore:
                t_generate_start = time.perf_counter()
                loop = asyncio.get_event_loop()
                gpu_future = loop.run_in_executor(
                    None,
                    lambda: generate(
                        _model, _tokenizer, prompt, temperature, top_p, max_tokens, stop
                    ),
                )
                try:
                    # shield ensures that if this request is cancelled (client disconnect),
                    # we still wait for the GPU thread to finish before releasing the
                    # semaphore — preventing a second generate() call overlapping on the GPU.
                    result = await asyncio.shield(gpu_future)
                except asyncio.CancelledError:
                    await gpu_future
                    raise
                t_first_token = time.perf_counter() - t_generate_start
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Request timed out")

    t_total = time.perf_counter() - t_start
    LATENCY_HIST.observe(t_total)
    TTFT_HIST.observe(t_first_token)

    if DEBUG:
        logger.debug(json.dumps({"request_id": request_id, "event": "raw_output", "text": result.text}))

    output_text = strip_think_tags(result.text)
    tool_calls_raw, content = parse_tool_calls(output_text)

    finish_reason = "stop"
    tool_calls_out = None

    if tool_calls_raw:
        TOOL_CALL_COUNTER.inc()
        finish_reason = "tool_calls"
        tool_calls_out = [
            ToolCall(
                id=tc["id"],
                function=FunctionCall(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in tool_calls_raw
        ]
        content = content if content else None
    else:
        content = output_text

    response = ChatCompletionResponse(
        id=request_id,
        model=SERVED_MODEL_ID,
        choices=[
            Choice(
                message=ResponseMessage(content=content, tool_calls=tool_calls_out),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )

    if DEBUG:
        logger.debug(json.dumps({"request_id": request_id, "event": "response", "body": response.model_dump()}))

    logger.info(
        json.dumps(
            {
                "request_id": request_id,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "ttft_s": round(t_first_token, 3),
                "total_s": round(t_total, 3),
                "tool_calls": bool(tool_calls_raw),
                "finish_reason": finish_reason,
                "orthrus_revision": os.environ.get("ORTHRUS_REVISION", "default"),
            }
        )
    )

    return response


async def _buffered_sse(request_id, t_start, prompt, temperature, top_p, max_tokens, stop):
    """
    Streaming response for tool-call requests.

    Runs generation in an executor thread while emitting SSE keepalive comments
    every 5 s so the client's read-timeout doesn't fire during long diffusion-mode
    generations. The full output is buffered, parsed for tool calls, then emitted
    as structured SSE chunks.
    """
    loop = asyncio.get_event_loop()
    created = int(time.time())

    def _make_chunk(delta: dict) -> str:
        return f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created, 'model': SERVED_MODEL_ID, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]})}\n\n"

    # Role chunk sent immediately so the client knows the stream is live.
    yield _make_chunk({"role": "assistant", "content": None})

    t_generate_start = time.perf_counter()

    async with _request_semaphore:
        gpu_future = loop.run_in_executor(
            None,
            lambda: generate(
                _model, _tokenizer, prompt, temperature, top_p, max_tokens, stop
            ),
        )
        try:
            while True:
                try:
                    result = await asyncio.wait_for(asyncio.shield(gpu_future), timeout=5.0)
                    break
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        except asyncio.CancelledError:
            # Client disconnected — drain the GPU before releasing the semaphore
            # so no second generate() call overlaps on the GPU.
            await gpu_future
            raise

    t_first_token = time.perf_counter() - t_generate_start
    t_total = time.perf_counter() - t_start
    LATENCY_HIST.observe(t_total)
    TTFT_HIST.observe(t_first_token)

    if DEBUG:
        logger.debug(json.dumps({"request_id": request_id, "event": "raw_output", "text": result.text}))

    output_text = strip_think_tags(result.text)
    tool_calls_raw, content = parse_tool_calls(output_text)

    finish_reason = "stop"
    tool_calls_out = None

    if tool_calls_raw:
        TOOL_CALL_COUNTER.inc()
        finish_reason = "tool_calls"
        tool_calls_out = [
            ToolCall(
                id=tc["id"],
                function=FunctionCall(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in tool_calls_raw
        ]

    logger.info(
        json.dumps(
            {
                "request_id": request_id,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "ttft_s": round(t_first_token, 3),
                "total_s": round(t_total, 3),
                "tool_calls": bool(tool_calls_raw),
                "finish_reason": finish_reason,
                "orthrus_revision": os.environ.get("ORTHRUS_REVISION", "default"),
            }
        )
    )

    if tool_calls_out:
        for i, tc in enumerate(tool_calls_out):
            yield _make_chunk({
                "tool_calls": [{"index": i, "id": tc.id, "type": "function",
                                "function": {"name": tc.function.name, "arguments": ""}}]
            })
            yield _make_chunk({
                "tool_calls": [{"index": i, "function": {"arguments": tc.function.arguments}}]
            })
    else:
        if content:
            yield _make_chunk({"content": content})

    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created, 'model': SERVED_MODEL_ID, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}]})}\n\n"
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
