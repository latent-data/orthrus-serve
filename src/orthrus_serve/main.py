from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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
    Usage,
)
from .tool_parse import parse_tool_calls, strip_think_tags

DEBUG = os.environ.get("ORTHRUS_DEBUG", "0") == "1"
BASE_MODEL = os.environ.get("ORTHRUS_BASE_MODEL", "0") == "1"
_thinking_env = os.environ.get("ORTHRUS_ENABLE_THINKING")
ENABLE_THINKING: bool | None = None if _thinking_env is None else (_thinking_env == "true")
SERVED_MODEL_ID = "qwen3-8b" if BASE_MODEL else "orthrus-qwen3-8b"

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":%(message)s}',
)
logger = logging.getLogger("orthrus_serve")

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
_request_lock: asyncio.Lock | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _ready, _request_lock
    _request_lock = asyncio.Lock()
    loop = asyncio.get_event_loop()
    _model, _tokenizer = await loop.run_in_executor(None, load_model_and_tokenizer)
    _ready = True
    _diffusion_on = not BASE_MODEL and os.environ.get("ORTHRUS_DIFFUSION", "1") != "0"
    logger.info(
        json.dumps({"event": "model_ready", "diffusion": _diffusion_on, "thinking": ENABLE_THINKING})
    )
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


def _dispatch_generate(prompt, temperature, top_p, max_tokens, stop):
    """Submit the generate call to the default executor; return the awaitable future."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(
        None,
        lambda: generate(_model, _tokenizer, prompt, temperature, top_p, max_tokens, stop),
    )


def _postprocess(result, request_id, t_first_token, t_total):
    """Strip think tags, parse tool calls, log INFO, observe Prometheus.

    Returns (tool_calls_out, content, finish_reason).
    """
    if DEBUG:
        logger.debug(json.dumps({"request_id": request_id, "event": "raw_output", "text": result.text}))

    output_text = strip_think_tags(result.text)
    tool_calls_raw, content = parse_tool_calls(output_text)

    finish_reason = "stop"
    tool_calls_out = None
    if tool_calls_raw:
        TOOL_CALL_COUNTER.inc()
        finish_reason = "tool_calls"
        tool_calls_out = [ToolCall(**tc) for tc in tool_calls_raw]
        content = content if content else None
    else:
        content = output_text

    LATENCY_HIST.observe(t_total)
    TTFT_HIST.observe(t_first_token)

    tok_per_s = round(result.completion_tokens / t_first_token, 2) if t_first_token > 0 else 0.0
    logger.info(
        json.dumps(
            {
                "request_id": request_id,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "ttft_s": round(t_first_token, 3),
                "total_s": round(t_total, 3),
                "tok_per_s": tok_per_s,
                "tool_calls": bool(tool_calls_out),
                "finish_reason": finish_reason,
                "orthrus_revision": os.environ.get("ORTHRUS_REVISION", "default"),
            }
        )
    )

    return tool_calls_out, content, finish_reason


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
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
    _thinking_default = {} if ENABLE_THINKING is None else {"enable_thinking": ENABLE_THINKING}
    chat_template_kwargs = {**_thinking_default, **(request.chat_template_kwargs or {})}

    messages = [m.model_dump(exclude_none=True) for m in request.messages]
    tools_raw = [t.model_dump() for t in request.tools] if request.tools else None

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

    # Streaming path: buffer generation, emit SSE keepalive pings every 5s so
    # the client's read-timeout doesn't fire during long diffusion-mode runs,
    # then emit the actual result chunks (with tool_calls if any) when done.
    # Diffusion mode generates in blocks of 32 so true token-by-token streaming
    # isn't meaningful — buffered + pings is the only viable shape here.
    if request.stream:
        return StreamingResponse(
            _buffered_sse(request_id, t_start, prompt, temperature, top_p, max_tokens, stop),
            media_type="text/event-stream",
        )

    try:
        async with asyncio.timeout(300):
            async with _request_lock:
                t_generate_start = time.perf_counter()
                gpu_future = _dispatch_generate(prompt, temperature, top_p, max_tokens, stop)
                try:
                    # shield ensures that if this request is cancelled (client disconnect),
                    # we still wait for the GPU thread to finish before releasing the
                    # lock — preventing a second generate() call overlapping on the GPU.
                    result = await asyncio.shield(gpu_future)
                except asyncio.CancelledError:
                    await gpu_future
                    raise
                t_first_token = time.perf_counter() - t_generate_start
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Request timed out")

    t_total = time.perf_counter() - t_start
    tool_calls_out, content, finish_reason = _postprocess(result, request_id, t_first_token, t_total)

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

    return response


async def _buffered_sse(request_id, t_start, prompt, temperature, top_p, max_tokens, stop):
    """SSE generator: role chunk → keepalive pings while GPU works → tool_calls / content chunks → finish + [DONE]."""
    created = int(time.time())

    def _chunk(delta: dict, finish_reason: str | None = None) -> str:
        body = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": SERVED_MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(body)}\n\n"

    yield _chunk({"role": "assistant", "content": None})

    t_generate_start = time.perf_counter()
    async with _request_lock:
        gpu_future = _dispatch_generate(prompt, temperature, top_p, max_tokens, stop)
        try:
            while True:
                try:
                    result = await asyncio.wait_for(asyncio.shield(gpu_future), timeout=5.0)
                    break
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        except asyncio.CancelledError:
            # Client disconnected — drain the GPU before releasing the lock
            # so no second generate() call overlaps on the GPU.
            await gpu_future
            raise

    t_first_token = time.perf_counter() - t_generate_start
    t_total = time.perf_counter() - t_start
    tool_calls_out, content, finish_reason = _postprocess(result, request_id, t_first_token, t_total)

    if tool_calls_out:
        for i, tc in enumerate(tool_calls_out):
            yield _chunk({"tool_calls": [{"index": i, "id": tc.id, "type": "function",
                                          "function": {"name": tc.function.name, "arguments": ""}}]})
            yield _chunk({"tool_calls": [{"index": i, "function": {"arguments": tc.function.arguments}}]})
    elif content:
        yield _chunk({"content": content})

    yield _chunk({}, finish_reason=finish_reason)
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
