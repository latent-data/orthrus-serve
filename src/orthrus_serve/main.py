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
from .settings import settings
from .tool_parse import parse_tool_calls, strip_think_tags

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
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
GENERATE_DURATION_HIST = Histogram(
    "orthrus_generate_duration_seconds",
    "GPU generation wall-time (excludes pre/post processing). Buffered API: not a TTFT.",
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60],
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
    logger.info(
        json.dumps({
            "event": "model_ready",
            "diffusion": settings.diffusion_enabled,
            "thinking": settings.enable_thinking,
            "orthrus_revision": settings.orthrus_revision,
        })
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
                id=settings.served_model_id,
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


def _postprocess(result, request_id, t_generate, t_total):
    """Strip think tags, parse tool calls, log INFO, observe Prometheus.

    Returns (tool_calls_out, content, finish_reason).
    """
    if settings.debug:
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
    GENERATE_DURATION_HIST.observe(t_generate)

    tok_per_s = round(result.completion_tokens / t_generate, 2) if t_generate > 0 else 0.0
    logger.info(
        json.dumps(
            {
                "request_id": request_id,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "generate_s": round(t_generate, 3),
                "total_s": round(t_total, 3),
                "tok_per_s": tok_per_s,
                "tool_calls": bool(tool_calls_out),
                "finish_reason": finish_reason,
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
    _thinking_default = {} if settings.enable_thinking is None else {"enable_thinking": settings.enable_thinking}
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

    if settings.debug:
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
                t_generate = time.perf_counter() - t_generate_start
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Request timed out")

    t_total = time.perf_counter() - t_start
    tool_calls_out, content, finish_reason = _postprocess(result, request_id, t_generate, t_total)

    response = ChatCompletionResponse(
        id=request_id,
        model=settings.served_model_id,
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

    if settings.debug:
        logger.debug(json.dumps({"request_id": request_id, "event": "response", "body": response.model_dump()}))

    return response


async def _buffered_sse(request_id, t_start, prompt, temperature, top_p, max_tokens, stop):
    """SSE generator: role chunk → keepalive pings while GPU works → tool_calls / content chunks → finish + [DONE]."""
    created = int(time.time())

    def _chunk(delta: dict, finish_reason: str | None = None, usage: dict | None = None) -> str:
        body = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": settings.served_model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            body["usage"] = usage
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

    t_generate = time.perf_counter() - t_generate_start
    t_total = time.perf_counter() - t_start
    tool_calls_out, content, finish_reason = _postprocess(result, request_id, t_generate, t_total)

    if tool_calls_out:
        for i, tc in enumerate(tool_calls_out):
            yield _chunk({"tool_calls": [{"index": i, "id": tc.id, "type": "function",
                                          "function": {"name": tc.function.name, "arguments": ""}}]})
            yield _chunk({"tool_calls": [{"index": i, "function": {"arguments": tc.function.arguments}}]})
    elif content:
        yield _chunk({"content": content})

    usage = {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
    }
    yield _chunk({}, finish_reason=finish_reason, usage=usage)
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
