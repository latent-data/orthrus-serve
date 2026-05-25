"""
Endpoint shape tests with a stubbed generate() and tokenizer.

Covers the FastAPI wrapper paths (request handling, post-processing, OpenAI
response shape, SSE chunk format). The model itself is not exercised —
tests/benchmark.py covers the model side.

Goal: guard the refactor in todo.md (collapse the two post-generate paths,
delete streaming.py, centralise env config) by pinning the response contract.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# torch / transformers only exist inside the orthrus-serve container — skip
# this file when running pytest on the host. tests/run_tests.sh runs pytest
# inside the image where these are present.
pytest.importorskip("transformers")

from fastapi.testclient import TestClient  # noqa: E402

from orthrus_serve import main  # noqa: E402
from orthrus_serve.generation import GenerationResult  # noqa: E402


def _stub_tokenizer():
    """Mock tokenizer: apply_chat_template returns a fixed prompt string."""
    tok = MagicMock()
    tok.apply_chat_template.return_value = "<stub-prompt>"
    return tok


def _result(text: str, prompt_tokens: int = 100, completion_tokens: int | None = None):
    if completion_tokens is None:
        completion_tokens = max(1, len(text.split()))
    return GenerationResult(text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


@pytest.fixture
def client():
    """TestClient with lifespan triggered; model/tokenizer load is stubbed."""
    with patch(
        "orthrus_serve.main.load_model_and_tokenizer",
        return_value=(MagicMock(), _stub_tokenizer()),
    ):
        with TestClient(main.app) as c:
            yield c


def _parse_sse(body: str) -> list[dict]:
    """Parse SSE body into a list of parsed JSON chunks (skipping comments and [DONE])."""
    out = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            continue
        out.append(json.loads(payload))
    return out


# ----------------- meta endpoints -----------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["id"] == main.SERVED_MODEL_ID


# ----------------- non-streaming -----------------

def test_non_streaming_no_tools(client):
    with patch("orthrus_serve.main.generate", return_value=_result("hello world", completion_tokens=2)):
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == main.SERVED_MODEL_ID
    assert len(body["choices"]) == 1

    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "hello world"
    assert choice["message"]["tool_calls"] is None
    assert choice["finish_reason"] == "stop"

    usage = body["usage"]
    assert usage["prompt_tokens"] == 100
    assert usage["completion_tokens"] == 2
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_non_streaming_with_tool_call(client):
    raw = '<tool_call>{"name": "get_weather", "arguments": {"city": "London"}}</tool_call>'
    with patch("orthrus_serve.main.generate", return_value=_result(raw, completion_tokens=15)):
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
            "stream": False,
        })
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None

    tcs = choice["message"]["tool_calls"]
    assert len(tcs) == 1
    assert tcs[0]["type"] == "function"
    assert tcs[0]["function"]["name"] == "get_weather"
    # arguments must be a JSON-encoded string per OpenAI spec, not a dict
    assert isinstance(tcs[0]["function"]["arguments"], str)
    assert json.loads(tcs[0]["function"]["arguments"]) == {"city": "London"}


def test_non_streaming_strips_think_tags(client):
    raw = "<think>internal monologue</think>actual answer"
    with patch("orthrus_serve.main.generate", return_value=_result(raw, completion_tokens=5)):
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
    assert r.json()["choices"][0]["message"]["content"] == "actual answer"


def test_non_streaming_stop_normalization(client):
    """stop=str and stop=list[str] should both work without 500."""
    with patch("orthrus_serve.main.generate", return_value=_result("ok", completion_tokens=1)) as mock_gen:
        r1 = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "stop": "###",
        })
        r2 = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "stop": ["###", "</end>"],
        })
    assert r1.status_code == 200 and r2.status_code == 200
    # stop is forwarded to generate() as a list either way
    for call in mock_gen.call_args_list:
        forwarded_stop = call.args[6]  # generate(model, tokenizer, prompt, temperature, top_p, max_tokens, stop)
        assert isinstance(forwarded_stop, list)


# ----------------- streaming with tools (buffered SSE path) -----------------

def test_streaming_with_tool_call(client):
    raw = '<tool_call>{"name": "ping", "arguments": {}}</tool_call>'
    with patch("orthrus_serve.main.generate", return_value=_result(raw, completion_tokens=10)):
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [{"type": "function", "function": {"name": "ping", "parameters": {}}}],
            "stream": True,
        })
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    chunks = _parse_sse(r.text)
    assert chunks, "no chunks parsed from SSE body"

    # Chunk envelope conforms to chat.completion.chunk
    for c in chunks:
        assert c["object"] == "chat.completion.chunk"
        assert c["model"] == main.SERVED_MODEL_ID

    # First chunk announces the role
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"

    # Somewhere in the middle there's a tool_calls delta with function name
    tc_chunks = [c for c in chunks if "tool_calls" in c["choices"][0]["delta"]]
    assert tc_chunks, "no tool_calls chunk emitted"
    names = [tc["function"].get("name") for c in tc_chunks for tc in c["choices"][0]["delta"]["tool_calls"]]
    assert "ping" in names

    # Final chunk carries finish_reason=tool_calls
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"

    # Stream terminates with [DONE]
    assert r.text.rstrip().endswith("data: [DONE]")


def test_streaming_buffered_no_tool_call_content_chunk(client):
    """When stream=True + tools requested but model returns plain text, content is emitted as a single delta."""
    with patch("orthrus_serve.main.generate", return_value=_result("hi there", completion_tokens=2)):
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            # tools forces the buffered SSE path even though model returns no <tool_call>
            "tools": [{"type": "function", "function": {"name": "anything", "parameters": {}}}],
            "stream": True,
        })
    assert r.status_code == 200
    chunks = _parse_sse(r.text)
    content_chunks = [c for c in chunks if "content" in c["choices"][0]["delta"] and c["choices"][0]["delta"]["content"]]
    assert content_chunks, "expected a content delta"
    assert content_chunks[0]["choices"][0]["delta"]["content"] == "hi there"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


# ----------------- streaming without tools (now also buffered SSE) -----------------

def test_streaming_no_tools(client):
    """Post-deletion of streaming.py: stream=True without tools goes through the
    same buffered SSE path as stream=True with tools."""
    with patch("orthrus_serve.main.generate", return_value=_result("hi there", completion_tokens=2)):
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        })
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    chunks = _parse_sse(r.text)
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    content_chunks = [c for c in chunks if c["choices"][0]["delta"].get("content")]
    assert content_chunks
    assert content_chunks[0]["choices"][0]["delta"]["content"] == "hi there"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert r.text.rstrip().endswith("data: [DONE]")


# ----------------- error paths -----------------

def test_returns_503_when_not_ready():
    """If the model never loaded, /v1/chat/completions returns 503."""
    # Build a TestClient without triggering lifespan by NOT using context-manager form.
    # _ready stays False since lifespan never ran.
    main._ready = False
    main._request_semaphore = None
    c = TestClient(main.app)
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503
