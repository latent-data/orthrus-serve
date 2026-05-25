# todo

Outstanding cleanup after the post-benchmark refactor pass.

## Big ones

### 1. Centralise env-var config

Env vars are read at module import in `main.py`, inside `model.load_model_and_tokenizer`, inside `generation.generate`, inside `lifespan`, and *per request* in the INFO log.

- `ORTHRUS_BASE_MODEL` read in 3 places.
- `ORTHRUS_DIFFUSION` read in 2 places.
- `ORTHRUS_REVISION` read on every request, though it cannot change.

Action: one `Settings` dataclass loaded once at startup. Move `orthrus_revision` from per-request log to the `model_ready` startup log.

## Smaller cleanups

- **Unreachable `model` defaults in schemas** — `openai_schemas.py` defaults `model = "orthrus-qwen3-8b"` on `ChatCompletionRequest`, `ChatCompletionResponse`, `ChatCompletionChunk`, but `main.py` computes `SERVED_MODEL_ID` from `BASE_MODEL` and uses that everywhere. Request defaults never reach a constructor. Drop them.
- **TTFT metric is misnamed** — `main.py:53-57`. The recorded value is full generate time, not first-token. Either rename the Prometheus metric or implement real TTFT (can't without a streamer).
- **`usage` field missing in SSE response** — non-streaming returns `Usage(prompt_tokens=…, completion_tokens=…)`; `_buffered_sse` never tells the client. Minor OpenAI-compat gap.

## Done

- **#2 (was)** Delete `streaming.py`, unify all SSE through buffered path — done (commit `a4e622d`).
- **#1 (was)** Collapse the two parallel post-generate paths in `main.py` — done (this commit). `_postprocess` and `_dispatch_generate` extracted; `chat_completions` and `_buffered_sse` both call them. Rolled in: `ToolCall(**tc)`, `asyncio.Lock` rename, `_make_chunk` → consistent `_chunk` with `finish_reason` arg, removed dead `tool_calls_raw` / `secrets` / `JSONResponse` / `Request` / `Any` / `FunctionCall` references.

## What's still deliberately left alone

- `model.py`, `generation.py`, `tool_parse.py`, `openai_schemas.py` (except dead defaults) — already small and focused.
- Logging format — JSON-in-JSON envelope is awkward but `tests/utils/log_parse.py` handles it and downstream consumers may depend on it.
