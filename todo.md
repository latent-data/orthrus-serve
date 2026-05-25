# todo

Outstanding cleanup after the post-benchmark refactor pass.

## Smaller cleanups

- **Unreachable `model` defaults in schemas** — `openai_schemas.py` defaults `model = "orthrus-qwen3-8b"` on `ChatCompletionRequest`, `ChatCompletionResponse`, `ChatCompletionChunk`, but `main.py` computes `SERVED_MODEL_ID` from `BASE_MODEL` and uses that everywhere. Request defaults never reach a constructor. Drop them.
- **TTFT metric is misnamed** — `main.py:53-57`. The recorded value is full generate time, not first-token. Either rename the Prometheus metric or implement real TTFT (can't without a streamer).
- **`usage` field missing in SSE response** — non-streaming returns `Usage(prompt_tokens=…, completion_tokens=…)`; `_buffered_sse` never tells the client. Minor OpenAI-compat gap.

## Done

- **streaming.py deleted, all SSE unified through buffered path** — commit `a4e622d`.
- **Collapsed parallel post-generate paths in main.py** — commit `a40c63a`. `_postprocess` and `_dispatch_generate` extracted; `chat_completions` and `_buffered_sse` both call them. Rolled in: `ToolCall(**tc)`, `asyncio.Lock` rename, `_make_chunk` → consistent `_chunk` with `finish_reason` arg, removed dead `tool_calls_raw` / `secrets` / `JSONResponse` / `Request` / `Any` / `FunctionCall` references.
- **Centralised env-var config in `settings.py`** — this commit. Single `Settings.from_env()` frozen dataclass loaded once. `main.py`, `model.py`, `generation.py` all read `settings.x` instead of `os.environ.get(...)`. `orthrus_revision` moved from per-request log to the `model_ready` startup log.

## What's still deliberately left alone

- `model.py`, `generation.py`, `tool_parse.py`, `openai_schemas.py` (except dead defaults) — already small and focused.
- Logging format — JSON-in-JSON envelope is awkward but `tests/utils/log_parse.py` handles it and downstream consumers may depend on it.
