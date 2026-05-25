# todo

Empty — todo.md cleanup pass is complete.

## Done

- **Delete `streaming.py`, unify all SSE through buffered path** — commit `a4e622d`.
- **Collapse parallel post-generate paths in main.py** — commit `a40c63a`. `_postprocess` and `_dispatch_generate` extracted. Rolled in: `ToolCall(**tc)`, `asyncio.Lock` rename, `_make_chunk` → consistent `_chunk` with `finish_reason` arg, removed dead `tool_calls_raw` / `secrets` / `JSONResponse` / `Request` / `Any` / `FunctionCall` references.
- **Centralised env-var config in `settings.py`** — commit `5c1ada1`. Single `Settings.from_env()` frozen dataclass loaded once; `main.py`, `model.py`, `generation.py` all read `settings.x`. `orthrus_revision` moved from per-request log to `model_ready` startup log.
- **Schema model defaults dropped** — `ChatCompletionRequest.model` / `Response.model` / `Chunk.model` are now required, matching OpenAI spec.
- **TTFT metric renamed to `orthrus_generate_duration_seconds`** — old name was misleading (it recorded full GPU time, not first-token). Per-request log field renamed from `ttft_s` to `generate_s`; `log_parse.py` accepts both for back-compat with older logs.
- **SSE response now carries `usage`** — final chunk in the buffered SSE stream includes `usage = {prompt_tokens, completion_tokens, total_tokens}`, closing the OpenAI-compat gap.

## Net package size

- Before: 820 lines across 7 files.
- After: ~660 lines across 7 files (streaming.py deleted, settings.py added).
- All 28 tests pass.

## What's still deliberately left alone

- `model.py`, `generation.py`, `tool_parse.py`, `openai_schemas.py` — already small and focused.
- Logging format — JSON-in-JSON envelope is awkward but `benchmarks/log_parse.py` handles it and downstream consumers may depend on it.
