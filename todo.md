# todo

Post-benchmark cleanup. Action after the diff vs no-diff comparison is finalised in `benchmarks.md` — the per-request logging instrumentation needs to stay stable until then.

## Big ones (ranked by payoff)

### 1. Collapse the two parallel post-generate paths in `main.py`

`chat_completions` (lines 181-270) and `_buffered_sse` (lines 273-372) duplicate everything after the GPU call:

- Same semaphore + executor dispatch
- Same `strip_think_tags` → `parse_tool_calls` → `ToolCall` rebuild (lines 211-229 vs 321-339)
- Same INFO request log dict (lines 253-268 vs 341-356) — exactly the place where `tok_per_s` had to be added in two spots
- Same Prometheus histogram observations

Action: extract a single `run_and_postprocess(prompt, …)` coroutine returning `(tool_calls_out, content, finish_reason, result, t_first_token, t_total)`. Have two thin emitters consume it — one returns a `ChatCompletionResponse`, the other yields SSE chunks. Expected diff: ~-80 lines, single source of truth for logging.

### 2. Delete `streaming.py`

`stream_completion` is a third generate path, used only when `request.stream and not has_tools`. Problems:

- Hardcodes `use_diffusion_mode: True` (line 45) — `--no-diffusion` doesn't affect it.
- Uses `TextIteratorStreamer` in a thread — the pattern that commit `3772692` removed from `_buffered_sse` because it hung.
- Uses `tokenizer(prompt, …)` not `tokenizer.encode(...)` — inconsistent with `generation.py` after `4e1f93f`.
- Reinvents stop-string handling rather than reusing `StringStoppingCriteria`.

You don't get real token-by-token streaming from diffusion mode anyway (blocks of 32). Action: delete `streaming.py`, route all `request.stream=True` through the unified buffered-SSE coroutine from item 1.

### 3. Centralise env-var config

Env vars are read at module import in `main.py`, inside `model.load_model_and_tokenizer`, inside `generation.generate`, inside `lifespan`, and *per request* in the INFO log.

- `ORTHRUS_BASE_MODEL` read in 3 places.
- `ORTHRUS_DIFFUSION` read in 2 places.
- `ORTHRUS_REVISION` read on every request, though it cannot change.

Action: one `Settings` dataclass loaded once at startup. Move `orthrus_revision` from per-request log to the `model_ready` startup log.

## Smaller cleanups (each ~5-15 lines)

- **Unreachable `model` defaults in schemas** — `openai_schemas.py:41,86,106` all default `model = "orthrus-qwen3-8b"`, but `main.py:38` computes `SERVED_MODEL_ID` from `BASE_MODEL` and uses that everywhere. Request defaults never reach a constructor. Drop them.
- **Pydantic re-pack ceremony** — `main.py:220-229` builds `ToolCall(id=…, function=FunctionCall(name=…, arguments=…))` from a dict that's already in OpenAI shape. Either have `parse_tool_calls` return `ToolCall` objects, or do `ToolCall(**tc)`.
- **`tool_calls_raw` is vestigial** — built, iterated once into `tool_calls_out`, then only used for a truthiness check at log time. Check `if tool_calls_out:` instead.
- **`_make_chunk` asymmetry** — defined as a helper in `_buffered_sse` (line 285), used three times, then bypassed for the final "finish" and "[DONE]" chunks (line 371). Use it consistently or drop it.
- **`asyncio.Semaphore(1)` is just `asyncio.Lock`** — `main.py:69`. Lock conveys intent.
- **TTFT metric is misnamed** — `main.py:54-58, 200, 313`. The recorded value is full generate time, not first-token. Either rename the Prometheus metric or implement real TTFT (can't without a streamer).
- **Inconsistent `t_generate_start` placement** — `main.py:184` is *inside* the semaphore, `main.py:291` is *outside*. One includes semaphore wait in "generate" time, the other doesn't. Pick one.
- **Dead imports** — `secrets` in `main.py` and `streaming.py`, `JSONResponse` in `main.py`.
- **`usage` field missing in SSE response** — non-streaming returns `Usage(prompt_tokens=…, completion_tokens=…)`; `_buffered_sse` never tells the client. Minor OpenAI-compat gap.

## What I deliberately left alone

- `model.py`, `generation.py`, `tool_parse.py`, `openai_schemas.py` (other than dead defaults) — these are already small and focused.
- Logging format — the JSON-in-JSON envelope is awkward but `tests/utils/log_parse.py` handles it and downstream consumers may depend on it.

## Expected outcome

After items 1-3 the package should land around ~500 lines (from ~820), with no second copy of anything load-bearing.
