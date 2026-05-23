# Prompt for Claude Code: orthrus-serve

Paste this as your initial prompt to Claude Code in a new empty repo. Provide the sibling `orthrus-bench-spark/` repo as a reference clone so Claude Code can read its Dockerfile and structural patterns.

---

Build `orthrus-serve`, an OpenAI-compatible HTTP serving endpoint for `chiennv/Orthrus-Qwen3-8B`, with tool calling support. Target hardware is a single NVIDIA DGX Spark (GB10, compute capability 12.1 / sm_121, 128 GB unified memory) running an NGC pytorch container. The endpoint will be evaluated by tool-eval-bench (a third-party OpenAI-compatible tool-calling benchmark, https://github.com/SeraphimSerapis/tool-eval-bench), which already supports vLLM, LiteLLM, and llama.cpp endpoints. Goal: orthrus-serve presents the same OpenAI wire format so it can be benchmarked alongside those backends with no special-case code.

## Reference repo

The sibling repo `orthrus-bench-spark/` is provided as a reference. Reuse its conventions:

- Dockerfile based on `nvcr.io/nvidia/pytorch:25.12-py3`
- `pip install --no-deps` for transformers and accelerate to avoid clobbering the container's custom torch
- Run script with `--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864`, HF cache mount at `~/.cache/huggingface`
- pyproject.toml with pinned `transformers==5.8.1` and `accelerate==1.13.0`
- SPEC.md pattern documenting design intent

Mirror that structure for orthrus-serve. Same Docker base, same pip discipline, same mount conventions. The container should expose the FastAPI server on port 8080 by default.

## Reference parity endpoint

A vanilla Qwen3-8B endpoint via llama.cpp is running on the same machine for parity testing. Its launch command is:

```
~/llama.cpp/build/bin/llama-server \
  -hf Qwen/Qwen3-8B-GGUF:Q8_0 \
  --jinja \
  --chat-template-file <(curl -s https://huggingface.co/Qwen/Qwen3-8B/raw/main/tokenizer_config.json | jq -r .chat_template) \
  -ngl 99 \
  --host 0.0.0.0 --port 8080
```

`orthrus-serve` should present the same external surface (port 8080 by default, OpenAI-compatible) so that tool-eval-bench can be pointed at either endpoint with no code changes, only a `--base-url` swap.

## Implementation

1. **FastAPI app** with these endpoints:
   - `POST /v1/chat/completions` (OpenAI ChatCompletions, request + response schemas matching OpenAI spec including `tools`, `tool_choice`, `stream`, `temperature`, `top_p`, `max_tokens`, `stop`, `chat_template_kwargs`)
   - `GET /v1/models` (returns the served model id so tool-eval-bench can auto-detect via its `/v1/models` discovery path)
   - `GET /health` (returns 200 once the model is loaded)
   - `GET /metrics` (optional but nice; Prometheus-style counters for total requests, tool-call rate, p50/p95 latency)

2. **Model loading** at startup:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       "chiennv/Orthrus-Qwen3-8B",
       revision=os.environ["ORTHRUS_REVISION"],
       trust_remote_code=True,
       dtype=torch.bfloat16,
       device_map="cuda",
       attn_implementation="flash_attention_2",
   ).eval()
   ```
   Pin the revision via env var (`ORTHRUS_REVISION`), with a sensible default baked in via the Dockerfile.

3. **Chat template inheritance**. The Orthrus tokenizer is missing `chat_template` in its tokenizer_config.json. After loading the Orthrus tokenizer, copy the template from `Qwen/Qwen3-8B` (also pinned via env var `QWEN_REVISION`):
   ```python
   tokenizer = AutoTokenizer.from_pretrained("chiennv/Orthrus-Qwen3-8B", revision=...)
   if tokenizer.chat_template is None:
       base = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", revision=...)
       tokenizer.chat_template = base.chat_template
       logger.warning("Orthrus tokenizer missing chat_template, inherited from Qwen/Qwen3-8B")
   ```
   This warning is intentional: if the Orthrus team ever adds the template upstream, the warning disappears and we know to drop the workaround.

4. **Request handling**:
   - Build prompt via `tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, **chat_template_kwargs, tokenize=False)`
   - Tokenize, move to device, generate with `use_diffusion_mode=True`
   - Map OpenAI request params: `temperature=0` means `do_sample=False`, otherwise `do_sample=True` with the given temperature/top_p; `max_tokens` defaults to 2048; `stop` strings are passed through stopping criteria

5. **Tool call parsing**. Qwen3 emits tool calls in this format in the assistant output:
   ```
   <tool_call>
   {"name": "get_weather", "arguments": {"location": "Oxford"}}
   </tool_call>
   ```
   Parse all `<tool_call>...</tool_call>` blocks from the generated text. For each, validate it's parseable JSON with a `name` and `arguments` field. Construct OpenAI-format tool_calls:
   ```python
   {
       "id": f"call_{secrets.token_hex(12)}",
       "type": "function",
       "function": {"name": parsed["name"], "arguments": json.dumps(parsed["arguments"])}
   }
   ```
   Note: OpenAI's `function.arguments` field is a JSON-encoded **string**, not a dict. Get this right; tool-eval-bench will fail otherwise.

   If any tool calls were emitted, set `finish_reason: "tool_calls"` and put them in `message.tool_calls`, with `message.content` set to whatever non-tool-call text was generated (often empty, but preserve any prefix the model emitted before the first tool_call block).
   
   If no tool calls, set `finish_reason: "stop"` and put the full text in `message.content`.

6. **Streaming** (SSE). Implement OpenAI-style streaming for the no-tool-call path: yield `data: {...}` chunks with delta content. For requests with `tools` provided, detect mid-stream if a `<tool_call>` token appears; if so, buffer until generation completes and return non-streaming. (Streaming partial tool calls is brittle and not worth v1 complexity.)

7. **Concurrency**. Single uvicorn worker is fine for v1 (one model on one GPU, requests serialised). Document this in the README. Consider adding a simple request queue with timeout for graceful degradation under load.

8. **Logging**. Structured JSON logs per request with: request_id, prompt token count, output token count, time-to-first-token, total time, whether tool_calls were emitted, finish_reason, model revision.

9. **/v1/models response shape**:
   ```json
   {
     "object": "list",
     "data": [{
       "id": "orthrus-qwen3-8b",
       "object": "model",
       "owned_by": "latent-data",
       "created": <unix-ts>,
       "max_model_len": 40960
     }]
   }
   ```
   tool-eval-bench reads `max_model_len` for context-pressure tests, so include it.

## Pitfalls (do not regress)

1. **Do not let pip upgrade torch.** NGC container ships a custom alpha build patched for sm_121. Use `pip install --no-deps`.
2. **`torch>=2.10.0`** does not satisfy NGC's `2.10.0a0+nv25.12` per PEP 440. Don't list torch as a dep.
3. **Do not install flash-attn from pip.** Container has FA2 preinstalled. FA4 does not support sm_121.
4. **Do not use `attn_implementation="sdpa"` for Orthrus generation.** SDPA breaks the bidirectional attention pattern Orthrus needs, producing degenerate repeating-token output. Use FA2.
5. **Do not assume the Orthrus chat template is present.** Inherit from `Qwen/Qwen3-8B` as in step 3 above.
6. **Tool call `arguments` must be a JSON-encoded string, not a dict.** Common OpenAI API mistake; tool-eval-bench will fail validation otherwise.
7. **Pin both model revisions.** Fetch current SHAs:
   ```bash
   curl -s https://huggingface.co/api/models/chiennv/Orthrus-Qwen3-8B | jq -r .sha
   curl -s https://huggingface.co/api/models/Qwen/Qwen3-8B | jq -r .sha
   ```
   Bake these as default env values in the Dockerfile.
8. **`trust_remote_code=True`** is required for Orthrus (loads `modeling_orthrus.py` from the HF cache). Do not try to remove it.

## Repo structure

```
orthrus-serve/
├── README.md
├── SPEC.md               # propagated; documents design
├── Dockerfile            # extends nvcr.io/nvidia/pytorch:25.12-py3
├── pyproject.toml
├── run.sh                # build + run container, mount HF cache
├── src/orthrus_serve/
│   ├── __init__.py
│   ├── main.py           # FastAPI app, uvicorn entry
│   ├── model.py          # model loading with chat template inheritance
│   ├── openai_schemas.py # pydantic models matching OpenAI spec
│   ├── generation.py     # generate() wrapper, sampling params mapping
│   ├── tool_parse.py     # <tool_call> block parsing to OpenAI format
│   └── streaming.py      # SSE response generator
├── tests/
│   ├── test_tool_parse.py    # unit tests for <tool_call> extraction
│   └── test_schemas.py       # pydantic schema round-trips
├── scripts/
│   └── smoke_test.sh     # curl-based health check + sample chat + sample tool call
└── LICENSE               # Apache-2.0
```

## Style

- No emdashes in any prose (user preference). Use commas, parentheses, or sentence breaks.
- Concise README. Include: what this is, how to run it, how it relates to llama.cpp parity, how tool-eval-bench connects.
- PEP 8 code, light comments. Reference relevant Qwen3 chat template behaviour with brief comments when it's non-obvious.
- Apache-2.0 license header optional, not required.

## Acceptance criteria

1. `./run.sh` builds the image and starts the container, exposing port 8080.
2. `curl http://localhost:8080/health` returns 200 after model load.
3. `curl http://localhost:8080/v1/models` returns one model entry with `max_model_len` populated.
4. `scripts/smoke_test.sh` runs three curl tests:
   - Plain chat: simple prompt, expects sensible English response
   - Single tool call: prompt with one tool defined, expects `finish_reason: "tool_calls"` and valid `tool_calls` array
   - No tool call: prompt with tools defined but a question that shouldn't use them, expects `finish_reason: "stop"` and no tool_calls
5. Pointing `tool-eval-bench --base-url http://localhost:8080 --backend vllm --scenarios TC-01 TC-02 TC-03` at the endpoint produces valid output (passes or fails are model behaviour, structural validity is the gate).
6. Unit tests in `tests/` pass under `pytest`.

You do not have GB10 hardware. Do not attempt to run the server. Final validation is the user's responsibility on their Spark.

## What to do if you are stuck

The non-obvious bits are: the chat template inheritance, the tool_call argument JSON-string serialization, and the streaming-vs-tool-call detection. For these, write the code with paper-side-by-side care, add unit tests, and surface to the user if uncertain. Everything else is standard FastAPI + transformers glue.

---
