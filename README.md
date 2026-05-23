# orthrus-serve

OpenAI-compatible HTTP endpoint for [chiennv/Orthrus-Qwen3-8B](https://huggingface.co/chiennv/Orthrus-Qwen3-8B), with tool calling support.

Target hardware: single NVIDIA DGX Spark (GB10, sm_121, 128 GB unified memory) running the NGC pytorch container.

## What this is

Orthrus is a diffusion-mode language model. This server wraps it in an OpenAI-compatible API so it can be evaluated alongside vLLM, LiteLLM, and llama.cpp endpoints with no benchmark-side code changes. The reference parity endpoint is a vanilla `Qwen3-8B` served via llama.cpp on the same machine.

## How to run

```bash
./run.sh
```

This builds the Docker image and starts the container. The server exposes port 8080 by default.

```bash
PORT=9090 ./run.sh              # use a different port
./run.sh --no-build             # skip docker build, mount source directly (faster iteration)
./run.sh --no-diffusion         # disable diffusion mode (autoregressive fallback)
./run.sh --debug                # enable verbose debug logging
./run.sh --with-base-model      # load Qwen3-8B instead of Orthrus
./run.sh --enable-thinking      # force thinking tokens on (model default is off)
./run.sh --disable-thinking     # force thinking tokens off
```

Environment variables you can override (defaults are baked into the Dockerfile):

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8080` | Listening port |
| `ORTHRUS_REVISION` | pinned SHA | HF revision for Orthrus model |
| `ORTHRUS_DIFFUSION` | `1` | Set to `0` to disable diffusion mode |
| `ORTHRUS_DEBUG` | `0` | Set to `1` for verbose JSON debug logs |
| `ORTHRUS_BASE_MODEL` | `0` | Set to `1` to load Qwen3-8B instead of Orthrus |
| `ORTHRUS_ENABLE_THINKING` | unset | Set to `true` or `false` to override the model default |

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Returns 200 once the model is loaded |
| GET | `/v1/models` | Lists the served model (OpenAI format) |
| POST | `/v1/chat/completions` | Chat completions with optional tool calling |
| GET | `/metrics` | Prometheus metrics |

## Tool calling

Orthrus emits tool calls in Qwen3 format (`<tool_call>...</tool_call>` blocks). This server parses them and converts to the OpenAI `tool_calls` array format, with `function.arguments` as a JSON-encoded string per the OpenAI spec.

If `tools` are present in the request, the full generation is buffered before tool calls are parsed and returned. The SSE stream is still opened immediately and keepalive comments (`": ping"`) are emitted every 5 s so the client's read-timeout doesn't fire during long generations. Plain chat requests (no tools) stream tokens as they are produced.

## Benchmarking with tool-eval-bench

```bash
python -m tool_eval_bench \
    --base-url http://localhost:8080 \
    --backend vllm \
    --scenarios TC-01 TC-02 TC-03
```

The `--backend vllm` flag tells tool-eval-bench to use the OpenAI-compatible wire format, which this server matches exactly.

## Concurrency

Single uvicorn worker, requests serialised behind an asyncio semaphore. One model, one GPU. Requests time out after 300 seconds.

## Smoke test

```bash
bash scripts/smoke_test.sh
# BASE_URL=http://localhost:8080 bash scripts/smoke_test.sh
```

## Unit tests

```bash
pip install -e ".[dev]"
pytest
```

## Relation to llama.cpp parity endpoint

The llama.cpp endpoint (`Qwen3-8B-GGUF:Q8_0`) runs on the same machine and exposes the same port 8080. Switch between them with `--base-url`. Both use the Qwen3 chat template and the same tool-call wire format.
