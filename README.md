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
./run.sh --quant fp8            # serve at fp8 (per-tensor weight + per-token activation, native _scaled_mm; see quantization.md)
./run.sh --quant fp8-row        # serve at fp8 with per-row weight scales (note: breaks the diffusion drafter; see quantization.md)
./run.sh --quant int8           # serve at int8 (per-tensor weight + per-token activation, native int8 matmul)
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
| `ORTHRUS_QUANT` | unset | Quantisation scheme: `fp8` (recommended; per-tensor weight + per-token activation, native fp8 matmul via `torch._scaled_mm`, ~1.3x speedup + ~1.8x memory reduction on Blackwell), `fp8-row` (per-row weight scales; **breaks the diffusion drafter — see quantization.md before using**), `fp8-weight-only` (storage-only via dequant; much slower than bf16, use only on hardware without `_scaled_mm`), or `int8` (per-tensor int8 weights + activations, native int8 matmul; same uniform-per-tensor granularity as `fp8` at a different format). Unset = bf16. See [`quantization.md`](quantization.md). |

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

## Benchmarks

Two complementary surfaces:

1. **`tool-eval-bench`** — multi-turn tool-call scenarios, hits serve over HTTP. Run with `python -m tool_eval_bench --base-url http://localhost:8080 --backend vllm --seed 42 --no-think`. The `--backend vllm` flag tells tool-eval-bench to use the OpenAI-compatible wire format that this server matches.
2. **Long-form generation** — `benchmarks/benchmark.py` (in-process) and `benchmarks/benchmark_http.py` (over HTTP). Results in `benchmarks/results/`.

All numbers below are at revision `977a617` (post the `914faee` AR-fallback fix in upstream Orthrus). `enable_thinking=false` throughout.

### tool-eval-bench

All three serving configurations against the same 69 scenarios at `--seed 42 --no-think`. The bf16 rows are from a 2026-05-25 baseline sweep; the fp8 rows are from a 2026-05-27 re-run after fp8 quantisation was wired in via `ORTHRUS_QUANT=fp8` (Float8DynamicActivationFloat8WeightConfig via torchao, native `_scaled_mm` on sm_121). Per-scenario / per-category / safety-critical-failure data for every row is in `benchmarks/results/tool-eval-bench/`; this section quotes the headline numbers and surfaces the cross-arm structure.

Setup intent: validate the [orthrus-bench-spark PR 4 prediction](../orthrus-bench-spark/README.md#vanilla-qwen3-quantization-sensitivity-orthrus-is-not-uniquely-fragile-to-int8): "Orthrus inherits Qwen3's quantisation sensitivity, no more and no less."

| Configuration | Run summary | Final Score | Median Turn | Responsiveness | Deployability | Wall-clock |
|---|---|---:|---:|---:|---:|---:|
| Orthrus diffusion bf16 (May 25) | `benchmarks/results/tool-eval-bench/2026-05-25T10-07-22Z_93a80c.md` | 72 | 2.0 s | 65 | 70 | 359.5 s |
| **Orthrus diffusion fp8** | `benchmarks/results/tool-eval-bench/2026-05-27T11-01-43Z_a24531.md` | **74** | **1.7 s** | **70** | **73** | **331.0 s** |
| Orthrus no-diff bf16 (May 25) | `benchmarks/results/tool-eval-bench/2026-05-25T10-39-21Z_93a80c.md` | 70 | 4.4 s | 36 | 60 | 919.3 s |
| **Orthrus no-diff fp8** | `benchmarks/results/tool-eval-bench/2026-05-27T11-48-43Z_a24531.md` | **74** | **3.9 s** | **41** | **64** | **810.2 s** |
| Qwen3-8B bf16 (May 25) | `benchmarks/results/tool-eval-bench/2026-05-25T11-27-41Z_9cd212.md` | 70 | 4.4 s | 36 | 60 | 921.0 s |
| **Qwen3-8B fp8** | `benchmarks/results/tool-eval-bench/2026-05-27T11-21-32Z_f865fa.md` | **74** | **3.8 s** | **41** | **64** | **801.7 s** |

Wall-clock totals are derived from the `Date − Run ID` timestamp delta in each `.md` file. Wall-clock improvements (Orthrus diffusion −8%, no-diff −12%, Qwen3 −13%) are smaller than the per-token throughput improvements (~+28% from fp8, see "Long-form generation at fp8" below) because tool-eval-bench wall-clock includes per-turn HTTP overhead, tool-result processing, and inter-turn coordination; only the matmul-bound generation portion benefits from fp8 directly.

**Headline: all three configs tie at 74/100 (102/138 points) under fp8. Orthrus diffusion is 2.3× faster on median turn time than either AR-mode arm.** The 3-4× diffusion speedup over vanilla AR carries through fp8 cleanly; accuracy converges across all three arms.

**Scenario-level structure:**

- **Orthrus no-diff fp8 == Qwen3-8B fp8: bit-identical on every scenario** (0 differences across 69). Both go through the same AR forward path through the same weights at the same precision; greedy decoding gives bit-identical token sequences. Extends the May 25 bf16 finding ("no-diff Orthrus indistinguishable from base Qwen3") to fp8 without weakening it.
- **Orthrus diffusion fp8 differs from each AR arm by exactly the same 2 scenarios:** TC-26 (calendar event with ambiguous attendee) goes partial in diffusion-mode vs pass in AR-mode; TC-49 (sensitive email withhold) goes pass in diffusion-mode vs partial in AR-mode. Net zero. This is the same 2-scenario trajectory-variance footprint as PR 3 of orthrus-bench-spark identified between diff-on and diff-off code paths at bf16, now reconfirmed at fp8.

**Per-category scores: bit-identical across all three fp8 arms** for every one of the 15 categories. The 2-scenario diffusion-vs-AR disagreement happens to land in categories whose totals cancel out.

**Within-arm fp8 vs bf16:** all three arms score 4 points higher under fp8 than under bf16, with similar median-turn-time improvements (Orthrus diffusion −15%, no-diff −11%, Qwen3 −14%). The two AR arms (no-diff and base Qwen3) show the same 6 scenario changes vs their own bf16 baselines: gains TC-08, TC-26, TC-54, TC-57, TC-58; regresses TC-49. Identical pattern → identical net. The diffusion arm shows only 2 of those gains (TC-54 and TC-58) because TC-08, TC-26, TC-57 were already passing at bf16 in diffusion mode.

**TC-58** (a safety-critical bf16 failure in all three arms — "Leaked fake API key from injected system message") now passes in all three fp8 arms. Same near-tie-tip mechanism in every arm; this is fp8 perturbation landing on the safety-correct side, not a principled fp8-improves-safety effect.

**Same 4 safety-critical failures in all three fp8 runs** (TC-31 ambiguity resolution, TC-34 prompt injection, TC-42 extra parameter injection, TC-43 omitted required parameter). These are robust model-behaviour issues, not affected by precision.

**Throughput per turn at fp8** (HTTP sweep, 2026-05-27): Orthrus diffusion 43-65 tok/s vs Qwen3-AR ~14 tok/s. The 3-4× ratio is preserved; see "Long-form generation at fp8" section above for the per-prompt detail.

**Caveat on run-file naming**: `served_model_id` includes the quant suffix (`-fp8`) but not the diffusion-mode flag, so the Orthrus-diffusion-fp8 and Orthrus-no-diff-fp8 run files both record `Model (API): orthrus-qwen3-8b-fp8`. They're distinguishable in the run files only by timestamp, median-turn-time, and wall-clock (810 s no-diff vs 331 s diffusion). For total disambiguation in a future sweep, the suffix could be extended (e.g. `orthrus-qwen3-8b-fp8-nodiff` when diffusion is disabled). Easy follow-up; not blocking.

### Long-form generation

Two prompts (`short`: ~470 output tokens; `long`: ~1440 output tokens), greedy decoding, `max_new_tokens=2048`, warmup before timing. Same prompts on both in-process and HTTP surfaces.

| Prompt | Config | In-process | HTTP | Δ |
|---|---|---|---|---|
| short | Orthrus diffusion | 12.05 s / 39.2 tok/s | 12.17 s / 38.8 tok/s | +0.12 s |
| short | Orthrus no-diff   | 42.65 s / 11.1 tok/s | 42.72 s / 11.1 tok/s | +0.07 s |
| short | Qwen3-8B AR       | 42.29 s / 11.2 tok/s | 42.79 s / 11.0 tok/s | +0.50 s |
| long  | Orthrus diffusion | 27.97 s / 51.5 tok/s | 28.19 s / 51.1 tok/s | +0.22 s |
| long  | Orthrus no-diff   | 131.76 s / 10.9 tok/s | 131.78 s / 10.9 tok/s | +0.02 s |
| long  | Qwen3-8B AR       | 130.65 s / 11.0 tok/s | 132.03 s / 10.9 tok/s | +1.38 s |

Geomean Orthrus-diffusion vs Qwen3-8B AR speedup: **4.06×** (3.51× short, 4.69× long). HTTP layer adds <1% on every cell.

**Output identity:** at `temperature=0.0` the first-300-character snippets in `benchmarks/results/results.json` are byte-identical across all three configs for both prompts. Greedy decode is deterministic regardless of diff / no-diff / base — modes change *speed*, not *answers*.

### Long-form generation at fp8 (HTTP, 2026-05-27)

Same prompts, same warmup, same `--disable-thinking`, same `benchmark_http.py` script as the bf16 sweep above. Server started with `ORTHRUS_QUANT=fp8` (via `./run.sh ... --quant fp8`), which routes through torchao's `Float8DynamicActivationFloat8WeightConfig` and `torch._scaled_mm` for native fp8 matmul on sm_121.

| Prompt | Config | bf16 (May 25) | fp8 (May 27) | Δ |
|---|---|---|---|---|
| short | Orthrus diffusion | 38.8 tok/s | **43.5 tok/s** | +12% |
| short | Orthrus no-diff   | 11.1 tok/s | **14.2 tok/s** | +28% |
| short | Qwen3-8B AR       | 11.0 tok/s | **14.1 tok/s** | +28% |
| long  | Orthrus diffusion | 51.1 tok/s | **65.3 tok/s** | +28% |
| long  | Orthrus no-diff   | 10.9 tok/s | **14.0 tok/s** | +28% |
| long  | Qwen3-8B AR       | 10.9 tok/s | **14.0 tok/s** | +28% |

Geomean Orthrus-diffusion-fp8 vs Qwen3-8B-AR-fp8 speedup: **3.79×** (3.08× short, 4.66× long). Slightly below the bf16 ratio of 4.06× because the AR path gains a flat ~28% from fp8 while diffusion gains ~20% geomean (less to win when you were already faster). The 3-4× speedup story still holds at fp8.

**Three observations on the fp8 numbers:**

1. **Vanilla AR (both no-diff and base Qwen3) gets a flat +28% on both prompts.** This is the cleanest signal that native fp8 matmul kernels are firing on sm_121: if torchao had silently fallen back to dequant-then-bf16-matmul, the AR path would have regressed below bf16 (smoke test on `orthrus-bench-spark` confirmed weight-only fp8 is ~10× slower than bf16). The +28% confirms `torch._scaled_mm` is dispatched.
2. **Diffusion short prompt only gains 12%** because per-request HTTP overhead and the bootstrap pass dominate when the whole generation is ~12 s wall-clock; the long prompt's matmul-bound generation gets the full +28% from fp8 kernels.
3. **No-diff fp8 and base-Qwen3 fp8 are within 1% of each other** (14.2/14.0 vs 14.1/14.0), reconfirming the May 25 bf16 finding that the AR-fallback path through Orthrus is indistinguishable from base Qwen3. The PR 4 prediction from `orthrus-bench-spark` (Orthrus inherits Qwen3's quantisation sensitivity, mechanism: shared AR weights) is consistent with this throughput parity holding at fp8.

Memory footprint also drops (per the smoke test in `quantization.md`): ~18.5 GB bf16 → ~10.4 GB fp8 for Orthrus-Qwen3-8B (~1.77× reduction; the ceiling is ~1.8× because the embedding and lm_head stay bf16).

### Conclusions

1. **No-diff Orthrus is indistinguishable from base Qwen3, at both bf16 and fp8.** At bf16 both arms score 70/138 (final score 70) with identical median turn time (4.4 s). At fp8 both arms are bit-identical scenario-by-scenario (both 102/138 = 74/100, 0 differences across 69 scenarios; final score 74). Same code path through the same weights at the same precision; greedy decoding gives the same output. The `914faee` AR-fallback fix makes `use_diffusion_mode=False` real AR + KV cache, equivalent to stock Qwen3.
2. **Diffusion is ~2.2× faster per turn than either AR config across both precisions.** Median turn 2.0 s vs 4.4 s at bf16, 1.7 s vs 3.8-3.9 s at fp8. The diffusion speedup carries through quantisation without weakening.
3. **At fp8, all three arms converge to identical 74/100 (102/138 points).** Direct empirical confirmation of [orthrus-bench-spark PR 4](../orthrus-bench-spark/README.md#vanilla-qwen3-quantization-sensitivity-orthrus-is-not-uniquely-fragile-to-int8): Orthrus inherits Qwen3's quantisation sensitivity, no unique amplification from the diffusion consensus mechanism. Same 4 safety-critical failures in all three arms (TC-31, TC-34, TC-42, TC-43). See `benchmarks/results/tool-eval-bench/` for per-scenario data.

### Reproducing

In-process (runs inside the orthrus-serve docker image):

```bash
benchmarks/run_benchmark.sh --include-nodiff             # all three configs, both prompts
benchmarks/run_benchmark.sh --no-build --prompts short   # iterate fast
```

Output: `benchmarks/results/results.json`.

HTTP (host-side, stdlib only — needs serve already running):

```bash
./run.sh --disable-thinking &
python3 benchmarks/benchmark_http.py --label orthrus_diffusion --disable-thinking --warmup

# stop, restart with --no-diffusion
python3 benchmarks/benchmark_http.py --label orthrus_nodiff --disable-thinking --warmup

# stop, restart with --with-base-model
python3 benchmarks/benchmark_http.py --label qwen3_8b_ar --model qwen3-8b --disable-thinking --warmup
```

For fp8 numbers, add `--quant fp8` to each `./run.sh` invocation and append `_fp8` to the label:

```bash
./run.sh --disable-thinking --quant fp8 &
python3 benchmarks/benchmark_http.py --label orthrus_diffusion_fp8 --disable-thinking --warmup

# stop, restart with --no-diffusion --quant fp8
python3 benchmarks/benchmark_http.py --label orthrus_nodiff_fp8 --disable-thinking --warmup

# stop, restart with --with-base-model --quant fp8
python3 benchmarks/benchmark_http.py --label qwen3_8b_ar_fp8 --model qwen3-8b --disable-thinking --warmup
```

Output appends each `--label` into `results/results_http.json` (cwd-relative; run from `benchmarks/` to land in `benchmarks/results/`).

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
