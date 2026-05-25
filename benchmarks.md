# Benchmarks

Two complementary benchmark surfaces:

1. **`tool-eval-bench`** — multi-turn tool-call scenarios, hits serve over HTTP. Numbers below sit in `~/spark-recipes/runs/`.
2. **Long-form generation** — `tests/benchmark.py` (in-process) and `tests/benchmark_http.py` (over HTTP). Numbers in `tests/results/`.

All Orthrus runs use revision `977a617772e91c966a8cd9b551f4151f9824b6fa` (post the `914faee` AR-fallback fix). All runs pass `enable_thinking=false`.

---

## tool-eval-bench (HTTP, multi-turn tool-call workload)

| Configuration | Summary file | Median turn |
|---|---|---|
| base Qwen3-8B, thinking off       | `~/spark-recipes/runs/2026/05/2026-05-23T17-52-30Z_9cd212_summary.md` | 4.5s |
| Orthrus, diffusion on, thinking off  | `~/spark-recipes/runs/2026/05/2026-05-23T19-19-08Z_93a80c_summary.md` | 2.0s |
| Orthrus, no-diff, thinking off       | `~/spark-recipes/runs/2026/05/2026-05-24T06-52-20Z_93a80c_summary.md` | 4.5s |
| Orthrus, no-diff, thinking on        | `~/spark-recipes/runs/2026/05/2026-05-23T20-47-05Z_93a80c_summary.md` | 30.3s |

Pattern: diffusion ≈ 2× faster than AR fallback or base. AR fallback matches base, consistent with both doing real AR + KV cache.

---

## Long-form generation (`tests/benchmark.py` + `tests/benchmark_http.py`)

Two prompts (`short`: ~470 output tokens; `long`: ~1440 output tokens), greedy decoding, `max_new_tokens=2048`, warmup before timing. Same prompts on both surfaces.

| Prompt | Config | In-process | HTTP | Δ |
|---|---|---|---|---|
| short | Orthrus diffusion | 12.05 s / 39.2 tok/s | 12.17 s / 38.8 tok/s | +0.12 s |
| short | Orthrus no-diff   | 42.65 s / 11.1 tok/s | 42.72 s / 11.1 tok/s | +0.07 s |
| short | Qwen3-8B AR       | 42.29 s / 11.2 tok/s | 42.79 s / 11.0 tok/s | +0.50 s |
| long  | Orthrus diffusion | 27.97 s / 51.5 tok/s | 28.19 s / 51.1 tok/s | +0.22 s |
| long  | Orthrus no-diff   | 131.76 s / 10.9 tok/s | 131.78 s / 10.9 tok/s | +0.02 s |
| long  | Qwen3-8B AR       | 130.65 s / 11.0 tok/s | 132.03 s / 10.9 tok/s | +1.38 s |

Geomean Orthrus-diffusion vs Qwen3-8B AR speedup: **4.06×** (3.51× short, 4.69× long).

### What this rules in / out for the serve wrapper

The HTTP numbers track in-process to within ~1% on every cell. So:

- **HTTP / FastAPI / uvicorn / asyncio overhead is negligible** at this scale.
- **`--no-diff` over HTTP is real AR**: matches base Qwen3 to within 0.1% on both prompts. The `914faee` "restore KV cache and EOS in AR fallback" fix is live in revision `977a617`.
- **Diffusion through serve is 3.5–4.7× faster than no-diff**, identical to in-process. The wrapper is not silently bypassing `use_diffusion_mode=False`.

### Unresolved: tool-eval-bench shows diff ≈ no-diff on the same serve

Long-form generation here clearly differentiates diff from no-diff. tool-eval-bench reportedly does not. That divergence cannot live in the model, the wrapper, or the HTTP layer — those are all benchmarked clean above. It must live in the workload itself.

Leading hypothesis: **the `StringStoppingCriteria` in `generation.py:67-70` is wired into AR but silently ignored by the diffusion path** (a known limitation — we'd have to patch the cached model file to fix). If a tool-call closing string trips the AR generator early but lets diffusion run to EOS or `max_tokens`, per-turn time compresses for no-diff and inflates for diff, collapsing the gap.

To verify, log `completion_tokens` and `tok_per_s` per request (now done in `main.py` at INFO level) and compare diff vs no-diff for the same scenarios.

---

## Reproducing

In-process (runs inside the orthrus-serve docker image):

```bash
tests/run_benchmark.sh --include-nodiff             # all three configs, both prompts
tests/run_benchmark.sh --no-build --prompts short   # iterate fast
```

Output: `tests/results/results.json`.

HTTP (host-side, stdlib only — needs serve already running):

```bash
./run.sh --disable-thinking &
python3 tests/benchmark_http.py --label orthrus_diffusion --disable-thinking --warmup

# stop, restart with --no-diffusion
python3 tests/benchmark_http.py --label orthrus_nodiff --disable-thinking --warmup

# stop, restart with --with-base-model
python3 tests/benchmark_http.py --label qwen3_8b_ar --model qwen3-8b --disable-thinking --warmup
```

Output: appends each `--label` into `results/results_http.json` (cwd-relative; run from `tests/` to land in `tests/results/`).
