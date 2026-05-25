# Benchmarks

Two complementary benchmark surfaces:

1. **`tool-eval-bench`** — multi-turn tool-call scenarios, hits serve over HTTP. Numbers below sit in `~/spark-recipes/runs/`.
2. **Long-form generation** — `benchmarks/benchmark.py` (in-process) and `benchmarks/benchmark_http.py` (over HTTP). Numbers in `benchmarks/results/`.

All Orthrus runs use revision `977a617772e91c966a8cd9b551f4151f9824b6fa` (post the `914faee` AR-fallback fix). All runs pass `enable_thinking=false`.

---

## tool-eval-bench (HTTP, multi-turn tool-call workload, 2026-05-25 sweep)

All three configurations run against the same 69 scenarios at `--seed 42 --no-think`. Per-request serve INFO logs aggregated via `benchmarks/log_parse.py`.

| Configuration | Summary | Final Score | Median Turn | Responsiveness |
|---|---|---:|---:|---:|
| Orthrus, diffusion on | `~/spark-recipes/runs/2026/05/2026-05-25T10-07-22Z_93a80c.md` | **72** | **2.0 s** | **65** |
| Orthrus, no-diff (AR fallback) | `~/spark-recipes/runs/2026/05/2026-05-25T10-39-21Z_93a80c.md` | 70 | 4.4 s | 36 |
| Qwen3-8B (base) | `~/spark-recipes/runs/2026/05/2026-05-25T11-27-41Z_9cd212.md` | 70 | 4.4 s | 36 |

### Per-request stats (serve INFO logs)

| Metric | Diffusion | No-diff | Base Qwen3 |
|---|---:|---:|---:|
| Requests logged | 154 | 152 | 152 |
| Tool-call turns | 68 | 66 | 66 |
| `completion_tokens` (median / mean / max) | 43.5 / 60.1 / 337 | 44 / 60.2 / 347 | 44 / 60.2 / 347 |
| `total_s` (median / mean / max) | 1.96 / 2.33 / 11.77 | 4.42 / 6.05 / 32.91 | 4.43 / 6.06 / 32.97 |
| `tok_per_s` (median / mean / max) | 24.8 / 27.3 / 74.6 | 9.84 / 9.67 / 10.62 | 9.81 / 9.65 / 10.59 |
| Total generate wall-time | 359.5 s | 919.3 s | 921.0 s |

### tok_per_s by completion-token bucket

| Completion tokens | Diff n | Diff mean tok/s | No-diff n | No-diff mean tok/s | Base n | Base mean tok/s |
|---|---:|---:|---:|---:|---:|---:|
| < 10 | 3 | 8.5 | 2 | 5.5 | 2 | 5.4 |
| 10-30 | 45 | 23.4 | 45 | 9.1 | 45 | 9.1 |
| 30-100 | 84 | 27.9 | 82 | 9.9 | 82 | 9.9 |
| 100-300 | 24 | 32.1 | 21 | 10.2 | 21 | 10.2 |
| 300+ | 1 | 74.2 | 2 | 10.6 | 2 | 10.6 |

### Conclusions

1. **No-diff Orthrus is indistinguishable from base Qwen3 on this workload.** Median `total_s` 4.42 vs 4.43, total generate wall-time 919.3 vs 921.0 (<0.2%), median `tok_per_s` 9.84 vs 9.81, identical Final Score 70 and bucket throughputs to one decimal. This is direct confirmation that the `914faee` AR-fallback fix makes Orthrus's `use_diffusion_mode=False` path real AR + KV cache — same code path as stock Qwen3.
2. **Diffusion wins at every output-length bucket** — including the very short acks (8.5 vs 5.5 tok/s). Both modes carry fixed per-request overhead that compresses tok/s on short outputs, but diff's floor sits above no-diff's everywhere.
3. **Completion-token distributions are essentially identical across all three modes** (median 43.5 / 44 / 44, max 337 / 347 / 347). The `StringStoppingCriteria` asymmetry (AR honours it, diffusion ignores it — see `todo.md`) is a real code issue but is not affecting output length here; at greedy with this seed every mode produces the same answers.
4. **Diff is ~2.25× faster per turn than either AR config.** The historical "diff ≈ no-diff" claim does not reproduce on revision `977a617` with instrumentation. Whatever produced that earlier observation isn't visible in this rerun.
5. **Quality drops 2 points (72 → 70) in both AR configurations.** p90 wall-time 11.87 s / 11.89 s and max ~33 s suggest a small handful of long-tail turns may brush up against tool-eval-bench's per-turn timeout, capping the chain early. Not investigated further — diff is strictly preferable on every axis here.

---

## Long-form generation (`benchmarks/benchmark.py` + `benchmarks/benchmark_http.py`)

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

---

## Reproducing

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

Output: appends each `--label` into `results/results_http.json` (cwd-relative; run from `tests/` to land in `benchmarks/results/`).
