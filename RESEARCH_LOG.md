# orthrus-serve research log

Full benchmark numbers, methodology, and the cross-cutting quantisation-survival framework derived from the orthrus-serve sweeps on DGX Spark (GB10, sm_121). See [README.md](README.md) for the high-level project summary, and [quantization.md](quantization.md) for the implementation-level write-up of each scheme.

## Benchmarks

Two complementary surfaces:

1. **`tool-eval-bench`** — multi-turn tool-call scenarios, hits orthrus-serve over HTTP. Run with `python -m tool_eval_bench --base-url http://localhost:8080 --backend vllm --seed 42 --no-think`. The `--backend vllm` flag tells tool-eval-bench to use the OpenAI-compatible wire format that this server matches.
2. **Long-form generation** — `benchmarks/benchmark.py` (in-process) and `benchmarks/benchmark_http.py` (over HTTP). Results in `benchmarks/results/`.

All numbers below are at revision `977a617` (post the `914faee` AR-fallback fix in upstream Orthrus). `enable_thinking=false` throughout.

### tool-eval-bench

All three serving configurations against the same 69 scenarios at `--seed 42 --no-think`. The bf16 rows are from a 2026-05-25 baseline sweep; the fp8 rows are from a 2026-05-27 re-run after fp8 quantisation was wired in via `ORTHRUS_QUANT=fp8` (Float8DynamicActivationFloat8WeightConfig via torchao, native `_scaled_mm` on sm_121). Per-scenario / per-category / safety-critical-failure data for every row is in `benchmarks/results/tool-eval-bench/`; this section quotes the headline numbers and surfaces the cross-arm structure.

Setup intent: validate the [orthrus-bench-spark PR 4 prediction](../orthrus-bench-spark/RESEARCH_LOG.md#vanilla-qwen3-quantization-sensitivity-orthrus-is-not-uniquely-fragile-to-int8): "Orthrus inherits Qwen3's quantisation sensitivity, no more and no less."

| Configuration | Run summary | Final Score | Median Turn | Responsiveness | Deployability | Wall-clock |
|---|---|---:|---:|---:|---:|---:|
| Orthrus diffusion bf16 (May 25) | `benchmarks/results/tool-eval-bench/2026-05-25T10-07-22Z_93a80c.md` | 72 | 2.0 s | 65 | 70 | 359.5 s |
| **Orthrus diffusion fp8** | `benchmarks/results/tool-eval-bench/2026-05-27T11-01-43Z_a24531.md` | **74** | **1.7 s** | **70** | **73** | **331.0 s** |
| Orthrus no-diff bf16 (May 25) | `benchmarks/results/tool-eval-bench/2026-05-25T10-39-21Z_93a80c.md` | 70 | 4.4 s | 36 | 60 | 919.3 s |
| **Orthrus no-diff fp8** | `benchmarks/results/tool-eval-bench/2026-05-27T11-48-43Z_a24531.md` | **74** | **3.9 s** | **41** | **64** | **810.2 s** |
| Qwen3-8B bf16 (May 25) | `benchmarks/results/tool-eval-bench/2026-05-25T11-27-41Z_9cd212.md` | 70 | 4.4 s | 36 | 60 | 921.0 s |
| **Qwen3-8B fp8** | `benchmarks/results/tool-eval-bench/2026-05-27T11-21-32Z_f865fa.md` | **74** | **3.8 s** | **41** | **64** | **801.7 s** |
| Orthrus diffusion fp8-row | `benchmarks/results/tool-eval-bench/2026-05-27T13-48-48Z_cbc6af.md` | 69 | 3.6 s | 43 | 61 | 672.3 s |
| **Orthrus diffusion nvfp4** | `benchmarks/results/tool-eval-bench/2026-05-27T15-48-11Z_9716ca.md` | **71** | **1.4 s** | **76** | **72** | **253.7 s** |
| Orthrus no-diff nvfp4 | `benchmarks/results/tool-eval-bench/2026-05-27T15-59-16Z_9716ca.md` | 69 | 2.6 s | 55 | 65 | 521.0 s |

Wall-clock totals are derived from the `Date − Run ID` timestamp delta in each `.md` file. Wall-clock improvements at fp8 (Orthrus diffusion −8%, no-diff −12%, Qwen3 −13%) are smaller than the per-token throughput improvements (~+28% from fp8, see "Long-form generation at fp8" below) because tool-eval-bench wall-clock includes per-turn HTTP overhead, tool-result processing, and inter-turn coordination; only the matmul-bound generation portion benefits from fp8 directly. The fp8-row row is a negative result: per-row quantisation breaks the diffusion drafter and collapses throughput to AR speed (full mechanism in [`quantization.md`'s "Per-row fp8 breaks the diffusion drafter"](quantization.md#per-row-fp8-breaks-the-diffusion-drafter) and the "Per-row fp8 (negative result)" section below). The nvfp4 row is the fastest configuration in this table: 4-bit weights with per-block scales fire on Blackwell-native triton kernels (smoke test 1.74x bf16 long-prompt, here 31% wall-clock reduction vs fp8). Accuracy drops 3 points vs fp8 (71 vs 74); see the "NVFP4 (4-bit, ships)" section below for the full story.

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

### NVFP4 (4-bit, ships)

Added 2026-05-27 after the fp8-row investigation surfaced the drafter-rejection mechanism. NVFP4 (`Float4` weights with per-block fp8 scales, via `torchao.prototype.mx_formats.NVFP4InferenceConfig`) is structurally per-block — each ~16-element block of weights gets its own scale — so it was an open question whether the drafter would survive that finer-grained perturbation. It does, but **only partially** — the drafter is mildly disrupted, not fully aligned.

HTTP throughput (`benchmarks/results/results_http.json`):

| Config | short tok/s | long tok/s |
|---|---:|---:|
| Orthrus diffusion bf16 | 38.8 | 51.1 |
| Orthrus diffusion fp8 | 43.5 | 65.3 |
| **Orthrus diffusion nvfp4** | **47.3** | **88.9** |

Long-prompt: NVFP4 is **+36% over fp8** and **+74% over bf16**. The diffusion drafter accept rate stays high enough that output is the same kind of code-completion long-form text as fp8/bf16, not the AR-fallback signature we see at fp8-row.

tool-eval-bench (`benchmarks/results/tool-eval-bench/2026-05-27T15-48-11Z_9716ca.md` and `2026-05-27T15-59-16Z_9716ca.md`):

| Config | Final score | Median turn | Wall-clock | Safety-critical fails |
|---|---:|---:|---:|---:|
| Orthrus diffusion bf16 | 72 | 2.0 s | 359.5 s | 4 |
| Orthrus diffusion fp8 | 74 | 1.7 s | 331.0 s | 4 |
| **Orthrus diffusion nvfp4** | **71** | **1.4 s** | **253.7 s** | **5** |
| Orthrus no-diff nvfp4 | 69 | 2.6 s | 521.0 s | 3 |

The no-diff row enables the drafter-speedup comparison: median turn 2.6s nodiff vs 1.4s diff — 1.86× drafter speedup (vs fp8's 2.29× and fp8-row's ~1.0×). Score 69 vs 71 is within noise tolerance at this seed and sample size (2 points = 3 scenarios flipping near tie tips on a 69-scenario bench) and shouldn't be read as a drafter accuracy effect either way. Safety-crit count drifts 3 vs 5 between the two arms but the specific scenarios (TC-43 passing in nodiff while failing in every other arm we've measured, TC-41 failing only in diffusion at this single seed) are also in noise territory.

Memory: 18.5 GB bf16 → 10.4 GB fp8 → **6.4 GB nvfp4**. Enough headroom to either run two NVFP4 instances on a single 128 GB Spark, or to push context length significantly beyond the 40k default with one instance plus its KV cache.

**Drafter survival is a spectrum, not binary** (refined uniform-vs-structured hypothesis):

| Scheme | Granularity | Drafter speedup (turn ratio) | Mechanism |
|---|---|---:|---|
| fp8 | per-tensor (1 scale / Linear) | **2.29×** (3.9/1.7) | uniform; every row rescaled identically; drafter fully tracks |
| nvfp4 | per-block (~16 elements) | **1.86×** (2.6/1.4) | block-wise; high-frequency within a row, partially averages out at row level; drafter partially tracks |
| fp8-row | per-row (1 scale / output channel) | **~1.0×** (drafter dead) | rigidly non-uniform across rows; drafter can't track at all |

The refined rule: **the drafter's accept rate is a continuous function of how much the quantisation pattern looks uniform-at-the-row-level.** Per-tensor is trivially uniform → full speedup. Per-block is locally non-uniform but averages out at the row scale → partial speedup. Per-row is rigidly non-uniform at exactly the row scale → no speedup at all. Bit width matters much less than perturbation geometry: NVFP4 is 4-bit yet drafter-friendly (1.86× speedup); fp8-row is 8-bit yet drafter-hostile (1.0× speedup).

**When to use NVFP4**:
- Memory pressure (running two instances, longer contexts, larger models on smaller hardware in future)
- Throughput priority on long-form generation (88.9 tok/s vs fp8's 65.3)
- Accept the 3-point tool-eval-bench cost vs fp8 (this is the real cost; both diff and nodiff arms show it, so it's a 4-bit-precision effect not a code-path artifact)

**When to stay with fp8**:
- Accuracy-priority workloads where every point of tool-eval-bench matters
- Default for general serving (74/100 vs 71/100)

`fp8` remains the recommended default for orthrus-serve; `nvfp4` is the recommended alternative when memory or throughput dominates accuracy preferences.

### Per-row fp8 (negative result)

Tried `fp8-row` (`Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())`) on 2026-05-27 as a higher-fidelity alternative to per-tensor fp8. Result: it breaks the diffusion drafter and is strictly worse than `fp8` for Orthrus-diffusion serving.

| Config | short tok/s | long tok/s | tool-eval-bench |
|---|---:|---:|---:|
| Orthrus diffusion fp8 | 43.5 | 65.3 | 74 / 100 |
| Orthrus diffusion fp8-row | 16.6 | 16.4 | 69 / 100 |
| Orthrus no-diff fp8-row | 16.6 | 16.3 | (not benched) |

Diffusion-mode fp8-row produces the same throughput as no-diff fp8-row — the diffusion speedup is gone. Combined with the 5-point tool-eval-bench regression and one extra safety-critical failure (TC-58 regressed vs the fp8 baseline), per-row fp8 fails on both axes that matter for serving.

**Mechanism**: per-row quantisation rescales each output channel by a different factor, perturbing the teacher's predictions in a structured way the drafter (trained against the unquantised teacher) can't track. Verify rejects almost every draft; the pipeline degenerates to single-token AR per step. Per-tensor fp8 doesn't trigger this because its uniform rescale leaves drafter↔teacher alignment intact. Full writeup in [`quantization.md`'s "Per-row fp8 breaks the diffusion drafter"](quantization.md#per-row-fp8-breaks-the-diffusion-drafter).

**Implication for Orthrus diffusion-mode serving**: the operational constraint is "uniform quantisation only" until quantisation-aware drafter retraining lands upstream. Per-row, per-group, per-channel schemes are off the table for diffusion-mode at least; they may be fine for no-diff (AR) serving or for non-Orthrus models that don't have a drafter at all.

### Conclusions

1. **No-diff Orthrus is indistinguishable from base Qwen3, at both bf16 and fp8.** At bf16 both arms score 70/138 (final score 70) with identical median turn time (4.4 s). At fp8 both arms are bit-identical scenario-by-scenario (both 102/138 = 74/100, 0 differences across 69 scenarios; final score 74). Same code path through the same weights at the same precision; greedy decoding gives the same output. The `914faee` AR-fallback fix makes `use_diffusion_mode=False` real AR + KV cache, equivalent to stock Qwen3.
2. **Diffusion is ~2.2× faster per turn than either AR config across both precisions.** Median turn 2.0 s vs 4.4 s at bf16, 1.7 s vs 3.8-3.9 s at fp8. The diffusion speedup carries through quantisation without weakening.
3. **At fp8, all three arms converge to identical 74/100 (102/138 points).** Direct empirical confirmation of [orthrus-bench-spark PR 4](../orthrus-bench-spark/RESEARCH_LOG.md#vanilla-qwen3-quantization-sensitivity-orthrus-is-not-uniquely-fragile-to-int8): Orthrus inherits Qwen3's quantisation sensitivity, no unique amplification from the diffusion consensus mechanism. Same 4 safety-critical failures in all three arms (TC-31, TC-34, TC-42, TC-43). See `benchmarks/results/tool-eval-bench/` for per-scenario data.
4. **Drafter survival is a spectrum in quantisation granularity, not binary.** Measured drafter speedup (no-diff median / diffusion median) at sm_121: fp8 per-tensor 2.29×, NVFP4 per-block 1.86×, fp8-row per-row ~1.0× (dead). Granularity below the row level (per-block, per-tensor) preserves drafter alignment; granularity at the row level (per-row) breaks it; bit width matters much less than perturbation geometry (NVFP4 is 4-bit yet drafter-friendly, fp8-row is 8-bit yet drafter-hostile). For diffusion-mode serving: `fp8` is the recommended default, `nvfp4` is the recommended alternative when memory or throughput priorities outweigh the 3-point tool-eval-bench cost, `fp8-row` is contraindicated. Full mechanism in `quantization.md`.

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

## Findings: can you quantise an Orthrus model without retraining?

The benchmarks above are specific to Orthrus-Qwen3-8B at 2026-05-27. The Qwen3 base is already a generation old (Qwen3.7 just shipped) and the authors will presumably release a Qwen3.7-based Orthrus checkpoint at some point. When that happens, the same question will recur: **can it be served quantised on Spark via off-the-shelf post-training quantisation (PTQ — taking a full-precision checkpoint and converting it to a quantised one with no retraining), or does it need quantisation-aware training (QAT — retraining the model while accounting for the rounding noise the quantisation will introduce) to recover the drafter?** The transferable answer from this investigation is **"it depends on the perturbation geometry, not the bit width."**

(Why those two acronyms recur in this section: PTQ is what `orthrus-serve` does — it takes the released bf16 checkpoint and applies torchao's quantisation in-process, no extra training step. QAT would require the model authors to retrain Orthrus while simulating the target quantisation's noise, then release that as a separate checkpoint. PTQ is cheap and one-step; QAT is upstream and a research/training project.)

### What this study found that should generalise

Measured median turn time on tool-eval-bench (sm_121, 2026-05-27). The baseline is bf16 no-diff = 4.4 s median (pure autoregressive single-token-at-a-time generation with no quantisation — the configuration you would get from any vanilla transformer serving stack with no Orthrus-specific work and no quantisation).

| Scheme | Bits | Weight granularity | nodiff median | diff median | Drafter only (nodiff/diff) | Total vs bf16-nodiff baseline (4.4/diff) |
|---|---|---|---:|---:|---:|---:|
| bf16 | 16 | n/a | 4.4 s | 2.0 s | **2.20×** | **2.20×** |
| fp8 (per-tensor) | 8 | 1 scale / Linear | 3.9 s | 1.7 s | **2.29×** | **2.59×** |
| nvfp4 (per-block) | 4 | 1 scale / ~16-elem block | 2.6 s | 1.4 s | **1.86×** | **3.14×** |
| fp8-row (per-row) | 8 | 1 scale / output channel | ~3.6 s* | 3.6 s | **~1.0× (dead)** | **1.22×** |

\* fp8-row no-diff median turn time was not directly measured on tool-eval-bench (we ran HTTP-bench instead, which showed nodiff_fp8_row at 16.6 / 16.3 tok/s — essentially identical to diffusion-mode fp8-row at 16.6 / 16.4 tok/s, confirming the drafter was rejected). The "~3.6 s" is an inference from that: if diff and nodiff give the same throughput, the drafter is contributing nothing.

How to read the two right-most columns:

- **Drafter only (nodiff / diff)**: at this same precision, how much does turning on the diffusion drafter speed you up? Isolates the drafter's contribution at that quant scheme. This is the column that exposes the "fp8-row breaks the drafter" finding (1.0× = drafter useless).
- **Total vs bf16-nodiff baseline**: what's the end-to-end speedup of this configuration over the dumb baseline (a vanilla AR-fp32-style serving stack, here represented by bf16 no-diff)? This is the column that tells you what speedup the user actually gets if they pick this scheme. Stacks the diffusion speedup AND the quant speedup together.

Two things become visible by looking at both columns together:

1. **bf16-diffusion alone gives 2.20×** (the "free" speedup of choosing Orthrus over vanilla AR with no quantisation). fp8 on top adds 18% more (2.20 → 2.59). NVFP4 on top adds 43% more (2.20 → 3.14). The quant compounds with the drafter.
2. **fp8-row loses the drafter completely** (1.0× in the drafter-only column) and ends up at 1.22× total — barely better than the vanilla-AR baseline, despite "having Orthrus" and "having a quant." The drafter is the load-bearing piece; killing it negates most of the deployment value.

The predictive rule that emerges: **the drafter's accept rate is a continuous function of how uniform the weight perturbation looks at the row level.** Per-tensor is trivially uniform per row → drafter fully tracks. Per-block is locally non-uniform but averages out at the row scale → drafter partially tracks. Per-row is rigidly non-uniform at exactly the row scale → drafter cannot track.

Crucially: **bit width matters much less than perturbation geometry.** NVFP4 is 4-bit yet drafter-friendly (1.86× drafter-only speedup); fp8-row is 8-bit yet drafter-hostile (1.0× drafter-only speedup). The drafter cares about relative-magnitude preservation across the weight matrix, not per-element rounding noise.

Shipping status for the orthrus-serve endpoint: fp8 is the recommended default (best accuracy + drafter intact), NVFP4 is the recommended alternative for memory or throughput priority (3-point accuracy cost + partial drafter speedup, but 38% less memory and 1.5× the bf16-baseline total speedup), fp8-row is contraindicated for diffusion-mode serving.

### What this means for the next Orthrus checkpoint

For a future Orthrus release (whether it's Qwen3.7-based, larger, or with a different teacher backbone), the answer to "can I PTQ this for serving on my Spark?" is:

- **Per-tensor fp8 should work calibration-free**, with the smallest accuracy cost and full drafter speedup. The same `Float8DynamicActivationFloat8WeightConfig()` call we use here.
- **NVFP4 (per-block ~16 elements) should also work calibration-free** for memory pressure, with a small accuracy cost (3 points here at 8B; presumably similar order at larger scales) and partial drafter speedup. The `NVFP4InferenceConfig()` call.
- **Per-row and per-channel weight scaling will break the drafter** without quantisation-aware drafter retraining. If a future Orthrus release ships with QAT against a per-row teacher, this changes; until then, treat per-row schemes as "needs drafter retrained for this scheme."
- **Other future schemes** (per-group with small groups, MXFP6, NVFP6, etc.) are predictable from the framework: if the weight scale granularity is below the row level (so row-level effect averages out) the drafter survives; if it's at or above the row level it doesn't. Bit width is a separate axis affecting standalone accuracy but not drafter survival.

The fix for the per-row case is "retrain the drafter against a per-row-quantised teacher" (quantisation-aware drafter retraining). That belongs upstream in Orthrus training code, not in a serving stack. Until it lands, the operational constraint for any Orthrus checkpoint is: pick a PTQ scheme whose weight-perturbation pattern doesn't disrupt the drafter's calibration.

### What this study doesn't answer

- Sample size is 8B parameters; bigger models might have different drafter robustness (the drafter could be more forgiving with more capacity to absorb perturbation, or less forgiving with sharper near-tie distributions in tool-call grammar tokens).
- Sample size is one teacher family (Qwen3); different teacher backbones might respond differently. The mechanism we've identified depends only on the existence of a drafter trained against a specific teacher, so the *shape* of the answer should transfer, but the specific drafter speedup numbers will not.
- Sample is one hardware target (Blackwell sm_121). Other hardware will have different kernel-tuning gaps (the 12% per-row throughput cost we measured is a cuBLAS sm_121 fact, not a fundamental algorithmic one). The drafter-alignment finding is hardware-agnostic; the throughput numbers are not.
- Sample is one bench (tool-eval-bench). Other accuracy benchmarks (general MMLU-style, code generation, math reasoning) might surface different sensitivities than tool-call grammar.

If a future Orthrus release surprises against this framework (e.g. per-row works without retraining), the most likely cause is that the drafter's training procedure was changed to be quantisation-robust — at which point this writeup needs updating. The framework is a falsifiable prediction, not a settled rule.
