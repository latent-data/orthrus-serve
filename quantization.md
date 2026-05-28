# Quantisation in orthrus-serve

In-process quantisation for Orthrus and vanilla Qwen3-8B served from this endpoint. Four production schemes: `fp8` (recommended default), `nvfp4` (4-bit per-block, recommended alternative for memory/throughput priority), `fp8-row` (per-row weights — the fastest 8-bit scheme, viable), `fp8-weight-only` (portable slow-path fallback). Plus two experimental probe schemes, `fp8-row-teacher-only` and `fp8-row-drafter-only`, that quantise only one side of the model to test whether single-side quantisation is worthwhile (it isn't — see "Is quantising only one side worth it?" below). Designed so additional schemes (AWQ-int4, etc.) can be added without changing the call sites.

This doc covers what the feature does, how to verify it works on the target hardware, how it's wired in, and the comparison workflow it enables under tool-eval-bench.

## What it does

`src/orthrus_serve/quantization.py` exposes `apply_quantization(model, scheme)`. The function:

1. Takes a freshly loaded bf16 model (from `AutoModelForCausalLM.from_pretrained`).
2. Wraps eligible `nn.Linear` weights in torchao's `Float8Tensor` subclass via `torchao.quantization.quantize_`. The wrapper keeps the wrapper's `.dtype` as `bfloat16` for transparency but stores the actual data at `.qdata` (`torch.float8_e4m3fn`, half the size).
3. Skips `lm_head` (output projection over the full vocabulary; quantisation here disproportionately affects perplexity and the parameter count is a small fraction of the layer stack).
4. Quantises the Orthrus-specific `_diff` projections alongside the AR projections. PR 2 of orthrus-bench-spark established empirically that the `_diff` projections are a "quantization passenger" (quantising them on top of the AR weights adds no measurable extra output divergence), so the memory savings come for free.
5. Logs memory footprint before and after (reading `.qdata` storage on wrapped tensors, not the wrapper's advertised `.element_size()` which lies about the real storage).

Calibration-free: the conversion is a deterministic per-tensor symmetric scale + cast, identical every run. No saved checkpoint is required, and the load path stays a single call to `from_pretrained` followed by an in-place transform.

## When to use it

Two main reasons:

- **Throughput on Blackwell**: with `ORTHRUS_QUANT=fp8`, torch routes fp8 weight × fp8 activation matmul through `torch._scaled_mm`, the native fp8 path on sm_121 (DGX Spark, B100, B200) and sm_90 (Hopper). Measured per-forward speedup on the Orthrus diffusion-mode short prompt at smoke-test conditions: 21.1 tok/s bf16 → 28.0 tok/s fp8, **+33%**.
- **Memory pressure**: ~1.8x smaller parameter footprint (~18.5 GB bf16 → ~10.4 GB fp8 for Orthrus-Qwen3-8B; not exactly 2x because the embedding and lm_head and norms stay bf16). On a 128 GB Spark this leaves more headroom for KV caches and concurrency.

Neither use case requires accuracy degradation provided you don't load the `_weight-only` scheme by mistake. The primary purpose for this endpoint is the third use case:

- **Apples-to-apples comparison**: serve Orthrus at fp8 and vanilla Qwen3-8B at fp8 through the same endpoint code, benchmark both with tool-eval-bench, and report whether Orthrus retains accuracy parity while delivering its usual ~3-4× wall-clock speedup (or larger, with fp8's additional 33% on top).

## How to enable

Set the `ORTHRUS_QUANT` env var:

| Value | Meaning |
|---|---|
| unset / empty | bf16, no quantisation (default) |
| `fp8` | Fp8 weights (per-tensor symmetric scale) + dynamic per-token activation quantisation, native `_scaled_mm` matmul. ~1.3x speedup, ~1.8x memory reduction on Blackwell. **Recommended default.** |
| `nvfp4` | 4-bit float weights with per-block (~16-element) fp8 scales, dynamic per-tensor activation scaling, native triton-fp4 kernel via `torchao.prototype.mx_formats.NVFP4InferenceConfig`. Smoke 1.24x bf16; long-prompt diffusion serving **1.74x bf16** (88.9 tok/s vs 51.1 bf16, vs 65.3 fp8). **2.89x memory reduction** (18.5 GB → 6.4 GB on 8B Qwen3). Tool-eval-bench 71/100 (vs fp8's 74). Drafter fully intact (median turn 1.4 s). Recommended when memory or throughput dominates accuracy. |
| `fp8-row` | Same as `fp8` but with per-row (a.k.a. per-output-channel) weight scales. Drafter fully intact: diffusion-mode long-prompt throughput **78.6 tok/s** — the fastest 8-bit scheme (vs fp8's 65.3, bf16's 51.1) — and tool-eval-bench 70/100, within a few points of fp8 (74). 1.8x memory, same as fp8. A viable scheme; `fp8` stays the default on a marginal accuracy preference. |
| `fp8-weight-only` | Fp8 weight storage with bf16 matmul (dequant on every forward). ~1.8x memory reduction, ~10x slower throughput. Use only on hardware without `_scaled_mm` support, or for offline analysis where storage is the only thing that matters. |
| `fp8-row-teacher-only` | **Experimental probe.** `fp8-row` applied only to the teacher (AR + shared) Linears; `_diff` drafter projections stay bf16. Probes whether single-side quantisation is worthwhile. See "Is quantising only one side worth it?" below. |
| `fp8-row-drafter-only` | **Experimental probe.** `fp8-row` applied only to the `_diff` drafter projections; teacher stays bf16. The other half of that probe; not a production config (near-zero memory win). See the same section below. |

The scheme names reflect what actually happens at runtime, not the historical torchao naming. The torchao mapping is: `fp8` → `Float8DynamicActivationFloat8WeightConfig()`, `fp8-row` (and the two `fp8-row-*-only` probes) → `Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())` with a teacher/drafter module filter, `fp8-weight-only` → `Float8WeightOnlyConfig()`, `nvfp4` → `torchao.prototype.mx_formats.NVFP4InferenceConfig()`.

Examples:

```bash
# Orthrus at fp8
docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e ORTHRUS_QUANT=fp8 \
    -p 8080:8080 \
    orthrus-serve

# Vanilla Qwen3-8B at fp8 (same image, flag flipped)
docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e ORTHRUS_BASE_MODEL=1 -e ORTHRUS_QUANT=fp8 \
    -p 8080:8080 \
    orthrus-serve
```

## Wire-up

Wired in across five files in this repo:

| File | Change |
|---|---|
| `src/orthrus_serve/quantization.py` | The module itself; `apply_quantization(model, scheme)` API |
| `src/orthrus_serve/settings.py` | `Settings.quant` field, read from `ORTHRUS_QUANT` env var |
| `src/orthrus_serve/model.py` | `apply_quantization(model, settings.quant)` called after `from_pretrained` in both Orthrus and base-model branches |
| `Dockerfile` | No torchao install needed: the NGC container (`nvcr.io/nvidia/pytorch:25.12-py3`) already ships torchao 0.15+ as part of its custom torch build, with `Float8WeightOnlyConfig` and friends. Overriding with a pypi version would either drop NGC patches or pull a generic torch wheel that clobbers sm_121. |
| `run.sh` | `--quant <scheme>` flag plumbs through as `-e ORTHRUS_QUANT=<scheme>` |

Setting `ORTHRUS_QUANT` (or passing `--quant fp8` to `run.sh`) is the only thing the user has to do at runtime. Unset = bf16 (no quantisation, same behaviour as before this feature existed).

## Verification

After wiring the above, before pointing tool-eval-bench at the endpoint, run the smoke test. It loads the model, runs a baseline bf16 generation with diagnostics, applies the requested quant scheme, runs the same generation again, and prints a structured verdict block.

### Smoke test

```bash
docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd):/workspace -w /workspace \
    nvcr.io/nvidia/pytorch:25.12-py3 \
    bash -c 'pip install --no-deps -q transformers==5.8.1 accelerate==1.13.0 && pip install -q -e ".[dev]" && python -m orthrus_serve.quantization --smoke --scheme fp8'
```

Expected verdict block (measured on DGX Spark sm_121, 2026-05-27, Orthrus diffusion-mode short prompt):

```
=== VERDICT ===
  Quantization applied:  PASS (396/396 target Linears wrapped in fp8 tensor subclass)
  Memory reduction:      PASS (1.77x; expected ~1.7-1.8x for fp8 on this model size)
  Throughput vs bf16:    28.0 vs 21.1 tok/s; PASS (1.33x faster than bf16; native fp8 matmul kernels firing as expected)
  Output sanity:         PASS (sensible decoded text; differs from bf16 as expected for a perturbed model)

OVERALL: GOOD (fp8 working as expected on this hardware: applied, memory halved, throughput 1.33x bf16).
```

If any of the four lines come back as FAIL or REGRESSION, the wire-up or the hardware is the place to look. The diagnostic distinguishes:

- **Quantization applied** counts how many `nn.Linear` weights are wrapped in `Float8Tensor` (torchao's tensor subclass with `.qdata` pointing at the fp8 storage). The wrapper reports its `.dtype` as `bfloat16` for transparency; the diagnostic looks past that to detect the wrapper.
- **Memory reduction** reads the underlying `.qdata` storage rather than the wrapper's lying `.element_size()`. The ceiling is ~1.8x (not 2.0x) for an 8B model because the embedding and lm_head stay bf16; the verdict accounts for this.
- **Throughput vs bf16** runs a 32-token generation under each precision with warmup. For `fp8` (activation+weight), expect ~1.2-1.4x speedup. For `fp8-weight-only`, expect ~10x SLOWER than bf16 (dequant-then-bf16-matmul); the verdict labels that as `EXPECTED`, not a failure.
- **Output sanity** checks that the post-quant output isn't empty, isn't a single token repeated, and decodes to sensible text. Bit-identity to bf16 is rare (quantisation usually perturbs at least one near-tie) and not a problem.

## Comparison workflow under tool-eval-bench

The headline claim this feature enables is "Orthrus fp8 has the same accuracy as Qwen fp8 but lower median turn time." To produce that comparison cleanly:

### 1. Run two endpoints sequentially (or in parallel on different ports)

```bash
# Terminal A: Orthrus fp8
docker run --rm --gpus all ... \
    -e ORTHRUS_QUANT=fp8 \
    -p 8080:8080 \
    orthrus-serve

# Terminal B: vanilla Qwen3-8B fp8 (same container, same precision, just the model differs)
docker run --rm --gpus all ... \
    -e ORTHRUS_BASE_MODEL=1 -e ORTHRUS_QUANT=fp8 \
    -p 8081:8081 \
    orthrus-serve
```

If running concurrently, each will use ~10 GB at fp8, fitting comfortably in 128 GB unified memory along with KV caches.

### 2. Run tool-eval-bench against each `--base-url` separately

```bash
tool-eval-bench --base-url http://localhost:8080/v1 --model orthrus-qwen3-8b ...
tool-eval-bench --base-url http://localhost:8081/v1 --model qwen3-8b ...
```

### 3. Report

Two numbers per benchmark:

- **Accuracy** (the bench's primary metric). Orthrus is expected to match vanilla Qwen3 within noise. If it does, the quantisation does not specifically harm Orthrus.
- **Median turn time**. Orthrus is expected to be ~3-4× faster (matching its bf16 speedup). At fp8 the advantage may be larger if the AR-mode forward path benefits less from fp8 matmul than the diffusion path (which is matmul-heavy through the AR head's verify call).

The structural argument that this comparison is fair: under the frozen-teacher claim, Orthrus's AR projections are vanilla Qwen3-8B weights, so fp8 cast-and-dequant perturbs them identically in both endpoints. PR 4 of orthrus-bench-spark established this empirically (bit-identical divergence patterns between Orthrus AR-mode and vanilla AR-mode at the simulated-int8 level).

## All PTQ schemes preserve the diffusion drafter

Every calibration-free post-training quantisation scheme we have measured — `fp8` (per-tensor), `fp8-row` (per-row), and `nvfp4` (per-block) — keeps the diffusion drafter accepting at high rates. None of them disrupts the drafter↔teacher alignment. The diffusion speedup carries through quantisation cleanly, and quant throughput stacks on top of it.

HTTP throughput (`benchmarks/results/results_http.json`), diffusion-mode long prompt vs the no-diffusion (AR) floor at the same precision:

| Scheme | diffusion short | diffusion long | no-diff long | drafter speedup (long) |
|---|---:|---:|---:|---:|
| bf16 | 38.8 | 51.1 | 10.9 | 4.7× |
| fp8 (per-tensor) | 43.5 | 65.3 | 14.0 | 4.7× |
| **fp8-row (per-row)** | **47.5** | **78.6** | 16.1 | **4.9×** |
| nvfp4 (per-block) | 47.3 | 88.9 | — | drafter intact (1.4 s median turn) |

The diffusion arm runs **3-5× faster than the AR floor under every scheme** — the unambiguous signature of an active drafter. Per-row is in fact the fastest 8-bit scheme on long-form generation (78.6 tok/s, above per-tensor fp8's 65.3), because its kernel path and accept rate are both healthy on sm_121.

tool-eval-bench accuracy (diffusion mode) clusters tightly across all schemes:

| Scheme | Final score | Median turn | Safety-critical fails |
|---|---:|---:|---:|
| bf16 | 72 | 2.0 s | 4 |
| fp8 | 74 | 1.7 s | 4 |
| **fp8-row** | **70** | **1.9 s** | 5 |
| nvfp4 | 71 | 1.4 s | 5 |

The 4-point spread (70-74) is within the bench's near-tie noise floor (≈3 scenarios flipping on a 69-scenario bench at this seed); treat the four schemes as accuracy-comparable. The safety-critical count drifts by one (TC-58, an API-key-leak near-tie that has flipped across bf16/fp8/nvfp4 too) — also noise, not a per-scheme property.

**Scheme choice** is therefore a memory/throughput/accuracy trade-off, not a drafter-survival question:
- `fp8` — recommended default: best accuracy (74), 1.8× memory.
- `fp8-row` — viable; fastest 8-bit long-form throughput, accuracy within noise of fp8, same 1.8× memory. `fp8` stays the default only on the marginal accuracy edge.
- `nvfp4` — recommended when memory or throughput dominates: 2.89× memory, fastest median turn, ~3-point accuracy cost vs fp8.

**Note on the smoke test:** it runs a single-prompt HF `generate` in AR mode, which never invokes the drafter, so it cannot measure drafter behaviour at all. The diffusion-mode HTTP throughput bench (diffusion long-prompt tok/s vs the AR floor) is the only thing that does. When benchmarking the diffusion arm, confirm the server actually started in diffusion mode (`"diffusion": true` in the `model_ready` log) — a diffusion-labelled run against an accidentally `--no-diffusion` server looks identical to a dead drafter.

## Is quantising only one side worth it?

The teacher (AR + shared backbone, ~84% of params) and the drafter (`_diff` projections, ~16%) can be quantised independently. The `fp8-row-teacher-only` and `fp8-row-drafter-only` probe schemes test whether a single-side split buys anything. It doesn't:

| Scheme | Teacher | Drafter | diffusion long tok/s | Memory | Verdict |
|---|---|---|---:|---:|---|
| fp8-row | per-row | per-row | 78.6 | 1.8× | the thing to ship |
| fp8-row-teacher-only | per-row | bf16 | 77.9 | 1.56× | drafter alive, but quantises 16% fewer params for no throughput gain |
| fp8-row-drafter-only | bf16 | per-row | 50.4 | 1.08× | drafter alive but teacher matmul unaccelerated; near-zero memory win |

The drafter survives in all three (consistent with the section above — per-row doesn't hurt it). But splitting gains nothing: full `fp8-row` already keeps the drafter *and* quantises the whole model, so teacher-only just leaves free memory on the table, and drafter-only barely moves memory while forgoing the teacher's fp8 matmul speedup. Conclusion: **quantise everything; there's no reason to spare one side.** (This also re-confirms PR 2 of orthrus-bench-spark's "`_diff` is a quantisation passenger" finding — quantising the drafter alongside the teacher is harmless.)

For non-Orthrus models that don't have a drafter, the drafter considerations are moot entirely — pick a scheme on the bit-width / memory / accuracy trade-off alone.

The two probe schemes share one code path: `_apply_fp8_dynamic_activation_weight(model, per_row=True, target=...)` with `target="teacher"` (skip `_diff` Linears) or `target="drafter"` (quantise only `_diff` Linears). The split is by FQN substring `_diff`, matching orthrus-bench-spark's teacher/drafter convention. To reproduce: serve with `--quant fp8-row-teacher-only` (or `-drafter-only`) and run `benchmarks/benchmark_http.py` in diffusion mode.

## What this is not

- **Not a saved checkpoint pipeline.** Each serve startup re-applies the quantisation in place. For fp8 weight-only this is fine (calibration-free, deterministic, fast). For calibrated formats (AWQ = Activation-aware Weight Quantization, GPTQ = Generative Pre-trained Transformer Quantization; both algorithms compute per-weight scales using a small representative dataset rather than from the weights alone) where "quantise once, distribute the artifact" is the right pattern, that work would graduate to a separate `orthrus-quant` project that produces HF-compatible quantised checkpoints.
- **Not a serving-stack swap.** Still HF transformers + custom Orthrus generate loop. vLLM does not currently support Orthrus's diffusion mode, so this implementation runs the same serving path as bf16, just with quantised weights underneath.
- **Not validated for accuracy beyond smoke testing.** The accuracy validation is what tool-eval-bench is for. If a quant scheme harms task accuracy meaningfully, the eval will surface it; the in-process verification only confirms forward-pass sanity.

## Caveats and known limitations

- **Torchao on NGC pytorch IS supported in this configuration** (smoke test on 2026-05-27 confirms `torch._scaled_mm` fires on sm_121 and produces the expected 1.33x speedup). The NGC container (`nvcr.io/nvidia/pytorch:25.12-py3`) ships its own torchao build matched to its custom torch; do not override it with a pypi version.
- **Two scheme names matter; pick the right one.** `fp8` is the recommended path: native fp8 matmul, ~33% faster than bf16 on Blackwell, ~1.8x memory reduction. `fp8-weight-only` is much slower than bf16 (~10x measured on a toy model) because torchao dequantises the weight back to bf16 for every matmul; use it only on hardware without `_scaled_mm` support.
- **Default weight scaling is per-tensor.** The `fp8` scheme uses per-token dynamic scaling for activations and per-tensor symmetric scaling for weights (one scalar per Linear). Per-row weight scaling is available via the `fp8-row` scheme (added 2026-05-27), which is slightly higher fidelity at near-zero memory cost; throughput on sm_121 is ~12% behind per-tensor (24.8 vs 28.0 tok/s short-prompt smoke test) because cuBLAS's per-row kernel is less tuned on Blackwell.
- **`lm_head` is skipped.** Standard practice; can be revisited if memory pressure on the head becomes meaningful.
- **No handling for activation outliers.** The dynamic per-token activation scaling in `fp8` handles most outlier cases for Qwen3-class models. For pathological activations, an outlier-aware scheme like SmoothQuant would be the next step.
- **The Orthrus consensus mechanism's TPF under real fp8 is not yet measured.** PR 2 of orthrus-bench-spark measured TPF drops of ~7-10% under simulated int8 (cast-and-dequant in bf16). Real fp8 with native matmul should land in a similar regime or better, but verify empirically with a longer run (the smoke test's 32-token short prompt is too small to give reliable TPF; run a full benchmark prompt).
- **int8 tried and rejected (2026-05-27).** Briefly added `Int8DynamicActivationInt8WeightConfig` as an `int8` scheme to test whether an int (rather than float) 8-bit format was viable on this stack. Two problems made it unusable: (1) torchao 0.15's default int8 path is **per-channel** (per-row weight scales), not per-tensor, so it can't act as the per-tensor control we wanted; (2) the int8 matmul kernel is not wired up on sm_121 in torchao 0.15, falling through to a dequant-then-bf16-matmul slow path ~6.4x slower than bf16 on a 4096x4096 toy linear. Code removed; the negative finding survives as this note. NVFP4 may be a better next candidate for the same "different precision, uniform per-tensor" experiment if its sm_121 kernel state is healthier.

## Future work

Wire-up validation is done (the smoke test passes OVERALL: GOOD on DGX Spark sm_121 as of 2026-05-27). Remaining in rough order of value:

1. **tool-eval-bench parity comparison.** Run Orthrus-fp8 vs Qwen3-fp8 endpoints under tool-eval-bench, report accuracy parity and median turn time. This is the immediate next experiment.
2. **Longer-prompt TPF measurement.** The smoke test's 32-token short prompt is too small for stable TPF; rerun on the orthrus-bench-spark long prompt to confirm the consensus mechanism still accepts at the rates predicted by PR 2 (TPF ~6.0+ at fp8 vs ~6.56 at bf16).
3. **Per-row tool-eval-bench comparison.** With the `fp8-row` scheme now wired in (2026-05-27), the open question is whether per-row scaling moves the tool-eval-bench score above the 74/100 ceiling that all three `fp8` arms tied at. Tiered plan: run the diffusion arm first as a gate, then full sweep only if it shows a meaningful gain over per-tensor.
4. **NVFP4 (fp4 with native Blackwell support).** Spark targets fp4-native paths through `torch._scaled_mm` for additional memory and throughput; torchao support exists.
5. **AWQ-int4 (calibrated).** Standard 4× memory reduction with calibration. Needs representative data, separate pipeline. Reasonable point to split out an `orthrus-quant` repo.
6. **Quantisation-aware diffusion drafter retraining.** If accuracy parity breaks at int4, the diffusion drafter could be retrained against the quantised teacher to restore alignment. Out of scope for orthrus-serve; would belong upstream in the Orthrus training code.

## Related

- `src/orthrus_serve/quantization.py`: the implementation module.
- orthrus-bench-spark/quant_benchmark.py: empirical investigation of how simulated int8/int4 quantisation affects Orthrus's TPF and output-equivalence properties. The findings there (passenger `_diff` projections, near-tie-cascade divergence mechanism, Orthrus not uniquely fragile vs vanilla Qwen3) inform the design choices in this module.
