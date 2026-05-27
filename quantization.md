# Quantisation in orthrus-serve

In-process quantisation for Orthrus and vanilla Qwen3-8B served from this endpoint. Four schemes: `fp8` (recommended default), `nvfp4` (4-bit per-block, recommended alternative for memory/throughput priority), `fp8-row` (per-row weights — breaks the diffusion drafter, see "Per-row fp8 breaks the diffusion drafter" below), `fp8-weight-only` (portable slow-path fallback). Designed so additional schemes (AWQ-int4, etc.) can be added without changing the call sites.

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
| `nvfp4` | 4-bit float weights with per-block (~16-element) fp8 scales, dynamic per-tensor activation scaling, native triton-fp4 kernel via `torchao.prototype.mx_formats.NVFP4InferenceConfig`. Smoke 1.24x bf16; long-prompt diffusion serving **1.74x bf16** (88.9 tok/s vs 51.1 bf16, vs 65.3 fp8). **2.89x memory reduction** (18.5 GB → 6.4 GB on 8B Qwen3). Tool-eval-bench 71/100 (vs fp8's 74), drafter partially survives (1.86× speedup, vs fp8's 2.29× and fp8-row's 1.0×). Recommended when memory or throughput dominates accuracy. See "Drafter survival is a spectrum" below. |
| `fp8-row` | Same as `fp8` but with per-row (a.k.a. per-output-channel) weight scales. **Breaks the diffusion drafter** (drafter rejected almost every iteration; diffusion-mode throughput collapses to AR speed; tool-eval-bench drops 5 points). Throughput on Blackwell sm_121 in pure AR mode: ~12% slower than `fp8` per-tensor (24.8 vs 28.0 tok/s short-prompt). May be useful for non-Orthrus models that don't have a drafter, or AR-mode-only Orthrus serving. See "Per-row breaks the diffusion drafter" below before enabling. |
| `fp8-weight-only` | Fp8 weight storage with bf16 matmul (dequant on every forward). ~1.8x memory reduction, ~10x slower throughput. Use only on hardware without `_scaled_mm` support, or for offline analysis where storage is the only thing that matters. |

The scheme names reflect what actually happens at runtime, not the historical torchao naming. The torchao mapping is: `fp8` → `Float8DynamicActivationFloat8WeightConfig()`, `fp8-row` → `Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())`, `fp8-weight-only` → `Float8WeightOnlyConfig()`, `nvfp4` → `torchao.prototype.mx_formats.NVFP4InferenceConfig()`.

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

## Per-row fp8 breaks the diffusion drafter

Empirical finding from 2026-05-27 sweeps. The mechanism is load-bearing for which scheme to pick, so it gets its own section here even though the code change for `fp8-row` was small.

### Setup and result

Three runs of HTTP benchmark (orthrus_diffusion mode, short + long prompts) and one tool-eval-bench run at fp8-row, compared to the fp8 baseline already documented above.

HTTP throughput (`benchmarks/results/results_http.json`):

| Config | short tok/s | long tok/s |
|---|---:|---:|
| Orthrus diffusion bf16 | 38.8 | 51.1 |
| Orthrus diffusion fp8 | **43.5** | **65.3** |
| Orthrus diffusion fp8-row | 16.6 | 16.4 |
| Orthrus no-diff fp8 | 14.2 | 14.0 |
| Orthrus no-diff fp8-row | 16.6 | 16.3 |

tool-eval-bench (`benchmarks/results/tool-eval-bench/2026-05-27T13-48-48Z_cbc6af.md`):

| Config | Final score | Median turn | Safety-critical fails |
|---|---:|---:|---:|
| Orthrus diffusion fp8 | 74 | 1.7 s | 4 |
| Orthrus diffusion fp8-row | **69** | **3.6 s** | **5** (TC-58 regressed) |

Two things to notice:

1. **fp8-row diffusion ≈ fp8-row no-diff** (16.6 vs 16.6 short, 16.4 vs 16.3 long). Diffusion mode contributes zero throughput benefit. Compare with fp8: diffusion 43.5/65.3 vs no-diff 14.2/14.0; the diffusion speedup at fp8 is 3.1×/4.7× over its no-diff counterpart.
2. **The smoke test does NOT catch this.** Smoke-test verdict for fp8-row on sm_121 is `GOOD` (1.17x bf16 throughput, kernels firing, output sane). The regression is invisible to a single-prompt HF `generate` call because the diffusion drafter's verify path isn't exercised.

### Mechanism

The Orthrus diffusion drafter was trained to predict tokens that the AR teacher would also predict. In serving, the drafter proposes a block of candidate tokens; the AR teacher verifies them in parallel; accepted prefix is committed, rejected suffix is discarded, drafter is re-invoked. The diffusion speedup comes from accepting multiple tokens per teacher forward.

Quantising the teacher introduces noise that perturbs the teacher's predictions. The drafter, trained against an unquantised teacher, will agree less often. Whether this is fatal depends on how the noise is patterned:

- **`fp8` (per-tensor)**: one symmetric scale per Linear. The perturbation is uniform across rows — every row of every weight matrix is rescaled by the same factor. Drafter and teacher's logits move together; agreement rate stays high enough that the verify path keeps accepting; the diffusion speedup is preserved.
- **`fp8-row` (per-row)**: one scale per output channel. Different rows get different rescalings. The teacher's output logits shift in a structured pattern that the drafter, which has internalised the unquantised teacher's relative logit magnitudes, doesn't track. Verify rejects almost every draft. The pipeline degenerates to single-token AR per step.

The empirical signature of this degenerate state is the diffusion-mode throughput collapsing to the no-diff throughput at the same precision: under fp8-row both modes land at ~16 tok/s, indistinguishable.

The 5-point tool-eval-bench accuracy regression is downstream of the same teacher perturbation: even in AR mode, per-row's structured noise produces different next-token decisions than per-tensor's uniform noise (smoke-test output already showed this: bit-different decoding even on a single-prompt greedy generation). The drafter-rejection cascade doesn't directly cause the accuracy loss, but the underlying teacher-perturbation pattern that causes the rejection cascade also lands tool-call-grammar tokens on different near-tie tips.

### What this means for scheme choice

- **For serving Orthrus diffusion mode**: use `fp8`, not `fp8-row`. The 12% per-tensor smoke-test gap was misleading; per-row's actual cost on serving is 2× slowdown plus 5-point accuracy regression.
- **For serving in AR mode** (no diffusion drafter): per-row vs per-tensor is a much smaller delta and choice is dominated by accuracy preferences. We did not run a full accuracy sweep at fp8-row AR mode because the drafter-rejection finding already invalidates fp8-row for the diffusion serving path that is this project's reason to exist.
- **For non-Orthrus models that don't have a drafter** (vanilla Qwen3 etc): the drafter-rejection effect does not apply. Per-row's cost is just the cuBLAS kernel-tuning gap (~12% on sm_121); per-row's benefit is the higher-fidelity weight scaling. Pick on accuracy preferences.

### Why this is interesting beyond this repo

The drafter-rejection collapse is a concrete example of how a quantisation scheme that looks numerically better at the weight level can be worse at the system level for diffusion-mode inference. The drafter is calibrated against a specific teacher; perturbing the teacher non-uniformly invalidates that calibration. The fix is not "use a different quantisation"; it's "retrain the drafter against the quantised teacher" (quantisation-aware drafter retraining). That work belongs upstream in Orthrus training code, not orthrus-serve.

## Drafter survival is a spectrum (refined hypothesis)

The fp8/fp8-row pair suggested "uniform per-tensor good, structured per-row bad." Adding NVFP4 (added 2026-05-27) to the comparison refines this into a continuous spectrum, parameterised by the granularity scale of weight quantisation relative to the row level.

Measured drafter speedup (no-diff median turn / diffusion median turn) on tool-eval-bench at sm_121, 2026-05-27:

| Scheme | Weight scale granularity | Drafter speedup | tool-eval-bench diff vs nodiff |
|---|---|---:|---|
| fp8 | per-tensor (1 scale / Linear) | **2.29×** (3.9 / 1.7 s) | 74 vs 74 (no change) |
| nvfp4 | per-block (~16 elements / scale) | **1.86×** (2.6 / 1.4 s) | 71 vs 69 (drafter ADDS 2 points) |
| fp8-row | per-row (1 scale / output channel) | **~1.00×** (3.8 / 3.6 s; drafter dead) | 69 vs ? (uninteresting; ≈ AR) |

Refined rule: **the drafter's accept rate is a continuous function of how uniform-at-the-row-level the quantisation pattern is.** Per-tensor is trivially uniform per row → drafter fully tracks. Per-block is locally non-uniform but averages out at the row scale (each row contains ~256 blocks with different scales whose effects average out) → drafter partially tracks. Per-row is rigidly non-uniform at exactly the row scale (each row has ONE scale that affects all its elements together) → drafter cannot track.

The orthogonal dimension is bit width, which barely matters for drafter survival: NVFP4 is 4-bit yet drafter-friendly (1.86×), fp8-row is 8-bit yet drafter-hostile (1.0×). The drafter cares about the *geometry* of the perturbation in weight space, not the *magnitude* of per-element rounding noise. Bit width affects standalone accuracy (per-element rounding noise) but not drafter alignment (relative-magnitude preservation across the weight matrix).

**On observed diff-vs-nodiff deltas within a scheme** (within noise tolerance, not load-bearing): at NVFP4 the tool-eval-bench score differs by 2 points between diff and nodiff (71 vs 69), and safety-critical fail counts differ by 2 (5 vs 3). 2 points = 3 scenarios flipping on a 69-scenario bench, which is well within the near-tie-cascade variance we've established between code paths at the same precision; specific scenarios that differ (TC-43 passing in nodiff_nvfp4 while failing in every other measured arm; TC-41 failing only in diffusion_nvfp4) are consistent with that noise floor. Don't read mechanism into these. The load-bearing observations are: (a) the drafter speedup is too large to be noise (1.86× at nvfp4, 1.0× at fp8-row, 2.29× at fp8), and (b) the 3-point accuracy gap fp8 → nvfp4 (74 → 71) reproduces in BOTH diff (74→71) and nodiff (74→69) directions, indicating it's a real cost of 4-bit precision rather than a code-path artifact.

For Orthrus diffusion-mode serving: `fp8` is the recommended default. `nvfp4` is the recommended alternative when memory pressure or throughput dominates accuracy preferences. `fp8-row` is contraindicated.

For non-Orthrus models that don't have a drafter, only the bit-width / kernel-tuning trade-offs apply — the drafter-spectrum considerations are moot.

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
- **int8 tried and rejected (2026-05-27).** Briefly added `Int8DynamicActivationInt8WeightConfig` as an `int8` scheme to test "uniform per-tensor quantisation preserves the diffusion drafter regardless of float-vs-int format." Two problems made it unusable: (1) torchao 0.15's default int8 path is **per-channel** (per-row weight scales), not per-tensor, so it can't act as the per-tensor control we wanted; (2) the int8 matmul kernel is not wired up on sm_121 in torchao 0.15, falling through to a dequant-then-bf16-matmul slow path ~6.4x slower than bf16 on a 4096x4096 toy linear. Code removed; the negative finding survives as this note. NVFP4 may be a better next candidate for the same "different precision, uniform per-tensor" experiment if its sm_121 kernel state is healthier.

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
