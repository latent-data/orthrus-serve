"""
In-process fp8 quantization for orthrus-serve.

Entry point for serving Orthrus (and vanilla Qwen3-8B) at fp8 precision.
Applies the quantization to a loaded bf16 model in place, after
from_pretrained() returns and before model.eval(). No separate saved
quantized checkpoint is required.

Wired in via:
  - settings.py: Settings.quant field, read from ORTHRUS_QUANT env var
  - model.py: apply_quantization(model, settings.quant) after from_pretrained
  - Dockerfile: torchao installed alongside transformers/accelerate
  - run.sh: --quant <scheme> flag sets ORTHRUS_QUANT for the container

Why in-process and not a saved fp8 checkpoint:
  - Fp8 weight-only is calibration-free. The conversion is a deterministic
    per-tensor scale + cast, identical every time. No reason to materialize a
    saved artifact for this.
  - Keeps the model-loading code path identical to the bf16 case (one call to
    from_pretrained); only an extra in-place transform is layered on top.
  - Trivial to swap quant schemes (fp8 weight-only -> fp8 activation+weight ->
    nvfp4 -> ...) without re-saving 8B-parameter checkpoints each time.
  - When/if we move to calibrated formats (AWQ, GPTQ) where saving is the
    normal workflow, we'd graduate to a separate orthrus-quant project.

Hardware notes:
  - On Blackwell (sm_121 / DGX Spark) torch supports fp8 weight storage and
    native fp8 matmul via torch._scaled_mm; torchao routes nn.Linear forward
    through that path automatically when the weight is fp8.
  - On pre-Blackwell GPUs torchao falls back to bf16-matmul with on-the-fly
    upcast from fp8 storage; you still get the ~2x memory savings, but not
    the throughput improvement.

Tested-by: the `python -m orthrus_serve.quantization --smoke` CLI in this
file (loads the model, applies quant, generates 32 tokens, prints memory
footprint and throughput before and after, emits a structured verdict).
Run inside the NGC container before pointing tool-eval-bench at the
quantised endpoint.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supported schemes
# ---------------------------------------------------------------------------
#
# Naming reflects what's actually fast on Blackwell, NOT what was historically
# called "the safe default" in torchao docs:
#   - "fp8"              -> Float8DynamicActivationFloat8WeightConfig with
#                           per-tensor symmetric weight scaling and per-token
#                           dynamic activation scaling. Uses torch._scaled_mm
#                           on Hopper / Blackwell for native fp8 matmul.
#                           Empirically: ~0.75x bf16 per-forward on Blackwell
#                           sm_121 (i.e. 25% faster than bf16). Same ~2x
#                           memory savings. No calibration. Recommended
#                           default for throughput.
#   - "fp8-row"          -> Same as "fp8" but with PerRow() granularity (per
#                           output-channel weight scales, per-token activation
#                           scales). Slightly higher numerical fidelity for
#                           weights with per-row outlier structure. Memory
#                           reduction matches "fp8" (the per-row scale tensor
#                           is negligible relative to the weight). Throughput
#                           measured on Blackwell sm_121 short-prompt smoke
#                           test (2026-05-27): 24.8 tok/s vs fp8's 28.0
#                           tok/s vs bf16's 21.1 tok/s -- per-row is 1.17x
#                           bf16, 0.88x fp8. The 12% gap vs per-tensor
#                           reflects cuBLAS dispatching a less-tuned per-row
#                           kernel on sm_121 (Hopper has better-tuned
#                           kernels; gap may close as Blackwell tuning
#                           lands). Opt-in for accuracy-sensitive workloads.
#   - "fp8-weight-only"  -> Float8WeightOnlyConfig. Weights stored fp8 but
#                           matmul DEQUANTIZES to bf16 then runs the bf16
#                           kernel. Empirically: ~10x SLOWER than bf16 on
#                           Blackwell. Memory still halves. Use only if you
#                           need fp8 storage on hardware without _scaled_mm
#                           (pre-Hopper), or for offline analysis.
#   - "nvfp4"            -> NVFP4InferenceConfig from torchao.prototype.
#                           mx_formats. 4-bit float weights with per-block
#                           (~16-element) fp8 scales, dynamic per-tensor
#                           activation scaling, triton-fp4 native kernel.
#                           Toy probe (4096x4096) measured 2.58x bf16,
#                           ~3.55x weight memory reduction. Inherently
#                           per-block, not per-tensor (4 bits is too few
#                           to make a single per-Linear scale work);
#                           prototype-namespace in torchao 0.15.
#   - "fp8-row-teacher-only" -> Same as "fp8-row" but applied ONLY to the
#                           teacher (AR + shared) weights; the Orthrus
#                           `_diff` drafter projections stay bf16. Mechanism
#                           probe: per-row fp8 breaks the diffusion drafter
#                           when applied everywhere ("fp8-row"); this isolates
#                           whether the breakage is driven by perturbing the
#                           TEACHER (whose argmax the drafter was trained to
#                           match) or the drafter's own weights. Accuracy is
#                           governed by the teacher, so this should match
#                           "fp8-row" on tool-eval-bench (~69); the question
#                           is whether the drafter accept rate recovers off
#                           the AR floor. Production-relevant: teacher is ~84%
#                           of params, so most of the memory win survives.
#   - "fp8-row-drafter-only" -> Same as "fp8-row" but applied ONLY to the
#                           `_diff` drafter projections; the teacher stays
#                           bf16. Pure mechanism control (NOT a useful
#                           production config: `_diff` is only ~16% of params,
#                           so near-zero memory win, and the teacher stays at
#                           full bf16 cost). Accuracy should return to bf16
#                           (~72) because the unquantised teacher emits the
#                           verified tokens; the question is whether the
#                           drafter tolerates per-row noise in its OWN
#                           proposals. NOTE: the AR-mode smoke test does not
#                           exercise `_diff` at all (those projections are
#                           only used in diffusion mode), so smoke throughput
#                           / output for this scheme reflect the unquantised
#                           teacher path; the real signal is the diffusion-
#                           mode HTTP throughput bench.
FP8 = "fp8"
FP8_ROW = "fp8-row"
FP8_ROW_TEACHER = "fp8-row-teacher-only"
FP8_ROW_DRAFTER = "fp8-row-drafter-only"
FP8_WEIGHT_ONLY = "fp8-weight-only"
NVFP4 = "nvfp4"

SUPPORTED_QUANT_SCHEMES = (
    FP8, FP8_ROW, FP8_ROW_TEACHER, FP8_ROW_DRAFTER, FP8_WEIGHT_ONLY, NVFP4,
)

# Schemes that go through a fast native-matmul path (torch._scaled_mm for
# fp8 on Hopper/Blackwell; triton-fp4 kernel for nvfp4). Used by the smoke
# test's verdict logic to decide whether a throughput regression is
# expected (weight-only) or a failure.
_FAST_FP8_SCHEMES = (FP8, FP8_ROW, FP8_ROW_TEACHER, FP8_ROW_DRAFTER, NVFP4)

# Per-scheme expected speedup over bf16 (PASS threshold for the smoke
# verdict). FP8 (per-tensor) hits ~1.33x on sm_121 short-prompt; FP8_ROW
# (per-row) lands at ~1.17x because cuBLAS's per-row `_scaled_mm` kernel
# is less tuned. NVFP4 toy probe measured 2.58x bf16 on a 4096x4096
# linear but the full-model HF generate amortises to 1.24x (smoke 2026-
# 05-27); long-prompt diffusion-mode HTTP serving recovers the win,
# landing at 1.74x bf16. Smoke threshold reflects the smoke ceiling.

#   - fp8-row-teacher-only: same per-row kernel as fp8-row on the 84% of
#     Linears that are the teacher; smoke (AR mode) should look like fp8-row.
#   - fp8-row-drafter-only: AR-mode smoke runs through the unquantised
#     teacher (the `_diff` projections aren't on the AR path), so throughput
#     is ~bf16 (1.0x) and the threshold is set accordingly — this scheme's
#     real signal is the diffusion-mode HTTP bench, not the smoke test.
_FAST_FP8_PASS_THRESHOLDS = {
    FP8: 1.20, FP8_ROW: 1.10, FP8_ROW_TEACHER: 1.05, FP8_ROW_DRAFTER: 0.90,
    NVFP4: 1.20,
}

# Per-scheme expected memory-reduction ceiling for the smoke verdict.
# fp8 family halves Linear bytes (~1.77x model-wide because embedding +
# lm_head + norms stay bf16). NVFP4 quarters them plus a small per-block
# scale overhead (~16-element blocks with fp8 scales) yielding a ~3.55x
# weight-byte reduction or ~2.5x model-wide ceiling on an 8B Qwen3.
# fp8-row-teacher-only leaves the ~16%-of-params `_diff` projections at bf16,
# so its model-wide reduction is lower than full fp8 (~1.4x vs ~1.77x).
# fp8-row-drafter-only quantises only the `_diff` projections (~16% of params),
# so the model-wide reduction is small (~1.05x).
_QUANT_MEMORY_PASS_THRESHOLDS = {
    FP8: 1.60, FP8_ROW: 1.60, FP8_ROW_TEACHER: 1.40, FP8_ROW_DRAFTER: 1.03,
    FP8_WEIGHT_ONLY: 1.60, NVFP4: 2.30,
}

# Human-readable expected memory-reduction string per scheme for the smoke
# verdict (the bare threshold above is the machine check).
_QUANT_MEMORY_EXPECTED = {
    NVFP4: "~2.4-2.6x (4-bit weights)",
    FP8_ROW_TEACHER: "~1.4x (teacher only; `_diff` stays bf16)",
    FP8_ROW_DRAFTER: "~1.05x (drafter only; ~16% of params quantised)",
}


# ---------------------------------------------------------------------------
# Filter: which Linear layers do we quantize?
# ---------------------------------------------------------------------------

# A parameter / module belongs to the Orthrus diffusion drafter iff its
# qualified name contains "_diff" (the `*_proj_diff` projections added on top
# of stock Qwen3). Everything else (the AR + shared backbone) is the teacher.
# Matches the split used in orthrus-bench-spark/quant_benchmark.py.
_DIFF_MARKER = "_diff"


def _target_for_scheme(scheme: Optional[str]) -> str:
    """Which side of the model a scheme quantises: 'all', 'teacher', 'drafter'."""
    if scheme == FP8_ROW_TEACHER:
        return "teacher"
    if scheme == FP8_ROW_DRAFTER:
        return "drafter"
    return "all"


def _should_quantize_linear(
    fqn: str, module: nn.Module, target: str = "all",
) -> bool:
    """Filter applied per-module before quantization.

    Skip:
      - lm_head: output projection over the full vocabulary. Often left in
        bf16 because quantization here disproportionately affects perplexity
        and the parameter count is small relative to the layer stack.
      - any non-Linear module (torchao already filters but be explicit).
      - layers on the wrong side of a teacher/drafter `target` split (see
        below).

    `target` selects which weights to quantize:
      - "all" (default): everything eligible, including the Orthrus-specific
        `_diff` projections. PR 2 of orthrus-bench-spark showed the `_diff`
        projections are a quantization passenger (quantizing them on top of
        the AR weights adds essentially no additional output divergence at
        per-tensor granularity), so we get the memory win for free.
      - "teacher": only the AR + shared backbone (everything NOT containing
        `_diff`). Used by fp8-row-teacher-only to test whether the per-row
        drafter breakage is driven by teacher-side perturbation.
      - "drafter": only the `_diff` projections. Used by fp8-row-drafter-only
        as a mechanism control.
    """
    if not isinstance(module, nn.Linear):
        return False
    if "lm_head" in fqn:
        return False
    is_diff = _DIFF_MARKER in fqn
    if target == "teacher" and is_diff:
        return False
    if target == "drafter" and not is_diff:
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_quantization(
    model: nn.Module,
    scheme: Optional[str],
) -> nn.Module:
    """Apply the requested quantization scheme to `model` in place.

    Args:
        model: a loaded HF model (bf16 weights expected). Modified in place.
        scheme: one of SUPPORTED_QUANT_SCHEMES, or None / empty string for
            no-op.

    Returns the same model object for fluent chaining.
    """
    if not scheme:
        logger.info("Quantization scheme = none; model stays at bf16")
        return model

    if scheme not in SUPPORTED_QUANT_SCHEMES:
        raise ValueError(
            f"Unknown quantization scheme {scheme!r}; "
            f"supported: {SUPPORTED_QUANT_SCHEMES}"
        )

    _check_runtime()

    pre = _memory_footprint_mb(model)
    logger.info("Applying quantization scheme %r (model footprint %.0f MB)",
                scheme, pre)

    if scheme == FP8:
        _apply_fp8_dynamic_activation_weight(model, per_row=False)
    elif scheme == FP8_ROW:
        _apply_fp8_dynamic_activation_weight(model, per_row=True)
    elif scheme == FP8_ROW_TEACHER:
        _apply_fp8_dynamic_activation_weight(model, per_row=True, target="teacher")
    elif scheme == FP8_ROW_DRAFTER:
        _apply_fp8_dynamic_activation_weight(model, per_row=True, target="drafter")
    elif scheme == FP8_WEIGHT_ONLY:
        _apply_fp8_weight_only(model)
    elif scheme == NVFP4:
        _apply_nvfp4(model)

    post = _memory_footprint_mb(model)
    logger.info("Quantization %r complete: footprint %.0f MB -> %.0f MB "
                "(%.1fx reduction)", scheme, pre, post,
                pre / post if post else float("nan"))
    return model


# ---------------------------------------------------------------------------
# Scheme implementations
# ---------------------------------------------------------------------------

def _apply_fp8_weight_only(model: nn.Module) -> None:
    """Cast all eligible Linear weights to fp8 e4m3fn (STORAGE ONLY).

    Despite the misleading name, this does NOT use fp8 matmul kernels under
    torchao 0.15: each forward dequantises the weight to bf16 and runs the
    bf16 matmul kernel. Memory drops ~2x, throughput drops ~10x on Blackwell
    (measured per-forward on a 3-layer 4096x4096 toy model). Use this only
    when storage is the goal and throughput doesn't matter, e.g. offline
    analysis, or on hardware without _scaled_mm support where weight-only is
    the only available fp8 path.

    For Blackwell-class inference, use FP8 (the activation+weight variant)
    instead.

    No calibration data needed. Per-tensor symmetric scale derived from each
    weight's absmax at quant time.
    """
    try:
        from torchao.quantization import quantize_, Float8WeightOnlyConfig
    except ImportError as e:
        raise RuntimeError(
            "torchao import failed. The NGC container "
            "(nvcr.io/nvidia/pytorch:25.12-py3) ships torchao 0.15+, which "
            "exports Float8WeightOnlyConfig. If you see this error, either "
            "the container's torchao was uninstalled / overridden (check pip "
            "install steps in the Dockerfile / run.sh) or you're on a "
            "non-NGC environment. Do NOT pip install a pypi version on top "
            "of NGC: that will either drop NGC-side patches or pull a fresh "
            "torch wheel that clobbers the sm_121 build."
        ) from e

    def _filter(module: nn.Module, fqn: str) -> bool:
        return _should_quantize_linear(fqn, module)

    quantize_(model, Float8WeightOnlyConfig(), filter_fn=_filter)


def _apply_fp8_dynamic_activation_weight(
    model: nn.Module, per_row: bool = False, target: str = "all",
) -> None:
    """Both weights and activations in fp8 (dynamic per-token activation
    scaling), with native fp8 matmul via torch._scaled_mm.

    This is the recommended path on Blackwell-class hardware: empirically
    ~25% faster than bf16 per-forward and ~2x smaller in memory on a 3-layer
    4096x4096 toy model, with no calibration data required (per-token
    activation scaling is dynamic). Falls back to dequant-then-bf16-matmul
    on pre-Hopper hardware without _scaled_mm support.

    Granularity:
      - per_row=False (default): per-tensor symmetric scale for weights
        (one scalar per Linear), per-token dynamic scale for activations.
        Matches torchao's default and the historical "fp8" entrypoint.
      - per_row=True: per-row (a.k.a. per-output-channel) scale for weights,
        per-token dynamic scale for activations. Higher numerical fidelity
        when weight rows have heterogeneous magnitudes; throughput is
        essentially the same on Hopper+ because _scaled_mm dispatches a
        per-row-scale kernel that fuses the rescale into the matmul.

    Slightly more aggressive than weight-only because activations are also
    quantised; in principle this could affect accuracy on outlier-heavy
    activations, but Qwen3-class models are well-behaved here. Verify with
    tool-eval-bench before deployment.
    """
    try:
        from torchao.quantization import (
            quantize_,
            Float8DynamicActivationFloat8WeightConfig,
        )
    except ImportError as e:
        raise RuntimeError(
            "torchao import failed; see fp8-weight-only error for details."
        ) from e

    if per_row:
        try:
            from torchao.quantization import PerRow
        except ImportError:
            # Older torchao layouts expose granularity types under a submodule.
            from torchao.quantization.granularity import PerRow  # type: ignore
        config = Float8DynamicActivationFloat8WeightConfig(
            granularity=PerRow(),
        )
    else:
        config = Float8DynamicActivationFloat8WeightConfig()

    def _filter(module: nn.Module, fqn: str) -> bool:
        return _should_quantize_linear(fqn, module, target=target)

    quantize_(model, config, filter_fn=_filter)
    if target != "all":
        n = sum(
            1 for fqn, m in model.named_modules()
            if isinstance(m, nn.Linear) and _should_quantize_linear(fqn, m, target)
        )
        logger.info("fp8 per-row applied to %s side only (%d Linear layers)",
                    target, n)


def _apply_nvfp4(model: nn.Module) -> None:
    """Weights in NVFP4 (4-bit float, per-block fp8 scales),
    dynamic per-tensor activation scaling, triton-fp4 native matmul.

    NVFP4 (a.k.a. micro-scaling fp4) is structurally per-block: weights
    are packed 2 fp4 values per byte and each ~16-element block gets its
    own fp8 scale. The "per-tensor" piece in the config name refers to
    the activation scaling, not the weight scaling -- you can't usefully
    represent a transformer Linear's weight matrix with a single fp4
    scale (4 bits is too few dynamic range).

    Empirical state on Blackwell sm_121 (2026-05-27):
      - Toy probe (4096x4096 linear): 2.58x bf16, ~3.55x weight bytes.
      - 8B smoke test (HF generate, short prompt): 1.24x bf16, 2.89x
        model-wide memory reduction (18.5 GB -> 6.4 GB).
      - Diffusion-mode HTTP serving, long prompt: 88.9 tok/s vs fp8's
        65.3 tok/s vs bf16's 51.1 tok/s, i.e. 1.36x fp8 / 1.74x bf16.

    The diffusion drafter SURVIVES NVFP4 (unlike fp8-row, which breaks
    the drafter through rigid per-row perturbation). The mechanism is
    that block-wise scales vary at high frequency WITHIN a row (~16
    weights per scale), so the row-level effect averages out to look
    approximately uniform -- which is what drafter↔teacher alignment
    needs. Per-row's rigid one-scale-per-row perturbation is what kills
    the drafter, not the granularity per se. NVFP4 is finer-grained
    than per-row and yet works, because the granularity is below the
    row level rather than at it. (4-bit precision turns out to matter
    much less than the structure of the perturbation.)

    For accuracy: tool-eval-bench result is in
    `benchmarks/results/tool-eval-bench/` and the README has the
    per-scheme score table.
    """
    try:
        from torchao.quantization import quantize_
        from torchao.prototype.mx_formats import NVFP4InferenceConfig
    except ImportError as e:
        raise RuntimeError(
            "NVFP4 import failed; torchao.prototype.mx_formats not "
            "available in this build. Requires torchao 0.15+ as shipped "
            "in nvcr.io/nvidia/pytorch:25.12-py3."
        ) from e

    def _filter(module: nn.Module, fqn: str) -> bool:
        return _should_quantize_linear(fqn, module)

    quantize_(model, NVFP4InferenceConfig(), filter_fn=_filter)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_runtime() -> None:
    """Warn (don't fail) if the hardware doesn't natively support fp8 matmul.

    Quantization still produces the memory savings on pre-Blackwell hardware
    via upcast-on-matmul; only the throughput benefit is lost.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("fp8 quantization requires CUDA")
    cap = torch.cuda.get_device_capability(0)
    if cap < (9, 0):
        logger.warning(
            "GPU compute capability %s is below sm_90 (Hopper); fp8 matmul "
            "kernels will not fire. Memory savings still apply.", cap,
        )
    else:
        logger.info("GPU compute capability %s supports fp8 matmul", cap)


def _memory_footprint_mb(model: nn.Module) -> float:
    """Approximate model parameter memory in MB.

    Handles torchao tensor subclasses (Float8Tensor, etc.) by reading the
    underlying storage tensor's element_size rather than the wrapper's
    advertised dtype. Float8Tensor reports p.dtype=bfloat16 (the dequantised
    "logical" dtype) but the real storage is in p.qdata.dtype with half the
    element size, so the naive Parameter-iteration overestimates by 2x for
    quantised models.

    Does not include activations or KV cache.
    """
    total_bytes = 0
    for p in model.parameters():
        if hasattr(p, "qdata"):
            qd = p.qdata
            total_bytes += qd.numel() * qd.element_size()
        else:
            total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# Smoke-test CLI: python -m orthrus_serve.quantization --smoke [--scheme fp8]
# ---------------------------------------------------------------------------

def _smoke_test(scheme: str = FP8) -> None:
    """Comprehensive diagnostic smoke test for quantization.

    Loads the model, runs a baseline bf16 generation with diagnostics
    (Linear-weight dtype audit, memory footprint, throughput), applies the
    requested quant scheme, re-runs the same diagnostics, then prints a
    verdict block assessing whether each layer behaved as expected.

    Catches the common silent-failure modes on new hardware:
      - quantization didn't apply (filter bug, torchao skipped layers)
      - memory didn't drop (some layers stayed bf16)
      - throughput didn't improve (native kernels not firing; dequant
        fallback path; memory savings hold but no speedup)
      - output is corrupted (matmul producing garbage)

    Run inside the NGC container:

        python -m orthrus_serve.quantization --smoke --scheme fp8
    """
    import time
    from collections import Counter
    from .model import load_model_and_tokenizer

    PROMPT = "Hello, what is 2+2?"
    MAX_NEW_TOKENS = 32
    WARMUP_TOKENS = 8

    def _audit_dtypes(model: nn.Module) -> Counter:
        """Count Linear weight dtypes, distinguishing torchao wrappers from
        plain Tensors. A weight wrapped by torchao (Float8Tensor etc.) reports
        its 'logical' dtype (often bfloat16 for transparency) via .dtype, but
        the real storage dtype is in .qdata.dtype.
        """
        c: Counter = Counter()
        for _, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                w = mod.weight
                if hasattr(w, "qdata"):
                    c[f"{type(w).__name__}({w.qdata.dtype})"] += 1
                else:
                    c[str(w.dtype)] += 1
        return c

    def _gen(model, input_ids, max_new):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids, max_new_tokens=max_new, do_sample=False,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        new_tokens = out[0, input_ids.shape[-1]:]
        return new_tokens, elapsed

    print("[smoke] loading model ...")
    t0 = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer()
    print(f"[smoke] loaded in {time.perf_counter() - t0:.1f}s")

    messages = [{"role": "user", "content": PROMPT}]
    # apply_chat_template with return_tensors="pt" returns a BatchEncoding
    # dict, not a tensor; extract .input_ids to get the actual tensor.
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
    ).input_ids.to(model.device)

    # -------- before quant --------
    target = _target_for_scheme(scheme)
    print("\n=== DIAGNOSTIC: BEFORE QUANTIZATION ===")
    pre_dtypes = _audit_dtypes(model)
    pre_mem = _memory_footprint_mb(model)
    n_linear = sum(pre_dtypes.values())
    n_to_skip = sum(
        1 for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and not _should_quantize_linear(n, m, target)
    )
    n_to_quantize = n_linear - n_to_skip
    if target != "all":
        print(f"  Target side:           {target!r} "
              f"(only the {'AR/shared teacher' if target == 'teacher' else 'diffusion `_diff` drafter'} "
              f"Linears are quantised)")
    print(f"  Linear modules:        {n_linear} total")
    print(f"  Linear dtypes:         {dict(pre_dtypes)}")
    print(f"  Memory footprint:      {pre_mem:.0f} MB")
    print(f"  Expected to quantize:  {n_to_quantize} "
          f"({n_to_skip} skipped by filter, e.g. lm_head)")

    print(f"\n[bf16] warmup ({WARMUP_TOKENS} tok; first call compiles kernels) ...")
    _gen(model, input_ids, WARMUP_TOKENS)
    print(f"[bf16] measuring ({MAX_NEW_TOKENS} tok) ...")
    bf16_tokens, bf16_elapsed = _gen(model, input_ids, MAX_NEW_TOKENS)
    bf16_tps = bf16_tokens.numel() / bf16_elapsed
    bf16_text = tokenizer.decode(bf16_tokens, skip_special_tokens=True)
    print(f"  Elapsed:               {bf16_elapsed:.2f}s")
    print(f"  Throughput:            {bf16_tps:.1f} tok/s")
    print(f"  Output: {bf16_text!r}")

    # -------- apply quant --------
    print(f"\n=== APPLYING QUANT: scheme={scheme!r} ===")
    apply_quantization(model, scheme)

    # -------- after quant --------
    print("\n=== DIAGNOSTIC: AFTER QUANTIZATION ===")
    post_dtypes = _audit_dtypes(model)
    post_mem = _memory_footprint_mb(model)
    print(f"  Linear dtypes:         {dict(post_dtypes)}")
    mem_ratio = pre_mem / post_mem if post_mem else float("nan")
    print(f"  Memory footprint:      {post_mem:.0f} MB")
    print(f"  Memory reduction:      {mem_ratio:.2f}x")
    # Granularity introspection: prove per-row vs per-tensor actually fired.
    # The wrapper (Float8Tensor) is the same class either way; the discriminator
    # is the SHAPE of the .scale tensor on the wrapped weight:
    #   - PerTensor() -> scale is a scalar (numel == 1)
    #   - PerRow()    -> scale has out_features entries (numel == out_features)
    # If scheme==fp8-row but the inspected scale is scalar, torchao silently
    # fell back to per-tensor and the "fp8-row" run is actually a "fp8" run.
    sample_scales = []
    for fqn, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and hasattr(mod.weight, "qdata"):
            scale = getattr(mod.weight, "scale", None)
            if scale is not None:
                sample_scales.append((fqn, tuple(scale.shape), scale.numel(),
                                      mod.weight.shape))
            if len(sample_scales) >= 3:
                break
    if sample_scales:
        print(f"  Weight scale shapes (first 3 quantised Linears):")
        for fqn, shp, n, wshp in sample_scales:
            wshp_t = tuple(wshp)
            kind = "PER-TENSOR" if n == 1 else (
                f"PER-ROW (={wshp_t[0]} out_features)" if n == wshp_t[0]
                else f"OTHER (numel={n}, weight={wshp_t})"
            )
            print(f"    {fqn}: scale shape {shp}, weight {wshp_t} -> {kind}")
    else:
        print(f"  Weight scale shapes:   (no .scale attribute found on "
              f"wrapped weights; torchao internal layout differs)")

    print(f"\n[{scheme}] warmup ({WARMUP_TOKENS} tok) ...")
    _gen(model, input_ids, WARMUP_TOKENS)
    print(f"[{scheme}] measuring ({MAX_NEW_TOKENS} tok) ...")
    q_tokens, q_elapsed = _gen(model, input_ids, MAX_NEW_TOKENS)
    q_tps = q_tokens.numel() / q_elapsed
    q_text = tokenizer.decode(q_tokens, skip_special_tokens=True)
    print(f"  Elapsed:               {q_elapsed:.2f}s")
    print(f"  Throughput:            {q_tps:.1f} tok/s")
    print(f"  Output: {q_text!r}")

    # -------- verdict --------
    print("\n=== VERDICT ===")

    # 1. Did quantization apply to the expected layers?
    # Torchao wraps weights in tensor subclasses (Float8Tensor etc.); a
    # wrapped weight is recognised by the wrapper class name appearing in
    # the audit's dtype key, or by the presence of a qdata attribute on the
    # module's weight.
    wrapped_count = 0
    for _, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and hasattr(mod.weight, "qdata"):
            wrapped_count += 1
    if n_to_quantize == 0:
        applied_pct = 0.0
    else:
        applied_pct = 100 * wrapped_count / n_to_quantize
    if applied_pct >= 95:
        applied_v = (f"PASS ({wrapped_count}/{n_to_quantize} target Linears "
                     f"wrapped in fp8 tensor subclass)")
    elif applied_pct >= 50:
        applied_v = (f"PARTIAL ({wrapped_count}/{n_to_quantize} target "
                     f"Linears wrapped; investigate filter / silent skips)")
    else:
        applied_v = (f"FAIL ({wrapped_count}/{n_to_quantize} target Linears "
                     f"wrapped; quantization mostly did not apply)")
    print(f"  Quantization applied:  {applied_v}")

    # 2. Memory drop in the expected range?
    # Naive "2x for fp8" overestimates because the embedding, lm_head, layer
    # norms, q_norm/k_norm etc. are NOT nn.Linear modules and stay bf16.
    # On an 8B Qwen3-class model that unquantised tail is ~13% of params
    # (~2.4 GB out of ~18.5 GB), so the realistic ceiling is ~1.77x for
    # 8-bit schemes and ~2.5x for 4-bit (NVFP4) schemes. Per-scheme PASS
    # threshold (see _QUANT_MEMORY_PASS_THRESHOLDS) reflects that ceiling.
    mem_threshold = _QUANT_MEMORY_PASS_THRESHOLDS.get(scheme, 1.60)
    expected = _QUANT_MEMORY_EXPECTED.get(
        scheme, f"~1.7-1.8x for {scheme} on this model size")
    if mem_ratio >= mem_threshold:
        mem_v = f"PASS ({mem_ratio:.2f}x; expected {expected})"
    elif mem_ratio >= mem_threshold * 0.8:
        mem_v = (f"LOW ({mem_ratio:.2f}x; expected {expected}; some Linear "
                 f"modules may have stayed bf16; check the filter)")
    else:
        mem_v = (f"FAIL ({mem_ratio:.2f}x; quantization mostly did not "
                 f"shrink memory; check that _memory_footprint_mb is reading "
                 f"qdata storage rather than wrapper element_size)")
    print(f"  Memory reduction:      {mem_v}")

    # 3. Throughput: which fp8 path actually fired?
    # The expectation differs by scheme:
    #   - FP8 / FP8_ROW (activation+weight, native matmul) should be faster
    #     than bf16 on Blackwell (~0.75x bf16 per-forward, ~25% speedup)
    #     because _scaled_mm fires. FP8_ROW pays a small extra cost for
    #     per-row scale application, but the fused kernel keeps it within
    #     ~5% of FP8 in practice.
    #   - FP8_WEIGHT_ONLY is dequant-then-bf16-matmul and is much slower
    #     than bf16 (~10x slower on toy models). Slow is the EXPECTED state
    #     for that scheme, not a failure.
    tps_ratio = q_tps / bf16_tps if bf16_tps else float("nan")
    if scheme == FP8_WEIGHT_ONLY:
        # Weight-only: slow is expected (memory savings only).
        if tps_ratio < 0.5:
            tps_v = (f"EXPECTED ({tps_ratio:.2f}x vs bf16; weight-only "
                     f"dequantises on every forward, much slower than bf16. "
                     f"Use the 'fp8' scheme instead for throughput.)")
        else:
            tps_v = (f"UNUSUAL ({tps_ratio:.2f}x vs bf16; weight-only is "
                     f"usually >5x slower on Blackwell. Possible fast-path "
                     f"hit; verify before relying on it.)")
    else:
        # FP8 / FP8_ROW (activation+weight): native fp8 matmul should give
        # speedup. Per-scheme PASS threshold (see _FAST_FP8_PASS_THRESHOLDS)
        # because per-row's measured ceiling on sm_121 (1.17x bf16) is
        # below per-tensor's (1.33x bf16).
        pass_threshold = _FAST_FP8_PASS_THRESHOLDS.get(scheme, 1.20)
        if tps_ratio >= pass_threshold:
            tps_v = (f"PASS ({tps_ratio:.2f}x faster than bf16; native fp8 "
                     f"matmul kernels firing as expected; threshold "
                     f"{pass_threshold:.2f}x)")
        elif tps_ratio >= 0.9:
            tps_v = (f"NEUTRAL ({tps_ratio:.2f}x vs bf16; quantization is "
                     f"running but below the {pass_threshold:.2f}x PASS "
                     f"threshold for {scheme}. Memory savings hold. May "
                     f"reflect overhead from the first-time kernel compile "
                     f"or a kernel selection mismatch on this hardware.)")
        else:
            tps_v = (f"REGRESSION ({tps_ratio:.2f}x vs bf16; slower than "
                     f"bf16. Native fp8 matmul probably did not fire; "
                     f"investigate kernel logs.)")
    print(f"  Throughput vs bf16:    {q_tps:.1f} vs {bf16_tps:.1f} tok/s; "
          f"{tps_v}")

    # 4. Output sanity (cheap heuristics; eyeball the snippets above too)
    unique = len(set(q_tokens.tolist()))
    if not q_text.strip():
        out_v = "FAIL (empty output)"
    elif unique < 3:
        out_v = (f"FAIL (output has only {unique} unique tokens; possible "
                 f"matmul corruption)")
    elif q_text == bf16_text:
        out_v = ("PASS (output bit-identical to bf16; quantization preserved "
                 "greedy choices on this prompt)")
    else:
        out_v = ("PASS (sensible decoded text; differs from bf16 as expected "
                 "for a perturbed model)")
    print(f"  Output sanity:         {out_v}")

    # Overall
    print()
    issues = []
    if applied_pct < 95:
        issues.append("quantization did not apply to expected layers")
    if mem_ratio < mem_threshold:
        issues.append(f"memory did not drop as expected (got {mem_ratio:.2f}x, "
                      f"expected >={mem_threshold:.2f}x for {scheme})")
    # Throughput regression is only a failure for fast-path schemes
    # (activation+weight, native matmul); weight-only is expected to be slow.
    if scheme in _FAST_FP8_SCHEMES and tps_ratio < 0.9:
        issues.append(f"throughput regressed below bf16 (expected "
                      f">={_FAST_FP8_PASS_THRESHOLDS.get(scheme, 1.20):.2f}x "
                      f"for {scheme})")
    if "FAIL" in out_v:
        issues.append("output corruption suspected")

    pass_threshold = _FAST_FP8_PASS_THRESHOLDS.get(scheme, 1.20)
    if not issues:
        if scheme in _FAST_FP8_SCHEMES and tps_ratio >= pass_threshold:
            print(f"OVERALL: GOOD ({scheme} working as expected on this "
                  f"hardware: applied, memory halved, throughput "
                  f"{tps_ratio:.2f}x bf16, threshold {pass_threshold:.2f}x).")
        elif scheme == FP8_WEIGHT_ONLY:
            print(f"OVERALL: AS EXPECTED ({scheme} applied, memory halved, "
                  f"throughput {tps_ratio:.2f}x bf16 (slow is expected for "
                  f"weight-only; use 'fp8' for throughput).")
        else:
            print(f"OVERALL: PARTIAL ({scheme} applied and memory halved, "
                  f"but throughput {tps_ratio:.2f}x bf16 is below the "
                  f"{pass_threshold:.2f}x threshold for {scheme}. Memory "
                  f"savings are intact.)")
    else:
        print(f"OVERALL: ISSUES: {'; '.join(issues)}. See individual "
              f"verdicts above. Common causes: torchao version mismatch with "
              f"NGC torch, sm_121 detection problems, or filter bugs.")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="Load model, apply quant, run diagnostic verdict, exit.")
    p.add_argument("--scheme", default=FP8,
                   choices=SUPPORTED_QUANT_SCHEMES,
                   help=f"Quant scheme (default {FP8!r})")
    args = p.parse_args()
    if args.smoke:
        _smoke_test(args.scheme)
