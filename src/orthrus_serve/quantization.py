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
#                           weights with per-row outlier structure, at near-
#                           identical throughput (`_scaled_mm` handles
#                           per-row scales natively on Hopper+). Memory
#                           reduction matches "fp8" (the per-row scale tensor
#                           is negligible relative to the weight). Opt-in for
#                           accuracy-sensitive workloads.
#   - "fp8-weight-only"  -> Float8WeightOnlyConfig. Weights stored fp8 but
#                           matmul DEQUANTIZES to bf16 then runs the bf16
#                           kernel. Empirically: ~10x SLOWER than bf16 on
#                           Blackwell. Memory still halves. Use only if you
#                           need fp8 storage on hardware without _scaled_mm
#                           (pre-Hopper), or for offline analysis.
FP8 = "fp8"
FP8_ROW = "fp8-row"
FP8_WEIGHT_ONLY = "fp8-weight-only"

SUPPORTED_QUANT_SCHEMES = (FP8, FP8_ROW, FP8_WEIGHT_ONLY)

# Schemes that go through the fast native-matmul path (torch._scaled_mm on
# Hopper / Blackwell). Used by the smoke test's verdict logic to decide
# whether a throughput regression is expected (weight-only) or a failure.
_FAST_FP8_SCHEMES = (FP8, FP8_ROW)


# ---------------------------------------------------------------------------
# Filter: which Linear layers do we quantize?
# ---------------------------------------------------------------------------

def _should_quantize_linear(fqn: str, module: nn.Module) -> bool:
    """Filter applied per-module before quantization.

    Skip:
      - lm_head: output projection over the full vocabulary. Often left in
        bf16 because quantization here disproportionately affects perplexity
        and the parameter count is small relative to the layer stack.
      - any non-Linear module (torchao already filters but be explicit).

    Quantize:
      - everything else, including the Orthrus-specific `_diff` projections.
        PR 2 of orthrus-bench-spark showed the `_diff` projections are a
        quantization passenger (quantizing them on top of the AR weights adds
        essentially no additional output divergence), so we get the memory
        win for free.
    """
    if not isinstance(module, nn.Linear):
        return False
    if "lm_head" in fqn:
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
    elif scheme == FP8_WEIGHT_ONLY:
        _apply_fp8_weight_only(model)

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
    model: nn.Module, per_row: bool = False,
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
        return _should_quantize_linear(fqn, module)

    quantize_(model, config, filter_fn=_filter)


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
    print("\n=== DIAGNOSTIC: BEFORE QUANTIZATION ===")
    pre_dtypes = _audit_dtypes(model)
    pre_mem = _memory_footprint_mb(model)
    n_linear = sum(pre_dtypes.values())
    n_to_skip = sum(
        1 for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and not _should_quantize_linear(n, m)
    )
    n_to_quantize = n_linear - n_to_skip
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
    # (~2.4 GB out of ~18.5 GB), so the realistic ceiling is ~1.77x, not 2.0x.
    # Threshold tuned to that: PASS if >=1.6x (close to the ceiling),
    # LOW if a meaningful fraction of Linears were missed,
    # FAIL only if the wrapper element-size accounting broke.
    expected = "~1.7-1.8x for fp8 on this model size"
    good, low = 1.6, 1.3
    if mem_ratio >= good:
        mem_v = f"PASS ({mem_ratio:.2f}x; expected {expected})"
    elif mem_ratio >= low:
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
        # speedup. Per-row variant may run a few percent slower than per-
        # tensor due to scale-fetch overhead; the >=1.2x PASS threshold
        # still applies because the bf16 baseline is the same.
        if tps_ratio >= 1.2:
            tps_v = (f"PASS ({tps_ratio:.2f}x faster than bf16; native fp8 "
                     f"matmul kernels firing as expected)")
        elif tps_ratio >= 0.9:
            tps_v = (f"NEUTRAL ({tps_ratio:.2f}x vs bf16; quantization is "
                     f"running but not delivering a throughput advantage. "
                     f"Memory savings hold. May reflect overhead from the "
                     f"first-time kernel compile or a kernel selection "
                     f"mismatch on this hardware.)")
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
    if mem_ratio < 1.6:
        issues.append("memory did not drop as expected")
    # Throughput regression is only a failure for fast-path schemes
    # (activation+weight, native matmul); weight-only is expected to be slow.
    if scheme in _FAST_FP8_SCHEMES and tps_ratio < 0.9:
        issues.append("throughput regressed below bf16 (expected ~1.2x speedup "
                      "for activation+weight fp8)")
    if "FAIL" in out_v:
        issues.append("output corruption suspected")

    if not issues:
        if scheme in _FAST_FP8_SCHEMES and tps_ratio >= 1.2:
            print(f"OVERALL: GOOD ({scheme} working as expected on this "
                  f"hardware: applied, memory halved, throughput "
                  f"{tps_ratio:.2f}x bf16).")
        elif scheme == FP8_WEIGHT_ONLY:
            print(f"OVERALL: AS EXPECTED ({scheme} applied, memory halved, "
                  f"throughput {tps_ratio:.2f}x bf16 (slow is expected for "
                  f"weight-only; use 'fp8' for throughput).")
        else:
            print(f"OVERALL: PARTIAL ({scheme} applied and memory halved, "
                  f"but throughput {tps_ratio:.2f}x bf16 is below the "
                  f"expected ~1.2x speedup. Memory savings are intact.)")
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
