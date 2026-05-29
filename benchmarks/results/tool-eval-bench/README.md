# tool-eval-bench run files

Full per-scenario / per-category outputs from the tool-eval-bench sweeps cited in `orthrus-serve/RESEARCH_LOG.md`. Filenames preserve the original tool-eval-bench `{ISO timestamp}_{hash}.md` convention so they can be cross-referenced against the spark-recipes archive at `~/spark-recipes/runs/2026/05/`.

## Index

| File | Date | Configuration | Final Score | Median Turn | Wall-clock |
|---|---|---|---|---|---|
| `2026-05-25T10-07-22Z_93a80c.md` | May 25 | Orthrus diffusion (bf16) | 72 | 2.0 s | 359.5 s |
| `2026-05-25T10-39-21Z_93a80c.md` | May 25 | Orthrus no-diff (bf16) | 70 | 4.4 s | 919.3 s |
| `2026-05-25T11-27-41Z_9cd212.md` | May 25 | Qwen3-8B base (bf16) | 70 | 4.4 s | 921.0 s |
| `2026-05-27T11-01-43Z_a24531.md` | May 27 | Orthrus diffusion (fp8) | 74 | 1.7 s | 331.0 s |
| `2026-05-27T11-21-32Z_f865fa.md` | May 27 | Qwen3-8B base (fp8) | 74 | 3.8 s | 801.7 s |
| `2026-05-27T11-48-43Z_a24531.md` | May 27 | Orthrus no-diff (fp8) | 74 | 3.9 s | 810.2 s |
| `2026-05-28T09-26-58Z_cbc6af.md` | May 28 | Orthrus diffusion (fp8-row) | 70 | 1.9 s | 326.0 s |
| `2026-05-27T15-48-11Z_9716ca.md` | May 27 | Orthrus diffusion (nvfp4) | 71 | 1.4 s | 253.7 s |
| `2026-05-27T15-59-16Z_9716ca.md` | May 27 | Orthrus no-diff (nvfp4) | 69 | 2.6 s | 521.0 s |

Note: the two Orthrus fp8 runs (diffusion and no-diff) share the same `Model (API)` field of `orthrus-qwen3-8b-fp8` because `served_model_id` includes the quant suffix but not the diffusion-mode flag. They're distinguished here by the median-turn-time and wall-clock columns above and (in the .md files themselves) by the `Run ID` timestamp.

All four Orthrus diffusion arms (bf16 72, fp8 74, fp8-row 70, nvfp4 71) cluster within a few points, and all run at diffusion-mode median turn times (1.4-2.0 s) — the drafter is active under every quantisation scheme tested. fp8-row matches fp8 on memory (1.8x) and is the fastest 8-bit scheme on long-form throughput; see `QUANTIZATION.md` and `RESEARCH_LOG.md`.

## Source

Each .md file is the verbatim output of `python -m tool_eval_bench ... > {Run ID}.md` per tool-eval-bench's `--out-dir` convention. The runs were executed against this repo's orthrus-serve endpoint on a DGX Spark (sm_121) at the dates shown. Reproducing instructions: see the `### Reproducing` section in `orthrus-serve/RESEARCH_LOG.md`.

## What's inside each file

- Headline scores: Final Score, Quality, Responsiveness, Deployability, median turn time
- Safety-critical failure list
- Full run context (backend, model, temperature, seed, max turns, timeout, etc.)
- Inference engine details (max model length, quantisation, host)
- Per-category breakdown (Tool Selection, Parameter Precision, Multi-Step Chains, ..., Structured Output)
- Per-scenario results: status (pass / partial / fail), points, summary
- Full per-turn tool-call / tool-result transcripts for every scenario
