"""
HTTP benchmark for orthrus-serve.

Hits POST /v1/chat/completions on a running server with the same prompts as
benchmarks/benchmark.py (in-process). Each server instance serves one config
(diff / no-diff / base), so run this script once per config with a unique
--label; results accumulate into a single JSON file for side-by-side comparison.

Usage:
    # serve started with default (Orthrus, diffusion on)
    python benchmarks/benchmark_http.py --label orthrus_diffusion

    # restart serve with --no-diffusion, then:
    python benchmarks/benchmark_http.py --label orthrus_nodiff

    # restart serve with --with-base-model, then:
    python benchmarks/benchmark_http.py --label qwen3_8b_ar --model qwen3-8b

Stdlib only — no extra deps needed on the host.
"""
import argparse
import datetime
import json
import os
import sys
import time
import urllib.request

# Kept in sync with benchmarks/benchmark.py — duplicated here so this script is
# importable from the host without torch/transformers in the environment.
PROMPTS = {
    "short": (
        "Write a program to count the frequency of each word in a paragraph."
    ),
    "long": (
        "Implement a Python class BoundedPriorityQueue backed by a binary heap. "
        "The class takes a capacity (int) and max_heap (bool, default False) at "
        "construction. Implement: push(item, priority) which adds the item and raises "
        "RuntimeError if at capacity; pop() which removes and returns the highest-priority "
        "item, raising IndexError if empty; peek() which returns the best item without "
        "removing it, raising IndexError if empty; __len__; __bool__; __iter__ yielding "
        "(item, priority) pairs in priority order without mutating the queue; and a "
        "classmethod from_items(items, capacity, max_heap=False) accepting an iterable "
        "of (item, priority) pairs. Use full type annotations throughout. Then write a "
        "complete unittest.TestCase covering: push/pop round-trip, capacity enforcement, "
        "min-heap and max-heap ordering, peek, iteration order, from_items bulk loading, "
        "empty-queue edge cases, and duplicate priorities."
    ),
}
DEFAULT_MAX_NEW_TOKENS = 2048

DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_OUTPUT = "results/results_http.json"


def parse_args():
    p = argparse.ArgumentParser(description="HTTP benchmark for orthrus-serve")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help=f"Server base URL (default: {DEFAULT_BASE_URL})")
    p.add_argument("--model", default="orthrus-qwen3-8b",
                   help="Model ID in the request body (use 'qwen3-8b' when serving --with-base-model)")
    p.add_argument("--label", required=True,
                   help="Identifier for this run in the output, e.g. orthrus_diffusion")
    p.add_argument("--prompts", action="append", metavar="NAME", default=None,
                   help=f"Named prompt (default: all). Valid: {', '.join(PROMPTS)}")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--runs", type=int, default=1, metavar="N")
    p.add_argument("--warmup", action="store_true",
                   help="Issue one short throwaway request before timing")
    p.add_argument("--disable-thinking", action="store_true",
                   help="Pass chat_template_kwargs.enable_thinking=false")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="Per-request timeout in seconds (default: 600)")
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    args = p.parse_args()

    if args.runs < 1:
        p.error("--runs must be at least 1")
    selected = args.prompts if args.prompts is not None else list(PROMPTS)
    unknown = [n for n in selected if n not in PROMPTS]
    if unknown:
        p.error(f"Unknown prompt name(s): {', '.join(unknown)}. Valid: {', '.join(PROMPTS)}")
    args.prompts = selected
    return args


def call(base_url, model, prompt, max_new_tokens, disable_thinking, timeout):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    if disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def run_one(args, prompt_name):
    prompt = PROMPTS[prompt_name]
    print(f"\n=== {args.label} [{prompt_name}] ===")

    if args.warmup:
        print("  warmup (32 tokens) ...")
        call(args.base_url, args.model, prompt, 32, args.disable_thinking, args.timeout)
        print("  warmup done")

    elapsed_list = []
    completion_tokens = 0
    prompt_tokens = 0
    snippet = ""
    for i in range(args.runs):
        label = f"run {i + 1}/{args.runs}" if args.runs > 1 else "measuring"
        print(f"  {label} (up to {args.max_new_tokens} tokens) ...")
        t0 = time.perf_counter()
        resp = call(args.base_url, args.model, prompt, args.max_new_tokens,
                    args.disable_thinking, args.timeout)
        elapsed_list.append(time.perf_counter() - t0)
        usage = resp.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        content = resp["choices"][0]["message"].get("content") or ""
        snippet = content[:300].replace("\n", " ")

    tps_list = [completion_tokens / e if e > 0 else 0.0 for e in elapsed_list]
    elapsed_mean = sum(elapsed_list) / len(elapsed_list)
    tps_mean = completion_tokens / elapsed_mean if elapsed_mean > 0 else 0.0

    print(f"  prompt tokens:  {prompt_tokens}")
    print(f"  output tokens:  {completion_tokens}")
    if args.runs > 1:
        run_summary = "  ".join(f"{e:.2f}s/{t:.1f}tok/s" for e, t in zip(elapsed_list, tps_list))
        print(f"  runs:           {run_summary}")
    print(f"  elapsed:        {elapsed_mean:.2f} s{'  (mean)' if args.runs > 1 else ''}")
    print(f"  throughput:     {tps_mean:.1f} tok/s{'  (mean)' if args.runs > 1 else ''}")
    print(f"  output[0:300]:  {snippet!r}")

    result = {
        "prompt_tokens": prompt_tokens,
        "tokens": completion_tokens,
        "elapsed": round(elapsed_mean, 3),
        "tps": round(tps_mean, 2),
        "snippet": snippet,
    }
    if args.runs > 1:
        result["elapsed_runs"] = [round(e, 3) for e in elapsed_list]
        result["tps_runs"] = [round(t, 2) for t in tps_list]
    return result


def main():
    args = parse_args()

    print(f"Base URL: {args.base_url}")
    print(f"Model:    {args.model}")
    print(f"Label:    {args.label}")

    try:
        with urllib.request.urlopen(f"{args.base_url.rstrip('/')}/health", timeout=10) as r:
            print(f"Health:   {r.read().decode().strip()}")
    except Exception as e:
        raise SystemExit(f"Cannot reach {args.base_url}/health: {e}")

    results = {}
    for prompt_name in args.prompts:
        results[prompt_name] = run_one(args, prompt_name)

    print("\n=== Summary ===")
    prompt_w = max(len(p) for p in args.prompts) + 2
    label_w = max(len(args.label), 8) + 2
    header = (f"  {'prompt':<{prompt_w}} {'config':<{label_w}}"
              f" {'tokens':>8}  {'elapsed':>10}  {'throughput':>12}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for prompt_name, r in results.items():
        print(
            f"  {prompt_name:<{prompt_w}} {args.label:<{label_w}}"
            f" {r['tokens']:>8}  {r['elapsed']:>8.2f} s  {r['tps']:>8.1f} tok/s"
        )

    run_record = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "label": args.label,
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "runs": args.runs,
            "warmup": args.warmup,
            "disable_thinking": args.disable_thinking,
            "prompts": args.prompts,
        },
        "results": results,
    }

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Append into a single file keyed by label so multiple invocations accumulate.
    existing = {"labels": {}}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("labels"), dict):
                existing = data
        except (json.JSONDecodeError, OSError):
            pass
    existing["labels"][args.label] = run_record

    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults appended to {args.output} under label '{args.label}'")


if __name__ == "__main__":
    main()
