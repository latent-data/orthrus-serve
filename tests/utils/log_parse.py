"""
Aggregate per-request stats from orthrus-serve INFO logs.

Each request log line looks like:
    {"time":"YYYY-MM-DD HH:MM:SS,sss","level":"INFO","logger":"orthrus_serve",
     "message":{"request_id":"...","prompt_tokens":...,"completion_tokens":...,
                "ttft_s":...,"total_s":...,"tok_per_s":...,"tool_calls":bool,
                "finish_reason":"...","orthrus_revision":"..."}}

Usage:
    python tests/utils/log_parse.py /tmp/serve.log
    python tests/utils/log_parse.py /tmp/serve.log --since "2026-05-25 10:07"
    cat /tmp/serve.log | python tests/utils/log_parse.py -
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from typing import Iterable

# Bucket boundaries by completion_tokens
BUCKETS: list[tuple[str, int, int]] = [
    ("<10", 0, 10),
    ("10-30", 10, 30),
    ("30-100", 30, 100),
    ("100-300", 100, 300),
    ("300+", 300, 10**9),
]


def parse_log(lines: Iterable[str], since: str | None = None) -> list[dict]:
    """Extract request log dicts, optionally filtered by timestamp string prefix."""
    reqs: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            outer = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = outer.get("message")
        if not isinstance(msg, dict) or "request_id" not in msg:
            continue
        ts = outer.get("time", "")
        if since and ts < since:
            continue
        msg["_ts"] = ts
        reqs.append(msg)
    return reqs


def _quantile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _print_stats(name: str, vals: list[float]) -> None:
    if not vals:
        print(f"  {name:<20}  (no data)")
        return
    print(
        f"  {name:<20}  "
        f"mean={statistics.mean(vals):>7.2f}  "
        f"median={statistics.median(vals):>7.2f}  "
        f"p10={_quantile(vals, 0.1):>7.2f}  "
        f"p90={_quantile(vals, 0.9):>7.2f}  "
        f"min={min(vals):>7.2f}  "
        f"max={max(vals):>7.2f}"
    )


def _print_bucket_throughput(reqs: list[dict]) -> None:
    print("\ntok_per_s by completion_tokens bucket:")
    for label, lo, hi in BUCKETS:
        vs = [r["tok_per_s"] for r in reqs if lo <= r["completion_tokens"] < hi]
        if not vs:
            continue
        print(
            f"  {label:>8} (n={len(vs):>4}): "
            f"mean={statistics.mean(vs):>5.1f} tok/s  "
            f"median={statistics.median(vs):>5.1f}  "
            f"range {min(vs):>5.1f}-{max(vs):>5.1f}"
        )


def summarize(reqs: list[dict]) -> None:
    print(f"Parsed: {len(reqs)} requests")
    if not reqs:
        return
    tc = sum(1 for r in reqs if r.get("tool_calls"))
    print(f"Tool-call turns: {tc} / {len(reqs)}")

    for field in ("completion_tokens", "prompt_tokens", "ttft_s", "total_s", "tok_per_s"):
        vals = [float(r[field]) for r in reqs if field in r]
        _print_stats(field, vals)

    total_gen = sum(r.get("total_s", 0) for r in reqs)
    print(f"Total wall-time generating: {total_gen:.1f}s")

    _print_bucket_throughput(reqs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="Log file paths, or '-' for stdin")
    p.add_argument(
        "--since",
        help="Drop requests with time < this string (e.g. '2026-05-25 10:07'). "
             "String compare, so use ISO-ish format.",
    )
    args = p.parse_args()

    lines: list[str] = []
    for f in args.files:
        if f == "-":
            lines.extend(sys.stdin.readlines())
        else:
            with open(f) as fh:
                lines.extend(fh)

    reqs = parse_log(lines, since=args.since)
    if not reqs:
        sys.exit("No request log lines found.")
    summarize(reqs)


if __name__ == "__main__":
    main()
