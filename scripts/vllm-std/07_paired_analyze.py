#!/usr/bin/env python
"""Paired analysis for the Phase 7 interleaved A/B timing runs.

Reports the paired difference per pair (B - A), its mean/std, and the paired
speedup A/B, which is what "does speculation help" actually asks.
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="results/vllm-std/spec_bench.jsonl")
    p.add_argument("--out", default="results/vllm-std/07_paired.json")
    args = p.parse_args()
    recs = {}
    for line in open(args.file):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = r.get("tag", "")
        if t.startswith("p7_"):
            recs[t] = r

    pairs = []
    i = 1
    while f"p7_A{i}" in recs and f"p7_B{i}" in recs:
        a, b = recs[f"p7_A{i}"], recs[f"p7_B{i}"]
        pairs.append({
            "pair": i,
            "eager_s": a["seconds_mean"], "spec_s": b["seconds_mean"],
            "diff_s": round(b["seconds_mean"] - a["seconds_mean"], 4),
            "speedup": round(a["seconds_mean"] / b["seconds_mean"], 4),
            "accept_rate": (b.get("acceptance_agg") or {}).get("accept_rate"),
            "eager_gpu": (a.get("gpu_before") or {}).get("visible"),
            "spec_gpu": (b.get("gpu_before") or {}).get("visible"),
        })
        i += 1

    if not pairs:
        print("no complete p7_A/p7_B pairs found")
        return 1

    diffs = [x["diff_s"] for x in pairs]
    sps = [x["speedup"] for x in pairs]
    summary = {
        "n_pairs": len(pairs),
        "pairs": pairs,
        "mean_diff_s": round(statistics.mean(diffs), 4),
        "std_diff_s": round(statistics.stdev(diffs), 4) if len(diffs) > 1 else 0.0,
        "mean_speedup": round(statistics.mean(sps), 4),
        "std_speedup": round(statistics.stdev(sps), 4) if len(sps) > 1 else 0.0,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"{'pair':<6}{'eager_s':>10}{'spec_s':>10}{'diff_s':>10}{'speedup':>10}{'accept':>9}")
    for x in pairs:
        print(f"{x['pair']:<6}{x['eager_s']:>10}{x['spec_s']:>10}{x['diff_s']:>10}"
              f"{x['speedup']:>10}{str(x['accept_rate']):>9}")
    print(f"\npaired speedup: {summary['mean_speedup']} ± {summary['std_speedup']} "
          f"(n={summary['n_pairs']})")
    print(f"paired diff   : {summary['mean_diff_s']} ± {summary['std_diff_s']} s")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
