#!/usr/bin/env python
"""Compare speculative-decoding runs against an eager baseline, bit for bit.

Phase 3's gate is "output identical, or the difference has an explicit cause", so
this tool answers precisely:

  * do two runs produce identical greedy token ids?
  * if not, at WHICH index do they first differ, and what are the two tokens?

It also surfaces the timing and acceptance numbers side by side, so one command
produces the whole phase table.

Usage
-----
    $PY scripts/vllm-std/compare_runs.py \
        --file results/vllm-std/spec_bench.jsonl \
        --ref p3_base_k0 --tags p3_empty_k4 p3_repeat_k4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    """Latest record per tag (the jsonl is append-only across reruns)."""
    out: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "tag" in r:
                out[r["tag"]] = r
    return out


def first_divergence(a: list[int], b: list[int]) -> dict | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return {"index": i, "ref_token": x, "got_token": y,
                    "ref_context": a[max(0, i - 5):i + 5],
                    "got_context": b[max(0, i - 5):i + 5]}
    if len(a) != len(b):
        return {"index": min(len(a), len(b)), "reason": "length mismatch",
                "ref_len": len(a), "got_len": len(b)}
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="results/vllm-std/spec_bench.jsonl")
    p.add_argument("--ref", required=True, help="tag of the reference run")
    p.add_argument("--tags", nargs="+", required=True, help="tags to compare")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    recs = load(Path(args.file))
    if args.ref not in recs:
        print(f"FATAL: reference tag '{args.ref}' not in {args.file}. "
              f"Known: {sorted(recs)}", file=sys.stderr)
        return 2

    ref = recs[args.ref]
    ref_ids = ref["token_ids"][0]
    print(f"reference: {args.ref}")
    print(f"  prompt_tokens={ref.get('num_prompt_tokens')} "
          f"output_tokens={len(ref_ids)} "
          f"seconds={ref.get('seconds_mean')}±{ref.get('seconds_std')}")
    print()

    results = []
    hdr = f"{'tag':<18} {'bit-exact':<10} {'1st diff':<9} {'sec mean':>9} {'sec std':>8} {'accept_rate':>11} {'accept_len':>10}"
    print(hdr)
    print("-" * len(hdr))
    for tag in args.tags:
        if tag not in recs:
            print(f"{tag:<18} MISSING")
            continue
        r = recs[tag]
        ids = r["token_ids"][0]
        div = first_divergence(ref_ids, ids)
        acc = r.get("acceptance_agg") or {}
        ar = acc.get("accept_rate")
        al = acc.get("accept_len")
        # A run that never drafts anything is trivially identical; say so explicitly.
        noted = "yes" if div is None else "NO"
        if r.get("num_spec_tokens", 0) == 0:
            noted += "*"
        print(f"{tag:<18} {noted:<10} {('-' if div is None else div['index']):<9} "
              f"{r.get('seconds_mean'):>9} {r.get('seconds_std'):>8} "
              f"{('-' if ar is None else ar):>11} {('-' if al is None else al):>10}")
        row = {"tag": tag, "bit_exact": div is None, "divergence": div,
               "seconds_mean": r.get("seconds_mean"),
               "seconds_std": r.get("seconds_std"),
               "acceptance": acc,
               "num_spec_tokens": r.get("num_spec_tokens"),
               "output_text": r.get("output_text")}
        results.append(row)
        if div is not None:
            print(f"    ↳ first divergence at token {div['index']}: "
                  f"ref={div.get('ref_token')} got={div.get('got_token')}")
            print(f"      ref ctx: {div.get('ref_context')}")
            print(f"      got ctx: {div.get('got_context')}")

    print()
    print("* = run declared no speculative tokens, so equality is expected by construction")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"reference": args.ref, "results": results}, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
