#!/usr/bin/env python
"""Phase 6: turn the batch-sweep JSONL evidence into results/vllm-std/batch_sweep.csv.

Emits one row per (batch, mode) with the raw timings plus the derived figures the
attribution needs: throughput, acceptance, and the framework tax that has to be
paid before any speculative gain exists.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path, args_prefix: str = "p6_") -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(r.get("tag", "")).startswith(args_prefix):
                rows.append(r)
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="results/vllm-std/spec_bench.jsonl")
    p.add_argument("--out", default="results/vllm-std/batch_sweep.csv")
    p.add_argument("--prefix", default="p6_",
                   help="tag prefix selecting which records to summarise")
    args = p.parse_args()

    rows = load(Path(args.file), args.prefix)
    if not rows:
        print(f"no {args.prefix}* records found")
        return 1

    # Latest record per tag.
    by_tag: dict[str, dict] = {}
    for r in rows:
        by_tag[r["tag"]] = r
    rows = list(by_tag.values())

    recs = []
    for r in rows:
        mode = "spec" if r.get("proposer") else "eager"
        b = r.get("batch_size")
        sec = r.get("seconds_mean") or 0.0
        out_tok = sum(x["num_output_tokens"] for x in r["runs"][0]["requests"]) if r.get("runs") else 0
        acc = r.get("acceptance_agg") or {}
        recs.append({
            "batch": b,
            "mode": mode,
            "proposer": r.get("proposer") or "",
            "num_spec_tokens": r.get("num_spec_tokens", 0),
            "num_frames": r.get("num_frames"),
            "num_prompt_tokens": r.get("num_prompt_tokens"),
            "seconds_mean": round(sec, 4),
            "seconds_std": round(r.get("seconds_std") or 0.0, 4),
            "output_tokens": out_tok,
            "throughput_tok_s": round(out_tok / sec, 3) if sec else None,
            "drafted": acc.get("drafted"),
            "accepted": acc.get("accepted"),
            "drafts": acc.get("drafts"),
            "accept_rate": acc.get("accept_rate"),
            "accept_len": acc.get("accept_len"),
            "gpu": (r.get("gpu_before") or {}).get("visible"),
        })

    recs.sort(key=lambda x: (x["batch"], x["mode"]))

    # Derived comparisons, joined on batch.
    eager = {r["batch"]: r for r in recs if r["mode"] == "eager"}
    for r in recs:
        e = eager.get(r["batch"])
        r["eager_seconds"] = e["seconds_mean"] if e else None
        r["speedup_vs_eager"] = (round(e["seconds_mean"] / r["seconds_mean"], 4)
                                 if e and r["seconds_mean"] else None)
        r["throughput_gain"] = (round(r["throughput_tok_s"] / e["throughput_tok_s"], 4)
                                if e and e["throughput_tok_s"] and r["throughput_tok_s"]
                                else None)

    cols = ["batch", "mode", "proposer", "num_spec_tokens", "num_frames",
            "num_prompt_tokens", "seconds_mean", "seconds_std", "output_tokens",
            "throughput_tok_s", "drafted", "accepted", "drafts", "accept_rate",
            "accept_len", "eager_seconds", "speedup_vs_eager",
            "throughput_gain", "gpu"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k) for k in cols})
    print(f"wrote {out} ({len(recs)} rows)")
    for r in recs:
        print(f"  batch={r['batch']:<3} {r['mode']:<6} sec={r['seconds_mean']:<8} "
              f"speedup={r['speedup_vs_eager']} accept_rate={r['accept_rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
