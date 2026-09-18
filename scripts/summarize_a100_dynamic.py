#!/usr/bin/env python
"""Summarize matched warmed trials; incomplete and unequal trials stay visible."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median


COMPONENT_KEYS = (
    "prefill_time", "cache_init_time", "selection_prefill_time", "selection_time",
    "dense_prefill_time", "sparse_cache_time", "draft_time", "verify_time",
    "bonus_time", "cache_adjust_time",
)


def headroom_summary(groups, samples, methods):
    """Per-sample paired acceptance plus H_headroom stratification.

    Pairs every dynamic method against `static` on the *same* sample. The
    aggregate mean is explicitly not trusted: on the 2026-09-09 A100 run it hid
    a sign flip (+0.044 / -0.102 / -0.126 over three samples) that correlated
    with how much headroom static still had.
    """
    dynamic_methods = [m for m in methods if m.startswith("dynamic_")]
    static_accept = {
        s: median(r.get("acceptance_rate", 0.0) for r in groups[s, "static"]) for s in samples
    }

    paired = []
    for s in samples:
        for m in dynamic_methods:
            dyn = median(r.get("acceptance_rate", 0.0) for r in groups[s, m])
            paired.append(
                {
                    "sample_id": s,
                    "method": m,
                    "static_accept": static_accept[s],
                    "dynamic_accept": dyn,
                    "delta_accept": dyn - static_accept[s],
                }
            )

    strata = []
    for m in dynamic_methods:
        rows = sorted(
            (p for p in paired if p["method"] == m), key=lambda p: p["static_accept"]
        )
        n = len(rows)
        n_bins = min(3, n)
        for b in range(n_bins):
            # Equal-count edges over n_bins; using //3 here would drop every
            # sample when n is smaller than the bin count.
            lo, hi = (b * n) // n_bins, ((b + 1) * n) // n_bins
            chunk = rows[lo:hi]
            if not chunk:
                continue
            strata.append(
                {
                    "method": m,
                    "bin": b,
                    "n": len(chunk),
                    "static_min": chunk[0]["static_accept"],
                    "static_max": chunk[-1]["static_accept"],
                    "mean_static_accept": sum(c["static_accept"] for c in chunk) / len(chunk),
                    "mean_delta_accept": sum(c["delta_accept"] for c in chunk) / len(chunk),
                }
            )
    return {"paired": paired, "strata": strata}


def summarize(path):
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    manifest = next(r for r in rows if r["kind"] == "manifest")
    samples = manifest["sample_ids"]
    methods = ["ar", "static"]
    methods.extend(sorted({r["method"] for r in rows if r.get("kind") == "trial" and r["method"].startswith("dynamic_")}))
    groups = defaultdict(list)
    comparisons = defaultdict(list)
    for r in rows:
        if r.get("phase") != "measure":
            continue
        if r["kind"] == "trial":
            groups[r["sample_id"], r["method"]].append(r)
        elif r["kind"] == "comparison":
            comparisons[r["method"]].append(r)
    complete = any(r["kind"] == "complete" for r in rows)
    expected = manifest["args"]["repeats"]
    if not complete or any(len(groups[s, m]) != expected for s in samples for m in methods):
        raise ValueError("Experiment incomplete or trial counts disagree with the manifest")
    totals = {}
    per_sample = []
    for s in samples:
        timings = {m: {k: median(r.get(k, 0.0) for r in groups[s, m])
                       for k in ("decoding_time", "inference_time", *COMPONENT_KEYS)} for m in methods}
        per_sample.append({"sample_id": s, "medians": timings})
    for m in methods:
        totals[m] = {k: sum(s["medians"][m][k] for s in per_sample)
                     for k in ("decoding_time", "inference_time", *COMPONENT_KEYS)}
        cmps = comparisons[m]
        if m != "ar" and len(cmps) != len(samples) * expected:
            raise ValueError("Missing correctness comparisons")
        totals[m]["exact_trials"] = len(samples) * expected if m == "ar" else sum(r["token_equal"] for r in cmps)
        totals[m]["total_trials"] = len(samples) * expected
        totals[m]["max_token_mismatches"] = max((r["mismatch_token_count"] for r in cmps), default=0)
        totals[m]["peak_memory_gib"] = max(r["peak_memory_gib"] for s in samples for r in groups[s, m])
    for m in methods:
        for baseline in ("ar", "static"):
            for metric in ("decoding_time", "inference_time"):
                totals[m][f"{metric}_speedup_vs_{baseline}"] = totals[baseline][metric] / totals[m][metric]
    return {
        "manifest": manifest,
        "totals": totals,
        "per_sample": per_sample,
        "headroom": headroom_summary(groups, samples, methods),
        "timing_valid": bool(manifest.get("timing_valid", True)),
        "complete": complete,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    (args.output_dir / f"{stem}_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    cfg = result["manifest"]["args"]
    # Co-tenancy makes wall-clock meaningless; acceptance/exactness stay valid.
    timing_valid = result["timing_valid"]

    def speedup(value: float) -> str:
        return f"{value:.3f}x" if timing_valid else "n/a"

    text = [f"A100 dynamic STD comparison — {stem}", "",
            f"Video-MME: {len(result['per_sample'])} available-subset samples, seed-42 order; "
            f"{cfg['frame_num']} requested frames; {cfg['max_new_tokens']} fixed output tokens; "
            f"{cfg['repeats']} measured repetitions per method/sample after {cfg['warmup_tokens']}-token warmups.",
            f"Physical GPU {cfg['gpu']}; FP16; sparse backend {cfg['sparse_attn_mode']}; "
            f"collector={cfg.get('dynamic_collector', 'legacy')}; refresh={cfg.get('refresh_mode', 'legacy')}; "
            f"update_interval={cfg.get('selection_update_interval', 1)}; "
            f"min_change={cfg.get('min_selection_change_ratio', 0.0)}; gamma={cfg['gamma']}; "
            f"query_mode={cfg.get('dynamic_query_mode', 'three')}; "
            f"bootstrap={cfg.get('dynamic_bootstrap', 'attention')}; "
            f"coverage={cfg.get('bootstrap_coverage_ratio', 0.25)}; "
            f"value_weight={cfg.get('bootstrap_value_weight', 0.25)}; "
            f"K+text={cfg['k_plus_text']}; fallback={cfg['verify_fallback']}.", ""]
    if not timing_valid:
        text += [
            "> **WARNING: `timing_valid=false`** — this run shared a non-idle GPU. All wall-clock "
            "numbers and speedups are invalid and are shown as `n/a`. Acceptance and exactness "
            "remain valid because they are deterministic for fixed inputs.", "",
        ]
    text += [
            "Times below sum each sample's median across repetitions. Speedup = baseline / method; greater than 1 is faster.", "",
            "| Method | Prefill (s) | Decode (s) | vs AR | vs static | Inference (s) | Inf. vs AR | Inf. vs static | Exact | Peak GiB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, r in result["totals"].items():
        text.append(f"| {method} | {r['prefill_time']:.3f} | {r['decoding_time']:.3f} | {speedup(r['decoding_time_speedup_vs_ar'])} | "
                    f"{speedup(r['decoding_time_speedup_vs_static'])} | {r['inference_time']:.3f} | "
                    f"{speedup(r['inference_time_speedup_vs_ar'])} | {speedup(r['inference_time_speedup_vs_static'])} | "
                    f"{r['exact_trials']}/{r['total_trials']} | {r['peak_memory_gib']:.2f} |")
    text += ["", "Component totals (when --profile-components is enabled; medians per sample):",
             "| Method | Cache init | Selection prefill | Selection/top-k | Dense prefill | Sparse cache | Draft | Verify | Bonus | Cache adjust |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, r in result["totals"].items():
        text.append(f"| {method} | {r['cache_init_time']:.3f} | {r['selection_prefill_time']:.3f} | "
                    f"{r['selection_time']:.3f} | {r['dense_prefill_time']:.3f} | {r['sparse_cache_time']:.3f} | "
                    f"{r['draft_time']:.3f} | {r['verify_time']:.3f} | {r['bonus_time']:.3f} | {r['cache_adjust_time']:.3f} |")
    headroom = result.get("headroom") or {"paired": [], "strata": []}
    if headroom["paired"]:
        text += ["", "Per-sample paired acceptance (H_headroom; the aggregate mean is not trusted):",
                 "| Method | Sample | Static accept | Dynamic accept | delta |",
                 "|---|---|---:|---:|---:|"]
        for p in headroom["paired"]:
            text.append(
                f"| {p['method']} | {p['sample_id']} | {p['static_accept']:.3f} | "
                f"{p['dynamic_accept']:.3f} | {p['delta_accept']:+.3f} |"
            )
    if headroom["strata"]:
        text += ["", "Stratified by static headroom (equal-count bins, LOW headroom first):",
                 "| Method | Bin | n | static range | mean static accept | mean delta accept |",
                 "|---|---:|---:|---|---:|---:|"]
        for s in headroom["strata"]:
            text.append(
                f"| {s['method']} | {s['bin']} | {s['n']} | "
                f"{s['static_min']:.3f}-{s['static_max']:.3f} | {s['mean_static_accept']:.3f} | "
                f"{s['mean_delta_accept']:+.3f} |"
            )

    text += ["", "Measured generation continues after EOS to equalize work. Inference includes prefill and decoding; "
             "both timing measures exclude model loading and video processing. Exact-trial counts compare complete "
             "token sequences against AR. Unequal outputs do not demonstrate lossless speedup.", "",
             "This small available-video subset is exploratory, not a full Video-MME accuracy evaluation. "
             "The synchronized total decode times include all dynamic selection/collection/refresh overhead.", "",
             f"Raw trials: {args.input.name}",
             f"Git base: {result['manifest']['git_head']}; worktree modified={result['manifest'].get('git_worktree_modified', 'unknown')}; "
             "exact source hashes are authoritative in the raw manifest."]
    markdown = "\n".join(text) + "\n"
    (args.output_dir / f"{stem}_report.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
