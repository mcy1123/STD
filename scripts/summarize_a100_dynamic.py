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
    return {"manifest": manifest, "totals": totals, "per_sample": per_sample, "complete": complete}


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
            f"K+text={cfg['k_plus_text']}; fallback={cfg['verify_fallback']}.", "",
            "Times below sum each sample's median across repetitions. Speedup = baseline / method; greater than 1 is faster.", "",
            "| Method | Prefill (s) | Decode (s) | vs AR | vs static | Inference (s) | Inf. vs AR | Inf. vs static | Exact | Peak GiB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, r in result["totals"].items():
        text.append(f"| {method} | {r['prefill_time']:.3f} | {r['decoding_time']:.3f} | {r['decoding_time_speedup_vs_ar']:.3f}x | "
                    f"{r['decoding_time_speedup_vs_static']:.3f}x | {r['inference_time']:.3f} | "
                    f"{r['inference_time_speedup_vs_ar']:.3f}x | {r['inference_time_speedup_vs_static']:.3f}x | "
                    f"{r['exact_trials']}/{r['total_trials']} | {r['peak_memory_gib']:.2f} |")
    text += ["", "Component totals (when --profile-components is enabled; medians per sample):",
             "| Method | Cache init | Selection prefill | Selection/top-k | Dense prefill | Sparse cache | Draft | Verify | Bonus | Cache adjust |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, r in result["totals"].items():
        text.append(f"| {method} | {r['cache_init_time']:.3f} | {r['selection_prefill_time']:.3f} | "
                    f"{r['selection_time']:.3f} | {r['dense_prefill_time']:.3f} | {r['sparse_cache_time']:.3f} | "
                    f"{r['draft_time']:.3f} | {r['verify_time']:.3f} | {r['bonus_time']:.3f} | {r['cache_adjust_time']:.3f} |")
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
