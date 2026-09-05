#!/usr/bin/env python
"""Benchmark Verification-Guided STD: Static / Dynamic MVP / Dynamic optimized.

For each sample it runs four decoders:

  * AR        — vanilla greedy decode (correctness reference);
  * Static    — the frozen baseline ``std_generate_qwen25vl``;
  * Dynamic MVP       — ``dynamic_std_generate_qwen25vl`` with
    ``policy="previous_verify_topk"``, V1 collector (all-query, in-forward GEMM
    with per-layer ``.cpu()``) and full compact-prompt rebuild refresh;
  * Dynamic optimized — same policy with V3 fused collector (all-query,
    in-forward GEMM, single end-of-round ``.cpu()``) and full rebuild refresh.

Usage:
  python scripts/benchmark_dynamic_std.py \
    --dataset VideoDetailCaption --data-dir /mnt/local2/mcy/datasets/VideoDetailCaption \
    --frame-num 128 --max-new-tokens 256 --gamma 9 --limit 10 --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_gpu = "0"
for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        _gpu = sys.argv[_i + 1]
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
os.environ.setdefault("SPECVLM_MAX_CACHE_LEN", "32768")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd  # noqa: E402

from benchmark_std import (  # noqa: E402
    VIDEO_TOKEN_ID,
    build_mismatch_diagnostic,
    load_qwen_model,
    make_qwen_video_inputs,
)
from std_repro.std_qwen25vl import (  # noqa: E402
    ar_generate_qwen25vl,
    generated_suffix,
    std_generate_qwen25vl,
    tokens_equal,
)
from std_repro.dynamic_std_qwen25vl import dynamic_std_generate_qwen25vl  # noqa: E402
from std_repro.verification_policy import positional_token_metrics  # noqa: E402


def iter_vdc_samples(data_dir: str, limit: int):
    parquet = Path(data_dir) / "data" / "test-00000-of-00001.parquet"
    df = pd.read_parquet(parquet)
    video_dir = Path(data_dir) / "Test_Videos"
    yielded = 0
    for _, row in df.iterrows():
        vpath = video_dir / f"{row['video_name']}.mp4"
        if not vpath.exists():
            continue
        yield {"sample_id": row["video_name"], "video_path": str(vpath), "question": row["question"]}
        yielded += 1
        if yielded >= limit:
            break


def iter_mlvu_samples(data_dir: str, limit: int):
    parquet = Path(data_dir) / "mlvu_test" / "test-00000-of-00001.parquet"
    df = pd.read_parquet(parquet)
    video_dir = Path(data_dir) / "videos"
    yielded = 0
    for _, row in df.iterrows():
        vpath = video_dir / row["video_name"]
        if not vpath.exists():
            continue
        yield {
            "sample_id": row.get("question_id", row["video_name"]),
            "video_path": str(vpath),
            "question": row["question"],
        }
        yielded += 1
        if yielded >= limit:
            break


def _std_fields(prefix: str, result, dstats, match: bool) -> dict:
    return {
        f"{prefix}_token_match": bool(match),
        f"{prefix}_mean_accept": result.mean_accept_length,
        f"{prefix}_accept_rate": result.acceptance_rate,
        f"{prefix}_decode_rounds": result.decode_rounds,
        f"{prefix}_decoding_time": result.decoding_time,
        f"{prefix}_draft_time": result.draft_time,
        f"{prefix}_verify_time": result.verify_time,
        f"{prefix}_total_refresh_ms": dstats["total_refresh_time_ms"],
        f"{prefix}_mean_refresh_ms": dstats["mean_refresh_time_ms"],
        f"{prefix}_total_collect_ms": dstats["total_collect_time_ms"],
        f"{prefix}_mean_jaccard": dstats["mean_jaccard_old_new"],
        f"{prefix}_mean_changed_tokens": dstats["mean_changed_tokens"],
        f"{prefix}_fallback_count": result.fallback_count,
        f"{prefix}_verify_margin_reruns": result.verify_margin_reruns,
        f"{prefix}_min_verify_margin": result.min_verify_margin,
    }


def _agreement_fields(prefix: str, candidate, reference, tokenizer, eos_token_id: int) -> dict:
    diagnostic = build_mismatch_diagnostic(reference, candidate, tokenizer, eos_token_id)
    metrics = positional_token_metrics(reference, candidate)
    return {
        f"{prefix}_mismatch_token_count": metrics["mismatch_token_count"],
        f"{prefix}_token_level_agreement": metrics["token_level_agreement"],
        f"{prefix}_common_prefix_length": diagnostic["common_prefix_length"],
        f"{prefix}_first_ar_token_id": diagnostic["ar_token_id"],
        f"{prefix}_first_candidate_token_id": diagnostic["std_token_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/mnt/local2/mcy/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset", choices=["VideoDetailCaption", "MLVU"], default="VideoDetailCaption")
    parser.add_argument("--data-dir", default="/mnt/local2/mcy/datasets/VideoDetailCaption")
    parser.add_argument("--frame-num", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gamma", type=int, default=9)
    parser.add_argument("--k-plus-text", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--verify-fallback",
        choices=[
            "none",
            "sequential_on_reject",
            "sequential_on_low_accept",
            "sequential_on_low_margin",
            "sequential_guard",
        ],
        default="sequential_on_low_margin",
        help="Correctness fallback applied after fast parallel verification.",
    )
    parser.add_argument(
        "--verify-margin-threshold",
        type=float,
        default=0.1,
        help="Top-1/top-2 logit margin below which sequential re-verification is used.",
    )
    parser.add_argument("--sequential-fallback-max-accept", type=int, default=1)
    parser.add_argument(
        "--max-mismatch-tokens",
        type=int,
        default=2,
        help="Maximum tolerated positional token mismatches per generated sample.",
    )
    parser.add_argument(
        "--fail-on-correctness",
        action="store_true",
        help="Abort after writing a record when any decoder exceeds max-mismatch-tokens.",
    )
    parser.add_argument("--output", default=str(ROOT / "results" / "dynamic_std_mvp" / "vdc10_3col.jsonl"))
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model on GPU {_gpu} ...", flush=True)
    model, processor = load_qwen_model(args.model_path, "0")
    eos = processor.tokenizer.eos_token_id

    if args.dataset == "VideoDetailCaption":
        samples = list(iter_vdc_samples(args.data_dir, args.limit))
    else:
        samples = list(iter_mlvu_samples(args.data_dir, args.limit))
    print(f"Dataset={args.dataset}  samples={len(samples)}  frame={args.frame_num} "
          f"max_new={args.max_new_tokens}  gamma={args.gamma}", flush=True)

    mismatch_totals = {"static": 0, "mvp": 0, "opt": 0}
    agreement_totals = {"static": 0.0, "mvp": 0.0, "opt": 0.0}
    within_tolerance = {"static": 0, "mvp": 0, "opt": 0}
    completed_samples = 0
    with out_path.open("w", encoding="utf-8") as f:
        for idx, s in enumerate(samples):
            inputs = make_qwen_video_inputs(
                processor, s["video_path"], s["question"], args.frame_num,
                min_pixels=None, max_pixels=None, target_device="cuda",
            )
            prompt_len = int(inputs["input_ids"].shape[1])

            # AR reference
            ar = ar_generate_qwen25vl(model, inputs, VIDEO_TOKEN_ID, eos,
                                      max_new_tokens=args.max_new_tokens, ignore_eos=True)

            # Static STD (frozen baseline)
            static, sel = std_generate_qwen25vl(
                model, inputs, VIDEO_TOKEN_ID, eos,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                target_k_plus_text=args.k_plus_text,
                ignore_eos=True,
                verify_fallback=args.verify_fallback,
                verify_margin_threshold=args.verify_margin_threshold,
                sequential_fallback_max_accept=args.sequential_fallback_max_accept,
            )

            # Dynamic MVP (V1 collector + full rebuild refresh)
            mvp, dsel_mvp, dstats_mvp = dynamic_std_generate_qwen25vl(
                model, inputs, VIDEO_TOKEN_ID, eos,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                target_k_plus_text=args.k_plus_text,
                policy="previous_verify_topk",
                ignore_eos=True,
                collector_version="v1",
                refresh_mode="full",
                verify_fallback=args.verify_fallback,
                verify_margin_threshold=args.verify_margin_threshold,
                sequential_fallback_max_accept=args.sequential_fallback_max_accept,
            )

            # Dynamic optimized (fused V3 collector + full rebuild refresh)
            opt, dsel_opt, dstats_opt = dynamic_std_generate_qwen25vl(
                model, inputs, VIDEO_TOKEN_ID, eos,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                target_k_plus_text=args.k_plus_text,
                policy="previous_verify_topk",
                ignore_eos=True,
                collector_version="v3",
                refresh_mode="full",
                verify_fallback=args.verify_fallback,
                verify_margin_threshold=args.verify_margin_threshold,
                sequential_fallback_max_accept=args.sequential_fallback_max_accept,
            )

            ar_suffix = generated_suffix(ar.output_ids, prompt_len)
            static_match = tokens_equal(generated_suffix(static.output_ids, prompt_len), ar_suffix)
            mvp_match = tokens_equal(generated_suffix(mvp.output_ids, prompt_len), ar_suffix)
            opt_match = tokens_equal(generated_suffix(opt.output_ids, prompt_len), ar_suffix)
            static_suffix = generated_suffix(static.output_ids, prompt_len)
            mvp_suffix = generated_suffix(mvp.output_ids, prompt_len)
            opt_suffix = generated_suffix(opt.output_ids, prompt_len)
            static_agreement = _agreement_fields(
                "static", static_suffix, ar_suffix, processor.tokenizer, eos
            )
            mvp_agreement = _agreement_fields(
                "mvp", mvp_suffix, ar_suffix, processor.tokenizer, eos
            )
            opt_agreement = _agreement_fields(
                "opt", opt_suffix, ar_suffix, processor.tokenizer, eos
            )
            opt_static_agreement = _agreement_fields(
                "opt_static", opt_suffix, static_suffix, processor.tokenizer, eos
            )

            record = {
                "dataset": args.dataset,
                "sample_id": s["sample_id"],
                "frame_num": args.frame_num,
                "prompt_len": prompt_len,
                "visual_len": sel.visual_len,
                "k": sel.k,
                "gamma": args.gamma,
                "max_new_tokens": args.max_new_tokens,
                "ar_generate_len": ar.generate_len,
                "verify_fallback": args.verify_fallback,
                "verify_margin_threshold": args.verify_margin_threshold,
                "max_mismatch_tokens": args.max_mismatch_tokens,
            }
            record.update(_std_fields("static", static, {
                "total_refresh_time_ms": 0.0, "mean_refresh_time_ms": 0.0,
                "total_collect_time_ms": 0.0, "mean_jaccard_old_new": 1.0,
                "mean_changed_tokens": 0.0,
            }, static_match))
            record.update(_std_fields("mvp", mvp, dstats_mvp, mvp_match))
            record.update(_std_fields("opt", opt, dstats_opt, opt_match))
            record.update(static_agreement)
            record.update(mvp_agreement)
            record.update(opt_agreement)
            record.update(opt_static_agreement)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            completed_samples += 1
            for name, fields in (
                ("static", static_agreement),
                ("mvp", mvp_agreement),
                ("opt", opt_agreement),
            ):
                mismatches = fields[f"{name}_mismatch_token_count"]
                mismatch_totals[name] += mismatches
                agreement_totals[name] += fields[f"{name}_token_level_agreement"]
                within_tolerance[name] += int(mismatches <= args.max_mismatch_tokens)

            print(
                f"[{idx}] {s['sample_id']}  prompt={prompt_len} visual={sel.visual_len} k={sel.k}\n"
                f"    token_match(AR): static={static_match} mvp={mvp_match} opt={opt_match}\n"
                f"    mismatch_tokens: static={static_agreement['static_mismatch_token_count']}  "
                f"mvp={mvp_agreement['mvp_mismatch_token_count']}  "
                f"opt={opt_agreement['opt_mismatch_token_count']}\n"
                f"    fallback_rounds: static={static.fallback_count}  "
                f"mvp={mvp.fallback_count}  opt={opt.fallback_count}\n"
                f"    mean_accept:     static={static.mean_accept_length:5.2f}  "
                f"mvp={mvp.mean_accept_length:5.2f}  opt={opt.mean_accept_length:5.2f}\n"
                f"    accept_rate:     static={static.acceptance_rate:5.3f}  "
                f"mvp={mvp.acceptance_rate:5.3f}  opt={opt.acceptance_rate:5.3f}\n"
                f"    decoding_time:   static={static.decoding_time:6.2f}s  "
                f"mvp={mvp.decoding_time:6.2f}s  opt={opt.decoding_time:6.2f}s\n"
                f"    vs-static:       mvp={static.decoding_time / mvp.decoding_time:.2f}x  "
                f"opt={static.decoding_time / opt.decoding_time:.2f}x\n"
                f"    refresh_total:   mvp={dstats_mvp['total_refresh_time_ms']:7.1f}ms  "
                f"opt={dstats_opt['total_refresh_time_ms']:7.1f}ms\n"
                f"    collect_total:   mvp={dstats_mvp['total_collect_time_ms']:7.1f}ms  "
                f"opt={dstats_opt['total_collect_time_ms']:7.1f}ms",
                flush=True,
            )

            failures = {
                name: fields[f"{name}_mismatch_token_count"]
                for name, fields in (
                    ("static", static_agreement),
                    ("mvp", mvp_agreement),
                    ("opt", opt_agreement),
                )
                if fields[f"{name}_mismatch_token_count"] > args.max_mismatch_tokens
            }
            if args.fail_on_correctness and failures:
                raise RuntimeError(
                    f"Correctness tolerance exceeded for sample {s['sample_id']}: {failures}"
                )

            del inputs, ar, static, mvp, opt, sel, dsel_mvp, dsel_opt
            torch.cuda.empty_cache()

    print(f"\nDone. -> {out_path}", flush=True)
    if completed_samples:
        print(
            f"Correctness tolerance (<= {args.max_mismatch_tokens} mismatches/sample):",
            flush=True,
        )
        for name in ("static", "mvp", "opt"):
            print(
                f"  {name}: {within_tolerance[name]}/{completed_samples} within tolerance, "
                f"total mismatches={mismatch_totals[name]}, "
                f"mean agreement={agreement_totals[name] / completed_samples:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
