#!/usr/bin/env python
"""CSV-0: measure the certificate ceiling and the middle-pass cost on one GPU.

Answers the two questions that decide whether a certified sparse verification
layer is worth building (see `src/std_repro/certificate_probe.py` and
`docs/superpowers/plans/2026-09-19-hsd-three-level-verification.md`):

  1. how often does a static top-K sparse pass agree with the dense pass at the
     same position, and can a statistic computable from the sparse pass alone
     find those positions without emitting wrong tokens;
  2. how expensive is one batched sparse pass over the draft block relative to
     the dense pass it would let us skip -- for the existing per-layer mask
     construction and for a variant that builds the mask once per forward.

The runner also verifies the probe's decode against greedy AR in the same
process, so a broken probe cannot look like a certificate result.

Usage:
  python scripts/benchmark_certificate.py \
    --model-path ... --data-path ... --video-root ... --output results/csv0.jsonl
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--frame-num", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--gamma", type=int, default=9)
    parser.add_argument("--target-k-plus-text", type=int, default=1024)
    parser.add_argument("--sparse-attn-mode", choices=("gqa_sdpa", "triton_gqa"), default="gqa_sdpa")
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--cache-len", type=int, default=40960)
    parser.add_argument("--allow-shared-gpu", action="store_true",
                        help="permit a GPU that other processes already occupy, if enough memory is free")
    parser.add_argument("--min-free-gib", type=float, default=40.0)
    parser.add_argument("--max-rss-gib", type=float, default=48.0)
    args = parser.parse_args(argv)
    if min(args.frame_num, args.max_new_tokens, args.limit, args.gamma) < 1:
        parser.error("frame, token, sample and gamma counts must be positive")
    if args.gamma < 2:
        parser.error("gamma must be at least 2 to form a verification block")
    if args.skip_samples < 0 or args.target_k_plus_text < 1 or args.cache_len < 1:
        parser.error("invalid skip, target-k or cache-len")
    if args.gpu < 0:
        parser.error("gpu must be non-negative")
    return args


def gpu_rows(snapshot: str) -> list[list[str]]:
    rows = []
    for line in snapshot.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0].isdigit():
            rows.append(fields)
    return rows


def require_usable_gpu(snapshot: str, gpu: int, *, allow_shared: bool, min_free_gib: float) -> None:
    """Refuse to start unless the selected GPU is idle, or free enough by request.

    The default mirrors the repository's rule: never share a busy GPU.  Timing is
    part of this probe, so sharing is opt-in and the free-memory floor still has
    to be met.
    """
    selected = None
    for fields in gpu_rows(snapshot):
        if int(fields[0]) == gpu:
            selected = fields
            break
    if selected is None:
        raise RuntimeError(f"selected GPU {gpu} is not present in the nvidia-smi snapshot")
    used_mib = int(float(selected[3]))
    free_mib = int(float(selected[4]))
    if used_mib != 0 and not allow_shared:
        raise RuntimeError(
            f"GPU {gpu} is not idle ({used_mib} MiB used); pass --allow-shared-gpu to share it explicitly"
        )
    if free_mib < min_free_gib * 1024:
        raise RuntimeError(
            f"GPU {gpu} has only {free_mib / 1024:.1f} GiB free, below the {min_free_gib:.1f} GiB floor"
        )


def _snapshot() -> str:
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def render_report(aggregate: Mapping[str, Any], *, sample_ids: Sequence[str], args: argparse.Namespace) -> str:
    """Cost + certificate report, ending in the gate the plan defines."""
    timing = aggregate["timing"]
    rounds = max(1, int(aggregate["rounds"]))
    sparse_ms = timing["sparse_pass_seconds"] / rounds * 1000.0
    cached_fwd_ms = timing["cached_sparse_pass_seconds"] / rounds * 1000.0
    mask_prepare_ms = timing["mask_prepare_seconds"] / rounds * 1000.0
    cached_ms = cached_fwd_ms + mask_prepare_ms
    dense_ms = timing["dense_pass_seconds"] / rounds * 1000.0
    dense_bonus_ms = timing["dense_bonus_seconds"] / rounds * 1000.0
    combined_bonus_ms = timing["bonus_seconds"] / rounds * 1000.0
    draft_ms = timing["draft_seconds"] / rounds * 1000.0
    per_layer_mask_ms = timing["per_layer_mask_build_seconds"] / rounds * 1000.0
    microbench_ms = timing["mask_build_microbench_seconds"] / max(1, len(sample_ids)) * 1000.0

    # Reuse the plan's gate so the verdict is not re-derived ad hoc here.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))
    import hsd_feasibility as hf  # noqa: E402

    baseline = {
        "verify_time_ms_per_round": dense_ms,
        "bonus_time_ms_per_round": dense_bonus_ms,
    }
    print_sparse_gate = hf.certification_gate(baseline, sparse_ms)
    cached_gate = hf.certification_gate(baseline, cached_ms)
    strict = aggregate["strict_operating_point"]
    risk = aggregate["risk_operating_point"]

    lines = [
        "# CSV-0 certificate probe",
        "",
        f"Config: {args.frame_num} frames, {args.max_new_tokens} tokens, gamma={args.gamma}, "
        f"K+text={args.target_k_plus_text}, {len(sample_ids)} sample(s): {', '.join(sample_ids)}",
        "",
        "## A. Correctness ceiling (paired sparse vs dense)",
        "",
        f"- verified positions: **{aggregate['positions']}** across **{aggregate['rounds']}** rounds",
        f"- position-level sparse/dense agreement: **{aggregate['position_agreement'] * 100:.2f}%**",
        f"- rounds where every position agreed (the certificate ceiling): "
        f"**{aggregate['round_ceiling'] * 100:.2f}%**",
        f"- rounds where the mask variants disagreed on logits: **{aggregate['logit_mismatch_rounds']}** "
        f"(max |delta| = {aggregate.get('max_logit_delta', 0.0):.3e}; the cached mask is an optimisation, "
        "not an approximation, so this must stay 0)",
        "",
    ]
    if strict:
        lines += [
            f"- strict operating point (precision 1.00): margin >= {strict['threshold']:.3f} certifies "
            f"**{strict['coverage'] * 100:.2f}%** of rounds with **{strict['wrong_skips']}** wrong skips",
        ]
    else:
        lines.append("- strict operating point: **none** -- no margin threshold certifies any round without error")
    if risk:
        lines.append(
            f"- 0.95-precision operating point: margin >= {risk['threshold']:.3f} certifies "
            f"**{risk['coverage'] * 100:.2f}%** of rounds ({risk['wrong_skips']} wrong skips)"
        )
    lines.append("")
    lines.append("Selected curve rows (min-margin threshold -> coverage / precision):")
    lines.append("")
    lines.append("| margin >= | certified rounds | coverage | precision | wrong skips |")
    lines.append("|---:|---:|---:|---:|---:|")
    curve = aggregate["curve"]
    if curve:
        step = max(1, len(curve) // 8)
        for row in curve[::step]:
            lines.append(
                f"| {row['threshold']:.3f} | {row['certified_rounds']} | {row['coverage'] * 100:.1f}% | "
                f"{row['precision'] * 100:.1f}% | {row['wrong_skips']} |"
            )
    lines.append("")
    lines.append("## B. Middle-pass cost (ms per round)")
    lines.append("")
    lines.append("| stage | ms/round | note |")
    lines.append("|---|---:|---|")
    lines.append(f"| sparse draft ({args.gamma} sequential passes) | {draft_ms:.1f} | needed only to obtain the block |")
    lines.append(f"| **sparse batched pass (mask rebuilt per layer)** | **{sparse_ms:.1f}** | existing implementation; all-inclusive |")
    lines.append(f"| sparse batched pass (mask built once) | {cached_fwd_ms:.1f} | forward only |")
    lines.append(f"| + one mask build per round | {mask_prepare_ms:.1f} | {args.gamma}x1 mask |")
    lines.append(f"| **= mask-cached variant, all-inclusive** | **{cached_ms:.1f}** | this probe's variant |")
    lines.append(f"| dense pass (what a certificate would skip) | {dense_ms:.1f} | authoritative |")
    lines.append(f"| bonus step, dense share | {dense_bonus_ms:.1f} | a skipped round avoids this; the sparse bonus still runs (combined {combined_bonus_ms:.1f}) |")
    lines.append("")
    lines.append(f"- mask builds: per-layer variant {aggregate.get('per_layer_mask_builds', 0)} "
                 f"({per_layer_mask_ms:.1f} ms of CPU launch time), cached variant "
                 f"{aggregate.get('cached_mask_hits', 0)} hits / {aggregate.get('cached_mask_misses', 0)} misses "
                 f"(0 misses means the cached mask was really reused)")
    lines.append(f"- standalone mask-build microbenchmark: {microbench_ms:.3f} ms "
                 "(CPU launch time, no sync; diagnostic only)")
    lines.append("")
    lines.append(f"- existing middle pass / dense pass = **{sparse_ms / dense_ms:.2f}x**")
    lines.append(f"- mask-cached middle pass / dense pass = **{cached_ms / dense_ms:.2f}x**")
    lines.append("")
    lines.append("## C. Gate (from `hsd_feasibility.py`)")
    lines.append("")
    lines.append(f"- saving when a round skips dense: **{print_sparse_gate['saving_ms_per_round']:.1f} ms/round**")
    lines.append(f"- existing middle pass: {print_sparse_gate['verdict']}")
    lines.append(f"- cached-mask middle pass: {cached_gate['verdict']}")
    lines.append("")
    lines.append("Compare the required rate against the ceiling in section A. A requirement above the ceiling")
    lines.append("cannot be met by any certificate, because no rule computable from the sparse pass alone can")
    lines.append("certify a round whose sparse and dense argmax differ.")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["SPECVLM_MAX_CACHE_LEN"] = str(args.cache_len)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    import torch
    from std_repro.streaming_video import install_streaming_video_reader
    from benchmark_std import VIDEO_TOKEN_ID, iter_generic_hf_video, load_qwen_model, make_qwen_video_inputs
    from std_repro.std_qwen25vl import ar_generate_qwen25vl
    from std_repro.certificate_probe import aggregate, certificate_probe_qwen25vl

    install_streaming_video_reader()
    snapshot = _snapshot()
    require_usable_gpu(snapshot, args.gpu, allow_shared=args.allow_shared_gpu, min_free_gib=args.min_free_gib)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one visible CUDA device")

    samples = list(iter_generic_hf_video(
        args.data_path, "test", args.skip_samples + args.limit, args.video_root, prompt_style="cot"
    ))[args.skip_samples:]
    if len(samples) != args.limit:
        raise RuntimeError(f"requested {args.limit} samples, found {len(samples)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_paths = [
        "scripts/benchmark_certificate.py", "scripts/benchmark_std.py",
        "src/std_repro/certificate_probe.py", "src/std_repro/sparse_verify_spike.py",
        "src/std_repro/std_qwen25vl.py", "scripts/analysis/hsd_feasibility.py",
    ]
    hashes = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in source_paths}

    stats_by_sample = []
    sample_ids = []
    with args.output.open("x", encoding="utf-8") as handle:
        def emit(item: Mapping[str, Any]) -> None:
            handle.write(json.dumps(_jsonable(dict(item)), ensure_ascii=False) + "\n")
            handle.flush()

        emit({
            "kind": "manifest", "label": "csv0-certificate-probe",
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "physical_gpu": args.gpu,
            "gpu_snapshot": snapshot, "dtype": "float16", "torch": torch.__version__,
            "cuda": torch.version.cuda, "source_sha256": hashes, "args": vars(args),
            "sample_ids": [sample["sample_id"] for sample in samples],
            "timing_note": "shared GPU is opt-in; timing is only meaningful if the GPU stayed quiet",
        })
        print("Loading target model ...", flush=True)
        model, processor = load_qwen_model(args.model_path, "0")
        eos = processor.tokenizer.eos_token_id
        for sample in samples:
            inputs, prep = make_qwen_video_inputs(
                processor, sample["video_path"], sample["question"], args.frame_num, None,
                args.max_pixels, target_device="cuda:0", return_timings=True,
            )
            prompt_len = int(inputs["input_ids"].shape[1])
            if prompt_len + args.max_new_tokens + args.gamma + 2 > args.cache_len:
                raise RuntimeError("prompt and generation exceed cache capacity")
            emit({"kind": "input", "sample_id": sample["sample_id"], **prep})

            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            reference = ar_generate_qwen25vl(
                model, {k: v.clone() if hasattr(v, "clone") else v for k, v in inputs.items()},
                VIDEO_TOKEN_ID, eos, max_new_tokens=args.max_new_tokens, ignore_eos=True,
            )
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            result, stats = certificate_probe_qwen25vl(
                model, inputs, VIDEO_TOKEN_ID, eos,
                max_new_tokens=args.max_new_tokens, gamma=args.gamma,
                target_k_plus_text=args.target_k_plus_text, sparse_attn_mode=args.sparse_attn_mode,
            )
            torch.cuda.synchronize()
            probe_tokens = result.output_ids[0, prompt_len:].tolist()
            reference_tokens = reference.output_ids[0, prompt_len:].tolist()
            stats["sample_id"] = sample["sample_id"]
            stats["probe_matches_ar"] = probe_tokens == reference_tokens
            stats["probe_token_count"] = len(probe_tokens)
            stats["probe_tokens"] = probe_tokens
            stats["reference_tokens"] = reference_tokens
            stats["first_mismatch_index"] = next(
                (index for index, (left, right) in enumerate(zip(probe_tokens, reference_tokens)) if left != right),
                None if len(probe_tokens) == len(reference_tokens)
                else min(len(probe_tokens), len(reference_tokens)),
            )
            stats_by_sample.append(stats)
            sample_ids.append(sample["sample_id"])
            emit({"kind": "certificate", "sample_id": sample["sample_id"], "stats": _jsonable(stats)})
            print(f"  {sample['sample_id']}: rounds={len(stats['rounds'])} matches_ar={stats['probe_matches_ar']}",
                  flush=True)
            del inputs, reference, result
            gc.collect(); torch.cuda.empty_cache()

        pooled = aggregate(stats_by_sample)
        mismatches = [s["sample_id"] for s in stats_by_sample if not s["probe_matches_ar"]]
        pooled["probe_mismatch_samples"] = mismatches
        emit({"kind": "aggregate", **pooled})
        report = render_report(pooled, sample_ids=sample_ids, args=args)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report + "\n")
        emit({"kind": "report", "markdown": report})
        print(report, flush=True)
    if mismatches:
        raise SystemExit(f"probe decode did not match greedy AR for: {mismatches}")
    print(f"COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    main()
