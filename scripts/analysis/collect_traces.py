"""Collect dense-verification visual attention traces for the Oracle Study.

Runs the frozen STD decoder (std_generate_qwen25vl) with a read-only attention
trace collector installed, and saves per-sample:

  - {sample_id}.pt : prefill A_0 [kv_heads, visual_len] and per-round A_t
                     [num_rounds, kv_heads, visual_len] (fp32, CPU).
  - metadata.jsonl : one line per sample with scalar metadata (visual_len, k,
                     gamma, frame_num, accept_lengths, query_lens, ...).

The collector is a pure pass-through hook (attribute lookup); it never changes
logits/KV/backend, so the frozen correctness baseline is preserved.

Usage:
  # VideoDetailCaption / MLVU (local paths)
  python scripts/analysis/collect_traces.py \
    --dataset VideoDetailCaption --data-dir /mnt/local2/mcy/datasets/VideoDetailCaption \
    --frame-num 128 --max-new-tokens 256 --gamma 9 --limit 10 --gpu 0

  # Video-MME on the A100 (see docs/superpowers/plans/2026-09-17-dynamic-routing-thorough-analysis.md)
  CUDA_VISIBLE_DEVICES=1 python scripts/analysis/collect_traces.py \
    --dataset Video-MME \
    --data-path  /public/home/xlwang/mcy/STD_assets/datasets/Video-MME \
    --video-root /public/home/xlwang/mcy/STD_assets/datasets/Video-MME/videos \
    --model-path /public/home/xlwang/mcy/STD_assets/models/Qwen2.5-VL-7B-Instruct \
    --frame-num 128 --max-new-tokens 256 --gamma 9 --k-plus-text 1024 \
    --limit 20 --record-per-query --prompt-style cot \
    --output-dir /public/home/xlwang/mcy/STD_assets/results/routing_traces_videomme

`--record-per-query` keeps the verification-round query axis so that
all/two/three-query collectors can be compared offline
(`scripts/analysis/analyze_collector_ablation.py`). The prefill trace is always
summed over queries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_gpu = "0"
for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        _gpu = sys.argv[_i + 1]
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts" / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd  # noqa: E402

from benchmark_std import (  # noqa: E402
    VIDEO_TOKEN_ID,
    iter_generic_hf_video,
    load_qwen_model,
    make_qwen_video_inputs,
)
from std_repro.std_qwen25vl import std_generate_qwen25vl, set_trace_collector  # noqa: E402
from attention_trace import AttentionTraceCollector  # noqa: E402


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


def iter_videomme_samples(data_path: str, video_root: str, limit: int, prompt_style: str = "cot"):
    """Video-MME samples, matching the A100 benchmark's seed-42 ordering.

    Reuses `benchmark_std.iter_generic_hf_video` so trace collection sees the
    exact same samples, prompt style and video resolution as
    `benchmark_a100_dynamic.py`.
    """
    yield from iter_generic_hf_video(
        data_path, "test", limit, video_root, prompt_style=prompt_style
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/mnt/local2/mcy/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--dataset",
        choices=["VideoDetailCaption", "MLVU", "Video-MME"],
        default="VideoDetailCaption",
    )
    parser.add_argument(
        "--data-dir",
        default="/mnt/local2/mcy/datasets/VideoDetailCaption",
        help="VideoDetailCaption / MLVU dataset root.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Video-MME dataset root (HF dataset dir). Defaults to --data-dir.",
    )
    parser.add_argument(
        "--video-root",
        default=None,
        help="Video-MME video directory. Defaults to --data-path.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["direct", "cot"],
        default="cot",
        help="Prompt style for Video-MME (must match the benchmark).",
    )
    parser.add_argument("--frame-num", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gamma", type=int, default=9)
    parser.add_argument("--k-plus-text", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "routing_traces"))
    parser.add_argument("--resume", action="store_true", help="Skip samples whose .pt trace already exists.")
    parser.add_argument(
        "--record-per-query",
        action="store_true",
        help=(
            "Keep the query axis on verification rounds ([kv_heads, q_len, visual_len]) so that "
            "all/two/three-query collectors can be compared offline. Prefill stays summed."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"{args.dataset}_frame{args.frame_num}.jsonl"

    print(f"Loading model on GPU {_gpu} ...", flush=True)
    model, processor = load_qwen_model(args.model_path, "0")
    eos = processor.tokenizer.eos_token_id

    if args.dataset == "VideoDetailCaption":
        samples = list(iter_vdc_samples(args.data_dir, args.limit))
    elif args.dataset == "MLVU":
        samples = list(iter_mlvu_samples(args.data_dir, args.limit))
    else:
        data_path = args.data_path or args.data_dir
        video_root = args.video_root or data_path
        samples = list(
            iter_videomme_samples(data_path, video_root, args.limit, args.prompt_style)
        )
    print(f"Dataset={args.dataset}  samples={len(samples)}  frame_num={args.frame_num}", flush=True)

    with meta_path.open("w", encoding="utf-8") as meta_f:
        for idx, s in enumerate(samples):
            pt_path = out_dir / f"{s['sample_id']}.pt"
            if args.resume and pt_path.exists():
                print(f"  [{idx}] {s['sample_id']} (resume: skip existing)", flush=True)
                continue
            t0 = time.time()
            inputs = make_qwen_video_inputs(
                processor, s["video_path"], s["question"], args.frame_num,
                min_pixels=None, max_pixels=None, target_device="cuda",
            )
            prompt_ids = inputs["input_ids"][0]
            prompt_len = int(prompt_ids.numel())
            visual_positions = torch.nonzero(prompt_ids == VIDEO_TOKEN_ID, as_tuple=False).flatten().cpu()
            visual_len = int(visual_positions.numel())

            collector = AttentionTraceCollector(
                model, visual_positions, record_per_query=args.record_per_query
            )
            collector.install()
            set_trace_collector(collector)
            try:
                result, selection = std_generate_qwen25vl(
                    model, inputs, VIDEO_TOKEN_ID, eos,
                    max_new_tokens=args.max_new_tokens,
                    gamma=args.gamma,
                    target_k_plus_text=args.k_plus_text,
                )
            finally:
                set_trace_collector(None)
                collector.uninstall()

            prefill = collector.prefill_scores
            if prefill is None:
                raise RuntimeError(f"No prefill attention captured for {s['sample_id']}.")
            if len(collector.rounds) != result.decode_rounds:
                raise RuntimeError(
                    f"Round count mismatch: collector={len(collector.rounds)} result={result.decode_rounds}"
                )
            query_lens = [r.query_len for r in collector.rounds]
            if args.record_per_query:
                # q_len varies per round (pending + propose), so the rounds
                # cannot be stacked into one tensor.
                round_scores = [r.visual_scores.float() for r in collector.rounds]
            else:
                round_scores = torch.stack(
                    [r.visual_scores for r in collector.rounds], dim=0
                ).float()

            payload = {
                "prefill_scores": prefill.float(),
                "round_scores": round_scores,
                "query_lens": torch.tensor(query_lens, dtype=torch.long),
                "per_query": bool(args.record_per_query),
            }
            torch.save(payload, out_dir / f"{s['sample_id']}.pt")

            meta = {
                "dataset": args.dataset,
                "sample_id": s["sample_id"],
                "video_name": s["sample_id"],
                "frame_num": args.frame_num,
                "prompt_len": prompt_len,
                "visual_len": visual_len,
                "text_len": selection.text_len,
                "k": selection.k,
                "gamma": args.gamma,
                "max_new_tokens": args.max_new_tokens,
                "decode_rounds": result.decode_rounds,
                "generate_len": result.generate_len,
                "accept_lengths": result.accept_lengths,
                "query_lens": query_lens,
                "proposed_lengths": result.proposed_lengths,
                "pending_lengths": result.pending_lengths,
                "mean_accept_length": result.mean_accept_length,
                "per_query": bool(args.record_per_query),
                "prompt_style": args.prompt_style if args.dataset == "Video-MME" else None,
            }
            # q_len must equal pending + proposed; otherwise the offline collector
            # ablation cannot reconstruct the exact query positions.
            if len(result.pending_lengths) == len(query_lens) == len(result.proposed_lengths):
                mismatched = [
                    i
                    for i in range(len(query_lens))
                    if query_lens[i] != result.pending_lengths[i] + result.proposed_lengths[i]
                ]
                if mismatched:
                    raise RuntimeError(
                        f"query_len != pending + proposed at rounds {mismatched[:5]} "
                        f"for {s['sample_id']}; collector ablation would be invalid."
                    )
            meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            meta_f.flush()

            print(
                f"  [{idx}] {s['sample_id']}  prompt={prompt_len} visual={visual_len} k={selection.k} "
                f"rounds={result.decode_rounds} gen={result.generate_len} "
                f"mean_accept={result.mean_accept_length:.2f} ({time.time()-t0:.1f}s)",
                flush=True,
            )
            del inputs, payload, collector, round_scores, prefill, visual_positions
            torch.cuda.empty_cache()

    print(f"Done. traces -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
