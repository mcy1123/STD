#!/usr/bin/env python
"""Phase 1 baseline: pure video / multi-frame inference with Qwen2.5-VL on vLLM.

What this establishes
---------------------
A *reference* greedy decode on a video input. Every later phase (dummy proposer,
STD single-token, STD multi-token, batch sweep) is judged against the token ids
produced here, so this script's output is the bit-exactness oracle.

Design notes
------------
* Greedy only (temperature=0). Any sampling randomness would make the
  cross-configuration token-id comparison meaningless.
* The first `--warmup` calls are untimed: they absorb CUDA-graph capture, kernel
  autotuning and lazy weight load. Timed runs start only after that.
* We assert the video was actually expanded into vision tokens. vLLM silently
  degrades to a text-only prompt if the multimodal payload is not recognised,
  which would make a "working" baseline that never touched a single frame.
* Every run is appended to a JSONL so a rerun never overwrites evidence.

Usage
-----
    source scripts/vllm-std/env.sh
    CUDA_VISIBLE_DEVICES=2 $PY scripts/vllm-std/01_baseline_video.py \
        --num-frames 32 --max-tokens 128 --runs 3 --warmup 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gpu_state() -> dict:
    """Snapshot the GPU so a contended run can never be misread as an algorithm effect."""
    try:
        q = "index,name,memory.total,memory.used,memory.free,utilization.gpu"
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip()
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        return {"visible": vis, "all_gpus": out.splitlines()}
    except Exception as e:  # never let telemetry break the experiment
        return {"error": f"{type(e).__name__}: {e}"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get(
        "QWEN25VL_7B", "/mnt/local/ldh/PhyLLM/Qwen2.5-VL-7B-Instruct"))
    p.add_argument("--video", default=None,
                   help="video path; defaults to the first Video-MME mp4")
    p.add_argument("--question", default="Describe what happens in this video in detail.")
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--fps", type=float, default=None,
                   help="if set, sample by fps instead of a fixed frame count")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=65536)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--runs", type=int, default=3, help="timed repetitions")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--allowed-local-media-path", default=None,
                   help="root dir the media connector may read video files from; "
                        "defaults to the video's own directory")
    p.add_argument("--out", default=os.environ.get(
        "VLLM_STD_RES", "results/vllm-std") + "/01_baseline_video.jsonl")
    p.add_argument("--tag", default="baseline")
    args = p.parse_args()

    video = args.video
    if video is None:
        root = Path(os.environ.get("VIDEOMME", "datasets/Video-MME")) / "videos"
        vids = sorted(root.rglob("*.mp4"))
        if not vids:
            print(f"FATAL: no .mp4 under {root}", file=sys.stderr)
            return 2
        video = str(vids[0])
    if not Path(video).is_file():
        print(f"FATAL: video not found: {video}", file=sys.stderr)
        return 2

    # media_io_kwargs is how vLLM's VideoMediaIO receives frame sampling controls.
    media_io_kwargs = {"video": {"num_frames": args.num_frames}}
    if args.fps is not None:
        media_io_kwargs = {"video": {"fps": args.fps}}

    from vllm import LLM, SamplingParams
    import vllm
    import torch

    # Local-file video only loads through the *chat* renderer's media connector,
    # and that connector refuses any file outside `allowed_local_media_path`
    # (vllm/multimodal/media/connector.py:_load_file_url). Pass the directory.
    allowed_root = args.allowed_local_media_path
    if allowed_root is None:
        allowed_root = str(Path(video).resolve().parent)
    if Path(video).resolve().parent != Path(allowed_root).resolve() and \
            Path(allowed_root).resolve() not in Path(video).resolve().parents:
        print(f"FATAL: {video} is not under allowed root {allowed_root}",
              file=sys.stderr)
        return 2

    rec = {
        "tag": args.tag,
        "phase": 1,
        "timestamp": utc(),
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "model": args.model,
        "video": video,
        "question": args.question,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "media_io_kwargs": media_io_kwargs,
        "allowed_local_media_path": allowed_root,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "gpu_mem_util": args.gpu_mem_util,
        "enforce_eager": args.enforce_eager,
        "gpu_before": gpu_state(),
    }

    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        limit_mm_per_prompt={"video": 1, "image": 1},
        media_io_kwargs=media_io_kwargs,
        allowed_local_media_path=allowed_root,
        enforce_eager=args.enforce_eager,
        disable_log_stats=False,
    )
    rec["load_seconds"] = round(time.perf_counter() - t0, 3)

    # The OpenAI-style `video_url` part is what the chat renderer understands;
    # a bare `{"type": "video", "video": <path>}` part is NOT resolved and would
    # reach the processor as a raw string (and get iterated char-by-char).
    messages = [{
        "role": "user",
        "content": [
            {"type": "video_url",
             "video_url": {"url": Path(video).resolve().as_uri()}},
            {"type": "text", "text": args.question},
        ],
    }]
    rec["messages"] = messages

    # temperature=0 -> greedy. No seed needed, but pin it anyway.
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)

    def once() -> dict:
        t = time.perf_counter()
        outs = llm.chat(messages, sp, use_tqdm=False)
        dt = time.perf_counter() - t
        o = outs[0]
        return {
            "seconds": round(dt, 4),
            "text": o.outputs[0].text,
            "token_ids": list(o.outputs[0].token_ids),
            "num_prompt_tokens": len(o.prompt_token_ids or []),
            "num_output_tokens": len(o.outputs[0].token_ids),
            "finish_reason": o.outputs[0].finish_reason,
        }

    # Probe run: always executed, never timed. It absorbs first-call laziness and
    # gives us a sample to validate that the video really became vision tokens.
    probe = once()
    rec["probe"] = {"num_prompt_tokens": probe["num_prompt_tokens"],
                    "num_output_tokens": probe["num_output_tokens"],
                    "seconds": probe["seconds"]}

    # Guard: did the video actually expand into vision tokens? A text-only prompt
    # for this question is ~30 tokens, so require a prompt that clearly scales with
    # the frame budget. Relative threshold => valid for any --num-frames.
    npt = probe["num_prompt_tokens"]
    floor = 100 + 20 * args.num_frames
    if npt < floor:
        rec["status"] = "FAIL_VIDEO_NOT_EXPANDED"
        rec["guard"] = {"num_prompt_tokens": npt, "required_at_least": floor}
        rec["gpu_after"] = gpu_state()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"FATAL: prompt has only {npt} tokens (< {floor} required for "
              f"{args.num_frames} frames); the video was probably dropped. "
              f"Recorded to {args.out}", file=sys.stderr)
        return 3

    for _ in range(args.warmup):
        w = once()
        rec.setdefault("warmup", []).append(
            {"num_prompt_tokens": w["num_prompt_tokens"],
             "num_output_tokens": w["num_output_tokens"],
             "seconds": w["seconds"]})

    runs = [once() for _ in range(args.runs)]
    rec["runs"] = runs
    rec["output_text"] = runs[0]["text"]
    rec["token_ids"] = runs[0]["token_ids"]
    rec["num_prompt_tokens"] = runs[0]["num_prompt_tokens"]
    rec["num_output_tokens"] = runs[0]["num_output_tokens"]

    secs = [r["seconds"] for r in runs]
    rec["seconds_mean"] = round(sum(secs) / len(secs), 4)
    rec["seconds_std"] = round(
        (sum((s - rec["seconds_mean"]) ** 2 for s in secs) / len(secs)) ** 0.5, 4)
    rec["seconds_min"] = round(min(secs), 4)
    rec["seconds_max"] = round(max(secs), 4)

    # Stability: greedy decode must be bit-identical across repetitions.
    ref = runs[0]["token_ids"]
    rec["bit_exact_across_runs"] = all(r["token_ids"] == ref for r in runs)
    rec["gpu_after"] = gpu_state()
    rec["status"] = "OK" if rec["bit_exact_across_runs"] else "UNSTABLE"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(json.dumps({
        "status": rec["status"],
        "num_prompt_tokens": rec["num_prompt_tokens"],
        "num_output_tokens": rec["num_output_tokens"],
        "seconds_mean": rec["seconds_mean"],
        "seconds_std": rec["seconds_std"],
        "bit_exact_across_runs": rec["bit_exact_across_runs"],
        "token_ids_head": rec["token_ids"][:16],
        "text_head": rec["output_text"][:300],
        "out": args.out,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
