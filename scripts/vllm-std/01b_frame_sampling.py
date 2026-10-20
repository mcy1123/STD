#!/usr/bin/env python
"""Phase 1b: verify vLLM's video frame-sampling controls actually take effect.

Phase 1 must show that `num_frames` / `fps` / resolution behave sanely, not just
that *some* video prompt was accepted. The cheapest honest check is end-to-end:
one engine per configuration, then compare the prompt token count the engine
actually built. If a knob were silently ignored, the token counts would not move.

`media_io_kwargs` is fixed at engine construction, so each configuration needs its
own engine -- hence the outer loop over configs rather than over requests.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get(
        "QWEN25VL_7B", "/mnt/local/ldh/PhyLLM/Qwen2.5-VL-7B-Instruct"))
    p.add_argument("--video", default=None)
    p.add_argument("--question", default="Describe what happens in this video.")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=65536)
    p.add_argument("--gpu-mem-util", type=float, default=0.80)
    p.add_argument("--num-frames-list", default="4,8,16,32")
    p.add_argument("--fps-list", default="1,2,4")
    p.add_argument("--configs-json", default="",
                   help="raw media_io_kwargs JSON objects joined by '||'")
    p.add_argument("--out", default=os.environ.get(
        "VLLM_STD_RES", "results/vllm-std") + "/01b_frame_sampling.jsonl")
    args = p.parse_args()

    video = args.video
    if video is None:
        root = Path(os.environ.get("VIDEOMME", "datasets/Video-MME")) / "videos"
        vids = sorted(root.rglob("*.mp4"))
        if not vids:
            print(f"FATAL: no .mp4 under {root}", file=sys.stderr)
            return 2
        video = str(vids[0])
    video = str(Path(video).resolve())

    from vllm import LLM, SamplingParams
    import vllm
    import torch

    messages = [{
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": Path(video).as_uri()}},
            {"type": "text", "text": args.question},
        ],
    }]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)

    configs: list[dict] = []
    for n in (int(x) for x in args.num_frames_list.split(",") if x.strip()):
        configs.append({"kind": "num_frames", "media_io_kwargs": {"video": {"num_frames": n}}})
    for f in (float(x) for x in args.fps_list.split(",") if x.strip()):
        configs.append({"kind": "fps", "media_io_kwargs": {"video": {"fps": f}}})
    # Raw configs, for probing interactions the named flags cannot express
    # (e.g. fps together with an explicit num_frames sentinel).
    for raw in (x for x in args.configs_json.split("||") if x.strip()):
        configs.append({"kind": "raw", "media_io_kwargs": json.loads(raw)})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for cfg in configs:
        rec = {
            "phase": "1b",
            "timestamp": utc(),
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "video": video,
            "config": cfg["kind"],
            "media_io_kwargs": cfg["media_io_kwargs"],
            "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        t0 = time.perf_counter()
        try:
            llm = LLM(
                model=args.model,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_mem_util,
                limit_mm_per_prompt={"video": 1, "image": 1},
                media_io_kwargs=cfg["media_io_kwargs"],
                allowed_local_media_path=str(Path(video).parent),
                disable_log_stats=True,
            )
            rec["load_seconds"] = round(time.perf_counter() - t0, 2)
            out = llm.chat(messages, sp, use_tqdm=False)[0]
            rec["num_prompt_tokens"] = len(out.prompt_token_ids or [])
            rec["num_output_tokens"] = len(out.outputs[0].token_ids)
            rec["text_head"] = out.outputs[0].text[:160]
            rec["status"] = "OK"
            del llm
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            rec["status"] = "ERROR"
            rec["error"] = f"{type(e).__name__}: {e}"
        with open(out_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        results.append(rec)
        print(json.dumps({k: rec.get(k) for k in
                          ("config", "media_io_kwargs", "num_prompt_tokens",
                           "status", "error")}, ensure_ascii=False), flush=True)

    print("\n=== summary ===")
    for r in results:
        if r["config"] == "num_frames":
            print(f"  num_frames={r['media_io_kwargs']['video']['num_frames']:>3} "
                  f"-> prompt_tokens={r.get('num_prompt_tokens')}")
    for r in results:
        if r["config"] == "fps":
            print(f"  fps={r['media_io_kwargs']['video']['fps']:>4} "
                  f"-> prompt_tokens={r.get('num_prompt_tokens')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
