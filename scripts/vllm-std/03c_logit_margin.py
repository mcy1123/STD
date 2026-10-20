#!/usr/bin/env python
"""Phase 3c: quantify the margin at the divergence point, and probe the cause.

Findings this is chasing
------------------------
1. `RepeatProposer` gets 0/1524 drafts accepted, yet its greedy output still differs
   from eager at token 62 -- reproducibly, across separate engine launches.
2. The two continuations are semantically equivalent
   ("vibrant, featuring a large screen" vs "vibrant, with a large screen"),
   which smells like a near-tie rather than a logic bug.
3. `03b_divergence_probe.py` with `logprobs=k` did NOT diverge. So either logprobs
   changes the sampling path (masking it), or the probe's shape differed.

This script therefore dumps the top-k logprob distribution around the decision point
under BOTH configurations, plus the same again with CUDA graphs disabled
(`--enforce-eager`), to separate "graph replay numerics" from "verify-kernel numerics".

Usage
-----
    $PY scripts/vllm-std/03c_logit_margin.py --eager-only
    $PY scripts/vllm-std/03c_logit_margin.py --enforce-eager
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get(
        "QWEN25VL_7B", "/mnt/local/ldh/PhyLLM/Qwen2.5-VL-7B-Instruct"))
    p.add_argument("--video", default=None)
    p.add_argument("--question", default="Describe what happens in this video in detail.")
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=70)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-mem-util", type=float, default=0.78)
    p.add_argument("--num-spec-tokens", type=int, default=4)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--probe-index", type=int, default=62)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--eager-only", action="store_true")
    p.add_argument("--out", default=os.environ.get(
        "VLLM_STD_RES", "results/vllm-std") + "/03c_logit_margin.json")
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

    messages = [[{
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": Path(video).as_uri()}},
            {"type": "text", "text": args.question},
        ],
    }]]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0,
                        logprobs=args.topk)

    common = dict(
        model=args.model, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        limit_mm_per_prompt={"video": 1, "image": 1},
        media_io_kwargs={"video": {"num_frames": args.num_frames}},
        allowed_local_media_path=str(Path(video).parent),
        enforce_eager=args.enforce_eager,
        disable_log_stats=True,
    )

    def run(spec: bool) -> dict:
        kw = dict(common)
        if spec:
            kw["speculative_config"] = {
                "method": "custom_class",
                "model": "std_vllm.proposer.RepeatProposer",
                "num_speculative_tokens": args.num_spec_tokens,
            }
        llm = LLM(**kw)
        o = llm.chat(messages, sp, use_tqdm=False)[0].outputs[0]
        toks = list(o.token_ids)
        rows = []
        for d in (o.logprobs or []):
            items = sorted(d.items(), key=lambda kv: -kv[1].logprob)
            rows.append([{"token_id": int(t), "logprob": round(float(v.logprob), 6)}
                         for t, v in items[:args.topk]])
        del llm
        import gc
        import torch
        gc.collect(); torch.cuda.empty_cache()
        return {"token_ids": toks, "topk": rows, "text": o.text}

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(args.model)

    eager = run(spec=False)
    report = {"vllm_version": vllm.__version__, "enforce_eager": args.enforce_eager,
              "num_frames": args.num_frames, "max_tokens": args.max_tokens,
              "probe_index": args.probe_index, "eager": eager}

    i = args.probe_index
    if i < len(eager["topk"]):
        top = eager["topk"][i]
        print(f"\n=== EAGER top-{args.topk} at token index {i} (enforce_eager={args.enforce_eager}) ===")
        for it in top:
            print(f"   token={it['token_id']:<8} logprob={it['logprob']:+.6f} "
                  f"{tk.decode([it['token_id']])!r}")
        if len(top) >= 2:
            margin = top[0]["logprob"] - top[1]["logprob"]
            report["eager_margin_top1_top2"] = round(margin, 6)
            print(f"   margin(top1-top2) = {margin:.6f}")

    if not args.eager_only:
        spec = run(spec=True)
        report["spec"] = spec
        div = None
        for j, (a, b) in enumerate(zip(eager["token_ids"], spec["token_ids"])):
            if a != b:
                div = j
                break
        report["divergence_index"] = div
        print(f"\n=== divergence index with logprobs enabled: {div} ===")
        if div is not None:
            print(f"   eager token {eager['token_ids'][div]} "
                  f"({tk.decode([eager['token_ids'][div]])!r}) "
                  f"vs spec token {spec['token_ids'][div]} "
                  f"({tk.decode([spec['token_ids'][div]])!r})")
            se = {it['token_id']: it['logprob'] for it in eager["topk"][div]}
            ss = {it['token_id']: it['logprob'] for it in spec["topk"][div]}
            report["eager_topk_at_div"] = eager["topk"][div]
            report["spec_topk_at_div"] = spec["topk"][div]
            print("   -- eager top-k there --")
            for it in eager["topk"][div]:
                print(f"      token={it['token_id']:<8} lp={it['logprob']:+.6f} "
                      f"{tk.decode([it['token_id']])!r}")
            print("   -- spec top-k there --")
            for it in spec["topk"][div]:
                print(f"      token={it['token_id']:<8} lp={it['logprob']:+.6f} "
                      f"{tk.decode([it['token_id']])!r}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
