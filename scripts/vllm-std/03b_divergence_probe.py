#!/usr/bin/env python
"""Phase 3b: locate the first divergent token and compare the LOGITS there.

The question this answers
-------------------------
With `RepeatProposer`, vLLM drafted 381 groups / 1524 tokens and the target model
accepted **zero** of them -- i.e. every single draft was thrown away and the target
model chose every emitted token itself. Naively the output should be identical to
the eager baseline. It is not: it first differs at token 62.

So the divergence cannot come from a wrong token being *accepted*. Either
  (a) the speculative verification forward pass produces slightly different logits
      than the eager single-token decode (floating-point / kernel-shape effect), and
      the argmax flips on a near-tie, or
  (b) there is a real logic bug in the draft/verify bookkeeping.
This probe distinguishes them by printing the top-k logprobs at the divergence
position for both configurations. A near-tie (small margin) supports (a); a large
margin with the eager token still on top would support (b).

Usage
-----
    $PY scripts/vllm-std/03b_divergence_probe.py --divergence-index 62
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
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-mem-util", type=float, default=0.78)
    p.add_argument("--num-spec-tokens", type=int, default=4)
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--out", default=os.environ.get(
        "VLLM_STD_RES", "results/vllm-std") + "/03b_divergence.json")
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
        disable_log_stats=False,
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
        out = llm.chat(messages, sp, use_tqdm=False)[0]
        o = out.outputs[0]
        # logprobs[i] is a dict {token_id: Logprob} for position i
        lp = []
        for d in (o.logprobs or []):
            items = sorted(d.items(), key=lambda kv: -kv[1].logprob)
            lp.append([{"token_id": int(t), "logprob": float(v.logprob),
                        "rank": int(v.rank)} for t, v in items])
        res = {"token_ids": list(o.token_ids), "logprobs": lp, "text": o.text}
        del llm
        import gc
        import torch
        gc.collect(); torch.cuda.empty_cache()
        return res

    print("running eager baseline ...", flush=True)
    base = run(spec=False)
    print("running RepeatProposer k=%d ..." % args.num_spec_tokens, flush=True)
    spec = run(spec=True)

    div = None
    for i, (a, b) in enumerate(zip(base["token_ids"], spec["token_ids"])):
        if a != b:
            div = i
            break
    if div is None and len(base["token_ids"]) != len(spec["token_ids"]):
        div = min(len(base["token_ids"]), len(spec["token_ids"]))

    report = {
        "vllm_version": vllm.__version__,
        "video": video, "num_frames": args.num_frames,
        "max_tokens": args.max_tokens, "num_spec_tokens": args.num_spec_tokens,
        "divergence_index": div,
        "eager_token_ids": base["token_ids"],
        "spec_token_ids": spec["token_ids"],
        "eager_text": base["text"], "spec_text": spec["text"],
    }

    if div is None:
        report["conclusion"] = "BIT_EXACT: no divergence"
        print("\nNo divergence.")
    else:
        report["eager_logprobs_at_div"] = base["logprobs"][div]
        report["spec_logprobs_at_div"] = spec["logprobs"][div]
        e_top = base["logprobs"][div]
        s_top = spec["logprobs"][div]
        print(f"\n=== divergence at token index {div} ===")
        print(f"eager chose {base['token_ids'][div]}, spec chose {spec['token_ids'][div]}")
        print("\n-- eager top-k at that position --")
        for it in e_top[:args.topk]:
            print(f"   token={it['token_id']:<8} logprob={it['logprob']:+.6f} rank={it['rank']}")
        print("-- speculative top-k at that position --")
        for it in s_top[:args.topk]:
            print(f"   token={it['token_id']:<8} logprob={it['logprob']:+.6f} rank={it['rank']}")
        # Margin between the two competing tokens under each configuration.
        def lp_of(tops, tok):
            for it in tops:
                if it["token_id"] == tok:
                    return it["logprob"]
            return None
        e_a, e_b = lp_of(e_top, base["token_ids"][div]), lp_of(e_top, spec["token_ids"][div])
        s_a, s_b = lp_of(s_top, base["token_ids"][div]), lp_of(s_top, spec["token_ids"][div])
        report["margins"] = {"eager": None if None in (e_a, e_b) else e_a - e_b,
                             "spec": None if None in (s_a, s_b) else s_a - s_b}
        print(f"\neager margin(its top1 - the other token) = "
              f"{None if None in (e_a, e_b) else round(e_a - e_b, 6)}")
        print(f"spec  margin(its top1 - the other token) = "
              f"{None if None in (s_a, s_b) else round(s_a - s_b, 6)}")
        report["conclusion"] = (
            "near-tie at divergence: consistent with floating-point differences in the "
            "verification forward pass rather than a bookkeeping bug"
            if (report["margins"]["eager"] is not None
                and abs(report["margins"]["eager"]) < 0.5) else
            "large margin: investigate bookkeeping")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
