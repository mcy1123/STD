#!/usr/bin/env python
"""General speculative-decoding benchmark for the STD-on-vLLM workstream.

Runs the same video prompt(s) through vLLM either eagerly (no speculation) or with
a `custom_class` proposer, and records everything needed to judge it:

  * per-run wall time,
  * per-request greedy token ids (the bit-exactness oracle),
  * vLLM's own spec-decode counters, snapshotted per run.

Deliberately configuration-agnostic: phases 3-6 all drive this one script so that
"baseline", "dummy" and "STD" numbers come from identical code paths and identical
inputs. Any difference is then attributable to the proposer alone.

Usage
-----
    # baseline (no speculation)
    $PY scripts/vllm-std/03_bench_spec.py --tag base_k0

    # dummy proposer, k=4
    $PY scripts/vllm-std/03_bench_spec.py \
        --proposer std_vllm.proposer.EmptyProposer --num-spec-tokens 4 --tag dummy_k4
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
    try:
        q = "index,name,memory.total,memory.used,memory.free,utilization.gpu"
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
        return {"visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "all_gpus": out.strip().splitlines()}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def snapshot_metrics(llm) -> dict:
    """Flatten vLLM's Prometheus snapshot into {name: value}.

    Spec-decode counters are cumulative for the engine's lifetime, so callers diff
    consecutive snapshots to get per-run figures.
    """
    out: dict[str, object] = {}
    try:
        metrics = llm.get_metrics()
    except Exception as e:
        return {"__error__": f"{type(e).__name__}: {e}"}
    for m in metrics:
        name = getattr(m, "name", None)
        if name is None:
            continue
        if hasattr(m, "values"):
            out[name] = list(m.values)
        elif hasattr(m, "value"):
            out[name] = m.value
        else:
            out[name] = None
    return out


SPEC_KEYS = [
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
    "vllm:spec_decode_num_accepted_tokens_per_pos",
]


def spec_delta(after: dict, before: dict) -> dict:
    d: dict[str, object] = {}
    for k in SPEC_KEYS:
        av, bv = after.get(k), before.get(k)
        if isinstance(av, list) and isinstance(bv, list):
            d[k] = [x - y for x, y in zip(av, bv)]
        elif isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            d[k] = av - bv
        else:
            d[k] = None
    return d


def acceptance(d: dict) -> dict:
    """Turn raw counters into acceptance figures.

    accept_rate  = accepted / drafted      (fraction of drafted tokens the target kept)
    accept_len   = accepted / drafts + 1   (mean tokens emitted per verification round)
    """
    drafted = d.get("vllm:spec_decode_num_draft_tokens")
    accepted = d.get("vllm:spec_decode_num_accepted_tokens")
    drafts = d.get("vllm:spec_decode_num_drafts")
    out = {"drafted": drafted, "accepted": accepted, "drafts": drafts}
    if isinstance(drafted, (int, float)) and drafted > 0:
        out["accept_rate"] = round(accepted / drafted, 4)
    if isinstance(drafts, (int, float)) and drafts > 0:
        out["accept_len"] = round(accepted / drafts + 1, 4)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get(
        "QWEN25VL_7B", "/mnt/local/ldh/PhyLLM/Qwen2.5-VL-7B-Instruct"))
    p.add_argument("--proposer", default=None,
                   help="dotted class path; omit for an eager (non-speculative) run")
    p.add_argument("--num-spec-tokens", type=int, default=4)
    p.add_argument("--videos", default=None,
                   help="comma-separated mp4 paths; default = first Video-MME clip")
    p.add_argument("--num-videos", type=int, default=1,
                   help="take the first N Video-MME clips when --videos is unset")
    p.add_argument("--question", default="Describe what happens in this video in detail.")
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=32768,
                   help="keep modest: KV for 65536 tokens alone needs 3.5 GiB, "
                        "which will not fit beside a 15.7 GiB model on a shared card")
    p.add_argument("--gpu-mem-util", type=float, default=0.80)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--tag", required=True)
    p.add_argument("--out", default=os.environ.get(
        "VLLM_STD_RES", "results/vllm-std") + "/spec_bench.jsonl")
    args = p.parse_args()

    # ---- resolve inputs -----------------------------------------------------
    if args.videos:
        videos = [str(Path(v).resolve()) for v in args.videos.split(",") if v.strip()]
    else:
        root = Path(os.environ.get("VIDEOMME", "datasets/Video-MME")) / "videos"
        found = sorted(root.rglob("*.mp4"))[: args.num_videos]
        if not found:
            print(f"FATAL: no .mp4 under {root}", file=sys.stderr)
            return 2
        videos = [str(v.resolve()) for v in found]
    for v in videos:
        if not Path(v).is_file():
            print(f"FATAL: video not found: {v}", file=sys.stderr)
            return 2
    common_root = Path(os.path.commonpath([str(Path(v).parent) for v in videos]))

    from vllm import LLM, SamplingParams
    import vllm
    import torch

    rec: dict = {
        "tag": args.tag,
        "timestamp": utc(),
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "model": args.model,
        "videos": videos,
        "batch_size": len(videos),
        "question": args.question,
        "num_frames": args.num_frames,
        "max_tokens": args.max_tokens,
        "gpu_mem_util": args.gpu_mem_util,
        "enforce_eager": args.enforce_eager,
        "proposer": args.proposer,
        "num_spec_tokens": args.num_spec_tokens if args.proposer else 0,
        "gpu_before": gpu_state(),
    }

    llm_kwargs: dict = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        limit_mm_per_prompt={"video": 1, "image": 1},
        media_io_kwargs={"video": {"num_frames": args.num_frames}},
        allowed_local_media_path=str(common_root),
        enforce_eager=args.enforce_eager,
        disable_log_stats=False,
    )
    if args.proposer:
        llm_kwargs["speculative_config"] = {
            "method": "custom_class",
            "model": args.proposer,
            "num_speculative_tokens": args.num_spec_tokens,
        }
        # Speculative decoding needs the draft/verify path to be graph-friendly;
        # padded drafter batch is the default and supported for custom_class.
    else:
        rec["speculative_config"] = None

    rec["llm_kwargs"] = {k: v for k, v in llm_kwargs.items()
                         if k not in ("model", "allowed_local_media_path")}

    from std_vllm.engine_guard import guard_engine, shutdown_all

    t0 = time.perf_counter()
    llm = guard_engine(LLM(**llm_kwargs))
    rec["load_seconds"] = round(time.perf_counter() - t0, 3)

    messages = [[
        {"role": "user",
         "content": [
             {"type": "video_url", "video_url": {"url": Path(v).as_uri()}},
             {"type": "text", "text": args.question},
         ]},
    ] for v in videos]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)

    def once() -> dict:
        m_before = snapshot_metrics(llm)
        t = time.perf_counter()
        outs = llm.chat(messages, sp, use_tqdm=False)
        dt = time.perf_counter() - t
        m_after = snapshot_metrics(llm)
        d = spec_delta(m_after, m_before)
        return {
            "seconds": round(dt, 4),
            "requests": [
                {"video": v,
                 "text": o.outputs[0].text,
                 "token_ids": list(o.outputs[0].token_ids),
                 "num_prompt_tokens": len(o.prompt_token_ids or []),
                 "num_output_tokens": len(o.outputs[0].token_ids),
                 "finish_reason": o.outputs[0].finish_reason}
                for v, o in zip(videos, outs)
            ],
            "spec_counters_delta": d,
            "acceptance": acceptance(d),
        }

    probe = once()
    rec["probe"] = {"num_prompt_tokens": probe["requests"][0]["num_prompt_tokens"],
                    "acceptance": probe["acceptance"]}
    npt = probe["requests"][0]["num_prompt_tokens"]
    floor = 100 + 20 * args.num_frames
    if npt < floor:
        rec["status"] = "FAIL_VIDEO_NOT_EXPANDED"
        rec["guard"] = {"num_prompt_tokens": npt, "required_at_least": floor}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(f"FATAL: prompt {npt} tokens < {floor}; video dropped?", file=sys.stderr)
        return 3

    for _ in range(args.warmup):
        once()

    runs = [once() for _ in range(args.runs)]
    secs = [r["seconds"] for r in runs]
    mean = sum(secs) / len(secs)
    std = (sum((s - mean) ** 2 for s in secs) / len(secs)) ** 0.5

    # Aggregate acceptance across timed runs only.
    agg = {}
    for k in SPEC_KEYS:
        tot = 0
        seen = False
        for r in runs:
            v = r["spec_counters_delta"].get(k)
            if isinstance(v, list):
                if k not in agg:
                    agg[k] = [0] * len(v)
                agg[k] = [a + b for a, b in zip(agg[k], v)]
                seen = True
            elif isinstance(v, (int, float)):
                tot += v
                seen = True
        if k not in agg and seen:
            agg[k] = tot

    rec["runs"] = runs
    rec["seconds_mean"] = round(mean, 4)
    rec["seconds_std"] = round(std, 4)
    rec["token_ids"] = [r["requests"][0]["token_ids"] for r in runs]
    rec["output_text"] = runs[0]["requests"][0]["text"]
    rec["acceptance_agg"] = acceptance(agg)
    rec["spec_counters_agg"] = agg
    rec["num_prompt_tokens"] = npt
    rec["bit_exact_across_runs"] = all(
        [x["token_ids"] for x in r["requests"]] ==
        [x["token_ids"] for x in runs[0]["requests"]] for r in runs)
    rec["gpu_after"] = gpu_state()
    rec["status"] = "OK"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    print(json.dumps({
        "tag": args.tag,
        "status": rec["status"],
        "proposer": args.proposer,
        "num_spec_tokens": rec["num_spec_tokens"],
        "batch_size": len(videos),
        "num_prompt_tokens": npt,
        "seconds_mean": rec["seconds_mean"],
        "seconds_std": rec["seconds_std"],
        "bit_exact_across_runs": rec["bit_exact_across_runs"],
        "acceptance": rec["acceptance_agg"],
        "out": args.out,
    }, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    # try/finally at the outermost frame: the engine is shut down on normal exit,
    # on exception, and (via signal handlers) on SIGTERM. A bare `finally` does not
    # run on SIGTERM, which is precisely how the 41 GiB EngineCore leaked before.
    from std_vllm.engine_guard import shutdown_all as _shutdown_all
    try:
        _rc = main()
    finally:
        _shutdown_all()
    raise SystemExit(_rc)
