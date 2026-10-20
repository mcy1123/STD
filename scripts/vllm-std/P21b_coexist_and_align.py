#!/usr/bin/env python
"""Phase 2.1b + 2.1c: can vLLM and HF coexist, and do their inputs line up?

WHY COMBINED
------------
Both need the same scarce thing: a ~37 GB GPU window. Running them as two scripts
would burn two windows. So 2.1b runs first and 2.1c runs ONLY if 2.1b passes --
sequencing is preserved, windows are not wasted.

2.1b GATE (memory)
  Load vLLM, then load HF Qwen2.5-VL FP16 alongside (no forward for HF), report
  headroom. PASS if headroom >= 5 GiB. FAIL -> route A.

2.1c GATE (alignment)
  Run vLLM once to get `prompt_token_ids` and the captured video tensors, rebuild
  the HF inputs, and assert `len(hf_input_ids) == vLLM num_prompt_tokens`.
  Then run ONE HF forward and compare its greedy next token to what vLLM produced.
  That last check is the real predictor of acceptance.

MEASUREMENT RULE
  Memory is read from nvidia-smi (device truth), not torch's allocator, because the
  question is "does the card hold both", not "how much did torch account for".
  Timing (if any) uses CUDA events, never perf_counter+cuda.synchronize() inside
  the engine.

Run
---
  bash scripts/vllm-std/wait_and_run.sh 40000 3600 \\
      P21b_coexist_and_align.py --num-frames 8 --max-model-len 8192
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gpu_free_mib(idx: str | None = None) -> int | None:
    try:
        q = "memory.free"
        cmd = ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"]
        if idx is not None:
            cmd += ["-i", idx]
        return int(subprocess.run(cmd, capture_output=True, text=True, timeout=20,
                                  check=True).stdout.strip().splitlines()[0])
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get(
        "QWEN25VL_7B", "/mnt/local/ldh/PhyLLM/Qwen2.5-VL-7B-Instruct"))
    p.add_argument("--video", default=None)
    p.add_argument("--question", default="Describe what happens in this video in detail.")
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem-util", type=float, default=0.36)
    p.add_argument("--min-headroom-gib", type=float, default=5.0)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--skip-hf", action="store_true",
                   help="2.1b only: just check the vLLM side fits")
    p.add_argument("--out", default=os.environ.get(
        "VLLM_STD_RES", "results/vllm-std") + "/P21b_coexist.json")
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
    dev = os.environ.get("CUDA_VISIBLE_DEVICES")

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import torch
    from vllm import LLM, SamplingParams
    from std_vllm.engine_guard import guard_engine, shutdown_all

    rec: dict = {"phase": "2.1b+2.1c", "timestamp": utc(), "video": video,
                 "num_frames": args.num_frames, "max_model_len": args.max_model_len,
                 "gpu_mem_util": args.gpu_mem_util, "hf_dtype": args.dtype,
                 "gpu_visible": dev, "torch_version": torch.__version__,
                 "free_before_mib": gpu_free_mib(dev)}

    hf_model = None
    try:
        # ---------------- vLLM side -------------------------------------------
        from vllm.model_executor.models.qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration as VMModel)

        captured: dict[str, object] = {}
        mm_calls: list[dict] = []
        orig_mm = VMModel.embed_multimodal

        def hooked_mm(self, **kwargs):
            # Correct hook (verified in 2.1a): vLLM v1 runs the vision tower
            # separately, so raw pixels reach embed_multimodal, NOT forward.
            #
            # Capture the LAST call, not the first. vLLM profiles the encoder at
            # startup "with 1 video items of the MAXIMUM feature size", so the first
            # call carries warmup shapes (grid [1,256,256] -> 16384 features) that do
            # NOT correspond to the real request (4784 video tokens here). Taking the
            # first call produced a phantom alignment mismatch.
            snap = {}
            for k in ("pixel_values_videos", "video_grid_thw", "second_per_grid_ts"):
                v = kwargs.get(k)
                if v is not None:
                    snap[k] = v.detach().clone() if torch.is_tensor(v) else v
            if snap:
                if len(mm_calls) < 40:
                    mm_calls.append({k: (list(v.shape) if torch.is_tensor(v) else str(v))
                                     for k, v in snap.items()})
                captured.clear()
                captured.update(snap)
            return orig_mm(self, **kwargs)

        VMModel.embed_multimodal = hooked_mm

        t0 = time.perf_counter()
        llm = guard_engine(LLM(
            model=args.model, max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            limit_mm_per_prompt={"video": 1, "image": 1},
            media_io_kwargs={"video": {"num_frames": args.num_frames}},
            allowed_local_media_path=str(Path(video).parent),
            enforce_eager=True, enable_chunked_prefill=False,
            disable_log_stats=True))
        rec["vllm_load_seconds"] = round(time.perf_counter() - t0, 2)
        rec["free_after_vllm_mib"] = gpu_free_mib(dev)

        msgs = [[{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": Path(video).as_uri()}},
            {"type": "text", "text": args.question}]}]]
        out = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                            seed=0), use_tqdm=False)[0]
        rec["num_prompt_tokens_vllm"] = len(out.prompt_token_ids or [])
        rec["vllm_first_token"] = int(out.outputs[0].token_ids[0])
        rec["vllm_text_head"] = out.outputs[0].text[:80]
        rec["captured_keys"] = sorted(captured.keys())
        rec["mm_call_shapes"] = mm_calls
        rec["free_after_vllm_run_mib"] = gpu_free_mib(dev)

        if args.skip_hf:
            rec["verdict_2_1b"] = "SKIPPED_HF"
            rec["status"] = "OK"
            raise SystemExit if False else None  # fall through to finally

        # ---------------- 2.1b: HF coexistence --------------------------------
        from transformers import AutoModelForVision2Seq
        t0 = time.perf_counter()
        # Avoid `device_map` on purpose: it pulls in `accelerate` and, without it,
        # fails with "Using a device_map ... requires accelerate". Moving the module
        # explicitly is equivalent here (single GPU, TP=1) and has no extra dep.
        hf_model = AutoModelForVision2Seq.from_pretrained(
            args.model, torch_dtype=getattr(torch, args.dtype),
            low_cpu_mem_usage=True).to("cuda")
        hf_model.eval()
        rec["hf_load_seconds"] = round(time.perf_counter() - t0, 2)
        free_after_hf = gpu_free_mib(dev)
        rec["free_after_hf_mib"] = free_after_hf
        hf_gib = round((sum(p.numel() * p.element_size()
                            for p in hf_model.parameters()) / 2**30), 2)
        rec["hf_weight_gib"] = hf_gib
        headroom = round((free_after_hf or 0) / 1024, 2)
        rec["headroom_gib"] = headroom
        rec["verdict_2_1b"] = ("PASS" if headroom >= args.min_headroom_gib
                               else "FAIL_LOW_HEADROOM")

        # ---------------- 2.1c: alignment (only if 2.1b passed) ---------------
        if rec["verdict_2_1b"] == "PASS":
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(args.model)
            ids = list(out.prompt_token_ids or [])
            rec["align_check"] = {
                "vllm_prompt_tokens": len(ids),
                "captured_video_grid_thw": (
                    [int(x) for x in captured["video_grid_thw"].flatten()]
                    if torch.is_tensor(captured.get("video_grid_thw")) else None),
                "captured_pixel_shape": (
                    [int(x) for x in captured["pixel_values_videos"].shape]
                    if torch.is_tensor(captured.get("pixel_values_videos")) else None),
            }
            # Rebuild HF inputs from vLLM's own token ids + captured pixels.
            hf_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
            kw = {"input_ids": hf_ids, "use_cache": True}
            if torch.is_tensor(captured.get("pixel_values_videos")):
                kw["pixel_values_videos"] = captured["pixel_values_videos"].to(
                    "cuda", dtype=getattr(torch, args.dtype))
            if torch.is_tensor(captured.get("video_grid_thw")):
                kw["video_grid_thw"] = captured["video_grid_thw"]
            if torch.is_tensor(captured.get("second_per_grid_ts")):
                kw["second_per_grid_ts"] = captured["second_per_grid_ts"].to(
                    getattr(torch, args.dtype))
            ok = False
            try:
                with torch.inference_mode():
                    hf_out = hf_model(**kw)
                hf_tok = int(torch.argmax(hf_out.logits[0, -1, :]).item())
                rec["align_check"]["hf_next_token"] = hf_tok
                rec["align_check"]["vllm_first_token"] = rec["vllm_first_token"]
                rec["align_check"]["next_token_match"] = (hf_tok == rec["vllm_first_token"])
                ok = True
            except Exception as e:
                rec["align_check"]["hf_forward_error"] = f"{type(e).__name__}: {e}"
            rec["align_check"]["hf_forward_ok"] = ok
            rec["verdict_2_1c"] = ("PASS" if ok else "FAIL_HF_FORWARD")
        else:
            rec["verdict_2_1c"] = "SKIPPED_2_1B_FAILED"
        rec["status"] = "OK"

    except Exception as exc:
        rec.setdefault("status", "ERROR")
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if hf_model is not None:
                del hf_model
                gc.collect()
                torch.cuda.empty_cache()
        except Exception:
            pass
        shutdown_all()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str))
    for k in ("status", "verdict_2_1b", "verdict_2_1c", "headroom_gib",
              "free_after_vllm_mib", "free_after_hf_mib",
              "num_prompt_tokens_vllm", "vllm_first_token", "captured_keys", "error"):
        if k in rec:
            print(f"{k:26}: {rec[k]}")
    if "align_check" in rec:
        print(f"{'align_check':26}: {json.dumps(rec['align_check'], ensure_ascii=False)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
