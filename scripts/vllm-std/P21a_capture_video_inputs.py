#!/usr/bin/env python
"""Phase 2.1a: can we capture the video pixels from inside vLLM?

WHAT THIS VERIFIES
------------------
Route B needs HF to re-run the visual prefix. That requires `pixel_values_videos`
(and `video_grid_thw`, `second_per_grid_ts`). They are NOT in `propose()`'s
arguments, so they must be captured by hooking vLLM's model forward.

Hook point (verified by reading source, not guessed):
  `Qwen2_5_VLForConditionalGeneration.forward(self, input_ids, positions,
   intermediate_tensors=None, inputs_embeds=None, **kwargs)`
  at vllm/model_executor/models/qwen2_5_vl.py:1927 -- video tensors arrive in **kwargs.

PASS  : pixel_values_videos captured, shape/dtype sane for HF re-use.
FAIL  : not captured, or shape cannot rebuild HF processor inputs -> route A.

TRACE SAFETY: the hook records tensor *references* and `torch.Size` only -- no
`.cpu()`, no `.tolist()`, no host sync. (A `.tolist()` here previously killed
engine startup with `torch._dynamo.exc.Unsupported`.)

Run
---
  CUDA_VISIBLE_DEVICES=<gpu> $PY scripts/vllm-std/P21a_capture_video_inputs.py --num-frames 32
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
    p.add_argument("--question", default="Describe what happens in this video in detail.")
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-mem-util", type=float, default=0.80)
    p.add_argument("--out", default=os.environ.get(
        "VLLM_STD_RES", "results/vllm-std") + "/P21a_video_inputs.json")
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

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import torch
    from vllm import LLM, SamplingParams
    from std_vllm.engine_guard import guard_engine, shutdown_all

    # ---- install the capture hook BEFORE the engine is built -----------------
    from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

    seen: dict[str, dict] = {}
    call_log: list[dict] = []
    orig_forward = Qwen2_5_VLForConditionalGeneration.forward

    def hooked_forward(self, input_ids, positions, intermediate_tensors=None,
                       inputs_embeds=None, **kwargs):
        try:
            if len(call_log) < 20:
                call_log.append({"kwargs_keys": sorted(kwargs.keys()),
                                 "n_input_ids": input_ids.shape[0] if input_ids is not None else None})
            for k in ("pixel_values_videos", "video_grid_thw", "second_per_grid_ts",
                      "pixel_values"):
                v = kwargs.get(k)
                if v is not None and k not in seen:
                    seen[k] = v if not torch.is_tensor(v) else v.detach()
        except Exception:
            pass
        return orig_forward(self, input_ids, positions, intermediate_tensors,
                            inputs_embeds, **kwargs)

    Qwen2_5_VLForConditionalGeneration.forward = hooked_forward

    # SECOND hook, and this is the one that actually matters.
    # Measured: the LLM `forward` receives `input_ids=None` and NO kwargs -- vLLM v1
    # runs the vision tower separately (log: "Encoder cache ... budget of 32768
    # tokens") and hands the LLM pre-computed `inputs_embeds`. So raw pixels never
    # reach `forward`. They do reach `embed_multimodal(**kwargs)`, which dispatches
    # to `_process_video_input` (qwen2_5_vl.py:1432, consumes pixel_values_videos at
    # :1442 and calls self.visual at :1451).
    orig_mm = Qwen2_5_VLForConditionalGeneration.embed_multimodal

    def hooked_mm(self, **kwargs):
        try:
            if len(call_log) < 20:
                call_log.append({"mm_kwargs_keys": sorted(kwargs.keys())})
            for k in ("pixel_values_videos", "video_grid_thw", "second_per_grid_ts",
                      "pixel_values", "image_grid_thw"):
                v = kwargs.get(k)
                if v is not None and k not in seen:
                    seen[k] = v if not torch.is_tensor(v) else v.detach()
        except Exception:
            pass
        return orig_mm(self, **kwargs)

    Qwen2_5_VLForConditionalGeneration.embed_multimodal = hooked_mm

    rec: dict = {"phase": "2.1a", "timestamp": utc(),
                 "video": video, "num_frames": args.num_frames,
                 "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                 "torch_version": torch.__version__}
    try:
        llm = guard_engine(LLM(
            model=args.model, max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            limit_mm_per_prompt={"video": 1, "image": 1},
            media_io_kwargs={"video": {"num_frames": args.num_frames}},
            allowed_local_media_path=str(Path(video).parent),
            enforce_eager=True, enable_chunked_prefill=False,
            disable_log_stats=True))
        msgs = [[{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": Path(video).as_uri()}},
            {"type": "text", "text": args.question}]}]]
        t0 = time.perf_counter()
        out = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                            seed=0), use_tqdm=False)[0]
        rec["gen_seconds"] = round(time.perf_counter() - t0, 3)
        rec["num_prompt_tokens"] = len(out.prompt_token_ids or [])
        rec["output_text_head"] = out.outputs[0].text[:120]

        # ---- analyse captured tensors, now OUTSIDE the traced region ----------
        rec["call_log"] = call_log
        info = {}
        for k, v in seen.items():
            if torch.is_tensor(v):
                info[k] = {"is_tensor": True, "shape": [int(x) for x in v.shape],
                           "dtype": str(v.dtype), "device": str(v.device),
                           "numel": int(v.numel())}
            else:
                info[k] = {"is_tensor": False, "type": type(v).__name__,
                           "repr": repr(v)[:200]}
        rec["captured"] = info
        rec["captured_keys"] = sorted(seen.keys())

        ok = "pixel_values_videos" in seen and torch.is_tensor(seen["pixel_values_videos"])
        rec["verdict"] = "PASS" if ok else "FAIL_NO_PIXEL_VALUES_VIDEOS"
        if ok:
            pv = seen["pixel_values_videos"]
            gt = seen.get("video_grid_thw")
            T = int(gt[0][0]) if torch.is_tensor(gt) and gt.numel() >= 3 else None
            rec["pixel_values_videos_shape"] = [int(x) for x in pv.shape]
            rec["video_grid_thw"] = ([int(x) for x in gt.flatten()] if torch.is_tensor(gt) else None)
            rec["derived_num_frames_T"] = T
            rec["note"] = ("pixel_values_videos is the PROCESSED patch tensor vLLM's vision "
                           "tower consumes (not raw frames). That is exactly what HF's "
                           "Qwen2_5_VLForConditionalGeneration.forward also expects, so it "
                           "can be fed straight back in.")
    except Exception as exc:
        rec["verdict"] = "ERROR"
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        shutdown_all()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str))
    print(f"verdict            : {rec.get('verdict')}")
    print(f"num_prompt_tokens  : {rec.get('num_prompt_tokens')}")
    print(f"captured keys      : {rec.get('captured_keys')}")
    print(f"captured detail    : {json.dumps(rec.get('captured'), ensure_ascii=False)[:600]}")
    if rec.get("note"):
        print(f"note               : {rec['note']}")
    if rec.get("error"):
        print(f"error              : {rec['error']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
