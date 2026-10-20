#!/usr/bin/env python
"""P0 Phase 1: break down the per-layer selection cost into A/B/C.

Question
--------
The naive selection implementation cost 157 ms for ONE layer (64 text rows x 2420
keys). The theoretical FLOP count for all 28 layers is only ~8.8 GFLOP, so the gap
is 2-3 orders of magnitude. We must show WHERE the time goes before deciding
whether a fused kernel could amortise it.

A/B/C split
-----------
  A = GPU kernel time        (what a fused kernel would still have to pay)
  B = host / launch / sync   (what batching + fusion removes)
  C = memory movement        (dtype casts, copies, D2H)

Three variants are measured so the split is attributable, not guessed:

  core          einsum(QK^T) -> softmax -> accumulate. The math only.
  core+book     core, plus the scalar extraction the *probe* needed for its own
                validation (.item()/float()/isfinite/topk->python list). These force
                host syncs and D2H copies that a real proposer would never do.
  naive         the original probe implementation verbatim, for continuity with
                the 157 ms number.

Run
---
  CUDA_VISIBLE_DEVICES=3 $PY scripts/vllm-std/P1_profile_selection.py \
      --num-frames 4 --repeats 3
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# the three variants under test
# --------------------------------------------------------------------------
def selection_core(q3, k3, n_rows, text_rows, nkv, grp, hs, chunk=512):
    """Only the arithmetic a fused kernel must perform."""
    scale = hs ** -0.5
    qg = q3[n_rows - text_rows:].reshape(text_rows, nkv, grp, hs)
    kf = k3.float()
    acc = globals()["_torch"].zeros(nkv, k3.shape[0], device=q3.device,
                                    dtype=globals()["_torch"].float32)
    for s0 in range(0, text_rows, chunk):
        qc = qg[s0:s0 + chunk].float()
        logits = globals()["_torch"].einsum("cqgh,klh->cqgk", qc, kf) * scale
        m = logits.max(dim=-1, keepdim=True).values
        e = (logits - m).exp()
        probs = e / e.sum(dim=-1, keepdim=True)
        acc += probs.sum(dim=(0, 2))
    return acc


def selection_with_bookkeeping(q3, k3, n_rows, text_rows, nkv, grp, hs, chunk=512):
    """core + the scalar extraction the probe added purely to validate itself.

    Every `float(...)`/`bool(...)`/`.tolist()` here is a host sync + D2H copy.
    A real proposer only needs the top-K *indices on device*, so this whole block
    is measurement overhead -- but it was part of the measured 157 ms.
    """
    torch = globals()["_torch"]
    acc = selection_core(q3, k3, n_rows, text_rows, nkv, grp, hs, chunk)
    out = {}
    out["shape"] = list(acc.shape)
    out["is_finite"] = bool(torch.isfinite(acc).all())          # sync
    out["row0_min"] = round(float(acc[0].min()), 6)             # sync + D2H
    out["row0_max"] = round(float(acc[0].max()), 6)             # sync + D2H
    out["mass"] = round(float(acc.sum()), 3)                    # sync + D2H
    tk = torch.topk(acc[0], k=5)
    out["top5_idx"] = [int(x) for x in tk.indices.tolist()]     # sync + D2H + tolist
    out["top5_val"] = [round(float(x), 4) for x in tk.values.tolist()]
    return out


def selection_naive(q3, k3, n_rows, text_rows, nkv, grp, hs, chunk=512):
    """The original probe body, kept for continuity with the 157 ms figure."""
    torch = globals()["_torch"]
    scale = hs ** -0.5
    qg = q3[n_rows - text_rows:].reshape(text_rows, nkv, grp, hs)
    kf = k3.float()
    t0 = time.perf_counter()
    acc = torch.zeros(nkv, k3.shape[0], device=q3.device, dtype=torch.float32)
    for s0 in range(0, text_rows, chunk):
        qc = qg[s0:s0 + chunk].float()
        logits = torch.einsum("cqgh,klh->cqgk", qc, kf) * scale
        m = logits.max(dim=-1, keepdim=True).values
        e = (logits - m).exp()
        probs = e / e.sum(dim=-1, keepdim=True)
        acc += probs.sum(dim=(0, 2))
    torch.cuda.synchronize()
    return {"seconds_inner": time.perf_counter() - t0, "acc": acc}


VARIANTS = {
    "core": selection_core,
    "core+book": selection_with_bookkeeping,
    "naive": selection_naive,
}


def time_with_events(fn, args):
    """Accurate GPU time via CUDA events (no profiler overhead)."""
    torch = globals()["_torch"]
    torch.cuda.synchronize()
    s_ev = torch.cuda.Event(enable_timing=True)
    e_ev = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    s_ev.record()
    fn(*args)
    e_ev.record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e3
    return {"gpu_ms": round(s_ev.elapsed_time(e_ev), 4), "wall_ms": round(wall, 4)}


def profile_variant(name, fn, args, repeats):
    """Time one variant and split it into A (kernel) / B (host) / C (copies)."""
    torch = globals()["_torch"]
    from torch.profiler import ProfilerActivity, profile

    runs = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            fn(*args)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0

        cuda_ns = 0          # all device work
        cuda_memcpy_ns = 0   # device-side copies
        cpu_ns = 0
        op_cuda: dict[str, int] = {}
        op_cpu: dict[str, int] = {}
        memcpy_ops: dict[str, int] = {}
        for evt in prof.key_averages():
            dt = getattr(evt, "device_time_total", 0) or 0
            ct = getattr(evt, "self_cpu_time_total", 0) or 0
            if dt > 0:
                cuda_ns += dt
                op_cuda[evt.key] = op_cuda.get(evt.key, 0) + dt
                if "memcpy" in evt.key.lower() or "Memcpy" in evt.key or \
                   "copy" in evt.key.lower():
                    cuda_memcpy_ns += dt
                    memcpy_ops[evt.key] = memcpy_ops.get(evt.key, 0) + dt
            if ct > 0:
                cpu_ns += ct
                op_cpu[evt.key] = op_cpu.get(evt.key, 0) + ct

        a_ms = cuda_ns / 1e6
        wall_ms = wall * 1e3
        runs.append({
            "wall_ms": round(wall_ms, 3),
            "A_kernel_ms": round(a_ms, 3),
            "B_host_ms": round(wall_ms - a_ms, 3),
            "C_memcpy_ms": round(cuda_memcpy_ns / 1e6, 3),
            "cpu_self_ms": round(cpu_ns / 1e6, 3),
            "top_kernels": sorted(op_cuda.items(), key=lambda kv: -kv[1])[:8],
            "top_cpu_ops": sorted(op_cpu.items(), key=lambda kv: -kv[1])[:8],
            "memcpy_ops": sorted(memcpy_ops.items(), key=lambda kv: -kv[1])[:5],
        })

    # COLD vs STEADY STATE. The original 157 ms figure came from a *first* call on a
    # fresh shape: cuBLAS workspace allocation, einsum kernel selection and allocator
    # growth all land in that first sample. Time the cold call explicitly, then warm up
    # before the event-timed repeats.
    torch.cuda.synchronize()
    _t0 = time.perf_counter()
    fn(*args)
    torch.cuda.synchronize()
    cold_ms = round((time.perf_counter() - _t0) * 1e3, 3)

    for _ in range(2):          # warm-up: allocator + kernel selection
        fn(*args)
    torch.cuda.synchronize()

    # Event-timed (accurate GPU time) steady-state repeats.
    ev = [time_with_events(fn, args) for _ in range(repeats)]
    bare = [r["wall_ms"] for r in ev]
    gpu = [r["gpu_ms"] for r in ev]

    def ms(key):
        vals = [r[key] for r in runs]
        return {"mean": round(statistics.mean(vals), 3),
                "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
                "runs": vals}

    def stat(vals):
        return {"mean": round(statistics.mean(vals), 3),
                "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
                "runs": vals}

    return {"variant": name, "repeats": repeats, "per_run": runs,
            "cold_first_call_ms": cold_ms,
            "gpu_ms_events": stat(gpu),
            "wall_ms_profiled": ms("wall_ms"),
            "wall_ms_unprofiled": stat(bare),
            "A_kernel_ms": ms("A_kernel_ms"),
            "B_host_ms": ms("B_host_ms"),
            "C_memcpy_ms": ms("C_memcpy_ms"),
            "cpu_self_ms": ms("cpu_self_ms")}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get(
        "QWEN25VL_7B", "/mnt/local/ldh/PhyLLM/Qwen2.5-VL-7B-Instruct"))
    p.add_argument("--video", default=None)
    p.add_argument("--question", default="Describe what happens in this video.")
    p.add_argument("--num-frames", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--text-rows", type=int, default=64)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default=os.environ.get(
        "VLLM_STD_RES", "results/vllm-std") + "/profile_single_layer.json")
    p.add_argument("--trace", default=os.environ.get(
        "VLLM_STD_RES", "results/vllm-std") + "/profile_single_layer_trace.json")
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
    import vllm
    from vllm import LLM, SamplingParams
    globals()["_torch"] = torch

    from std_vllm.engine_guard import guard_engine, shutdown_all

    # Capture Q/K/V for one layer via the same hook validated in Phase 0.
    from vllm.model_executor.layers.attention.attention import Attention
    grabbed: list[dict] = []
    orig_forward = Attention.forward

    def hooked_forward(self, query, key, value, output_shape=None):
        try:
            if query.shape[0] > 1 and key is not None and len(grabbed) < 200:
                grabbed.append({"layer": getattr(self, "layer_name", None),
                                "q": query.detach().clone(),
                                "k": key.detach().clone(),
                                "nh": getattr(self, "num_heads", -1),
                                "nkv": getattr(self, "num_kv_heads", -1),
                                "hs": getattr(self, "head_size", -1)})
        except Exception:
            pass
        return orig_forward(self, query, key, value, output_shape)

    Attention.forward = hooked_forward

    rec = {"phase": "P0-1", "timestamp": utc(), "vllm_version": vllm.__version__,
           "torch_version": torch.__version__, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "num_frames": args.num_frames, "text_rows": args.text_rows,
           "chunk": args.chunk, "repeats": args.repeats}

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
        out = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0),
                       use_tqdm=False)[0]
        rec["prompt_tokens"] = len(out.prompt_token_ids or [])

        # Measure the PREFILL cost for the same input, because the stop-loss is a
        # RATIO (selection <= 10% of prefill). max_tokens=1 keeps decode out of it.
        prefill_ms = []
        for _ in range(3):
            _t = time.perf_counter()
            llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=1, seed=0),
                     use_tqdm=False)
            prefill_ms.append(round((time.perf_counter() - _t) * 1e3, 2))
        rec["prefill_ms"] = {"mean": round(statistics.mean(prefill_ms), 2),
                             "std": round(statistics.stdev(prefill_ms), 2),
                             "runs": prefill_ms,
                             "note": "max_tokens=1 -> dominated by prefill"}
        rec["prefill_caveat"] = ("Prefix caching is ON by default, so repeats after the "
                                 "first may hit the cache and understate prefill. Treat "
                                 "runs[0] as the honest figure.")

        if not grabbed:
            rec["status"] = "FAIL_NO_PREFILL_CAPTURE"
        else:
            # Warmup/profiling passes use the max_num_batched_tokens bucket; the
            # real request is the smallest captured prefill (same rule as the B probe).
            # Pick the pass that actually covers the prompt: the largest captured
            # prefill whose length is <= prompt_tokens. (The 8192 pass is the warmup
            # bucket; a 256 pass appears too; the real request has q_len == prompt.)
            npt = rec["prompt_tokens"]
            cands = [x for x in grabbed if int(x["q"].shape[0]) <= npt]
            g = max(cands, key=lambda x: int(x["q"].shape[0])) if cands else grabbed[0]
            q, k = g["q"], g["k"]
            nh, nkv, hs = int(g["nh"]), int(g["nkv"]), int(g["hs"])
            q3 = q.view(-1, nh, hs) if q.dim() == 2 else q
            k3 = k.view(-1, nkv, hs) if k.dim() == 2 else k
            n_rows = int(q3.shape[0])
            text_rows = min(args.text_rows, n_rows)
            grp = nh // nkv
            rec.update({"layer": g["layer"], "q_len": n_rows,
                        "kv_len": int(k3.shape[0]), "text_rows_used": text_rows,
                        "num_heads": nh, "num_kv_heads": nkv, "head_size": hs,
                        "heads_per_kv": grp,
                        "q_dtype": str(q.dtype), "k_dtype": str(k.dtype)})

            call_args = (q3, k3, n_rows, text_rows, nkv, grp, hs, args.chunk)
            rec["variants"] = {}
            for name, fn in VARIANTS.items():
                # warm-up once per variant (kernel autotune / allocator warm)
                fn(*call_args)
                torch.cuda.synchronize()
                rec["variants"][name] = profile_variant(name, fn, call_args, args.repeats)

            # Save a chrome trace of the last variant for offline inspection.
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                VARIANTS["core+book"](*call_args)
                torch.cuda.synchronize()
            prof.export_chrome_trace(args.trace)
            rec["trace"] = args.trace
            rec["status"] = "OK"
    finally:
        shutdown_all()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str))

    if rec.get("status") == "OK":
        print(f"layer={rec['layer']} prompt={rec['prompt_tokens']} "
              f"q={rec['q_len']} kv={rec['kv_len']} text_rows={rec['text_rows_used']}")
        print(f"{'variant':<12}{'COLD_ms':>10}{'wall_mean':>11}{'gpu_mean':>10}"
              f"{'gpu_std':>9}{'prof_ms':>10}")
        for name, v in rec["variants"].items():
            print(f"{name:<12}{v['cold_first_call_ms']:>10}"
                  f"{v['wall_ms_unprofiled']['mean']:>11}"
                  f"{v['gpu_ms_events']['mean']:>10}"
                  f"{v['gpu_ms_events']['std']:>9}"
                  f"{v['wall_ms_profiled']['mean']:>10}")
    else:
        print(f"status={rec.get('status')}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    rc = main()
    raise SystemExit(rc)
