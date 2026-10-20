#!/usr/bin/env python
"""LINE B PROBE (go/no-go): can a sparse-KV selection statistic be obtained in vLLM?

Why this is the decisive question
---------------------------------
STD's draft step needs `build_sparse_selection` (src/std_repro/std_qwen25vl.py:185):
from the *prefill attention*, sum over query heads per KV head, restrict to visual
positions, and keep the top-K per layer. Everything downstream depends on that.

vLLM's fast attention backends (FLASH_ATTN / FLASHINFER) do NOT materialise the
softmax matrix -- that is the risk flagged in docs/vllm-std/04-std-1token.md 4.5.

BUT the fused kernel is not the only possible source. `Attention.forward`
(vllm/model_executor/layers/attention/attention.py:437) is a *Python* method that
receives `query, key, value` as real tensors before dispatching to the backend:

    def forward(self, query, key, value, output_shape=None) -> torch.Tensor

If that Python frame actually executes during a video prefill, then Q/K/V (and
hence the score statistic) are reachable by a plain monkey-patch, with no vLLM
patch and no backend change. That is what this probe tests.

Three things are measured
-------------------------
1. Does the hook fire at all? (torch.compile / CUDA graphs may bypass it.)
2. What are the shapes -- is `key` the FULL prompt during prefill, or only the
   newly-scheduled chunk? This decides whether one pass yields the whole signal.
3. Can the STD statistic (per-KV-head summed scores over visual positions) be
   computed from what we captured, and at what cost?

Usage
-----
    STD_FORCE_GPU=7 bash scripts/vllm-std/run_bench.sh <log> B_probe_attn_weights.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get(
        "QWEN25VL_7B", "/mnt/local/ldh/PhyLLM/Qwen2.5-VL-7B-Instruct"))
    p.add_argument("--video", default=None)
    p.add_argument("--question", default="Describe what happens in this video.")
    p.add_argument("--num-frames", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem-util", type=float, default=0.30)
    p.add_argument("--text-rows", type=int, default=64,
                   help="STD only needs the TEXT query rows, not every prompt row "
                        "(build_sparse_selection slices to [.., text_len, prompt_len])")
    p.add_argument("--chunked-prefill", action="store_true",
                   help="ENABLE chunked prefill (default OFF). Hard precondition #2: "
                        "with chunked prefill on, K is only the current chunk and "
                        "text->prompt selection is impossible.")
    p.add_argument("--no-enforce-eager", action="store_true",
                   help="probe defaults to eager so the hook is not traced by dynamo")
    p.add_argument("--out", default=os.environ.get(
        "VLLM_STD_RES", "results/vllm-std") + "/B_attn_weight_probe.json")
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

    # Hard precondition #1: the hook lives in this process, but vLLM's EngineCore
    # runs in a forked child by default, so anything the hook records (and any
    # monkey-patch that must be visible) would be lost. Force in-process execution.
    # Must be set BEFORE `import vllm` because vllm.envs reads it at import time.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    # ---- install the hook BEFORE the engine is built -------------------------
    from vllm.model_executor.layers.attention.attention import Attention

    from collections import deque
    # Tensors only for the LAST few prefill passes (the real request runs after
    # warmup), but a cheap uncapped log of every shape so nothing is truncated.
    captured: deque = deque(maxlen=4)
    prefill_shapes: list[dict] = []
    counts = {"n": 0, "prefill": 0, "decode": 0}
    orig_forward = Attention.forward

    # TRACE SAFETY: an earlier version of this hook called `.tolist()` on a float
    # tensor and killed engine startup with
    #   torch._dynamo.exc.Unsupported: Tensor.tolist() with non-integer tensor
    # because torch.compile traces THROUGH this Python method. Any host sync or
    # data-dependent Python branch here breaks graph capture. So the hook only
    # keeps tensor *references* (no .cpu(), no .tolist(), no int(shape)), and all
    # measurement happens after the run, outside the traced region.
    def hooked_forward(self, query, key, value, output_shape=None):
        try:
            counts["n"] += 1
            # Compare on the shape only; no host sync, no data-dependent Python.
            if query.shape[0] > 1:
                counts["prefill"] += 1
                # Keep PREFILL calls only. An earlier version kept every call in a
                # 64-slot ring buffer and the prefill entries were evicted by the
                # decode calls that followed, which made the probe wrongly report
                # "decode only". Decode calls are counted, never stored.
                if len(prefill_shapes) < 400:
                    prefill_shapes.append({
                        "layer": getattr(self, "layer_name", None),
                        "q": [int(x) for x in query.shape],
                        "k": None if key is None else [int(x) for x in key.shape],
                    })
                if True:
                    captured.append({
                        "layer": getattr(self, "layer_name", None),
                        "query": query.detach(),
                        "key": None if key is None else key.detach(),
                        "num_tokens_meta": query.shape[0],
                        "num_heads": getattr(self, "num_heads", -1),
                        "num_kv_heads": getattr(self, "num_kv_heads", -1),
                        "head_size": getattr(self, "head_size", -1),
                    })
            else:
                counts["decode"] += 1
        except Exception:
            pass
        return orig_forward(self, query, key, value, output_shape)

    Attention.forward = hooked_forward

    rec: dict = {
        "probe": "attn_weight_availability",
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "video": video, "num_frames": args.num_frames,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    from std_vllm.engine_guard import guard_engine

    llm = guard_engine(LLM(
        model=args.model, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        limit_mm_per_prompt={"video": 1, "image": 1},
        media_io_kwargs={"video": {"num_frames": args.num_frames}},
        allowed_local_media_path=str(Path(video).parent),
        enforce_eager=not args.no_enforce_eager,
        enable_chunked_prefill=args.chunked_prefill,
        disable_log_stats=True,
    ))

    messages = [[{
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": Path(video).as_uri()}},
            {"type": "text", "text": args.question},
        ],
    }]]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)
    out = llm.chat(messages, sp, use_tqdm=False)[0]

    rec["num_prompt_tokens"] = len(out.prompt_token_ids or [])
    npt = rec["num_prompt_tokens"]

    # Analyse the captured tensors now that the traced region is over.
    hook_calls = dict(counts)
    prefill = list(captured)
    rec["hook_calls"] = hook_calls
    rec["all_prefill_shapes_observed"] = prefill_shapes
    rec["captured_summary"] = [{
        "layer": c["layer"],
        "query_shape": [int(x) for x in c["query"].shape],
        "key_shape": None if c["key"] is None else [int(x) for x in c["key"].shape],
        "dtype": str(c["query"].dtype),
        "num_heads": int(c["num_heads"]), "num_kv_heads": int(c["num_kv_heads"]),
        "head_size": int(c["head_size"]),
    } for c in list(captured)[:4]]

    # ---- can we build the STD statistic from what the hook saw? --------------
    # build_sparse_selection (src/std_repro/std_qwen25vl.py:185) computes, per layer:
    #     grouped = attn.reshape(nkv, heads_per_kv, q_len, kv_len)
    #     scores  = grouped.sum(dim=(1, 2))          # -> [nkv, kv_len]
    # where `attn` is the POST-SOFTMAX attention. The sum runs over query positions
    # too, and softmax normalises each (query, head) row independently, so the
    # softmax denominator does NOT cancel -- we must actually form probabilities.
    # This does that chunked over query positions, which is what a real proposer
    # would have to do since the fused kernel never hands us the matrix.
    stats = {}
    try:
        if prefill:
            # Prefer the shortest captured prefill: the others are warmup passes
            # padded to the max_num_batched_tokens bucket (8192).
            # Warmup/profiling passes use the max_num_batched_tokens bucket; the
            # real request is the smallest captured prefill. If several share the
            # smallest size, take the last one.
            c0 = sorted(prefill, key=lambda c: int(c["num_tokens_meta"]))[0]
            for c in reversed(prefill):
                if int(c["num_tokens_meta"]) == int(c0["num_tokens_meta"]):
                    c0 = c
                    break
            q_raw, k_raw = c0["query"], c0["key"]
            nh, nkv = int(c0["num_heads"]), int(c0["num_kv_heads"])
            hs = int(c0["head_size"])
            q3 = q_raw.view(-1, nh, hs) if q_raw.dim() == 2 else q_raw
            k3 = k_raw.view(-1, nkv, hs) if k_raw.dim() == 2 else k_raw
            q_len, kv_len = int(q3.shape[0]), int(k3.shape[0])
            grp = nh // nkv
            stats.update({
                "layer": c0["layer"], "q_len": q_len, "kv_len": kv_len,
                "num_heads": nh, "num_kv_heads": nkv, "head_size": hs,
                "heads_per_kv": grp,
                "key_is_full_prompt": kv_len == npt,
                "captured_prefill_token_counts": sorted(
                    {int(c["num_tokens_meta"]) for c in prefill}),
                "all_prefill_shapes": [
                    {"layer": c["layer"],
                     "q": [int(x) for x in c["query"].shape],
                     "k": [int(x) for x in c["key"].shape]} for c in prefill],
                "prompt_tokens": npt,
            })
            # STD only needs text query rows: build_sparse_selection consumes
            # attn[batch, q_heads, text_len, prompt_len]. Use the trailing rows.
            n_rows = min(args.text_rows, q_len)
            q_use = q3[q_len - n_rows:]
            t0 = time.perf_counter()
            scale = hs ** -0.5
            qg = q_use.reshape(n_rows, nkv, grp, hs)
            kf = k3.float()
            acc = torch.zeros(nkv, kv_len, device=q3.device, dtype=torch.float32)
            CH = 512
            for s0 in range(0, n_rows, CH):
                qc = qg[s0:s0 + CH].float()
                logits = torch.einsum("cqgh,klh->cqgk", qc, kf) * scale
                m = logits.max(dim=-1, keepdim=True).values
                e = (logits - m).exp()
                probs = e / e.sum(dim=-1, keepdim=True)
                acc += probs.sum(dim=(0, 2))
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            stats["statistic_compute_seconds"] = round(dt, 4)
            stats["statistic_shape"] = list(acc.shape)
            stats["statistic_is_finite"] = bool(torch.isfinite(acc).all())
            stats["statistic_row0_top5_idx"] = [
                int(x) for x in torch.topk(acc[0], k=min(5, kv_len)).indices]
            stats["statistic_row0_top5_val"] = [
                round(float(x), 4) for x in torch.topk(acc[0], k=min(5, kv_len)).values]
            stats["statistic_row0_min"] = round(float(acc[0].min()), 6)
            stats["statistic_row0_max"] = round(float(acc[0].max()), 6)
            # Sanity: softmax rows sum to 1, so summing over q gives total mass q_len
            # per (kv head, ...) -- acc summed over kv positions should equal q_len*grp.
            stats["text_rows_used"] = n_rows
            stats["mass_check_expected"] = float(n_rows * grp * nkv)
            stats["mass_check_actual"] = round(float(acc.sum()), 3)
            stats["softmax_weights_materialised_by_kernel"] = False
            stats["note"] = ("Post-softmax attention probabilities were reconstructed "
                             "from hooked Q/K (chunked). Top-K selection is therefore "
                             "possible without any vLLM patch and without backend "
                             "support for returning weights -- but the reconstruct cost "
                             "is O(q_len * kv_len) and must be paid on every prefill.")
    except Exception as e:
        stats["stat_error"] = f"{type(e).__name__}: {e}"
    rec["statistics"] = stats

    # ---- verdict -------------------------------------------------------------
    if not captured:
        rec["verdict"] = "NO_HOOK"
        rec["conclusion"] = ("Attention.forward never ran in Python -- the compiled/"
                             "custom-op path bypasses it. Q/K/V are NOT reachable "
                             "this way.")
    elif not prefill:
        rec["verdict"] = "DECODE_ONLY"
        rec["conclusion"] = ("Hook fires but only for single-token decode steps; no "
                             "prefill Q/K/V captured.")
    elif rec["statistics"].get("key_is_full_prompt"):
        rec["verdict"] = "YES_FULL_PROMPT"
        rec["conclusion"] = ("Prefill Q/K/V are reachable from a plain Python hook and "
                             "K spans the FULL prompt -> the STD sparse-selection "
                             "statistic is obtainable without patching vLLM.")
    else:
        rec["verdict"] = "YES_BUT_CHUNKED"
        rec["conclusion"] = ("Prefill Q/K/V are reachable, but K is only the scheduled "
                             "chunk, so several chunks must be accumulated (or chunked "
                             "prefill disabled) to see the whole prompt.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    print(json.dumps({k: rec[k] for k in
                      ("verdict", "conclusion", "hook_calls", "num_prompt_tokens",
                       "statistics")}, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    from std_vllm.engine_guard import shutdown_all as _shutdown_all
    try:
        _rc = main()
    finally:
        _shutdown_all()
    raise SystemExit(_rc)
