#!/usr/bin/env python
"""P0 Path A: does STD's sparse selection actually preserve what the next token needs?

THE QUESTION
------------
STD's premise is "top-K visual KV is enough to draft". Whether that yields a non-zero
ACCEPTANCE RATE cannot be measured without a sparse forward pass -- but its
precondition can be measured cheaply and exactly from attention we already capture:

    if the top-K positions the selection keeps do NOT carry the attention mass that
    the LAST query row (the one that predicts the next token) depends on, then no
    sparse draft built on them can be accepted, and acceptance is 0 by construction.

So this probe measures, per layer, for a range of K:
  * mass_recall   = attention mass on the selected visual positions / total mass on
                    all visual positions, measured ON THE LAST QUERY ROW.
  * overlap       = Jaccard overlap between the STD selection and the last-row oracle.

AND it tests a specific, sharp hypothesis about STD's own criterion:

    `build_sparse_selection` scores a visual position by summing attention over ALL
    query positions and all grouped heads:  grouped.sum(dim=(1, 2)).
    But token n+1 depends on the LAST query row only.
    So the all-query criterion can select positions irrelevant to the next token,
    while missing the ones that matter.

  criterion "std_all_query"  = sum over all text query rows and group heads  (STD)
  criterion "last_query"     = last query row only                          (oracle-ish)

If `std_all_query` has much lower last-row recall than `last_query`, that is a
mechanistic explanation for the 0.0 acceptance rate, and it is actionable.

Run
---
  CUDA_VISIBLE_DEVICES=3 $PY scripts/vllm-std/P2_acceptance_probe.py --num-frames 4
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get(
        "QWEN25VL_7B", "/mnt/local/ldh/PhyLLM/Qwen2.5-VL-7B-Instruct"))
    p.add_argument("--video", default=None)
    p.add_argument("--question", default="Describe what happens in this video in detail.")
    p.add_argument("--num-frames", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--text-rows", type=int, default=32)
    p.add_argument("--ks", default="16,32,64,128,256,512")
    p.add_argument("--out", default=os.environ.get(
        "VLLM_STD_RES", "results/vllm-std") + "/P2_acceptance_probe.json")
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
    from std_vllm.engine_guard import guard_engine, shutdown_all

    from vllm.model_executor.layers.attention.attention import Attention
    caps: list[dict] = []
    orig_forward = Attention.forward

    def hooked_forward(self, query, key, value, output_shape=None):
        try:
            if query.shape[0] > 1 and key is not None and len(caps) < 64:
                caps.append({"layer": getattr(self, "layer_name", None),
                             "q": query.detach().clone(),
                             "k": key.detach().clone(),
                             "nh": getattr(self, "num_heads", -1),
                             "nkv": getattr(self, "num_kv_heads", -1),
                             "hs": getattr(self, "head_size", -1),
                             "n": int(query.shape[0])})
        except Exception:
            pass
        return orig_forward(self, query, key, value, output_shape)

    Attention.forward = hooked_forward
    rec = {"probe": "acceptance_precondition", "timestamp": utc(),
           "vllm_version": vllm.__version__, "torch_version": torch.__version__,
           "num_frames": args.num_frames, "text_rows": args.text_rows,
           "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}

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
        out = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                            seed=0), use_tqdm=False)[0]
        ids = list(out.prompt_token_ids or [])
        rec["prompt_tokens"] = len(ids)

        # Locate the video token span. Qwen2.5-VL uses <|video_pad|>.
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(args.model)
        vid_id = proc.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        rec["video_token_id"] = int(vid_id)
        vis = [i for i, t in enumerate(ids) if t == vid_id]
        rec["num_video_tokens"] = len(vis)
        if not vis:
            rec["status"] = "FAIL_NO_VIDEO_TOKENS"
            raise RuntimeError("no video tokens found in prompt")
        npt = len(ids)
        cands = [c for c in caps if c["n"] <= npt]
        if not cands:
            rec["status"] = "FAIL_NO_CAPTURE"
            raise RuntimeError("no prefill captured")
        # Analyse EVERY captured layer at the real prompt length, not just one:
        # a single layer cannot tell us whether the result is a layer-0 artefact.
        maxn = max(c["n"] for c in cands)
        by_layer: dict[str, dict] = {}
        for c in cands:
            if c["n"] == maxn:
                by_layer[c["layer"]] = c
        rec["layers_analysed"] = sorted(by_layer)
        rec["num_layers_analysed"] = len(by_layer)
        rec["captured_n"] = maxn

        per_layer = []
        for lname in sorted(by_layer):
          g = by_layer[lname]
          try:
            q, k = g["q"], g["k"]
            nh, nkv, hs = int(g["nh"]), int(g["nkv"]), int(g["hs"])
            q3 = q.view(-1, nh, hs) if q.dim() == 2 else q
            k3 = k.view(-1, nkv, hs) if k.dim() == 2 else k
            L = int(q3.shape[0])
            grp = nh // nkv
            tr = min(args.text_rows, L)
            rows = slice(L - tr, L)      # text query rows (video tokens precede text)
            vis_t = torch.tensor(vis, device=q3.device)
            scale = hs ** -0.5

            qg = q3[rows].reshape(tr, nkv, grp, hs).float()
            kf = k3.float()
            logits = torch.einsum("tqgh,klh->tqgk", qg, kf) * scale
            m = logits.max(dim=-1, keepdim=True).values
            e = (logits - m).exp()
            probs = e / e.sum(dim=-1, keepdim=True)
            torch.cuda.synchronize()

            pv = probs[..., vis_t]                                 # [tr,nkv,grp,nvis]
            score_std = pv.sum(dim=(0, 2))                         # [nkv, nvis]
            last_mass = pv[-1].sum(dim=0)                          # [nkv, nvis]
            lrec = {"layer": lname, "text_rows": tr,
                    "last_row_video_mass": round(float(pv[-1].sum()), 6), "by_K": []}
            for K in (int(x) for x in args.ks.split(",") if x.strip()):
                Kx = min(K, len(vis))
                e_std = torch.topk(score_std, k=Kx, dim=-1).indices
                e_orc = torch.topk(last_mass, k=Kx, dim=-1).indices
                mr_std = float(torch.gather(last_mass, 1, e_std).sum() / last_mass.sum())
                mr_orc = float(torch.gather(last_mass, 1, e_orc).sum() / last_mass.sum())
                inter = len(set(e_std[0].tolist()) & set(e_orc[0].tolist()))
                lrec["by_K"].append({"K": Kx,
                    "std_mass_recall": round(mr_std, 4),
                    "std_overlap_oracle": round(inter / Kx, 4),
                    "last_mass_recall": round(mr_orc, 4)})
            per_layer.append(lrec)
          except Exception as le:
            per_layer.append({"layer": lname, "error": f"{type(le).__name__}: {le}"})
        rec["per_layer"] = per_layer

        # aggregate across layers
        ok = [x for x in per_layer if "by_K" in x]
        ks = [int(x) for x in args.ks.split(",") if x.strip()]
        agg = []
        for i, K in enumerate(ks):
            a = [x["by_K"][i]["std_mass_recall"] for x in ok]
            b = [x["by_K"][i]["last_mass_recall"] for x in ok]
            o = [x["by_K"][i]["std_overlap_oracle"] for x in ok]
            agg.append({"K": K,
                        "std_mass_recall_mean": round(statistics.mean(a), 4) if a else None,
                        "std_mass_recall_std": round(statistics.stdev(a), 4) if len(a) > 1 else 0.0,
                        "last_mass_recall_mean": round(statistics.mean(b), 4) if b else None,
                        "std_overlap_oracle_mean": round(statistics.mean(o), 4) if o else None})
        rec["aggregate_by_K"] = agg
        rec["n_layers_ok"] = len(ok)

        # criterion A: STD's -- sum over ALL text query rows and group heads
        score_std = pv.sum(dim=(0, 2))                             # [nkv, nvis]
        # criterion B: last query row only, summed over group heads
        score_last = pv[-1].sum(dim=0)                             # [nkv, nvis]
        # the mass the LAST row actually places on video positions (ground truth)
        last_mass = pv[-1].sum(dim=0)                              # [nkv, nvis]

        rec["status"] = "OK"
    except Exception as exc:
        rec.setdefault("status", "ERROR")
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        shutdown_all()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str))
    if rec.get("status") == "OK":
        print(f"prompt={rec['prompt_tokens']} video_tokens={rec['num_video_tokens']} "
              f"layers={rec['n_layers_ok']} captured_n={rec['captured_n']}")
        print(f"{'K':>5} | {'STD all-query mass_recall':>26} | {'last-query oracle':>18} | {'std overlap':>12}")
        for r in rec["aggregate_by_K"]:
            print(f"{r['K']:>5} | {r['std_mass_recall_mean']:>18.4f} +/- {r['std_mass_recall_std']:<5.4f}"
                  f" | {r['last_mass_recall_mean']:>18.4f} | {r['std_overlap_oracle_mean']:>12.4f}")
    else:
        print(f"status={rec.get('status')} error={rec.get('error')}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
