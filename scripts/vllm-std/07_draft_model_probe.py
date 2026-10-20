#!/usr/bin/env python
"""Phase 7: pin down exactly where `method="draft_model"` dies on a multimodal target.

Phase 2 predicted, from source, that the blocker is NOT multimodality (which is only
a warning) but M-RoPE (`_raise_if_mrope`, a hard NotImplementedError), and that the
gate lives in `LLMBaseProposer.__init__` -- which `custom_class` never runs.

This script tests that prediction by actually trying it, and records the verbatim
exception plus the traceback frames so the claim is evidence, not inference.

Usage
-----
    $PY scripts/vllm-std/07_draft_model_probe.py --draft <draft_model_path>
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get(
        "QWEN25VL_7B", "/mnt/local/ldh/PhyLLM/Qwen2.5-VL-7B-Instruct"))
    p.add_argument("--draft", default=os.environ.get(
        "QWEN25VL_7B", "/mnt/local/ldh/PhyLLM/Qwen2.5-VL-7B-Instruct"),
        help="draft model path; any Qwen2.5-VL works since M-RoPE is what triggers")
    p.add_argument("--num-spec-tokens", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem-util", type=float, default=0.80)
    p.add_argument("--out", default=os.environ.get(
        "VLLM_STD_RES", "results/vllm-std") + "/07_draft_model_probe.json")
    args = p.parse_args()

    rec = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_model": args.model,
        "draft_model": args.draft,
        "method": "draft_model",
        "num_speculative_tokens": args.num_spec_tokens,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    from vllm import LLM
    import vllm
    from vllm.transformers_utils.config import uses_mrope
    from transformers import AutoConfig

    rec["vllm_version"] = vllm.__version__

    # Record the premise: the target really is M-RoPE.
    try:
        tcfg = AutoConfig.from_pretrained(args.model)
        dcfg = AutoConfig.from_pretrained(args.draft)
        # NOTE: vLLM v0.22.0 `_uses_mrope` reads `config.rope_parameters`, which is
        # the *normalised* name vLLM creates while building its own config. A raw
        # `transformers.AutoConfig` still exposes the HF name `rope_scaling`, so
        # calling `uses_mrope()` on it returns False even for Qwen2.5-VL -- a false
        # negative that almost hid the real cause. Check BOTH names.
        def mrope(cfg):
            for name in ("rope_parameters", "rope_scaling"):
                rp = getattr(cfg, name, None)
                if isinstance(rp, dict) and "mrope_section" in rp:
                    return True
            t = cfg.get_text_config() if hasattr(cfg, "get_text_config") else None
            if t is not None and t is not cfg:
                for name in ("rope_parameters", "rope_scaling"):
                    rp = getattr(t, name, None)
                    if isinstance(rp, dict) and "mrope_section" in rp:
                        return True
            return False
        rec["target_uses_mrope"] = mrope(tcfg) or bool(uses_mrope(tcfg))
        rec["draft_uses_mrope"] = mrope(dcfg) or bool(uses_mrope(dcfg))
        rec["target_rope_scaling"] = getattr(tcfg, "rope_scaling", None)
    except Exception as e:
        rec["config_probe_error"] = f"{type(e).__name__}: {e}"

    try:
        LLM(model=args.model, max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            speculative_config={
                "method": "draft_model",
                "model": args.draft,
                "num_speculative_tokens": args.num_spec_tokens,
            },
            disable_log_stats=True)
        rec["outcome"] = "STARTED"
        rec["note"] = "draft_model unexpectedly started; the Phase 2 prediction is wrong"
    except Exception as e:
        rec["outcome"] = "FAILED"
        rec["exception_type"] = type(e).__name__
        rec["exception_message"] = str(e)
        tb = traceback.format_exc()
        rec["traceback_tail"] = tb.strip().splitlines()[-12:]
        # Classify against the two guards documented in Phase 2.
        blob = tb + "\n" + str(e)
        rec["hit_raise_if_mrope"] = "M-RoPE" in blob or "does not support M-RoPE" in blob
        rec["hit_warn_if_multimodal"] = "does not fully support multimodal" in blob
        rec["mentions_multimodal"] = "multimodal" in blob.lower()
        rec["frames_of_interest"] = [ln.strip() for ln in tb.splitlines()
                                     if "llm_base_proposer" in ln or "draft_model" in ln]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    print(json.dumps({k: rec.get(k) for k in
                      ("outcome", "exception_type", "exception_message",
                       "hit_raise_if_mrope", "target_uses_mrope",
                       "draft_uses_mrope")}, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
