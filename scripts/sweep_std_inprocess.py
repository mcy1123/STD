#!/usr/bin/env python
"""In-process K/gamma sweep for STD.

Unlike scripts/sweep_std.py, this loads the model once and runs AR once per
sample, then evaluates multiple STD parameter settings against that baseline.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import benchmark_std as bench  # noqa: E402
from std_repro.std_qwen25vl import std_generate_qwen25vl  # noqa: E402


def parse_int_list(value: str):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(ROOT / "models" / "Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--dataset", choices=["VideoDetailCaption", "MLVU", "Video-MME"], default="Video-MME")
    parser.add_argument("--data-path", default=str(ROOT / "datasets" / "Video-MME"))
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--prompt-style", choices=["direct", "cot"], default="direct")
    parser.add_argument("--eval-num", type=int, default=1)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--frame-num", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--ks", default="2048,4096", help="Comma-separated explicit visual K values. Use an empty string to skip explicit-K sweeps.")
    parser.add_argument("--target-k-plus-text", type=int, default=None, help="Also sweep paper-style K+text target with per-sample K.")
    parser.add_argument("--gammas", default="5,7,9")
    parser.add_argument("--verify-mode", choices=["parallel", "sequential"], default="parallel")
    parser.add_argument("--sparse-attn-mode", choices=["gqa_sdpa", "repeat_sdpa", "triton_gqa"], default="gqa_sdpa")
    parser.add_argument(
        "--no-copy-sparse-prefill",
        action="store_true",
        help="Run a separate sparse prompt prefill instead of copying the normal dense prefill cache.",
    )
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--output", default=str(ROOT / "results" / "std_qwen2_5_vl_7b" / "sweep_inprocess.jsonl"))
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, processor = bench.load_qwen_model(args.model_path, args.gpu_ids)
    eos_token_id = processor.tokenizer.eos_token_id
    ks = parse_int_list(args.ks)
    k_settings = [(k, None, f"k={k}") for k in ks]
    if args.target_k_plus_text is not None:
        k_settings.append((None, args.target_k_plus_text, f"k_plus_text={args.target_k_plus_text}"))
    if not k_settings:
        raise ValueError("No K settings requested; pass --ks or --target-k-plus-text.")
    gammas = parse_int_list(args.gammas)

    with output_path.open("w", encoding="utf-8") as f:
        for sample_index, sample in enumerate(bench.iter_samples(args)):
            print(f"[sample {sample_index}] {sample['sample_id']} {sample['video_path']}", flush=True)
            inputs = bench.make_qwen_video_inputs(
                processor,
                sample["video_path"],
                sample["question"],
                args.frame_num,
                args.min_pixels,
                args.max_pixels,
            )
            prompt_len = int(inputs["input_ids"].shape[1])
            ar = bench.ar_generate_qwen25vl(
                model,
                inputs,
                video_token_id=bench.VIDEO_TOKEN_ID,
                eos_token_id=eos_token_id,
                max_new_tokens=args.max_new_tokens,
            )

            for explicit_k, target_k_plus_text, k_label in k_settings:
                for gamma in gammas:
                    print(f"  [std] {k_label} gamma={gamma} verify={args.verify_mode}", flush=True)
                    std, selection = std_generate_qwen25vl(
                        model,
                        inputs,
                        video_token_id=bench.VIDEO_TOKEN_ID,
                        eos_token_id=eos_token_id,
                        max_new_tokens=args.max_new_tokens,
                        gamma=gamma,
                        target_k_plus_text=target_k_plus_text or 1024,
                        explicit_k=explicit_k,
                        verify_mode=args.verify_mode,
                        sparse_attn_mode=args.sparse_attn_mode,
                        copy_sparse_prefill=not args.no_copy_sparse_prefill,
                    )
                    speedup = ar.decoding_time / std.decoding_time if std.decoding_time else 0.0
                    token_equal = bench.tokens_equal(
                        bench.generated_suffix(ar.output_ids, prompt_len),
                        bench.generated_suffix(std.output_ids, prompt_len),
                    )
                    record = {
                        "dataset": args.dataset,
                        "sample_index": sample_index,
                        "sample_id": sample["sample_id"],
                        "frame_num": args.frame_num,
                        "prompt_len": prompt_len,
                        "visual_len": selection.visual_len,
                        "text_len": selection.text_len,
                        "k": selection.k,
                        "explicit_k": explicit_k,
                        "target_k_plus_text": target_k_plus_text,
                        "gamma": gamma,
                        "verify_mode": args.verify_mode,
                        "sparse_attn_mode": args.sparse_attn_mode,
                        "copy_sparse_prefill": not args.no_copy_sparse_prefill,
                        "prompt_style": args.prompt_style,
                        "max_new_tokens": args.max_new_tokens,
                        "token_equal": token_equal,
                        "ar_generate_len": ar.generate_len,
                        "std_generate_len": std.generate_len,
                        "ar_inference_time": ar.inference_time,
                        "std_inference_time": std.inference_time,
                        "ar_decoding_time": ar.decoding_time,
                        "std_decoding_time": std.decoding_time,
                        "speedup": speedup,
                        "accepted_draft_tokens": std.accepted_draft_tokens,
                        "proposed_draft_tokens": std.proposed_draft_tokens,
                        "acceptance_rate": std.acceptance_rate,
                        "mean_accept_length": std.mean_accept_length,
                        "decode_rounds": std.decode_rounds,
                    }
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    del std
                    gc.collect()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
