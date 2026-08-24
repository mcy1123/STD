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
    parser.add_argument("--k-plus-texts", default="", help="Comma-separated K+text budgets to sweep.")
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
    parser.add_argument("--input-cache-dir", default=None)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--profile-prefill", action="store_true")
    parser.add_argument("--profile-decode", action="store_true")
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
    for k_plus_text in parse_int_list(args.k_plus_texts):
        k_settings.append((None, k_plus_text, f"k_plus_text={k_plus_text}"))
    if not k_settings:
        raise ValueError("No K settings requested; pass --ks or --target-k-plus-text.")
    gammas = parse_int_list(args.gammas)

    with output_path.open("w", encoding="utf-8") as f:
        for sample_index, sample in enumerate(bench.iter_samples(args)):
            print(f"[sample {sample_index}] {sample['sample_id']} {sample['video_path']}", flush=True)
            inputs, input_timings = bench.prepare_qwen_video_inputs(
                processor,
                sample,
                args.frame_num,
                args.min_pixels,
                args.max_pixels,
                args.input_cache_dir,
            )
            prompt_len = int(inputs["input_ids"].shape[1])
            bench.reset_peak_memory()
            ar = bench.ar_generate_qwen25vl(
                model,
                inputs,
                video_token_id=bench.VIDEO_TOKEN_ID,
                eos_token_id=eos_token_id,
                max_new_tokens=args.max_new_tokens,
                profile_prefill=args.profile_prefill,
                ignore_eos=args.ignore_eos,
            )
            ar_peak_memory_gib = bench.peak_memory_gib()

            for explicit_k, target_k_plus_text, k_label in k_settings:
                for gamma in gammas:
                    print(f"  [std] {k_label} gamma={gamma} verify={args.verify_mode}", flush=True)
                    bench.reset_peak_memory()
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
                        profile_prefill=args.profile_prefill,
                        profile_decode=args.profile_decode,
                        ignore_eos=args.ignore_eos,
                    )
                    std_peak_memory_gib = bench.peak_memory_gib()
                    speedup = ar.decoding_time / std.decoding_time if std.decoding_time else 0.0
                    inference_speedup = ar.inference_time / std.inference_time if std.inference_time else 0.0
                    ar_tokens_per_second = ar.generate_len / ar.decoding_time if ar.decoding_time else 0.0
                    std_tokens_per_second = std.generate_len / std.decoding_time if std.decoding_time else 0.0
                    token_equal = bench.tokens_equal(
                        bench.generated_suffix(ar.output_ids, prompt_len),
                        bench.generated_suffix(std.output_ids, prompt_len),
                    )
                    agreement = bench.build_mismatch_diagnostic(
                        bench.generated_suffix(ar.output_ids, prompt_len),
                        bench.generated_suffix(std.output_ids, prompt_len),
                        processor.tokenizer,
                        eos_token_id,
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
                        "profile_prefill": args.profile_prefill,
                        "profile_decode": args.profile_decode,
                        "prompt_style": args.prompt_style,
                        "max_new_tokens": args.max_new_tokens,
                        "ignore_eos": args.ignore_eos,
                        "input_cache_dir": args.input_cache_dir,
                        **input_timings,
                        "token_equal": token_equal,
                        "token_level_agreement": agreement["token_level_agreement"],
                        "common_prefix_length": agreement["common_prefix_length"],
                        "ar_generate_len": ar.generate_len,
                        "std_generate_len": std.generate_len,
                        "ar_inference_time": ar.inference_time,
                        "std_inference_time": std.inference_time,
                        "ar_decoding_time": ar.decoding_time,
                        "std_decoding_time": std.decoding_time,
                        "speedup": speedup,
                        "inference_speedup": inference_speedup,
                        "ar_tokens_per_second": ar_tokens_per_second,
                        "std_tokens_per_second": std_tokens_per_second,
                        "committed_tokens_per_second": std_tokens_per_second,
                        "ar_peak_memory_gib": ar_peak_memory_gib,
                        "std_peak_memory_gib": std_peak_memory_gib,
                        "accepted_draft_tokens": std.accepted_draft_tokens,
                        "proposed_draft_tokens": std.proposed_draft_tokens,
                        "acceptance_rate": std.acceptance_rate,
                        "mean_accept_length": std.mean_accept_length,
                        "decode_rounds": std.decode_rounds,
                        "draft_time_per_round": std.draft_time / std.decode_rounds if std.decode_rounds else 0.0,
                        "verify_time_per_round": std.verify_time / std.decode_rounds if std.decode_rounds else 0.0,
                        "committed_tokens_per_round": std.generate_len / std.decode_rounds if std.decode_rounds else 0.0,
                        "gamma_history": std.gamma_history,
                        "proposed_lengths": std.proposed_lengths,
                        "accept_lengths": std.accept_lengths,
                        "draft_time": std.draft_time,
                        "verify_time": std.verify_time,
                        "bonus_time": std.bonus_time,
                        "cache_adjust_time": std.cache_adjust_time,
                        "ar_cache_init_time": ar.cache_init_time,
                        "ar_prefill_time": ar.prefill_time,
                        "std_cache_init_time": std.cache_init_time,
                        "std_prefill_time": std.prefill_time,
                        "std_selection_prefill_time": std.selection_prefill_time,
                        "std_selection_time": std.selection_time,
                        "std_dense_prefill_time": std.dense_prefill_time,
                        "std_sparse_cache_time": std.sparse_cache_time,
                    }
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    del std
                    gc.collect()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
