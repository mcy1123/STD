#!/usr/bin/env python
"""Benchmark Sparse-to-Dense on Qwen2.5-VL-7B.

Examples:
  conda run -n specvlm python scripts/benchmark_std.py \
    --dataset VideoDetailCaption \
    --data-path datasets/VideoDetailCaption \
    --eval-num 1 --frame-num 32 --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import av
import torch
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from specvlm.models.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration  # noqa: E402
from specvlm.models.processing_qwen2_5_vl import Qwen2_5_VLProcessor  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402
from std_repro.std_qwen25vl import (  # noqa: E402
    ar_generate_qwen25vl,
    generated_suffix,
    std_generate_qwen25vl,
    tokens_equal,
)


VIDEO_TOKEN_ID = 151656


def build_max_memory(gpu_ids: str) -> Optional[Dict[int, str]]:
    if not gpu_ids:
        return None
    max_mem = {}
    for local_id, _physical_id in enumerate(gpu_ids.split(",")):
        props = torch.cuda.get_device_properties(local_id)
        usable_mib = max(1, (props.total_memory - 2 * 1024**3) // 1024**2)
        max_mem[local_id] = f"{usable_mib}MiB"
    return max_mem


def load_qwen_model(model_path: str, gpu_ids: str):
    max_memory = build_max_memory(gpu_ids)
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.eval()
    return model, processor


def video_fps_for_frame_budget(video_path: str, frame_num: int) -> float:
    container = av.open(video_path)
    try:
        duration = container.duration / 1_000_000 if container.duration else 0
    finally:
        container.close()
    if duration <= 0:
        return 1.0
    return max(0.01, frame_num / duration)


def make_qwen_video_inputs(
    processor,
    video_path: str,
    question: str,
    frame_num: int,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
) -> Dict[str, torch.Tensor]:
    video_content = {
        "type": "video",
        "video": f"file://{video_path}",
        "fps": video_fps_for_frame_budget(video_path, frame_num),
        "max_pixels": max_pixels if max_pixels is not None else 448 * 448,
    }
    if min_pixels is not None:
        video_content["min_pixels"] = min_pixels

    messages = [
        {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    return inputs.to("cuda")


def _find_existing_video(base_dir: Path, stem: str) -> Optional[str]:
    candidate = Path(stem)
    if candidate.exists():
        return str(candidate)
    for ext in ("", ".mp4", ".mkv", ".webm", ".avi", ".mov"):
        path = base_dir / f"{stem}{ext}"
        if path.exists():
            return str(path)
    if base_dir.exists():
        for ext in (".mp4", ".mkv", ".webm", ".avi", ".mov"):
            matches = list(base_dir.rglob(f"{stem}{ext}"))
            if matches:
                return str(matches[0])
    return None


def _question_with_options(row: Dict, prompt_style: str = "direct") -> str:
    question = row.get("question") or row.get("Question") or row.get("query") or ""
    options = row.get("options") or row.get("candidates") or row.get("choices")
    if isinstance(options, (list, tuple)):
        option_text = "\n".join(str(x) for x in options)
        if prompt_style == "cot":
            return f"{question}\n{option_text}\nThink step by step, then provide the final answer option."
        return f"{question}\n{option_text}\nOnly give the best option."
    if prompt_style == "cot":
        return f"{question}\nThink step by step before answering."
    return str(question)


def iter_videodetailcaption(data_path: str, limit: int):
    ds = load_dataset(data_path, split="test").shuffle(seed=42)
    video_dir = Path(data_path) / "Test_Videos"
    yielded = 0
    for row in ds:
        path = _find_existing_video(video_dir, row["video_name"])
        if path is None:
            continue
        yield {
            "sample_id": row.get("video_name", str(yielded)),
            "video_path": path,
            "question": row.get("question", "Describe the video in detail."),
        }
        yielded += 1
        if yielded >= limit:
            break


def iter_generic_hf_video(
    dataset_name_or_path: str,
    split: str,
    limit: int,
    video_root: Optional[str],
    prompt_style: str = "direct",
):
    ds = load_dataset(dataset_name_or_path, split=split).shuffle(seed=42)
    base_dir = Path(video_root or dataset_name_or_path)
    yielded = 0
    for idx, row in enumerate(ds):
        video_path = None
        for key in ("video_path", "video", "path", "filepath", "file", "video_name", "videoID", "video_id"):
            if key not in row:
                continue
            value = row[key]
            if isinstance(value, dict):
                value = value.get("path") or value.get("filename")
            if isinstance(value, str):
                video_path = _find_existing_video(base_dir, value)
            if video_path:
                break
        if not video_path:
            continue
        yield {
            "sample_id": row.get("question_id") or row.get("id") or row.get("video_id") or str(idx),
            "video_path": video_path,
            "question": _question_with_options(row, prompt_style=prompt_style),
            "duration": row.get("duration"),
        }
        yielded += 1
        if yielded >= limit:
            break


def iter_samples(args):
    limit = args.eval_num + args.skip_samples
    if args.dataset == "VideoDetailCaption":
        source = iter_videodetailcaption(args.data_path, limit)
    elif args.dataset in {"MLVU", "Video-MME"}:
        source = iter_generic_hf_video(args.data_path, args.split, limit, args.video_root, args.prompt_style)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    for idx, sample in enumerate(source):
        if idx < args.skip_samples:
            continue
        yield sample


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
    parser.add_argument("--frame-num", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--gamma", type=int, default=9)
    parser.add_argument("--target-k-plus-text", type=int, default=1024)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--verify-mode", choices=["parallel", "sequential"], default="parallel")
    parser.add_argument("--sparse-attn-mode", choices=["gqa_sdpa", "repeat_sdpa", "triton_gqa"], default="gqa_sdpa")
    parser.add_argument("--strict-equality", action="store_true", help="Abort if STD generated tokens differ from AR.")
    parser.add_argument(
        "--no-copy-sparse-prefill",
        action="store_true",
        help="Run a separate sparse prompt prefill instead of copying the normal dense prefill cache.",
    )
    parser.add_argument("--profile-decode", action="store_true", help="Synchronize and record STD decode stage timings.")
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--output", default=str(ROOT / "results" / "std_qwen2_5_vl_7b" / "metrics.jsonl"))
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, processor = load_qwen_model(args.model_path, args.gpu_ids)
    eos_token_id = processor.tokenizer.eos_token_id

    with output_path.open("w", encoding="utf-8") as f:
        for sample_index, sample in enumerate(iter_samples(args)):
            print(f"[{sample_index}] {sample['sample_id']} {sample['video_path']}", flush=True)
            inputs = make_qwen_video_inputs(
                processor,
                sample["video_path"],
                sample["question"],
                args.frame_num,
                args.min_pixels,
                args.max_pixels,
            )
            prompt_len = int(inputs["input_ids"].shape[1])

            ar = ar_generate_qwen25vl(
                model,
                inputs,
                video_token_id=VIDEO_TOKEN_ID,
                eos_token_id=eos_token_id,
                max_new_tokens=args.max_new_tokens,
            )
            std, selection = std_generate_qwen25vl(
                model,
                inputs,
                video_token_id=VIDEO_TOKEN_ID,
                eos_token_id=eos_token_id,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                target_k_plus_text=args.target_k_plus_text,
                explicit_k=args.k,
                verify_mode=args.verify_mode,
                profile_decode=args.profile_decode,
                sparse_attn_mode=args.sparse_attn_mode,
                copy_sparse_prefill=not args.no_copy_sparse_prefill,
            )

            speedup = ar.decoding_time / std.decoding_time if std.decoding_time else 0.0
            token_equal = tokens_equal(
                generated_suffix(ar.output_ids, prompt_len),
                generated_suffix(std.output_ids, prompt_len),
            )
            if args.strict_equality and not token_equal:
                raise RuntimeError(f"STD token mismatch on sample {sample['sample_id']}.")

            record = {
                "dataset": args.dataset,
                "sample_index": sample_index,
                "sample_id": sample["sample_id"],
                "frame_num": args.frame_num,
                "prompt_len": prompt_len,
                "visual_len": selection.visual_len,
                "text_len": selection.text_len,
                "k": selection.k,
                "gamma": args.gamma,
                "verify_mode": args.verify_mode,
                "sparse_attn_mode": args.sparse_attn_mode,
                "copy_sparse_prefill": not args.no_copy_sparse_prefill,
                "profile_decode": args.profile_decode,
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
                "draft_time": std.draft_time,
                "verify_time": std.verify_time,
                "bonus_time": std.bonus_time,
                "cache_adjust_time": std.cache_adjust_time,
            }
            print(json.dumps(record, ensure_ascii=False), flush=True)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
