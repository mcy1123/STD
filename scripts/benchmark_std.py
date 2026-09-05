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
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import av
import torch
from datasets import load_dataset
from tqdm import tqdm


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
    target_device: Optional[str] = "cuda",
    return_timings: bool = False,
):
    probe_start = time.perf_counter()
    fps = video_fps_for_frame_budget(video_path, frame_num)
    video_probe_time = time.perf_counter() - probe_start
    video_content = {
        "type": "video",
        "video": f"file://{video_path}",
        "fps": fps,
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
    decode_start = time.perf_counter()
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    video_decode_sampling_time = time.perf_counter() - decode_start
    # qwen-vl-utils may emit this key for newer processor versions, while the
    # bundled Qwen2.5-VL processor ignores it. Sampling has already happened in
    # process_vision_info(), so dropping it only removes the invalid-argument warning.
    video_kwargs.pop("do_sample_frames", None)
    if video_inputs is None:
        video_items = []
    elif torch.is_tensor(video_inputs):
        video_items = [video_inputs]
    else:
        video_items = list(video_inputs)
    video_input_shapes = [list(video.shape) for video in video_items if hasattr(video, "shape")]
    decoded_frame_count = sum(shape[0] for shape in video_input_shapes if shape)
    processor_start = time.perf_counter()
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    processor_time = time.perf_counter() - processor_start
    transfer_time = 0.0
    if target_device is not None:
        transfer_start = time.perf_counter()
        inputs = inputs.to(target_device)
        torch.cuda.synchronize()
        transfer_time = time.perf_counter() - transfer_start
    timings = {
        "input_cache_hit": False,
        "video_probe_time": video_probe_time,
        "video_decode_sampling_time": video_decode_sampling_time,
        "processor_time": processor_time,
        "input_cache_load_time": 0.0,
        "input_cache_save_time": 0.0,
        "input_transfer_time": transfer_time,
        "decoded_frame_count": decoded_frame_count,
        "video_input_shapes": video_input_shapes,
    }
    return (inputs, timings) if return_timings else inputs


def _input_cache_path(
    cache_dir: str,
    processor,
    sample_id: str,
    video_path: str,
    question: str,
    frame_num: int,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
) -> Path:
    video = Path(video_path).resolve()
    stat = video.stat()
    identity = {
        "processor": getattr(processor, "name_or_path", processor.__class__.__name__),
        "sample_id": str(sample_id),
        "video_path": str(video),
        "video_size": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
        "question": question,
        "frame_num": frame_num,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(sample_id))[:80]
    return Path(cache_dir) / f"{safe_id}_{digest}.pt"


def prepare_qwen_video_inputs(
    processor,
    sample: Dict,
    frame_num: int,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    input_cache_dir: Optional[str],
):
    cache_path = None
    if input_cache_dir:
        cache_path = _input_cache_path(
            input_cache_dir,
            processor,
            sample["sample_id"],
            sample["video_path"],
            sample["question"],
            frame_num,
            min_pixels,
            max_pixels,
        )
        if cache_path.exists():
            load_start = time.perf_counter()
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            cache_load_time = time.perf_counter() - load_start
            if "inputs" in payload and "metadata" in payload:
                inputs = payload["inputs"]
                cached_metadata = payload["metadata"]
            else:
                # Backward compatibility with caches produced before metadata was stored.
                inputs = payload
                cached_metadata = {}
            transfer_start = time.perf_counter()
            inputs = {
                key: value.to("cuda") if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
            torch.cuda.synchronize()
            transfer_time = time.perf_counter() - transfer_start
            return inputs, {
                "input_cache_hit": True,
                "video_probe_time": 0.0,
                "video_decode_sampling_time": 0.0,
                "processor_time": 0.0,
                "input_cache_load_time": cache_load_time,
                "input_cache_save_time": 0.0,
                "input_transfer_time": transfer_time,
                "decoded_frame_count": cached_metadata.get("decoded_frame_count"),
                "video_input_shapes": cached_metadata.get("video_input_shapes", []),
            }

    inputs, timings = make_qwen_video_inputs(
        processor,
        sample["video_path"],
        sample["question"],
        frame_num,
        min_pixels,
        max_pixels,
        target_device=None if cache_path is not None else "cuda",
        return_timings=True,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cpu_inputs = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        save_start = time.perf_counter()
        torch.save(
            {
                "inputs": cpu_inputs,
                "metadata": {
                    "decoded_frame_count": timings.get("decoded_frame_count"),
                    "video_input_shapes": timings.get("video_input_shapes", []),
                },
            },
            cache_path,
        )
        timings["input_cache_save_time"] = time.perf_counter() - save_start
        transfer_start = time.perf_counter()
        inputs = {
            key: value.to("cuda") if torch.is_tensor(value) else value
            for key, value in cpu_inputs.items()
        }
        torch.cuda.synchronize()
        timings["input_transfer_time"] = time.perf_counter() - transfer_start
    return inputs, timings


def reset_peak_memory() -> None:
    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_idx)


def peak_memory_gib() -> float:
    return sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1024**3


class CudaModuleProfiler:
    """Synchronized CUDA timing/counting for one module during prefill profiling."""

    def __init__(self, module, enabled: bool):
        self.module = module
        self.enabled = enabled
        self.elapsed = 0.0
        self.count = 0
        self._start = 0.0
        self._handles = []

    def _pre(self, _module, _inputs):
        torch.cuda.synchronize()
        self._start = time.perf_counter()

    def _post(self, _module, _inputs, _output):
        torch.cuda.synchronize()
        self.elapsed += time.perf_counter() - self._start
        self.count += 1

    def __enter__(self):
        if self.enabled:
            self._handles = [
                self.module.register_forward_pre_hook(self._pre),
                self.module.register_forward_hook(self._post),
            ]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles = []


def prefill_module_profilers(model, enabled: bool):
    return (
        CudaModuleProfiler(model.visual, enabled),
        CudaModuleProfiler(model.visual.merger, enabled),
    )


def build_mismatch_diagnostic(ar_suffix, std_suffix, tokenizer, eos_token_id: int) -> Dict:
    ar_tokens = [int(token) for token in ar_suffix.flatten().detach().cpu().tolist()]
    std_tokens = [int(token) for token in std_suffix.flatten().detach().cpu().tolist()]
    common_prefix = 0
    for ar_token, std_token in zip(ar_tokens, std_tokens):
        if ar_token != std_token:
            break
        common_prefix += 1
    ar_token = ar_tokens[common_prefix] if common_prefix < len(ar_tokens) else None
    std_token = std_tokens[common_prefix] if common_prefix < len(std_tokens) else None
    compared = min(len(ar_tokens), len(std_tokens))
    positional_matches = sum(ar_tokens[i] == std_tokens[i] for i in range(compared))
    return {
        "diagnostic_level": "first_divergence_basic",
        "first_divergence_position": common_prefix,
        "common_prefix_length": common_prefix,
        "ar_token_id": ar_token,
        "std_token_id": std_token,
        "ar_token_text": tokenizer.decode([ar_token]) if ar_token is not None else None,
        "std_token_text": tokenizer.decode([std_token]) if std_token is not None else None,
        "ar_generate_len": len(ar_tokens),
        "std_generate_len": len(std_tokens),
        "token_level_agreement": positional_matches / compared if compared else 1.0,
        "eos_related": ar_token == eos_token_id or std_token == eos_token_id,
        "ar_top1_top2_margin": None,
        "parallel_verifier_top1_top2_margin": None,
        "divergence_round": None,
        "accepted_length_at_divergence": None,
        "bonus_token_state": None,
    }


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
    parser.add_argument(
        "--verify-fallback",
        choices=[
            "none",
            "sequential_on_reject",
            "sequential_on_low_accept",
            "sequential_on_low_margin",
            "sequential_guard",
        ],
        default="none",
    )
    parser.add_argument("--sequential-fallback-max-accept", type=int, default=1)
    parser.add_argument("--verify-margin-threshold", type=float, default=None)
    parser.add_argument(
        "--verify-attn-backend",
        choices=["default", "math", "math_on_full_accept"],
        default="default",
    )
    parser.add_argument("--sparse-attn-mode", choices=["gqa_sdpa", "repeat_sdpa", "triton_gqa"], default="gqa_sdpa")
    parser.add_argument("--strict-equality", action="store_true", help="Abort if STD generated tokens differ from AR.")
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Performance mode: continue after EOS so AR and STD both emit exactly max-new-tokens.",
    )
    parser.add_argument(
        "--no-copy-sparse-prefill",
        action="store_true",
        help="Run a separate sparse prompt prefill instead of copying the normal dense prefill cache.",
    )
    parser.add_argument("--profile-prefill", action="store_true", help="Synchronize and record STD prefill stage timings.")
    parser.add_argument("--profile-decode", action="store_true", help="Synchronize and record STD decode stage timings.")
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument(
        "--input-cache-dir",
        default=None,
        help="Optional directory for cached processor outputs; useful for repeated ablations.",
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--output", default=str(ROOT / "results" / "std_qwen2_5_vl_7b" / "metrics.jsonl"))
    parser.add_argument("--mismatch-output", default=None, help="Optional JSONL path for first-divergence diagnostics.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mismatch_path = Path(args.mismatch_output) if args.mismatch_output else None
    if mismatch_path is not None:
        mismatch_path.parent.mkdir(parents=True, exist_ok=True)

    model, processor = load_qwen_model(args.model_path, args.gpu_ids)
    eos_token_id = processor.tokenizer.eos_token_id

    # Pre-load samples for accurate progress bar
    samples = list(iter_samples(args))
    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}  |  Samples: {len(samples)}  |  Frames: {args.frame_num}")
    print(f"Max tokens: {args.max_new_tokens}  |  Gamma: {args.gamma}  |  K+text: {args.target_k_plus_text}")
    print(f"Sparse attn: {args.sparse_attn_mode}  |  Verify: {args.verify_mode}  |  Prompt: {args.prompt_style}")
    print(f"Profile prefill: {args.profile_prefill}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    total_ar_time = 0.0
    total_std_time = 0.0
    total_ar_inference_time = 0.0
    total_std_inference_time = 0.0
    total_speedups = []
    total_inference_speedups = []
    token_matches = 0

    mismatch_file = mismatch_path.open("w", encoding="utf-8") if mismatch_path is not None else None
    try:
      with output_path.open("w", encoding="utf-8") as f:
        pbar = tqdm(total=len(samples), desc="STD benchmark", unit="sample",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        for sample_index, sample in enumerate(samples):
            pbar.set_postfix_str(f"loading: {sample['sample_id']}")

            input_start = time.perf_counter()
            inputs, input_timings = prepare_qwen_video_inputs(
                processor,
                sample,
                args.frame_num,
                args.min_pixels,
                args.max_pixels,
                args.input_cache_dir,
            )
            input_preparation_time = time.perf_counter() - input_start
            input_runtime_time = sum(
                float(input_timings.get(field, 0.0))
                for field in (
                    "video_probe_time",
                    "video_decode_sampling_time",
                    "processor_time",
                    "input_cache_load_time",
                    "input_transfer_time",
                )
            )
            prompt_len = int(inputs["input_ids"].shape[1])
            print(f"  [{sample_index}] {sample['sample_id']}  prompt_len={prompt_len}", flush=True)

            # AR decoding
            print(f"         AR decoding...", end=" ", flush=True)
            reset_peak_memory()
            t_ar = time.time()
            ar_visual_profiler, ar_projector_profiler = prefill_module_profilers(model, args.profile_prefill)
            with ar_visual_profiler, ar_projector_profiler:
                ar = ar_generate_qwen25vl(
                    model,
                    inputs,
                    video_token_id=VIDEO_TOKEN_ID,
                    eos_token_id=eos_token_id,
                    max_new_tokens=args.max_new_tokens,
                    profile_prefill=args.profile_prefill,
                    ignore_eos=args.ignore_eos,
                )
            ar_peak_memory_gib = peak_memory_gib()
            ar_dt = time.time() - t_ar
            print(f"done ({ar_dt:.1f}s, {ar.generate_len} tokens)", flush=True)

            # STD decoding
            print(f"         STD decoding...", end=" ", flush=True)
            reset_peak_memory()
            t_std = time.time()
            std_visual_profiler, std_projector_profiler = prefill_module_profilers(model, args.profile_prefill)
            with std_visual_profiler, std_projector_profiler:
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
                    verify_fallback=args.verify_fallback,
                    sequential_fallback_max_accept=args.sequential_fallback_max_accept,
                    profile_decode=args.profile_decode,
                    sparse_attn_mode=args.sparse_attn_mode,
                    copy_sparse_prefill=not args.no_copy_sparse_prefill,
                    profile_prefill=args.profile_prefill,
                    verify_attn_backend=args.verify_attn_backend,
                    verify_margin_threshold=args.verify_margin_threshold,
                    ignore_eos=args.ignore_eos,
                )
            std_peak_memory_gib = peak_memory_gib()
            std_dt = time.time() - t_std
            print(f"done ({std_dt:.1f}s, {std.generate_len} tokens)", flush=True)
            if args.ignore_eos and (
                ar.generate_len != args.max_new_tokens or std.generate_len != args.max_new_tokens
            ):
                raise RuntimeError(
                    "Fixed-token performance mode failed: "
                    f"AR={ar.generate_len}, STD={std.generate_len}, expected={args.max_new_tokens}."
                )

            speedup = ar.decoding_time / std.decoding_time if std.decoding_time else 0.0
            inference_speedup = ar.inference_time / std.inference_time if std.inference_time else 0.0
            ar_tokens_per_second = ar.generate_len / ar.decoding_time if ar.decoding_time else 0.0
            std_tokens_per_second = std.generate_len / std.decoding_time if std.decoding_time else 0.0
            application_speedup = (
                (input_runtime_time + ar.inference_time) / (input_runtime_time + std.inference_time)
                if input_runtime_time + std.inference_time else 0.0
            )
            token_equal = tokens_equal(
                generated_suffix(ar.output_ids, prompt_len),
                generated_suffix(std.output_ids, prompt_len),
            )
            agreement_diagnostic = build_mismatch_diagnostic(
                generated_suffix(ar.output_ids, prompt_len),
                generated_suffix(std.output_ids, prompt_len),
                processor.tokenizer,
                eos_token_id,
            )
            if args.strict_equality and not token_equal:
                raise RuntimeError(f"STD token mismatch on sample {sample['sample_id']}.")

            total_ar_time += ar.decoding_time
            total_std_time += std.decoding_time
            total_ar_inference_time += ar.inference_time
            total_std_inference_time += std.inference_time
            total_speedups.append(speedup)
            total_inference_speedups.append(inference_speedup)
            token_matches += int(token_equal)
            avg_speedup = sum(total_speedups) / len(total_speedups)

            pbar.set_postfix_str(
                f"AR={ar.decoding_time:.1f}s STD={std.decoding_time:.1f}s "
                f"speedup={speedup:.2f}x acc={std.acceptance_rate:.2f} "
                f"ok={'✓' if token_equal else '✗'}"
            )
            pbar.update(1)
            print(f"         AR={ar.decoding_time:.2f}s  STD={std.decoding_time:.2f}s  "
                  f"speedup={speedup:.2f}x  accept={std.acceptance_rate:.2f}  "
                  f"inference={inference_speedup:.2f}x  "
                  f"match={'✓' if token_equal else '✗'}  "
                  f"avg_speedup={avg_speedup:.2f}x", flush=True)

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
                "verify_fallback": args.verify_fallback,
                "verify_margin_threshold": args.verify_margin_threshold,
                "sequential_fallback_max_accept": args.sequential_fallback_max_accept,
                "verify_attn_backend": args.verify_attn_backend,
                "sparse_attn_mode": args.sparse_attn_mode,
                "copy_sparse_prefill": not args.no_copy_sparse_prefill,
                "profile_decode": args.profile_decode,
                "profile_prefill": args.profile_prefill,
                "prompt_style": args.prompt_style,
                "max_new_tokens": args.max_new_tokens,
                "ignore_eos": args.ignore_eos,
                "input_cache_dir": args.input_cache_dir,
                **input_timings,
                "input_preparation_time": input_preparation_time,
                "input_runtime_time": input_runtime_time,
                "token_equal": token_equal,
                "token_level_agreement": agreement_diagnostic["token_level_agreement"],
                "common_prefix_length": agreement_diagnostic["common_prefix_length"],
                "ar_generate_len": ar.generate_len,
                "std_generate_len": std.generate_len,
                "ar_inference_time": ar.inference_time,
                "std_inference_time": std.inference_time,
                "ar_decoding_time": ar.decoding_time,
                "std_decoding_time": std.decoding_time,
                "speedup": speedup,
                "inference_speedup": inference_speedup,
                "application_speedup": application_speedup,
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
                "fallback_count": std.fallback_count,
                "fallback_accepted_extra": std.fallback_accepted_extra,
                "verify_margin_reruns": std.verify_margin_reruns,
                "min_verify_margin": std.min_verify_margin,
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
                "ar_vision_encoder_time": max(0.0, ar_visual_profiler.elapsed - ar_projector_profiler.elapsed),
                "ar_projector_time": ar_projector_profiler.elapsed,
                "ar_dense_lm_prefill_time": max(0.0, ar.prefill_time - ar_visual_profiler.elapsed),
                "ar_vision_encoder_forward_count": ar_visual_profiler.count,
                "ar_projector_forward_count": ar_projector_profiler.count,
                "ar_prompt_prefill_count": 1,
                "std_vision_encoder_time": max(0.0, std_visual_profiler.elapsed - std_projector_profiler.elapsed),
                "std_projector_time": std_projector_profiler.elapsed,
                "std_dense_lm_prefill_time": max(0.0, std.selection_prefill_time - std_visual_profiler.elapsed),
                "std_vision_encoder_forward_count": std_visual_profiler.count,
                "std_projector_forward_count": std_projector_profiler.count,
                "std_prompt_prefill_count": 1,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            if mismatch_file is not None and not token_equal:
                diagnostic = dict(agreement_diagnostic)
                diagnostic.update({
                    "dataset": args.dataset,
                    "sample_index": sample_index,
                    "sample_id": sample["sample_id"],
                    "frame_num": args.frame_num,
                    "prompt_len": prompt_len,
                    "k": selection.k,
                    "gamma": args.gamma,
                })
                mismatch_file.write(json.dumps(diagnostic, ensure_ascii=False) + "\n")
                mismatch_file.flush()

        pbar.close()
        print(f"\n{'='*60}")
        print(f"Done. {len(samples)} samples → {output_path}")
        print(f"Total AR decoding:  {total_ar_time:.1f}s")
        print(f"Total STD decoding: {total_std_time:.1f}s")
        print(f"Overall speedup:     {total_ar_time / total_std_time:.2f}x" if total_std_time > 0 else "")
        print(f"Mean per-sample:     {sum(total_speedups) / len(total_speedups):.2f}x")
        print(
            f"Inference speedup:   {total_ar_inference_time / total_std_inference_time:.2f}x"
            if total_std_inference_time > 0 else ""
        )
        print(f"Mean inference:      {sum(total_inference_speedups) / len(total_inference_speedups):.2f}x")
        print(f"Token match:         {token_matches}/{len(total_speedups)}")
        print(f"{'='*60}")
    finally:
        if mismatch_file is not None:
            mismatch_file.close()


if __name__ == "__main__":
    main()
