#!/usr/bin/env python
"""Probe Qwen2.5-VL input token counts without loading the model."""

from __future__ import annotations

import argparse
from pathlib import Path

import benchmark_std as bench
from specvlm.models.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info


def make_inputs_cpu(processor, video_path: str, question: str, frame_num: int, max_pixels: int):
    video_content = {
        "type": "video",
        "video": f"file://{video_path}",
        "fps": bench.video_fps_for_frame_budget(video_path, frame_num),
        "max_pixels": max_pixels,
    }
    messages = [{"role": "user", "content": [video_content, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--dataset", choices=["VideoDetailCaption", "MLVU", "Video-MME"], default="Video-MME")
    parser.add_argument("--data-path", default=str(Path(__file__).resolve().parents[1] / "datasets" / "Video-MME"))
    parser.add_argument("--video-root", default=str(Path(__file__).resolve().parents[1] / "datasets" / "Video-MME" / "videos"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--prompt-style", choices=["direct", "cot"], default="cot")
    parser.add_argument("--eval-num", type=int, default=1)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--frame-nums", default="144,160,192,224,256")
    parser.add_argument("--max-pixels", default="200704,150528,112896,100352,75264,50176")
    args = parser.parse_args()

    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path)
    frame_nums = [int(x) for x in args.frame_nums.split(",") if x]
    max_pixels_values = [int(x) for x in args.max_pixels.split(",") if x]
    sample = next(iter(bench.iter_samples(args)))

    print(f"sample_id={sample['sample_id']} video={Path(sample['video_path']).name}")
    for frame_num in frame_nums:
        for max_pixels in max_pixels_values:
            try:
                inputs = make_inputs_cpu(processor, sample["video_path"], sample["question"], frame_num, max_pixels)
                input_ids = inputs["input_ids"][0]
                visual_len = int((input_ids == bench.VIDEO_TOKEN_ID).sum().item())
                print(
                    f"frame_num={frame_num} max_pixels={max_pixels} "
                    f"prompt_len={int(input_ids.numel())} visual_len={visual_len}"
                )
            except Exception as exc:
                print(f"frame_num={frame_num} max_pixels={max_pixels} error={type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
