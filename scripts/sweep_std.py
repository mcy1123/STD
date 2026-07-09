#!/usr/bin/env python
"""Run a small K/gamma sweep for STD reproduction."""

from __future__ import annotations

import argparse
import itertools
import subprocess
from pathlib import Path


def parse_int_list(value: str):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", default="512,1024,2048,4096")
    parser.add_argument("--gammas", default="5,7,9,11")
    parser.add_argument("--frame-num", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--eval-num", type=int, default=1)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--verify-mode", choices=["parallel", "sequential"], default="parallel")
    parser.add_argument("--sparse-attn-mode", choices=["gqa_sdpa", "repeat_sdpa", "triton_gqa"], default="gqa_sdpa")
    parser.add_argument("--strict-equality", action="store_true")
    parser.add_argument("--dataset", default="VideoDetailCaption")
    parser.add_argument("--data-path", default=str(Path(__file__).resolve().parents[1] / "datasets" / "Video-MME"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "results" / "std_qwen2_5_vl_7b" / "sweeps"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for k, gamma in itertools.product(parse_int_list(args.ks), parse_int_list(args.gammas)):
        output = output_dir / f"{args.dataset}_{args.frame_num}f_{args.max_new_tokens}tok_k{k}_g{gamma}.jsonl"
        cmd = [
            "conda",
            "run",
            "-n",
            "specvlm",
            "python",
            "scripts/benchmark_std.py",
            "--dataset",
            args.dataset,
            "--data-path",
            args.data_path,
            "--eval-num",
            str(args.eval_num),
            "--frame-num",
            str(args.frame_num),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--gamma",
            str(gamma),
            "--k",
            str(k),
            "--gpu-ids",
            args.gpu_ids,
            "--verify-mode",
            args.verify_mode,
            "--sparse-attn-mode",
            args.sparse_attn_mode,
            "--output",
            str(output),
        ]
        if args.strict_equality:
            cmd.append("--strict-equality")
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
