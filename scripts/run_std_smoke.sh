#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mcy/projects/Std"
cd "$ROOT"

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export SPECVLM_MAX_CACHE_LEN="${SPECVLM_MAX_CACHE_LEN:-40960}"

conda run -n specvlm python scripts/benchmark_std.py \
  --model-path /home/mcy/projects/models/Qwen2.5-VL-7B-Instruct \
  --dataset VideoDetailCaption \
  --data-path /home/mcy/projects/SpecVLM/datasets/VideoDetailCaption \
  --eval-num 1 \
  --frame-num 32 \
  --max-new-tokens 64 \
  --gamma 9 \
  --target-k-plus-text 1024 \
  --gpu-ids "${GPU_IDS:-0,1,2,3}" \
  --strict-equality \
  --output results/std_qwen2_5_vl_7b/smoke_videodetailcaption.jsonl
