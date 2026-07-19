#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-specvlm}"
GPU_IDS="${GPU_IDS:-0}"
EVAL_NUM="${EVAL_NUM:-3}"
FRAME_NUM="${FRAME_NUM:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
GAMMA="${GAMMA:-9}"
K_PLUS_TEXT="${K_PLUS_TEXT:-8192}"
MAX_CACHE_LEN="${MAX_CACHE_LEN:-40960}"
MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-VL-7B-Instruct}"
DATA_PATH="${DATA_PATH:-datasets/Video-MME}"
VIDEO_ROOT="${VIDEO_ROOT:-datasets/Video-MME/videos}"
OUTPUT="${OUTPUT:-results/h100_videomme_256f_256tok_kplus8192_g9_reuse_prefill.jsonl}"
PROFILE="${PROFILE:-1}"

export SPECVLM_MAX_CACHE_LEN="$MAX_CACHE_LEN"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Running H100 STD with dense-prefill reuse"
echo "GPU=$GPU_IDS samples=$EVAL_NUM frames=$FRAME_NUM output_tokens=$MAX_NEW_TOKENS"
echo "gamma=$GAMMA K+text=$K_PLUS_TEXT max_cache_len=$MAX_CACHE_LEN"
echo "profile=$PROFILE output=$OUTPUT"

PROFILE_ARGS=()
if [[ "$PROFILE" == "1" ]]; then
  PROFILE_ARGS+=(--profile-prefill --profile-decode)
fi

conda run --no-capture-output -n "$CONDA_ENV" \
  python scripts/benchmark_std.py \
  --model-path "$MODEL_PATH" \
  --dataset Video-MME \
  --data-path "$DATA_PATH" \
  --video-root "$VIDEO_ROOT" \
  --eval-num "$EVAL_NUM" \
  --frame-num "$FRAME_NUM" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --gamma "$GAMMA" \
  --target-k-plus-text "$K_PLUS_TEXT" \
  --prompt-style cot \
  --verify-mode parallel \
  --sparse-attn-mode gqa_sdpa \
  --reuse-dense-prefill \
  "${PROFILE_ARGS[@]}" \
  --gpu-ids "$GPU_IDS" \
  --output "$OUTPUT"

conda run --no-capture-output -n "$CONDA_ENV" \
  python scripts/summarize_metrics.py "$OUTPUT"
