#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PHASE="${1:-baseline}"
CONDA_ENV="${CONDA_ENV:-specvlm}"
GPU_IDS="${GPU_IDS:-0}"
EVAL_NUM="${EVAL_NUM:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_CACHE_LEN="${MAX_CACHE_LEN:-40960}"
MAX_PIXELS="${MAX_PIXELS:-200704}"
MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-VL-7B-Instruct}"
DATA_PATH="${DATA_PATH:-datasets/Video-MME}"
VIDEO_ROOT="${VIDEO_ROOT:-datasets/Video-MME/videos}"
RESULT_DIR="${RESULT_DIR:-results/h100_next_stage}"
INPUT_CACHE_DIR="${INPUT_CACHE_DIR:-cache/processor_qwen25vl}"
BEST_K_PLUS_TEXT="${BEST_K_PLUS_TEXT:-4096}"
BEST_GAMMA="${BEST_GAMMA:-9}"

export SPECVLM_MAX_CACHE_LEN="$MAX_CACHE_LEN"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$RESULT_DIR" "$INPUT_CACHE_DIR"

COMMON=(
  --model-path "$MODEL_PATH"
  --dataset Video-MME
  --data-path "$DATA_PATH"
  --video-root "$VIDEO_ROOT"
  --prompt-style cot
  --max-new-tokens "$MAX_NEW_TOKENS"
  --max-pixels "$MAX_PIXELS"
  --verify-mode parallel
  --sparse-attn-mode gqa_sdpa
  --input-cache-dir "$INPUT_CACHE_DIR"
  --gpu-ids "$GPU_IDS"
)

run_python() {
  conda run --no-capture-output -n "$CONDA_ENV" python "$@"
}

case "$PHASE" in
  baseline)
    OUTPUT="$RESULT_DIR/baseline_fixed_tokens.jsonl"
    run_python scripts/benchmark_std.py \
      "${COMMON[@]}" \
      --eval-num "$EVAL_NUM" \
      --frame-num 256 \
      --gamma 9 \
      --target-k-plus-text 8192 \
      --ignore-eos \
      --output "$OUTPUT"
    run_python scripts/summarize_metrics.py "$OUTPUT"
    ;;

  correctness)
    OUTPUT="$RESULT_DIR/correctness_fixed_tokens.jsonl"
    MISMATCH="$RESULT_DIR/mismatch_diagnostics.jsonl"
    run_python scripts/benchmark_std.py \
      "${COMMON[@]}" \
      --eval-num "$EVAL_NUM" \
      --frame-num 256 \
      --gamma 9 \
      --target-k-plus-text 8192 \
      --mismatch-output "$MISMATCH" \
      --output "$OUTPUT"
    run_python scripts/summarize_metrics.py "$OUTPUT"
    ;;

  prefill)
    OUTPUT="$RESULT_DIR/prefill_breakdown.jsonl"
    run_python scripts/benchmark_std.py \
      "${COMMON[@]}" \
      --eval-num "${PREFILL_EVAL_NUM:-3}" \
      --frame-num 256 \
      --gamma 9 \
      --target-k-plus-text 8192 \
      --ignore-eos \
      --profile-prefill \
      --profile-decode \
      --output "$OUTPUT"
    run_python scripts/summarize_metrics.py "$OUTPUT"
    ;;

  k)
    OUTPUT="$RESULT_DIR/k_ablation.jsonl"
    run_python scripts/sweep_std_inprocess.py \
      "${COMMON[@]}" \
      --eval-num "$EVAL_NUM" \
      --frame-num 256 \
      --ks "" \
      --k-plus-texts 8192,4096,2048,1024 \
      --gammas 9 \
      --ignore-eos \
      --profile-decode \
      --output "$OUTPUT"
    run_python scripts/export_ablation_summary.py "$OUTPUT" \
      --group-by target_k_plus_text \
      --output "$RESULT_DIR/k_ablation_summary.json"
    ;;

  gamma)
    OUTPUT="$RESULT_DIR/gamma_ablation.jsonl"
    run_python scripts/sweep_std_inprocess.py \
      "${COMMON[@]}" \
      --eval-num "$EVAL_NUM" \
      --frame-num 256 \
      --ks "" \
      --k-plus-texts "$BEST_K_PLUS_TEXT" \
      --gammas 5,7,9,12 \
      --ignore-eos \
      --profile-decode \
      --output "$OUTPUT"
    run_python scripts/export_ablation_summary.py "$OUTPUT" \
      --group-by target_k_plus_text,gamma \
      --output "$RESULT_DIR/gamma_ablation_summary.json"
    ;;

  context)
    PARTS=()
    for FRAMES in 32 64 128 256; do
      OUTPUT="$RESULT_DIR/context_${FRAMES}f.jsonl"
      run_python scripts/benchmark_std.py \
        "${COMMON[@]}" \
        --eval-num "$EVAL_NUM" \
        --frame-num "$FRAMES" \
        --gamma "$BEST_GAMMA" \
        --target-k-plus-text "$BEST_K_PLUS_TEXT" \
        --ignore-eos \
        --output "$OUTPUT"
      PARTS+=("$OUTPUT")
    done
    MERGED="$RESULT_DIR/context_crossover.jsonl"
    run_python scripts/merge_jsonl.py "${PARTS[@]}" --output "$MERGED"
    run_python scripts/export_ablation_summary.py "$MERGED" \
      --group-by frame_num \
      --output "$RESULT_DIR/context_crossover_summary.json"
    ;;

  *)
    echo "Usage: bash scripts/run_h100_next_stage.sh {baseline|correctness|prefill|k|gamma|context}" >&2
    exit 2
    ;;
esac
