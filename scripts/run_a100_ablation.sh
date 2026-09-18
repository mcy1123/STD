#!/usr/bin/env bash
# Dynamic STD ablation rerun on the ECNU Phase-8 A100 node.
#
# Design and decision gates:
#   docs/superpowers/plans/2026-09-17-dynamic-std-ablation-rerun.md
#
# SAFETY: this driver uses physical GPU 1 only. GPU 0 hosts an unrelated vLLM
# workload and must never be touched. The benchmark itself also refuses to start
# unless the selected physical GPU is idle, so a busy GPU fails fast instead of
# contending.
#
# Usage:
#   bash scripts/run_a100_ablation.sh stage0   # cheap reproduce + fallback A/B
#   bash scripts/run_a100_ablation.sh stage1   # 6 single-axis ablations
#
# Overridable environment:
#   REPO PY ASSETS MODEL DATA VIDEOS OUTDIR GPU LIMIT REPEATS FRAMES TOKENS
set -euo pipefail

REPO="${REPO:-/public/home/xlwang/mcy/Project/STD-latest}"
PY="${PY:-/public/home/xlwang/mcy/conda_envs/specvlm/bin/python}"
ASSETS="${ASSETS:-/public/home/xlwang/mcy/STD_assets}"
MODEL="${MODEL:-$ASSETS/models/Qwen2.5-VL-7B-Instruct}"
DATA="${DATA:-$ASSETS/datasets/Video-MME}"
VIDEOS="${VIDEOS:-$ASSETS/datasets/Video-MME/videos}"
GPU="${GPU:-1}"
STAGE="${1:-stage1}"

case "$STAGE" in
  stage0) DEFAULT_LIMIT=3;  DEFAULT_TOKENS=128 ;;
  stage1) DEFAULT_LIMIT=10; DEFAULT_TOKENS=128 ;;
  *) echo "unknown stage '$STAGE' (expected stage0 or stage1)" >&2; exit 2 ;;
esac

LIMIT="${LIMIT:-$DEFAULT_LIMIT}"
REPEATS="${REPEATS:-2}"
FRAMES="${FRAMES:-128}"
TOKENS="${TOKENS:-$DEFAULT_TOKENS}"

OUTDIR="${OUTDIR:-$ASSETS/results/ablation_$(date +%Y%m%d_%H%M%S)_${STAGE}}"

if [[ "$GPU" == "0" ]]; then
  echo "refusing to run on GPU 0: it hosts an unrelated vLLM workload" >&2
  exit 3
fi
if [[ ! -x "$PY" ]]; then
  echo "python not found or not executable: $PY" >&2
  exit 4
fi

mkdir -p "$OUTDIR"

run_case() {
  local name="$1"; shift
  local out="$OUTDIR/${name}.jsonl"
  if [[ -e "$out" ]]; then
    echo "refusing to overwrite existing output: $out" >&2
    exit 5
  fi
  echo
  echo "==============================================================="
  echo "[$name] $(date -Iseconds)  ->  $out"
  echo "==============================================================="
  "$PY" "$REPO/scripts/benchmark_a100_dynamic.py" \
    --gpu "$GPU" \
    --model-path "$MODEL" \
    --data-path "$DATA" \
    --video-root "$VIDEOS" \
    --output "$out" \
    --frame-num "$FRAMES" \
    --max-new-tokens "$TOKENS" \
    --limit "$LIMIT" \
    --repeats "$REPEATS" \
    --gamma 9 \
    --k-plus-text 1024 \
    --dynamic-collector v2 \
    --profile-components \
    --assert-equal-s0 \
    --assert-consistency \
    "$@"
  "$PY" "$REPO/scripts/summarize_a100_dynamic.py" "$out" --output-dir "$OUTDIR"
  echo "[$name] report: $OUTDIR/${name}_report.md"
}

echo "stage=$STAGE limit=$LIMIT repeats=$REPEATS frames=$FRAMES tokens=$TOKENS"
echo "gpu=$GPU outdir=$OUTDIR"

if [[ "$STAGE" == "stage0" ]]; then
  # Recorded 2026-09-09 configuration, with the fallback that made it exact.
  run_case R1_full_three_att \
    --refresh-mode full --dynamic-query-mode three --dynamic-bootstrap attention \
    --selection-update-interval 1 --min-selection-change-ratio 0.05 \
    --verify-fallback sequential_on_low_margin

  # Fallback A/B: does exactness survive without the sequential guard?
  run_case R6_full_three_att_nofallback \
    --refresh-mode full --dynamic-query-mode three --dynamic-bootstrap attention \
    --selection-update-interval 1 --min-selection-change-ratio 0.05 \
    --verify-fallback none
fi

if [[ "$STAGE" == "stage1" ]]; then
  # R1: static-faithful control (equal S_0, full rebuild, every update).
  run_case R1_full_three_att \
    --refresh-mode full --dynamic-query-mode three --dynamic-bootstrap attention \
    --selection-update-interval 1 --min-selection-change-ratio 0.05 \
    --verify-fallback sequential_on_low_margin

  # R2: H1 -- incremental refresh slot-order regression.
  run_case R2_incr_three_att \
    --refresh-mode incremental --dynamic-query-mode three --dynamic-bootstrap attention \
    --selection-update-interval 1 --min-selection-change-ratio 0.05 \
    --verify-fallback sequential_on_low_margin

  # R3: H2 -- attention-free bootstrap (different S_0 by construction).
  run_case R3_full_three_attfree \
    --refresh-mode full --dynamic-query-mode three --dynamic-bootstrap attention_free \
    --selection-update-interval 1 --min-selection-change-ratio 0.05 \
    --verify-fallback sequential_on_low_margin

  # R4: H3 -- two-query collector.
  run_case R4_full_two_att \
    --refresh-mode full --dynamic-query-mode two --dynamic-bootstrap attention \
    --selection-update-interval 1 --min-selection-change-ratio 0.05 \
    --verify-fallback sequential_on_low_margin

  # R5: H6 -- no hysteresis, apply every candidate update.
  run_case R5_full_three_att_nohyst \
    --refresh-mode full --dynamic-query-mode three --dynamic-bootstrap attention \
    --selection-update-interval 1 --min-selection-change-ratio 0.0 \
    --verify-fallback sequential_on_low_margin

  # R6: F -- drop the sequential fallback to remove the §12 confound.
  run_case R6_full_three_att_nofallback \
    --refresh-mode full --dynamic-query-mode three --dynamic-bootstrap attention \
    --selection-update-interval 1 --min-selection-change-ratio 0.05 \
    --verify-fallback none
fi

echo
echo "all runs complete. reports in $OUTDIR"
