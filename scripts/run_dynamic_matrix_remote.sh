#!/usr/bin/env bash
# E1 / E6 single-variable matrix for dynamic-routing, executed ON gpu23.
#
# Launched by scripts/run_dynamic_matrix.sh (which syncs the code first); it is
# written to survive a dropped SSH session, so it can also be started by hand:
#
#   ssh a100-gpu 'cd $REPO && TAG=e1e6_x nohup setsid bash scripts/run_dynamic_matrix_remote.sh \
#                 > /tmp/matrix.log 2>&1 < /dev/null & echo $!'
#
# Every configuration differs from the R1 baseline in exactly one factor, so the
# paired comparison against R1 stays interpretable (see PROGRESS.md §17.6/§17.7):
#
#   A (baseline, already measured as R1) collector=v2  interval=1  min_change=0.00
#   B (E1, throttling)                   collector=v2  interval=4  min_change=0.05
#   C (E6, all-query scoring)            collector=v3  interval=1  min_change=0.00
#   D (E1+E6 combined)                   collector=v3  interval=4  min_change=0.05
#
# The benchmark opens its output with mode "x", so an existing file is never
# overwritten and re-running the driver resumes at the first missing config.
set -uo pipefail

REPO="${REPO:-/public/home/xlwang/mcy/Project/STD-latest}"
ASSETS="${ASSETS:-/public/home/xlwang/mcy/STD_assets}"
PY="${PY:-/public/home/xlwang/mcy/conda_envs/specvlm/bin/python}"
TAG="${TAG:-e1e6_$(date +%Y%m%d)}"
GPU="${GPU:-1}"
LIMIT="${LIMIT:-10}"
MIN_FREE_GIB="${MIN_FREE_GIB:-26.0}"

MODEL="$ASSETS/models/Qwen2.5-VL-7B-Instruct"
DATA="$ASSETS/datasets/Video-MME"
VIDEOS="$DATA/videos"
OUT_DIR="$ASSETS/results/$TAG"

# name|collector|selection_update_interval|min_selection_change_ratio
CONFIGS=(
  "B_v2_i4|v2|4|0.05"
  "C_v3_i1|v3|1|0.00"
  "D_v3_i4|v3|4|0.05"
)

mkdir -p "$OUT_DIR"
DRIVER_LOG="$OUT_DIR/driver.log"
DONE_MARKER="$OUT_DIR/MATRIX_DONE"
rm -f "$DONE_MARKER"

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$DRIVER_LOG"; }

# GPU1 is shared with other users, so free memory swings by tens of GiB. Poll
# until the guard in benchmark_a100_dynamic.py can pass instead of burning the
# whole matrix on "not enough free memory" failures (precedent: wait_r1.log).
WAIT_GPU_MAX_S="${WAIT_GPU_MAX_S:-21600}"
need_mib() { awk -v g="$MIN_FREE_GIB" 'BEGIN { printf "%d", g * 1024 }'; }

wait_for_free_gpu() {
  local want waited=0 free
  want="$(need_mib)"
  while :; do
    free="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
    [[ "$free" =~ ^[0-9]+$ ]] || free=0
    if (( free >= want )); then
      say "GPU$GPU ready: ${free} MiB free (need ${want} MiB)"
      return 0
    fi
    if (( waited >= WAIT_GPU_MAX_S )); then
      say "GPU$GPU still short after ${waited}s (${free} MiB free, need ${want} MiB); giving up"
      return 1
    fi
    say "waiting for GPU$GPU: ${free} MiB free, need ${want} MiB (waited ${waited}s)"
    sleep 120
    waited=$(( waited + 120 ))
  done
}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src"
cd "$REPO" || { say "FATAL repo not found: $REPO"; exit 2; }

say "TAG=$TAG GPU=$GPU LIMIT=$LIMIT repo=$REPO"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | tee -a "$DRIVER_LOG"

for spec in "${CONFIGS[@]}"; do
  IFS='|' read -r name coll iv mc <<<"$spec"
  jsonl="$OUT_DIR/${TAG}_${name}.jsonl"

  if [[ -e "$jsonl" ]]; then
    say "[$name] SKIP: $jsonl already exists"
    continue
  fi

  say "[$name] START collector=$coll interval=$iv min_change=$mc"
  if ! wait_for_free_gpu; then
    say "[$name] ABORT: no GPU with enough free memory; stopping matrix"
    break
  fi
  "$PY" scripts/benchmark_a100_dynamic.py \
    --gpu "$GPU" \
    --model-path "$MODEL" \
    --data-path  "$DATA" \
    --video-root "$VIDEOS" \
    --output     "$jsonl" \
    --frame-num 128 --max-new-tokens 128 --limit "$LIMIT" --repeats 1 \
    --warmup-tokens 16 --gamma 9 --k-plus-text 1024 --max-pixels 200704 \
    --cache-len 20480 --sparse-attn-mode gqa_sdpa \
    --verify-fallback none --verify-margin-threshold 0.1 \
    --dynamic-collector "$coll" --refresh-mode incremental \
    --selection-update-interval "$iv" --min-selection-change-ratio "$mc" \
    --dynamic-query-mode three --dynamic-bootstrap attention \
    --bootstrap-coverage-ratio 0.25 --bootstrap-value-weight 0.25 \
    --bootstrap-window-tokens 0 --bootstrap-layer-stride 1 \
    --assert-equal-s0 --assert-consistency --profile-components \
    --max-rss-gib 48 --allow-shared-gpu --min-free-gib "$MIN_FREE_GIB" \
    > "$OUT_DIR/${name}.log" 2>&1
  rc=$?

  if [[ $rc -ne 0 ]]; then
    say "[$name] FAILED rc=$rc (see ${name}.log); continuing"
    continue
  fi

  "$PY" scripts/summarize_a100_dynamic.py "$jsonl" --output-dir "$OUT_DIR" \
    >> "$OUT_DIR/${name}.log" 2>&1
  say "[$name] DONE (summarize rc=$?)"
done

say "MATRIX_DONE"
printf 'MATRIX_DONE %s\n' "$(date -Is)" > "$DONE_MARKER"
