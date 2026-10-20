#!/usr/bin/env bash
# Wait for a GPU window and launch IMMEDIATELY, in one shell invocation.
# Rationale: this box's free memory swings by >15 GiB in under a minute, so
# "poll, then run" reliably loses the window between the two steps. Here the
# check and the launch are adjacent with no user-visible gap.
#
# Usage: wait_and_run.sh <min_free_MiB> <max_wait_s> <script.py> [args...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/env.sh"
NEED="${1:?min free MiB}"; shift
MAXW="${1:-900}"; shift
SCRIPT="${1:?script}"; shift
A6000_TOTAL=49140; T4090_TOTAL=24564

waited=0
while [ "$waited" -lt "$MAXW" ]; do
  read -r IDX FREE NAME < <(nvidia-smi --query-gpu=index,name,memory.free \
      --format=csv,noheader,nounits | awk -F', *' '{print $1, $3, $2}' | sort -k2,2nr | head -1)
  if [ "${FREE:-0}" -ge "$NEED" ]; then
    TOT=$A6000_TOTAL; case "$NAME" in *4090*) TOT=$T4090_TOTAL;; esac
    # Use a FIXED conservative budget, not a fraction of the transient peak.
    # Scaling to the peak asked for 33.5 GiB when a neighbour then dropped the
    # card to 27 GiB, so the engine refused to start. A 7B needs ~17.5 GiB
    # (15.67 weights + ~1 encoder + ~0.8 KV), so request that and no more.
    GMU="${GMU_FIXED:-0.360}"
    echo "=== window: GPU=$IDX ($NAME) free=${FREE}MiB -> gpu_mem_util=$GMU ==="
    CUDA_VISIBLE_DEVICES="$IDX" exec "$PY" "$VLLM_STD_SCRIPTS/$SCRIPT" \
        --gpu-mem-util "$GMU" "$@"
  fi
  echo "[$(date +%H:%M:%S)] best GPU$IDX ($NAME) free=${FREE}MiB < ${NEED}; waiting"
  sleep 20; waited=$((waited+20))
done
echo "=== no window after ${MAXW}s ===" >&2
exit 4
