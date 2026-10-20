#!/usr/bin/env bash
# Launch a vLLM benchmark on the best available A6000, surviving a shared box.
#
# Two failure modes this exists for, both of which otherwise look like algorithm
# failures:
#
#  1. CONTENTION. Free memory can drop by tens of GiB between choosing a GPU and
#     the engine finishing startup. vLLM then dies with
#       "Free memory on device cuda:0 (X/GiB) ... is less than desired GPU memory
#        utilization"
#     We detect that and retry on the next-best GPU.
#
#  2. A FIXED --gpu-memory-utilization is wrong on a shared box. The KV budget is
#     `util*48GiB - 15.7GiB(model) - ~0.5GiB(cudagraph) - activations`, and
#     max_model_len=65536 alone needs 3.5GiB of KV. A value that works on an empty
#     card starves the cache on a half-full one, so the engine refuses to start.
#     We therefore derive the utilization from the chosen GPU's *actual* free
#     memory and pass it through.
#
# Usage: run_bench.sh <log_path> <bench_script.py> [bench args...]
#   Do NOT pass --gpu-mem-util; it is computed here.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

LOG="${1:?usage: run_bench.sh <log_path> <bench_script.py> [args...]}"; shift
SCRIPT="${1:?usage: run_bench.sh <log_path> <bench_script.py> [args...]}"; shift
MAX_ATTEMPTS="${MAX_ATTEMPTS:-4}"
A6000_TOTAL_MIB=49140
MIN_FREE_MIB="${MIN_FREE_MIB:-24000}"

mkdir -p "$(dirname "$LOG")"

# If this driver is killed, take the python child (and its EngineCore) with it.
# Killing only the shell leaves an orphaned EngineCore holding tens of GiB -- that
# is exactly how pid 1442533 leaked.
CHILD_PID=""
cleanup() {
  if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
    echo "=== driver exiting; terminating child $CHILD_PID ===" >>"$LOG"
    kill -TERM "$CHILD_PID" 2>/dev/null || true
    sleep 3
    kill -KILL "$CHILD_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# STD_FORCE_GPU pins the run to one card (used when we have verified it is
# exclusively ours and want timing free of neighbour noise).
if [ -n "${STD_FORCE_GPU:-}" ]; then
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$STD_FORCE_GPU")
  CANDS=("$STD_FORCE_GPU $FREE $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$STD_FORCE_GPU")")
  echo "=== forced GPU=$STD_FORCE_GPU free=${FREE}MiB ===" | tee -a "$LOG"
fi

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  if [ -z "${STD_FORCE_GPU:-}" ]; then
  mapfile -t CANDS < <(REQ_FREE_MIB="$MIN_FREE_MIB" MAX_UTIL="${MAX_UTIL:-70}" \
                        "$HERE/pick_gpu.sh" 2>/dev/null || true)
  fi
  if [ "${#CANDS[@]}" -eq 0 ]; then
    echo "=== [attempt $attempt] no A6000 with >=${MIN_FREE_MIB}MiB free ===" | tee -a "$LOG"
    sleep 30; continue
  fi
  for cand in "${CANDS[@]}"; do
    GPU=$(echo "$cand" | awk '{print $1}')
    FREE=$(echo "$cand" | awk '{print $2}')
    UTIL=$(echo "$cand" | awk '{print $3}')
    # Claim ~93% of currently-free memory, capped so we never monopolise the card.
    GMU=$(awk -v f="$FREE" -v t="$A6000_TOTAL_MIB" 'BEGIN{u=f/t*0.93; if(u>0.90)u=0.90; printf "%.3f", u}')
    echo "=== [attempt $attempt] GPU=$GPU free=${FREE}MiB util=${UTIL}% gpu_mem_util=$GMU ===" | tee -a "$LOG"
    # Only bytes appended by THIS attempt may be inspected: the log is appended
    # across attempts, so grepping the whole file would match stale text forever.
    BEFORE=$(wc -c < "$LOG")
    CUDA_VISIBLE_DEVICES="$GPU" STD_GPU_MEM_UTIL="$GMU" \
      "$PY" "$VLLM_STD_SCRIPTS/$SCRIPT" --gpu-mem-util "$GMU" "$@" >>"$LOG" 2>&1 &
    CHILD_PID=$!
    wait "$CHILD_PID"
    rc=$?
    CHILD_PID=""
    if [ $rc -eq 0 ]; then
      echo "=== OK on GPU=$GPU (rc=0) ===" | tee -a "$LOG"
      exit 0
    fi
    if tail -c +"$((BEFORE + 1))" "$LOG" | grep -q \
        "is less than desired GPU memory utilization\|CUDA out of memory\|No available memory for the cache blocks"; then
      echo "=== GPU=$GPU lost to contention (rc=$rc); trying next ===" | tee -a "$LOG"
      continue
    fi
    echo "=== FAILED on GPU=$GPU rc=$rc (not contention); not retrying ===" | tee -a "$LOG"
    exit $rc
  done
  sleep 20
done

echo "=== FATAL: exhausted $MAX_ATTEMPTS attempts ===" | tee -a "$LOG" >&2
exit 4
