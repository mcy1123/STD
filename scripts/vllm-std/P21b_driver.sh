#!/usr/bin/env bash
# Long-wait driver for 2.1b (vLLM + HF coexistence) + 2.1c (alignment).
#
# 2.1b needs ~33.7 GiB of weights alone (vLLM 15.67 + HF 15.5) plus both sides'
# KV/activations plus a 5 GiB safety margin, so it demands a near-empty 48 GiB card.
# Configs are tried largest-first; a memory failure means "keep waiting", a
# non-memory failure means "stop and look" (a real failure is information).
#
# Only bytes appended by the CURRENT attempt are inspected: this log is appended
# across sessions, and grepping the whole file matched a stale error once already.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/env.sh"
LOG="$VLLM_STD_RES/logs/P21b_driver.log"
DEADLINE=$(( $(date +%s) + ${MAX_WAIT_S:-14400} ))   # 4h
NEED="${NEED:-40000}"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== 2.1b driver start (deadline $(date -d @$DEADLINE +%H:%M:%S), need ${NEED}MiB) ==="

attempt=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  read -r IDX FREE NAME < <(nvidia-smi --query-gpu=index,name,memory.free \
      --format=csv,noheader,nounits | awk -F', *' '{print $1, $3, $2}' | sort -k2,2nr | head -1)
  if [ "${FREE:-0}" -ge "$NEED" ]; then
    attempt=$((attempt+1))
    # largest config first; fall back to smaller frames if the window is just OK
    if [ "$FREE" -ge 44000 ]; then FR=8;  ML=8192; GMU=0.400
    else                          FR=4;  ML=4096; GMU=0.380; fi
    say "=== WINDOW: GPU$IDX ($NAME) free=${FREE}MiB -> attempt #$attempt (frames=$FR, util=$GMU) ==="
    BEFORE=$(wc -c < "$LOG")
    CUDA_VISIBLE_DEVICES="$IDX" "$PY" "$VLLM_STD_SCRIPTS/P21b_coexist_and_align.py" \
        --num-frames "$FR" --max-model-len "$ML" --gpu-mem-util "$GMU" \
        >>"$LOG" 2>&1
    rc=$?
    V=$(grep -oE '"verdict_2_1b": *"[A-Z_0-9]+"' "$VLLM_STD_RES/P21b_coexist.json" 2>/dev/null | head -1)
    say "attempt #$attempt rc=$rc $V"
    if [ "$rc" -eq 0 ] && echo "$V" | grep -q PASS; then
      say "=== 2.1b PASSED on GPU$IDX (frames=$FR) ==="; exit 0
    fi
    if tail -c +"$((BEFORE + 1))" "$LOG" | grep -qE \
        "OutOfMemoryError|Free memory on device|No available memory for the cache blocks|larger than the available KV cache memory"; then
      say "memory failure; continuing to wait"
    else
      say "=== NON-MEMORY failure; stopping for inspection ==="; exit 1
    fi
  else
    say "best GPU$IDX ($NAME) free=${FREE}MiB < ${NEED}; waiting"
  fi
  sleep 30
done
say "=== deadline reached without a window ==="
exit 4
