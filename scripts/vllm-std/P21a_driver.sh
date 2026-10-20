#!/usr/bin/env bash
# Long-wait driver for 2.1a on a volatile shared box.
#
# Requires a LARGE margin (not just "enough"): measuring on this host showed free
# memory falling 23.8 -> 6.28 GiB in 6 seconds, so a window that just barely fits
# at probe time is gone by the time the 15.67 GiB of weights are on the card.
# We therefore demand a big margin and accept a long wait.
#
# Also tries shrinking max_model_len on each successive attempt, so a smaller
# window can still be used rather than failing forever on one fixed size.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/env.sh"

LOG="$VLLM_STD_RES/logs/P21a_driver.log"
DEADLINE=$(( $(date +%s) + ${MAX_WAIT_S:-10800} ))   # default 3h
NEED_1="${NEED_1:-30000}"     # preferred window
NEED_2="${NEED_2:-22000}"     # fallback after 45 min
attempt=0

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== 2.1a driver start (deadline $(date -d @$DEADLINE +%H:%M:%S)) ==="

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  elapsed=$(( $(date +%s) - (DEADLINE - ${MAX_WAIT_S:-10800}) ))
  if [ "$elapsed" -gt 2700 ]; then NEED="$NEED_2"; else NEED="$NEED_1"; fi

  read -r IDX FREE NAME < <(nvidia-smi --query-gpu=index,name,memory.free \
      --format=csv,noheader,nounits | awk -F', *' '{print $1, $3, $2}' | sort -k2,2nr | head -1)

  if [ "${FREE:-0}" -ge "$NEED" ]; then
    attempt=$((attempt+1))
    # Prefer 32 frames; fall back to 8 frames if the window is on the smaller side.
    # Budget must cover weights (15.67 GiB) + encoder cache + KV + activations.
    # 0.360 gave a NEGATIVE KV budget (-3.3 GiB) at 32 frames -- measured, not guessed.
    # Budget: weights 15.67 + non-KV peak ~5.4 (encoder cache @32 frames)
    # + KV 1.75  =>  ~22.8 GiB minimum.  0.430 (21.1 GiB) measured out at 0.03 GiB KV.
    if [ "$FREE" -ge 32000 ]; then FR=32; ML=32768; GMU=0.560; else FR=8; ML=8192; GMU=0.450; fi
    say "=== WINDOW: GPU$IDX ($NAME) free=${FREE}MiB -> attempt #$attempt (frames=$FR) ==="
    BEFORE=$(wc -c < "$LOG")
    CUDA_VISIBLE_DEVICES="$IDX" \
      "$PY" "$VLLM_STD_SCRIPTS/P21a_capture_video_inputs.py" \
        --num-frames "$FR" --max-model-len "$ML" --gpu-mem-util "$GMU" \
        --out "$VLLM_STD_RES/P21a_video_inputs.json" >>"$LOG" 2>&1
    rc=$?
    verdict=$(grep -oE '"verdict": *"[A-Z_]+"' "$VLLM_STD_RES/P21a_video_inputs.json" 2>/dev/null | head -1)
    say "attempt #$attempt rc=$rc $verdict"
    if [ "$rc" -eq 0 ] && echo "$verdict" | grep -q PASS; then
      say "=== 2.1a PASSED on GPU$IDX (frames=$FR) ==="
      exit 0
    fi
    # If it was not a memory failure, stop -- a real technical failure is information.
    if tail -c +"$((BEFORE + 1))" "$LOG" | grep -qE \
        "OutOfMemoryError|Free memory on device|No available memory for the cache blocks|larger than the available KV cache memory"; then
      say "memory failure; continuing to wait"
    else
      say "=== NON-MEMORY failure; stopping driver for inspection ==="
      exit 1
    fi
  else
    say "best GPU$IDX ($NAME) free=${FREE}MiB < ${NEED}; waiting"
  fi
  sleep 25
done
say "=== deadline reached without a window ==="
exit 4
