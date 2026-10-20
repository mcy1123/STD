#!/usr/bin/env bash
# Line A: exclusive-card re-measurement on GPU0 (verified: no compute apps on it).
#
# Answers three questions the shared-card runs could not:
#   1. Is 0.9227x stable? (paired A/B, 3 pairs)
#   2. Is the b1-b8 batch curve monotonic and explainable?
#   3. Do b16/b32 still collapse on an exclusive card? (shared-card throughput was 2-4 tok/s)
#
# Pinned with STD_FORCE_GPU so no neighbour can move in mid-comparison.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/env.sh"
export STD_FORCE_GPU=0
export STD_GPU=0

echo "=== LINE A start $(date +%H:%M:%S) on exclusive GPU0 ==="

echo "--- A1: paired A/B x3 (validates 0.9227x) ---"
FRAMES=32 MAXTOK=128 WINDOW=64 bash "$HERE/07_paired_ab.sh" 3
"$PY" "$HERE/07_paired_analyze.py" --file "$VLLM_STD_RES/spec_bench.jsonl" \
    --out "$VLLM_STD_RES/A1_paired_exclusive.json" || true

echo "--- A2: batch sweep b1..b32 on exclusive card ---"
FRAMES=8 MAXTOK=64 RUNS=3 K=4 WINDOW=256 bash "$HERE/06_batch_sweep.sh" || true

echo "=== LINE A done $(date +%H:%M:%S) ==="
