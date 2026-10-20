#!/usr/bin/env bash
# Phase 6: batch sweep (1,2,4,8,16,32) for eager vs sparse-context speculation.
#
# Frame count is deliberately low (default 8). At 32 frames each request carries
# ~19.2k prompt tokens, so batch 32 would need ~613k tokens of KV cache (~34 GiB at
# 56 KB/token) and cannot fit beside the 15.7 GiB model on a 48 GiB A6000. 8 frames
# (~4.8k tokens) keeps batch 32 inside the card while preserving the comparison.
#
# Every row is the SAME videos, frames, greedy settings and GPU-selection policy;
# only batch size and proposer vary.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/env.sh"
mkdir -p "$VLLM_STD_RES"

FRAMES="${FRAMES:-8}"
MAXTOK="${MAXTOK:-64}"
RUNS="${RUNS:-3}"
K="${K:-4}"
WINDOW="${WINDOW:-256}"
CSV="$VLLM_STD_RES/batch_sweep.csv"

for B in 1 2 4 8 16 32; do
  for MODE in eager spec; do
    TAG="p6_${MODE}_b${B}"
    ARGS=(--num-videos "$B" --num-frames "$FRAMES" --max-tokens "$MAXTOK"
          --runs "$RUNS" --warmup 1 --tag "$TAG")
    if [ "$MODE" = spec ]; then
      export STD_DRAFT_WINDOW="$WINDOW" STD_DRAFT_NGRAM=3
      export STD_PROPOSER_STATS="$VLLM_STD_RES/proposer_stats/$TAG.json"
      ARGS+=(--num-spec-tokens "$K" --proposer std_vllm.proposer.SparseContextDraftProposer)
    fi
    echo "### $TAG"
    bash "$HERE/run_bench.sh" "$VLLM_STD_RES/logs/06_$TAG.log" 03_bench_spec.py "${ARGS[@]}"
    echo "### $TAG rc=$?"
  done
done

# Materialise the required CSV from the jsonl evidence.
"$PY" "$HERE/summarize_batch_sweep.py" --file "$VLLM_STD_RES/spec_bench.jsonl" --out "$CSV"
echo "### wrote $CSV"
