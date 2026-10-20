#!/usr/bin/env bash
# Phase 7: interleaved A/B/A/B paired timing, to defeat shared-machine drift.
#
# WHY: Phase 5 produced non-monotonic window timings (64 -> 4.15s, 256 -> 5.40s,
# 1024 -> 4.51s) purely because one arm happened to land on an idle card. On this
# host a single point measurement is not comparable across time; alternating the
# two arms makes drift a common-mode error that cancels in the paired difference.
#
# Each arm still needs its own engine (speculative_config is fixed at construction),
# so "paired" is enforced at the level of *time*: A,B,A,B,...
#
# Usage: 07_paired_ab.sh <pairs> [extra bench args]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/env.sh"

PAIRS="${1:-3}"; shift || true
COMMON="--num-frames ${FRAMES:-32} --max-tokens ${MAXTOK:-128} --runs 1 --warmup 1"
export STD_DRAFT_WINDOW="${WINDOW:-64}" STD_DRAFT_NGRAM=3

for i in $(seq 1 "$PAIRS"); do
  echo "### pair $i arm A (eager)"
  bash "$HERE/run_bench.sh" "$VLLM_STD_RES/logs/07_paired_A$i.log" 03_bench_spec.py \
      $COMMON --tag "p7_A$i" "$@"
  echo "### pair $i arm B (sparse k=4 W=$STD_DRAFT_WINDOW)"
  export STD_PROPOSER_STATS="$VLLM_STD_RES/proposer_stats/p7_B$i.json"
  bash "$HERE/run_bench.sh" "$VLLM_STD_RES/logs/07_paired_B$i.log" 03_bench_spec.py \
      $COMMON --tag "p7_B$i" --num-spec-tokens 4 \
      --proposer std_vllm.proposer.SparseContextDraftProposer "$@"
done
echo "### paired run complete"
