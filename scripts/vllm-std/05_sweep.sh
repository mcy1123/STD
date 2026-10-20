#!/usr/bin/env bash
# Phase 5 sweep: draft length (k) x sparse context window (W), one variable at a time.
#
# Runs the SAME video / frames / greedy settings as the Phase 3 baseline so every
# row is comparable. Proposer-side counters land in results/vllm-std/proposer_stats/.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/env.sh"
mkdir -p "$VLLM_STD_RES/proposer_stats"

COMMON="--num-frames 32 --max-tokens 128 --runs 3 --warmup 1"
NGRAM="${NGRAM:-3}"

run_one() {  # tag window k
  local tag="$1" win="$2" k="$3"
  export STD_DRAFT_WINDOW="$win" STD_DRAFT_NGRAM="$NGRAM"
  export STD_PROPOSER_STATS="$VLLM_STD_RES/proposer_stats/$tag.json"
  echo "### $tag: window=$win k=$k ngram=$NGRAM"
  bash "$HERE/run_bench.sh" "$VLLM_STD_RES/logs/05_$tag.log" 03_bench_spec.py \
      $COMMON --tag "$tag" --num-spec-tokens "$k" \
      --proposer std_vllm.proposer.SparseContextDraftProposer
  echo "### $tag rc=$?"
}

# Draft-length curve at a fixed sparse window.
for k in 1 2 4 8; do run_one "p5_w256_k$k" 256 "$k"; done
# Sparsity curve at fixed k=4 (0 = full context, i.e. no sparsity).
for w in 0 64 1024; do run_one "p5_w${w}_k4" "$w" 4; done
