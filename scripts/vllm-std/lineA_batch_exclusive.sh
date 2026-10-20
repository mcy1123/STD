#!/usr/bin/env bash
# Line A (P0 round): batch b1..b32 sweep on an EXCLUSIVE A6000.
# Distinct tags (p6x_*) so these never mix with the shared-card p6_* evidence.
# Prior shared-card attempt died at b16 (engine throughput 2-4 tok/s); this retests it.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/env.sh"
export STD_FORCE_GPU="${STD_FORCE_GPU:-2}"
FRAMES="${FRAMES:-8}"; MAXTOK="${MAXTOK:-64}"; RUNS="${RUNS:-3}"

echo "=== LINE A batch sweep start $(date +%H:%M:%S) on exclusive GPU$STD_FORCE_GPU ==="
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader | tee -a "$VLLM_STD_RES/logs/lineA_batch.log"

for B in 1 2 4 8 16 32; do
  for MODE in eager spec; do
    TAG="p6x_${MODE}_b${B}"
    ARGS=(--num-videos "$B" --num-frames "$FRAMES" --max-tokens "$MAXTOK"
          --runs "$RUNS" --warmup 1 --tag "$TAG")
    if [ "$MODE" = spec ]; then
      export STD_DRAFT_WINDOW=256 STD_DRAFT_NGRAM=3
      ARGS+=(--num-spec-tokens 4 --proposer std_vllm.proposer.SparseContextDraftProposer)
    fi
    echo "### $TAG" | tee -a "$VLLM_STD_RES/logs/lineA_batch.log"
    timeout 2400 bash "$HERE/run_bench.sh" "$VLLM_STD_RES/logs/06x_$TAG.log" \
        03_bench_spec.py "${ARGS[@]}" >>"$VLLM_STD_RES/logs/lineA_batch.log" 2>&1
    echo "### $TAG rc=$?" | tee -a "$VLLM_STD_RES/logs/lineA_batch.log"
  done
done

"$PY" "$HERE/summarize_batch_sweep.py" --file "$VLLM_STD_RES/spec_bench.jsonl" \
    --prefix p6x_ --out "$VLLM_STD_RES/batch_sweep_exclusive.csv" || true
echo "=== LINE A batch sweep done $(date +%H:%M:%S) ==="
