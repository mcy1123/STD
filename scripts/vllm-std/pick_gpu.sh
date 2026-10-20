#!/usr/bin/env bash
# List usable A6000 GPUs, best first, given how much memory a run actually needs.
#
# This box is shared and its occupancy changes minute to minute; a fixed GPU
# choice silently turns "someone else started a job" into "my experiment OOMs".
# So callers state the free memory they require and get back a ranked candidate
# list to try in order.
#
# Only A6000s are considered: the box also has 2x RTX 4090 (24 GiB), and mixing
# architectures makes timings non-comparable -- the exact cross-hardware confound
# documented in docs/experiment-provenance.md.
#
# We never touch other users' processes; we only choose where to run.
#
# Env:
#   REQ_FREE_MIB  required free memory per GPU (default 12000)
#   MAX_UTIL      ignore GPUs busier than this percent (default 60)
#
# stdout: "INDEX FREE_MIB UTIL" per candidate line, best first.
# exit 0 with candidates, exit 1 with none.
set -uo pipefail
REQ_FREE_MIB="${REQ_FREE_MIB:-12000}"
MAX_UTIL="${MAX_UTIL:-60}"

cands=$(nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu \
    --format=csv,noheader,nounits \
| awk -F', *' -v minf="$REQ_FREE_MIB" -v maxu="$MAX_UTIL" '
    $2 ~ /A6000/ && $3+0 >= minf && $4+0 <= maxu { printf "%d %d %d\n", $1, $3, $4 }
' | sort -k2,2nr -k3,3n)

if [ -z "$cands" ]; then
    echo "no A6000 with >=${REQ_FREE_MIB} MiB free and <=${MAX_UTIL}% util" >&2
    exit 1
fi
echo "$cands"
