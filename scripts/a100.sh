#!/usr/bin/env bash
# Reusable ECNU Phase-8 A100 access helper.
#
#   bash scripts/a100.sh help
#
# Full manual: docs/a100-access.md
#
# Connection chain (the GPU node is NOT directly reachable from the internet):
#   local --ssh -p 2323--> login2 --ProxyJump--> gpu23
#
# Authentication: an SSH key is installed on the cluster, so no password is
# needed. `~/.ssh/config` should define the aliases `a100` (login2) and
# `a100-gpu` (gpu23 via ProxyJump); this script falls back to explicit
# host/port targets if those aliases are absent.
#
# This script NEVER stores or embeds a password.
set -euo pipefail

A100_HOST="${A100_HOST:-59.78.189.133}"
A100_PORT="${A100_PORT:-2323}"
A100_USER="${A100_USER:-xlwang}"
A100_GPU_HOST="${A100_GPU_HOST:-10.11.200.23}"
A100_LOGIN_ALIAS="${A100_LOGIN_ALIAS:-a100}"
A100_GPU_ALIAS="${A100_GPU_ALIAS:-a100-gpu}"

A100_REPO="${A100_REPO:-/public/home/xlwang/mcy/Project/STD-latest}"
A100_ASSETS="${A100_ASSETS:-/public/home/xlwang/mcy/STD_assets}"
A100_LOCAL_REPO="${A100_LOCAL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
A100_LOCAL_RESULTS="${A100_LOCAL_RESULTS:-$A100_LOCAL_REPO/results/a100_ablation}"

# Default paths synced from the local checkout to the remote working copy.
DEFAULT_SYNC_PATHS=(src scripts tests docs PROGRESS.md README.md)

log() { printf '[a100] %s\n' "$*" >&2; }
die() { printf '[a100] error: %s\n' "$*" >&2; exit 1; }

alias_hostname() { ssh -G "$1" 2>/dev/null | awk '/^hostname /{print $2; exit}'; }

# Prefer the ssh_config aliases; otherwise build explicit targets.
if [[ "$(alias_hostname "$A100_LOGIN_ALIAS")" == "$A100_HOST" ]]; then
  LOGIN_SPEC="$A100_LOGIN_ALIAS"
  RSH="ssh"
else
  LOGIN_SPEC="${A100_USER}@${A100_HOST}"
  RSH="ssh -p ${A100_PORT} -o ConnectTimeout=15"
fi

if [[ "$(alias_hostname "$A100_GPU_ALIAS")" == "$A100_GPU_HOST" ]]; then
  GPU_SPEC="$A100_GPU_ALIAS"
else
  GPU_SPEC="-o ProxyJump=${A100_USER}@${A100_HOST}:${A100_PORT} ${A100_USER}@${A100_GPU_HOST}"
fi

on_login() { ssh $LOGIN_SPEC "$@"; }
on_login_tty() { ssh -t $LOGIN_SPEC "$@"; }

on_gpu() {
  if [[ $# -eq 0 ]]; then ssh -t $GPU_SPEC; else ssh $GPU_SPEC "$@"; fi
}

cmd_status() {
  on_gpu "nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader; echo '--- compute apps ---'; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"
}

cmd_login() { if [[ $# -eq 0 ]]; then on_login_tty; else on_login "$@"; fi; }
cmd_gpu()   { if [[ $# -eq 0 ]]; then on_gpu; else on_gpu "$@"; fi; }

cmd_sync() {
  local paths=("$@")
  [[ ${#paths[@]} -eq 0 ]] && paths=("${DEFAULT_SYNC_PATHS[@]}")
  ( cd "$A100_LOCAL_REPO"
    for p in "${paths[@]}"; do [[ -e "$p" ]] || die "sync source not found locally: $p"; done
    log "syncing to ${LOGIN_SPEC}:${A100_REPO}/ : ${paths[*]}"
    rsync -az --exclude='__pycache__' --exclude='*.pyc' -e "$RSH" "${paths[@]}" "${LOGIN_SPEC}:${A100_REPO}/"
  )
}

cmd_ablation() {
  local stage="${1:-stage0}"
  on_gpu "cd ${A100_REPO} && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && REPO='${A100_REPO}' bash scripts/run_a100_ablation.sh ${stage}"
}

cmd_e2e() {
  local stage="${1:-stage0}" latest
  cmd_sync
  log "GPU status before launch:"
  cmd_status || true
  cmd_ablation "$stage"
  latest="$(on_login "ls -1dt ${A100_ASSETS}/results/ablation_* 2>/dev/null | head -1" | tr -d '\r' | tail -1)"
  [[ -n "$latest" ]] || die "no ablation output directory found on the remote"
  mkdir -p "$A100_LOCAL_RESULTS"
  log "pulling reports from $latest"
  rsync -az -e "$RSH" \
    --include='*/' --include='*_report.md' --include='*_summary.json' --exclude='*' \
    "${LOGIN_SPEC}:${latest}/" "$A100_LOCAL_RESULTS/"
  log "done. reports in $A100_LOCAL_RESULTS"
}

cmd_push() {
  [[ $# -eq 2 ]] || die "usage: $0 push <local-path> <remote-absolute-path>"
  [[ -e "$1" ]] || die "local path not found: $1"
  log "uploading $1 -> ${LOGIN_SPEC}:$2"
  rsync -az -e "$RSH" "$1" "${LOGIN_SPEC}:$2"
}

cmd_pull() {
  [[ $# -eq 2 ]] || die "usage: $0 pull <remote-absolute-path> <local-path>"
  log "downloading ${LOGIN_SPEC}:$1 -> $2"
  rsync -az -e "$RSH" "${LOGIN_SPEC}:$1" "$2"
}

cmd_close() {
  ssh -O exit $LOGIN_SPEC >/dev/null 2>&1 || true
  ssh -O exit $GPU_SPEC  >/dev/null 2>&1 || true
  log "master connections closed (if any)"
}

cmd_help() {
  cat <<EOF
ECNU Phase-8 A100 helper  (see docs/a100-access.md)

usage: bash scripts/a100.sh <command> [args...]

  status                 GPU status + compute apps on gpu23
  login [cmd...]         run cmd on login2, or open an interactive shell
  gpu   [cmd...]         run cmd on gpu23,   or open an interactive shell
  sync  [paths...]       rsync local code to the remote working copy
                         (default: ${DEFAULT_SYNC_PATHS[*]})
  ablation [stage]       run scripts/run_a100_ablation.sh on gpu23 (default stage0)
  e2e   [stage]          sync + GPU status + ablation + pull *_report.md/_summary.json
  push <local> <remote>  upload a single path
  pull <remote> <local>  download a single path
  close                  close multiplexed master connections

targets (override via environment):
  login2  ${LOGIN_SPEC}
  gpu23   ${GPU_SPEC}
  repo    ${A100_REPO}
  assets  ${A100_ASSETS}
  local   ${A100_LOCAL_REPO}  -> results: ${A100_LOCAL_RESULTS}

rules: GPU 0 runs a vLLM workload and GPU 1 may be held by another user;
       never kill processes you did not start. The benchmark refuses to run
       on a busy GPU by design.
EOF
}

main() {
  local cmd="${1:-help}"
  shift || true
  case "$cmd" in
    status)    cmd_status "$@" ;;
    login)     cmd_login "$@" ;;
    gpu)       cmd_gpu "$@" ;;
    sync)      cmd_sync "$@" ;;
    ablation)  cmd_ablation "$@" ;;
    e2e)       cmd_e2e "$@" ;;
    push)      cmd_push "$@" ;;
    pull)      cmd_pull "$@" ;;
    close)     cmd_close "$@" ;;
    help|-h|--help) cmd_help ;;
    *) die "unknown command '$cmd' (try: $0 help)" ;;
  esac
}

main "$@"
