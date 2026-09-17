#!/usr/bin/env bash
# Reusable ECNU Phase-8 A100 access helper.
#
#   bash scripts/a100.sh help
#
# Full manual: docs/a100-access.md
#
# Connection chain (the GPU node is NOT directly reachable):
#   local --ssh -p 2323--> login2 --ssh--> gpu23
#
# This script NEVER stores or embeds a password. Use an SSH key
# (`a100.sh setup-key`, recommended) or let it prompt once and reuse a
# multiplexed ControlMaster connection for A100_PERSIST seconds.
set -euo pipefail

A100_HOST="${A100_HOST:-59.78.189.133}"
A100_PORT="${A100_PORT:-2323}"
A100_USER="${A100_USER:-xlwang}"
A100_GPU_HOST="${A100_GPU_HOST:-10.11.200.23}"
A100_PERSIST="${A100_PERSIST:-600}"
A100_SOCK="${A100_SOCK:-${TMPDIR:-/tmp}/a100-ssh-$(id -u)/control}"
A100_REPO="${A100_REPO:-/public/home/xlwang/mcy/Project/STD-latest}"
A100_PY="${A100_PY:-/public/home/xlwang/mcy/conda_envs/specvlm/bin/python}"

LOGIN="${A100_USER}@${A100_HOST}"
SSH_OPTS=(-p "$A100_PORT" -o ServerAliveInterval=30 -o ServerAliveCountMax=5 -o ConnectTimeout=15)

log() { printf '[a100] %s\n' "$*" >&2; }
die() { printf '[a100] error: %s\n' "$*" >&2; exit 1; }

# Optional sshpass support; only active when the user points at a secret file
# that lives OUTSIDE this repository.
AUTH=()
if [[ -n "${A100_PASSWORD_FILE:-}" ]]; then
  command -v sshpass >/dev/null 2>&1 || die "A100_PASSWORD_FILE is set but sshpass is not installed"
  [[ -r "$A100_PASSWORD_FILE" ]] || die "A100_PASSWORD_FILE is not readable: $A100_PASSWORD_FILE"
  AUTH=(sshpass -f "$A100_PASSWORD_FILE")
fi

master_alive() {
  "${AUTH[@]}" ssh -O check -S "$A100_SOCK" "${SSH_OPTS[@]}" "$LOGIN" >/dev/null 2>&1
}

ensure_master() {
  master_alive && return 0
  mkdir -p "$(dirname "$A100_SOCK")"
  log "opening multiplexed master to $LOGIN (ControlPersist=${A100_PERSIST}s)"
  if [[ -z "${A100_PASSWORD_FILE:-}" ]]; then
    log "a password prompt is expected unless an SSH key is installed (run: $0 setup-key)"
  fi
  "${AUTH[@]}" ssh -M -S "$A100_SOCK" -o ControlPersist="$A100_PERSIST" "${SSH_OPTS[@]}" "$LOGIN" true
}

on_login() {
  ensure_master
  "${AUTH[@]}" ssh -S "$A100_SOCK" "${SSH_OPTS[@]}" "$LOGIN" "$@"
}

on_login_tty() {
  ensure_master
  "${AUTH[@]}" ssh -t -S "$A100_SOCK" "${SSH_OPTS[@]}" "$LOGIN" "$@"
}

on_gpu() {
  ensure_master
  if [[ $# -eq 0 ]]; then
    on_login_tty "ssh -t ${A100_GPU_HOST}"
    return 0
  fi
  local remote
  remote="$(printf '%q ' "$@")"
  on_login "ssh -o BatchMode=yes -o ConnectTimeout=15 ${A100_GPU_HOST} ${remote}"
}

cmd_status() {
  on_gpu "nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader"
}

cmd_login() {
  if [[ $# -eq 0 ]]; then on_login_tty; else on_login "$@"; fi
}

cmd_gpu() {
  if [[ $# -eq 0 ]]; then on_gpu; else on_gpu "$@"; fi
}

cmd_push() {
  [[ $# -eq 2 ]] || die "usage: $0 push <local-path> <remote-absolute-path>"
  [[ -e "$1" ]] || die "local path not found: $1"
  ensure_master
  log "uploading $1 -> $LOGIN:$2"
  "${AUTH[@]}" scp -P "$A100_PORT" -o ControlPath="$A100_SOCK" -o ServerAliveInterval=30 "$1" "$LOGIN:$2"
}

cmd_pull() {
  [[ $# -eq 2 ]] || die "usage: $0 pull <remote-absolute-path> <local-path>"
  ensure_master
  log "downloading $LOGIN:$1 -> $2"
  "${AUTH[@]}" scp -P "$A100_PORT" -o ControlPath="$A100_SOCK" -o ServerAliveInterval=30 "$LOGIN:$1" "$2"
}

cmd_setup_key() {
  local key="${1:-$HOME/.ssh/id_ed25519.pub}"
  [[ -f "$key" ]] || die "public key not found: $key (create one with: ssh-keygen -t ed25519)"
  log "installing $key on $LOGIN (one password prompt; afterwards no password is needed)"
  ssh-copy-id -i "$key" -p "$A100_PORT" "$LOGIN"
  log "done. verify with: $0 login hostname"
}

cmd_close() {
  if master_alive; then
    "${AUTH[@]}" ssh -O exit -S "$A100_SOCK" "${SSH_OPTS[@]}" "$LOGIN" >/dev/null 2>&1 || true
    log "master connection closed"
  else
    log "no master connection running"
  fi
}

cmd_ablation() {
  local stage="${1:-stage0}"
  on_gpu "cd ${A100_REPO} && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && bash scripts/run_a100_ablation.sh ${stage}"
}

cmd_help() {
  cat <<EOF
ECNU Phase-8 A100 helper

usage: bash scripts/a100.sh <command> [args...]

  status                 show GPU status on gpu23 (index, memory, util)
  login [cmd...]         run cmd on login2, or open an interactive shell
  gpu   [cmd...]         run cmd on gpu23,   or open an interactive shell
  ablation [stage]       run scripts/run_a100_ablation.sh on gpu23 (default stage0)
  push <local> <remote>  upload via login2
  pull <remote> <local>  download via login2
  setup-key [pubkey]     install an SSH key (recommended; one password prompt)
  close                  close the multiplexed master connection

environment overrides:
  A100_HOST=${A100_HOST}   A100_PORT=${A100_PORT}   A100_USER=${A100_USER}
  A100_GPU_HOST=${A100_GPU_HOST}   A100_PERSIST=${A100_PERSIST}
  A100_REPO=${A100_REPO}
  A100_SOCK=${A100_SOCK}
  A100_PASSWORD_FILE=<path outside this repo>   (requires sshpass)

topology: local --ssh -p ${A100_PORT}--> ${LOGIN} (login2) --ssh--> ${A100_GPU_HOST} (gpu23)
rules:    GPU 0 runs a vLLM workload; use GPU 1 only.
          Never commit a password. See docs/a100-access.md.
EOF
}

main() {
  local cmd="${1:-help}"
  shift || true
  case "$cmd" in
    status)     cmd_status "$@" ;;
    login)      cmd_login "$@" ;;
    gpu)        cmd_gpu "$@" ;;
    ablation)   cmd_ablation "$@" ;;
    push)       cmd_push "$@" ;;
    pull)       cmd_pull "$@" ;;
    setup-key)  cmd_setup_key "$@" ;;
    close)      cmd_close "$@" ;;
    help|-h|--help) cmd_help ;;
    *) die "unknown command '$cmd' (try: $0 help)" ;;
  esac
}

main "$@"
