#!/usr/bin/env bash
# Orchestrate the dynamic-routing E1/E6 matrix on ECNU gpu23 from this checkout.
#
#   bash scripts/run_dynamic_matrix.sh [tag]
#
# Steps: sync code to the shared remote working copy -> launch the remote driver
# under nohup/setsid -> poll for MATRIX_DONE -> pull the raw JSONL, reports and
# summaries back into results/l2_screening/<tag>/.
#
# SSH: the ~/.ssh/config aliases are used as-is when ~/.ssh is writable (the
# normal case). If it is not -- e.g. a read-only-home sandbox, where the
# multiplexed ControlPath cannot be created -- an equivalent config is
# generated in a temp dir from the effective `ssh -G` values, without touching
# ~/.ssh. A100_SSH_OPTS overrides both paths explicitly.
#
# Remote paths / GPU are overridable via A100_REPO, A100_ASSETS, GPU, LIMIT.
set -euo pipefail

REPO_LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-${TAG:-e1e6_$(date +%Y%m%d)}}"

REMOTE_REPO="${A100_REPO:-/public/home/xlwang/mcy/Project/STD-latest}"
REMOTE_ASSETS="${A100_ASSETS:-/public/home/xlwang/mcy/STD_assets}"
LOGIN="${A100_LOGIN:-a100}"
GPU_NODE="${A100_GPU_NODE:-a100-gpu}"
GPU="${GPU:-1}"
LIMIT="${LIMIT:-10}"
LOCAL_OUT="${A100_LOCAL_RESULTS:-$REPO_LOCAL/results/l2_screening}/$TAG"

log() { printf '[matrix] %s\n' "$*" >&2; }

# shellcheck disable=SC2206  # word splitting is the point: "-F /tmp/x" -> two args
SSH_OPTS_STR="${A100_SSH_OPTS:-}"
TMP_SSH_DIR=""
if [[ -z "$SSH_OPTS_STR" && ! -w "$HOME/.ssh" ]]; then
  # TMPDIR may itself be unwritable; fall back to a workspace-local scratch dir.
  TMP_SSH_DIR="$(mktemp -d 2>/dev/null || true)"
  if [[ -z "$TMP_SSH_DIR" ]]; then
    TMP_SSH_DIR="$REPO_LOCAL/.matrix_tmp"
    rm -rf "$TMP_SSH_DIR"
    mkdir -p "$TMP_SSH_DIR"
  fi
  eff() { ssh -G "$1" 2>/dev/null | awk -v k="$2" '$1==k {print $2; exit}'; }
  l_host="$(eff "$LOGIN" hostname)";    l_port="$(eff "$LOGIN" port)"
  l_user="$(eff "$LOGIN" user)";        l_ident="$(eff "$LOGIN" identityfile)"
  g_host="$(eff "$GPU_NODE" hostname)"; g_user="$(eff "$GPU_NODE" user)"
  : "${l_ident:=$HOME/.ssh/id_ed25519}"
  cat > "$TMP_SSH_DIR/config" <<EOF
Host $LOGIN
    HostName ${l_host:-59.78.189.133}
    Port ${l_port:-22}
    User ${l_user:-$(id -un)}
    IdentityFile $l_ident
    ControlMaster no
    ControlPath $TMP_SSH_DIR/cm-%r@%h:%p
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null

Host $GPU_NODE
    HostName ${g_host:?cannot resolve $GPU_NODE}
    User ${g_user:-$(id -un)}
    IdentityFile $l_ident
    ProxyJump $LOGIN
    ControlMaster no
    ControlPath $TMP_SSH_DIR/cmg-%r@%h:%p
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
  SSH_OPTS_STR="-F $TMP_SSH_DIR/config"
  log "~/.ssh is read-only; generated fallback ssh config at $TMP_SSH_DIR/config"
fi
trap '[[ -n "$TMP_SSH_DIR" ]] && rm -rf "$TMP_SSH_DIR"' EXIT
# shellcheck disable=SC2206
SSH_OPTS=($SSH_OPTS_STR)

gpu_ssh() { ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$GPU_NODE" "$@"; }
login_ssh() { ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$LOGIN" "$@"; }

log "tag=$TAG gpu=$GPU limit=$LIMIT"
log "remote repo  : $REMOTE_REPO"
log "remote assets: $REMOTE_ASSETS"
log "local output : $LOCAL_OUT"

log "=== 1/4 syncing code ==="
( cd "$REPO_LOCAL"
  rsync -az --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh ${SSH_OPTS_STR}" \
    src scripts tests docs PROGRESS.md README.md \
    "${LOGIN}:${REMOTE_REPO}/" )

log "=== 2/4 launching remote driver ==="
REMOTE_CMD="cd '$REMOTE_REPO' && mkdir -p '$REMOTE_ASSETS/results/$TAG' && \
  TAG='$TAG' GPU='$GPU' LIMIT='$LIMIT' REPO='$REMOTE_REPO' ASSETS='$REMOTE_ASSETS' \
  nohup setsid bash scripts/run_dynamic_matrix_remote.sh \
  > '$REMOTE_ASSETS/results/$TAG/outer.log' 2>&1 < /dev/null & echo LAUNCHED_PID=\$!"
launch_out="$(gpu_ssh "$REMOTE_CMD")"
printf '%s\n' "$launch_out" >&2
case "$launch_out" in
  *LAUNCHED_PID=*) ;;
  *) log "error: remote launch did not report a pid"; exit 1 ;;
esac

log "=== 3/4 waiting for MATRIX_DONE (driver.log tail follows) ==="
deadline=$(( $(date +%s) + ${MATRIX_TIMEOUT_S:-21600} ))
while :; do
  if gpu_ssh "test -f '$REMOTE_ASSETS/results/$TAG/MATRIX_DONE'" 2>/dev/null; then
    log "MATRIX_DONE seen"
    break
  fi
  if [[ $(date +%s) -gt $deadline ]]; then
    log "error: timed out waiting for MATRIX_DONE; remote driver may still be running"
    gpu_ssh "tail -5 '$REMOTE_ASSETS/results/$TAG/driver.log'" 2>/dev/null || true
    exit 1
  fi
  gpu_ssh "tail -3 '$REMOTE_ASSETS/results/$TAG/driver.log' 2>/dev/null" 2>/dev/null || true
  sleep 120
done

log "=== 4/4 pulling results ==="
mkdir -p "$LOCAL_OUT"
rsync -az -e "ssh ${SSH_OPTS_STR}" \
  "${LOGIN}:${REMOTE_ASSETS}/results/$TAG/" "$LOCAL_OUT/"
log "done. results in $LOCAL_OUT"
ls -1 "$LOCAL_OUT" >&2
