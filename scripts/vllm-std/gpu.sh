#!/usr/bin/env bash
# Workspace-local SSH wrapper for the ECNU A100 cluster.
# Why this exists: the local ~/.ssh/config `a100-gpu` entry uses ProxyJump, whose
# direct-tcpip channel is refused (kex_exchange_identification) from this host.
# The explicit `ssh -W` ProxyCommand form does work, and ControlPath is forced
# into /tmp because the DSH file sandbox cannot write to ~/.ssh.
# Usage: gpu.sh 'remote command'      (runs on gpu23)
#        gpu.sh -l 'remote command'   (runs on login2)
set -uo pipefail
LOGIN_CP=/tmp/dsh-cm-a100-%r@%h:%p
GPU_CP=/tmp/dsh-cm-a100gpu-%r@%h:%p
mode=gpu
if [ "${1:-}" = "-l" ]; then mode=login; shift; fi
if [ "$mode" = login ]; then
  exec ssh -o BatchMode=yes -o ControlPath="$LOGIN_CP" a100 "$@"
fi
exec ssh -o BatchMode=yes -o ControlPath="$GPU_CP" \
  -o ProxyCommand="ssh -o BatchMode=yes -o ControlPath=$LOGIN_CP -W %h:%p a100" \
  a100-gpu "$@"
