#!/usr/bin/env bash
# Phase 0: build the isolated vLLM environment for the STD workstream.
#
# WHY vLLM 0.22.0 + cu129 (this is the crux of the whole workstream):
#   * The injection point we need -- `speculative_config.method="custom_class"` --
#     first appears in vLLM v0.22.0. It does NOT exist in <=0.21.0.
#   * vLLM >=0.22.0 pins torch==2.11.0, whose *PyPI* wheel is built for CUDA 13
#     (nvidia-*-cu13 deps). Host A6000 has driver 550.163.01 == CUDA 12.4, and a
#     CUDA 13 runtime refuses to load there (libcudart.so.13 missing). We proved
#     this: the pre-existing `vllm-omni-dev-fast` env ships vllm 0.28.0 and fails
#     `import vllm` with exactly that error.
#   * BUT vLLM publishes alternate-CUDA wheels and PyTorch publishes alternate-CUDA
#     torch builds. `vllm-0.22.0+cu129` and `torch-2.11.0+cu129` both exist.
#     CUDA 12.9 is covered by CUDA 12.x *minor version compatibility*
#     (driver >= 525.60.13 runs any CUDA 12.x app), so cu129 loads on driver 550.
#
#   Net effect: we get the real, unpatched custom_class proposer API on this box.
#
# Fallback if cu129 fails to initialise on this driver: pin vLLM 0.19.1
# (torch 2.10.0, cu128, the last CUDA-12-native release) and backport the
# custom_class hook (config/speculative.py + v1/worker/gpu_model_runner.py).
set -euo pipefail
source "$(dirname "$0")/env.sh"

PYPI="https://pypi.org/simple"
TUNA="https://pypi.tuna.tsinghua.edu.cn/simple"
PT_CU129="https://download.pytorch.org/whl/cu129"

VLLM_WHEEL_NAME="vllm-0.22.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_WHEEL_URL="https://wheels.vllm.ai/0b3ba88f165976e77ca5e6a7a3f5bba4562b80af/$VLLM_WHEEL_NAME"
VLLM_WHEEL_SHA256="405319a7fde1af416eef6cbab0315d354d8b9c1a45b97097a9f1c3707bf05f74"

PIP="$PY -m pip"

echo "=== [0/3] land pinned vLLM source tree (github.com git clone times out here) ==="
VLLM_SRC_TARBALL_SHA256="72e40a471069d5285a1f57d2d056b18a10acd82f9fa5a403104affa2cfe23fd8"
if [ ! -f "$VLLM_SRC/vllm/v1/spec_decode/custom_class_proposer.py" ]; then
  mkdir -p "$(dirname "$VLLM_SRC")"
  curl -sSL -o /tmp/vllm-0.22.0-src.tar.gz \
    https://codeload.github.com/vllm-project/vllm/tar.gz/refs/tags/v0.22.0
  echo "$VLLM_SRC_TARBALL_SHA256  /tmp/vllm-0.22.0-src.tar.gz" | sha256sum -c -
  rm -rf "$VLLM_SRC"; mkdir -p "$VLLM_SRC"
  tar xzf /tmp/vllm-0.22.0-src.tar.gz -C "$VLLM_SRC" --strip-components=1
  ( cd "$VLLM_SRC"
    git init -q; git config user.email agent@local; git config user.name "STD agent"
    printf '__pycache__/\n*.pyc\n' > .git/info/exclude
    git add -A && git commit -qm "pristine vLLM v0.22.0 (codeload tarball)"
    git tag -a v0.22.0-pristine -m "upstream v0.22.0 release tarball, unmodified"
    git checkout -q -b feat/std-proposer )
fi
echo "source tree: $(cd "$VLLM_SRC" && git rev-parse --short HEAD) on $(cd "$VLLM_SRC" && git rev-parse --abbrev-ref HEAD)"

$PIP install --upgrade pip setuptools wheel -i "$TUNA" 2>&1 | tail -2

echo "=== [1/3] torch 2.11.0+cu129 (CUDA 12.9 runtime) ==="
$PIP install \
  "torch==2.11.0+cu129" "torchvision==0.26.0+cu129" "torchaudio==2.11.0+cu129" \
  --index-url "$PT_CU129" 2>&1 | tail -5

echo "=== [2/3] vllm 0.22.0+cu129 (direct wheel URL) ==="
# vLLM's alt-CUDA wheels are NOT a PEP 503 simple index -- the listing at
# https://wheels.vllm.ai/0.22.0/cu129/vllm/ is a plain directory listing, so
# `pip install vllm==0.22.0+cu129 --extra-index-url ...` dies with
# "No matching distribution found". Fetch the wheel by direct object URL.
WL="/tmp/$VLLM_WHEEL_NAME"
[ -f "$WL" ] || curl -sSL -o "$WL" "$VLLM_WHEEL_URL"
echo "$VLLM_WHEEL_SHA256  $WL" | sha256sum -c - || { echo "WHEEL CHECKSUM MISMATCH"; exit 1; }
# torch==2.11.0 (vllm's exact pin) is satisfied by the installed 2.11.0+cu129:
# PEP 440 `==2.11.0` ignores the local version segment, so pip will not pull cu13.
$PIP install "$WL" \
  --extra-index-url "$TUNA" \
  --extra-index-url "$PT_CU129" 2>&1 | tail -8

echo "=== [3/3] verify ==="
$PY - <<'EOF'
import torch, vllm
print("torch      :", torch.__version__)
print("vllm       :", vllm.__version__)
print("cuda build :", torch.version.cuda)
print("devices    :", torch.cuda.device_count())
EOF
