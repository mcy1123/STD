# Shared environment for the vLLM STD workstream. `source` this in every script.
# The harness exports a proxy (127.0.0.1:7904) that is DEAD on this host; leaving
# it set makes every outbound request fail with "Connection refused". Strip it.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"

export STD_ROOT="/home/mcy/Projects/SpecVLM/STD"
export VLLM_SRC="$STD_ROOT/third_party/vllm"
export VLLM_STD_DOCS="$STD_ROOT/docs/vllm-std"
export VLLM_STD_RES="$STD_ROOT/results/vllm-std"
export VLLM_STD_SCRIPTS="$STD_ROOT/scripts/vllm-std"

export CONDA_ENV="vllm-std"
export PY="/home/mcy/.conda/envs/$CONDA_ENV/bin/python"

# Custom proposer classes live here; vLLM imports them by module path at engine
# startup, so this must be on PYTHONPATH for every run.
export PYTHONPATH="$STD_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# --- CUDA toolkit used for RUNTIME JIT -------------------------------------------
# `/usr/local/cuda` symlinks to CUDA 11.8 on this host. vLLM pulls flashinfer, which
# JIT-compiles sampling kernels on first use with whatever `nvcc` is on PATH, and its
# bundled CCCL hard-errors under CUDA < 12:
#     #error "CUDA versions below 12 are not supported."
# which aborts engine startup. Several 12.x toolkits are installed side by side;
# point JIT at 12.6 (>=12, same major as the cu129 wheel runtime).
export CUDA_HOME="/usr/local/cuda-12.6"
export PATH="$CUDA_HOME/bin:$PATH"

# Local assets (verified to exist on host A6000)
export QWEN25VL_7B="/mnt/local/ldh/PhyLLM/Qwen2.5-VL-7B-Instruct"
export VIDEOMME="/home/mcy/Projects/SpecVLM/STD/datasets/Video-MME"

# GPU pinning: this box is shared and its occupancy changes by the minute, so the
# GPU is chosen per run (see scripts/vllm-std/pick_gpu.sh). NEVER touch other
# users' processes.
export STD_GPU="${STD_GPU:-3}"
