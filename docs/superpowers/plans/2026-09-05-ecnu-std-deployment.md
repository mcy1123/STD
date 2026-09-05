# ECNU Phase-8 STD Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare the current STD code, an isolated reproducible environment, Qwen2.5-VL-7B-Instruct, and Video-MME chunk 01 on the internet-connected ECNU Phase-8 login node without connecting to a GPU node.

**Architecture:** Keep source code in the existing Git checkout under the user's personal project directory. Create a path-addressed Conda environment and store all large/generated assets in a sibling `STD_assets` tree, then pass absolute paths to benchmark commands so the repository remains clean and movable.

**Tech Stack:** Git, Bash, Conda, Python 3.10, PyTorch 2.6.0 (CUDA 12.4 wheels), torchvision 0.21.0, Transformers 4.48.0, Triton 3.2.0, Hugging Face Hub, Video-MME.

**Spec:** `docs/superpowers/specs/2026-09-05-ecnu-std-deployment-design.md`

## Global Constraints

- Run every deployment command on `login2`; do not SSH to `gpu23` in this stage.
- Keep code at `/public/home/xlwang/mcy/Project/STD`.
- Keep the environment at `/public/home/xlwang/mcy/conda_envs/specvlm` and address it with `conda run -p`, never by the shared name `specvlm`.
- Keep model, dataset, processor cache, and results under `/public/home/xlwang/mcy/STD_assets`.
- Install Python 3.10, PyTorch 2.6.0, torchvision 0.21.0, Transformers 4.48.0, qwen-vl-utils 0.0.10, PyAV 14.0.0, Triton 3.2.0, datasets 2.14 or newer, accelerate, and huggingface_hub.
- Preserve the untracked remote artifact `/public/home/xlwang/mcy/Project/STD/=2.14,`.
- Do not reuse or modify `/public/home/xlwang/jyy/anaconda/envs/specvlm`.
- Stop before a large download if `/public` free space is below 100 GB.
- Download only Video-MME metadata, subtitles, and chunk 01; do not download full Video-MME or MLVU.
- Do not run CUDA or model inference verification until the user separately authorizes a GPU-node connection.

---

### Task 1: Verify the target and synchronize the STD checkout

**Files:**
- Preserve: `/public/home/xlwang/mcy/Project/STD/=2.14,`
- Update through Git fast-forward: `/public/home/xlwang/mcy/Project/STD/`

**Interfaces:**
- Consumes: SSH session already authenticated as `xlwang` on the Phase-8 cluster.
- Produces: Clean, current STD source tree at `/public/home/xlwang/mcy/Project/STD` for all later tasks.

- [ ] **Step 1: Prove the session is on the login node and not the GPU node**

Run:

```bash
hostname
pwd
```

Expected: hostname is `login2`. The current directory may vary; no command in
this task depends on the starting directory.

- [ ] **Step 2: Check the checkout for tracked user changes**

Run:

```bash
cd /public/home/xlwang/mcy/Project/STD
GIT_PAGER=cat git status --short
git rev-parse HEAD
```

Expected before update: only `?? =2.14,` is reported and HEAD is
`c2a65f1889e4522a94397759199242f50225bb31`. If any tracked file is modified,
stop without pulling.

- [ ] **Step 3: Fetch and fast-forward the checkout**

Run:

```bash
cd /public/home/xlwang/mcy/Project/STD
git pull --ff-only origin main
```

Expected: fast-forward succeeds. A merge, rebase, reset, or forced checkout is
not an acceptable substitute.

- [ ] **Step 4: Verify the synchronized source tree**

Run:

```bash
cd /public/home/xlwang/mcy/Project/STD
git rev-parse HEAD
GIT_PAGER=cat git status --short
test -f src/std_repro/dynamic_std_qwen25vl.py
test -f scripts/benchmark_dynamic_std.py
test -f tests/test_adaptive_k_offline.py
```

Expected: HEAD is `36046dd0663419e7c2762575dc94d0464eaf4627`, the only untracked
entry remains `?? =2.14,`, and all three files exist.

---

### Task 2: Create and validate the deployment directory layout

**Files:**
- Create: `/public/home/xlwang/mcy/conda_envs/specvlm/` (populated in Task 3)
- Create: `/public/home/xlwang/mcy/STD_assets/models/`
- Create: `/public/home/xlwang/mcy/STD_assets/datasets/`
- Create: `/public/home/xlwang/mcy/STD_assets/cache/processor_qwen25vl/`
- Create: `/public/home/xlwang/mcy/STD_assets/results/`

**Interfaces:**
- Consumes: Writable personal directory `/public/home/xlwang/mcy`.
- Produces: Exact absolute directories consumed by environment, download, and benchmark commands.

- [ ] **Step 1: Confirm enough free space and write access**

Run:

```bash
test -w /public/home/xlwang/mcy
df -Pk /public/home/xlwang/mcy
free_kb=$(df -Pk /public/home/xlwang/mcy | awk 'NR==2 {print $4}')
test "$free_kb" -ge 104857600
```

Expected: both `test` commands exit 0 and at least 100 GiB is free.

- [ ] **Step 2: Create the isolated directory tree**

Run:

```bash
mkdir -p \
  /public/home/xlwang/mcy/conda_envs \
  /public/home/xlwang/mcy/STD_assets/models \
  /public/home/xlwang/mcy/STD_assets/datasets \
  /public/home/xlwang/mcy/STD_assets/cache/processor_qwen25vl \
  /public/home/xlwang/mcy/STD_assets/results
```

Expected: command exits 0 without creating anything under `/hpc_stor` or
another member's directory.

- [ ] **Step 3: Verify ownership and paths**

Run:

```bash
find /public/home/xlwang/mcy/STD_assets -maxdepth 2 -type d \
  -printf '%u:%g %M %p\n' | sort
```

Expected: every listed path is owned by `xlwang:xlwang` and is user-writable.

---

### Task 3: Build the explicit-prefix STD environment

**Files:**
- Create: `/public/home/xlwang/mcy/conda_envs/specvlm/`

**Interfaces:**
- Consumes: Cluster Conda executable and internet access from `login2`.
- Produces: Python executable `/public/home/xlwang/mcy/conda_envs/specvlm/bin/python` with all STD imports installed.

- [ ] **Step 1: Confirm that the target prefix is not an existing foreign environment**

Run:

```bash
if test -e /public/home/xlwang/mcy/conda_envs/specvlm/conda-meta/history; then
  echo 'Existing environment found; inspect before modifying.' >&2
  exit 3
fi
```

Expected on a fresh deployment: exit 0 with no output.

- [ ] **Step 2: Create Python 3.10 at the explicit prefix**

Run:

```bash
conda create -y -p /public/home/xlwang/mcy/conda_envs/specvlm python=3.10 pip
```

Expected: Conda completes successfully and creates the target interpreter.

- [ ] **Step 3: Install the pinned CUDA 12.4 PyTorch wheels**

Run:

```bash
conda run -p /public/home/xlwang/mcy/conda_envs/specvlm \
  python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

Expected: pip installs both pinned packages successfully.

- [ ] **Step 4: Install STD's remaining dependencies**

Run:

```bash
conda run -p /public/home/xlwang/mcy/conda_envs/specvlm \
  python -m pip install \
  transformers==4.48.0 \
  'datasets>=2.14' \
  accelerate \
  qwen-vl-utils==0.0.10 \
  av==14.0.0 \
  triton==3.2.0 \
  huggingface_hub
```

Expected: pip exits 0. The quoted datasets requirement must remain quoted so
the shell does not create another `=2.14` artifact.

- [ ] **Step 5: Verify imports and exact pinned versions without probing CUDA**

Run:

```bash
conda run -p /public/home/xlwang/mcy/conda_envs/specvlm python - <<'PY'
import av
import datasets
import huggingface_hub
import qwen_vl_utils
import torch
import transformers
import triton

assert torch.__version__.startswith('2.6.0')
assert torch.version.cuda == '12.4'
assert transformers.__version__ == '4.48.0'
assert av.__version__ == '14.0.0'
assert triton.__version__ == '3.2.0'
print({
    'torch': torch.__version__,
    'torch_cuda_wheel': torch.version.cuda,
    'transformers': transformers.__version__,
    'datasets': datasets.__version__,
    'av': av.__version__,
    'triton': triton.__version__,
    'huggingface_hub': huggingface_hub.__version__,
    'cuda_probe_skipped': True,
})
PY
```

Expected: all assertions pass. `torch.cuda.is_available()` is deliberately not
used on the login node.

---

### Task 4: Download and validate Qwen2.5-VL-7B-Instruct

**Files:**
- Create: `/public/home/xlwang/mcy/STD_assets/models/Qwen2.5-VL-7B-Instruct/`

**Interfaces:**
- Consumes: `huggingface_hub` from the Task 3 environment and internet access.
- Produces: A local Transformers model directory consumed by `--model-path`.

- [ ] **Step 1: Recheck the 100 GiB free-space floor**

Run:

```bash
free_kb=$(df -Pk /public/home/xlwang/mcy | awk 'NR==2 {print $4}')
test "$free_kb" -ge 104857600
```

Expected: exit 0 before the model transfer starts.

- [ ] **Step 2: Download the model with resumable snapshot semantics**

Run:

```bash
/public/home/xlwang/mcy/conda_envs/specvlm/bin/python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id='Qwen/Qwen2.5-VL-7B-Instruct',
    local_dir='/public/home/xlwang/mcy/STD_assets/models/Qwen2.5-VL-7B-Instruct',
)
print(path)
PY
```

Expected: the command exits 0. If authentication is requested, stop and ask
the user to authenticate interactively without exposing a token in logs.

- [ ] **Step 3: Validate the model index and every referenced weight shard**

Run:

```bash
conda run -p /public/home/xlwang/mcy/conda_envs/specvlm python - <<'PY'
import json
from pathlib import Path

root = Path('/public/home/xlwang/mcy/STD_assets/models/Qwen2.5-VL-7B-Instruct')
for name in ('config.json', 'preprocessor_config.json', 'tokenizer_config.json'):
    assert (root / name).is_file(), name
index = root / 'model.safetensors.index.json'
assert index.is_file(), index
weight_map = json.loads(index.read_text())['weight_map']
shards = sorted(set(weight_map.values()))
missing = [name for name in shards if not (root / name).is_file()]
assert not missing, missing
partials = list(root.rglob('*.incomplete'))
assert not partials, partials
print({'weight_shards': len(shards), 'model_bytes': sum((root / x).stat().st_size for x in shards)})
PY
```

Expected: no missing shards or incomplete files are reported.

---

### Task 5: Download, extract, and inventory Video-MME chunk 01

**Files:**
- Create: `/public/home/xlwang/mcy/STD_assets/datasets/Video-MME/`
- Create: `/public/home/xlwang/mcy/STD_assets/results/videomme_asset_check.jsonl`

**Interfaces:**
- Consumes: Updated STD dataset scripts and the Task 3 environment.
- Produces: Local Video-MME metadata plus extracted videos consumed by `--data-path` and `--video-root`.

- [ ] **Step 1: Recheck the 100 GiB free-space floor**

Run:

```bash
free_kb=$(df -Pk /public/home/xlwang/mcy | awk 'NR==2 {print $4}')
test "$free_kb" -ge 104857600
```

Expected: exit 0 before the dataset transfer starts.

- [ ] **Step 2: Download only the approved Video-MME subset**

Run:

```bash
cd /public/home/xlwang/mcy/Project/STD
conda run --no-capture-output \
  -p /public/home/xlwang/mcy/conda_envs/specvlm \
  python scripts/prepare_std_datasets.py \
  --dataset Video-MME \
  --output-root /public/home/xlwang/mcy/STD_assets/datasets \
  --videomme-chunks 01
```

Expected: metadata, `subtitle.zip`, and `videos_chunked_01.zip` download; no
other video chunks are requested.

- [ ] **Step 3: Extract chunk 01 into the external video root**

Run:

```bash
cd /public/home/xlwang/mcy/Project/STD
conda run --no-capture-output \
  -p /public/home/xlwang/mcy/conda_envs/specvlm \
  python scripts/extract_videomme_chunks.py \
  --dataset-dir /public/home/xlwang/mcy/STD_assets/datasets/Video-MME \
  --output-dir /public/home/xlwang/mcy/STD_assets/datasets/Video-MME/videos \
  --chunks 01
```

Expected: extraction exits 0 and creates playable video files under `videos/`.

- [ ] **Step 4: Record Video-MME asset coverage**

Run:

```bash
cd /public/home/xlwang/mcy/Project/STD
conda run --no-capture-output \
  -p /public/home/xlwang/mcy/conda_envs/specvlm \
  python scripts/check_videomme_assets.py \
  --metadata-path /public/home/xlwang/mcy/STD_assets/datasets/Video-MME \
  --video-root /public/home/xlwang/mcy/STD_assets/datasets/Video-MME/videos \
  --output /public/home/xlwang/mcy/STD_assets/results/videomme_asset_check.jsonl
```

Expected: command prints `found=<positive number> total=<number>` and writes
one JSON object per checked metadata row.

---

### Task 6: Perform login-node-only final verification and prepare the GPU command

**Files:**
- Read: `/public/home/xlwang/mcy/Project/STD/src/`
- Read: `/public/home/xlwang/mcy/Project/STD/scripts/`
- Read: `/public/home/xlwang/mcy/STD_assets/`

**Interfaces:**
- Consumes: Synchronized code, verified environment, model, and dataset from Tasks 1-5.
- Produces: Evidence that preparation is complete plus an absolute-path command for a later A100 smoke test.

- [ ] **Step 1: Compile the Python source without importing CUDA workloads**

Run:

```bash
cd /public/home/xlwang/mcy/Project/STD
conda run -p /public/home/xlwang/mcy/conda_envs/specvlm \
  python -m compileall -q src scripts tests
```

Expected: exit 0 with no syntax errors.

- [ ] **Step 2: Summarize installed asset sizes and remaining disk space**

Run:

```bash
du -sh \
  /public/home/xlwang/mcy/conda_envs/specvlm \
  /public/home/xlwang/mcy/STD_assets/models/Qwen2.5-VL-7B-Instruct \
  /public/home/xlwang/mcy/STD_assets/datasets/Video-MME \
  /public/home/xlwang/mcy/STD_assets/cache \
  /public/home/xlwang/mcy/STD_assets/results
df -h /public/home/xlwang/mcy
```

Expected: all required paths exist and at least 100 GB remains free.

- [ ] **Step 3: Confirm the session never left the login node**

Run:

```bash
hostname
```

Expected: `login2`.

- [ ] **Step 4: Preserve this unexecuted A100 smoke-test command for handoff**

Do not execute it in this deployment stage:

```bash
cd /public/home/xlwang/mcy/Project/STD
SPECVLM_MAX_CACHE_LEN=40960 \
conda run --no-capture-output \
  -p /public/home/xlwang/mcy/conda_envs/specvlm \
  python scripts/benchmark_std.py \
  --model-path /public/home/xlwang/mcy/STD_assets/models/Qwen2.5-VL-7B-Instruct \
  --dataset Video-MME \
  --data-path /public/home/xlwang/mcy/STD_assets/datasets/Video-MME \
  --video-root /public/home/xlwang/mcy/STD_assets/datasets/Video-MME/videos \
  --eval-num 3 \
  --frame-num 32 \
  --max-new-tokens 64 \
  --gamma 9 \
  --target-k-plus-text 1024 \
  --prompt-style cot \
  --gpu-ids 0 \
  --output /public/home/xlwang/mcy/STD_assets/results/a100_smoke.jsonl
```

Expected: the command is included in the deployment handoff, and no GPU
process is started during this plan.
