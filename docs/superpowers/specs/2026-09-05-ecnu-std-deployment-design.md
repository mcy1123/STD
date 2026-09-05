# ECNU Phase-8 STD Deployment Design

## Goal

Prepare the STD reproduction project on the ECNU Phase-8 cluster so that code,
the Python environment, the Qwen2.5-VL-7B-Instruct model, and a usable
Video-MME subset are available for a later A100 smoke test. This deployment
stage must remain on the internet-connected `login2` node and must not connect
to or run work on `gpu23`.

## Current State

- The working cluster account is the shared Unix account `xlwang`.
- The personal project area is `/public/home/xlwang/mcy`.
- The existing remote checkout is `/public/home/xlwang/mcy/Project/STD` at
  commit `c2a65f1`; the current local/main revision is `36046dd`.
- The remote checkout has one unrelated untracked artifact named `=2.14,`.
  Deployment will preserve it rather than delete user data implicitly.
- `/hpc_stor` has ample capacity, but `xlwang` has no allocated or writable
  directory there.
- `/public` has approximately 531 GB available under the account quota.
- The active Conda installation is owned under another member's directory,
  `/public/home/xlwang/jyy/anaconda`; its environments must not be modified.

## Directory Layout

Use the existing personal code location and an explicit path-based Conda
environment. Keep large or generated assets outside the Git checkout.

```text
/public/home/xlwang/mcy/
├── Project/STD/                         # Git checkout
├── conda_envs/specvlm/                  # isolated Python environment
└── STD_assets/
    ├── models/Qwen2.5-VL-7B-Instruct/   # model weights and processor
    ├── datasets/Video-MME/              # metadata, subtitle archive, chunk 01
    ├── cache/processor_qwen25vl/         # generated processor cache
    └── results/                          # experiment output
```

All benchmark invocations will pass these asset paths explicitly through
`MODEL_PATH`, `DATA_PATH`, `VIDEO_ROOT`, `INPUT_CACHE_DIR`, and `RESULT_DIR`.
This avoids symlinks and avoids changing the repository's path defaults.

## Code Synchronization

Update the existing remote checkout from its configured GitHub origin with a
fast-forward-only pull. Before and after synchronization, record `git status`
and the exact commit. Do not overwrite local modifications. If the checkout
cannot fast-forward cleanly, stop and report the conflict instead of resetting
or recloning.

## Environment

Create the environment at the explicit prefix
`/public/home/xlwang/mcy/conda_envs/specvlm`, with:

- Python 3.10
- PyTorch 2.6.0 and torchvision 0.21.0 from the CUDA 12.4 wheel index
- transformers 4.48.0
- datasets 2.14 or newer
- accelerate
- qwen-vl-utils 0.0.10
- PyAV 14.0.0
- Triton 3.2.0
- huggingface_hub

The setup commands will use `conda run -p <prefix>` rather than the shared
environment name `specvlm`. Login-node verification will import the pinned
packages and report versions without requiring CUDA. GPU availability and the
Triton kernel will be verified later on `gpu23`, only after explicit approval
to connect to the GPU node.

## Model and Dataset

Download `Qwen/Qwen2.5-VL-7B-Instruct` into the external model directory using
the environment's Hugging Face CLI. Prepare the default partial Video-MME
payload using the repository script with `--videomme-chunks 01`; this includes
metadata, subtitles, and video archive chunk 01. Extract chunk 01 into the
external Video-MME directory and run `check_videomme_assets.py` to report
coverage.

Do not download the full Video-MME collection or MLVU during this deployment.
MLVU may require separate access approval, and full video assets should wait
for an administrator-provisioned `/hpc_stor` directory.

## Safety and Failure Handling

- Do not connect to `gpu23` during this stage.
- Do not run compute-heavy verification on `login2`.
- Do not modify or reuse environments under another member's directory.
- Do not delete the untracked `=2.14,` artifact automatically.
- Use resumable Hugging Face downloads and retain partial files if a transfer
  is interrupted.
- Check free space before model download, dataset download, and extraction.
- Stop if projected or observed free space drops below 100 GB.
- Never print or persist SSH or Hugging Face credentials in project files.

## Completion Criteria

This deployment stage is complete when:

1. The remote STD checkout matches the approved main revision with no
   overwritten user changes.
2. The explicit-prefix environment imports all required packages and reports
   the requested versions on `login2`.
3. The Qwen2.5-VL-7B-Instruct model files pass Hugging Face cache/local-file
   validation.
4. Video-MME metadata and chunk 01 are downloaded and extracted, and asset
   coverage is reported.
5. A ready-to-run A100 smoke-test command using absolute paths is documented,
   but it is not executed until the user authorizes a GPU-node connection.

## Later Migration to `/hpc_stor`

When an administrator allocates a writable large-storage directory, move only
`STD_assets` there and update the absolute path variables. The code checkout
and Conda environment can remain in `/public/home/xlwang/mcy`.
