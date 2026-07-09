# Sparse-to-Dense (STD) Reproduction

Reproducing the paper **"Sparse-to-Dense: A Free Lunch for Lossless Acceleration
of Video Understanding in LLMs"** with Qwen2.5-VL-7B-Instruct.

This repository is **self-contained** — no external code dependencies.
Clone, install, and run.

## Quick Start

```bash
# 1. Install environment
bash scripts/setup_env.sh

# 2. Download model
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir models/Qwen2.5-VL-7B-Instruct

# 3. Download Video-MME dataset
conda run -n specvlm python scripts/prepare_std_datasets.py --dataset Video-MME
conda run -n specvlm python scripts/extract_videomme_chunks.py --chunks 01

# 4. Smoke test (3 samples, 32 frames, single GPU)
SPECVLM_MAX_CACHE_LEN=40960 conda run -n specvlm python scripts/benchmark_std.py \
  --eval-num 3 --frame-num 32 --max-new-tokens 64 \
  --gamma 9 --target-k-plus-text 1024 --prompt-style cot \
  --gpu-ids 0
```

## Repository Structure

```text
STD/
├── src/
│   ├── std_repro/          # STD core algorithm
│   │   ├── std_qwen25vl.py   # AR & STD decoding, sparse KV selection
│   │   └── triton_attention.py  # Fused Triton GQA kernel (optional)
│   └── specvlm/            # Bundled SpecVLM (Qwen2.5-VL model + KV cache)
│       ├── models/           # modeling_qwen2_5_vl, processing, config
│       ├── kv_cache/         # KV cache initialization
│       └── utils/            # get_last_video_idx
├── scripts/
│   ├── benchmark_std.py       # Main benchmark entry point
│   ├── sweep_std_inprocess.py # In-process K/gamma sweep (load model once)
│   ├── sweep_std.py           # Multi-process K/gamma sweep
│   ├── summarize_metrics.py   # Pretty-print JSONL results
│   ├── setup_env.sh           # One-click conda environment setup
│   ├── prepare_std_datasets.py  # Download MLVU / Video-MME
│   ├── extract_videomme_chunks.py
│   ├── download_videomme_youtube.py
│   └── check_videomme_assets.py
├── datasets/               # (downloaded) MLVU / Video-MME
├── models/                 # (downloaded) Qwen2.5-VL-7B-Instruct
└── results/                # JSONL output files
```

## Paper-Aligned Configuration

The paper uses `K + text = 1024` and `gamma = 9` with CoT prompting:

```bash
SPECVLM_MAX_CACHE_LEN=163840 conda run -n specvlm python scripts/benchmark_std.py \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --dataset Video-MME \
  --data-path datasets/Video-MME \
  --video-root datasets/Video-MME/videos \
  --eval-num 30 \
  --frame-num 256 \
  --max-new-tokens 128 \
  --gamma 9 \
  --target-k-plus-text 1024 \
  --prompt-style cot \
  --sparse-attn-mode triton_gqa \
  --gpu-ids 0
```

## Key Parameters

| Parameter | Description | Default |
|---|---|---|
| `--frame-num` | Video frames sampled | 32 |
| `--gamma` | Draft tokens per round | 9 |
| `--target-k-plus-text` | Sparse KV budget (K + text tokens) | 1024 |
| `--k` | Explicit visual K (overrides target-k-plus-text) | None |
| `--max-new-tokens` | Max generated tokens | 64 |
| `--verify-mode` | `parallel` (fast) or `sequential` (exact) | parallel |
| `--sparse-attn-mode` | `gqa_sdpa`, `repeat_sdpa`, or `triton_gqa` | gqa_sdpa |
| `--prompt-style` | `direct` or `cot` (chain-of-thought) | direct |
| `--gpu-ids` | Comma-separated GPU indices | 0,1,2,3 |

## Environment Variables

- `SPECVLM_MAX_CACHE_LEN` — KV cache pre-allocation length. Higher frames need larger values:
  - 32 frames → `40960`
  - 144 frames → `81920`
  - 256 frames → `163840`
  - 384 frames → `245760`

## Viewing Results

```bash
conda run -n specvlm python scripts/summarize_metrics.py results/std_qwen2_5_vl_7b/*.jsonl
conda run -n specvlm python scripts/summarize_metrics.py --group-by gamma results/std_qwen2_5_vl_7b/sweep.jsonl
```

## H100 / A100 GPU Notes

- A100/H100 80GB can run 256+ frames on a **single GPU** (RTX 3090 is limited to ~144 frames across 4 GPUs)
- Higher frames mean more visual KV → bigger sparse attention benefit
- Use `--sparse-attn-mode triton_gqa` for the fused Triton kernel on H100/A100

## Current Implementation Notes

- Uses one Qwen2.5-VL model instance with separate KV caches for selection, dense verification, and sparse drafting
- Dense cache verifies with full attention; sparse draft cache uses top-K visual KV selected from text-to-video attention during prefill
- Greedy decoding only — correctness check is exact token equality with vanilla AR
- `--verify-mode parallel` matches the paper-style parallel dense verification
- `--verify-attn-backend math` forces the math SDPA backend for stricter lossless verification
- `--verify-margin-threshold` + `--verify-fallback sequential_on_low_accept` provides low-overhead strict verification

## Remaining Differences From The Paper

- Current model: Qwen2.5-VL-7B-Instruct (paper uses Qwen2-VL / LLaVA-OneVision)
- Current STD runner: batch size 1 (paper reports batch size 8)
- PyTorch SDPA instead of fused top-K attention kernel
- Speedup ceiling depends on GPU compute bandwidth (A100/H100 > RTX 3090)
