# H100 Reproduction Notes

Target: reproduce Sparse-to-Dense speedups on a single H100.

## Setup

```bash
git clone <this-repo-url> STD
cd STD
bash scripts/setup_env.sh
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/Qwen2.5-VL-7B-Instruct
conda run -n specvlm python scripts/prepare_std_datasets.py --dataset Video-MME
conda run -n specvlm python scripts/extract_videomme_chunks.py --chunks 01
```

## Paper-Aligned First Run

`K + text = 1024`, `gamma = 9`, CoT, Triton GQA:

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
  --gpu-ids 0 \
  --output results/h100_videomme_256f_128tok_kplus1024_g9_triton.jsonl
```

## Higher-Frame Run

If 256 frames works comfortably, try 384:

```bash
SPECVLM_MAX_CACHE_LEN=245760 conda run -n specvlm python scripts/benchmark_std.py \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --dataset Video-MME \
  --data-path datasets/Video-MME \
  --video-root datasets/Video-MME/videos \
  --eval-num 30 \
  --frame-num 384 \
  --max-new-tokens 128 \
  --gamma 9 \
  --target-k-plus-text 1024 \
  --prompt-style cot \
  --sparse-attn-mode triton_gqa \
  --gpu-ids 0 \
  --output results/h100_videomme_384f_128tok_kplus1024_g9_triton.jsonl
```

## Higher-Acceptance Run

If acceptance is low, increase explicit K:

```bash
SPECVLM_MAX_CACHE_LEN=245760 conda run -n specvlm python scripts/benchmark_std.py \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --dataset Video-MME \
  --data-path datasets/Video-MME \
  --video-root datasets/Video-MME/videos \
  --eval-num 30 \
  --frame-num 384 \
  --max-new-tokens 128 \
  --gamma 9 \
  --k 4096 \
  --prompt-style cot \
  --sparse-attn-mode triton_gqa \
  --gpu-ids 0 \
  --output results/h100_videomme_384f_128tok_k4096_g9_triton.jsonl
```

## Sweep

```bash
SPECVLM_MAX_CACHE_LEN=245760 conda run -n specvlm python scripts/sweep_std_inprocess.py \
  --dataset Video-MME \
  --data-path datasets/Video-MME \
  --video-root datasets/Video-MME/videos \
  --prompt-style cot \
  --ks 2048,4096,8192 \
  --gammas 9,13,15,17 \
  --frame-num 384 \
  --max-new-tokens 128 \
  --eval-num 10 \
  --sparse-attn-mode triton_gqa \
  --gpu-ids 0 \
  --output results/h100_videomme_384f_sweep.jsonl
```

## View Results

```bash
conda run -n specvlm python scripts/summarize_metrics.py results/h100_*.jsonl
```

## Expected Outcomes

- `>1.0x`: realistic first target.
- `1.2x-1.5x`: possible with high frames + `triton_gqa`.
- `1.7x-1.9x`: not guaranteed without batch size 8, paper models, and optimized top-K attention kernel.

## Remaining Differences From The Paper

- Current model: Qwen2.5-VL-7B-Instruct (paper: Qwen2-VL / LLaVA-OneVision)
- Current STD runner: batch size 1 (paper: batch size 8)
- Current data: downloaded subset, not full paper benchmark split
- `triton_gqa` is an optimization path, not exact paper kernel parity


cd /czsun/lsw/mcy/STD/STD
SPECVLM_MAX_CACHE_LEN=40960 conda run --no-capture-output -n specvlm python scripts/benchmark_std.py \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --dataset Video-MME \
  --data-path datasets/Video-MME \
  --video-root datasets/Video-MME/videos \
  --eval-num 100000 \
  --frame-num 256 \
  --max-new-tokens 128 \
  --gamma 9 \
  --target-k-plus-text 8192 \
  --prompt-style cot \
  --verify-mode parallel \
  --sparse-attn-mode gqa_sdpa \
  --gpu-ids 0 \
  --output results/h100_videomme_256f_kplus8192_g9_sdpa.jsonl