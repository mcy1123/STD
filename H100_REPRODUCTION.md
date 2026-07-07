# H100 Reproduction Notes

This note records the current practical plan for reproducing Sparse-to-Dense
speedups on a single H100.

## Goal

The target is an approximate reproduction of the paper's speedup trend rather
than an exact Table 1 reproduction. The current codebase uses Qwen2.5-VL and a
batch-1 STD decode loop, while the paper reports Qwen2-VL/LLaVA-OneVision with
batch size 8.

Expected H100 outcomes:

- `>1.0x`: realistic first target.
- `1.2x-1.5x`: possible if high-frame inputs and `triton_gqa` perform well.
- `1.7x-1.9x`: not guaranteed without batch size 8, paper models, full
  benchmark protocol, and more optimized sparse/top-K attention.

## Paper-Aligned First Run

Use the paper hyperparameters `K + text = 1024` and `gamma = 9`, with CoT
prompting. This run uses the fused Triton single-query GQA draft path.

```bash
SPECVLM_MAX_CACHE_LEN=163840 conda run -n specvlm python scripts/benchmark_std.py \
  --model-path /data/models/Qwen2.5-VL-7B-Instruct \
  --dataset Video-MME \
  --data-path /data/Std/datasets/Video-MME \
  --video-root /data/Std/datasets/Video-MME/videos \
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

If 256 frames works comfortably, try 384 frames to increase visual KV length and
make sparse attention more valuable.

```bash
SPECVLM_MAX_CACHE_LEN=245760 conda run -n specvlm python scripts/benchmark_std.py \
  --model-path /data/models/Qwen2.5-VL-7B-Instruct \
  --dataset Video-MME \
  --data-path /data/Std/datasets/Video-MME \
  --video-root /data/Std/datasets/Video-MME/videos \
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

If 384 frames runs out of memory or becomes too slow, return to 256 frames and
use the first run as the baseline.

## Higher-Acceptance Run

If `K+text=1024` has low acceptance, raise the explicit visual K. This deviates
from the paper default, but it is useful for finding the best achievable speed
on the current Qwen2.5-VL implementation.

```bash
SPECVLM_MAX_CACHE_LEN=245760 conda run -n specvlm python scripts/benchmark_std.py \
  --model-path /data/models/Qwen2.5-VL-7B-Instruct \
  --dataset Video-MME \
  --data-path /data/Std/datasets/Video-MME \
  --video-root /data/Std/datasets/Video-MME/videos \
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

After the baseline runs, sweep K and gamma to find the best H100 setting.

```bash
SPECVLM_MAX_CACHE_LEN=245760 conda run -n specvlm python scripts/sweep_std_inprocess.py \
  --dataset Video-MME \
  --data-path /data/Std/datasets/Video-MME \
  --video-root /data/Std/datasets/Video-MME/videos \
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

## Interpreting Results

Summarize the runs:

```bash
conda run -n specvlm python scripts/summarize_metrics.py \
  results/h100_videomme_256f_128tok_kplus1024_g9_triton.jsonl \
  results/h100_videomme_384f_128tok_kplus1024_g9_triton.jsonl \
  results/h100_videomme_384f_128tok_k4096_g9_triton.jsonl
```

Key fields:

- `speedup`: primary decode speedup.
- `acceptance_rate`: whether sparse draft agrees with dense enough.
- `draft_time`: if this dominates, sparse draft kernel/runtime is still the
  bottleneck.
- `verify_time`: dense verification cost.
- `retained_ratio` and `acceptance_minus_threshold`: whether the paper I/O
  condition is satisfied.

If acceptance is high but speedup is still low, the bottleneck is likely the
sparse draft kernel/runtime. If acceptance is low, increase K or sweep gamma.

## Remaining Differences From The Paper

- Current model: Qwen2.5-VL-7B-Instruct, not Qwen2-VL-7B-Instruct.
- Current STD runner: batch size 1, not paper batch size 8.
- Current data may be a downloaded subset, not the full paper benchmark split.
- `triton_gqa` is an optimization path for the current implementation, not a
  claim of exact paper kernel parity.
