# Sparse-to-Dense Qwen2.5-VL Reproduction

This project reproduces the STD paper with the local Qwen2.5-VL-7B-Instruct
checkpoint and the existing SpecVLM Qwen2.5-VL fork.

## Paths

- Model: `/home/mcy/projects/models/Qwen2.5-VL-7B-Instruct`
- SpecVLM codebase: `/home/mcy/projects/SpecVLM`
- Smoke dataset: `/home/mcy/projects/SpecVLM/datasets/VideoDetailCaption`
- Paper datasets:
  - MLVU: `MLVU/MVLU`
  - Video-MME: `lmms-lab/Video-MME`

## Quick Smoke Test

```bash
cd /home/mcy/projects/Std
GPU_IDS=0,1,2,3 bash scripts/run_std_smoke.sh
```

The smoke test writes JSONL metrics to:

```text
results/std_qwen2_5_vl_7b/smoke_videodetailcaption.jsonl
```

## Download Paper Datasets

```bash
cd /home/mcy/projects/Std
conda run -n specvlm python scripts/prepare_std_datasets.py --dataset MLVU
conda run -n specvlm python scripts/prepare_std_datasets.py --dataset Video-MME
```

By default the script avoids full 400GB/94GB downloads:

- MLVU: downloads annotations only.
- Video-MME: downloads metadata plus `videos_chunked_01.zip`.

Use `--full` only when the target disk has enough free space. Current observed
dataset constraints on this machine:

- `MLVU/MVLU` is gated on Hugging Face and returned 401 without authentication.
- `lmms-lab/Video-MME` metadata downloaded successfully; video chunk 01 supports
  resume but was interrupted before completion.

After a Video-MME chunk completes:

```bash
conda run -n specvlm python scripts/extract_videomme_chunks.py --chunks 01
```

## Main Benchmark Entry

```bash
cd /home/mcy/projects/Std
conda run -n specvlm python scripts/benchmark_std.py \
  --model-path /home/mcy/projects/models/Qwen2.5-VL-7B-Instruct \
  --dataset VideoDetailCaption \
  --data-path /home/mcy/projects/SpecVLM/datasets/VideoDetailCaption \
  --eval-num 1 \
  --frame-num 32 \
  --max-new-tokens 64 \
  --gamma 9 \
  --target-k-plus-text 1024 \
  --gpu-ids 0,1,2,3
```

Each output record includes token equality, AR/STD decoding time, AR/STD total
inference time, speedup, acceptance rate, mean accepted draft length, visual
token count, text length, and selected K.

## Current Implementation

- Uses one Qwen2.5-VL model instance. The correctness-first path keeps separate
  selection, dense verifier, and sparse draft KV caches because the
  `output_attentions=True` selection prefill does not stay numerically
  interchangeable with the normal SDPA verifier prefill.
- Dense cache verifies with full attention.
- Sparse draft cache uses top-K visual KV selected from text-to-video attention
  during prefill. By default it copies the normal dense verifier prompt KV into
  the sparse cache, then physically compacts that cache before draft decoding;
  this avoids a redundant sparse prompt prefill while keeping the verifier on
  the normal SDPA path.
- Sparse draft decode passes original absolute `position_ids` after compaction,
  so RoPE positions stay aligned with the full prompt.
- Sparse draft attention defaults to `--sparse-attn-mode gqa_sdpa`, which uses
  PyTorch SDPA GQA support instead of explicitly expanding KV heads with
  `repeat_kv`. Use `--sparse-attn-mode repeat_sdpa` to reproduce the older
  path. Use `--sparse-attn-mode triton_gqa` to try the fused Triton
  single-query GQA draft kernel on the current batch-1 decode path.
- Greedy decoding only (`do_sample=False` equivalent), so the required
  correctness check is exact token equality with vanilla AR.
- Formal AR and STD generation run under `torch.inference_mode()` to avoid
  autograd metadata overhead during decode.
- `scripts/benchmark_std.py` records token mismatch and continues by default;
  pass `--strict-equality` to abort on mismatch.
- `--verify-mode parallel` matches the paper-style parallel dense verification
  and is the speed path.
- `--verify-mode sequential` verifies draft tokens with one-token dense steps.
  It is slower, but useful to prove strict token equality when fp16/SDPA
  parallel verification drifts numerically from vanilla AR.
- `--verify-attn-backend math` forces the parallel dense verifier's SDPA calls
  onto the math backend. It is slower than the default SDPA backend, but it
  fixed the observed 144-frame false-accept mismatch and is the stricter
  lossless verification path.
- `--prompt-style cot` is available for MLVU/Video-MME style multiple-choice
  rows and asks the model to produce chain-of-thought before the final option,
  matching the paper's long-response evaluation setting more closely.
- `--profile-decode` records synchronized STD decode stage timings for
  profiling; leave it off for speed comparisons.
- `--adaptive-gamma-min` enables an experimental adaptive draft length. It is
  off by default because the current simple heuristic has not improved the
  measured results.
- `--verify-fallback sequential_on_reject` rechecks rejected parallel
  verification rounds sequentially. It is useful for diagnosing false
  parallel-verification rejection, but it is off by default because it adds
  dense work.
- `--verify-margin-threshold` reruns low-confidence dense verifier rounds with
  the math SDPA backend before accepting tokens. Combining it with
  `--verify-fallback sequential_on_low_accept --sequential-fallback-max-accept 1`
  is the current lowest-overhead strict guard found for high-acceptance runs.
- `--reuse-dense-prefill` is an experimental diagnostic only. A 32-frame
  Video-MME smoke test showed that reusing the attention-selection prefill as
  the dense verifier cache caused token mismatch, so formal runs leave it off.
- `--no-copy-sparse-prefill` restores the older path that recomputes a separate
  sparse prompt prefill. The default copy path preserved exact token equality in
  the current tests and reduced STD total inference time substantially.

## Current Local Results

All rows below use `VideoDetailCaption`, one sample, Qwen2.5-VL-7B-Instruct,
and 4x RTX 3090.

| Setting | Token equal | Acceptance | Speedup |
| --- | --- | ---: | ---: |
| 8 frames, 8 tokens, gamma=3, K+text=256 | true | 1.000 | 0.880x |
| 32 frames, 64 tokens, gamma=9, K+text=1024 | true | 0.778 | 0.686x |
| 128 frames, 64 tokens, gamma=9, K+text=1024 | true | 0.460 | 0.443x |
| 128 frames, 64 tokens, gamma=5, K=4096 | true | 0.930 | 0.732x |
| 128 frames, 64 tokens, gamma=9, K=4096 | false | 0.877 | 0.766x |

Five-sample local run at `128 frames / 64 tokens / gamma=5 / K=4096`:

```text
samples: 5
token_equal: 3/5
speedup: 0.710x
acceptance_rate: 0.879
mean_accept_length: 4.247
ar_decoding_time: 4.796s
std_decoding_time: 6.780s
```

The third sample from that run diverged under `parallel` verification but passed
with `--verify-mode sequential --strict-equality`, confirming that the observed
mismatches are caused by parallel verification numerical differences rather than
sparse cache corruption.

Video-MME short, one downloaded video (`fFjv93ACGo8`), three CoT questions,
`32 frames / 64 tokens / gamma=9 / K=2048`:

```text
samples: 3
token_equal: 3/3
speedup: 0.921x
acceptance_rate: 0.983
mean_accept_length: 8.286
ar_decoding_time: 4.669s
std_decoding_time: 5.068s
```

After deferring dense bonus-token cache updates, a high-acceptance setting now
exceeds 1x on the same Video-MME CoT mini-set:

```text
Setting: 32 frames / 64 tokens / gamma=31 / K=4096
samples: 3
token_equal: 3/3
speedup: 1.024x
acceptance_rate: 1.000
mean_accept_length: 31.000
ar_decoding_time: 4.678s
std_decoding_time: 4.568s
```

Note: for this particular video, `K=4096` clips to all `3780` visual tokens, so
this is best interpreted as the current engineering upper-bound / self-draft
speculative result. The genuinely sparse `K=2048, gamma=31` setting remained
lossless but averaged `0.818x` because acceptance dropped to `0.806`.

Video-MME short, five downloaded videos, fifteen CoT questions, `32 frames /
64 tokens / gamma=9`:

```text
K+text=1024:
  samples: 15
  token_equal: 15/15
  speedup: 0.882x
  acceptance_rate: 0.942
  mean_accept_length: 7.839

K=2048:
  samples: 15
  token_equal: 15/15
  speedup: 0.926x
  acceptance_rate: 0.992
  mean_accept_length: 8.276
```

These 15-question runs are the closest current paper-style evidence: Video-MME,
CoT prompting, sparse visual KV selection, and lossless token equality. They are
still below the paper's reported speedups because this reproduction runs on
Qwen2.5-VL-7B with PyTorch SDPA on RTX 3090s rather than the paper's Qwen2-VL /
LLaVA-OneVision setup on A100 80GB, and it does not yet include a fused top-K
attention kernel.

After downloading five more short videos from Video-MME YouTube metadata, the
paper-like `K+text=1024, gamma=9` setting was expanded to ten videos / thirty
CoT questions:

```text
samples: 30
token_equal: 30/30
speedup: 0.885x
acceptance_rate: 0.937
mean_accept_length: 7.860
ar_decoding_time: 4.567s
std_decoding_time: 5.188s
selected K range: 938-978
visual tokens: 27 samples at 3780, 3 samples at 4032
```

This is the most stable current lossless reproduction result. It confirms the
STD sparse-to-dense decoding logic on Qwen2.5-VL-7B, but it still does not
reproduce the paper's claimed wall-clock speedup on the local RTX 3090 setup.

Keeping `K+text=1024` fixed and increasing gamma did not improve speed on a
ten-question subset:

```text
gamma=5:  samples=10, token_equal=10/10, speedup=0.844x, acceptance=0.948
gamma=7:  samples=10, token_equal=10/10, speedup=0.870x, acceptance=0.934
gamma=9:  samples=10, token_equal=10/10, speedup=0.868x, acceptance=0.920
gamma=11: samples=10, token_equal=10/10, speedup=0.870x, acceptance=0.911
gamma=13: samples=10, token_equal=10/10, speedup=0.840x, acceptance=0.869
gamma=17: samples=10, token_equal=10/10, speedup=0.834x, acceptance=0.850
```

The current best `K+text=1024` setting is effectively between `gamma=7` and
`gamma=9`; larger gamma values lose speed because acceptance drops.

Relaxing sparse K improves acceptance and speed but is still below 1x on the
ten-question subset:

```text
K=2048, gamma=9:  samples=10, token_equal=10/10, speedup=0.927x, acceptance=0.992
K=2048, gamma=13: samples=10, token_equal=10/10, speedup=0.932x, acceptance=0.967
K=2560, gamma=9:  samples=10, token_equal=10/10, speedup=0.923x, acceptance=0.987
K=2560, gamma=13: samples=10, token_equal=10/10, speedup=0.934x, acceptance=0.969
```

`K=2560, gamma=13` is the fastest genuinely sparse local setting found so far,
although it still trails vanilla AR on this hardware/software stack.

Expanded to the full ten-video / thirty-question local set with the original
`repeat_kv + SDPA` sparse attention path:

```text
K=2560, gamma=13
samples: 30
token_equal: 30/30
speedup: 0.949x
acceptance_rate: 0.980
mean_accept_length: 11.770
ar_decoding_time: 4.696s
std_decoding_time: 4.956s
per-sample speedup range: 0.810x-1.154x
samples above 1x: 1/30
```

Switching sparse draft attention to PyTorch SDPA's GQA mode avoids explicit
`repeat_kv` in the sparse draft path and improves the same setting:

```text
K=2560, gamma=13, sparse_attn_mode=gqa_sdpa
samples: 30
token_equal: 30/30
speedup: 0.965x
acceptance_rate: 0.975
mean_accept_length: 11.691
ar_decoding_time: 4.612s
std_decoding_time: 4.788s
per-sample speedup range: 0.830x-1.001x
samples above 1x: 1/30
```

This is the best current 30-question result, improving over `K+text=1024,
gamma=9` (`0.885x`) and over the previous `repeat_kv + SDPA` best (`0.949x`),
but it is still short of the STD paper's reported acceleration.

Profiling the best setting on five questions shows where the remaining wall
time goes:

```text
draft_time: 86.5% of STD decode
verify_time: 7.7% of STD decode
bonus_time: 5.8% of STD decode
```

The remaining speed gap is therefore dominated by sparse draft single-token
forwards, not dense verification.

Longer 128-token generation did not improve the result on the current
parallel-verification path:

```text
K=2560, gamma=13, sparse_attn_mode=gqa_sdpa, 128 max new tokens
samples: 10
token_equal: 8/10
speedup: 0.949x
acceptance_rate: 0.971
mean_accept_length: 12.141
ar_decoding_time: 8.753s
std_decoding_time: 9.230s
```

One mismatched sample (`010-3`) passed with
`--verify-mode sequential --strict-equality`, confirming that this long-output
failure mode is parallel verification numerical drift rather than sparse cache
corruption. Sequential verification is much slower (`0.527x` on that sample),
so it is useful for diagnosis rather than acceleration.

Increasing the frame count to 128 creates a longer visual context
(`visual_len=15876`) and can cross 1x on easy high-acceptance samples:

```text
K=4096, gamma=13, sparse_attn_mode=gqa_sdpa, 128 frames / 64 tokens
samples: 3
token_equal: 3/3
speedup: 1.019x
acceptance_rate: 1.000
mean_accept_length: 12.000
ar_decoding_time: 4.814s
std_decoding_time: 4.723s
```

However, the same setting was not stable when expanded beyond the first three
questions:

```text
completed samples: 6
token_equal: 6/6
speedup: 0.877x
acceptance_rate: 0.861
mean_accept_length: 10.465
per-sample acceptance: 1.000, 1.000, 1.000, 0.756, 0.766, 0.640
```

The fourth sample (`001-2`) stayed lossless but slowed down at `K=4096,
gamma=13` (`0.778x`). Raising K to `8192` did not improve acceptance
(`0.756`), lowering gamma to `9` improved speed only to `0.895x`, and the
experimental adaptive gamma heuristic was worse (`0.797x`). This suggests the
remaining gap is not just K size or gamma tuning; it is draft-model agreement
under long visual contexts.

Sequential fallback on rejected parallel-verification rounds also did not help
that sample:

```text
K=4096, gamma=13, verify_fallback=sequential_on_reject
token_equal: true
speedup: 0.721x
acceptance_rate: 0.756
fallback_count: 2
fallback_accepted_extra: 0
```

So this low-acceptance case is not a false rejection from parallel numerical
drift; the sparse draft itself disagrees with dense decoding.

Pushing the frame count higher improves the easy high-acceptance upper bound,
but the local 24GB GPUs hit the visual tower memory limit before the paper-scale
A100 setting:

```text
K=4096, gamma=13, sparse_attn_mode=gqa_sdpa, 144 frames / 64 tokens
samples: 1
token_equal: 1/1
speedup: 1.100x
acceptance_rate: 1.000
mean_accept_length: 12.000
ar_decoding_time: 5.208s
std_decoding_time: 4.736s
visual_len: 17892
```

Expanding the 144-frame setting to ten Video-MME CoT questions separates the
strict lossless result from the faster default-verifier upper bound:

The closest local run to the paper's default implementation setting keeps
`K + text = 1024` and `gamma = 9`; with 144 frames this leaves roughly
`942-976` selected visual KV entries per sample after all text/nonvisual KV are
kept. It uses text-guided top-K sparse draft KV and parallel dense verification:

```text
K+text=1024, gamma=9, sparse_attn_mode=gqa_sdpa, default verifier, torch.inference_mode
144 frames / 64 tokens
samples: 5
token_equal: 5/5
speedup: 0.857x
acceptance_rate: 0.742
mean_accept_length: 6.227
retained_ratio: 0.057
paper_speed_threshold: 0.168
acceptance_minus_threshold: 0.574
ar_decoding_time: 5.142s
std_decoding_time: 6.327s
ar_inference_time: 58.975s
std_inference_time: 115.900s
visual_len: 17892
```

This paper-like setting confirms that the sparse draft acceptance is in the
expected speculative range rather than being forced toward 100%, but the
current PyTorch SDPA/GQA implementation does not turn the reduced KV traffic
into paper-level wall-clock speedup on the local RTX 3090 setup. Switching the
formal generation loops from `torch.no_grad()` to `torch.inference_mode()`
improved this same five-question baseline from `0.801x` to `0.857x` without
changing acceptance.

A synchronized single-sample profile of the same setting shows where the time
goes:

```text
K+text=1024, gamma=9, sparse_attn_mode=gqa_sdpa, profile_decode, torch.inference_mode
144 frames / 64 tokens, sample 006-3
token_equal: true
speedup: 1.111x
acceptance_rate: 0.951
ar_decoding_time: 5.174s
std_decoding_time: 4.658s
draft_time: 3.696s
verify_time: 0.598s
bonus_time: 0.363s
cache_adjust_time: 0.000s
retained_ratio: 0.057
paper_speed_threshold: 0.168
acceptance_minus_threshold: 0.783
```

Even with high acceptance on this sample, the sparse draft only barely beats
vanilla AR per token because the full model still runs once per drafted token.
The inference-mode change reduced draft overhead enough for this sample to
exceed 1x, but low-acceptance samples still drag the five-question mean below
1x.

A three-question gamma sweep after the inference-mode change shows that the
best fixed gamma can shift by sample, and that larger gamma is not always
better:

```text
K+text=1024, sparse_attn_mode=gqa_sdpa, default verifier, torch.inference_mode
144 frames / 64 tokens, samples: 3
gamma=5:  token_equal=3/3, speedup=0.922x, acceptance=0.866
gamma=7:  token_equal=3/3, speedup=0.890x, acceptance=0.793
gamma=9:  token_equal=3/3, speedup=0.904x, acceptance=0.791
gamma=11: token_equal=3/3, speedup=0.859x, acceptance=0.736
gamma=13: token_equal=3/3, speedup=0.834x, acceptance=0.704
```

On these three samples, `gamma=5` is the best fixed setting, while per-sample
winners differ (`9`, `5`, and `11`). This matches the paper's warning that
overlarge gamma can reduce speed when consecutive draft accuracy is insufficient.
Expanding `gamma=5` to five questions improves the fixed-gamma local baseline
but remains below 1x:

```text
K+text=1024, gamma=5, sparse_attn_mode=gqa_sdpa, default verifier, torch.inference_mode
144 frames / 64 tokens
samples: 5
token_equal: 5/5
speedup: 0.887x
acceptance_rate: 0.836
mean_accept_length: 3.979
retained_ratio: 0.057
paper_speed_threshold: 0.257
acceptance_minus_threshold: 0.579
ar_decoding_time: 5.138s
std_decoding_time: 5.869s
```

Compared with `gamma=9`, `gamma=5` raises acceptance and helps low-acceptance
samples, but it also increases decode rounds and loses speed on high-acceptance
samples. The next speed path is therefore adaptive gamma rather than another
fixed-gamma sweep.

An otherwise identical `repeat_sdpa` sparse-attention profile was slightly
slower than the default GQA SDPA path:

```text
K+text=1024, gamma=9, sparse_attn_mode=repeat_sdpa, profile_decode
144 frames / 64 tokens, sample 006-3
token_equal: true
speedup: 0.994x
acceptance_rate: 0.951
draft_time: 4.205s
verify_time: 0.600s
bonus_time: 0.414s
```

This keeps `gqa_sdpa` as the current best local sparse-attention mode. The
remaining gap to the paper is therefore more likely from the execution regime
(`batch size = 8`, A100 80GB, and optimized attention/runtime path) than from
the top-K selection rule itself.

A diagnostic hand-written grouped-matmul GQA attention path was also tested on
the same sample, but it was rejected because it changed the sparse draft
distribution and collapsed acceptance:

```text
K+text=1024, gamma=9, diagnostic grouped-matmul attention
144 frames / 64 tokens, sample 006-3
token_equal: true
speedup: 0.112x
acceptance_rate: 0.008
draft_time: 36.956s
```

This diagnostic path is not exposed in the benchmark CLI.

```text
K=8192, gamma=13, sparse_attn_mode=gqa_sdpa, verify_attn_backend=math
144 frames / 64 tokens
samples: 10
token_equal: 10/10
speedup: 0.986x
acceptance_rate: 0.972
mean_accept_length: 11.783
ar_decoding_time: 5.146s
std_decoding_time: 5.247s
ar_inference_time: 59.627s
std_inference_time: 116.473s
visual_len: 17892
```

This is the strictest current paper-style local result: exact-token lossless on
all ten local Video-MME CoT questions, but slightly below 1x decode speed. The
default SDPA verifier is faster on the same ten questions but produced one
false-accept mismatch:

```text
K=8192, gamma=13, sparse_attn_mode=gqa_sdpa, default verifier
samples: 10
token_equal: 9/10
speedup: 1.053x
acceptance_rate: 0.966
ar_decoding_time: 5.141s
std_decoding_time: 4.895s
```

Adding a low-confidence verifier guard plus sequential fallback only on
extremely low-acceptance rejected rounds gives the best strict lossless result
found so far:

```text
K=8192, gamma=13, sparse_attn_mode=gqa_sdpa
verify_margin_threshold=0.1
verify_fallback=sequential_on_low_accept
sequential_fallback_max_accept=1
144 frames / 64 tokens
samples: 10
token_equal: 10/10
speedup: 1.084x
acceptance_rate: 0.972
mean_accept_length: 11.783
verify_margin_reruns: 1.900 per sample
ar_decoding_time: 5.138s
std_decoding_time: 4.770s
```

This is the best strict ten-question local run so far. Narrowing the sequential
fallback from every rejected round to only `accept_len <= 1` preserves
`10/10` exactness while improving the guarded run from `1.005x` to `1.084x`.
The remaining gap to the paper's speedups is mostly verifier/draft overhead:
profiled stages are still dominated by sparse draft time (`3.789s`) and guarded
verify time (`0.729s`).

An in-process gamma sweep at the same `K=8192`, `144 frames / 64 tokens`, and
strict guard confirmed that increasing gamma past 13 does not improve the
lossless average on the local ten-question slice:

```text
K=8192, sparse_attn_mode=gqa_sdpa
verify_margin_threshold=0.1
verify_fallback=sequential_on_low_accept
sequential_fallback_max_accept=1
144 frames / 64 tokens
gamma=11: samples=10, token_equal=9/10,  speedup=1.003x, acceptance=0.936
gamma=13: samples=10, token_equal=10/10, speedup=1.061x, acceptance=0.972
gamma=15: samples=10, token_equal=10/10, speedup=1.060x, acceptance=0.957
gamma=17: samples=10, token_equal=9/10,  speedup=1.010x, acceptance=0.916
```

Within this same-process timing comparison, `gamma=13` is the fastest strict
setting by a tiny margin. `gamma=11` and `gamma=17` are rejected because they
produce token mismatches on the ten-question slice.

The matching K sweep with `gamma=13` shows that lowering K reduces retained KV
traffic but hurts acceptance enough to lose speed:

```text
K=4096: samples=10, token_equal=10/10, speedup=0.920x, acceptance=0.843
K=6144: samples=10, token_equal=10/10, speedup=0.969x, acceptance=0.930
K=8192: samples=10, token_equal=10/10, speedup=1.042x, acceptance=0.972
```

If exact token matching is treated as diagnostic rather than a hard gate for
Qwen2.5-VL numerical behavior, the closest paper-style speed-focused run is the
plain STD verifier path: sparse top-K draft, dense full-KV parallel
verification, and no extra margin or sequential fallback guard.

```text
K=8192, sparse_attn_mode=gqa_sdpa
verify_fallback=none
144 frames / 64 tokens
gamma=13: samples=10, token_equal=9/10, speedup=1.118x, acceptance=0.966
gamma=15: samples=10, token_equal=9/10, speedup=1.128x, acceptance=0.948
gamma=17: samples=10, token_equal=9/10, speedup=1.085x, acceptance=0.916
gamma=21: samples=10, token_equal=9/10, speedup=1.109x, acceptance=0.939
gamma=25: samples=10, token_equal=9/10, speedup=1.122x, acceptance=0.923
```

Under this speed-first metric, `gamma=15` is the current best ten-question local
setting. Larger gamma values reduce decode rounds, but the lower acceptance on
hard samples such as `006-2`, `007-1`, and `011-1` cancels the gain.

The earlier five-question default-verifier run remains the best exact-token
decode-speed result (`5/5`, `1.085x`), but the ten-question expansion shows why
the paper's lossless requirement needs verifier numerical care. Copying the
dense verifier prompt KV into the sparse cache reduced the three-question
setting's STD total inference time from `170.613s` to `115.045s` while
preserving exact token equality. Total inference time is still slower than
vanilla AR because this correctness-first implementation still performs a
separate attention-selection prefill in addition to the normal dense prefill.

Default-resolution `160`, `192`, and `256` frame attempts OOMed during
Qwen2.5-VL visual-tower prefill on the 4x RTX 3090 24GB machine. A
`256`-frame low-resolution smoke run (`max_pixels=112896`) fit, but only had
`visual_len=18288` and reached `0.993x` on one question, so it is not a useful
speed path. This makes `144` default-resolution frames the highest confirmed
useful point so far for the current local setup and reinforces that the paper's
A100 80GB environment is a meaningful part of the reported speed regime.

Video-MME short, one CoT question, `32 frames / 64 tokens`, mid-K sweep:

```text
K=2560, gamma=13: speedup=0.958x, acceptance=0.967, token_equal=true
K=2560, gamma=21: speedup=0.783x, acceptance=0.769, token_equal=true
K=2560, gamma=31: speedup=0.934x, acceptance=0.910, token_equal=true
K=3072, gamma=13: speedup=0.874x, acceptance=0.894, token_equal=true
K=3072, gamma=21: speedup=0.674x, acceptance=0.659, token_equal=true
K=3072, gamma=31: speedup=0.703x, acceptance=0.681, token_equal=true
K=3584, gamma=13: speedup=0.873x, acceptance=0.879, token_equal=true
K=3584, gamma=21: speedup=0.873x, acceptance=0.857, token_equal=true
K=3584, gamma=31: speedup=0.701x, acceptance=0.670, token_equal=true
```

This sweep did not find a genuinely sparse setting above 1x. The best sparse
point was `K=2560, gamma=13` at `0.958x`; larger gamma values often lost speed
because lower acceptance outweighed fewer dense verification rounds.

Video-MME short, one CoT question, `128 frames / 64 tokens / gamma=9` sweep:

```text
K=1024: speedup=0.858x, acceptance=0.778, token_equal=true
K=2048: speedup=0.857x, acceptance=0.778, token_equal=true
K=4096: speedup=0.921x, acceptance=0.864, token_equal=true
```

The current version is a correctness-first reproduction. It does not yet match
the STD paper speedups because the local validation data is not the paper
benchmark, Qwen2.5-VL differs from the paper's Qwen2-VL setting, and the current
implementation still uses regular PyTorch SDPA rather than a fused top-K
attention kernel.

## Useful Commands

Summarize one or more metric files:

```bash
conda run -n specvlm python scripts/summarize_metrics.py results/std_qwen2_5_vl_7b/*.jsonl
conda run -n specvlm python scripts/summarize_metrics.py --group-by gamma results/std_qwen2_5_vl_7b/videomme_short_10questions_cot_32f_64tok_kplus1024_gamma_sweep.jsonl
```

Resume Video-MME partial data download:

```bash
conda run -n specvlm python scripts/prepare_std_datasets.py --dataset Video-MME --videomme-chunks 01
conda run -n specvlm python scripts/extract_videomme_chunks.py --chunks 01
```

Check how many Video-MME rows have local videos:

```bash
conda run -n specvlm python scripts/check_videomme_assets.py
```

Download Video-MME videos on demand from YouTube metadata URLs:

```bash
conda run -n specvlm python scripts/download_videomme_youtube.py --duration short --limit 10
```

Run the strict ten-question lossless Video-MME short setting:

```bash
SPECVLM_MAX_CACHE_LEN=40960 conda run -n specvlm python scripts/benchmark_std.py \
  --dataset Video-MME \
  --data-path /home/mcy/projects/Std/datasets/Video-MME \
  --video-root /home/mcy/projects/Std/datasets/Video-MME/videos \
  --eval-num 10 \
  --frame-num 144 \
  --max-new-tokens 64 \
  --gamma 13 \
  --k 8192 \
  --verify-mode parallel \
  --verify-attn-backend math \
  --prompt-style cot \
  --sparse-attn-mode triton_gqa \
  --gpu-ids 0,1,2,3 \
  --output results/std_qwen2_5_vl_7b/videomme_short_10questions_cot_144f_64tok_k8192_g13_gqa_mathverify_copy_sparse.jsonl
```

Run the current best strict ten-question Video-MME short setting:

```bash
SPECVLM_MAX_CACHE_LEN=40960 conda run -n specvlm python scripts/benchmark_std.py \
  --dataset Video-MME \
  --data-path /home/mcy/projects/Std/datasets/Video-MME \
  --video-root /home/mcy/projects/Std/datasets/Video-MME/videos \
  --eval-num 10 \
  --frame-num 144 \
  --max-new-tokens 64 \
  --gamma 13 \
  --k 8192 \
  --verify-margin-threshold 0.1 \
  --verify-fallback sequential_on_low_accept \
  --sequential-fallback-max-accept 1 \
  --profile-decode \
  --verify-mode parallel \
  --prompt-style cot \
  --sparse-attn-mode gqa_sdpa \
  --gpu-ids 0,1,2,3 \
  --output results/std_qwen2_5_vl_7b/videomme_short_10questions_cot_144f_64tok_k8192_g13_margin0p1_seqlow1_profile.jsonl
```

Run the current best 30-question Video-MME short setting:

```bash
SPECVLM_MAX_CACHE_LEN=40960 conda run -n specvlm python scripts/benchmark_std.py \
  --dataset Video-MME \
  --data-path /home/mcy/projects/Std/datasets/Video-MME \
  --video-root /home/mcy/projects/Std/datasets/Video-MME/videos \
  --eval-num 30 \
  --frame-num 32 \
  --max-new-tokens 64 \
  --gamma 13 \
  --k 2560 \
  --verify-mode parallel \
  --prompt-style cot \
  --sparse-attn-mode gqa_sdpa \
  --gpu-ids 0,1,2,3 \
  --output results/std_qwen2_5_vl_7b/videomme_short_30questions_cot_32f_64tok_k2560_g13_gqa_sdpa.jsonl
```

Run an in-process sweep that loads the model once and reuses the AR baseline:

```bash
SPECVLM_MAX_CACHE_LEN=40960 conda run -n specvlm python scripts/sweep_std_inprocess.py \
  --dataset Video-MME \
  --data-path /home/mcy/projects/Std/datasets/Video-MME \
  --video-root /home/mcy/projects/Std/datasets/Video-MME/videos \
  --prompt-style cot \
  --ks '' \
  --target-k-plus-text 1024 \
  --gammas 9,11,13 \
  --sparse-attn-mode gqa_sdpa \
  --frame-num 32 \
  --max-new-tokens 64 \
  --eval-num 1
```

Run a local K/gamma sweep:

```bash
SPECVLM_MAX_CACHE_LEN=40960 conda run -n specvlm python scripts/sweep_std.py \
  --ks 2048,4096 \
  --gammas 5,9 \
  --frame-num 128 \
  --max-new-tokens 64 \
  --eval-num 1
```

## A100 服务器迁移指南

### 代码迁移

需要迁移两个目录（纯代码，总计约 1.2 MB）：

```bash
# 在本地打包
tar czf std_project.tar.gz -C /home/mcy/projects Std/src Std/scripts Std/datasets
tar czf specvlm.tar.gz -C /home/mcy/projects SpecVLM --exclude=datasets --exclude=results --exclude=reproduction_results --exclude=.git --exclude=assets

# 传输到 A100 服务器
scp std_project.tar.gz specvlm.tar.gz user@a100-server:/path/to/projects/
```

在 A100 上解压，保证以下目录结构：

```text
/path/to/projects/
├── Std/
│   ├── src/std_repro/std_qwen25vl.py
│   ├── src/std_repro/triton_attention.py
│   └── scripts/benchmark_std.py
└── SpecVLM/
    ├── models/modeling_qwen2_5_vl.py
    ├── kv_cache/kv_cache.py
    └── utils/utils.py
```

```bash
mkdir -p /path/to/projects
cd /path/to/projects
tar xzf std_project.tar.gz
tar xzf specvlm.tar.gz
```

### 第一步：安装 Conda 环境

```bash
conda create -n specvlm python=3.10 -y
conda activate specvlm

# PyTorch（A100 建议 CUDA 12.1+）
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121

# Triton
pip install triton==3.2.0

# 核心依赖
pip install transformers==4.48.0
pip install "datasets>=2.14,<3"
pip install accelerate
pip install "numpy<2.0"
pip install qwen-vl-utils==0.0.10
pip install av==14.0.0

# 辅助工具
pip install huggingface_hub
```

验证环境：

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
python -c "import triton; print('triton OK')"
python -c "import av; print('av OK')"
```

### 第二步：下载模型

```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir /path/to/projects/models/Qwen2.5-VL-7B-Instruct
```

约 16 GB，A100 服务器带宽充足，几分钟即可完成。

### 第三步：下载 Video-MME 数据集

```bash
cd /path/to/projects/Std

# 下载 metadata + 第一批视频压缩包
conda run -n specvlm python scripts/prepare_std_datasets.py --dataset Video-MME

# 解压视频
conda run -n specvlm python scripts/extract_videomme_chunks.py --chunks 01

# 从 YouTube 元数据补充下载更多 short 视频（用于测试）
conda run -n specvlm python scripts/download_videomme_youtube.py --duration short --limit 20
```

### 第四步：运行复现实验

#### 4a. Smoke test（验证链路正确）

```bash
cd /path/to/projects/Std

SPECVLM_MAX_CACHE_LEN=81920 conda run -n specvlm python scripts/benchmark_std.py \
  --model-path /path/to/projects/models/Qwen2.5-VL-7B-Instruct \
  --dataset Video-MME \
  --data-path /path/to/projects/Std/datasets/Video-MME \
  --video-root /path/to/projects/Std/datasets/Video-MME/videos \
  --eval-num 3 \
  --frame-num 32 \
  --max-new-tokens 64 \
  --gamma 9 \
  --target-k-plus-text 1024 \
  --prompt-style cot \
  --gpu-ids 0 \
  --output results/std_qwen2_5_vl_7b/smoke_32f.jsonl
```

#### 4b. 论文级配置 —— 高帧数 + 稀疏 K

A100 80GB 的核心优势是可以跑 256 帧以上，这是 RTX 3090 做不到的：
论文表 1 使用 batch size 8；当前 benchmark 的 STD 解码状态仍按单样本
维护，因此下面命令复现的是论文的 `K+text=1024 / gamma=9 / CoT`
超参，而不是表 1 的 batched throughput setting。

```bash
# 256 帧，论文的核心 setting
SPECVLM_MAX_CACHE_LEN=163840 conda run -n specvlm python scripts/benchmark_std.py \
  --model-path /path/to/projects/models/Qwen2.5-VL-7B-Instruct \
  --dataset Video-MME \
  --data-path /path/to/projects/Std/datasets/Video-MME \
  --video-root /path/to/projects/Std/datasets/Video-MME/videos \
  --eval-num 30 \
  --frame-num 256 \
  --max-new-tokens 64 \
  --gamma 9 \
  --target-k-plus-text 1024 \
  --prompt-style cot \
  --sparse-attn-mode triton_gqa \
  --gpu-ids 0 \
  --output results/std_qwen2_5_vl_7b/videomme_short_256f_kplus1024_g9_triton.jsonl
```

#### 4c. 参数联合 sweep

```bash
SPECVLM_MAX_CACHE_LEN=163840 conda run -n specvlm python scripts/sweep_std_inprocess.py \
  --dataset Video-MME \
  --data-path /path/to/projects/Std/datasets/Video-MME \
  --video-root /path/to/projects/Std/datasets/Video-MME/videos \
  --prompt-style cot \
  --ks '2048,4096,8192' \
  --gammas '9,11,13,15,17' \
  --frame-num 256 \
  --max-new-tokens 64 \
  --eval-num 10 \
  --sparse-attn-mode triton_gqa \
  --gpu-ids 0 \
  --output results/std_qwen2_5_vl_7b/sweep_256f.jsonl
```

#### 4d. 查看结果

```bash
conda run -n specvlm python scripts/summarize_metrics.py results/std_qwen2_5_vl_7b/*.jsonl
```

### 参数对照表

| 参数 | RTX 3090 最佳 | A100 80GB 建议 |
|---|---|---|
| `--frame-num` | 144（显存上限） | **256 起步**，可尝试 384、512 |
| `--target-k-plus-text` | 1024 | 1024（稀疏）或 4096（高接受率） |
| `--k` | 8192 | 4096–8192 sweep |
| `--gamma` | 13 | **9 for paper config**, 9–17 for speed sweep |
| `--gpu-ids` | 0,1,2,3（4 卡） | 0（**单卡即可**） |
| `SPECVLM_MAX_CACHE_LEN` | 40960 | **163840** 或更高（对应 256+ 帧的 KV cache 长度） |

### 关键环境变量

- `SPECVLM_MAX_CACHE_LEN`：KV cache 预分配长度。高帧数时必须调大，否则 OOM 或 cache 截断。256 帧建议 `163840`，384 帧建议 `245760`。
- `CUDA_VISIBLE_DEVICES`：由 `--gpu-ids` 自动设置。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：脚本自动设置，避免显存碎片。

### 3090 vs A100 结果预期

在 A100 上预期能获得更好的加速比，原因：

1. **计算带宽更高**：sparse draft 的 single-token forward 在 A100 上更快，压低 draft_time 占比（3090 上 draft 占 86.5% 的解码时间）
2. **可以跑更高帧数**：visual KV 越多，稀疏选择的收益越大。256 帧下 visual_len ≈ 32000，K+text=1024 仅保留约 3% 的 visual KV
3. **单卡即够**：80GB 显存远大于 4×3090 24GB 的实际可用显存

如果 A100 上 256 帧 + K+text=1024 + `--sparse-attn-mode triton_gqa`
仍然低于 1x 加速，瓶颈就不只是 PyTorch SDPA，下一步需要做真正的
batch=8 decode 路径和更完整的 top-K/cache kernel 融合。
