# W1 — decode-only optimisation knobs (2026-09-22)

> **Scope.** Decode only. Prefill (61 % of `inference`) is explicitly out of scope.
> Everything here attacks the two free variables that are actually left:
> **how many forwards per accepted token** and **how many bytes per forward**.
>
> Implementation: `src/std_repro/std_qwen25vl.py` (helpers + `std_generate_qwen25vl`),
> `src/std_repro/dynamic_std_qwen25vl.py`, `scripts/benchmark_a100_dynamic.py`.
> Tests: `tests/test_w1_decode_knobs.py` (40 CPU regressions).

## 0. Why these three, and why not KV selection

The decode budget measured on `local512_A` (10 samples, 128 frames, 512 tokens):

| component | time | share of decode | paid |
|---|---:|---:|---|
| sparse draft forwards | 146.8 s | 71.8 % | 9 × per round |
| bonus sparse forward | 16.3 s | 8.0 % | 1 × per round |
| dense verification | 41.3 s | 20.2 % | 1 × per round |
| selection / cache copy | 0.4 s | 0.2 % | once |

So `c_s = 28.5 ms` per sparse step is **80 % of the round** and is paid ten times,
while the whole sparse-attention saving is **0.83 ms of that 28.5 ms (2.9 %)**.
Continuing to tune *which* KV are selected cannot move decode; the levers are the
per-step cost and the number of steps. W1 adds one knob for each.

## 1. The three knobs

| flag | what it changes | why it is output-preserving |
|---|---|---|
| `--compile-draft` | `torch.compile(mode="reduce-overhead")` around the single-token sparse step (CUDA graphs) | the compiled region only produces *draft proposals*; accepted tokens are still the dense verifier's argmax |
| `--draft-layers N` | the sparse draft runs only the first `N` decoder blocks, then the model's own final norm + LM head (early exit) | same argument — a weaker draft can only lower the acceptance rate |
| `--fair-dense-verify` | the **batched verification** uses `enable_gqa` instead of `repeat_kv` | ⚠️ **not** output-preserving: it changes verifier numerics (kernel / reduction order). Read the `Exact` column |

`--fair-dense-verify` has two modes:

* `mask` (default) — keep the model's explicit 4D additive mask; only the 7× K/V
  materialisation disappears. **Semantics identical to stock.**
* `causal` — additionally drop the mask and ask SDPA for `is_causal=True`, which is
  what lets a fused kernel run at all. **Only equivalent if SDPA anchors the causal
  mask at the bottom-right for `q_len < kv_len`.** PyTorch's math backend anchors it
  top-left, which would silently attend to the wrong keys, so this mode is gated
  behind `scripts/analysis/verify_attention_variants.py`.

### Deliberately *not* included

* **AR is untouched.** `--fair-dense-attention` (E3b) is the opposite knob: it patches
  `q_len == 1`, i.e. the baseline, and is what makes the speedup collapse to 0.78×.
  `--fair-dense-verify` patches `q_len > 1` only, so the AR decode, both prefills and
  the selection branch keep their stock numerics. The patch is also inert until a
  verify call marks it active, so prefills cannot be affected even by accident.
* The sparse *draft* kernel (`--sparse-attn-mode`) is unchanged. E2 already showed
  removing `enable_gqa` there costs only +0.93 ms/step.

## 2. How the three were wired

* `_CompiledSparseStep` — preallocates the token/position buffers that CUDA-graph
  capture needs, and is **fail-soft**: a failing compile snapshots and restores the
  sparse cache write cursors, disables itself permanently, and returns `None` so the
  caller falls back to eager. (`KVCache.cat` writes at `current_length` and only then
  advances it, so restoring the cursors is exactly the rollback the decode loop
  already performs.)
* `_draft_layer_window` — swaps `model.model.layers` for a slice inside a context
  manager. The final norm and LM head still run, so the early-exit head is the
  model's own output head: no new parameters, no training.
* `sparse_draft_context` — one context that both enables the controller and applies
  the window, used by `_draft_tokens` and `_append_sparse_tokens`.
* `install_fair_verify` / `uninstall_fair_verify` / `fair_verify_active` — install the
  patched forward once per trial and toggle it around the three parallel-verify call
  sites. `_is_exactly_causal` recognises the mask `_prepare_decoder_attention_mask`
  builds; anything else (padding, tree mask, other shape) falls back to stock and is
  counted in the audit.
* `_call_trace_hook` — the dynamic-routing collectors
  (`RuntimeVerificationCollector`, `VerificationCollectorV2/V3`) only *attach* a
  `_std_trace_hook` attribute; the call site lives inside the vendored
  `Qwen2_5_VLSdpaAttention.forward`. Both replacement attention paths now reproduce
  that call at the same point (post-RoPE, post-cache, pre-`repeat_kv`). Without it,
  `--fair-dense-verify` would have made the dynamic runs silently collect nothing and
  turn their selection refresh into a no-op.

The old `use_compile` plumbing was **dead code**: `_build_compiled_sparse_step` called
the compiled step with two arguments while the closure took three, and no caller ever
passed `use_compile=True` (`grep use_compile scripts/` was empty). Both are fixed, and
the draft loop now actually routes through the compiled path.

## 3. Measured results (2026-09-22, A6000 GPU 1)

### 3.1 `--fair-dense-verify` is **counterproductive** — measured, do not use

`scripts/analysis/verify_attention_variants.py --gpu 1 --q-len 10 --kv-len 16460`
(raw: `results/w1_decode/verify_variants.md` / `.json`), 28 steps per timing
sample, 20 reps × 4 interleaved rounds, fp16:

| variant | equivalent to the math reference? | ms / verify step | vs stock |
|---|---|---:|---:|
| `stock_repeat_mask` (what we pay today) | ✅ 1.9e-06 | **43.36** | **1.000×** |
| `gqa_mask` (`--fair-dense-verify-mode mask`) | ✅ **0.0** (bit-equal) | 99.34 | 2.291× |
| `gqa_boolmask` | ✅ **0.0** | 98.09 | 2.262× |
| `gqa_causal` (`--fair-dense-verify-mode causal`) | ❌ **0.325** | 98.05 | 2.261× |

Two conclusions, both the opposite of the pre-run estimate:

1. **Removing `repeat_kv` makes the verification 2.3× slower, not faster.** The
   verification needs a causal mask, and *any* mask keeps SDPA off the fused
   kernel; `enable_gqa` then routes to the math backend. This also explains E1's
   `dense_gqa_mask` = 113 ms being the worst variant. At `q_len=10` the bytes saved
   by not materialising 6.6 GB are far smaller than the kernel that is lost.
2. **`causal` is not equivalent and must not be used.** The 0.325 deviation proves
   SDPA anchored `is_causal` somewhere other than the bottom-right for
   `q_len < kv_len` on this torch/GPU pair. The gate caught a configuration that
   would have silently attended to the wrong keys — the mode stays in the code as
   a recorded negative, not as a recommendation.

The flag is retained for the record and its `--help` now says so.

### 3.2 `--compile-draft` works but buys **nothing** — the step has no launch overhead to remove

`results/w1_decode/smoke_compile_layers16.jsonl` (1 sample, 128 tokens):

* `draft_compiled: true` with an **empty** `draft_compile_note` — `torch.compile`
  traced the custom `KVCache` object without a graph break and captured the CUDA
  graph. The soft-fallback was not needed.
* The compile cost lands entirely in the warmup trial (`draft_time` 52.3 s for 12
  rounds) and is gone by the measure trial (17.8 s for 121 rounds).

End-to-end A/B, 3 samples, 512 tokens, same idle GPU and sample list
(`results/w1_decode/w1_decode_20260922/`):

| run | static/AR decode | `c_s` (ms per sparse step) | accept | accept len | rounds | exact |
|---|---:|---:|---:|---:|---:|---:|
| `W0_base` | **1.855×** | **26.95** | 0.9245 | 8.278 | 166 | 2/3 |
| `W1_compile` | 1.849× | **27.04** | 0.9245 | 8.278 | 166 | 2/3 |

`draft_compiled: true` in **3/3** static measure trials. Acceptance, accepted
length, round count and exactness are byte-identical — exactly as the
output-preserving argument predicts. And `c_s` moved by **+0.3 %**, i.e. nothing.

**The pre-run estimate ("~7.6 ms of the 28.5 ms step is kernel-launch / Python
overhead") was wrong.** A single-token 7B forward at batch 1 is ~100 % DRAM-bound:
~27 ms of the 28.5 ms step is streaming 15.45 GiB of weights, and the achievable
fraction of peak bandwidth is not something a CUDA graph improves. There is no
launch overhead worth capturing.

### 3.3 `--draft-layers` reduces `c_s` exactly as predicted — and loses anyway

Early exit does shrink the weight read **precisely** in proportion to the layer
fraction. It also destroys acceptance, because the model was never trained to be
read out from an intermediate block:

| run | layers | `c_s` (ms) | predicted `c_s` (26.95 × N/28) | accept len `n` | rounds | static/AR decode |
|---|---:|---:|---:|---:|---:|---:|
| `W0_base` | 28 | **26.95** | — | 9.253 | 166 | **1.855×** |
| `W2_layers26` | **26** | **24.70** | 25.02 | **1.664** | **923** | **0.358×** |
| smoke | 16 | — | 15.40 | 0.058 | 121 / 128 tok | — |

`c_s` matches the layer-fraction prediction to **1.3 %**, so the rest of the step
has no hidden constant: **the sparse draft step is a linear function of the bytes
of weights it reads.** Acceptance, however, falls off a cliff — dropping just two
of twenty-eight blocks takes the accepted length from 9.25 to 1.66 and makes
static **2.8× slower than AR**.

### 3.4 Verdict on all three knobs

| knob | runs? | output preserved? | decode effect | usable? |
|---|---|---|---|---|
| `--compile-draft` | ✅ traced + CUDA graph | ✅ identical | **±0.3 %** (`c_s` 26.95 → 27.04) | ❌ no headroom to capture |
| `--draft-layers N` | ✅ | ✅ identical | `c_s` −(N/28) but `n` −82 % at N=26 | ❌ acceptance collapse |
| `--fair-dense-verify` | ✅ | ⚠️ `mask` bit-equal, `causal` **wrong** | **2.3× slower** | ❌ counterproductive |

**All three fail, and together they prove the diagnosis rather than leaving it a
hypothesis:**

1. `compile` moves nothing → there is **no launch / Python overhead** in the step;
   the residual over peak bandwidth is DRAM efficiency, not removable work.
2. `draft_layers` moves `c_s` by exactly the byte fraction → the step **is** the
   weight read, with no hidden constant.
3. `fair-dense-verify` is slower → the verify's `repeat_kv` is not reclaimable
   either, because the mask that verification needs already costs the fused kernel.

So every knob that leaves the weight bytes alone is worthless. The single
remaining lever is the one that changes the bytes.

## 4. What to run

### 4.1 Gate `causal` on the GPU first (5 minutes)

```bash
python scripts/analysis/verify_attention_variants.py --gpu 3 \
    --q-len 10 --kv-len 16460 \
    --output results/w1_decode/verify_variants.md
```

It reports, for the real verification shapes, each variant's **max absolute deviation
from a math-backend reference** and its **ms per step**. It refuses to recommend
`causal` unless the deviation is within tolerance. Kernel behaviour here cannot be
predicted offline — this is the measurement that decides it.

### 4.2 The matrix (needs an idle A6000)

```bash
GPU=3 LIMIT=10 bash scripts/run_w1_matrix_local.sh w1_decode_20260923
```

Runs `W0_base` plus one-factor-at-a-time runs for each knob and a combined run, waits
for a genuinely idle GPU (`timing_valid=true`), and summarizes each run into
`results/w1_decode/<TAG>/`. `CONFIGS` env var overrides the config list
(newline-separated `name|collector|interval|min_change|extra args`, so a config may
carry several flags).

### 4.3 What each column is supposed to move

| config | expected decode-ratio effect |
|---|---|
| `W0_base` | reference; must reproduce ≈1.71× — if it does not, nothing below is comparable |
| `W1_compile` | −10…−15 % on `draft_time`; acceptance and `Exact` must be **unchanged** |
| `W2…W5_layers{24,16,12,8}` | `draft_time` scales with N/28; **acceptance is the measurement** — watch `mean_accept_length` and `decode_rounds` |
| `W6_fairmask` | `verify_time` down; `Exact` may move — this is the numeric-risk cell |
| `W7_faircausal` | same as W6 but larger, **only if 3.1 passed** |
| `W8_combo` | the best combination of the above |

### 4.4 How to read the result

```
decode ms/token = (S · c_s + c_v) / n         S = 10 sparse forwards per round
decode ratio    = c_d · n / (S · c_s + c_v)   c_d = 68.0 ms, n ≈ 9.25
```

and, from §3.3, `c_s` is a linear function of the bytes of weights the draft reads:

```
c_s ≈ 1.74 ms per GiB of draft weights
    =  26.95 ms for the 15.45 GiB fp16 stack (28 layers)
```

so a −10 % `c_s` is worth **+~9 %** on the ratio — which is why `compile`
(±0 %) and `draft_layers` (`c_s` −8 %, `n` −82 %) both failed.

## 5. Where the leverage actually is: quantise the **draft only**

The three negative results point at exactly one remaining lever. `c_s` is bought
by the byte, and only two things reduce the bytes: fewer layers (proven fatal —
the model was never trained to be read out early) or **fewer bytes per weight**.

The reason this is worth stating carefully: **quantising the draft alone does not
break exactness.** The same argument that made `--compile-draft` and
`--draft-layers` output-preserving applies — accepted tokens are the fp16 dense
verifier's own argmax, and a worse draft can only lower the acceptance rate. An
earlier note in this project ("quantisation breaks bit-exactness") is true only
when the **verifier** is quantised; a draft-only quantisation spends no
correctness budget at all.

Predicted from the measured 1.74 ms/GiB and the measured `c_d`, `c_v`:

| draft precision | draft weights | `c_s` (ms) | predicted static/AR decode |
|---|---:|---:|---:|
| fp16 (today) | 15.45 GiB | 26.95 | **1.855×** (measured) |
| int8 | 7.7 GiB | ≈ 14 | **≈ 3.0×** |
| int4 | 3.9 GiB | ≈ 8 | **≈ 4.2×** |
| free draft | 0 | 0 | 9.1× (asymptote: `c_d·n/c_v`) |

This escapes the `c = c_d/c_s = 2.5×` ceiling precisely because the draft stops
reading the same bytes as the target. Two caveats that must be measured, not
assumed: the quantised draft will propose worse tokens (`n` falls), and `c_v`
(69 ms/round) becomes the dominant term past int4, so the curve flattens.

## 6. What W1 cannot fix

* **The `c = 1/ρ = 2.40×` ceiling.** The draft and the target are the same 7B model,
  so both pay the ~22 ms weight read per step. W1 lowers `c_s`, which raises the ratio
  toward that ceiling; it cannot raise the ceiling itself.
* **A fair dense baseline.** If the verification is compared against an AR path that
  also uses `enable_gqa`, the honest ratio is ≈1.05×. W1 improves STD's own numbers
  against the stock baseline; it does not change that attribution.
* **Honest sparse benefit.** Still 1.10 ms/step. Nothing in W1 touches it.

## 7. Verification status

* 52 CPU regressions across `tests/test_w1_decode_knobs.py` (43) and
  `tests/test_compare_w1_configs.py` (9); the pre-existing 324-test suite is
  unchanged (**376 passed** total).
* **The GPU runs were executed** on 2026-09-22 from 20:15 on a genuinely idle
  A6000 (physical GPU 1, `timing_valid=true` for every run):
  `results/w1_decode/verify_variants.{md,json}`,
  `results/w1_decode/smoke_compile_layers16.jsonl`,
  `results/w1_decode/w1_decode_20260922/` (W0/W1/W2 + `comparison.md`).
* `torch.compile` **did** trace the custom `KVCache` without a graph break
  (`draft_compiled: true`, empty note, 3/3 trials) — the fail-soft path was
  never needed.
* Still unmeasured: whether a **draft-only quantised** draft keeps enough
  acceptance to realise the §5 prediction. That is the W4 experiment, not W1.
* `W3_layers24` was cancelled after `W2_layers26` already answered the question:
  dropping two blocks collapses the accepted length from 9.25 to 1.66, so a
  four-block cut has nothing left to show.
