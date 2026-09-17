# Dynamic STD Ablation Rerun (A100 / Video-MME)

**Goal:** determine whether the verifier-guided dynamic STD acceptance advantage
is real, or whether the 2026-09-09 A100 negative result is an implementation
regression. Produce a paired, equal-`S_0`, enlarged-sample comparison that either
restores the advantage or retires the dynamic line as a documented negative
result.

**Spec / context:** `PROGRESS.md` §14 (negative result), §12 (fallback risk),
§16.2 (this plan). Baseline harness: `scripts/benchmark_a100_dynamic.py` and
`scripts/summarize_a100_dynamic.py`.

**Runner:** `scripts/run_a100_ablation.sh`.

## Global constraints

- Use **physical GPU 1 only**. GPU 0 hosts an unrelated vLLM workload and must
  never be touched or queried for work. The benchmark refuses to start unless
  the selected GPU is idle.
- Every method must emit the fixed requested token budget and be compared to AR
  on paired trials; `results/` is gitignored, so raw JSONL stays in
  `STD_assets/results/`.
- Do not claim wall-clock acceleration from a run whose correctness gate failed.
- Output files are created with `open(..., "x")`; never reuse an output path.

## Why `limit=3` is not decisive

The 2026-09-09 runs used 3 samples. Static acceptance was identical across every
run (0.823; 0.857 at 256 tokens), which proves the harness is deterministic but
leaves no variance estimate. The observed dynamic deficit (0.731–0.777) could be
a real algorithm effect or a small-sample artefact, and several implementation
axes were changed at once, so they cannot be attributed.

## Hypotheses

| id | hypothesis | axis to vary |
|---|---|---|
| H1 | `incremental` refresh slot-order regression depresses accept | `--refresh-mode` |
| H2 | `attention_free` bootstrap produces a worse `S_0` than static | `--dynamic-bootstrap` |
| H3 | `two`-query collector picks worse query positions than `three` | `--dynamic-query-mode` |
| H4 | `TopK(A_{t-1})` has no signal on Video-MME/CoT/128-frame inputs | all axes stay at the static-faithful control |
| H5 | 3 samples is too few to decide | `--limit` |
| H6 | the 0.05 hysteresis suppresses useful updates | `--min-selection-change-ratio` |
| F  | `sequential_on_low_margin` is what makes trials exact | `--verify-fallback` |

## Controls that make the comparison interpretable

1. **Equal `S_0`.** `--dynamic-bootstrap attention` makes dynamic start from the
   exact same static Top-K selection as the `static` method, so any accept
   difference is attributable to the *update rule*, not the initial mask.
   `attention_free` deliberately breaks this; treat it as a different algorithm.
2. **`static` is inside every run.** Each benchmark emits `ar`, `static`, and
   `dynamic_v2` on identical tensors, so static is a per-run control and its
   run-to-run stability is itself a sanity check.
3. **One axis at a time.** The control run R1 keeps every dynamic axis at its
   most static-faithful value; R2–R6 each change exactly one thing.
4. **Matched trials.** `summarize_a100_dynamic.py` pairs by sample and repeat and
   refuses incomplete runs, so partial output cannot be misread.

## Stage 0 — reproduce and sanity-check (cheap)

`limit=3`, `repeats=2`, 128 frames, 128 tokens.

- R1 (control) and R6 (`--verify-fallback none`).
- **Gate S0-a:** R1 must reproduce the recorded negative direction
  (static accept > dynamic accept) within noise.
- **Gate S0-b:** the fallback A/B must show whether exactness survives
  `--verify-fallback none`. If it does, drop the fallback for stage 1 to remove
  the §12 confound; if it does not, keep it and record that exactness depends on
  an unverified fallback.

## Stage 1 — decisive single-axis ablations

`limit=10`, `repeats=2`, 128 frames, 128 tokens.

| run | refresh | query | bootstrap | interval | min-change | fallback |
|---|---|---|---|---|---|---|
| R1 | full | three | attention | 1 | 0.05 | sequential_on_low_margin |
| R2 | incremental | three | attention | 1 | 0.05 | sequential_on_low_margin |
| R3 | full | three | attention_free | 1 | 0.05 | sequential_on_low_margin |
| R4 | full | two | attention | 1 | 0.05 | sequential_on_low_margin |
| R5 | full | three | attention | 1 | 0.00 | sequential_on_low_margin |
| R6 | full | three | attention | 1 | 0.05 | none |

Interpretation:

- If **R1 or R5 ≥ static** but R2 < static, H1 is confirmed → keep full rebuild.
- If **R3 ≥ static** but R1 < static, H2 is confirmed → switch bootstrap.
- If **R4 ≥ static** but R1 < static, H3 is confirmed → switch to two-query.
- If **every R < static**, H4 is the surviving explanation → retire dynamic as a
  negative result and record it in `PROGRESS.md`.
- If static itself drifts across runs beyond the deterministic baseline, stop and
  investigate the harness before interpreting anything.

## Stage 2 — conditional

Only if some stage-1 run shows `dynamic accept ≥ static accept`:

- switch `--dynamic-collector v3` (fused, no CPU sync) to test whether the
  accept gain can survive into wall-clock;
- switch `--dynamic-collector v1` (full-query control) to confirm the
  two-query reduction is not the cause;
- raise `--limit` to 30 and re-run the winning configuration plus R1.

## Acceptance criteria

A dynamic configuration is declared beneficial **only if all hold**:

1. correctness: every measured trial is positionally exact against AR;
2. acceptance: matched-trial `acceptance_rate` for dynamic ≥ static;
3. wall-clock: synchronized `decoding_time` total for dynamic < static;
4. stability: static's own totals agree with its baseline across runs.

Otherwise the run is reported as a negative ablation that names the measured
bottleneck, per the existing harness contract.

## Concrete commands

```bash
cd /public/home/xlwang/mcy/Project/STD

# Stage 0 (repeat the recorded configuration + fallback A/B)
bash scripts/run_a100_ablation.sh stage0

# Stage 1 (decisive, 6 runs x 10 samples)
bash scripts/run_a100_ablation.sh stage1
```

Override defaults by exporting variables, e.g. a smaller exploratory smoke:

```bash
LIMIT=3 TOKENS=64 REPEATS=1 bash scripts/run_a100_ablation.sh stage1
```

A single run without the driver:

```bash
/public/home/xlwang/mcy/conda_envs/specvlm/bin/python scripts/benchmark_a100_dynamic.py \
  --gpu 1 \
  --model-path  /public/home/xlwang/mcy/STD_assets/models/Qwen2.5-VL-7B-Instruct \
  --data-path   /public/home/xlwang/mcy/STD_assets/datasets/Video-MME \
  --video-root  /public/home/xlwang/mcy/STD_assets/datasets/Video-MME/videos \
  --output      /public/home/xlwang/mcy/STD_assets/results/R1_full_three_att.jsonl \
  --frame-num 128 --max-new-tokens 128 --limit 10 --repeats 2 \
  --gamma 9 --k-plus-text 1024 --profile-components \
  --dynamic-collector v2 --refresh-mode full \
  --dynamic-query-mode three --dynamic-bootstrap attention \
  --selection-update-interval 1 --min-selection-change-ratio 0.05 \
  --verify-fallback sequential_on_low_margin

/public/home/xlwang/mcy/conda_envs/specvlm/bin/python scripts/summarize_a100_dynamic.py \
  /public/home/xlwang/mcy/STD_assets/results/R1_full_three_att.jsonl \
  --output-dir /public/home/xlwang/mcy/STD_assets/results
```
