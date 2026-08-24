# Adaptive-K Offline Simulator Design

## Purpose

Determine whether verification feedback contains enough signal to vary the visual KV budget and whether that signal justifies a later runtime Adaptive-K experiment. This phase is read-only with respect to decoding: it consumes the existing VideoDetailCaption traces and must not modify or invoke the decoding loop, sparse cache, generated KV, dense verification, gamma, or correctness path.

## Dataset and Evidence Boundary

The primary input is `results/routing_traces/VideoDetailCaption_frame128.jsonl` plus the ten matching `.pt` files. The dataset contains ten VideoDetailCaption samples collected with 128 frames, up to 256 generated tokens, gamma 9, and approximately 15K visual tokens per sample.

All existing primary traces were generated at visual `K=996`, because the collection run used `K+text=1024`. Consequently:

- observed accepted lengths and accept rates are valid only for that recorded trajectory;
- controller K decisions can be replayed exactly from recorded attention and acceptance feedback;
- acceptance at static K values 1024, 2048, 4096, and 8192, or at adaptive K values, is not directly observed;
- no report may describe a counterfactual acceptance estimate as measured evidence.

## Considered Evaluation Approaches

### 1. Evidence-layered replay (selected)

Replay controller decisions exactly, report observed acceptance separately, and add a clearly labeled attention-mass acceptance proxy as a sensitivity analysis. This preserves the no-new-decoding constraint and prevents proxy results from being mistaken for causal evidence.

### 2. Proxy-only frontier

Scale accepted length directly by counterfactual attention-mass coverage and present the result as the Adaptive-K frontier. This produces all requested table cells but is rejected because a single-K trace cannot validate the proxy response curve and the resulting Pareto claim would be circular.

### 3. Multi-K static sweep

Run frozen STD at visual K values 1024, 2048, 4096, and 8192, then calibrate or directly compare adaptive policies. This would provide stronger evidence but is rejected in this phase because the user explicitly prohibited new decoding.

## Trace Semantics and Causality

For recorded verification round `t`, the simulator exposes:

- `A_t`: the recorded visual relevance vector, averaged across KV heads;
- `accepted_t`: recorded accepted draft tokens;
- `gamma`: recorded speculative width;
- `visual_len`: number of visual tokens;
- `recorded_k`: the visual K used to produce the trace.

To avoid future leakage, feedback from round `t` determines the budget for round `t+1`. Round zero starts from `clamp(recorded_k, K_min, K_max)`. If `visual_len < K_max`, the effective upper bound is `visual_len`.

## Controllers

All budgets are integer visual-token counts clamped to `[K_min, min(K_max, visual_len)]`, with defaults `K_min=512` and `K_max=8192`.

### Controller A: Attention Mass Top-P

For each feedback vector, sort non-negative visual scores descending and choose the smallest K whose cumulative score reaches `rho * total_score`, for `rho` in `{0.80, 0.90, 0.95}`. Clamp the result. If the score vector is empty, non-finite, or has non-positive total mass, retain the previous K.

### Controller B: Acceptance Feedback

Compute `accept_rate_t = accepted_t / gamma`. For the next round:

- if `accept_rate_t < 0.5`, double K;
- if `accept_rate_t > 0.8`, halve K using integer floor division;
- otherwise retain K.

Clamp after the update. Strict inequalities match the requested definition.

### Controller C: Hybrid

For each next-round decision, compute Attention Mass Top-P K and Acceptance Feedback K from the same previous-round feedback, then choose their maximum and clamp it. No smoothing, EMA, predictive model, or additional routing rule is allowed.

## Acceptance and Efficiency Metrics

### Observed replay metrics

The report gives the recorded mean accepted length and accept rate (`sum(accepted) / sum(query_lens where available, otherwise rounds * gamma)`). These values are repeated only as the observed K=996 baseline and are not attributed to a counterfactual controller.

### Proxy sensitivity metrics

For each decision K, compute predictive mass coverage by applying the selection induced by the previous feedback vector to the current round's relevance vector. Static K uses the prefill relevance ranking for every round. Adaptive Top-P and Hybrid use the previous verification relevance ranking; acceptance feedback changes only the size of that ranking.

The proxy is anchored to observed accepted length at recorded K:

`estimated_accept_t(K) = clip(observed_accept_t * coverage_t(K) / coverage_t(recorded_k), 0, gamma)`.

This estimate is reported under an explicit `proxy` label and cannot independently satisfy the Q2 GO gate. Rounds with zero denominator fall back to observed acceptance and are counted in a warning field.

Draft cost is estimated as proportional to visual K. Effective efficiency is:

`tokens_per_kv_budget_unit = sum(accepted_or_proxy_tokens) / sum(K / 1024)`.

Both observed replay efficiency and proxy efficiency are reported; only the former uses measured tokens, while neither is a wall-clock measurement.

## Outputs

The simulator writes to `results/adaptive_k_offline/` by default:

- `rounds.csv`: per-sample, per-round controller K, observed acceptance, mass coverage, and proxy acceptance;
- `summary.csv`: K distribution, acceptance, efficiency, evidence class, and controller configuration;
- `report.md`: the requested seven-section report;
- `figure1_adaptive_k_by_round.png`: round ID versus mean adaptive K, with sample variability;
- `figure2_acceptance_vs_average_k.png`: static points and adaptive points, visually distinguishing observed and proxy values;
- `manifest.json`: inputs, sample count, trace settings, simulator arguments, and warnings.

The static comparison rows are visual K values 1024, 2048, 4096, and 8192. Adaptive rows include Attention Top-P and Hybrid at rho 0.80, 0.90, and 0.95, plus Acceptance Feedback. K distribution statistics include mean, median, minimum, and maximum.

## Go / No-Go Rules

### Q1: Does visual budget vary?

Report the coefficient of variation, interquartile range, and fraction of transitions where K changes. If adaptive K is effectively constant (coefficient of variation below 0.05 and change fraction below 0.10), answer NO and stop recommendation.

### Q2: Is adaptive better than fixed K?

The requested gate is either mean accepted length at least 0.5 token higher or average K at least 30% lower at the same accepted length. Because current traces contain no multi-K acceptance measurements, proxy results are hypothesis-generating only. The phase may answer GO only if the measured evidence supports the threshold; otherwise it answers `INSUFFICIENT EVIDENCE / NO-GO` even when the proxy crosses it.

### Q3: Could this become a wall-clock speedup?

Estimate the draft-attention saving from the ratio of average adaptive K to the comparable static K. Compare that optimistic saving with measured historical controller-related overhead from `PROGRESS.md` only as context. The answer must state that wall-clock speedup is unverified without a runtime experiment.

## Error Handling

The CLI fails with a non-zero exit code and actionable message when metadata is missing, a `.pt` file is missing, tensor round counts disagree with metadata, score width disagrees with `visual_len`, accepted-length counts disagree with rounds, gamma is invalid, or no matching sample exists. Non-finite and zero-mass score vectors use the retain-previous-K fallback and emit counts in the manifest and report.

## Code Structure

- `src/std_repro/adaptive_k_offline.py`: pure controller, replay, proxy, aggregation, and validation functions; no model or CUDA dependency beyond loading CPU tensors at the CLI boundary.
- `scripts/analysis/simulate_adaptive_k.py`: command-line orchestration, trace loading, CSV/JSON/report writing, and plotting.
- `tests/test_adaptive_k_offline.py`: unit tests for controller boundaries, causal one-round shift, validation, proxy anchoring, efficiency, and aggregation.
- `tests/test_simulate_adaptive_k_cli.py`: integration test using synthetic trace files to verify all required artifacts and evidence labels.

No existing decoder, cache, collector, model, or correctness file is modified.

## Verification

Implementation follows red-green-refactor TDD. Completion requires:

1. focused unit and CLI tests passing;
2. the full local test suite passing;
3. running the CLI against the ten existing VDC traces without GPU/model loading;
4. checking that all seven report sections and both PNG figures exist;
5. confirming by Git diff that `src/std_repro/std_qwen25vl.py`, dynamic decoding, sparse cache, and model correctness paths were untouched.
