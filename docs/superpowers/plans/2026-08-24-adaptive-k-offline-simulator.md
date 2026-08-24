# Adaptive-K Offline Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a CPU-only offline simulator that replays three Adaptive-K controllers over the ten existing VDC verification traces and produces evidence-labeled statistics, figures, and a Go/No-Go report.

**Architecture:** A pure NumPy core validates traces, computes causally shifted budgets, evaluates attention-mass coverage, and aggregates metrics. A thin CLI loads existing CPU tensors, writes CSV/JSON/Markdown artifacts, and renders plots using Matplotlib's non-interactive backend. Tests use synthetic traces and invoke the real CLI without loading a model or GPU.

**Tech Stack:** Python 3, NumPy, PyTorch CPU serialization, Matplotlib Agg, pytest, csv/json/pathlib.

**Spec:** `docs/superpowers/specs/2026-08-24-adaptive-k-offline-simulator-design.md`

## Global Constraints

- Do not modify or invoke `src/std_repro/std_qwen25vl.py`.
- Do not modify decoding loops, sparse cache layout, generated KV, dense verification, gamma, correctness paths, or runtime routing.
- Use only existing VideoDetailCaption traces; do not run new decoding.
- Use feedback from round `t` only for the budget of round `t+1`.
- Label all counterfactual acceptance values as `proxy`; they cannot independently produce a GO decision.
- Defaults are `K_min=512`, `K_max=8192`, rho values `0.80`, `0.90`, and `0.95`, and static K values 1024, 2048, 4096, and 8192.

---

### Task 1: Controller primitives and causal budget replay

**Files:**
- Create: `src/std_repro/adaptive_k_offline.py`
- Create: `tests/test_adaptive_k_offline.py`

**Interfaces:**
- Produces: `BudgetBounds(k_min: int, k_max: int, visual_len: int)` with `clamp(k: int) -> int`.
- Produces: `attention_mass_top_p(scores: np.ndarray, rho: float, bounds: BudgetBounds, previous_k: int) -> tuple[int, bool]` where the boolean marks a fallback.
- Produces: `acceptance_feedback(previous_k: int, accepted: float, proposed: int, bounds: BudgetBounds) -> int`.
- Produces: `budget_series(prefill_scores, round_scores, accepted, proposed, recorded_k, controller, rho, bounds) -> tuple[np.ndarray, int]`.

- [ ] **Step 1: Write failing controller tests**

Add literal tests covering the smallest cumulative Top-P K, lower/upper clamping, exact 0.5/0.8 acceptance boundaries, doubling/halving, invalid-score fallback, and causal shifting. The causal fixture is:

```python
prefill = np.array([0.7, 0.2, 0.1, 0.0])
round_scores = np.array([
    [0.1, 0.1, 0.1, 0.7],
    [0.6, 0.2, 0.1, 0.1],
])
bounds = BudgetBounds(k_min=1, k_max=4, visual_len=4)
k, fallbacks = budget_series(
    prefill, round_scores, accepted=[0, 2], proposed=[2, 2],
    recorded_k=2, controller="attention", rho=0.8, bounds=bounds,
)
assert k.tolist() == [2, 3]
assert fallbacks == 0
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `PYTHONPATH=src pytest -q tests/test_adaptive_k_offline.py`

Expected: collection fails because `std_repro.adaptive_k_offline` does not exist.

- [ ] **Step 3: Implement minimal controller code**

Implement strict validation for rho `(0, 1]`, positive bounds, `visual_len >= k_min`, one-dimensional scores, consistent round counts, positive proposed counts, and `accepted <= proposed`. Clip negative scores to zero, treat any non-finite or non-positive-total vector as a fallback, and use stable descending ordering (`np.argsort(-scores, kind="stable")`). Controller names are exactly `attention`, `acceptance`, and `hybrid`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `PYTHONPATH=src pytest -q tests/test_adaptive_k_offline.py`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit controller primitives**

```bash
git add src/std_repro/adaptive_k_offline.py tests/test_adaptive_k_offline.py
git commit -m "feat: add offline adaptive k controllers"
```

### Task 2: Strategy replay, proxy metrics, and aggregation

**Files:**
- Modify: `src/std_repro/adaptive_k_offline.py`
- Modify: `tests/test_adaptive_k_offline.py`

**Interfaces:**
- Produces: `SampleTrace(sample_id, dataset, visual_len, recorded_k, gamma, prefill_scores, round_scores, accepted, proposed)` dataclass with validation.
- Produces: `StrategySpec(name: str, kind: str, static_k: int | None = None, rho: float | None = None)`.
- Produces: `RoundResult` and `StrategySummary` dataclasses serializable through `dataclasses.asdict`.
- Produces: `default_strategies() -> list[StrategySpec]` containing the four static rows, three attention rows, one acceptance row, and three hybrid rows.
- Produces: `replay_strategy(trace: SampleTrace, spec: StrategySpec, bounds: BudgetBounds) -> tuple[list[RoundResult], StrategySummary]`.
- Produces: `aggregate_summaries(round_rows: list[RoundResult], specs: list[StrategySpec]) -> list[StrategySummary]`.

- [ ] **Step 1: Write failing replay and metric tests**

Use a hand-derived two-round fixture and assert:

```python
assert reconstruct_proposed([2, 3]) == np.array([2, 2]).tolist()
assert observed_mean_accept == 1.0
assert observed_accept_rate == 0.5
assert static_recorded_proxy_accept == observed_accept
assert tokens_per_kv_budget_unit == accepted_sum / (k_sum / 1024.0)
```

Add tests proving static selection always uses prefill ranking, adaptive selection uses the previous feedback ranking, proxy values are clipped to proposed length, K statistics use all rounds, and change fraction excludes sample boundaries.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `PYTHONPATH=src pytest -q tests/test_adaptive_k_offline.py`

Expected: failures name the missing dataclasses and replay functions.

- [ ] **Step 3: Implement replay and aggregation**

For every round, compute the actual recorded-static denominator from prefill Top-K at `recorded_k`, compute candidate coverage from the strategy's selector and K, then calculate:

```python
proxy = np.clip(observed * candidate_coverage / recorded_coverage, 0.0, proposed)
efficiency = accepted_sum / (np.sum(k_values) / 1024.0)
```

If recorded coverage is zero, retain observed acceptance and increment `zero_denominator_fallbacks`. Summaries contain evidence class, mean/median/min/max K, coefficient of variation, IQR, change fraction, observed mean acceptance/rate, proxy mean acceptance/rate, observed efficiency, proxy efficiency, and fallback counts.

- [ ] **Step 4: Run focused and mutation-oriented tests**

Run: `PYTHONPATH=src pytest -q tests/test_adaptive_k_offline.py`

Expected: all tests pass. Then temporarily reason through mutations for reversed thresholds, current-round leakage, wrong static ranking, and missing proxy clipping; each must be caught by a named test.

- [ ] **Step 5: Commit replay engine**

```bash
git add src/std_repro/adaptive_k_offline.py tests/test_adaptive_k_offline.py
git commit -m "feat: replay adaptive k budgets offline"
```

### Task 3: CLI, artifacts, plots, and report

**Files:**
- Create: `scripts/analysis/simulate_adaptive_k.py`
- Create: `tests/test_simulate_adaptive_k_cli.py`

**Interfaces:**
- CLI: `python scripts/analysis/simulate_adaptive_k.py --traces-dir PATH --dataset VideoDetailCaption --frame-num 128 --output-dir PATH`.
- Produces: `rounds.csv`, `summary.csv`, `manifest.json`, `report.md`, `figure1_adaptive_k_by_round.png`, and `figure2_acceptance_vs_average_k.png`.

- [ ] **Step 1: Write a failing end-to-end CLI test**

Create two synthetic `.pt` traces and a JSONL metadata file under pytest's `tmp_path`, invoke the script through `subprocess.run(..., check=False)`, and assert exit code zero plus all six artifacts. Parse the outputs and assert:

```python
assert manifest["new_decoding_runs"] == 0
assert manifest["sample_count"] == 2
assert "proxy" in {row["evidence"] for row in summary_rows}
assert report.count("## ") >= 7
assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
```

Add a second integration test where tensor round count disagrees with metadata and assert a non-zero exit code containing `round count mismatch`.

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `PYTHONPATH=src pytest -q tests/test_simulate_adaptive_k_cli.py`

Expected: failure because the CLI does not exist.

- [ ] **Step 3: Implement trace loading and artifact writers**

Load tensors with `torch.load(..., map_location="cpu", weights_only=True)`, average KV heads, reconstruct proposed lengths from query lengths, and pass NumPy arrays to the pure core. Use `matplotlib.use("Agg")` before importing pyplot. Write UTF-8 Markdown/JSON and deterministic CSV column order.

The report headings are exactly:

```markdown
## 1. Dataset
## 2. Controller definitions
## 3. K distribution
## 4. Acceptance comparison table
## 5. Efficiency comparison
## 6. Pareto plot
## 7. Go/No-Go conclusion
```

Q1 uses coefficient of variation `< 0.05` and change fraction `< 0.10` as the constant-budget gate. Q2 is `INSUFFICIENT EVIDENCE / NO-GO` because all candidate acceptance values are proxy-only. Q3 reports optimistic draft cost reduction and explicitly states wall-clock speedup is unverified.

- [ ] **Step 4: Run CLI tests and verify GREEN**

Run: `PYTHONPATH=src pytest -q tests/test_simulate_adaptive_k_cli.py`

Expected: both CLI tests pass and Matplotlib emits no interactive-backend error.

- [ ] **Step 5: Run all feature tests**

Run: `PYTHONPATH=src pytest -q tests/test_adaptive_k_offline.py tests/test_simulate_adaptive_k_cli.py`

Expected: all feature tests pass.

- [ ] **Step 6: Commit CLI and report generation**

```bash
git add scripts/analysis/simulate_adaptive_k.py tests/test_simulate_adaptive_k_cli.py
git commit -m "feat: report adaptive k offline simulation"
```

### Task 4: Real trace simulation and final evidence audit

**Files:**
- Generate (ignored): `results/adaptive_k_offline/*`
- Modify only if findings require documentation correction: `PROGRESS.md`

**Interfaces:**
- Consumes the ten existing `results/routing_traces/VideoDetailCaption_frame128.jsonl` samples and matching `.pt` files.
- Produces the final seven-section report and plots without model/GPU access.

- [ ] **Step 1: Run the simulator on real traces**

Run:

```bash
PYTHONPATH=src python scripts/analysis/simulate_adaptive_k.py \
  --traces-dir results/routing_traces \
  --dataset VideoDetailCaption \
  --frame-num 128 \
  --output-dir results/adaptive_k_offline
```

Expected: exit zero, `sample_count=10`, and no model-loading or CUDA output.

- [ ] **Step 2: Audit generated evidence**

Run a read-only check that parses `manifest.json` and `summary.csv`, verifies 10 samples, four static strategies, seven adaptive strategies, K bounds 512..8192, finite metrics, explicit evidence labels, and six output artifacts. Open both PNG files to verify axes, legends, and readable labels.

- [ ] **Step 3: Update project progress with measured outcome**

Append a concise Adaptive-K Offline Simulation section to `PROGRESS.md` containing the dataset, K variation result, the evidence limitation, efficiency finding, and final Go/No-Go decision. Do not edit historical conclusions.

- [ ] **Step 4: Run full verification**

Run:

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src/std_repro/adaptive_k_offline.py scripts/analysis/simulate_adaptive_k.py tests
git diff --check
git status --short
```

Expected: zero test failures, compile exit zero, no whitespace errors, and no modifications to decoder/cache/model correctness files caused by this phase.

- [ ] **Step 5: Commit progress documentation**

```bash
git add PROGRESS.md
git commit -m "docs: record adaptive k offline findings"
```

