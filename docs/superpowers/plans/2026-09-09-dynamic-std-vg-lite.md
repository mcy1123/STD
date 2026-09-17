# Dynamic STD-VG Lite Implementation Plan

> **For agentic workers:** Use the plan task-by-task with test and benchmark checkpoints.

**Goal:** Implement and evaluate verifier-guided dynamic STD changes without sacrificing exact output parity.

**Architecture:** Keep AR and static STD unchanged. Add selectable verifier query policies and bootstrap selectors behind the dynamic entry point, with synchronized component timing and paired JSONL comparisons.

**Tech Stack:** PyTorch, Qwen2.5-VL, Video-MME, pytest, JSONL reports.

**Spec:** User-approved STD-VG Lite proposal in the conversation.

**Global Constraints:** GPU 1 only; never disturb the existing GPU 0 vLLM workload; fixed output budgets; offline model/data paths on GPU23.

## Goal

Evaluate and implement the lowest-risk parts of the verifier-guided dynamic STD proposal while preserving the existing AR/static paths and exact-output checks. The final benchmark must report prefill, decode, selection/collection, refresh, acceptance, and end-to-end timing on the Video-MME subset.

## Work sequence

1. Add a configurable two-query verifier collector. Keep the current three-query collector as the control, expose the query policy in the benchmark CLI, and unit-test query-position selection and score-shape invariants.
2. Add explicit refresh scheduling and hysteresis controls. Keep incremental cache refresh as the default safe path, test interval 1/2/4 and selection-change thresholds, and unit-test that skipped updates leave cache metadata and token order correct.
3. Add an optional attention-free bootstrap selector based on windowed key-centroid residual plus value norm and temporal/spatial coverage. Keep attention-based bootstrap as the default until a paired benchmark demonstrates no acceptance regression. Unit-test deterministic budget and coverage invariants on synthetic K/V tensors.
4. Record disjoint timing components and acceptance statistics in JSONL. Update the summarizer to compare every dynamic variant against AR and static.
5. Run local compile/tests, then on GPU1 run matched warmups and repetitions for baseline, each ablation, and the combined candidate. Stop using GPU1 if it is no longer idle; do not touch GPU0.

## Acceptance criteria

- All existing tests plus new unit tests pass.
- Every measured method emits the fixed requested token budget and is positionally exact against AR on paired trials.
- Reports show separate prefill/decode and dynamic overhead timings, not only one aggregate speedup.
- The combined candidate is declared beneficial only if synchronized decode and inference totals improve; otherwise the report identifies the measured bottleneck and retains the result as a negative ablation.
