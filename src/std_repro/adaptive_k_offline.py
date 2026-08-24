"""Pure offline Adaptive-K controller and replay primitives.

This module operates only on recorded CPU-side arrays.  It does not import or
invoke any decoder, model, cache, or runtime routing implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BudgetBounds:
    """Valid visual-token budget interval for one sample."""

    k_min: int
    k_max: int
    visual_len: int

    def __post_init__(self) -> None:
        if self.k_min <= 0 or self.k_max <= 0:
            raise ValueError("K bounds must be positive")
        if self.k_min > self.k_max:
            raise ValueError("k_min must not exceed k_max")
        if self.visual_len < self.k_min:
            raise ValueError("visual_len must be at least k_min")

    @property
    def effective_max(self) -> int:
        return min(self.k_max, self.visual_len)

    def clamp(self, k: int) -> int:
        return max(self.k_min, min(int(k), self.effective_max))


def attention_mass_top_p(
    scores: np.ndarray,
    rho: float,
    bounds: BudgetBounds,
    previous_k: int,
) -> Tuple[int, bool]:
    """Return the smallest clamped K reaching ``rho`` of total score mass."""

    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must be in (0, 1]")
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if values.size == 0 or not np.all(np.isfinite(values)):
        return bounds.clamp(previous_k), True
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= 0.0:
        return bounds.clamp(previous_k), True

    ordered = values[np.argsort(-values, kind="stable")]
    target = rho * total
    tolerance = max(abs(total) * 1e-12, np.finfo(np.float64).eps)
    raw_k = int(np.searchsorted(np.cumsum(ordered), target - tolerance, side="left") + 1)
    return bounds.clamp(raw_k), False


def acceptance_feedback(
    previous_k: int,
    accepted: float,
    proposed: int,
    bounds: BudgetBounds,
) -> int:
    """Apply the strict 0.5/0.8 acceptance feedback thresholds."""

    if proposed <= 0:
        raise ValueError("proposed must be positive")
    if accepted < 0 or accepted > proposed:
        raise ValueError("accepted must be between zero and proposed")
    rate = float(accepted) / int(proposed)
    if rate < 0.5:
        return bounds.clamp(int(previous_k) * 2)
    if rate > 0.8:
        return bounds.clamp(int(previous_k) // 2)
    return bounds.clamp(previous_k)


def budget_series(
    prefill_scores: np.ndarray,
    round_scores: np.ndarray,
    accepted: Sequence[float],
    proposed: Sequence[int],
    recorded_k: int,
    controller: str,
    rho: Optional[float],
    bounds: BudgetBounds,
) -> Tuple[np.ndarray, int]:
    """Replay per-round budgets with a one-round feedback delay."""

    prefill = np.asarray(prefill_scores, dtype=np.float64)
    rounds = np.asarray(round_scores, dtype=np.float64)
    if prefill.ndim != 1 or prefill.size != bounds.visual_len:
        raise ValueError("prefill score width must equal visual_len")
    if rounds.ndim != 2 or rounds.shape[1] != bounds.visual_len:
        raise ValueError("round score width must equal visual_len")
    if rounds.shape[0] != len(accepted) or rounds.shape[0] != len(proposed):
        raise ValueError("round count mismatch between scores, accepted, and proposed")
    if controller not in {"attention", "acceptance", "hybrid"}:
        raise ValueError(f"unsupported controller: {controller}")
    if controller in {"attention", "hybrid"} and rho is None:
        raise ValueError("rho is required for attention-based controllers")

    budgets = np.empty(rounds.shape[0], dtype=np.int64)
    if budgets.size == 0:
        return budgets, 0
    budgets[0] = bounds.clamp(recorded_k)
    fallbacks = 0
    for round_id in range(1, rounds.shape[0]):
        previous_k = int(budgets[round_id - 1])
        if controller == "acceptance":
            next_k = acceptance_feedback(
                previous_k, accepted[round_id - 1], int(proposed[round_id - 1]), bounds
            )
        else:
            attention_k, fallback = attention_mass_top_p(
                rounds[round_id - 1], float(rho), bounds, previous_k
            )
            fallbacks += int(fallback)
            if controller == "attention":
                next_k = attention_k
            else:
                feedback_k = acceptance_feedback(
                    previous_k, accepted[round_id - 1], int(proposed[round_id - 1]), bounds
                )
                next_k = bounds.clamp(max(attention_k, feedback_k))
        budgets[round_id] = next_k
    return budgets, fallbacks


def reconstruct_proposed(query_lens: Sequence[int]) -> List[int]:
    """Recover draft lengths from verification query lengths."""

    proposed: List[int] = []
    for round_id, query_len in enumerate(query_lens):
        value = int(query_len) if round_id == 0 else int(query_len) - 1
        if value <= 0:
            raise ValueError("query lengths imply a non-positive proposed length")
        proposed.append(value)
    return proposed


@dataclass
class SampleTrace:
    sample_id: str
    dataset: str
    visual_len: int
    recorded_k: int
    gamma: int
    prefill_scores: np.ndarray
    round_scores: np.ndarray
    accepted: np.ndarray
    proposed: np.ndarray

    def __post_init__(self) -> None:
        self.prefill_scores = np.asarray(self.prefill_scores, dtype=np.float64)
        self.round_scores = np.asarray(self.round_scores, dtype=np.float64)
        self.accepted = np.asarray(self.accepted, dtype=np.float64)
        self.proposed = np.asarray(self.proposed, dtype=np.int64)
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if self.visual_len <= 0 or self.gamma <= 0 or self.recorded_k <= 0:
            raise ValueError("visual_len, recorded_k, and gamma must be positive")
        if self.prefill_scores.shape != (self.visual_len,):
            raise ValueError("prefill score width must equal visual_len")
        if self.round_scores.ndim != 2 or self.round_scores.shape[1] != self.visual_len:
            raise ValueError("round score width must equal visual_len")
        round_count = self.round_scores.shape[0]
        if self.accepted.shape != (round_count,) or self.proposed.shape != (round_count,):
            raise ValueError("round count mismatch in SampleTrace")
        if np.any(self.proposed <= 0):
            raise ValueError("proposed lengths must be positive")
        if np.any(self.accepted < 0) or np.any(self.accepted > self.proposed):
            raise ValueError("accepted lengths must be between zero and proposed")


@dataclass(frozen=True)
class StrategySpec:
    name: str
    kind: str
    static_k: Optional[int] = None
    rho: Optional[float] = None

    def __post_init__(self) -> None:
        if self.kind not in {"static", "attention", "acceptance", "hybrid"}:
            raise ValueError(f"unsupported strategy kind: {self.kind}")
        if self.kind == "static" and (self.static_k is None or self.static_k <= 0):
            raise ValueError("static strategies require positive static_k")
        if self.kind in {"attention", "hybrid"} and self.rho is None:
            raise ValueError("attention-based strategies require rho")


@dataclass
class RoundResult:
    sample_id: str
    dataset: str
    strategy: str
    kind: str
    round_id: int
    k: int
    accepted_observed: float
    proposed: int
    recorded_coverage: float
    candidate_coverage: float
    proxy_accept: float
    controller_fallbacks: int = 0

    @classmethod
    def minimal(
        cls,
        sample_id: str,
        strategy: str,
        round_id: int,
        k: int,
        accepted: float,
        proposed: int,
    ) -> "RoundResult":
        return cls(
            sample_id=sample_id,
            dataset="fixture",
            strategy=strategy,
            kind="static",
            round_id=round_id,
            k=k,
            accepted_observed=accepted,
            proposed=proposed,
            recorded_coverage=1.0,
            candidate_coverage=1.0,
            proxy_accept=accepted,
        )


@dataclass
class StrategySummary:
    strategy: str
    kind: str
    evidence: str
    rounds: int
    mean_k: float
    median_k: float
    min_k: int
    max_k: int
    coefficient_of_variation: float
    iqr_k: float
    change_fraction: float
    observed_mean_accept: float
    observed_accept_rate: float
    proxy_mean_accept: float
    proxy_accept_rate: float
    observed_efficiency: float
    proxy_efficiency: float
    controller_fallbacks: int


def default_strategies() -> List[StrategySpec]:
    strategies = [
        StrategySpec(name=f"static_k{k}", kind="static", static_k=k)
        for k in (1024, 2048, 4096, 8192)
    ]
    strategies.extend(
        StrategySpec(name=f"attention_rho{rho:.2f}", kind="attention", rho=rho)
        for rho in (0.8, 0.9, 0.95)
    )
    strategies.append(StrategySpec(name="acceptance_feedback", kind="acceptance"))
    strategies.extend(
        StrategySpec(name=f"hybrid_rho{rho:.2f}", kind="hybrid", rho=rho)
        for rho in (0.8, 0.9, 0.95)
    )
    return strategies


def _ranking(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    safe = np.where(np.isfinite(values), np.clip(values, 0.0, None), 0.0)
    return np.argsort(-safe, kind="stable")


def _coverage(scores: np.ndarray, ranking: np.ndarray, k: int) -> float:
    values = np.asarray(scores, dtype=np.float64)
    safe = np.where(np.isfinite(values), np.clip(values, 0.0, None), 0.0)
    total = float(safe.sum())
    if total <= 0.0:
        return 0.0
    return float(safe[ranking[: int(k)]].sum() / total)


def replay_strategy(
    trace: SampleTrace,
    spec: StrategySpec,
    bounds: BudgetBounds,
) -> Tuple[List[RoundResult], StrategySummary]:
    """Replay one strategy over one recorded sample."""

    if trace.visual_len != bounds.visual_len:
        raise ValueError("trace visual_len does not match bounds")
    round_count = trace.round_scores.shape[0]
    if spec.kind == "static":
        budgets = np.full(round_count, bounds.clamp(int(spec.static_k)), dtype=np.int64)
        controller_fallbacks = 0
    else:
        budgets, controller_fallbacks = budget_series(
            trace.prefill_scores,
            trace.round_scores,
            trace.accepted,
            trace.proposed,
            trace.recorded_k,
            spec.kind,
            spec.rho,
            bounds,
        )

    prefill_ranking = _ranking(trace.prefill_scores)
    recorded_k = min(trace.recorded_k, trace.visual_len)
    rows: List[RoundResult] = []
    zero_denominator_fallbacks = 0
    for round_id in range(round_count):
        current_scores = trace.round_scores[round_id]
        recorded_coverage = _coverage(current_scores, prefill_ranking, recorded_k)
        if spec.kind == "static" or round_id == 0:
            candidate_ranking = prefill_ranking
        else:
            candidate_ranking = _ranking(trace.round_scores[round_id - 1])
        candidate_coverage = _coverage(current_scores, candidate_ranking, int(budgets[round_id]))
        if recorded_coverage <= 0.0:
            proxy_accept = float(trace.accepted[round_id])
            zero_denominator_fallbacks += 1
        else:
            proxy_accept = float(
                np.clip(
                    trace.accepted[round_id] * candidate_coverage / recorded_coverage,
                    0.0,
                    trace.proposed[round_id],
                )
            )
        rows.append(
            RoundResult(
                sample_id=trace.sample_id,
                dataset=trace.dataset,
                strategy=spec.name,
                kind=spec.kind,
                round_id=round_id,
                k=int(budgets[round_id]),
                accepted_observed=float(trace.accepted[round_id]),
                proposed=int(trace.proposed[round_id]),
                recorded_coverage=recorded_coverage,
                candidate_coverage=candidate_coverage,
                proxy_accept=proxy_accept,
                controller_fallbacks=(controller_fallbacks + zero_denominator_fallbacks)
                if round_id == round_count - 1
                else 0,
            )
        )
    return rows, _summarize(spec, rows)


def _summarize(spec: StrategySpec, rows: Sequence[RoundResult]) -> StrategySummary:
    if not rows:
        raise ValueError("cannot summarize an empty strategy")
    k_values = np.asarray([row.k for row in rows], dtype=np.float64)
    observed = np.asarray([row.accepted_observed for row in rows], dtype=np.float64)
    proxy = np.asarray([row.proxy_accept for row in rows], dtype=np.float64)
    proposed = np.asarray([row.proposed for row in rows], dtype=np.float64)
    transitions = 0
    changes = 0
    grouped: Dict[str, List[RoundResult]] = {}
    for row in rows:
        grouped.setdefault(row.sample_id, []).append(row)
    for sample_rows in grouped.values():
        ordered = sorted(sample_rows, key=lambda row: row.round_id)
        transitions += max(0, len(ordered) - 1)
        changes += sum(left.k != right.k for left, right in zip(ordered, ordered[1:]))
    mean_k = float(k_values.mean())
    budget_units = float(k_values.sum() / 1024.0)
    return StrategySummary(
        strategy=spec.name,
        kind=spec.kind,
        evidence="proxy",
        rounds=len(rows),
        mean_k=mean_k,
        median_k=float(np.median(k_values)),
        min_k=int(k_values.min()),
        max_k=int(k_values.max()),
        coefficient_of_variation=float(k_values.std() / mean_k) if mean_k else 0.0,
        iqr_k=float(np.percentile(k_values, 75) - np.percentile(k_values, 25)),
        change_fraction=float(changes / transitions) if transitions else 0.0,
        observed_mean_accept=float(observed.mean()),
        observed_accept_rate=float(observed.sum() / proposed.sum()),
        proxy_mean_accept=float(proxy.mean()),
        proxy_accept_rate=float(proxy.sum() / proposed.sum()),
        observed_efficiency=float(observed.sum() / budget_units),
        proxy_efficiency=float(proxy.sum() / budget_units),
        controller_fallbacks=sum(row.controller_fallbacks for row in rows),
    )


def aggregate_summaries(
    round_rows: Sequence[RoundResult], specs: Sequence[StrategySpec]
) -> List[StrategySummary]:
    """Aggregate per-round rows across samples without boundary transitions."""

    summaries: List[StrategySummary] = []
    for spec in specs:
        selected = [row for row in round_rows if row.strategy == spec.name]
        summaries.append(_summarize(spec, selected))
    return summaries
