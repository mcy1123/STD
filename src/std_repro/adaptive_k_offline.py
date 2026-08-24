"""Pure offline Adaptive-K controller and replay primitives.

This module operates only on recorded CPU-side arrays.  It does not import or
invoke any decoder, model, cache, or runtime routing implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

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
