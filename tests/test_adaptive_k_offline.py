from __future__ import annotations

import numpy as np
import pytest

from std_repro.adaptive_k_offline import (
    BudgetBounds,
    acceptance_feedback,
    attention_mass_top_p,
    budget_series,
)


def test_attention_mass_top_p_chooses_smallest_budget_reaching_rho() -> None:
    bounds = BudgetBounds(k_min=1, k_max=4, visual_len=4)

    k, fallback = attention_mass_top_p(
        np.array([0.5, 0.3, 0.15, 0.05]), rho=0.8, bounds=bounds, previous_k=4
    )

    assert k == 2
    assert fallback is False


def test_attention_mass_top_p_clamps_to_effective_bounds() -> None:
    lower = BudgetBounds(k_min=3, k_max=8, visual_len=4)
    upper = BudgetBounds(k_min=1, k_max=8, visual_len=3)

    low_k, _ = attention_mass_top_p(
        np.array([0.9, 0.05, 0.03, 0.02]), rho=0.8, bounds=lower, previous_k=4
    )
    high_k, _ = attention_mass_top_p(
        np.array([0.34, 0.33, 0.33]), rho=1.0, bounds=upper, previous_k=1
    )

    assert low_k == 3
    assert high_k == 3


@pytest.mark.parametrize(
    ("scores", "previous_k"),
    [
        (np.array([0.0, 0.0]), 2),
        (np.array([1.0, np.nan]), 1),
        (np.array([]), 2),
    ],
)
def test_attention_mass_top_p_retains_previous_budget_for_invalid_mass(
    scores: np.ndarray, previous_k: int
) -> None:
    bounds = BudgetBounds(k_min=1, k_max=2, visual_len=2)

    k, fallback = attention_mass_top_p(scores, rho=0.9, bounds=bounds, previous_k=previous_k)

    assert k == previous_k
    assert fallback is True


@pytest.mark.parametrize("rho", [0.0, -0.1, 1.1])
def test_attention_mass_top_p_rejects_invalid_rho(rho: float) -> None:
    bounds = BudgetBounds(k_min=1, k_max=2, visual_len=2)

    with pytest.raises(ValueError, match="rho"):
        attention_mass_top_p(np.array([0.5, 0.5]), rho, bounds, previous_k=1)


def test_acceptance_feedback_uses_strict_thresholds_and_clamps() -> None:
    bounds = BudgetBounds(k_min=2, k_max=8, visual_len=8)

    assert acceptance_feedback(4, accepted=0, proposed=2, bounds=bounds) == 8
    assert acceptance_feedback(8, accepted=0, proposed=2, bounds=bounds) == 8
    assert acceptance_feedback(8, accepted=2, proposed=2, bounds=bounds) == 4
    assert acceptance_feedback(2, accepted=2, proposed=2, bounds=bounds) == 2
    assert acceptance_feedback(4, accepted=1, proposed=2, bounds=bounds) == 4
    assert acceptance_feedback(4, accepted=4, proposed=5, bounds=bounds) == 4


def test_budget_series_shifts_attention_feedback_by_one_round() -> None:
    prefill = np.array([0.7, 0.2, 0.1, 0.0])
    round_scores = np.array(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.6, 0.2, 0.1, 0.1],
        ]
    )
    bounds = BudgetBounds(k_min=1, k_max=4, visual_len=4)

    k, fallbacks = budget_series(
        prefill,
        round_scores,
        accepted=[0, 2],
        proposed=[2, 2],
        recorded_k=2,
        controller="attention",
        rho=0.8,
        bounds=bounds,
    )

    assert k.tolist() == [2, 3]
    assert fallbacks == 0


def test_budget_series_hybrid_takes_max_of_attention_and_acceptance() -> None:
    bounds = BudgetBounds(k_min=1, k_max=8, visual_len=8)
    scores = np.array(
        [
            [0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    k, _ = budget_series(
        scores[0],
        scores,
        accepted=[0, 2],
        proposed=[2, 2],
        recorded_k=2,
        controller="hybrid",
        rho=0.8,
        bounds=bounds,
    )

    assert k.tolist() == [2, 4]


def test_budget_series_rejects_inconsistent_rounds() -> None:
    bounds = BudgetBounds(k_min=1, k_max=2, visual_len=2)

    with pytest.raises(ValueError, match="round count"):
        budget_series(
            np.array([0.5, 0.5]),
            np.array([[0.5, 0.5]]),
            accepted=[1, 1],
            proposed=[2, 2],
            recorded_k=1,
            controller="acceptance",
            rho=None,
            bounds=bounds,
        )
