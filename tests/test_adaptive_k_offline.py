from __future__ import annotations

import numpy as np
import pytest

from std_repro.adaptive_k_offline import (
    BudgetBounds,
    RoundResult,
    SampleTrace,
    StrategySpec,
    acceptance_feedback,
    aggregate_summaries,
    attention_mass_top_p,
    budget_series,
    default_strategies,
    reconstruct_proposed,
    replay_strategy,
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


def _sample_trace(sample_id: str = "sample-1") -> SampleTrace:
    return SampleTrace(
        sample_id=sample_id,
        dataset="VideoDetailCaption",
        visual_len=4,
        recorded_k=2,
        gamma=2,
        prefill_scores=np.array([0.6, 0.3, 0.1, 0.0]),
        round_scores=np.array(
            [
                [0.1, 0.1, 0.2, 0.6],
                [0.5, 0.3, 0.1, 0.1],
            ]
        ),
        accepted=np.array([1.0, 1.0]),
        proposed=np.array([2, 2]),
    )


def test_reconstruct_proposed_accounts_for_pending_token_after_round_zero() -> None:
    assert reconstruct_proposed([2, 3]) == [2, 2]


def test_recorded_static_replay_anchors_proxy_to_observed_acceptance() -> None:
    trace = _sample_trace()
    spec = StrategySpec(name="recorded", kind="static", static_k=2)

    rows, summary = replay_strategy(
        trace, spec, BudgetBounds(k_min=1, k_max=4, visual_len=4)
    )

    assert [row.proxy_accept for row in rows] == pytest.approx([1.0, 1.0])
    assert summary.observed_mean_accept == pytest.approx(1.0)
    assert summary.observed_accept_rate == pytest.approx(0.5)
    assert summary.proxy_mean_accept == pytest.approx(1.0)
    assert summary.observed_efficiency == pytest.approx(2.0 / (4.0 / 1024.0))


def test_static_and_adaptive_selectors_use_different_causal_rankings() -> None:
    trace = _sample_trace()
    bounds = BudgetBounds(k_min=1, k_max=4, visual_len=4)

    static_rows, _ = replay_strategy(
        trace, StrategySpec(name="static", kind="static", static_k=2), bounds
    )
    adaptive_rows, _ = replay_strategy(
        trace, StrategySpec(name="adaptive", kind="attention", rho=0.8), bounds
    )

    assert static_rows[1].candidate_coverage == pytest.approx(0.8)
    assert adaptive_rows[1].candidate_coverage == pytest.approx(0.2)
    assert adaptive_rows[1].proxy_accept == pytest.approx(0.25)


def test_proxy_acceptance_is_clipped_to_proposed_length() -> None:
    trace = _sample_trace()
    trace.accepted[:] = 2.0
    spec = StrategySpec(name="large", kind="static", static_k=4)

    rows, _ = replay_strategy(trace, spec, BudgetBounds(k_min=1, k_max=4, visual_len=4))

    assert [row.proxy_accept for row in rows] == [2.0, 2.0]


def test_aggregate_summaries_excludes_sample_boundaries_from_change_fraction() -> None:
    rows = [
        RoundResult.minimal("a", "s", 0, k=1, accepted=1.0, proposed=2),
        RoundResult.minimal("a", "s", 1, k=2, accepted=1.0, proposed=2),
        RoundResult.minimal("b", "s", 0, k=8, accepted=1.0, proposed=2),
        RoundResult.minimal("b", "s", 1, k=8, accepted=1.0, proposed=2),
    ]

    summary = aggregate_summaries(rows, [StrategySpec(name="s", kind="static", static_k=1)])[0]

    assert summary.mean_k == pytest.approx(4.75)
    assert summary.min_k == 1
    assert summary.max_k == 8
    assert summary.change_fraction == pytest.approx(0.5)


def test_default_strategies_have_requested_static_and_adaptive_rows() -> None:
    strategies = default_strategies()

    assert [s.static_k for s in strategies if s.kind == "static"] == [1024, 2048, 4096, 8192]
    assert [s.rho for s in strategies if s.kind == "attention"] == [0.8, 0.9, 0.95]
    assert len([s for s in strategies if s.kind == "acceptance"]) == 1
    assert [s.rho for s in strategies if s.kind == "hybrid"] == [0.8, 0.9, 0.95]


def test_sample_trace_rejects_invalid_acceptance() -> None:
    trace = _sample_trace()

    with pytest.raises(ValueError, match="accepted"):
        SampleTrace(
            **{
                **trace.__dict__,
                "accepted": np.array([3.0, 1.0]),
            }
        )
