from __future__ import annotations

import pytest
import torch

from std_repro.verification_policy import (
    min_prediction_margin,
    needs_sequential_fallback,
    positional_token_metrics,
)


@pytest.mark.parametrize(
    ("mode", "accept_len", "draft_len", "low_margin", "expected"),
    [
        ("none", 0, 9, True, False),
        ("sequential_guard", 9, 9, False, True),
        ("sequential_on_low_margin", 9, 9, True, True),
        ("sequential_on_low_margin", 0, 9, False, False),
        ("sequential_on_reject", 8, 9, False, True),
        ("sequential_on_reject", 9, 9, False, False),
        ("sequential_on_low_accept", 1, 9, False, True),
        ("sequential_on_low_accept", 2, 9, False, False),
    ],
)
def test_needs_sequential_fallback(
    mode: str,
    accept_len: int,
    draft_len: int,
    low_margin: bool,
    expected: bool,
) -> None:
    assert (
        needs_sequential_fallback(
            mode,
            accept_len=accept_len,
            draft_len=draft_len,
            sequential_fallback_max_accept=1,
            low_margin=low_margin,
        )
        is expected
    )


def test_min_prediction_margin_with_pending_uses_requested_predictions() -> None:
    logits = torch.tensor(
        [
            [
                [3.0, 2.8, 0.0, -1.0],
                [5.0, 1.0, 0.0, -1.0],
                [9.0, 0.0, 0.0, -1.0],
            ]
        ]
    )

    margin = min_prediction_margin(logits, [0, 1], has_dense_pending=True)

    assert margin == pytest.approx(0.2)


def test_min_prediction_margin_without_pending_skips_canonical_prefill_token() -> None:
    logits = torch.tensor(
        [
            [
                [4.0, 3.95, 0.0],
                [8.0, 1.0, 0.0],
            ]
        ]
    )

    margin = min_prediction_margin(logits, [0, 1], has_dense_pending=False)

    assert margin == pytest.approx(0.05)


def test_positional_token_metrics_counts_cascade_and_length_difference() -> None:
    reference = torch.tensor([[1, 2, 3, 4]])
    candidate = torch.tensor([[1, 9, 3]])

    metrics = positional_token_metrics(reference, candidate)

    assert metrics["mismatch_token_count"] == 2
    assert metrics["token_level_agreement"] == pytest.approx(0.5)
