"""Unit tests for the offline collector-estimator ablation (L1.2).

The point of the ablation is to test whether the *query subset* a collector
samples explains the churn the online A100 runs showed (~0.52 adjacent-round
Jaccard). These tests pin the estimator semantics, the reconstructed query
positions, and the churn comparison on a synthetic case with a known answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts" / "analysis") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import analyze_collector_ablation as abl  # noqa: E402

H, Q, V, K = 2, 5, 20, 5
HOT = (0, 1, 2, 3, 4)


def _round(noise_pos=None, noise_row=0, noise_val=3.0):
    r = torch.zeros(H, Q, V)
    r[:, :, list(HOT)] = 1.0
    if noise_pos is not None:
        r[:, noise_row, noise_pos] = noise_val
    return r


def _meta(**over):
    meta = {
        "sample_id": "s1",
        "k": K,
        "accept_lengths": [3, 3, 3, 3],
        "pending_lengths": [0, 0, 0, 0],
        "proposed_lengths": [5, 5, 5, 5],
    }
    meta.update(over)
    return meta


def _payload(rounds):
    return {"per_query": True, "round_scores": rounds}


class TestEstimatorScores:
    def test_none_positions_sums_every_query(self):
        r = torch.ones(H, Q, V)
        assert torch.allclose(abl.estimator_scores(r, None), torch.full((H, V), float(Q)))

    def test_subset_sums_only_selected_rows(self):
        r = torch.zeros(H, Q, V)
        r[:, 0, :] = 1.0
        r[:, 2, :] = 2.0
        r[:, 3, :] = 4.0
        out = abl.estimator_scores(r, [0, 2])
        assert torch.allclose(out, torch.full((H, V), 3.0))

    def test_out_of_range_positions_are_ignored(self):
        r = torch.ones(H, Q, V)
        out = abl.estimator_scores(r, [0, 99])
        assert torch.allclose(out, torch.ones(H, V))

    def test_all_positions_invalid_raises(self):
        r = torch.ones(H, Q, V)
        with pytest.raises(ValueError):
            abl.estimator_scores(r, [99])


class TestRoundPositions:
    def test_all_mode_is_the_full_reference(self):
        assert abl.round_positions("all", 3, 0, 5) is None

    def test_two_and_three_query_positions(self):
        assert abl.round_positions("two", 3, 0, 5) == [0, 2]
        assert abl.round_positions("three", 3, 0, 5) == [0, 2, 3]

    def test_positions_shift_with_pending_context(self):
        assert abl.round_positions("two", 1, 2, 5) == [0, 2]
        assert abl.round_positions("three", 1, 2, 5) == [0, 2, 3]


class TestAnalyzeSample:
    def test_requires_per_query_traces(self):
        with pytest.raises(ValueError, match="record-per-query"):
            abl.analyze_sample(_meta(), {"per_query": False, "round_scores": [_round()]})

    def test_rejects_stacked_tensor(self):
        with pytest.raises(ValueError, match="per-round list"):
            abl.analyze_sample(_meta(), {"per_query": True, "round_scores": torch.ones(4, H, V)})

    def test_rejects_mismatched_per_round_metadata(self):
        meta = _meta(pending_lengths=[0, 0])  # only two entries for four rounds
        with pytest.raises(ValueError, match="length mismatch"):
            abl.analyze_sample(meta, _payload([_round() for _ in range(4)]))

    def test_all_query_fidelity_is_one_and_stable(self):
        rounds = [_round() for _ in range(4)]
        result = abl.analyze_sample(_meta(), _payload(rounds))

        assert result["T"] == 4
        assert result["per_mode"]["all"]["fidelity"] == [1.0] * 4
        # The full-query reference never changes, so its churn is a perfect 1.0.
        assert all(c == pytest.approx(1.0) for c in result["per_mode"]["all"]["churn_jaccard"])

    def test_two_query_is_less_stable_than_the_full_reference(self):
        # Inject per-round noise only into the query row the two-query estimator
        # samples, so the subsampled selection moves while the full sum does not.
        rounds = [_round(noise_pos=6 + t, noise_row=0) for t in range(4)]
        result = abl.analyze_sample(_meta(), _payload(rounds))

        two = result["per_mode"]["two"]
        allq = result["per_mode"]["all"]
        assert np.mean(two["churn_jaccard"]) < np.mean(allq["churn_jaccard"]) - 0.1
        assert np.mean(two["fidelity"]) < 1.0
        assert np.mean(two["churn_jaccard"]) < 1.0


class TestAggregate:
    def test_macro_average_over_samples(self):
        rounds = [_round(noise_pos=6 + t, noise_row=0) for t in range(4)]
        samples = [
            abl.analyze_sample(_meta(sample_id="s1"), _payload(rounds)),
            abl.analyze_sample(_meta(sample_id="s2"), _payload([_round() for _ in range(4)])),
        ]

        summary = abl.aggregate(samples)

        assert summary["all"]["samples"] == 2
        assert summary["all"]["fidelity"] == pytest.approx(1.0)
        # s1's two-query fidelity is 0.8, s2's is 1.0 -> macro mean 0.9
        assert summary["two"]["fidelity"] == pytest.approx(0.9)
        assert summary["two"]["churn_jaccard"] < summary["all"]["churn_jaccard"]

    def test_empty_input_yields_nan(self):
        summary = abl.aggregate([])
        assert np.isnan(summary["two"]["fidelity"])
