"""Unit tests for the CSV-0 certificate probe.

The probe decides whether a certified sparse verification layer is worth
building, so its two load-bearing outputs are pinned here:

  * the position-level sparse/dense agreement (the correctness ceiling), and
  * the coverage/precision curve of a certificate that must fire only on rounds
    where the whole block agrees.

Both are pure functions, so they are tested on synthetic logits without a model.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import std_repro.certificate_probe as cp  # noqa: E402


def logits(*rows):
    return torch.tensor(rows, dtype=torch.float32)


class TestPositionRecords:
    def test_identical_distributions_agree(self):
        block = logits([10.0, 0.0, 1.0], [0.0, 4.0, 1.0])
        records = cp.position_records(block, block.clone())
        assert [record["agree"] for record in records] == [True, True]
        assert records[0]["sparse_top1"] == records[0]["dense_top1"] == 0
        assert records[1]["sparse_top1"] == records[1]["dense_top1"] == 1

    def test_different_argmax_disagrees(self):
        sparse = logits([5.0, 0.0, 1.0])
        dense = logits([0.0, 5.0, 1.0])
        record = cp.position_records(sparse, dense)[0]
        assert record["agree"] is False
        assert record["sparse_top1"] == 0
        assert record["dense_top1"] == 1

    def test_margin_is_the_top1_top2_gap(self):
        record = cp.position_records(logits([3.0, 1.0, 0.0]), logits([3.0, 1.0, 0.0]))[0]
        assert record["margin"] == pytest.approx(2.0)

    def test_entropy_is_lower_for_peaked_distributions(self):
        peaked = cp.position_records(logits([20.0, 0.0]), logits([20.0, 0.0]))[0]
        flat = cp.position_records(logits([1.0, 1.0]), logits([1.0, 1.0]))[0]
        assert peaked["entropy"] < flat["entropy"]
        assert flat["entropy"] == pytest.approx(math.log(2.0), rel=1e-5)

    def test_shape_or_rank_mismatch_is_rejected(self):
        with pytest.raises(ValueError):
            cp.position_records(logits([1.0, 0.0]), logits([1.0, 0.0, 0.0]))
        with pytest.raises(ValueError):
            cp.position_records(torch.zeros(2, 2, 2), torch.zeros(2, 2, 2))


class TestRoundSummary:
    def test_collapses_a_round(self):
        summary = cp.round_summary(
            [
                {"agree": True, "margin": 4.0, "entropy": 0.1},
                {"agree": False, "margin": 1.0, "entropy": 0.3},
            ]
        )
        assert summary["positions"] == 2
        assert summary["agreements"] == 1
        assert summary["all_agree"] is False
        assert summary["min_margin"] == 1.0
        assert summary["mean_margin"] == pytest.approx(2.5)
        assert summary["max_entropy"] == 0.3
        assert summary["mean_entropy"] == pytest.approx(0.2)

    def test_fully_agreeing_round(self):
        summary = cp.round_summary([{"agree": True, "margin": 2.0, "entropy": 0.1}])
        assert summary["all_agree"] is True

    def test_empty_round_is_rejected(self):
        with pytest.raises(ValueError):
            cp.round_summary([])


class TestThresholds:
    def test_monotonic_and_spanning(self):
        rounds = [{"min_margin": value} for value in (0.0, 1.0, 2.0, 3.0, 100.0)]
        thresholds = cp.default_thresholds(rounds, count=8)
        assert thresholds == sorted(thresholds)
        assert thresholds[0] == pytest.approx(0.0)
        assert thresholds[-1] == pytest.approx(100.0)

    def test_constant_margins_yield_a_single_threshold(self):
        rounds = [{"min_margin": 2.0} for _ in range(4)]
        assert cp.default_thresholds(rounds) == [2.0]

    def test_no_rounds_yields_no_thresholds(self):
        assert cp.default_thresholds([]) == []


class TestCertificateCurve:
    def test_ceiling_and_precision(self):
        rounds = [
            {"min_margin": 5.0, "all_agree": True},
            {"min_margin": 4.0, "all_agree": True},
            {"min_margin": 3.0, "all_agree": False},
        ]
        curve = cp.certificate_curve(rounds, [0.0, 3.5, 4.5])
        assert all(row["ceiling"] == pytest.approx(2 / 3) for row in curve)

        loose = next(row for row in curve if row["threshold"] == 0.0)
        assert loose["coverage"] == pytest.approx(1.0)
        assert loose["precision"] == pytest.approx(2 / 3)
        assert loose["wrong_skips"] == 1

        strict = next(row for row in curve if row["threshold"] == 4.5)
        assert strict["certified_rounds"] == 1
        assert strict["coverage"] == pytest.approx(1 / 3)
        assert strict["precision"] == pytest.approx(1.0)
        assert strict["wrong_skips"] == 0

    def test_threshold_above_everything_certifies_nothing(self):
        rounds = [{"min_margin": 1.0, "all_agree": True}]
        row = cp.certificate_curve(rounds, [99.0])[0]
        assert row["certified_rounds"] == 0
        assert row["coverage"] == 0.0
        assert row["precision"] == 0.0

    def test_empty_rounds_rejected(self):
        with pytest.raises(ValueError):
            cp.certificate_curve([], [0.0])


class TestOperatingPoint:
    def test_picks_highest_coverage_with_perfect_precision(self):
        curve = [
            {"threshold": 0.0, "coverage": 1.0, "precision": 0.5},
            {"threshold": 1.0, "coverage": 0.6, "precision": 1.0},
            {"threshold": 2.0, "coverage": 0.3, "precision": 1.0},
        ]
        point = cp.select_operating_point(curve, min_precision=1.0)
        assert point["threshold"] == 1.0
        assert point["coverage"] == pytest.approx(0.6)

    def test_returns_none_when_no_point_is_precise_enough(self):
        curve = [{"threshold": 0.0, "coverage": 1.0, "precision": 0.5}]
        assert cp.select_operating_point(curve, min_precision=0.9) is None

    def test_relaxed_precision_admits_more_coverage(self):
        curve = [
            {"threshold": 0.0, "coverage": 1.0, "precision": 0.96},
            {"threshold": 1.0, "coverage": 0.5, "precision": 1.0},
        ]
        assert cp.select_operating_point(curve, min_precision=0.95)["coverage"] == pytest.approx(1.0)


class TestAggregate:
    def _stats(self, sample_id, rounds, **timing):
        return {"sample_id": sample_id, "rounds": rounds, **timing}

    def test_pools_rounds_and_sums_timing(self):
        stats = [
            self._stats(
                "a",
                [{"positions": 2, "agreements": 2, "all_agree": True, "min_margin": 5.0}],
                draft_seconds=1.0, sparse_pass_seconds=2.0,
                cached_sparse_pass_seconds=1.5, dense_pass_seconds=3.0, mask_prepare_seconds=0.1,
            ),
            self._stats(
                "b",
                [{"positions": 2, "agreements": 1, "all_agree": False, "min_margin": 0.5}],
                draft_seconds=1.0, sparse_pass_seconds=2.0,
                cached_sparse_pass_seconds=1.5, dense_pass_seconds=3.0, mask_prepare_seconds=0.1,
            ),
        ]
        pooled = cp.aggregate(stats)
        assert pooled["rounds"] == 2
        assert pooled["positions"] == 4
        assert pooled["position_agreement"] == pytest.approx(0.75)
        assert pooled["round_ceiling"] == pytest.approx(0.5)
        assert pooled["timing"]["sparse_pass_seconds"] == pytest.approx(4.0)
        assert pooled["timing"]["mask_prepare_seconds"] == pytest.approx(0.2)

    def test_curve_reflects_the_pooled_rounds(self):
        stats = [
            self._stats("a", [{"positions": 2, "agreements": 2, "all_agree": True, "min_margin": 5.0}]),
            self._stats("b", [{"positions": 2, "agreements": 0, "all_agree": False, "min_margin": 0.0}]),
        ]
        pooled = cp.aggregate(stats)
        strict = pooled["strict_operating_point"]
        assert strict is not None
        assert strict["coverage"] == pytest.approx(0.5)
        assert strict["precision"] == pytest.approx(1.0)

    def test_no_rounds_rejected(self):
        with pytest.raises(ValueError):
            cp.aggregate([])
        with pytest.raises(ValueError):
            cp.aggregate([{"sample_id": "a", "rounds": []}])


class TestDraftAgreement:
    """Sparse/draft agreement is the second free opinion the conjunction rule uses."""

    def test_records_draft_agreement_per_position(self):
        block = logits([10.0, 0.0], [0.0, 10.0], [5.0, 0.0])
        # position 0 predicts draft[1]=0 (sparse top1 is 0 -> agrees)
        # position 1 predicts draft[2]=1 (sparse top1 is 1 -> agrees)
        # position 2 predicts the bonus, which no draft proposed
        records = cp.position_records(block, block.clone(), [0, 1, None])
        assert [record["draft_agree"] for record in records] == [True, True, None]

    def test_disagreement_is_recorded(self):
        sparse = logits([0.0, 10.0])
        records = cp.position_records(sparse, sparse.clone(), [3])
        assert records[0]["draft_agree"] is False

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError):
            cp.position_records(logits([1.0, 0.0]), logits([1.0, 0.0]), [1, 2])

    def test_round_summary_excludes_the_bonus_row_from_the_conjunction(self):
        summary = cp.round_summary(
            [
                {"agree": True, "margin": 3.0, "entropy": 0.1, "draft_agree": True},
                {"agree": True, "margin": 3.0, "entropy": 0.1, "draft_agree": None},
            ]
        )
        assert summary["all_draft_agree"] is True
        assert summary["draft_judged"] == 1

    def test_round_summary_fails_the_conjunction_on_one_disagreement(self):
        summary = cp.round_summary(
            [
                {"agree": True, "margin": 3.0, "entropy": 0.1, "draft_agree": True},
                {"agree": True, "margin": 3.0, "entropy": 0.1, "draft_agree": False},
            ]
        )
        assert summary["all_draft_agree"] is False

    def test_round_with_no_drafted_positions_is_not_certified(self):
        summary = cp.round_summary([{"agree": True, "margin": 3.0, "entropy": 0.1, "draft_agree": None}])
        assert summary["all_draft_agree"] is False


def _round(min_margin, all_agree, all_draft_agree=True):
    return {"min_margin": min_margin, "all_agree": all_agree, "all_draft_agree": all_draft_agree}


class TestConjunctionRule:
    def test_conjunction_can_only_remove_rounds(self):
        rounds = [_round(5.0, True, True), _round(5.0, True, False), _round(1.0, False, True)]
        loose = cp.certificate_curve(rounds, [0.0], "margin")
        strict = cp.certificate_curve(rounds, [0.0], "margin_and_draft")
        assert loose[0]["certified_rounds"] == 3
        assert strict[0]["certified_rounds"] == 2

    def test_conjunction_removes_a_wrong_skip(self):
        rounds = [_round(5.0, True, True), _round(5.0, False, False)]
        loose = cp.certificate_curve(rounds, [0.0], "margin")[0]
        strict = cp.certificate_curve(rounds, [0.0], "margin_and_draft")[0]
        assert loose["precision"] == pytest.approx(0.5)
        assert strict["precision"] == pytest.approx(1.0)
        assert strict["wrong_skips"] == 0

    def test_unknown_rule_is_rejected(self):
        with pytest.raises(ValueError):
            cp.certificate_curve([_round(1.0, True)], [0.0], "nonsense")


class TestHeldoutOperatingPoint:
    def test_threshold_is_chosen_on_one_fold_and_scored_on_the_other(self):
        # Margins alternate so the even fold is clean and the odd fold is not.
        rounds = []
        for index in range(8):
            if index % 2 == 0:
                rounds.append(_round(9.0, True))
            else:
                rounds.append(_round(9.0, False))
        result = cp.heldout_operating_point(rounds, "margin", min_precision=1.0)
        assert result["usable"] is True
        assert len(result["folds"]) == 2
        # every fold's test half is the contaminated odd fold, so precision is 0
        assert result["mean_test_precision"] == pytest.approx(0.0)
        assert result["total_test_wrong_skips"] > 0

    def test_too_few_rounds_is_reported_not_guessed(self):
        result = cp.heldout_operating_point([_round(1.0, True)], "margin")
        assert result["usable"] is False
        assert "reason" in result

    def test_clean_rounds_survive_out_of_sample(self):
        rounds = [_round(9.0, True) for _ in range(8)]
        result = cp.heldout_operating_point(rounds, "margin", min_precision=1.0)
        assert result["mean_test_coverage"] == pytest.approx(1.0)
        assert result["total_test_wrong_skips"] == 0


class TestProjection:
    def _timing(self):
        return {
            "draft_seconds": 2.0,
            "cached_sparse_pass_seconds": 0.3,
            "mask_prepare_seconds": 0.0,
            "dense_pass_seconds": 0.6,
            "dense_bonus_seconds": 0.2,
            "bonus_seconds": 0.5,
        }

    def test_zero_coverage_is_static_plus_the_middle_pass(self):
        row = cp.projected_speedup(self._timing(), 0.0)
        # static = 2.0 + 0.6 + 0.5 = 3.1 ; csv = 2.0 + 0.3 + 0.3 + 0.8 = 3.4
        assert row["static_seconds"] == pytest.approx(3.1)
        assert row["csv_seconds"] == pytest.approx(3.4)
        assert row["speedup_vs_static"] < 1.0

    def test_full_coverage_keeps_only_the_sparse_bonus(self):
        row = cp.projected_speedup(self._timing(), 1.0)
        assert row["csv_seconds"] == pytest.approx(2.0 + 0.3 + 0.3)
        assert row["speedup_vs_static"] > 1.0

    def test_speedup_is_monotonic_in_coverage(self):
        rates = [cp.projected_speedup(self._timing(), c)["speedup_vs_static"] for c in (0.0, 0.3, 0.6, 1.0)]
        assert rates == sorted(rates)

    def test_draft_dominates_so_the_upside_is_bounded(self):
        # Even skipping every dense pass cannot beat draft + middle, so the ceiling
        # speedup is static/(draft+middle+sparse_bonus).
        row = cp.projected_speedup(self._timing(), 1.0)
        assert row["speedup_vs_static"] == pytest.approx(3.1 / 2.6, rel=1e-6)
