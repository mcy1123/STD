"""Unit tests for the H_headroom stratification added to analyze_routing.py.

The thorough-analysis plan (docs/superpowers/plans/2026-09-17-dynamic-routing-
thorough-analysis.md) requires the recall gain of `Previous` over `Static` to be
reported per accept-length stratum, because the aggregate mean hid a sign flip
(+0.044 / -0.102 / -0.126 across three samples).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts" / "analysis") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import analyze_routing  # noqa: E402


def _sample(sample_id, accepts, static, prev):
    return {
        "sample_id": sample_id,
        "_dataset": "Video-MME",
        "T": len(static),
        "accept_lengths": accepts,
        "static_recall": static,
        "prev_recall": prev,
    }


class TestHeadroomRows:
    def test_joins_accept_length_with_recall_gain(self):
        rows = analyze_routing.headroom_rows(
            [_sample("s1", [1, 5], [0.50, 0.80], [0.70, 0.82])]
        )

        assert len(rows) == 2
        assert rows[0]["sample_id"] == "s1"
        assert rows[0]["accept_length"] == 1.0
        assert rows[0]["delta_recall"] == pytest.approx(0.20)
        assert rows[1]["delta_recall"] == pytest.approx(0.02)

    def test_missing_accept_entries_become_nan(self):
        rows = analyze_routing.headroom_rows(
            [_sample("s1", [1], [0.50, 0.60], [0.70, 0.65])]
        )

        assert len(rows) == 2
        assert not math.isfinite(rows[1]["accept_length"])
        # NaN rounds are excluded from stratification.
        assert len(analyze_routing.quantile_strata(rows)) == 1


class TestQuantileStrata:
    def test_lowest_headroom_first_and_equal_counts(self):
        # Larger accept length -> smaller gain, i.e. the H_headroom shape.
        rows = [
            {
                "accept_length": float(i),
                "static_recall": 0.5,
                "prev_recall": 0.5 + (5 - i) * 0.01,
                "delta_recall": (5 - i) * 0.01,
                "sample_id": "s",
                "dataset": "d",
                "round": i,
            }
            for i in range(6)
        ]

        strata = analyze_routing.quantile_strata(rows, bins=3)

        assert [s["n"] for s in strata] == [2, 2, 2]
        assert (strata[0]["accept_min"], strata[0]["accept_max"]) == (0.0, 1.0)
        assert strata[-1]["accept_max"] == 5.0
        assert strata[0]["delta_recall"] > strata[-1]["delta_recall"]
        assert strata[0]["mean_accept"] < strata[-1]["mean_accept"]

    def test_empty_input_returns_empty(self):
        assert analyze_routing.quantile_strata([]) == []

    def test_all_nan_returns_empty(self):
        rows = [{"accept_length": float("nan"), "static_recall": 0.5, "prev_recall": 0.6,
                 "delta_recall": 0.1}]
        assert analyze_routing.quantile_strata(rows) == []


class TestPearson:
    def test_perfect_positive(self):
        assert analyze_routing._pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)

    def test_perfect_negative(self):
        assert analyze_routing._pearson([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)

    def test_degenerate_is_nan(self):
        assert math.isnan(analyze_routing._pearson([1, 1, 1], [1, 2, 3]))
        assert math.isnan(analyze_routing._pearson([1], [1]))

    def test_spearman_matches_pearson_on_monotone_transform(self):
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.0, 4.0, 9.0, 16.0]  # strictly increasing but non-linear
        spearman = analyze_routing._pearson(
            analyze_routing._rankdata(x), analyze_routing._rankdata(y)
        )
        assert spearman == pytest.approx(1.0)


class TestHeadroomReport:
    def test_writes_artifacts_and_flags_consistency(self, tmp_path):
        # accept length grows while the gain shrinks -> H_headroom consistent.
        analyzed = [
            _sample(
                "s1",
                [0, 1, 3, 6],
                [0.50, 0.50, 0.50, 0.50],
                [0.70, 0.66, 0.55, 0.51],
            )
        ]

        summary = analyze_routing.headroom_report(analyzed, tmp_path)

        assert (tmp_path / "headroom_rounds.csv").exists()
        assert (tmp_path / "headroom_strata.json").exists()
        assert summary["n_rounds"] == 4
        assert summary["low_headroom_delta_recall"] > summary["high_headroom_delta_recall"]
        assert summary["low_headroom_delta_recall"] > 0
        # gain falls as accept length rises -> negative correlation
        assert summary["spearman_accept_vs_delta_recall"] < 0

        persisted = json.loads((tmp_path / "headroom_strata.json").read_text())
        assert persisted["n_rounds"] == 4
        header = (tmp_path / "headroom_rounds.csv").read_text().splitlines()[0]
        assert header.startswith("sample_id,dataset,round,accept_length")
