"""Unit tests for scripts/analysis/analyze_anchor_protection.py.

The script decides whether a "visual anchor protection" proposal has anything to
protect, so the two quantities it turns on -- how many tokens a cumulative-mass
target needs, and how much attention mass the anchor criterion would rescue --
are pinned down here on hand-computable cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts" / "analysis") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import analyze_anchor_protection as anchor  # noqa: E402


class TestMassNeeded:
    def test_counts_the_token_that_crosses_the_target(self):
        v = torch.tensor([0.5, 0.3, 0.2])
        assert anchor.mass_needed(v, 0.5) == 1     # the first token already reaches 50%
        assert anchor.mass_needed(v, 0.9) == 3     # 0.5+0.3 = 0.8 < 0.9
        assert anchor.mass_needed(v, 0.99) == 3

    def test_single_token_distribution_needs_one(self):
        assert anchor.mass_needed(torch.tensor([1.0, 0.0, 0.0]), 0.9) == 1

    def test_all_zero_mass_is_safe(self):
        assert anchor.mass_needed(torch.zeros(4), 0.9) == 0


class TestAnalyzeRound:
    def _scores(self):
        # 1 head, 3 queries, 3 visual tokens. Each query row sums to 1.
        # token 0 wins on a single query (anchor criterion);
        # token 1 wins on the sum (selector criterion).
        return torch.tensor([[[0.60, 0.25, 0.15],
                              [0.10, 0.45, 0.45],
                              [0.10, 0.45, 0.45]]])

    def test_anchor_criterion_disagrees_with_the_selector(self):
        rows = anchor.analyze_round(self._scores(), k=1)
        assert len(rows) == 1
        row = rows[0]
        # top-1 by sum is token 1; top-1 by per-token max is token 0.
        assert row["jaccard_selector_vs_anchor"] == pytest.approx(0.0)
        assert row["anchors_evicted"] == 1
        # aggregate mass on token 0 is 0.60+0.10+0.10 = 0.80 of a total of 3.0
        assert row["mass_of_evicted_anchors"] == pytest.approx(0.80 / 3.0)
        assert row["mass_kept_by_topk"] == pytest.approx(1.15 / 3.0)

    def test_identical_criteria_evict_nothing(self):
        scores = torch.tensor([[[0.7, 0.2, 0.1]]])  # single query: sum == max
        row = anchor.analyze_round(scores, k=1)[0]
        assert row["jaccard_selector_vs_anchor"] == pytest.approx(1.0)
        assert row["anchors_evicted"] == 0
        assert row["mass_of_evicted_anchors"] == 0.0

    def test_k_clamped_to_the_number_of_tokens(self):
        scores = torch.tensor([[[0.6, 0.4]]])
        row = anchor.analyze_round(scores, k=99)[0]
        assert row["anchors_evicted"] == 0
        assert row["mass_kept_by_topk"] == pytest.approx(1.0)

    def test_diffuseness_columns_present(self):
        row = anchor.analyze_round(self._scores(), k=1)[0]
        for f in (0.5, 0.9, 0.99):
            assert row[f"n_for_{f}"] >= 1


class TestRender:
    def test_render_reports_the_headline_numbers(self):
        obs = [
            {"sample_id": "a", "round": 0, "k": 2, "visual_len": 100,
             "mass_kept_by_topk": 0.5, "jaccard_selector_vs_anchor": 0.75,
             "anchors_evicted": 3.0, "mass_of_evicted_anchors": 0.02,
             "n_for_0.5": 8.0, "n_for_0.9": 48.0, "n_for_0.99": 90.0},
            {"sample_id": "b", "round": 0, "k": 2, "visual_len": 100,
             "mass_kept_by_topk": 0.5, "jaccard_selector_vs_anchor": 0.75,
             "anchors_evicted": 3.0, "mass_of_evicted_anchors": 0.02,
             "n_for_0.5": 8.0, "n_for_0.9": 48.0, "n_for_0.99": 90.0},
        ]
        text = anchor.render(obs, Path("results/routing_traces_x"), 2)
        assert "0.750" in text          # jaccard
        assert "2.0000%" in text        # rescued mass
        assert "48" in text             # tokens for 90% mass
