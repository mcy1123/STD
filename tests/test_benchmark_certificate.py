"""Unit tests for the CSV-0 benchmark runner's guard and report.

The guard is the only thing standing between a timing probe and a GPU that
another user is still using, and the report is what a reader uses to accept or
reject the certified-verification idea, so both are pinned here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "scripts" / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import benchmark_certificate as bc  # noqa: E402

IDLE = "0, GPU-aaa, NVIDIA A100 80GB PCIe, 0, 81050, 0\n1, GPU-bbb, NVIDIA A100 80GB PCIe, 0, 81050, 0"
SHARED = "0, GPU-aaa, NVIDIA A100 80GB PCIe, 79033, 2229, 100\n1, GPU-bbb, NVIDIA A100 80GB PCIe, 21361, 60000, 0"
CROWDED = "1, GPU-bbb, NVIDIA A100 80GB PCIe, 70000, 10000, 55"


class TestGpuGuard:
    def test_parses_only_numeric_rows(self):
        rows = bc.gpu_rows(IDLE)
        assert [row[0] for row in rows] == ["0", "1"]
        assert bc.gpu_rows("") == []

    def test_idle_gpu_is_accepted(self):
        bc.require_usable_gpu(IDLE, 1, allow_shared=False, min_free_gib=40)

    def test_busy_gpu_is_rejected_by_default(self):
        with pytest.raises(RuntimeError, match="not idle"):
            bc.require_usable_gpu(SHARED, 1, allow_shared=False, min_free_gib=40)

    def test_shared_gpu_is_accepted_when_explicitly_allowed(self):
        bc.require_usable_gpu(SHARED, 1, allow_shared=True, min_free_gib=40)

    def test_shared_gpu_still_needs_enough_free_memory(self):
        with pytest.raises(RuntimeError, match="below the"):
            bc.require_usable_gpu(CROWDED, 1, allow_shared=True, min_free_gib=40)

    def test_missing_gpu_is_rejected(self):
        with pytest.raises(RuntimeError, match="not present"):
            bc.require_usable_gpu(IDLE, 7, allow_shared=True, min_free_gib=1)


def _aggregate(round_ceiling, strict, risk, curve, heldout_coverage=0.0, heldout_precision=1.0):
    rounds = 10
    heldout = {
        rule: {
            "rule": rule,
            "usable": True,
            "folds": [],
            "mean_test_coverage": heldout_coverage,
            "mean_test_precision": heldout_precision,
            "total_test_wrong_skips": 0,
        }
        for rule in ("margin", "margin_and_draft")
    }
    points = {"margin": strict, "margin_and_draft": strict}
    timing = {
        "draft_seconds": 2.0,
        "sparse_pass_seconds": 1.0,
        "cached_sparse_pass_seconds": 0.4,
        "dense_pass_seconds": 1.2,
        "bonus_seconds": 0.2,
        "dense_bonus_seconds": 0.1,
        "mask_prepare_seconds": 0.01,
        "per_layer_mask_build_seconds": 0.5,
        "mask_build_microbench_seconds": 0.002,
    }
    return {
        "rounds": rounds,
        "positions": 90,
        "position_agreement": 0.87,
        "draft_position_agreement": 0.91,
        "round_ceiling": round_ceiling,
        "curve": curve,
        "curves": {"margin": curve, "margin_and_draft": curve},
        "strict_operating_point": strict,
        "risk_operating_point": risk,
        "strict_operating_points": points,
        "risk_operating_points": points,
        "heldout_operating_points": heldout,
        "projections": {
            "ceiling": {"coverage": round_ceiling, "static_seconds": 3.4, "csv_seconds": 3.0,
                        "speedup_vs_static": 1.13},
            "margin_strict": {"coverage": strict["coverage"] if strict else 0.0, "static_seconds": 3.4,
                              "csv_seconds": 3.3, "speedup_vs_static": 1.03},
        },
        "timing": timing,
        "cached_mask_hits": 280,
        "cached_mask_misses": 0,
        "per_layer_mask_builds": 280,
        "logit_mismatch_rounds": 0,
    }


def _args(**overrides):
    base = dict(frame_num=32, max_new_tokens=64, gamma=9, target_k_plus_text=1024)
    base.update(overrides)
    return argparse.Namespace(**base)


class TestRenderReport:
    def test_reports_ceiling_cost_and_gate(self):
        curve = [
            {"threshold": 0.0, "certified_rounds": 10, "coverage": 1.0, "precision": 0.6, "wrong_skips": 4},
            {"threshold": 5.0, "certified_rounds": 4, "coverage": 0.4, "precision": 1.0, "wrong_skips": 0},
        ]
        strict = curve[1]
        aggregate = _aggregate(0.6, strict, None, curve)
        text = bc.render_report(aggregate, sample_ids=["050-1"], args=_args())
        assert "Correctness ceiling" in text
        assert "Middle-pass cost" in text
        assert "Gate" in text
        assert "87.00%" in text
        assert "60.00%" in text
        assert "max |delta|" in text

    def test_derives_per_round_costs(self):
        curve = [{"threshold": 0.0, "certified_rounds": 10, "coverage": 1.0, "precision": 1.0, "wrong_skips": 0}]
        aggregate = _aggregate(1.0, curve[0], None, curve)
        text = bc.render_report(aggregate, sample_ids=["a"], args=_args())
        # 1.0 s over 10 rounds = 100 ms/round; cached 0.4 s = 40 ms/round
        assert "100.0" in text
        assert "40.0" in text
        assert "120.0" in text

    def test_missing_strict_point_is_stated_explicitly(self):
        curve = [{"threshold": 0.0, "certified_rounds": 10, "coverage": 1.0, "precision": 0.4, "wrong_skips": 6}]
        aggregate = _aggregate(0.0, None, None, curve)
        text = bc.render_report(aggregate, sample_ids=["a"], args=_args())
        assert "none" in text

    def test_gate_verdict_reflects_a_cheap_middle_pass(self):
        curve = [{"threshold": 0.0, "certified_rounds": 10, "coverage": 1.0, "precision": 1.0, "wrong_skips": 0}]
        aggregate = _aggregate(1.0, curve[0], None, curve)
        # cached pass 40 ms vs saving 120+20 ms -> requirement below 100%
        text = bc.render_report(aggregate, sample_ids=["a"], args=_args())
        assert "NECESSARY" in text

    def test_gate_rejects_an_expensive_middle_pass(self):
        curve = [{"threshold": 0.0, "certified_rounds": 10, "coverage": 1.0, "precision": 1.0, "wrong_skips": 0}]
        aggregate = _aggregate(1.0, curve[0], None, curve, )
        aggregate["timing"]["sparse_pass_seconds"] = 30.0
        aggregate["timing"]["cached_sparse_pass_seconds"] = 30.0
        text = bc.render_report(aggregate, sample_ids=["a"], args=_args())
        assert "IMPOSSIBLE" in text


class TestJsonable:
    def test_paths_become_strings(self):
        assert bc._jsonable(Path("results/x.jsonl")) == "results/x.jsonl"

    def test_nested_arg_namespaces_are_serializable(self):
        payload = bc._jsonable({"args": dict(vars(_args()), output=Path("a/b.jsonl"))})
        assert payload["args"]["output"] == "a/b.jsonl"

    def test_tensor_like_values_become_lists(self):
        class Fake:
            ndim = 1

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return [1, 2]

        assert bc._jsonable(Fake()) == [1, 2]


class TestViabilityVerdict:
    """Section D is the decision the whole probe exists to make."""

    def _curve(self, coverage, precision):
        return [{"threshold": 1.0, "certified_rounds": 5, "coverage": coverage,
                 "precision": precision, "wrong_skips": 0}]

    def test_open_and_reachable_when_the_heldout_split_confirms_it(self):
        curve = self._curve(0.40, 1.0)
        text = bc.render_report(
            _aggregate(0.87, curve[0], None, curve, heldout_coverage=0.40),
            sample_ids=["a"], args=_args(),
        )
        assert "OPEN AND REACHABLE" in text

    def test_in_sample_only_when_the_split_misses_the_rate_by_a_little(self):
        # required rate is 41/130 = 31.5%; in-sample 40%, held-out 31% -> below the
        # rate but within the noise band, so the rule reports in-sample-only.
        curve = self._curve(0.40, 1.0)
        text = bc.render_report(
            _aggregate(0.87, curve[0], None, curve, heldout_coverage=0.31),
            sample_ids=["a"], args=_args(),
        )
        assert "OPEN, IN-SAMPLE ONLY" in text

    def test_unresolved_when_the_split_disagrees_wildly(self):
        curve = self._curve(0.40, 1.0)
        text = bc.render_report(
            _aggregate(0.87, curve[0], None, curve, heldout_coverage=0.05),
            sample_ids=["a"], args=_args(),
        )
        assert "UNRESOLVED" in text
        assert "sample too small" in text

    def test_closed_by_magnitude_when_even_a_perfect_certificate_barely_helps(self):
        curve = self._curve(0.40, 1.0)
        aggregate = _aggregate(0.87, curve[0], None, curve, heldout_coverage=0.40)
        aggregate["projections"]["ceiling"]["speedup_vs_static"] = 1.02
        text = bc.render_report(aggregate, sample_ids=["a"], args=_args())
        assert "CLOSED BY MAGNITUDE" in text
        assert "cannot move this bound" in text

    def test_closed_when_the_ceiling_is_below_the_required_rate(self):
        curve = self._curve(0.10, 1.0)
        text = bc.render_report(_aggregate(0.20, curve[0], None, curve), sample_ids=["a"], args=_args())
        assert "CLOSED" in text
        assert "ceiling" in text

    def test_open_but_unreached_when_only_a_lossy_certificate_gets_there(self):
        curve = self._curve(0.10, 1.0)
        text = bc.render_report(_aggregate(0.87, curve[0], None, curve), sample_ids=["a"], args=_args())
        assert "OPEN BUT NOT YET REACHED" in text

    def test_closed_when_even_a_perfect_certificate_cannot_pay(self):
        curve = self._curve(1.0, 1.0)
        aggregate = _aggregate(1.0, curve[0], None, curve)
        aggregate["timing"]["cached_sparse_pass_seconds"] = 30.0
        text = bc.render_report(aggregate, sample_ids=["a"], args=_args())
        assert "CLOSED" in text
        assert "fires on every round" in text

    def test_reports_the_three_deciding_numbers(self):
        curve = self._curve(0.40, 1.0)
        text = bc.render_report(_aggregate(0.87, curve[0], None, curve), sample_ids=["a"], args=_args())
        assert "87.0%" in text
        assert "40.0%" in text
        assert "32%" in text


class TestParseArgs:
    def test_rejects_gamma_below_two(self):
        with pytest.raises(SystemExit):
            bc.parse_args([
                "--model-path", "m", "--data-path", "d", "--video-root", "v", "--output", "o", "--gamma", "1",
            ])

    def test_defaults_keep_the_idle_gpu_rule(self):
        args = bc.parse_args([
            "--model-path", "m", "--data-path", "d", "--video-root", "v", "--output", "o",
        ])
        assert args.allow_shared_gpu is False
        assert args.min_free_gib == 40.0
