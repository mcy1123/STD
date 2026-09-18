"""Unit tests for the HSD three-level feasibility gate.

The gate exists to decide, without new GPU work, whether inserting a middle
verification level can ever pay for itself.  These tests pin the two cost
inequalities and the funnel-collapse detector, because a silently wrong gate
would either resurrect an already-falsified design or kill a viable one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts" / "analysis") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import hsd_feasibility as hf  # noqa: E402


def trial(
    method,
    *,
    decode=1.0,
    rounds=10.0,
    accept_len=8.0,
    generated=80.0,
    accepted=80.0,
    proposed=90.0,
    draft=0.5,
    verify=0.2,
    bonus=0.1,
    middle=None,
    phase="measure",
):
    row = {
        "kind": "trial",
        "phase": phase,
        "method": method,
        "decoding_time": decode,
        "decode_rounds": rounds,
        "mean_accept_length": accept_len,
        "generate_len": generated,
        "accepted_draft_tokens": accepted,
        "proposed_draft_tokens": proposed,
        "draft_time": draft,
        "verify_time": verify,
        "bonus_time": bonus,
        "sparse_cache_time": 0.0,
    }
    if middle is not None:
        row["hsd_stats"] = {"sparse_time": middle}
    return row


def write_jsonl(path: Path, rows) -> Path:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


class TestLoadTrials:
    def test_keeps_only_requested_phase_and_trial_rows(self, tmp_path):
        path = write_jsonl(
            tmp_path / "r.jsonl",
            [
                {"kind": "manifest", "phase": "measure", "method": "ar"},
                trial("ar", phase="warmup"),
                trial("ar", phase="measure"),
            ],
        )
        loaded = hf.load_trials(path, phase="measure")
        assert len(loaded) == 1
        assert loaded[0]["method"] == "ar"

    def test_phase_any_keeps_both_phases(self, tmp_path):
        path = write_jsonl(tmp_path / "r.jsonl", [trial("ar", phase="warmup"), trial("ar", phase="measure")])
        assert len(hf.load_trials(path, phase="any")) == 2

    def test_empty_selection_raises(self, tmp_path):
        path = write_jsonl(tmp_path / "r.jsonl", [trial("ar", phase="warmup")])
        with pytest.raises(ValueError):
            hf.load_trials(path, phase="measure")


class TestSummarize:
    def test_per_round_and_per_token_rates(self):
        rows = [trial("static", decode=1.2, rounds=4.0, generated=40.0, draft=0.8, verify=0.3, bonus=0.1)]
        summary = hf.summarize(rows)
        assert summary["ms_per_round"] == pytest.approx(300.0)
        assert summary["ms_per_token"] == pytest.approx(30.0)
        assert summary["draft_time_ms_per_round"] == pytest.approx(200.0)
        assert summary["verify_time_ms_per_round"] == pytest.approx(75.0)

    def test_averages_across_repeats(self):
        rows = [trial("static", decode=1.0), trial("static", decode=3.0)]
        assert hf.summarize(rows)["decode_s"] == pytest.approx(2.0)

    def test_zero_rounds_does_not_divide_by_zero(self):
        summary = hf.summarize([trial("ar", decode=1.0, rounds=0.0, generated=0.0)])
        assert summary["ms_per_round"] == 0.0
        assert summary["ms_per_token"] == 0.0


class TestMiddleLayerRecovery:
    def test_recovers_nested_sparse_time(self):
        rows = [trial("hsd", middle=0.9), trial("hsd", middle=1.1)]
        assert hf.middle_layer_seconds(rows) == pytest.approx([0.9, 1.1])

    def test_absent_nested_stats_yields_nothing(self):
        assert hf.middle_layer_seconds([trial("static")]) == []

    def test_summary_exposes_middle_ms_per_round(self):
        rows = [trial("hsd", middle=1.0, rounds=10.0)]
        assert hf.summarize(rows)["middle_ms_per_round"] == pytest.approx(100.0)


class TestFunnelCollapse:
    def test_identical_accept_and_rounds_is_degenerate(self):
        baseline = hf.summarize([trial("static", accept_len=7.598, rounds=7.667, proposed=65.3)])
        candidate = hf.summarize([trial("hsd", accept_len=7.598, rounds=7.667, proposed=73.3)])
        result = hf.funnel_collapse_check(baseline, candidate)
        assert result["degenerate"] is True
        assert "DEGENERATE" in result["verdict"]
        assert result["proposed_ratio"] == pytest.approx(73.3 / 65.3)

    def test_changed_accept_length_is_not_degenerate(self):
        baseline = hf.summarize([trial("static", accept_len=5.0)])
        candidate = hf.summarize([trial("hsd", accept_len=6.5)])
        result = hf.funnel_collapse_check(baseline, candidate)
        assert result["degenerate"] is False
        assert "INCONCLUSIVE" in result["verdict"]


class TestTruncationGate:
    def test_weight_bound_regime_has_no_truncation_value(self):
        # A batched dense pass costs about one AR step, so extra verified tokens are nearly free.
        ar = hf.summarize([trial("ar", decode=1.6, generated=64.0)])
        baseline = hf.summarize([trial("static", verify=0.201, rounds=7.667)])
        gate = hf.truncation_gate(ar, baseline, gamma=9, middle_ms_per_round=100.0)
        assert gate["marginal_ms_per_token"] < 1.0
        assert gate["feasible"] is False
        assert "IMPOSSIBLE" in gate["verdict"]

    def test_long_context_truncation_is_larger_but_can_still_lose(self):
        ar = hf.summarize([trial("ar", decode=6.439, generated=128.0)])
        baseline = hf.summarize([trial("static", verify=1.841, rounds=16.0)])
        gate = hf.truncation_gate(ar, baseline, gamma=9, middle_ms_per_round=111.878)
        assert gate["marginal_ms_per_token"] == pytest.approx((115.06 - 50.31) / 9, rel=1e-3)
        assert gate["feasible"] is False

    def test_negative_marginal_is_clamped(self):
        ar = hf.summarize([trial("ar", decode=2.0, generated=10.0)])
        baseline = hf.summarize([trial("static", verify=0.05, rounds=10.0)])
        gate = hf.truncation_gate(ar, baseline, gamma=9, middle_ms_per_round=1.0)
        assert gate["marginal_ms_per_token"] == 0.0
        assert gate["max_saving_ms_per_round"] == 0.0

    def test_small_middle_pass_can_be_feasible(self):
        ar = hf.summarize([trial("ar", decode=6.439, generated=128.0)])
        baseline = hf.summarize([trial("static", verify=1.841, rounds=16.0)])
        gate = hf.truncation_gate(ar, baseline, gamma=9, middle_ms_per_round=10.0)
        assert gate["feasible"] is True


class TestCertificationGate:
    def test_required_rate_is_cost_over_saving(self):
        baseline = hf.summarize([trial("static", verify=1.0, bonus=0.2, rounds=10.0)])
        gate = hf.certification_gate(baseline, middle_ms_per_round=60.0)
        # saving = 100 + 20 = 120 ms/round
        assert gate["saving_ms_per_round"] == pytest.approx(120.0)
        assert gate["required_skip_rate"] == pytest.approx(0.5)
        assert gate["feasible"] is True
        assert ">50%" in gate["verdict"]

    def test_impossible_when_middle_costs_more_than_the_saving(self):
        baseline = hf.summarize([trial("static", verify=0.201, bonus=0.157, rounds=7.667)])
        gate = hf.certification_gate(baseline, middle_ms_per_round=111.878)
        assert gate["required_skip_rate"] > 1.0
        assert gate["feasible"] is False
        assert "IMPOSSIBLE" in gate["verdict"]


class TestCertificationSensitivity:
    def test_rows_include_measured_cost_and_are_sorted(self):
        baseline = hf.summarize([trial("static", verify=1.841, bonus=0.363, rounds=16.0)])
        rows = hf.certification_sensitivity(baseline, middle_ms_per_round=111.878)
        costs = [row["middle_ms_per_round"] for row in rows]
        assert 111.878 in costs
        assert costs == sorted(costs)
        assert all(row["required_skip_rate"] > 0 for row in rows)

    def test_cheaper_middle_pass_requires_lower_rate(self):
        baseline = hf.summarize([trial("static", verify=1.841, bonus=0.363, rounds=16.0)])
        rows = hf.certification_sensitivity(baseline, middle_ms_per_round=111.878)
        rates = [row["required_skip_rate"] for row in rows]
        assert rates == sorted(rates)

    def test_zero_saving_yields_no_rows(self):
        baseline = hf.summarize([trial("static", verify=0.0, bonus=0.0, rounds=10.0)])
        assert hf.certification_sensitivity(baseline, middle_ms_per_round=10.0) == []


class TestRenderAndCli:
    def test_report_contains_all_gates_and_stop_verdict(self):
        ar = hf.summarize([trial("ar", decode=1.64, generated=64.0)])
        baseline = hf.summarize([trial("static", accept_len=7.598, rounds=7.667, verify=0.201, bonus=0.157)])
        candidate = hf.summarize(
            [trial("hsd", accept_len=7.598, rounds=7.667, verify=0.203, bonus=0.164, middle=0.858)]
        )
        text = hf.render_report({"ar": ar, "static": baseline, "hsd": candidate}, "static", "hsd", "ar", 9)
        assert "Gate 1" in text and "Gate 2" in text and "Gate 3" in text
        assert "DEGENERATE" in text
        assert "**STOP.**" in text

    def test_hypothetical_run_needs_an_explicit_cost(self):
        summaries = {"ar": hf.summarize([trial("ar")]), "static": hf.summarize([trial("static")])}
        with pytest.raises(ValueError):
            hf.render_report(summaries, "static", "static", "ar", 9)

    def test_hypothetical_run_marks_gate_one_not_applicable(self):
        ar = hf.summarize([trial("ar", decode=6.439, generated=128.0)])
        baseline = hf.summarize([trial("static", verify=1.841, bonus=0.363, rounds=16.0)])
        text = hf.render_report({"ar": ar, "static": baseline}, "static", "static", "ar", 9, 111.878)
        assert "not applicable" in text
        assert "CONDITIONAL GO" in text

    def test_main_runs_end_to_end(self, tmp_path, capsys):
        path = write_jsonl(
            tmp_path / "r.jsonl",
            [
                trial("ar", decode=1.64, generated=64.0),
                trial("static", accept_len=7.598, rounds=7.667, verify=0.201, bonus=0.157),
                trial("hsd", accept_len=7.598, rounds=7.667, verify=0.203, bonus=0.164, middle=0.858),
            ],
        )
        out = tmp_path / "nested" / "gate.md"
        code = hf.main(["--jsonl", str(path), "--baseline", "static", "--candidate", "hsd", "--ar", "ar",
                        "--gamma", "9", "--out", str(out)])
        assert code == 0
        assert out.exists()
        assert "HSD three-level feasibility gate" in capsys.readouterr().out

    def test_main_reports_missing_method(self, tmp_path):
        path = write_jsonl(tmp_path / "r.jsonl", [trial("ar"), trial("static")])
        with pytest.raises(SystemExit):
            hf.main(["--jsonl", str(path), "--baseline", "static", "--candidate", "hsd"])

    def test_main_json_payload_is_serializable(self, tmp_path, capsys):
        path = write_jsonl(
            tmp_path / "r.jsonl",
            [
                trial("ar", decode=6.439, generated=128.0),
                trial("static", verify=1.841, bonus=0.363, rounds=16.0),
            ],
        )
        code = hf.main(["--jsonl", str(path), "--baseline", "static", "--candidate", "static",
                        "--ar", "ar", "--middle-ms-per-round", "111.878", "--json"])
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["hypothetical"] is True
        assert payload["certification"]["required_skip_rate"] == pytest.approx(111.878 / 137.752, rel=1e-3)
