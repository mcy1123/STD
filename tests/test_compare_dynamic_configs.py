"""Unit tests for scripts/analysis/compare_dynamic_configs.py.

The E1/E6 matrix is judged by comparing several single-variable runs, so the
comparator has to reduce each run to the same paired, headroom-stratified view
the summarizer uses -- including the per-round cost breakdown, because "select
better" and "swap cheaper" are two factors of one product.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts" / "analysis") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import compare_dynamic_configs as cmp  # noqa: E402


def _write_run(path, *, samples, collector="v2", refresh="incremental",
               timing_valid=True, refresh_ms=20.0, mismatches=0, checks=3):
    """Emit a JSONL shaped exactly like benchmark_a100_dynamic.py."""
    records = [
        {"kind": "manifest", "timing_valid": timing_valid, "git_head": "deadbeef"},
        {"kind": "input", "sample_id": samples[0]["sample_id"]},
    ]
    for spec in samples:
        sid = spec["sample_id"]
        records.append({"kind": "trial", "sample_id": sid, "phase": "warmup", "repeat": -1,
                        "method": "dynamic_v2", "acceptance_rate": 0.0, "decode_rounds": 0,
                        "dynamic_stats": {"collector_version": "v1", "refresh_mode": "full",
                                          "consistency_checks": 3, "consistency_mismatches": 0,
                                          "per_round": [{"refresh_time_ms": 999.0}]}})
        for method, accept, rounds, decode in (
            ("static", spec["static"], spec["rounds"], spec["static_decode"]),
            ("dynamic_v2", spec["dynamic"], spec["rounds"], spec["dynamic_decode"]),
        ):
            rec = {"kind": "trial", "sample_id": sid, "phase": "measure", "repeat": 0,
                   "method": method, "acceptance_rate": accept,
                   "mean_accept_length": accept * 10, "decode_rounds": rounds,
                   "decoding_time": decode, "draft_time": 1.0, "verify_time": 0.5}
            if method == "dynamic_v2":
                rec["dynamic_stats"] = {
                    "collector_version": collector, "refresh_mode": refresh,
                    "consistency_checks": checks, "consistency_mismatches": mismatches,
                    "total_selection_update_time_ms": 100.0,
                    "per_round": [{"refresh_time_ms": refresh_ms, "changed_ratio": 0.4},
                                  {"refresh_time_ms": refresh_ms, "changed_ratio": 0.5}],
                }
            records.append(rec)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


SAMPLES = [
    {"sample_id": "a", "static": 0.50, "dynamic": 0.70, "rounds": 10,
     "static_decode": 5.0, "dynamic_decode": 6.0},
    {"sample_id": "b", "static": 0.80, "dynamic": 0.75, "rounds": 10,
     "static_decode": 5.0, "dynamic_decode": 6.0},
    {"sample_id": "c", "static": 0.95, "dynamic": 0.80, "rounds": 10,
     "static_decode": 5.0, "dynamic_decode": 6.0},
]


class TestLoadRun:
    def test_headroom_sign_flip_and_correlation(self, tmp_path):
        run = cmp.load_run(_write_run(tmp_path / "r.jsonl", samples=SAMPLES))
        deltas = {r["sample_id"]: r["delta_accept"] for r in run["paired"]}
        assert deltas["a"] == pytest.approx(+0.20)
        assert deltas["b"] == pytest.approx(-0.05)
        assert deltas["c"] == pytest.approx(-0.15)
        # Also verifies the mean does not hide the spread the strata expose.
        assert run["mean_delta"] == pytest.approx(0.0, abs=1e-9)
        assert run["corr"] < -0.5

    def test_dynamic_stats_come_from_measure_not_warmup(self, tmp_path):
        """Warmup trials carry full/v1 stats; the measure trial must win."""
        run = cmp.load_run(_write_run(tmp_path / "r.jsonl", samples=SAMPLES,
                                      collector="v3", refresh="incremental", refresh_ms=22.0))
        assert run["collector"] == "v3"
        assert run["refresh_mode"] == "incremental"
        assert run["refresh_ms_per_round"] == pytest.approx(22.0)

    def test_per_round_cost_uses_round_counts(self, tmp_path):
        run = cmp.load_run(_write_run(tmp_path / "r.jsonl", samples=SAMPLES, refresh_ms=20.0))
        # 30 rounds total; static decode is 5.0 s over 10 rounds per sample.
        assert run["static_decode_ms_per_round"] == pytest.approx(500.0, abs=0.5)
        assert run["dynamic_decode_ms_per_round"] == pytest.approx(600.0, abs=0.5)
        assert run["toll_ms_per_round"] == pytest.approx(100.0, abs=0.5)
        # selection time is already in ms and must not be scaled twice.
        assert run["selection_update_ms_per_round"] == pytest.approx(100.0 * 3 / 30, abs=1e-6)

    def test_t1_counts_surface_mismatches(self, tmp_path):
        run = cmp.load_run(_write_run(tmp_path / "r.jsonl", samples=SAMPLES, mismatches=1))
        assert run["consistency_checks"] == 9
        assert run["consistency_mismatches"] == 3


class TestStrata:
    def test_low_headroom_bin_comes_first(self, tmp_path):
        run = cmp.load_run(_write_run(tmp_path / "r.jsonl", samples=SAMPLES))
        bins = cmp.strata(run)
        assert bins[0][2] == pytest.approx(0.50)   # lowest static accept
        assert bins[-1][3] == pytest.approx(0.95)  # highest static accept
        assert bins[0][4] == pytest.approx(+0.20)  # the winning end


class TestRender:
    def test_renders_every_run_and_intersects_samples(self, tmp_path):
        a = cmp.load_run(_write_run(tmp_path / "a.jsonl", samples=SAMPLES))
        b = cmp.load_run(_write_run(tmp_path / "b.jsonl", samples=SAMPLES[:2]))
        text = cmp.render({"E1": a, "E6": b})
        assert "| E1 |" in text and "| E6 |" in text
        assert "Per-sample d accept" in text
        # sample "c" exists only in E1, so the paired table must omit it.
        pane = text.split("## Per-sample")[1].split("## Headroom")[0]
        assert "| c |" not in pane
        assert "| a |" in pane and "| b |" in pane


class TestCli:
    def test_writes_markdown_output(self, tmp_path):
        run = _write_run(tmp_path / "a.jsonl", samples=SAMPLES)
        out = tmp_path / "cmp.md"
        assert cmp.main([f"E1={run}", "--output", str(out)]) == 0
        assert "dynamic-routing configuration comparison" in out.read_text(encoding="utf-8")
