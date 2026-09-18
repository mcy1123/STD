"""Unit tests for the H_headroom stratification in summarize_a100_dynamic.py.

The aggregate mean hid a sign flip on the 2026-09-09 A100 run
(+0.044 / -0.102 / -0.126 over three samples), so the summarizer must always
emit the per-sample pairing and the headroom strata.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import summarize_a100_dynamic as summ  # noqa: E402


def _groups(static_dynamic):
    groups = defaultdict(list)
    for sample, (static_acc, dyn_acc) in static_dynamic.items():
        groups[sample, "static"].append({"acceptance_rate": static_acc})
        groups[sample, "dynamic_v2"].append({"acceptance_rate": dyn_acc})
    return groups


class TestHeadroomSummary:
    def test_reproduces_the_recorded_sign_flip(self):
        pairs = {
            "050-1": (0.623, 0.667),
            "496-3": (0.864, 0.762),
            "717-1": (0.983, 0.857),
        }
        out = summ.headroom_summary(_groups(pairs), list(pairs), ["ar", "static", "dynamic_v2"])

        deltas = {p["sample_id"]: p["delta_accept"] for p in out["paired"]}
        assert deltas["050-1"] == pytest.approx(0.044, abs=1e-3)
        assert deltas["496-3"] == pytest.approx(-0.102, abs=1e-3)
        assert deltas["717-1"] == pytest.approx(-0.126, abs=1e-3)

    def test_strata_are_ordered_by_static_headroom(self):
        pairs = {
            "low": (0.30, 0.60),
            "mid": (0.60, 0.62),
            "high": (0.95, 0.70),
            "higher": (0.99, 0.80),
            "highest": (1.00, 0.90),
            "top": (1.00, 0.95),
        }
        out = summ.headroom_summary(_groups(pairs), list(pairs), ["static", "dynamic_v2"])

        strata = out["strata"]
        assert [s["n"] for s in strata] == [2, 2, 2]
        # Gain must shrink as static headroom grows.
        assert strata[0]["mean_delta_accept"] > strata[-1]["mean_delta_accept"]
        assert strata[0]["static_max"] <= strata[-1]["static_min"]

    def test_single_sample_yields_one_stratum_per_method(self):
        pairs = {"only": (0.60, 0.65)}
        out = summ.headroom_summary(_groups(pairs), list(pairs), ["static", "dynamic_v2"])

        assert len(out["strata"]) == 1
        assert out["strata"][0]["n"] == 1
        assert out["strata"][0]["mean_delta_accept"] == pytest.approx(0.05, abs=1e-3)

    def test_multiple_dynamic_methods_are_stratified_separately(self):
        groups = defaultdict(list)
        for sample, acc in (("a", 0.4), ("b", 0.6), ("c", 0.9)):
            groups[sample, "static"].append({"acceptance_rate": acc})
            groups[sample, "dynamic_v2"].append({"acceptance_rate": acc + 0.05})
            groups[sample, "dynamic_v1"].append({"acceptance_rate": acc - 0.05})

        out = summ.headroom_summary(groups, ["a", "b", "c"], ["static", "dynamic_v2", "dynamic_v1"])

        methods = {s["method"] for s in out["strata"]}
        assert methods == {"dynamic_v2", "dynamic_v1"}
        v1 = [s for s in out["strata"] if s["method"] == "dynamic_v1"]
        assert all(s["mean_delta_accept"] == pytest.approx(-0.05, abs=1e-3) for s in v1)


def _write_run(path, *, timing_valid, tmp_path):
    """Minimal but complete run so `summarize` accepts it."""
    rows = [
        {
            "kind": "manifest",
            "sample_ids": ["s1"],
            "timing_valid": timing_valid,
            "git_head": "deadbeef",
            "args": {
                "repeats": 1,
                "frame_num": 128,
                "max_new_tokens": 128,
                "warmup_tokens": 16,
                "gpu": 0,
                "sparse_attn_mode": "gqa_sdpa",
                "gamma": 9,
                "k_plus_text": 1024,
                "verify_fallback": "none",
            },
        }
    ]
    component_keys = list(summ.COMPONENT_KEYS)
    for method in ("ar", "static", "dynamic_v2"):
        trial = {
            "kind": "trial",
            "sample_id": "s1",
            "phase": "measure",
            "repeat": 0,
            "method": method,
            "decoding_time": 10.0,
            "inference_time": 20.0,
            "acceptance_rate": 0.8,
            "peak_memory_gib": 30.0,
        }
        trial.update({k: 1.0 for k in component_keys})
        rows.append(trial)
        if method != "ar":
            rows.append(
                {
                    "kind": "comparison",
                    "sample_id": "s1",
                    "phase": "measure",
                    "repeat": 0,
                    "method": method,
                    "token_equal": True,
                    "mismatch_token_count": 0,
                }
            )
    rows.append({"kind": "complete", "samples": 1, "repeats": 1})
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


class TestTimingValidityPropagation:
    def test_idle_run_reports_valid_timing(self, tmp_path):
        path = _write_run(tmp_path / "a.jsonl", timing_valid=True, tmp_path=tmp_path)
        result = summ.summarize(path)
        assert result["timing_valid"] is True

    def test_shared_gpu_run_reports_invalid_timing(self, tmp_path):
        path = _write_run(tmp_path / "b.jsonl", timing_valid=False, tmp_path=tmp_path)
        result = summ.summarize(path)
        assert result["timing_valid"] is False

    def test_missing_field_defaults_to_valid(self, tmp_path):
        """Runs recorded before the flag existed must keep reporting speedups."""
        path = _write_run(tmp_path / "c.jsonl", timing_valid=True, tmp_path=tmp_path)
        lines = [json.loads(l) for l in path.read_text().splitlines()]
        del lines[0]["timing_valid"]
        path.write_text("\n".join(json.dumps(r) for r in lines) + "\n")

        assert summ.summarize(path)["timing_valid"] is True
