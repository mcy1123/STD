"""CPU-only contracts for the disposable HSD feasibility runner."""
from __future__ import annotations

import runpy
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_a100_hsd_spike.py"


def runner():
    assert SCRIPT.is_file(), "The isolated HSD spike runner has not been implemented"
    return runpy.run_path(str(SCRIPT))


def required_args():
    return ["--model-path", "/target", "--draft-model-path", "/draft",
            "--data-path", "/data", "--video-root", "/videos", "--output", "/run.jsonl"]


def test_cli_defaults_are_a_small_gpu1_only_spike():
    args = runner()["parse_args"](required_args())
    assert (args.gpu, args.frame_num, args.max_new_tokens, args.limit, args.repeats) == (1, 32, 64, 1, 2)
    assert (args.gamma, args.inner_gamma) == (9, 3)


@pytest.mark.parametrize("extra", [["--gpu", "0"], ["--repeats", "0"],
                                  ["--inner-gamma", "10"], ["--max-rss-gib", "0"]])
def test_cli_refuses_unsafe_or_invalid_runs(extra):
    parse_args = runner()["parse_args"]
    with pytest.raises(SystemExit) as error:
        parse_args(required_args() + extra)
    assert error.value.code == 2


def test_idle_guard_ignores_gpu0_but_refuses_a_busy_gpu1():
    guard = runner()["require_idle_gpu"]
    guard("0, uuid0, A100, 78821, 2229, 100\n1, uuid1, A100, 0, 81050, 0")
    for selected in ("1, uuid1, A100, 900, 80150, 0", "1, uuid1, A100, 0, 81050, 1"):
        with pytest.raises(RuntimeError, match="not idle"):
            guard(selected)
    with pytest.raises(RuntimeError, match="not present"):
        guard("0, uuid0, A100, 0, 81050, 0")


def test_method_rotation_does_not_pin_ar_first():
    order = runner()["method_order"]
    assert order(0, 0) == ["ar", "static", "small_dense", "hsd"]
    assert order(0, 1) == ["static", "small_dense", "hsd", "ar"]
    assert order(1, 1) == ["small_dense", "hsd", "ar", "static"]


def test_exact_comparison_counts_a_missing_tail_as_mismatch():
    compare = runner()["compare_tokens"]
    assert compare([1, 2, 3], [1, 2, 3]) == {
        "token_equal": True, "mismatch_token_count": 0, "token_level_agreement": 1.0,
        "first_mismatch_index": None, "reference_token_count": 3, "candidate_token_count": 3,
    }
    unequal = compare([1, 2, 3], [1, 9])
    assert unequal["token_equal"] is False
    assert unequal["mismatch_token_count"] == 2
    assert unequal["token_level_agreement"] == pytest.approx(1 / 3)
    assert unequal["first_mismatch_index"] == 1


def trial_fixture():
    rows = []
    times = {"ar": [(8, 12), (18, 22)], "static": [(4, 6), (9, 11)],
             "small_dense": [(5, 7), (11, 13)], "hsd": [(3, 5), (7, 9)]}
    for method, samples in times.items():
        for sample_id, pair in zip(("a", "b"), samples):
            for repeat, value in enumerate(pair):
                rows.append({"kind": "trial", "phase": "measure", "sample_id": sample_id,
                             "repeat": repeat, "method": method, "decoding_time": value,
                             "inference_time": value + 10, "output_tokens": [1, 2]})
    rows[-1]["output_tokens"] = [1, 3]
    rows.append({"kind": "trial", "phase": "warmup", "decoding_time": 1000})
    return rows


def test_summary_uses_per_sample_medians_and_exposes_unequal_outputs():
    result = runner()["summarize_trials"](trial_fixture(), ["a", "b"], 2)
    assert result["ar"]["decoding_time"] == 30
    assert result["hsd"]["decoding_time"] == 12
    assert result["hsd"]["inference_time"] == 32
    assert result["hsd"]["decode_speedup_vs_ar"] == 2.5
    assert result["hsd"]["decode_speedup_vs_static"] == 1.25
    assert result["hsd"]["exact_trials"] == 3
    assert result["hsd"]["total_trials"] == 4
    assert result["hsd"]["lossless_speedup_eligible"] is False


@pytest.mark.parametrize("duplicate", [False, True])
def test_summary_rejects_missing_or_duplicate_paired_trials(duplicate):
    summarize = runner()["summarize_trials"]
    rows = trial_fixture()
    if duplicate:
        rows.append(dict(rows[0]))
    else:
        rows.pop(0)
    with pytest.raises(ValueError, match="trial"):
        summarize(rows, ["a", "b"], 2)
