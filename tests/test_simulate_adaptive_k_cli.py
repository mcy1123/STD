from __future__ import annotations

import csv
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from std_repro.adaptive_k_offline import StrategySummary


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analysis" / "simulate_adaptive_k.py"
SCRIPT_GLOBALS = runpy.run_path(str(SCRIPT))
matched_static_cost_reduction = SCRIPT_GLOBALS["matched_static_cost_reduction"]
frontier_annotation_style = SCRIPT_GLOBALS["frontier_annotation_style"]


def _write_trace_set(base: Path, *, corrupt_round_count: bool = False) -> None:
    metadata = base / "VideoDetailCaption_frame128.jsonl"
    rows = []
    for sample_number in range(2):
        sample_id = f"sample-{sample_number}"
        rows.append(
            {
                "dataset": "VideoDetailCaption",
                "sample_id": sample_id,
                "frame_num": 128,
                "visual_len": 4,
                "k": 2,
                "gamma": 2,
                "max_new_tokens": 4,
                "decode_rounds": 2,
                "accept_lengths": [1, 2],
                "query_lens": [2, 3],
            }
        )
        round_scores = torch.tensor(
            [
                [[0.1, 0.1, 0.2, 0.6]],
                [[0.5, 0.3, 0.1, 0.1]],
            ],
            dtype=torch.float32,
        )
        if corrupt_round_count and sample_number == 0:
            round_scores = round_scores[:1]
        torch.save(
            {
                "prefill_scores": torch.tensor([[0.6, 0.3, 0.1, 0.0]]),
                "round_scores": round_scores,
                "query_lens": torch.tensor([2, 3]),
            },
            base / f"{sample_id}.pt",
        )
    metadata.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _run_cli(traces_dir: Path, output_dir: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--traces-dir",
            str(traces_dir),
            "--dataset",
            "VideoDetailCaption",
            "--frame-num",
            "128",
            "--output-dir",
            str(output_dir),
            "--k-min",
            "1",
            "--k-max",
            "4",
            "--static-k",
            "1",
            "2",
            "3",
            "4",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_writes_evidence_labeled_report_and_figures(tmp_path: Path) -> None:
    traces_dir = tmp_path / "traces"
    output_dir = tmp_path / "output"
    traces_dir.mkdir()
    _write_trace_set(traces_dir)

    result = _run_cli(traces_dir, output_dir)

    assert result.returncode == 0, result.stderr
    expected = {
        "rounds.csv",
        "summary.csv",
        "manifest.json",
        "report.md",
        "figure1_adaptive_k_by_round.png",
        "figure2_acceptance_vs_average_k.png",
    }
    assert {path.name for path in output_dir.iterdir()} == expected

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    with (output_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    report = (output_dir / "report.md").read_text(encoding="utf-8")

    assert manifest["new_decoding_runs"] == 0
    assert manifest["sample_count"] == 2
    assert "proxy" in {row["evidence"] for row in summary_rows}
    assert "observed" in {row["evidence"] for row in summary_rows}
    assert report.count("## ") >= 7
    for filename in (
        "figure1_adaptive_k_by_round.png",
        "figure2_acceptance_vs_average_k.png",
    ):
        assert (output_dir / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_cli_rejects_trace_metadata_round_count_mismatch(tmp_path: Path) -> None:
    traces_dir = tmp_path / "traces"
    output_dir = tmp_path / "output"
    traces_dir.mkdir()
    _write_trace_set(traces_dir, corrupt_round_count=True)

    result = _run_cli(traces_dir, output_dir)

    assert result.returncode != 0
    assert "round count mismatch" in result.stderr.lower()


def _summary(name: str, kind: str, mean_k: float, proxy_accept: float) -> StrategySummary:
    return StrategySummary(
        strategy=name,
        kind=kind,
        evidence="proxy",
        rounds=1,
        mean_k=mean_k,
        median_k=mean_k,
        min_k=int(mean_k),
        max_k=int(mean_k),
        coefficient_of_variation=0.0,
        iqr_k=0.0,
        change_fraction=0.0,
        observed_mean_accept=4.0,
        observed_accept_rate=0.5,
        proxy_mean_accept=proxy_accept,
        proxy_accept_rate=proxy_accept / 9.0,
        observed_efficiency=1.0,
        proxy_efficiency=1.0,
        controller_fallbacks=0,
    )


def test_matched_cost_reduction_uses_acceptance_comparable_static_point() -> None:
    summaries = [
        _summary("static_k4096", "static", 4096.0, 6.20),
        _summary("static_k8192", "static", 8192.0, 6.71),
        _summary("acceptance_feedback", "acceptance", 4608.0, 6.07),
        _summary("attention_rho0.80", "attention", 5429.0, 6.67),
    ]

    adaptive, static, reduction = matched_static_cost_reduction(summaries)

    assert adaptive.strategy == "attention_rho0.80"
    assert static.strategy == "static_k8192"
    assert reduction == pytest.approx(1.0 - 5429.0 / 8192.0)


def test_frontier_annotations_use_unique_short_labels_and_cluster_offsets() -> None:
    strategies = [
        "attention_rho0.80",
        "attention_rho0.90",
        "attention_rho0.95",
        "acceptance_feedback",
        "hybrid_rho0.80",
        "hybrid_rho0.90",
        "hybrid_rho0.95",
    ]

    styles = [frontier_annotation_style(strategy) for strategy in strategies]

    assert len({label for label, _ in styles}) == len(strategies)
    assert len({offset for _, offset in styles[1:]}) == len(styles[1:])
