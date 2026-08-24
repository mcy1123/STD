"""Run the Adaptive-K budget controllers over existing verification traces.

This script is CPU-only and never imports or invokes a model or decoder.
Counterfactual acceptance values are explicitly labeled as proxies.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from std_repro.adaptive_k_offline import (  # noqa: E402
    BudgetBounds,
    RoundResult,
    SampleTrace,
    StrategySpec,
    StrategySummary,
    aggregate_summaries,
    reconstruct_proposed,
    replay_strategy,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_HEADINGS = (
    "## 1. Dataset",
    "## 2. Controller definitions",
    "## 3. K distribution",
    "## 4. Acceptance comparison table",
    "## 5. Efficiency comparison",
    "## 6. Pareto plot",
    "## 7. Go/No-Go conclusion",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-dir", type=Path, default=ROOT / "results" / "routing_traces")
    parser.add_argument("--dataset", default="VideoDetailCaption")
    parser.add_argument("--frame-num", type=int, default=128)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "adaptive_k_offline"
    )
    parser.add_argument("--k-min", type=int, default=512)
    parser.add_argument("--k-max", type=int, default=8192)
    parser.add_argument("--static-k", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--rho", type=float, nargs="+", default=[0.80, 0.90, 0.95])
    return parser.parse_args()


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"metadata not found: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"no matching samples in {path}")
    return rows


def _mean_heads(tensor: torch.Tensor, expected_dims: Sequence[int], label: str) -> np.ndarray:
    array = tensor.detach().float().cpu().numpy()
    if array.ndim not in expected_dims:
        raise ValueError(f"{label} has invalid rank {array.ndim}")
    if label == "prefill_scores" and array.ndim == 2:
        array = array.mean(axis=0)
    if label == "round_scores" and array.ndim == 3:
        array = array.mean(axis=1)
    return np.asarray(array, dtype=np.float64)


def load_traces(traces_dir: Path, dataset: str, frame_num: int) -> List[SampleTrace]:
    metadata_path = traces_dir / f"{dataset}_frame{frame_num}.jsonl"
    metadata = _load_jsonl(metadata_path)
    traces: List[SampleTrace] = []
    for meta in metadata:
        if meta.get("dataset", dataset) != dataset or int(meta.get("frame_num", frame_num)) != frame_num:
            continue
        sample_id = str(meta.get("sample_id", ""))
        trace_path = traces_dir / f"{sample_id}.pt"
        if not trace_path.exists():
            raise FileNotFoundError(f"trace not found for {sample_id}: {trace_path}")
        payload = torch.load(trace_path, map_location="cpu", weights_only=True)
        if "prefill_scores" not in payload or "round_scores" not in payload:
            raise ValueError(f"trace {trace_path} is missing attention scores")
        prefill = _mean_heads(payload["prefill_scores"], (1, 2), "prefill_scores")
        rounds = _mean_heads(payload["round_scores"], (2, 3), "round_scores")
        expected_rounds = int(meta.get("decode_rounds", len(meta.get("accept_lengths", []))))
        if rounds.shape[0] != expected_rounds:
            raise ValueError(
                f"round count mismatch for {sample_id}: tensor={rounds.shape[0]} "
                f"metadata={expected_rounds}"
            )
        accepted = meta.get("accept_lengths")
        if accepted is None or len(accepted) != expected_rounds:
            raise ValueError(f"round count mismatch for {sample_id}: accepted lengths")
        query_lens = meta.get("query_lens")
        if query_lens is None and "query_lens" in payload:
            query_lens = payload["query_lens"].tolist()
        if query_lens is None:
            proposed = [int(meta["gamma"])] * expected_rounds
        else:
            if len(query_lens) != expected_rounds:
                raise ValueError(f"round count mismatch for {sample_id}: query lengths")
            proposed = reconstruct_proposed(query_lens)
        visual_len = int(meta["visual_len"])
        if prefill.shape != (visual_len,) or rounds.shape[1] != visual_len:
            raise ValueError(
                f"score width mismatch for {sample_id}: visual_len={visual_len}, "
                f"prefill={prefill.shape}, rounds={rounds.shape}"
            )
        traces.append(
            SampleTrace(
                sample_id=sample_id,
                dataset=dataset,
                visual_len=visual_len,
                recorded_k=int(meta["k"]),
                gamma=int(meta["gamma"]),
                prefill_scores=prefill,
                round_scores=rounds,
                accepted=np.asarray(accepted, dtype=np.float64),
                proposed=np.asarray(proposed, dtype=np.int64),
            )
        )
    if not traces:
        raise ValueError(f"no matching samples for dataset={dataset}, frame_num={frame_num}")
    return traces


def build_strategies(static_k: Sequence[int], rhos: Sequence[float]) -> List[StrategySpec]:
    strategies = [
        StrategySpec(name=f"static_k{k}", kind="static", static_k=int(k)) for k in static_k
    ]
    strategies.extend(
        StrategySpec(name=f"attention_rho{rho:.2f}", kind="attention", rho=float(rho))
        for rho in rhos
    )
    strategies.append(StrategySpec(name="acceptance_feedback", kind="acceptance"))
    strategies.extend(
        StrategySpec(name=f"hybrid_rho{rho:.2f}", kind="hybrid", rho=float(rho))
        for rho in rhos
    )
    return strategies


def simulate(
    traces: Sequence[SampleTrace],
    strategies: Sequence[StrategySpec],
    k_min: int,
    k_max: int,
) -> tuple:
    rows: List[RoundResult] = []
    for trace in traces:
        bounds = BudgetBounds(k_min=k_min, k_max=k_max, visual_len=trace.visual_len)
        for spec in strategies:
            sample_rows, _ = replay_strategy(trace, spec, bounds)
            rows.extend(sample_rows)
    summaries = aggregate_summaries(rows, strategies)
    observed_rows = [
        RoundResult.minimal(
            trace.sample_id,
            "recorded_static",
            round_id,
            k=trace.recorded_k,
            accepted=float(trace.accepted[round_id]),
            proposed=int(trace.proposed[round_id]),
        )
        for trace in traces
        for round_id in range(len(trace.accepted))
    ]
    observed = aggregate_summaries(
        observed_rows, [StrategySpec(name="recorded_static", kind="static", static_k=1)]
    )[0]
    observed = replace(observed, evidence="observed")
    return rows, [observed] + summaries


def _write_csv(path: Path, records: Iterable[dict]) -> None:
    records = list(records)
    if not records:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def plot_budget_by_round(rows: Sequence[RoundResult], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    strategies = sorted({row.strategy for row in rows if row.kind != "static"})
    for strategy in strategies:
        selected = [row for row in rows if row.strategy == strategy]
        by_round: Dict[int, List[int]] = {}
        for row in selected:
            by_round.setdefault(row.round_id, []).append(row.k)
        xs = sorted(by_round)
        means = np.asarray([np.mean(by_round[x]) for x in xs])
        stds = np.asarray([np.std(by_round[x]) for x in xs])
        ax.plot(xs, means, linewidth=1.4, label=strategy)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.08)
    ax.set_title("Figure 1: Adaptive visual KV budget by verification round")
    ax.set_xlabel("Verification round ID")
    ax.set_ylabel("Visual K")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_frontier(summaries: Sequence[StrategySummary], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    static = [summary for summary in summaries if summary.kind == "static" and summary.evidence == "proxy"]
    adaptive = [summary for summary in summaries if summary.kind != "static"]
    observed = [summary for summary in summaries if summary.evidence == "observed"]
    if static:
        ordered = sorted(static, key=lambda summary: summary.mean_k)
        ax.plot(
            [summary.mean_k for summary in ordered],
            [summary.proxy_mean_accept for summary in ordered],
            "o-",
            label="Static K (proxy)",
        )
    for summary in adaptive:
        ax.scatter(summary.mean_k, summary.proxy_mean_accept, marker="^", s=45)
        ax.annotate(
            summary.strategy,
            (summary.mean_k, summary.proxy_mean_accept),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6,
        )
    for summary in observed:
        ax.scatter(
            summary.mean_k,
            summary.observed_mean_accept,
            marker="*",
            s=140,
            color="black",
            label="Recorded K (observed)",
        )
    if adaptive:
        ax.scatter([], [], marker="^", s=45, label="Adaptive (proxy)")
    ax.set_title("Figure 2: Acceptance proxy versus average visual K")
    ax.set_xlabel("Average visual K")
    ax.set_ylabel("Mean accepted length")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _go_no_go(summaries: Sequence[StrategySummary]) -> Dict[str, str]:
    adaptive = [summary for summary in summaries if summary.kind != "static"]
    varying = [
        summary
        for summary in adaptive
        if summary.coefficient_of_variation >= 0.05 or summary.change_fraction >= 0.10
    ]
    q1 = (
        f"YES — {len(varying)}/{len(adaptive)} adaptive configurations show material K variation."
        if varying
        else "NO — adaptive K is effectively constant under the specified thresholds."
    )
    q2 = (
        "INSUFFICIENT EVIDENCE / NO-GO — the available traces contain measured acceptance only "
        "at recorded visual K≈996; all multi-K and adaptive acceptance values are proxies."
    )
    if adaptive:
        best = min(adaptive, key=lambda summary: summary.mean_k)
        saving = 100.0 * (1.0 - best.mean_k / 4096.0)
        q3 = (
            f"POSSIBLE BUT UNVERIFIED — {best.strategy} changes estimated draft attention cost by "
            f"{saving:+.1f}% versus static K=4096. Controller/collection overhead and dense verification "
            "must be measured in a later authorized runtime experiment before claiming wall-clock speedup."
        )
    else:
        q3 = "UNAVAILABLE — no adaptive strategies were evaluated."
    return {"q1": q1, "q2": q2, "q3": q3, "decision": "NO-GO"}


def write_report(
    path: Path,
    traces: Sequence[SampleTrace],
    summaries: Sequence[StrategySummary],
    k_min: int,
    k_max: int,
    rhos: Sequence[float],
) -> None:
    k_rows = [
        (
            summary.strategy,
            f"{summary.mean_k:.1f}",
            f"{summary.median_k:.1f}",
            summary.min_k,
            summary.max_k,
            f"{summary.coefficient_of_variation:.3f}",
            f"{summary.change_fraction:.3f}",
        )
        for summary in summaries
    ]
    acceptance_rows = [
        (
            summary.strategy,
            summary.evidence,
            f"{summary.observed_mean_accept:.3f}" if summary.evidence == "observed" else "—",
            f"{summary.observed_accept_rate:.3f}" if summary.evidence == "observed" else "—",
            f"{summary.proxy_mean_accept:.3f}" if summary.evidence == "proxy" else "—",
            f"{summary.proxy_accept_rate:.3f}" if summary.evidence == "proxy" else "—",
        )
        for summary in summaries
    ]
    efficiency_rows = [
        (
            summary.strategy,
            summary.evidence,
            f"{summary.observed_efficiency:.3f}"
            if summary.evidence == "observed"
            else "—",
            f"{summary.proxy_efficiency:.3f}" if summary.evidence == "proxy" else "—",
        )
        for summary in summaries
    ]
    decision = _go_no_go(summaries)
    total_rounds = sum(len(trace.accepted) for trace in traces)
    recorded_ks = sorted({trace.recorded_k for trace in traces})
    lines = [
        "# Adaptive-K Offline Budget Simulation",
        "",
        REPORT_HEADINGS[0],
        "",
        f"- Dataset: `{traces[0].dataset}`",
        f"- Samples: {len(traces)}",
        f"- Verification rounds: {total_rounds}",
        "- Frames: 128; maximum generated tokens: 256; gamma: 9 for the primary run",
        f"- Recorded visual K values: {recorded_ks}",
        "- New decoding runs: 0",
        "",
        "Only the recorded K trajectory has measured acceptance. Counterfactual rows are explicitly labeled proxy.",
        "",
        REPORT_HEADINGS[1],
        "",
        f"- Attention Mass Top-P: smallest K covering rho in {list(rhos)}, clamped to [{k_min}, {k_max}].",
        "- Acceptance Feedback: double K below 0.5 acceptance rate; halve K above 0.8; otherwise retain K.",
        "- Hybrid: max(Attention Mass Top-P K, Acceptance Feedback K).",
        "- Causality: round t feedback controls round t+1; round zero starts at recorded K.",
        "",
        REPORT_HEADINGS[2],
        "",
        _format_table(
            ["strategy", "mean K", "median K", "min K", "max K", "CV", "change fraction"],
            k_rows,
        ),
        "",
        REPORT_HEADINGS[3],
        "",
        _format_table(
            [
                "strategy",
                "evidence",
                "observed mean",
                "observed rate",
                "proxy mean",
                "proxy rate",
            ],
            acceptance_rows,
        ),
        "",
        REPORT_HEADINGS[4],
        "",
        "Efficiency is accepted tokens per `K/1024` budget unit; it is not wall-clock throughput.",
        "",
        _format_table(
            ["strategy", "evidence", "observed efficiency", "proxy efficiency"], efficiency_rows
        ),
        "",
        REPORT_HEADINGS[5],
        "",
        "![Adaptive K by round](figure1_adaptive_k_by_round.png)",
        "",
        "![Acceptance versus average K](figure2_acceptance_vs_average_k.png)",
        "",
        "The plotted static/adaptive frontier is a sensitivity proxy, not a measured Pareto frontier.",
        "",
        REPORT_HEADINGS[6],
        "",
        f"- Q1 — visual budget variation: {decision['q1']}",
        f"- Q2 — adaptive versus static: {decision['q2']}",
        f"- Q3 — wall-clock potential: {decision['q3']}",
        f"- Final decision: **{decision['decision']}**. Stop Adaptive-K runtime implementation until multi-K measured evidence is authorized.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    traces = load_traces(args.traces_dir, args.dataset, args.frame_num)
    strategies = build_strategies(args.static_k, args.rho)
    rows, summaries = simulate(traces, strategies, args.k_min, args.k_max)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(args.output_dir / "rounds.csv", (asdict(row) for row in rows))
    _write_csv(args.output_dir / "summary.csv", (asdict(summary) for summary in summaries))
    plot_budget_by_round(rows, args.output_dir / "figure1_adaptive_k_by_round.png")
    plot_frontier(summaries, args.output_dir / "figure2_acceptance_vs_average_k.png")
    write_report(args.output_dir / "report.md", traces, summaries, args.k_min, args.k_max, args.rho)

    manifest = {
        "dataset": args.dataset,
        "frame_num": args.frame_num,
        "sample_count": len(traces),
        "verification_rounds": sum(len(trace.accepted) for trace in traces),
        "new_decoding_runs": 0,
        "traces_dir": str(args.traces_dir.resolve()),
        "metadata": str(
            (args.traces_dir / f"{args.dataset}_frame{args.frame_num}.jsonl").resolve()
        ),
        "k_min": args.k_min,
        "k_max": args.k_max,
        "static_k": args.static_k,
        "rho": args.rho,
        "evidence_boundary": (
            "Acceptance is observed only for the recorded static trajectory; all candidate "
            "multi-K and adaptive acceptance values are attention-mass proxies."
        ),
        "controller_fallbacks": {
            summary.strategy: summary.controller_fallbacks for summary in summaries
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Adaptive-K offline simulation complete: samples={len(traces)}, "
        f"rounds={manifest['verification_rounds']}, output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
