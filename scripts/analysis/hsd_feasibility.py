"""Offline feasibility gate for hierarchical (three-level) speculative decoding.

This script exists because the three-level idea ("small draft -> sparse verify ->
dense verify") was already implemented and measured in this repository and came
out negative (``results/a100_hsd_spike_report.md``).  Rather than re-running a
GPU experiment, this tool turns the *already measured* stage costs into the two
inequalities that decide whether inserting a middle level can ever pay for
itself.

Two independent mechanisms can, in principle, let a middle level save work:

  1. **Truncation** -- the middle level predicts the accept prefix, so the dense
     verification block is shortened from ``gamma+1`` to ``n+1`` tokens.
     Saving = ``(gamma - n) * marginal_dense_cost_per_token``.
  2. **Certified skipping** -- the middle level certifies that its own prediction
     equals the dense one, so the dense pass is skipped entirely for that round.
     Saving = one dense pass (+ its bonus step) per skipped round.

Both are compared against the cost of the middle pass itself, which -- unlike a
trained sub-network -- is a full model forward whenever the middle level is
"the target model with sparse attention".  That is the crux: attention sparsity
removes attention work, not the per-layer weight reads that dominate batch-1
decode.

Usage:
  python scripts/analysis/hsd_feasibility.py \
    --jsonl results/a100_hsd_spike_20260909_l3.jsonl \
    --baseline static --candidate hsd --ar ar --gamma 9
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

# Stage timers recorded by scripts/benchmark_std.py / benchmark_a100_hsd_spike.py.
CORE_STAGES = ("draft_time", "verify_time", "bonus_time", "sparse_cache_time")
# The nested spike records its (otherwise uncounted) middle-level sparse pass here.
NESTED_STATS_KEY = "hsd_stats"
NESTED_MIDDLE_KEY = "sparse_time"


def load_trials(path: Path, phase: str = "measure") -> List[Dict[str, Any]]:
    """Load JSONL trial rows, keeping only the requested phase."""
    trials: List[Dict[str, Any]] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("kind") != "trial":
                continue
            if phase != "any" and row.get("phase") != phase:
                continue
            trials.append(row)
    if not trials:
        raise ValueError(f"no trial rows with phase={phase!r} in {path}")
    return trials


def group_by_method(trials: Iterable[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in trials:
        grouped.setdefault(str(row.get("method")), []).append(row)
    return grouped


def _mean(values: Sequence[float]) -> float:
    return float(stats.mean(values)) if values else 0.0


def _numbers(rows: Iterable[Mapping[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def middle_layer_seconds(rows: Iterable[Mapping[str, Any]]) -> List[float]:
    """Recover the nested verify pass time that the flat trial schema drops.

    ``hsd_spike`` accumulates the middle-level sparse pass in
    ``stats['sparse_time']`` and stores it under ``hsd_stats``, but
    ``benchmark_a100_hsd_spike.py`` only promotes a fixed list of stage timers
    to the top level, so the middle pass is invisible in the flat record.  A
    feasibility claim must not silently omit it.
    """
    out: List[float] = []
    for row in rows:
        nested = row.get(NESTED_STATS_KEY)
        if isinstance(nested, Mapping):
            value = nested.get(NESTED_MIDDLE_KEY)
            if isinstance(value, (int, float)):
                out.append(float(value))
    return out


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate one method's measured cost and throughput."""
    rounds = _mean(_numbers(rows, "decode_rounds"))
    generated = _mean(_numbers(rows, "generate_len"))
    decode = _mean(_numbers(rows, "decoding_time"))
    accepted = _mean(_numbers(rows, "accepted_draft_tokens"))
    summary: Dict[str, Any] = {
        "n": len(rows),
        "decode_s": decode,
        "rounds": rounds,
        "accept_len": _mean(_numbers(rows, "mean_accept_length")),
        "accepted_tokens": accepted,
        "generated_tokens": generated,
        "proposed_tokens": _mean(_numbers(rows, "proposed_draft_tokens")),
        "ms_per_token": (decode / generated * 1000.0) if generated else 0.0,
        "ms_per_round": (decode / rounds * 1000.0) if rounds else 0.0,
    }
    for stage in CORE_STAGES:
        total = _mean(_numbers(rows, stage))
        summary[stage] = total
        summary[f"{stage}_ms_per_round"] = (total / rounds * 1000.0) if rounds else 0.0
    middle = middle_layer_seconds(rows)
    summary["middle_s"] = _mean(middle)
    summary["middle_ms_per_round"] = (_mean(middle) / rounds * 1000.0) if rounds else 0.0
    return summary


def funnel_collapse_check(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """Test whether the middle level changed the outcome at all.

    A middle level that only re-orders which candidates reach the dense verifier
    cannot raise the accept length.  If the candidate's accept length and round
    count equal the baseline's, the funnel filtered nothing and the extra level
    is pure overhead.
    """
    accept_delta = candidate["accept_len"] - baseline["accept_len"]
    round_delta = candidate["rounds"] - baseline["rounds"]
    proposed_ratio = (
        candidate["proposed_tokens"] / baseline["proposed_tokens"]
        if baseline["proposed_tokens"]
        else 0.0
    )
    identical = abs(accept_delta) < 1e-9 and abs(round_delta) < 1e-9
    return {
        "accept_len_baseline": baseline["accept_len"],
        "accept_len_candidate": candidate["accept_len"],
        "accept_len_delta": accept_delta,
        "rounds_baseline": baseline["rounds"],
        "rounds_candidate": candidate["rounds"],
        "rounds_delta": round_delta,
        "proposed_ratio": proposed_ratio,
        "degenerate": identical,
        "verdict": (
            "DEGENERATE: middle level filtered nothing; accept length and round count unchanged"
            if identical
            else "INCONCLUSIVE: accept length or round count changed; inspect per-round evidence"
        ),
    }


def truncation_gate(
    ar: Mapping[str, Any],
    baseline: Mapping[str, Any],
    gamma: int,
    middle_ms_per_round: float,
) -> Dict[str, Any]:
    """Can shortening the dense block pay for the middle pass?

    ``marginal`` is the extra dense cost of verifying one more token, estimated
    from the difference between a batched dense verification pass and a single
    autoregressive dense step.  Its sign can be zero: at batch 1 the dense pass
    is dominated by weight reads, so extra verified tokens are nearly free.
    """
    verified_tokens = max(gamma, 1)
    dense_pass = baseline["verify_time_ms_per_round"]
    ar_step = ar["ms_per_token"]
    marginal = (dense_pass - ar_step) / verified_tokens
    if marginal < 0.0:
        marginal = 0.0
    max_saving = marginal * (verified_tokens - 1)
    cost = middle_ms_per_round
    ratio = (max_saving / cost) if cost > 0 else float("inf")
    return {
        "ar_step_ms": ar_step,
        "dense_pass_ms": dense_pass,
        "marginal_ms_per_token": marginal,
        "max_saving_ms_per_round": max_saving,
        "middle_cost_ms_per_round": cost,
        "saving_to_cost": ratio,
        "feasible": max_saving > cost,
        "verdict": (
            "POSSIBLE: truncation can pay for the middle pass"
            if max_saving > cost
            else "IMPOSSIBLE: maximum truncation saving is below the middle-pass cost"
        ),
    }


def certification_gate(
    baseline: Mapping[str, Any],
    middle_ms_per_round: float,
) -> Dict[str, Any]:
    """What skip rate must a certificate reach to pay for the middle pass?

    Skipping the dense level saves one full dense verification pass plus its
    bonus step for that round.  The certificate must therefore hold at least
    ``middle_cost / saving`` of rounds -- a rate above 1.0 means the mechanism
    cannot break even at this configuration, whatever the certificate quality.
    """
    saving = baseline["verify_time_ms_per_round"] + baseline["bonus_time_ms_per_round"]
    cost = middle_ms_per_round
    required = (cost / saving) if saving > 0 else float("inf")
    return {
        "saving_ms_per_round": saving,
        "middle_cost_ms_per_round": cost,
        "required_skip_rate": required,
        "feasible": required < 1.0,
        "verdict": (
            f"NECESSARY: certificates must hold in >{required * 100:.0f}% of rounds"
            if required < 1.0
            else f"IMPOSSIBLE: even a 100% certificate rate is insufficient "
            f"(needs >{required * 100:.0f}%)"
        ),
    }


def optimal_gamma_note(ar: Mapping[str, Any], baseline: Mapping[str, Any], gamma: int) -> Dict[str, Any]:
    """Where does the baseline's round budget actually go?

    This is the diagnostic that matters most: if drafting dominates, no change to
    the verification level can matter, because verification is the cheap part.
    """
    draft = baseline["draft_time_ms_per_round"]
    verify = baseline["verify_time_ms_per_round"]
    bonus = baseline["bonus_time_ms_per_round"]
    total = draft + verify + bonus
    middle = baseline["middle_ms_per_round"]
    return {
        "draft_share": (draft / total) if total else 0.0,
        "verify_share": (verify / total) if total else 0.0,
        "bonus_share": (bonus / total) if total else 0.0,
        "middle_share": (middle / (total + middle)) if (total + middle) else 0.0,
        "draft_passes": gamma,
        "draft_ms_per_pass": (draft / gamma) if gamma else 0.0,
        "ar_step_ms": ar["ms_per_token"],
    }


def certification_sensitivity(
    baseline: Mapping[str, Any],
    middle_ms_per_round: float,
    cost_fractions: Sequence[float] = (0.25, 0.5, 1.0),
) -> List[Dict[str, float]]:
    """Required certificate rate as a function of the middle pass's real cost.

    The measured middle pass in the nested spike is pathologically slow (a
    hand-built additive mask over a compacted cache, no compilation), so its cost
    is an upper bound rather than a property of sparse attention.  This table
    states the break-even the middle layer must beat, whatever its implementation.
    """
    saving = baseline["verify_time_ms_per_round"] + baseline["bonus_time_ms_per_round"]
    rows: List[Dict[str, float]] = []
    if saving <= 0:
        return rows
    candidates = [float(middle_ms_per_round)]
    candidates.extend(float(f) * baseline["verify_time_ms_per_round"] for f in cost_fractions)
    for cost in sorted(set(candidates)):
        rows.append(
            {
                "middle_ms_per_round": cost,
                "required_skip_rate": cost / saving,
                "feasible_at_full_rate": cost < saving,
            }
        )
    return rows


def _fmt(value: float, unit: str = "") -> str:
    return f"{value:.3f}{unit}"


def render_report(
    summaries: Mapping[str, Mapping[str, Any]],
    baseline_name: str,
    candidate_name: str,
    ar_name: str,
    gamma: int,
    middle_override_ms: Optional[float] = None,
) -> str:
    baseline = summaries[baseline_name]
    candidate = summaries[candidate_name]
    ar = summaries[ar_name]
    hypothetical = candidate_name == baseline_name
    if middle_override_ms is not None:
        middle_ms = float(middle_override_ms)
    elif hypothetical:
        raise ValueError("a hypothetical candidate (candidate == baseline) needs --middle-ms-per-round")
    else:
        middle_ms = candidate["middle_ms_per_round"]
    collapse = funnel_collapse_check(baseline, candidate)
    truncation = truncation_gate(ar, baseline, gamma, middle_ms)
    certification = certification_gate(baseline, middle_ms)
    budget = optimal_gamma_note(ar, baseline, gamma)

    lines: List[str] = []
    lines.append("# HSD three-level feasibility gate")
    lines.append("")
    lines.append("## Measured stage costs")
    lines.append("")
    lines.append("| method | n | decode (s) | rounds | accept len | ms/token | draft/round | verify/round | bonus/round | middle/round |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in sorted(summaries):
        row = summaries[name]
        lines.append(
            f"| {name} | {row['n']} | {_fmt(row['decode_s'])} | {_fmt(row['rounds'])} | "
            f"{_fmt(row['accept_len'])} | {_fmt(row['ms_per_token'])} | "
            f"{_fmt(row['draft_time_ms_per_round'])} | {_fmt(row['verify_time_ms_per_round'])} | "
            f"{_fmt(row['bonus_time_ms_per_round'])} | {_fmt(row['middle_ms_per_round'])} |"
        )
    lines.append("")
    lines.append("All figures are milliseconds per round unless the column says otherwise;")
    lines.append("`middle/round` is the nested spike's sparse pass, recovered from `hsd_stats.sparse_time`.")
    lines.append("")
    lines.append("## Gate 1 -- did the middle level filter anything?")
    lines.append("")
    if hypothetical:
        lines.append(f"- not applicable: `{baseline_name}` is used as both baseline and candidate, so this run")
        lines.append("  evaluates a *hypothetical* middle level whose per-round cost was supplied on the command line.")
    else:
        lines.append(f"- baseline `{baseline_name}` accept len **{_fmt(collapse['accept_len_baseline'])}**, "
                     f"candidate `{candidate_name}` accept len **{_fmt(collapse['accept_len_candidate'])}** "
                     f"(delta {collapse['accept_len_delta']:+.3f})")
        lines.append(f"- baseline rounds **{_fmt(collapse['rounds_baseline'])}**, "
                     f"candidate rounds **{_fmt(collapse['rounds_candidate'])}** "
                     f"(delta {collapse['rounds_delta']:+.3f})")
        lines.append(f"- candidate proposed **{collapse['proposed_ratio'] * 100:.1f}%** of the baseline's draft tokens")
        lines.append(f"- **{collapse['verdict']}**")
    lines.append("")
    lines.append("## Gate 2 -- can truncating the dense block pay for the middle pass?")
    lines.append("")
    lines.append(f"- AR single step: {_fmt(truncation['ar_step_ms'], ' ms')}")
    lines.append(f"- dense verification pass ({gamma}+1 tokens): {_fmt(truncation['dense_pass_ms'], ' ms')}")
    lines.append(f"- marginal dense cost per extra verified token: {_fmt(truncation['marginal_ms_per_token'], ' ms')}")
    lines.append(f"- maximum truncation saving: {_fmt(truncation['max_saving_ms_per_round'], ' ms/round')}")
    lines.append(f"- middle pass cost: {_fmt(truncation['middle_cost_ms_per_round'], ' ms/round')}")
    lines.append(f"- **{truncation['verdict']}** (saving/cost = {truncation['saving_to_cost']:.4f})")
    lines.append("")
    lines.append("## Gate 3 -- what certificate rate would skipping the dense level need?")
    lines.append("")
    lines.append(f"- saving when a round skips dense: {_fmt(certification['saving_ms_per_round'], ' ms/round')}")
    lines.append(f"- middle pass cost: {_fmt(certification['middle_cost_ms_per_round'], ' ms/round')}")
    lines.append(f"- **{certification['verdict']}**")
    lines.append("")
    sensitivity = certification_sensitivity(baseline, middle_ms)
    if sensitivity:
        lines.append("### Sensitivity: how cheap must the middle pass be?")
        lines.append("")
        lines.append("| middle pass (ms/round) | required certificate rate | even 100% is enough? |")
        lines.append("|---:|---:|:--:|")
        for row in sensitivity:
            lines.append(
                f"| {_fmt(row['middle_ms_per_round'])} | {row['required_skip_rate'] * 100:.0f}% | "
                f"{'yes' if row['feasible_at_full_rate'] else 'no'} |"
            )
        lines.append("")
        lines.append("The middle pass measured in `hsd_spike` is a hand-built mask over a compacted cache with")
        lines.append("compilation disabled; treat it as an upper bound, not as the cost of sparse attention.")
        lines.append("")
    lines.append("## Where the baseline's round budget goes")
    lines.append("")
    lines.append(f"- draft: {budget['draft_share'] * 100:.1f}% ({_fmt(budget['draft_ms_per_pass'], ' ms')} x {budget['draft_passes']} passes)")
    lines.append(f"- verify: {budget['verify_share'] * 100:.1f}%")
    lines.append(f"- bonus: {budget['bonus_share'] * 100:.1f}%")
    if budget["middle_share"]:
        lines.append(f"- middle level (candidate): {budget['middle_share'] * 100:.1f}% of the candidate round")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    gates_open = truncation["feasible"] or certification["feasible"]
    if hypothetical:
        if gates_open:
            lines.append("**CONDITIONAL GO.** A hypothetical middle level at the supplied cost has an open gate:")
            lines.append("validate the mechanism it relies on before building an engine.")
        else:
            lines.append("**STOP.** No saving mechanism can pay for a middle pass of the supplied cost at this")
            lines.append("configuration, whatever the middle level's internal quality.")
    elif collapse["degenerate"] and not gates_open:
        lines.append("**STOP.** The measured middle level changed neither the accept length nor the round count,")
        lines.append("and neither saving mechanism can pay for a full extra model pass at this configuration.")
    elif collapse["degenerate"]:
        lines.append("**STOP as specified.** The funnel degenerated, but a gate above is open: reconsider the")
        lines.append("middle level as a certificate source rather than as a pre-verification filter.")
    else:
        lines.append("**CONTINUE / INSPECT.** The funnel changed the outcome; validate per-round before building an engine.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--phase", default="measure")
    parser.add_argument("--ar", default="ar")
    parser.add_argument("--baseline", default="static")
    parser.add_argument("--candidate", default="hsd")
    parser.add_argument("--gamma", type=int, default=9)
    parser.add_argument(
        "--middle-ms-per-round",
        type=float,
        default=None,
        help="override the middle-level cost (required when --candidate equals --baseline, i.e. to "
             "evaluate a hypothetical middle level on a baseline-only run)",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="emit the raw gate results as JSON")
    args = parser.parse_args(argv)
    if args.gamma < 1:
        parser.error("--gamma must be positive")
    if args.middle_ms_per_round is not None and args.middle_ms_per_round < 0:
        parser.error("--middle-ms-per-round must be non-negative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    grouped = group_by_method(load_trials(args.jsonl, args.phase))
    summaries = {name: summarize(rows) for name, rows in grouped.items()}
    for required in (args.ar, args.baseline, args.candidate):
        if required not in summaries:
            raise SystemExit(
                f"method {required!r} not found in {args.jsonl}; available: {sorted(summaries)}"
            )
    hypothetical = args.candidate == args.baseline
    if hypothetical and args.middle_ms_per_round is None:
        raise SystemExit(
            "--candidate equals --baseline; pass --middle-ms-per-round to evaluate a hypothetical middle level"
        )
    middle_ms = (
        float(args.middle_ms_per_round)
        if args.middle_ms_per_round is not None
        else summaries[args.candidate]["middle_ms_per_round"]
    )
    if args.json:
        payload = {
            "summaries": summaries,
            "middle_ms_per_round": middle_ms,
            "hypothetical": hypothetical,
            "collapse": funnel_collapse_check(summaries[args.baseline], summaries[args.candidate]),
            "truncation": truncation_gate(summaries[args.ar], summaries[args.baseline], args.gamma, middle_ms),
            "certification": certification_gate(summaries[args.baseline], middle_ms),
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
    else:
        text = render_report(
            summaries, args.baseline, args.candidate, args.ar, args.gamma, middle_ms if hypothetical else None
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
