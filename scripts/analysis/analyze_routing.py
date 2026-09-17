"""Offline Oracle Study analysis for Verification-Guided Visual Routing.

Reads per-sample attention traces collected by collect_traces.py and evaluates
four visual selectors (Static / Previous / EMA / Oracle) against the per-round
dense-verification visual relevance A_t. Produces:

  - drift / local-predictability tables (Jaccard, Recall, attention mass)
  - accepted-length proxy (mass-coverage based; token-level proxy would require
    re-running the sparse draft, which is out of scope for Phase 1)
  - two ASCII figures (no matplotlib dependency) + CSV for external plotting
  - a GO / NO-GO decision.

Usage:
  python scripts/analysis/analyze_routing.py \
    --traces-dir results/routing_traces --datasets VideoDetailCaption MLVU \
    --frame-num 128 --out results/routing_analysis
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]

EMA_LAMBDAS = [0.5, 0.8, 0.9]


def topk_indices(relevance: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(relevance)[::-1][:k]


def _as_set(idx: np.ndarray) -> set:
    return set(int(x) for x in idx)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = _as_set(a), _as_set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def recall(sel: np.ndarray, oracle: np.ndarray) -> float:
    so = _as_set(oracle)
    return len(_as_set(sel) & so) / len(so) if so else 0.0


def attention_mass(sel: np.ndarray, r: np.ndarray) -> float:
    denom = float(r.sum())
    return float(r[sel].sum() / denom) if denom > 0 else 0.0


def load_meta(meta_path: Path) -> list:
    meta = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meta.append(json.loads(line))
    return meta


def analyze_sample(meta: dict, traces_dir: Path) -> dict:
    payload = torch.load(traces_dir / f"{meta['sample_id']}.pt", map_location="cpu", weights_only=True)
    r0 = payload["prefill_scores"].float().mean(dim=0).numpy()          # [visual_len]
    rounds = payload["round_scores"]
    if torch.is_tensor(rounds):
        # Summed convention: [num_rounds, kv_heads, visual_len].
        R = rounds.float().mean(dim=1).numpy()                          # [num_rounds, visual_len]
    else:
        # Per-query convention from --record-per-query: reduce each round to the
        # summed (V1) scores first so the Oracle Study convention is unchanged.
        R = torch.stack([r.float().sum(dim=1) for r in rounds], dim=0).mean(dim=1).numpy()
    T = R.shape[0]
    k = meta["k"]
    S0 = topk_indices(r0, k)
    oracles = [topk_indices(R[t], k) for t in range(T)]

    static_recall = [recall(S0, oracles[t]) for t in range(T)]
    static_jaccard = [jaccard(S0, oracles[t]) for t in range(T)]
    static_mass = [attention_mass(S0, R[t]) for t in range(T)]

    prev_recall, prev_jaccard, prev_mass = [], [], []
    for t in range(T):
        prev = S0 if t == 0 else oracles[t - 1]
        prev_recall.append(recall(prev, oracles[t]))
        prev_jaccard.append(jaccard(prev, oracles[t]))
        prev_mass.append(attention_mass(prev, R[t]))

    ema_recall = {lam: [] for lam in EMA_LAMBDAS}
    ema_jaccard = {lam: [] for lam in EMA_LAMBDAS}
    ema_mass = {lam: [] for lam in EMA_LAMBDAS}
    for lam in EMA_LAMBDAS:
        E = r0.copy()
        for t in range(T):
            sel = topk_indices(E, k)
            ema_recall[lam].append(recall(sel, oracles[t]))
            ema_jaccard[lam].append(jaccard(sel, oracles[t]))
            ema_mass[lam].append(attention_mass(sel, R[t]))
            E = lam * E + (1.0 - lam) * R[t]

    oracle_mass = [attention_mass(oracles[t], R[t]) for t in range(T)]

    return {
        "sample_id": meta["sample_id"],
        "visual_len": meta["visual_len"],
        "k": k,
        "T": T,
        "accept_lengths": meta["accept_lengths"],
        "mean_accept_length": meta["mean_accept_length"],
        "static_recall": static_recall,
        "static_jaccard": static_jaccard,
        "static_mass": static_mass,
        "prev_recall": prev_recall,
        "prev_jaccard": prev_jaccard,
        "prev_mass": prev_mass,
        "ema_recall": ema_recall,
        "ema_jaccard": ema_jaccard,
        "ema_mass": ema_mass,
        "oracle_mass": oracle_mass,
    }


def align_mean(seqs, method: str):
    """Mean over samples per round index (aligned from round 0)."""
    max_t = max(len(s) for s in seqs)
    means, counts = [], []
    for t in range(max_t):
        vals = [s[t] for s in seqs if t < len(s)]
        means.append(float(np.mean(vals)))
        counts.append(len(vals))
    return means, counts


def ascii_plot(xs, ys, width=60, height=16, ylabel="", title="", ymin=None, ymax=None):
    if ymin is None:
        ymin = min(ys) - 0.05 * (max(ys) - min(ys) if max(ys) != min(ys) else 1.0)
    if ymax is None:
        ymax = max(ys) + 0.05 * (max(ys) - min(ys) if max(ys) != min(ys) else 1.0)
    span = ymax - ymin if ymax != ymin else 1.0
    grid = [[" "] * width for _ in range(height)]
    for x, y in zip(xs, ys):
        col = int((x - min(xs)) / (max(xs) - min(xs)) * (width - 1)) if max(xs) != min(xs) else 0
        row = height - 1 - int((y - ymin) / span * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][col] = "*"
    lines = [f"  {title}"]
    lines.append(f"  y^ {ymax:.2f}")
    for i in range(height):
        lines.append("  |" + "".join(grid[i]))
    lines.append(f"  +{'—' * width}-> round (0..{max(xs)})  ymin={ymin:.2f}")
    return "\n".join(lines)


def headroom_rows(analyzed: list) -> list:
    """Per-round rows joining a round's measured accept length to its recall gain.

    `accept_length` is the static STD run's accepted length for that round, used
    as the "remaining headroom" proxy: a high value means the static selection
    was already good for that round.
    """
    rows = []
    for a in analyzed:
        acc = list(a.get("accept_lengths") or [])
        for t in range(a["T"]):
            static_r = float(a["static_recall"][t])
            prev_r = float(a["prev_recall"][t])
            rows.append(
                {
                    "dataset": a.get("_dataset"),
                    "sample_id": a["sample_id"],
                    "round": t,
                    "accept_length": float(acc[t]) if t < len(acc) else float("nan"),
                    "static_recall": static_r,
                    "prev_recall": prev_r,
                    "delta_recall": prev_r - static_r,
                }
            )
    return rows


def _rankdata(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(len(order), dtype=float)
    return ranks


def _pearson(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def quantile_strata(rows: list, bins: int = 3) -> list:
    """Equal-count bins over `accept_length`, lowest headroom first."""
    valid = [r for r in rows if np.isfinite(r["accept_length"])]
    if not valid:
        return []
    valid.sort(key=lambda r: r["accept_length"])
    out = []
    n = len(valid)
    for b in range(bins):
        lo = (b * n) // bins
        hi = ((b + 1) * n) // bins
        chunk = valid[lo:hi]
        if not chunk:
            continue
        out.append(
            {
                "bin": b,
                "n": len(chunk),
                "accept_min": min(r["accept_length"] for r in chunk),
                "accept_max": max(r["accept_length"] for r in chunk),
                "mean_accept": float(np.mean([r["accept_length"] for r in chunk])),
                "static_recall": float(np.mean([r["static_recall"] for r in chunk])),
                "prev_recall": float(np.mean([r["prev_recall"] for r in chunk])),
                "delta_recall": float(np.mean([r["delta_recall"] for r in chunk])),
            }
        )
    return out


def headroom_report(analyzed: list, out_dir: Path, bins: int = 3) -> dict:
    """Print and persist the H_headroom stratification; return the summary."""
    import csv

    rows = headroom_rows(analyzed)
    strata = quantile_strata(rows, bins=bins)

    print("\n" + "-" * 78)
    print("H_headroom: recall(Previous) - recall(Static), stratified by that round's")
    print("measured static accept length (equal-count bins, LOW headroom first)")
    print("-" * 78)
    print(
        f"{'bin':<4} {'n':>6} {'accept':>9} {'mean acc':>9} "
        f"{'static':>8} {'prev':>8} {'delta(pp)':>10}"
    )
    for s in strata:
        rng = f"{s['accept_min']:.0f}-{s['accept_max']:.0f}"
        print(
            f"{s['bin']:<4} {s['n']:>6} {rng:>9} {s['mean_accept']:>9.2f} "
            f"{s['static_recall']:>8.4f} {s['prev_recall']:>8.4f} "
            f"{100 * s['delta_recall']:>10.2f}"
        )

    valid = [r for r in rows if np.isfinite(r["accept_length"])]
    pearson = spearman = float("nan")
    if len(valid) >= 3:
        acc = [r["accept_length"] for r in valid]
        delta = [r["delta_recall"] for r in valid]
        pearson = _pearson(acc, delta)
        spearman = _pearson(_rankdata(acc), _rankdata(delta))
        print(
            f"\ncorr(accept_length, delta_recall): pearson={pearson:+.3f}  "
            f"spearman={spearman:+.3f}"
        )
        print("H_headroom predicts a NEGATIVE correlation: the gain shrinks as the")
        print("static selection already performs well.")

    low = strata[0]["delta_recall"] if strata else float("nan")
    high = strata[-1]["delta_recall"] if strata else float("nan")
    if strata:
        print(f"\nlow-headroom delta={100 * low:+.2f}pp   high-headroom delta={100 * high:+.2f}pp")
        consistent = np.isfinite(low) and np.isfinite(high) and low > high and low > 0
        print(
            "=> H_headroom CONSISTENT (gain concentrated in low-headroom rounds)"
            if consistent
            else "=> H_headroom NOT supported by this run"
        )

    with (out_dir / "headroom_rounds.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["sample_id", "dataset", "round", "accept_length", "static_recall",
             "prev_recall", "delta_recall"]
        )
        for r in rows:
            writer.writerow(
                [
                    r["sample_id"], r["dataset"], r["round"],
                    f"{r['accept_length']:.4f}", f"{r['static_recall']:.6f}",
                    f"{r['prev_recall']:.6f}", f"{r['delta_recall']:.6f}",
                ]
            )

    summary = {
        "bins": strata,
        "pearson_accept_vs_delta_recall": pearson,
        "spearman_accept_vs_delta_recall": spearman,
        "low_headroom_delta_recall": low,
        "high_headroom_delta_recall": high,
        "n_rounds": len(rows),
    }
    with (out_dir / "headroom_strata.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", default=str(ROOT / "results" / "routing_traces"))
    parser.add_argument("--datasets", nargs="+", default=["VideoDetailCaption", "MLVU"])
    parser.add_argument("--frame-num", type=int, default=128)
    parser.add_argument("--out", default=str(ROOT / "results" / "routing_analysis"))
    args = parser.parse_args()

    traces_dir = Path(args.traces_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    for ds in args.datasets:
        meta_path = traces_dir / f"{ds}_frame{args.frame_num}.jsonl"
        if not meta_path.exists():
            print(f"[skip] {meta_path} not found")
            continue
        for m in load_meta(meta_path):
            m["_dataset"] = ds
            all_samples.append(m)

    print(f"Analyzing {len(all_samples)} samples from {args.datasets} @ frame={args.frame_num}\n")

    analyzed = []
    for m in all_samples:
        a = analyze_sample(m, traces_dir)
        a["_dataset"] = m["_dataset"]
        analyzed.append(a)
    if not analyzed:
        print("No samples analyzed.")
        return

    # ---- aggregate recall ----
    static_recall_mean, _ = align_mean([a["static_recall"] for a in analyzed], "mean")
    prev_recall_mean, _ = align_mean([a["prev_recall"] for a in analyzed], "mean")
    ema_recall_mean = {lam: align_mean([a["ema_recall"][lam] for a in analyzed], "mean")[0] for lam in EMA_LAMBDAS}

    # ---- aggregate mass ----
    static_mass_mean, _ = align_mean([a["static_mass"] for a in analyzed], "mean")
    prev_mass_mean, _ = align_mean([a["prev_mass"] for a in analyzed], "mean")
    ema_mass_mean = {lam: align_mean([a["ema_mass"][lam] for a in analyzed], "mean")[0] for lam in EMA_LAMBDAS}
    oracle_mass_mean, _ = align_mean([a["oracle_mass"] for a in analyzed], "mean")

    # ---- overall means (macro over rounds, then over samples) ----
    def overall(seqs):
        vals = [np.mean(s) for s in seqs]
        return float(np.mean(vals))

    print("=" * 78)
    print("ORACLE STUDY: Verification-Guided Visual Routing (Phase 1)")
    print("=" * 78)
    print(f"datasets: {args.datasets}   frame_num: {args.frame_num}")
    print(f"samples: {len(analyzed)}")
    total_rounds = sum(a['T'] for a in analyzed)
    print(f"total verification rounds: {total_rounds}")
    print(f"visual_len: {analyzed[0]['visual_len']}   K: {analyzed[0]['k']}")
    print(f"mean accept length (Static STD, measured): "
          f"{np.mean([a['mean_accept_length'] for a in analyzed]):.2f}")

    # ---- Table 1: Recall vs oracle ----
    print("\n" + "-" * 78)
    print("Recall(S_selector, S_oracle)  [higher = better; NO-GO if Static ~ Oracle]")
    print("-" * 78)
    print(f"{'selector':<12} {'mean recall':>12} {'delta vs Static (pp)':>20}")
    sr = overall([a["static_recall"] for a in analyzed])
    pr = overall([a["prev_recall"] for a in analyzed])
    print(f"{'Static':<12} {sr:>12.4f} {'—':>20}")
    print(f"{'Previous':<12} {pr:>12.4f} {100*(pr-sr):>19.2f}")
    for lam in EMA_LAMBDAS:
        er = overall([a["ema_recall"][lam] for a in analyzed])
        print(f"{'EMA λ='+str(lam):<12} {er:>12.4f} {100*(er-sr):>19.2f}")

    # ---- Table 2: attention mass ----
    print("\n" + "-" * 78)
    print("Attention mass coverage M(S) = Σ_{j∈S} A_t[j] / Σ_j A_t[j]")
    print("-" * 78)
    print(f"{'selector':<12} {'mean mass':>12} {'mass gap vs Oracle':>20}")
    sm = overall([a["static_mass"] for a in analyzed])
    pm = overall([a["prev_mass"] for a in analyzed])
    om = overall([a["oracle_mass"] for a in analyzed])
    print(f"{'Static':<12} {sm:>12.4f} {om-sm:>19.4f}")
    print(f"{'Previous':<12} {pm:>12.4f} {om-pm:>19.4f}")
    for lam in EMA_LAMBDAS:
        em = overall([a["ema_mass"][lam] for a in analyzed])
        print(f"{'EMA λ='+str(lam):<12} {em:>12.4f} {om-em:>19.4f}")
    print(f"{'Oracle':<12} {om:>12.4f} {'0.0':>20}")

    # ---- accepted-length proxy (mass-coverage based) ----
    print("\n" + "-" * 78)
    print("Accepted-length proxy (mass-coverage scaled; token-level proxy requires")
    print("re-running sparse draft, which is out of scope for Phase 1)")
    print("-" * 78)
    a_bar = np.mean([a["mean_accept_length"] for a in analyzed])
    def proxy(mass):
        return a_bar * mass / sm if sm > 0 else a_bar
    print(f"{'selector':<12} {'proxy accept (tokens)':>22}")
    print(f"{'Static':<12} {proxy(sm):>22.2f}")
    print(f"{'Previous':<12} {proxy(pm):>22.2f}")
    for lam in EMA_LAMBDAS:
        em = overall([a["ema_mass"][lam] for a in analyzed])
        print(f"{'EMA λ='+str(lam):<12} {proxy(em):>22.2f}")
    print(f"{'Oracle':<12} {proxy(om):>22.2f}")

    # ---- H_headroom stratification (see the thorough-analysis plan) ----
    headroom = headroom_report(analyzed, out_dir)

    # ---- Figures (ASCII) ----
    static_jaccard_mean, _ = align_mean([a["static_jaccard"] for a in analyzed], "mean")
    prev_jaccard_mean, _ = align_mean([a["prev_jaccard"] for a in analyzed], "mean")
    rounds = list(range(len(static_jaccard_mean)))

    # Figure 1: static drift Jaccard(S0, St*)
    print("\n" + ascii_plot(rounds, static_jaccard_mean, title="Figure 1: Static drift  Jaccard(S0, St*) vs round"))

    # Figure 2: local predictability Jaccard(S_{t-1}*, St*) — use prev_jaccard (from round 1 onward)
    local_rounds = rounds[1:]
    local_vals = prev_jaccard_mean[1:]
    print("\n" + ascii_plot(local_rounds, local_vals, title="Figure 2: Local predictability  Jaccard(S_{t-1}*, St*) vs round"))

    # ---- CSV export ----
    import csv
    def write_csv(name, xs, series):
        with (out_dir / name).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round"] + list(series.keys()))
            for i, x in enumerate(xs):
                w.writerow([x] + [f"{series[k][i]:.6f}" for k in series])

    write_csv("drift_jaccard.csv", rounds, {"static_jaccard": static_jaccard_mean, "prev_jaccard": prev_jaccard_mean})
    write_csv("recall.csv", rounds, {
        "static": static_recall_mean,
        "prev": prev_recall_mean,
        **{f"ema{lam}": ema_recall_mean[lam] for lam in EMA_LAMBDAS},
    })
    write_csv("mass.csv", rounds, {
        "static": static_mass_mean,
        "prev": prev_mass_mean,
        "oracle": oracle_mass_mean,
        **{f"ema{lam}": ema_mass_mean[lam] for lam in EMA_LAMBDAS},
    })
    print(f"\nCSV + data written to {out_dir}")

    # ---- GO / NO-GO ----
    print("\n" + "=" * 78)
    print("FINAL DECISION")
    print("=" * 78)
    drift_gap = om - sm                      # oracle mass advantage over static
    recall_gain_prev = 100 * (pr - sr)       # pp
    oracle_still_gap = (om - pm) > 0.01

    cond_no_go = drift_gap < 0.01            # Oracle ≈ Static
    cond_a = proxy(om) - proxy(sm) >= 0.5    # oracle accept proxy >= +0.5 token
    cond_b = recall_gain_prev >= 2.0 and oracle_still_gap

    print(f"Oracle mass − Static mass (drift gap): {drift_gap:.4f}")
    print(f"Previous recall gain vs Static:       {recall_gain_prev:.2f} pp")
    print(f"Oracle still gaps Previous (mass):     {oracle_still_gap}")
    print(f"Oracle accept-proxy delta vs Static:   {proxy(om)-proxy(sm):.2f} tokens")
    print()
    if cond_no_go:
        decision = "NO-GO"
        reason = "Oracle ≈ Static: visual relevance does not drift enough to justify dynamic routing."
    elif cond_a or cond_b:
        decision = "GO"
        parts = []
        if cond_a:
            parts.append(f"Oracle accept proxy ≥ Static + 0.5 token ({proxy(om)-proxy(sm):.2f})")
        if cond_b:
            parts.append(f"Previous recall ≥ Static + 2pp ({recall_gain_prev:.2f}pp) with oracle still gapped")
        reason = "; ".join(parts)
    else:
        decision = "NO-GO (weak signal)"
        reason = "Drift exists but neither GO condition met; signal too weak for dynamic routing."
    print(f"Dynamic visual routing: {decision}")
    print(f"Reason: {reason}")

    # ---- G-L1 gate: is a controlled online ablation (L2) warranted? ----
    print("\n" + "-" * 78)
    print("G-L1 GATE  (proceed to the L2 controlled ablation only if it passes)")
    print("-" * 78)
    g1_recall = recall_gain_prev >= 5.0
    g1_strata = (
        np.isfinite(headroom["low_headroom_delta_recall"])
        and headroom["low_headroom_delta_recall"] > headroom["high_headroom_delta_recall"]
        and headroom["low_headroom_delta_recall"] > 0
    )
    print(f"  recall(Previous)-recall(Static) >= +5.0pp : {recall_gain_prev:+.2f}pp -> {g1_recall}")
    print(f"  gain concentrated in low-headroom rounds  : {g1_strata}")
    print(
        "  => "
        + ("PROCEED to L2" if (g1_recall and g1_strata)
           else "STOP: no exploitable signal on this dataset (H4)")
    )


if __name__ == "__main__":
    main()
