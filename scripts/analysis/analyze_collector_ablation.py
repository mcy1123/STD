#!/usr/bin/env python
"""Offline collector-estimator ablation (L1.2 of the thorough-analysis plan).

Reads per-query traces produced by
`collect_traces.py --record-per-query`, and asks whether the *query subset* a
collector samples can explain the churn that the online A100 runs showed
(adjacent-round Jaccard ~0.52, ~30% of the visual KV replaced every round).

For every verification round it builds a visual top-K from three estimators:

  * ``all``   — sum over every verification query (the V1 reference);
  * ``two``   — exactly the positions `verification_query_positions(mode="two")`
                would have chosen, reconstructed from the recorded
                ``accept_lengths`` / ``pending_lengths`` / ``proposed_lengths``;
  * ``three`` — the legacy three-query control.

Metrics per mode:

  * ``fidelity``   — recall(estimator S_t, all-query S_t): how much of the
                     full-query top-K survives the subsampling (1.0 for ``all``);
  * ``predictive`` — recall(estimator S_t, all-query S_{t+1}): whether the
                     selection still covers the *next* block's needs;
  * ``churn``      — Jaccard(S_t, S_{t+1}): run-to-run stability. If ``two`` is
                     far less stable than ``all``, the estimator — not the
                     routing idea — is what injects the noise.

Usage:
  python scripts/analysis/analyze_collector_ablation.py \
    --traces-dir results/routing_traces_videomme \
    --dataset Video-MME --frame-num 128 \
    --out results/routing_collector_ablation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts" / "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_routing import jaccard, recall, topk_indices  # noqa: E402
from std_repro.dynamic_selection import verification_query_positions  # noqa: E402

MODES = ("all", "two", "three")


def estimator_scores(
    per_query: torch.Tensor, positions: Optional[Sequence[int]]
) -> torch.Tensor:
    """Sum per-query attention over the selected query rows -> [kv_heads, visual_len]."""
    if not positions:
        return per_query.sum(dim=1)
    q_len = int(per_query.shape[1])
    valid = [int(p) for p in positions if 0 <= int(p) < q_len]
    if not valid:
        raise ValueError(f"no valid query position in {list(positions)} for q_len={q_len}")
    return per_query[:, valid, :].sum(dim=1)


def round_positions(
    mode: str, accept_len: int, pending_len: int, propose_len: int
) -> Optional[List[int]]:
    if mode == "all":
        return None
    return verification_query_positions(accept_len, pending_len, propose_len, mode=mode)


def _selection(scores: torch.Tensor, k: int) -> np.ndarray:
    """Mean over kv heads, then top-K -- matching analyze_routing's convention."""
    return topk_indices(scores.float().mean(dim=0).numpy(), k)


def analyze_sample(meta: dict, payload: dict, modes: Sequence[str] = MODES) -> dict:
    if not payload.get("per_query", False):
        raise ValueError(
            f"{meta['sample_id']}: trace lacks --record-per-query; re-collect it "
            "before running the collector ablation."
        )
    rounds = payload["round_scores"]
    if torch.is_tensor(rounds):
        raise ValueError(f"{meta['sample_id']}: expected a per-round list, got a stacked tensor")

    k = int(meta["k"])
    accept = list(meta["accept_lengths"])
    pending = list(meta.get("pending_lengths") or [])
    proposed = list(meta.get("proposed_lengths") or [])
    n = len(rounds)
    if not (len(accept) == len(pending) == len(proposed) == n):
        raise ValueError(
            f"{meta['sample_id']}: per-round metadata length mismatch "
            f"(rounds={n}, accept={len(accept)}, pending={len(pending)}, proposed={len(proposed)}). "
            "Traces collected before `pending_lengths` existed cannot be used."
        )

    all_scores = torch.stack([r.float().sum(dim=1) for r in rounds], dim=0)  # [T, H, V]
    reference = [_selection(all_scores[t], k) for t in range(n)]

    per_mode: Dict[str, dict] = {}
    for mode in modes:
        selections = []
        for t in range(n):
            positions = round_positions(mode, int(accept[t]), int(pending[t]), int(proposed[t]))
            selections.append(_selection(estimator_scores(rounds[t], positions), k))
        per_mode[mode] = {
            "fidelity": [recall(selections[t], reference[t]) for t in range(n)],
            "predictive": [recall(selections[t], reference[t + 1]) for t in range(n - 1)],
            "churn_jaccard": [jaccard(selections[t], selections[t + 1]) for t in range(n - 1)],
        }

    return {"sample_id": meta["sample_id"], "T": n, "per_mode": per_mode}


def aggregate(samples: Sequence[dict], modes: Sequence[str] = MODES) -> dict:
    """Macro average over samples, then rounds."""
    out: Dict[str, dict] = {}
    for mode in modes:
        entry = {}
        for metric in ("fidelity", "predictive", "churn_jaccard"):
            per_sample = [
                float(np.mean(s["per_mode"][mode][metric]))
                for s in samples
                if s["per_mode"][mode][metric]
            ]
            entry[metric] = float(np.mean(per_sample)) if per_sample else float("nan")
        entry["samples"] = len(samples)
        out[mode] = entry
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", default=str(ROOT / "results" / "routing_traces"))
    parser.add_argument("--dataset", default="Video-MME")
    parser.add_argument("--frame-num", type=int, default=128)
    parser.add_argument("--out", default=str(ROOT / "results" / "routing_collector_ablation"))
    return parser


def load_meta(meta_path: Path) -> list:
    rows = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = build_parser().parse_args()
    traces_dir = Path(args.traces_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = traces_dir / f"{args.dataset}_frame{args.frame_num}.jsonl"
    if not meta_path.exists():
        print(f"no metadata at {meta_path}; nothing to analyze.")
        return

    analyzed = []
    for meta in load_meta(meta_path):
        pt = traces_dir / f"{meta['sample_id']}.pt"
        if not pt.exists():
            print(f"[skip] missing trace for {meta['sample_id']}")
            continue
        payload = torch.load(pt, map_location="cpu", weights_only=True)
        try:
            analyzed.append(analyze_sample(meta, payload))
        except ValueError as exc:
            print(f"[skip] {exc}")
    if not analyzed:
        print("No usable per-query traces found.")
        return

    summary = aggregate(analyzed)

    print("=" * 78)
    print("COLLECTOR-ESTIMATOR ABLATION (offline, L1.2)")
    print("=" * 78)
    print(f"dataset: {args.dataset}   frame_num: {args.frame_num}   samples: {len(analyzed)}")
    print(f"rounds: {sum(s['T'] for s in analyzed)}")
    print()
    print(f"{'estimator':<10} {'fidelity':>10} {'predictive':>12} {'churn Jaccard':>15}")
    for mode in MODES:
        e = summary[mode]
        print(
            f"{mode:<10} {e['fidelity']:>10.4f} {e['predictive']:>12.4f} "
            f"{e['churn_jaccard']:>15.4f}"
        )

    all_churn = summary["all"]["churn_jaccard"]
    two_churn = summary["two"]["churn_jaccard"]
    two_fidelity = summary["two"]["fidelity"]
    print()
    print(f"online A100 observed adjacent-round Jaccard: ~0.52 (for the v2/two-query collector)")
    if np.isfinite(all_churn) and np.isfinite(two_churn):
        print(
            f"offline all-query churn={all_churn:.4f}  vs  two-query churn={two_churn:.4f}"
        )
        if two_churn < all_churn - 0.05:
            print(
                "=> The two-query estimator is measurably LESS stable than the full-query "
                "reference: the churn is (at least partly) an ESTIMATOR problem, not the "
                "routing idea itself."
            )
        else:
            print(
                "=> Subsampling the queries does not explain the churn; the reference "
                "selection is itself unstable."
            )
    print(f"two-query fidelity vs full-query top-K: {two_fidelity:.4f}")

    payload_out = {
        "dataset": args.dataset,
        "frame_num": args.frame_num,
        "samples": len(analyzed),
        "summary": summary,
        "per_sample": analyzed,
    }
    with (out_dir / "collector_ablation.json").open("w", encoding="utf-8") as f:
        json.dump(payload_out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_dir / 'collector_ablation.json'}")


if __name__ == "__main__":
    main()
