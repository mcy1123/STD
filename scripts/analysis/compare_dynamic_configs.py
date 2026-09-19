#!/usr/bin/env python
"""Compare several dynamic-routing runs of ``benchmark_a100_dynamic.py`` side by side.

Every run is reduced to the same paired, headroom-stratified view that
``summarize_a100_dynamic.py`` uses, so configurations that differ in one factor
(the E1/E6 matrix: collector version and selection-update throttle) can be read
off one table instead of three reports.

    python3 scripts/analysis/compare_dynamic_configs.py \
        R0_full=results/l2_screening/R0_full_three_att.jsonl \
        R1_incr=results/l2_screening/R1_incr_three_att.jsonl \
        E1_i4=results/l2_screening/e1e6_20260919/e1e6_20260919_B_v2_i4.jsonl

The aggregate mean is deliberately *not* the headline: on the 2026-09-09 A100 run
it hid a sign flip that correlated with how much headroom static still had. The
correlation and the stratified bins are printed instead, alongside the per-round
cost breakdown, because "select better" and "swap cheaper" are two factors of one
product and a config that looks neutral on acceptance can still be a regression
on cost.
"""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean


COMPONENT_KEYS = ("draft_time", "verify_time", "bonus_time", "selection_time",
                  "sparse_cache_time", "cache_adjust_time")


def _read(path):
    """Return (manifest, trials, dynamic_stats) from one benchmark JSONL."""
    manifest, trials, stats = {}, {}, {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            kind = rec.get("kind")
            if kind == "manifest":
                manifest = rec
                continue
            if kind != "trial":
                continue
            if rec.get("phase") == "measure" and rec.get("repeat") == 0:
                trials[(rec["sample_id"], rec["method"])] = rec
                if rec.get("method") == "dynamic_v2":
                    stats[rec["sample_id"]] = rec.get("dynamic_stats") or {}
    return manifest, trials, stats


def _corr(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float("nan")


def load_run(path):
    """Reduce one JSONL into the paired view plus per-round costs."""
    manifest, trials, stats = _read(path)
    samples = sorted({sid for sid, method in trials if method == "static"}
                     & {sid for sid, method in trials if method == "dynamic_v2"})

    paired = []
    for sid in samples:
        st, dy = trials[(sid, "static")], trials[(sid, "dynamic_v2")]
        paired.append({
            "sample_id": sid,
            "static_accept": float(st.get("acceptance_rate", 0.0)),
            "dynamic_accept": float(dy.get("acceptance_rate", 0.0)),
            "static_accept_len": float(st.get("mean_accept_length", 0.0)),
            "dynamic_accept_len": float(dy.get("mean_accept_length", 0.0)),
        })
    for row in paired:
        row["delta_accept"] = row["dynamic_accept"] - row["static_accept"]
        row["delta_accept_len"] = row["dynamic_accept_len"] - row["static_accept_len"]

    rounds = {sid: int(trials[(sid, "dynamic_v2")].get("decode_rounds", 0) or 0) for sid in samples}
    static_rounds = sum(int(trials[(sid, "static")].get("decode_rounds", 0) or 0) for sid in samples)
    total_rounds = sum(rounds.values())

    def per_round(key, method="dynamic_v2"):
        total = sum(float(trials[(sid, method)].get(key, 0.0) or 0.0) for sid in samples)
        n = total_rounds if method == "dynamic_v2" else static_rounds
        return 1000.0 * total / n if n else float("nan")

    refresh_ms, selection_ms, changed, mismatches, checks = [], [], [], 0, 0
    for sid in samples:
        d = stats.get(sid, {})
        refresh_ms += [float(r.get("refresh_time_ms", 0.0) or 0.0) for r in d.get("per_round", [])]
        changed += [float(r.get("changed_ratio", 0.0) or 0.0) for r in d.get("per_round", [])]
        mismatches += int(d.get("consistency_mismatches", 0) or 0)
        checks += int(d.get("consistency_checks", 0) or 0)
        selection_ms.append(float(d.get("total_selection_update_time_ms", 0.0) or 0.0))

    static_decode_ms = per_round("decoding_time", "static")
    dynamic_decode_ms = per_round("decoding_time", "dynamic_v2")
    return {
        "path": str(path),
        "timing_valid": bool(manifest.get("timing_valid", False)),
        "collector": (stats.get(samples[0], {}).get("collector_version") if samples else None),
        "refresh_mode": (stats.get(samples[0], {}).get("refresh_mode") if samples else None),
        "paired": paired,
        "mean_static": mean(r["static_accept"] for r in paired) if paired else float("nan"),
        "mean_delta": mean(r["delta_accept"] for r in paired) if paired else float("nan"),
        "mean_delta_len": mean(r["delta_accept_len"] for r in paired) if paired else float("nan"),
        "corr": _corr([r["static_accept"] for r in paired],
                      [r["delta_accept"] for r in paired]) if paired else float("nan"),
        "low_delta": (mean(r["delta_accept"] for r in sorted(paired, key=lambda r: r["static_accept"])[:max(1, len(paired) // 3)])
                      if paired else float("nan")),
        "rounds": total_rounds,
        "static_rounds": static_rounds,
        "static_decode_ms_per_round": static_decode_ms,
        "dynamic_decode_ms_per_round": dynamic_decode_ms,
        "toll_ms_per_round": dynamic_decode_ms - static_decode_ms,
        "refresh_ms_per_round": (sum(refresh_ms) / len(refresh_ms)) if refresh_ms else float("nan"),
        "refresh_ms_p99": sorted(refresh_ms)[int(0.99 * (len(refresh_ms) - 1))] if refresh_ms else float("nan"),
        "selection_update_ms_per_round": (sum(selection_ms) / total_rounds) if total_rounds else float("nan"),
        "mean_changed_ratio": (sum(changed) / len(changed)) if changed else float("nan"),
        "consistency_mismatches": mismatches,
        "consistency_checks": checks,
    }


def strata(run, n_bins=3):
    """Equal-count headroom bins, LOW headroom first (matches the summarizer)."""
    rows = sorted(run["paired"], key=lambda r: r["static_accept"])
    out = []
    for b in range(min(n_bins, len(rows))):
        lo, hi = (b * len(rows)) // min(n_bins, len(rows)), ((b + 1) * len(rows)) // min(n_bins, len(rows))
        chunk = rows[lo:hi]
        if chunk:
            out.append((b, len(chunk), chunk[0]["static_accept"], chunk[-1]["static_accept"],
                        mean(r["delta_accept"] for r in chunk)))
    return out


def render(runs, title="dynamic-routing configuration comparison"):
    names = list(runs)
    lines = [f"# {title}", ""]
    lines.append("| run | collector | refresh | timing_valid | samples | rounds | exact T1 |")
    lines.append("|---|---|---|---|---:|---:|---|")
    for n in names:
        r = runs[n]
        lines.append(f"| {n} | {r['collector']} | {r['refresh_mode']} | "
                     f"{r['timing_valid']} | {len(r['paired'])} | {r['rounds']} | "
                     f"{r['consistency_checks'] - r['consistency_mismatches']}/{r['consistency_checks']} |")

    lines += ["", "## Paired acceptance vs the run's own static baseline", "",
              "| run | mean static | mean d accept | mean d accept_len | corr(static, d) | low-headroom d |",
              "|---|---:|---:|---:|---:|---:|"]
    for n in names:
        r = runs[n]
        lines.append(f"| {n} | {r['mean_static']:.4f} | {r['mean_delta']:+.4f} | "
                     f"{r['mean_delta_len']:+.4f} | {r['corr']:+.3f} | {r['low_delta']:+.4f} |")

    common = None
    for r in runs.values():
        s = {row["sample_id"] for row in r["paired"]}
        common = s if common is None else (common & s)
    common = sorted(common or [])
    if common:
        lines += ["", "## Per-sample d accept (only samples present in every run)", "",
                  "| sample | static | " + " | ".join(names) + " |", "|---|---:|" + "---:|" * len(names)]
        for sid in common:
            base = next(row for row in runs[names[0]]["paired"] if row["sample_id"] == sid)["static_accept"]
            cells = []
            for n in names:
                row = next((x for x in runs[n]["paired"] if x["sample_id"] == sid), None)
                cells.append(f"{row['delta_accept']:+.3f}" if row else "n/a")
            lines.append(f"| {sid} | {base:.3f} | " + " | ".join(cells) + " |")

    lines += ["", "## Headroom strata (equal-count bins, LOW headroom first)", ""]
    for n in names:
        lines.append(f"**{n}**")
        lines.append("")
        lines.append("| bin | n | static range | mean d accept |")
        lines.append("|---:|---:|---|---:|")
        for b, cnt, lo, hi, d in strata(runs[n]):
            lines.append(f"| {b} | {cnt} | {lo:.3f}-{hi:.3f} | {d:+.4f} |")
        lines.append("")

    lines += ["## Per-round cost (wall-clock is only comparable when timing_valid)", "",
              "| run | static decode ms/round | dynamic decode ms/round | toll ms/round | refresh ms/round | refresh p99 | selection ms/round | mean changed ratio |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        r = runs[n]
        lines.append(f"| {n} | {r['static_decode_ms_per_round']:.1f} | {r['dynamic_decode_ms_per_round']:.1f} | "
                     f"{r['toll_ms_per_round']:+.1f} | {r['refresh_ms_per_round']:.1f} | "
                     f"{r['refresh_ms_p99']:.1f} | {r['selection_update_ms_per_round']:.1f} | "
                     f"{r['mean_changed_ratio']:.3f} |")
    lines.append("")
    return "\n".join(lines)


def parse_run(spec):
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    if not Path(path).is_file():
        raise argparse.ArgumentTypeError(f"no such file: {path}")
    return name, Path(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=parse_run, metavar="NAME=PATH.jsonl")
    ap.add_argument("--output", type=Path, help="write markdown here instead of stdout")
    args = ap.parse_args(argv)

    runs = {}
    for name, path in args.runs:
        if name in runs:
            ap.error(f"duplicate run name {name!r}")
        runs[name] = load_run(path)

    text = render(runs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
