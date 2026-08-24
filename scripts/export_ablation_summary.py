#!/usr/bin/env python
"""Export grouped STD JSONL results as a machine-readable JSON summary."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METRICS = (
    "prompt_len",
    "visual_len",
    "k",
    "acceptance_rate",
    "mean_accept_length",
    "token_level_agreement",
    "ar_decoding_time",
    "std_decoding_time",
    "speedup",
    "ar_inference_time",
    "std_inference_time",
    "inference_speedup",
    "ar_tokens_per_second",
    "std_tokens_per_second",
    "committed_tokens_per_second",
    "draft_time",
    "verify_time",
    "bonus_time",
    "decode_rounds",
    "draft_time_per_round",
    "verify_time_per_round",
    "committed_tokens_per_round",
    "ar_peak_memory_gib",
    "std_peak_memory_gib",
)


def stats(values):
    values = [float(value) for value in values]
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize(records):
    result = {
        "samples": len(records),
        "sequence_exact_rate": sum(bool(r.get("token_equal")) for r in records) / len(records),
        "fixed_length_rate": sum(
            r.get("ar_generate_len") == r.get("std_generate_len") == r.get("max_new_tokens")
            for r in records
        ) / len(records),
    }
    for metric in METRICS:
        values = [record[metric] for record in records if record.get(metric) is not None]
        if values:
            result[metric] = stats(values)
    total_ar_decode = sum(float(r["ar_decoding_time"]) for r in records)
    total_std_decode = sum(float(r["std_decoding_time"]) for r in records)
    result["overall_decode_speedup"] = total_ar_decode / total_std_decode if total_std_decode else 0.0
    if all("ar_inference_time" in r and "std_inference_time" in r for r in records):
        total_ar_inference = sum(float(r["ar_inference_time"]) for r in records)
        total_std_inference = sum(float(r["std_inference_time"]) for r in records)
        result["overall_inference_speedup"] = (
            total_ar_inference / total_std_inference if total_std_inference else 0.0
        )
    retained = []
    for record in records:
        full = float(record.get("visual_len", 0) + record.get("text_len", 0))
        if full:
            retained.append(float(record.get("k", 0) + record.get("text_len", 0)) / full)
    if retained:
        result["retained_ratio"] = stats(retained)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--group-by", required=True, help="Comma-separated JSON fields.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No records found in {args.input}")
    fields = [field.strip() for field in args.group_by.split(",") if field.strip()]
    groups = {}
    for record in records:
        key = tuple(record.get(field) for field in fields)
        groups.setdefault(key, []).append(record)
    output = {
        "source": str(Path(args.input)),
        "group_by": fields,
        "groups": [
            {
                "key": {field: value for field, value in zip(fields, key)},
                **summarize(group_records),
            }
            for key, group_records in sorted(groups.items(), key=lambda item: str(item[0]))
        ],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()
