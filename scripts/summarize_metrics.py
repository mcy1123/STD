#!/usr/bin/env python
"""Summarize STD JSONL metric files."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable, Sequence


def mean(values):
    return sum(values) / len(values) if values else 0.0


def print_distribution(label: str, values) -> None:
    values = [float(value) for value in values]
    if not values:
        return
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    print(f"  {label}: mean={mean(values):.3f} median={statistics.median(values):.3f} std={std:.3f}")


def mean_list_field(records: Sequence[dict], field: str) -> float:
    values = []
    for record in records:
        values.extend(float(value) for value in record.get(field, []))
    return mean(values)


def retained_k_plus_text(record: dict) -> float:
    if record.get("target_k_plus_text") is not None:
        return float(record["target_k_plus_text"])
    return float(record.get("k", 0) + record.get("text_len", 0))


def retained_ratio(record: dict) -> float:
    full_kv = float(record.get("visual_len", 0) + record.get("text_len", 0))
    if full_kv <= 0:
        return 0.0
    return retained_k_plus_text(record) / full_kv


def paper_speed_threshold(record: dict) -> float:
    gamma = float(record.get("gamma", 0))
    if gamma <= 0:
        return retained_ratio(record)
    return retained_ratio(record) + 1.0 / gamma


def format_key(record: dict, fields: Sequence[str]) -> str:
    return ", ".join(f"{field}={record.get(field)}" for field in fields)


def print_summary(label: str, records: Iterable[dict]) -> None:
    records = list(records)
    exact = sum(1 for r in records if r.get("token_equal"))
    print(label)
    print(f"  samples: {len(records)}")
    print(f"  token_equal: {exact}/{len(records)} ({exact / len(records):.3f})")
    print(f"  speedup: {mean([r['speedup'] for r in records]):.3f}x")
    print_distribution("speedup_distribution", [r["speedup"] for r in records])
    total_ar_decode = sum(float(r["ar_decoding_time"]) for r in records)
    total_std_decode = sum(float(r["std_decoding_time"]) for r in records)
    if total_std_decode:
        print(f"  overall_decode_speedup: {total_ar_decode / total_std_decode:.3f}x")
    print(f"  acceptance_rate: {mean([r['acceptance_rate'] for r in records]):.3f}")
    print(f"  mean_accept_length: {mean([r['mean_accept_length'] for r in records]):.3f}")
    if any(r.get("gamma_history") for r in records):
        print(f"  mean_gamma: {mean_list_field(records, 'gamma_history'):.3f}")
        print(f"  mean_propose_length: {mean_list_field(records, 'proposed_lengths'):.3f}")
        print(f"  mean_round_accept_length: {mean_list_field(records, 'accept_lengths'):.3f}")
    print(f"  retained_ratio: {mean([retained_ratio(r) for r in records]):.3f}")
    print(f"  paper_speed_threshold: {mean([paper_speed_threshold(r) for r in records]):.3f}")
    print(
        "  acceptance_minus_threshold: "
        f"{mean([r['acceptance_rate'] - paper_speed_threshold(r) for r in records]):.3f}"
    )
    print(f"  ar_decoding_time: {mean([r['ar_decoding_time'] for r in records]):.3f}s")
    print(f"  std_decoding_time: {mean([r['std_decoding_time'] for r in records]):.3f}s")
    input_time_fields = [
        "video_probe_time",
        "video_decode_sampling_time",
        "processor_time",
        "input_cache_load_time",
        "input_cache_save_time",
        "input_transfer_time",
        "input_preparation_time",
    ]
    if all(all(field in r for field in input_time_fields) for r in records):
        print("  input preparation:")
        for field in input_time_fields:
            print(f"    {field}: {mean([float(r[field]) for r in records]):.3f}s")
        cache_hits = sum(bool(r.get("input_cache_hit")) for r in records)
        print(f"    cache_hits: {cache_hits}/{len(records)}")
        decoded_counts = [r["decoded_frame_count"] for r in records if r.get("decoded_frame_count") is not None]
        if decoded_counts:
            print_distribution("decoded_frame_count", decoded_counts)
    if all("ar_tokens_per_second" in r and "std_tokens_per_second" in r for r in records):
        print_distribution("ar_tokens_per_second", [r["ar_tokens_per_second"] for r in records])
        print_distribution("std_tokens_per_second", [r["std_tokens_per_second"] for r in records])
    if all("ar_peak_memory_gib" in r and "std_peak_memory_gib" in r for r in records):
        print(f"  ar_peak_memory_gib: {mean([r['ar_peak_memory_gib'] for r in records]):.3f}")
        print(f"  std_peak_memory_gib: {mean([r['std_peak_memory_gib'] for r in records]):.3f}")
    if any("verify_margin_reruns" in r for r in records):
        print(f"  verify_margin_reruns: {mean([float(r.get('verify_margin_reruns', 0)) for r in records]):.3f}")
        print(f"  min_verify_margin: {min(float(r.get('min_verify_margin', 0.0)) for r in records):.6f}")
    if all("ar_inference_time" in r and "std_inference_time" in r for r in records):
        print(f"  ar_inference_time: {mean([r['ar_inference_time'] for r in records]):.3f}s")
        print(f"  std_inference_time: {mean([r['std_inference_time'] for r in records]):.3f}s")
        total_ar_inference = sum(float(r["ar_inference_time"]) for r in records)
        total_std_inference = sum(float(r["std_inference_time"]) for r in records)
        if total_std_inference:
            print(f"  overall_inference_speedup: {total_ar_inference / total_std_inference:.3f}x")
        if all("inference_speedup" in r for r in records):
            print(f"  mean_inference_speedup: {mean([r['inference_speedup'] for r in records]):.3f}x")
    prefill_fields = [
        "ar_cache_init_time",
        "ar_prefill_time",
        "std_cache_init_time",
        "std_selection_prefill_time",
        "std_selection_time",
        "std_dense_prefill_time",
        "std_sparse_cache_time",
    ]
    if any(r.get("profile_prefill") for r in records) and all(
        all(field in r for field in prefill_fields) for r in records
    ):
        print("  prefill profile:")
        for field in prefill_fields:
            print(f"    {field}: {mean([float(r[field]) for r in records]):.3f}s")
        module_fields = [
            "ar_vision_encoder_time",
            "ar_projector_time",
            "ar_dense_lm_prefill_time",
            "std_vision_encoder_time",
            "std_projector_time",
            "std_dense_lm_prefill_time",
            "ar_vision_encoder_forward_count",
            "ar_projector_forward_count",
            "std_vision_encoder_forward_count",
            "std_projector_forward_count",
        ]
        if all(all(field in r for field in module_fields) for r in records):
            for field in module_fields:
                print(f"    {field}: {mean([float(r[field]) for r in records]):.3f}")
    stage_fields = ["draft_time", "verify_time", "bonus_time", "cache_adjust_time"]
    if any(r.get("profile_decode") for r in records) and all(
        all(field in r for field in stage_fields) for r in records
    ):
        stage_total = mean([sum(float(r[field]) for field in stage_fields) for r in records])
        for field in stage_fields:
            field_mean = mean([float(r[field]) for r in records])
            share = field_mean / stage_total if stage_total else 0.0
            print(f"  {field}: {field_mean:.3f}s ({share:.1%} of profiled stages)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--group-by", default="", help="Comma-separated JSON fields to summarize separately.")
    args = parser.parse_args()
    group_fields = [field.strip() for field in args.group_by.split(",") if field.strip()]

    for path_str in args.paths:
        path = Path(path_str)
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            print(f"{path}: empty")
            continue

        print_summary(f"{path}", records)
        if group_fields:
            grouped = {}
            for record in records:
                key = tuple(record.get(field) for field in group_fields)
                grouped.setdefault(key, []).append(record)
            for _key, group_records in sorted(grouped.items(), key=lambda item: str(item[0])):
                print_summary(f"  [{format_key(group_records[0], group_fields)}]", group_records)


if __name__ == "__main__":
    main()
