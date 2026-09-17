#!/usr/bin/env python
"""Small, disposable HSD feasibility spike on the idle A100 (GPU 1 only).

This runner deliberately keeps the four methods paired and fixed-length.  It
does not claim that ``hsd`` is lossless: the JSONL records exact comparisons
against AR so that a failed spike remains useful evidence.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Iterable, Mapping, Sequence


METHODS = ("ar", "static", "small_dense", "hsd")
TIMING_KEYS = (
    "decoding_time", "inference_time", "prefill_time", "cache_init_time",
    "selection_prefill_time", "selection_time", "dense_prefill_time",
    "sparse_cache_time", "draft_time", "verify_time", "bonus_time",
    "cache_adjust_time",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--draft-model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-num", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--gamma", type=int, default=9)
    parser.add_argument("--inner-gamma", type=int, default=3)
    parser.add_argument("--target-k-plus-text", type=int, default=1024)
    parser.add_argument("--cache-len", type=int, default=40960)
    parser.add_argument("--sparse-attn-mode", choices=("gqa_sdpa", "triton_gqa"), default="gqa_sdpa")
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--max-rss-gib", type=float, default=48)
    args = parser.parse_args(argv)
    if args.gpu != 1:
        parser.error("this spike is restricted to physical GPU 1")
    if min(args.frame_num, args.max_new_tokens, args.limit, args.repeats, args.warmup_tokens) < 1:
        parser.error("frame, token, sample, repeat and warmup counts must be positive")
    if args.gamma < 1 or args.inner_gamma < 1 or args.inner_gamma > args.gamma:
        parser.error("inner-gamma must be in [1, gamma]")
    if args.skip_samples < 0 or args.target_k_plus_text < 1 or args.cache_len < 1 or args.max_rss_gib <= 0:
        parser.error("invalid skip, cache, budget or memory limit")
    return args


def require_idle_gpu(snapshot: str, gpu: int = 1) -> None:
    """Refuse to start if the selected GPU is not idle; ignore other GPUs."""
    if gpu != 1:
        raise RuntimeError("spike is restricted to GPU 1")
    selected = []
    for line in snapshot.splitlines():
        fields = [field.strip() for field in line.split(",")]
        try:
            if int(fields[0]) == gpu:
                selected.append(fields)
        except (ValueError, IndexError):
            continue
    if not selected:
        raise RuntimeError(f"selected GPU {gpu} is not present")
    fields = selected[0]
    if len(fields) < 6 or int(float(fields[3])) > 256 or int(float(fields[5])) != 0:
        raise RuntimeError(f"selected GPU {gpu} is not idle")


def method_order(sample_index: int, repeat: int) -> list[str]:
    shift = (sample_index + max(repeat, 0)) % len(METHODS)
    return list(METHODS[shift:] + METHODS[:shift])


def _flat_tokens(tokens: Any) -> list[int]:
    if hasattr(tokens, "detach"):
        return [int(item) for item in tokens.detach().cpu().flatten().tolist()]
    if hasattr(tokens, "flatten") and not isinstance(tokens, (list, tuple)):
        tokens = tokens.flatten().tolist()
    return [int(item) for item in tokens]


def compare_tokens(reference: Any, candidate: Any) -> dict[str, Any]:
    ref = _flat_tokens(reference)
    got = _flat_tokens(candidate)
    compared = min(len(ref), len(got))
    mismatches = sum(left != right for left, right in zip(ref[:compared], got[:compared]))
    mismatches += abs(len(ref) - len(got))
    first = next((idx for idx, (left, right) in enumerate(zip(ref, got)) if left != right), None)
    if first is None and len(ref) != len(got):
        first = compared
    denominator = max(len(ref), len(got))
    return {
        "token_equal": ref == got,
        "mismatch_token_count": mismatches,
        "token_level_agreement": 1.0 - mismatches / denominator if denominator else 1.0,
        "first_mismatch_index": first,
        "reference_token_count": len(ref),
        "candidate_token_count": len(got),
    }


def summarize_trials(rows: Iterable[Mapping[str, Any]], sample_ids: Sequence[str], repeats: int) -> dict[str, dict[str, Any]]:
    measured = [row for row in rows if row.get("kind") == "trial" and row.get("phase") == "measure"]
    expected = {(sample, repeat, method) for sample in sample_ids for repeat in range(repeats) for method in METHODS}
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
    for row in measured:
        key = (str(row.get("sample_id")), int(row.get("repeat", -999)), str(row.get("method")))
        if key in grouped:
            raise ValueError(f"duplicate trial row: {key}")
        grouped[key] = [row]
    if set(grouped) != expected:
        missing = expected - set(grouped)
        extra = set(grouped) - expected
        raise ValueError(f"trial pairing mismatch; missing={sorted(missing)} extra={sorted(extra)}")

    totals: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        medians: dict[str, float] = {}
        for key in TIMING_KEYS:
            values = [float(grouped[(sample, repeat, method)][0].get(key, 0.0))
                      for sample in sample_ids for repeat in range(repeats)]
            # One median per sample avoids a sample with a slow repeat dominating.
            sample_medians = [statistics.median(
                float(grouped[(sample, repeat, method)][0].get(key, 0.0)) for repeat in range(repeats)
            ) for sample in sample_ids]
            medians[key] = sum(sample_medians)
        exact = 0
        for sample in sample_ids:
            for repeat in range(repeats):
                if method == "ar":
                    exact += 1
                else:
                    ar = grouped[(sample, repeat, "ar")][0].get("output_tokens", [])
                    got = grouped[(sample, repeat, method)][0].get("output_tokens", [])
                    exact += int(_flat_tokens(ar) == _flat_tokens(got))
        totals[method] = {**medians, "exact_trials": exact,
                          "total_trials": len(sample_ids) * repeats}
    for method in METHODS:
        totals[method]["decode_speedup_vs_ar"] = totals["ar"]["decoding_time"] / totals[method]["decoding_time"]
        totals[method]["decode_speedup_vs_static"] = totals["static"]["decoding_time"] / totals[method]["decoding_time"]
        totals[method]["inference_speedup_vs_ar"] = totals["ar"]["inference_time"] / totals[method]["inference_time"]
        totals[method]["lossless_speedup_eligible"] = totals[method]["exact_trials"] == totals[method]["total_trials"]
    return totals


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _snapshot() -> str:
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip()


def _memory_guard(max_rss_gib: float) -> None:
    def monitor() -> None:
        page_size = os.sysconf("SC_PAGE_SIZE")
        while True:
            with open("/proc/self/statm", encoding="utf-8") as handle:
                rss = int(handle.read().split()[1]) * page_size
            if rss > max_rss_gib * 1024**3:
                print(f"MEMORY_GUARD: RSS {rss / 1024**3:.2f} GiB exceeds {max_rss_gib}", flush=True)
                os.kill(os.getpid(), signal.SIGTERM)
                return
            time.sleep(0.5)
    threading.Thread(target=monitor, daemon=True).start()


def _clone_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if hasattr(value, "clone") else value for key, value in inputs.items()}


def _result_record(result: Any, method: str, sample_id: str, phase: str, repeat: int,
                   order: Sequence[str], prompt_len: int) -> tuple[dict[str, Any], Any]:
    suffix = result.output_ids[:, prompt_len:].detach().cpu()
    record = {"kind": "trial", "method": method, "sample_id": sample_id, "phase": phase,
              "repeat": repeat, "order": list(order), "prompt_len": prompt_len,
              "generate_len": int(getattr(result, "generate_len", suffix.shape[-1])),
              "output_tokens": suffix.tolist()[0]}
    for key in TIMING_KEYS:
        record[key] = float(getattr(result, key, 0.0))
    for key in ("acceptance_rate", "mean_accept_length", "decode_rounds", "accepted_draft_tokens",
                "proposed_draft_tokens", "fallback_count", "peak_memory_gib"):
        if hasattr(result, key):
            value = getattr(result, key)
            record[key] = float(value) if isinstance(value, (int, float)) else _jsonable(value)
    return record, suffix


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["SPECVLM_MAX_CACHE_LEN"] = str(args.cache_len)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    _memory_guard(args.max_rss_gib)

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    import torch
    from std_repro.streaming_video import install_streaming_video_reader
    from benchmark_std import VIDEO_TOKEN_ID, iter_generic_hf_video, load_qwen_model, make_qwen_video_inputs
    from std_repro.std_qwen25vl import ar_generate_qwen25vl, std_generate_qwen25vl
    from std_repro.hsd_spike import hsd_generate_qwen25vl

    install_streaming_video_reader()
    snapshot = _snapshot()
    require_idle_gpu(snapshot, args.gpu)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    samples = list(iter_generic_hf_video(args.data_path, "test", args.skip_samples + args.limit,
                                         args.video_root, prompt_style="cot"))[args.skip_samples:]
    if len(samples) != args.limit:
        raise RuntimeError(f"requested {args.limit} samples, found {len(samples)}")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source_paths = ["scripts/benchmark_a100_hsd_spike.py", "scripts/benchmark_std.py",
                    "src/std_repro/std_qwen25vl.py", "src/std_repro/dynamic_selection.py"]
    hsd_path = root / "src/std_repro/hsd_spike.py"
    if hsd_path.exists():
        source_paths.append("src/std_repro/hsd_spike.py")
    hashes = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in source_paths}
    with out_path.open("x", encoding="utf-8") as handle:
        def emit(item: Mapping[str, Any]) -> None:
            handle.write(json.dumps(_jsonable(dict(item)), ensure_ascii=False) + "\n")
            handle.flush()

        emit({"kind": "manifest", "label": "hsd-feasibility-spike", "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "physical_gpu": args.gpu, "gpu_snapshot": snapshot, "dtype": "float16",
              "torch": torch.__version__, "cuda": torch.version.cuda, "source_sha256": hashes,
              "args": vars(args), "sample_ids": [sample["sample_id"] for sample in samples],
              "timing_note": "fixed-length; ignore EOS; excludes model loading and video processing"})
        print("Loading target and draft models ...", flush=True)
        target_model, processor = load_qwen_model(args.model_path, "0")
        draft_model, _ = load_qwen_model(args.draft_model_path, "0")
        eos = processor.tokenizer.eos_token_id
        for sample_index, sample in enumerate(samples):
            inputs, prep = make_qwen_video_inputs(processor, sample["video_path"], sample["question"],
                args.frame_num, None, args.max_pixels, target_device="cuda:0", return_timings=True)
            prompt_len = int(inputs["input_ids"].shape[1])
            if prompt_len + args.max_new_tokens + args.gamma + 2 > args.cache_len:
                raise RuntimeError("prompt and generation exceed cache capacity")
            emit({"kind": "input", "sample_id": sample["sample_id"], **prep})
            for repeat in range(-1, args.repeats):
                phase = "warmup" if repeat < 0 else "measure"
                tokens = args.warmup_tokens if repeat < 0 else args.max_new_tokens
                order = method_order(sample_index, repeat)
                trial_results: dict[str, tuple[dict[str, Any], Any]] = {}
                for method in order:
                    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
                    trial_inputs = _clone_inputs(inputs)
                    if method == "ar":
                        result = ar_generate_qwen25vl(
                            target_model, trial_inputs, VIDEO_TOKEN_ID, eos,
                            max_new_tokens=tokens, ignore_eos=True, profile_prefill=True,
                        )
                    elif method == "static":
                        result, selection = std_generate_qwen25vl(
                            target_model, trial_inputs, VIDEO_TOKEN_ID, eos, gamma=args.gamma,
                            target_k_plus_text=args.target_k_plus_text, sparse_attn_mode=args.sparse_attn_mode,
                            max_new_tokens=tokens, ignore_eos=True, profile_prefill=True, profile_decode=True,
                        )
                        del selection
                    else:
                        result, hsd_stats = hsd_generate_qwen25vl(
                            target_model, draft_model, trial_inputs, VIDEO_TOKEN_ID, eos,
                            gamma=args.gamma, inner_gamma=args.inner_gamma,
                            target_k_plus_text=args.target_k_plus_text, mode=method,
                            max_new_tokens=tokens, ignore_eos=True, profile_decode=True,
                        )
                    torch.cuda.synchronize()
                    rec, suffix = _result_record(result, method, sample["sample_id"], phase, repeat, order, prompt_len)
                    rec["peak_memory_gib"] = float(torch.cuda.max_memory_allocated() / 1024**3)
                    if method in ("small_dense", "hsd"):
                        rec["hsd_stats"] = _jsonable(hsd_stats)
                    emit(rec); trial_results[method] = (rec, suffix)
                    del result, trial_inputs
                ar_suffix = trial_results["ar"][1]
                for method in METHODS[1:]:
                    rec, suffix = trial_results[method]
                    comparison = {"kind": "comparison", "sample_id": sample["sample_id"], "phase": phase,
                                  "repeat": repeat, "method": method, **compare_tokens(ar_suffix, suffix)}
                    emit(comparison)
            del inputs
            gc.collect(); torch.cuda.empty_cache()
        # Re-read records from memory would be wasteful; the summary helper is
        # also intentionally available to CPU callers and tests.
        handle.flush()
    print(f"COMPLETE {out_path}", flush=True)


if __name__ == "__main__":
    main()
