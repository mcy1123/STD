#!/usr/bin/env python
"""Paired, warmed AR/static/dynamic comparison on one physical GPU.

Uses bounded streaming video decoding. All methods receive identical tensors,
precision, sparse backend and fixed output budget. JSONL retains every trial,
including warmups and output tokens; it does not assume token equality.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", required=True, type=int)
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--video-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--frame-num", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--skip-samples", type=int, default=0)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--warmup-tokens", type=int, default=16)
    p.add_argument("--gamma", type=int, default=9)
    p.add_argument("--k-plus-text", type=int, default=1024)
    p.add_argument("--max-pixels", type=int, default=448 * 448)
    p.add_argument("--cache-len", type=int, default=40960)
    p.add_argument("--sparse-attn-mode", choices=["gqa_sdpa", "triton_gqa"], default="gqa_sdpa")
    p.add_argument("--verify-fallback", choices=["none", "sequential_on_low_margin", "sequential_guard"], default="none")
    p.add_argument("--verify-margin-threshold", type=float, default=0.1)
    p.add_argument("--dynamic-collector", choices=["v1", "v2", "v3"], default="v2")
    p.add_argument("--refresh-mode", choices=["full", "incremental"], default="incremental")
    p.add_argument("--selection-update-interval", type=int, default=1)
    p.add_argument("--min-selection-change-ratio", type=float, default=0.05)
    p.add_argument("--dynamic-query-mode", choices=["two", "three"], default="three")
    p.add_argument("--dynamic-bootstrap", choices=["attention", "attention_free"], default="attention")
    p.add_argument("--bootstrap-coverage-ratio", type=float, default=0.25)
    p.add_argument("--bootstrap-value-weight", type=float, default=0.25)
    p.add_argument("--bootstrap-window-tokens", type=int, default=0,
                   help="0 infers one temporal slice from video_grid_thw")
    p.add_argument("--bootstrap-layer-stride", type=int, default=1)
    p.add_argument(
        "--assert-equal-s0",
        action="store_true",
        help=(
            "T2: fail the run unless the dynamic method starts from exactly the same "
            "initial selection as static (required for a clean paired comparison)."
        ),
    )
    p.add_argument(
        "--assert-consistency",
        action="store_true",
        help=(
            "T1: after every sparse-cache refresh, verify the compact prompt still "
            "matches the canonical dense KV at the selected positions."
        ),
    )
    p.add_argument("--profile-components", action="store_true",
                   help="record synchronized prefill/decode component timings")
    p.add_argument("--max-rss-gib", type=float, default=48)
    p.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help=(
            "Proceed on a non-idle GPU. Acceptance and exactness are still valid, "
            "wall-clock is not: the manifest records timing_valid=false and the "
            "summarizer will not report speedups. Requires enough free memory."
        ),
    )
    p.add_argument(
        "--min-free-gib",
        type=float,
        default=0.0,
        help="Refuse to start unless the selected GPU has at least this much free memory.",
    )
    args = p.parse_args()
    if min(args.limit, args.repeats, args.frame_num, args.max_new_tokens, args.warmup_tokens) < 1:
        p.error("sample, repeat, frame and token counts must be positive")
    if args.gpu < 0 or args.skip_samples < 0 or args.max_rss_gib <= 0 or args.selection_update_interval < 1:
        p.error("invalid GPU, skip count or memory limit")
    if not 0.0 <= args.min_selection_change_ratio <= 1.0:
        p.error("min-selection-change-ratio must be between 0 and 1")
    if not 0.0 <= args.bootstrap_coverage_ratio <= 1.0 or args.bootstrap_value_weight < 0:
        p.error("invalid attention-free bootstrap weights")
    if args.bootstrap_window_tokens < 0 or args.bootstrap_layer_stride < 1:
        p.error("invalid attention-free bootstrap window/layer stride")
    return args


def gpu_snapshot():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip()


def start_memory_guard(max_rss_gib):
    """Terminate only this experiment before it exhausts shared host RAM."""
    def monitor():
        page_size = os.sysconf("SC_PAGE_SIZE")
        while True:
            with open("/proc/self/statm") as f:
                rss = int(f.read().split()[1]) * page_size
            if rss > max_rss_gib * 1024**3:
                print(f"MEMORY_GUARD: RSS {rss / 1024**3:.2f} GiB exceeds {max_rss_gib}", flush=True)
                os.kill(os.getpid(), signal.SIGTERM)
                return
            time.sleep(0.5)
    threading.Thread(target=monitor, daemon=True).start()


def selection_digest(selection) -> str:
    """Stable digest of an initial visual selection (T2 equal-S_0 check)."""
    hasher = hashlib.sha256()
    for layer_positions in selection.topk_positions:
        hasher.update(layer_positions.detach().to("cpu").contiguous().numpy().tobytes())
    hasher.update(str(int(selection.k)).encode())
    return hasher.hexdigest()[:16]


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["SPECVLM_MAX_CACHE_LEN"] = str(args.cache_len)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    start_memory_guard(args.max_rss_gib)

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    import torch
    from std_repro.streaming_video import install_streaming_video_reader
    from benchmark_std import VIDEO_TOKEN_ID, iter_generic_hf_video, load_qwen_model, make_qwen_video_inputs
    from std_repro.std_qwen25vl import ar_generate_qwen25vl, std_generate_qwen25vl
    from std_repro.dynamic_std_qwen25vl import dynamic_std_generate_qwen25vl
    from std_repro.verification_policy import positional_token_metrics

    torch.set_num_threads(4)
    torch.manual_seed(42)
    install_streaming_video_reader()
    snapshot = gpu_snapshot()
    print(snapshot, flush=True)
    selected = [line.split(",") for line in snapshot.splitlines() if int(line.split(",")[0]) == args.gpu]
    gpu_idle = len(selected) == 1 and int(selected[0][3]) <= 256 and int(selected[0][5]) == 0
    if not gpu_idle:
        if not args.allow_shared_gpu:
            raise RuntimeError("Selected physical GPU is not idle; experiment not started")
        # Co-tenancy invalidates wall-clock comparisons but NOT acceptance or
        # exactness, which are deterministic for fixed inputs. The run is
        # recorded as timing-invalid and the summarizer refuses to report
        # speedups from it.
        print(f"SHARED_GPU: proceeding on a non-idle GPU ({selected[0][3]} MiB used, "
              f"{selected[0][5]}% util); timing is invalid, acceptance is not.", flush=True)
    if int(selected[0][4]) < args.min_free_gib * 1024:
        raise RuntimeError(
            f"Selected GPU has only {int(selected[0][4]) // 1024} GiB free; "
            f"--min-free-gib requires {args.min_free_gib}."
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")

    samples = list(iter_generic_hf_video(args.data_path, "test", args.skip_samples + args.limit,
                                       args.video_root, prompt_style="cot"))[args.skip_samples:]
    if len(samples) != args.limit:
        raise RuntimeError(f"Requested {args.limit} samples; only {len(samples)} available")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as f:
        def emit(record):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

        hashes = {}
        for rel in ("scripts/benchmark_a100_dynamic.py", "src/std_repro/streaming_video.py",
                    "src/std_repro/std_qwen25vl.py", "src/std_repro/dynamic_std_qwen25vl.py",
                    "src/std_repro/dynamic_selection.py", "src/std_repro/sparse_cache_refresh.py"):
            hashes[rel] = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        emit({"kind": "manifest", "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "host": socket.gethostname(), "physical_gpu": args.gpu,
              "gpu_snapshot": snapshot, "torch": torch.__version__, "cuda": torch.version.cuda,
              "timing_valid": bool(gpu_idle),
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "git_worktree_modified": git_dirty,
              "source_sha256": hashes, "args": vars(args), "dtype": "float16",
              "dataset": "Video-MME available chunk subset, seed=42 shuffle",
              "sample_ids": [s["sample_id"] for s in samples],
              "timing_note": "fixed-length; continue after EOS; model load and video processing excluded",
              "refresh_timing_note": "existing refresh sub-timer double-counts; not used; total decode timing is synchronized"})
        print("Loading model ...", flush=True)
        model, processor = load_qwen_model(args.model_path, "0")
        eos = processor.tokenizer.eos_token_id
        methods = ["ar", "static", f"dynamic_{args.dynamic_collector}"]

        for idx, sample in enumerate(samples):
            print(f"PREPARE {idx + 1}/{len(samples)} {sample['sample_id']}", flush=True)
            inputs, prep = make_qwen_video_inputs(processor, sample["video_path"], sample["question"],
                args.frame_num, None, args.max_pixels, target_device="cuda:0", return_timings=True)
            prompt_len = int(inputs["input_ids"].shape[1])
            if prompt_len + args.max_new_tokens + args.gamma + 2 > args.cache_len:
                raise RuntimeError("Prompt and generation exceed preallocated KV capacity")
            input_digest = hashlib.sha256()
            for key, value in sorted(inputs.items()):
                if torch.is_tensor(value):
                    input_digest.update(key.encode())
                    input_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            emit({"kind": "input", "sample_id": sample["sample_id"], "video": sample["video_path"],
                  "prompt_len": prompt_len, "sha256": input_digest.hexdigest(),
                  "gpu_snapshot": gpu_snapshot(), **prep})

            for repeat in range(-1, args.repeats):
                phase = "warmup" if repeat == -1 else "measure"
                tokens = args.warmup_tokens if repeat == -1 else args.max_new_tokens
                # Rotate order between repeats/samples to reduce fixed-order bias.
                shift = (idx + max(repeat, 0)) % len(methods)
                order = methods[shift:] + methods[:shift]
                trial_results = {}
                trial_digests = {}
                for method in order:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    common = dict(max_new_tokens=tokens, ignore_eos=True,
                                  profile_prefill=args.profile_components)
                    print(f"RUN {sample['sample_id']} {phase} repeat={repeat} {method}", flush=True)
                    dynamic_stats = None
                    digest = None
                    if method == "ar":
                        result = ar_generate_qwen25vl(model, inputs, VIDEO_TOKEN_ID, eos, **common)
                    else:
                        common["profile_decode"] = args.profile_components
                        common.update(gamma=args.gamma, target_k_plus_text=args.k_plus_text,
                            sparse_attn_mode=args.sparse_attn_mode, verify_fallback=args.verify_fallback,
                            verify_margin_threshold=None if args.verify_fallback == "none" else args.verify_margin_threshold)
                        if method == "static":
                            result, selection = std_generate_qwen25vl(model, inputs, VIDEO_TOKEN_ID, eos, **common)
                        else:
                            result, selection, dynamic_stats = dynamic_std_generate_qwen25vl(
                                model, inputs, VIDEO_TOKEN_ID, eos, policy="previous_verify_topk",
                                collector_version=args.dynamic_collector,
                                refresh_mode=args.refresh_mode,
                                assert_selection_cache_consistency=args.assert_consistency,
                                selection_update_interval=args.selection_update_interval,
                                min_selection_change_ratio=args.min_selection_change_ratio,
                                query_mode=args.dynamic_query_mode,
                                bootstrap_mode=args.dynamic_bootstrap,
                                bootstrap_coverage_ratio=args.bootstrap_coverage_ratio,
                                bootstrap_value_weight=args.bootstrap_value_weight,
                                bootstrap_window_tokens=args.bootstrap_window_tokens,
                                bootstrap_layer_stride=args.bootstrap_layer_stride,
                                **common)
                        digest = selection_digest(selection)
                        del selection
                    trial_digests[method] = digest
                    torch.cuda.synchronize()
                    suffix = result.output_ids[:, prompt_len:].detach().cpu()
                    rec = {"kind": "trial", "sample_id": sample["sample_id"], "phase": phase,
                           "repeat": repeat, "method": method, "order": order, "prompt_len": prompt_len,
                           "generate_len": result.generate_len, "decoding_time": result.decoding_time,
                           "inference_time": result.inference_time,
                           "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                           "acceptance_rate": result.acceptance_rate, "mean_accept_length": result.mean_accept_length,
                           "fallback_count": result.fallback_count, "decode_rounds": result.decode_rounds,
                           "accepted_draft_tokens": result.accepted_draft_tokens,
                           "proposed_draft_tokens": result.proposed_draft_tokens,
                           "prefill_time": result.prefill_time,
                           "cache_init_time": result.cache_init_time,
                           "selection_prefill_time": result.selection_prefill_time,
                           "selection_time": result.selection_time,
                           "dense_prefill_time": result.dense_prefill_time,
                           "sparse_cache_time": result.sparse_cache_time,
                           "draft_time": result.draft_time,
                           "verify_time": result.verify_time,
                           "bonus_time": result.bonus_time,
                           "cache_adjust_time": result.cache_adjust_time,
                           "initial_selection_digest": digest,
                           "output_tokens": suffix.tolist()[0]}
                    if dynamic_stats is not None:
                        rec["dynamic_stats"] = dynamic_stats
                    emit(rec)
                    trial_results[method] = (rec, suffix)
                    print(f"DONE {method}: decode={result.decoding_time:.3f}s inference={result.inference_time:.3f}s", flush=True)
                    del result
                    if suffix.numel() != tokens:
                        raise RuntimeError("Decoder did not emit the fixed token budget")

                ref = trial_results["ar"]
                static = trial_results["static"]
                # T2: with the attention bootstrap the dynamic method must start
                # from exactly the static S_0, otherwise the paired accept delta
                # is confounded by a different initial selection.
                equal_s0 = {}
                for method in [m for m in methods if m.startswith("dynamic_")]:
                    same = trial_digests.get(method) == trial_digests.get("static")
                    equal_s0[method] = same
                    if (
                        args.assert_equal_s0
                        and args.dynamic_bootstrap == "attention"
                        and not same
                    ):
                        raise RuntimeError(
                            f"T2 violated: {method} S_0 digest {trial_digests.get(method)} != "
                            f"static {trial_digests.get('static')} on "
                            f"{sample['sample_id']} repeat={repeat}"
                        )
                for method in methods[1:]:
                    rec, suffix = trial_results[method]
                    metrics = positional_token_metrics(ref[1], suffix)
                    summary = {"kind": "comparison", "sample_id": sample["sample_id"],
                               "phase": phase, "repeat": repeat, "method": method,
                               "token_equal": bool(torch.equal(ref[1], suffix)), **metrics,
                               "decode_speedup_vs_ar": ref[0]["decoding_time"] / rec["decoding_time"],
                               "decode_speedup_vs_static": static[0]["decoding_time"] / rec["decoding_time"],
                               "inference_speedup_vs_ar": ref[0]["inference_time"] / rec["inference_time"],
                               "inference_speedup_vs_static": static[0]["inference_time"] / rec["inference_time"],
                               "equal_s0": equal_s0.get(method)}
                    emit(summary)
                    print(json.dumps(summary), flush=True)
            del inputs, trial_results
            gc.collect()
            torch.cuda.empty_cache()
        emit({"kind": "complete", "samples": len(samples), "repeats": args.repeats,
              "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    print(f"COMPLETE {out_path}", flush=True)


if __name__ == "__main__":
    main()
