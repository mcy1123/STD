"""Verification-Guided STD (MVP): Fixed-K Previous Verification Guided Top-K Selection.

This is a *separate* entry point from the frozen static baseline in
``std_qwen25vl.py``. It reuses that module's prefill / compaction / sparse-draft /
append helpers verbatim, and only changes the decode loop to (a) observe the dense
verifier's visual attention with the read-only runtime collector, and (b) refresh
the sparse cache's visual prefix to ``TopK(A_t)`` after every verification round.

Correctness invariants (unchanged from the static baseline):

  * the dense verifier is a separate canonical cache fed with ``output_attentions=False``;
  * the speculative acceptance loop is byte-for-byte identical to ``parallel`` mode;
  * the collector and refresh never alter logits, KV layout beyond the visual
    prefix, or the attention backend.

Supported policies: ``static`` (S_next = S_0, identical to the frozen baseline) and
``previous_verify_topk`` (S_next = TopK(A_t)).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch

from specvlm.kv_cache.kv_cache import initialize_past_key_values
from std_repro.dynamic_selection import (
    PreviousVerifyTopKPolicy,
    RuntimeVerificationCollector,
    SelectionState,
    StaticPolicy,
    VerificationCollectorV2,
    VerificationCollectorV3,
    attention_free_visual_topk,
    should_refresh_selection,
    verification_query_positions,
)
from std_repro.sparse_cache_refresh import (
    count_changed_tokens,
    incremental_refresh_sparse_visual_kv,
    refresh_sparse_visual_kv,
    verify_sparse_visual_consistency,
)
from std_repro.std_qwen25vl import (
    GenerateResult,
    SparseDraftController,
    SparseSelection,
    _append_sparse_tokens,
    _contains_eos,
    _draft_tokens,
    _fill_cache_length,
    _first_model_device,
    _min_prediction_margin,
    _needs_sequential_fallback,
    _profile_mark,
    _sequential_verify_draft,
    _split_video_text_inputs,
    _token_argmax,
    build_sparse_selection,
    compact_sparse_prompt_cache,
    copy_prompt_cache,
)


def build_attention_free_selection_from_cache(
    past_key_values,
    full_input_ids: torch.Tensor,
    video_token_id: int,
    text_start: int,
    target_k_plus_text: int = 1024,
    explicit_k: Optional[int] = None,
    coverage_ratio: float = 0.25,
    value_weight: float = 0.25,
    window_size: int = 0,
    layer_stride: int = 1,
) -> SparseSelection:
    """Build the initial visual mask from canonical video-prefix K/V only."""
    prompt_ids = full_input_ids[0]
    visual_positions = torch.nonzero(prompt_ids == video_token_id, as_tuple=False).flatten().cpu()
    if visual_positions.numel() == 0:
        raise ValueError("Cannot build attention-free selection without visual/video token positions.")
    non_visual_positions = torch.nonzero(prompt_ids != video_token_id, as_tuple=False).flatten().cpu()
    prompt_len = int(prompt_ids.numel())
    text_len = max(1, prompt_len - int(text_start))
    k = int(explicit_k) if explicit_k is not None else max(1, int(target_k_plus_text) - text_len)
    k = min(k, int(visual_positions.numel()))
    device = past_key_values[0][0].data.device
    visual_gpu = visual_positions.to(device)
    selected_layers = past_key_values[::layer_stride]
    keys = torch.stack([layer[0].data[0, :, visual_gpu, :].detach() for layer in selected_layers], dim=0)
    values = torch.stack([layer[1].data[0, :, visual_gpu, :].detach() for layer in selected_layers], dim=0)
    local_topk = attention_free_visual_topk(
        keys, values, k=k, coverage_ratio=coverage_ratio, value_weight=value_weight,
        window_size=window_size,
    ).cpu()
    selected = torch.sort(visual_positions[local_topk], dim=-1).values
    return SparseSelection(
        topk_positions=[selected.clone() for _ in range(len(past_key_values))],
        non_visual_positions=non_visual_positions,
        prompt_len=prompt_len,
        visual_len=int(visual_positions.numel()),
        text_len=text_len,
        k=k,
    )


@torch.inference_mode()
def dynamic_std_generate_qwen25vl(
    model,
    inputs: Dict[str, torch.Tensor],
    video_token_id: int,
    eos_token_id: int,
    max_new_tokens: int = 256,
    gamma: int = 9,
    target_k_plus_text: int = 1024,
    explicit_k: Optional[int] = None,
    policy: str = "previous_verify_topk",
    profile_decode: bool = False,
    profile_prefill: bool = False,
    sparse_attn_mode: str = "gqa_sdpa",
    copy_sparse_prefill: bool = True,
    ignore_eos: bool = False,
    collector_version: str = "v2",
    refresh_mode: str = "incremental",
    assert_selection_cache_consistency: bool = False,
    consistency_check_limit: int = 3,
    selection_update_interval: int = 1,
    min_selection_change_ratio: float = 0.05,
    query_mode: str = "three",
    bootstrap_mode: str = "attention",
    bootstrap_coverage_ratio: float = 0.25,
    bootstrap_value_weight: float = 0.25,
    bootstrap_window_tokens: int = 0,
    bootstrap_layer_stride: int = 1,
    verify_fallback: str = "none",
    verify_margin_threshold: Optional[float] = None,
    sequential_fallback_max_accept: int = 1,
) -> Tuple[GenerateResult, SparseSelection, Dict]:
    """Verification-guided greedy decoding with dynamic visual selection.

    Returns ``(result, selection, dynamic_stats)`` where ``dynamic_stats`` holds
    per-round selection metrics and aggregate refresh timing.
    """
    if policy not in {"static", "previous_verify_topk"}:
        raise ValueError(f"Unsupported policy={policy!r}; expected 'static' or 'previous_verify_topk'.")
    if collector_version not in {"v1", "v2", "v3"}:
        raise ValueError(f"Unsupported collector_version={collector_version!r}; expected 'v1', 'v2' or 'v3'.")
    if refresh_mode not in {"full", "incremental"}:
        raise ValueError(f"Unsupported refresh_mode={refresh_mode!r}; expected 'full' or 'incremental'.")
    if verify_fallback not in {
        "none",
        "sequential_on_reject",
        "sequential_on_low_accept",
        "sequential_on_low_margin",
        "sequential_guard",
    }:
        raise ValueError(f"Unsupported verify_fallback={verify_fallback!r}.")
    if verify_fallback == "sequential_on_low_margin" and verify_margin_threshold is None:
        raise ValueError("sequential_on_low_margin requires verify_margin_threshold.")
    if verify_margin_threshold is not None and verify_margin_threshold < 0:
        raise ValueError("verify_margin_threshold must be non-negative.")
    if sequential_fallback_max_accept < 0:
        raise ValueError("sequential_fallback_max_accept must be non-negative.")
    if selection_update_interval < 1:
        raise ValueError("selection_update_interval must be positive.")
    if not 0.0 <= min_selection_change_ratio <= 1.0:
        raise ValueError("min_selection_change_ratio must be between 0 and 1.")
    if query_mode not in {"two", "three"}:
        raise ValueError("query_mode must be 'two' or 'three'.")
    if bootstrap_mode not in {"attention", "attention_free"}:
        raise ValueError("bootstrap_mode must be 'attention' or 'attention_free'.")
    if not 0.0 <= bootstrap_coverage_ratio <= 1.0:
        raise ValueError("bootstrap_coverage_ratio must be between 0 and 1.")
    if bootstrap_value_weight < 0:
        raise ValueError("bootstrap_value_weight must be non-negative.")
    if bootstrap_window_tokens < 0 or bootstrap_layer_stride < 1:
        raise ValueError("bootstrap_window_tokens must be non-negative and bootstrap_layer_stride positive.")

    torch.cuda.synchronize()
    start = time.time()

    stage_start = _profile_mark(profile_prefill)
    selection_pkv, _, _ = initialize_past_key_values(model)
    cache_init_time = _profile_mark(profile_prefill) - stage_start if profile_prefill else 0.0

    stage_start = _profile_mark(profile_prefill)
    input_ids, text_start, text_input_ids, video_inputs = _split_video_text_inputs(
        inputs, video_token_id, _first_model_device(model)
    )
    prompt_ids = input_ids.clone()

    # 1. Canonical video-prefix prefill, shared by selection and dense branches.
    model(**video_inputs, past_key_values=selection_pkv)
    # 2. Dense verifier cache: separate canonical cache (Scheme B correctness invariant).
    dense_pkv, _, dense_lengths = initialize_past_key_values(model)
    copy_prompt_cache(selection_pkv, dense_pkv, dense_lengths, text_start)
    # 3. Selection branch (custom attention) -> static S_0, never enters dense verifier.
    # The attention-free option derives the initial mask from the canonical
    # video-prefix K/V and therefore skips this extra text prefill.
    attentions = None
    if bootstrap_mode == "attention":
        selection_output = model(input_ids=text_input_ids, past_key_values=selection_pkv, output_attentions=True)
        attentions = selection_output.attentions
    # 4. Dense branch (canonical) -> verifier KV + canonical next token.
    dense_output = model(input_ids=text_input_ids, past_key_values=dense_pkv, output_attentions=False)
    if dense_output.attentions is not None:
        raise RuntimeError("Correctness invariant violated: dense prefill returned attentions.")
    dense_next = _token_argmax(dense_output.logits)
    selection_prefill_time = _profile_mark(profile_prefill) - stage_start if profile_prefill else 0.0
    stage_start = _profile_mark(profile_prefill)
    if bootstrap_mode == "attention":
        if attentions is None:
            raise RuntimeError("Selection prefill did not return attentions.")
        selection = build_sparse_selection(
            attentions, prompt_ids, video_token_id, text_start,
            target_k_plus_text=target_k_plus_text, explicit_k=explicit_k,
            num_key_value_heads=model.config.num_key_value_heads,
        )
        del attentions
    else:
        # Qwen merges each temporal slice's H*W visual patches spatially.
        # Auto windows follow those temporal slices rather than a global
        # centroid, which can mistake between-frame offsets for importance.
        if bootstrap_window_tokens == 0:
            grid = inputs.get("video_grid_thw")
            if grid is not None and grid.shape[0] == 1:
                merge = model.config.vision_config.spatial_merge_size
                bootstrap_window_tokens = int(grid[0, 1] * grid[0, 2]) // (merge * merge)
        selection = build_attention_free_selection_from_cache(
            selection_pkv, prompt_ids, video_token_id, text_start,
            target_k_plus_text=target_k_plus_text, explicit_k=explicit_k,
            coverage_ratio=bootstrap_coverage_ratio, value_weight=bootstrap_value_weight,
            window_size=bootstrap_window_tokens, layer_stride=bootstrap_layer_stride,
        )
    selection_time = _profile_mark(profile_prefill) - stage_start if profile_prefill else 0.0
    controller = SparseDraftController(model, selection, sparse_attn_mode=sparse_attn_mode, use_compile=False)
    controller.install()
    del selection_pkv

    dense_prefill_time = 0.0
    stage_start = _profile_mark(profile_prefill)
    sparse_pkv, _, sparse_lengths = initialize_past_key_values(model)
    if copy_sparse_prefill:
        sparse_next = dense_next.clone()
        copy_prompt_cache(dense_pkv, sparse_pkv, sparse_lengths, int(prompt_ids.shape[1]))
    else:
        from std_repro.std_qwen25vl import prefill_prompt
        _, sparse_next, _, _ = prefill_prompt(model, inputs, sparse_pkv, video_token_id, output_attentions=False)
    sparse_prompt_len = compact_sparse_prompt_cache(sparse_pkv, sparse_lengths, selection)
    sparse_cache_time = _profile_mark(profile_prefill) - stage_start if profile_prefill else 0.0

    # Visual positions / lengths needed by the collector and the refresh.
    visual_positions = torch.nonzero(prompt_ids[0] == video_token_id, as_tuple=False).flatten().cpu()
    k = selection.k
    non_visual_positions = selection.non_visual_positions

    if collector_version == "v2":
        collector = VerificationCollectorV2(model, visual_positions)
    elif collector_version == "v3":
        collector = VerificationCollectorV3(model, visual_positions)
    else:
        collector = RuntimeVerificationCollector(model, visual_positions)
    collector.install()
    policy_impl = StaticPolicy() if policy == "static" else PreviousVerifyTopKPolicy(visual_positions)
    state = policy_impl.initialize(selection.topk_positions, k)

    generated: List[int] = []
    accepted_total = 0
    proposed_total = 0
    accept_lengths: List[int] = []
    proposed_lengths: List[int] = []
    dense_pending: List[int] = []
    draft_time = 0.0
    verify_time = 0.0
    bonus_time = 0.0
    cache_adjust_time = 0.0
    decode_rounds = 0
    refresh_records: List[Dict] = []
    selection_update_time = 0.0
    skipped_selection_updates = 0
    consistency_checks = 0
    consistency_mismatches = 0
    fallback_count = 0
    fallback_accepted_extra = 0
    verify_margin_reruns = 0
    verify_margins: List[float] = []

    torch.cuda.synchronize()
    decode_start = time.time()
    device = _first_model_device(model)

    while len(generated) < max_new_tokens:
        decode_rounds += 1
        remaining = max_new_tokens - len(generated)
        propose_len = min(gamma, remaining)
        context_len = prompt_ids.shape[1] + len(generated)
        dense_cached_len = context_len - len(dense_pending)
        sparse_prev_len = sparse_prompt_len + len(generated)
        collect_due = decode_rounds % selection_update_interval == 0
        update_due = policy == "previous_verify_topk" and collect_due

        # Step 1: sparse draft using current selection S_t.
        stage_start = _profile_mark(profile_decode)
        draft = _draft_tokens(model, sparse_next, propose_len, sparse_pkv, controller, start_position=context_len)
        draft_time += _profile_mark(profile_decode) - stage_start
        proposed_total += len(draft)
        proposed_lengths.append(len(draft))

        # Step 2: dense verification (canonical, exact) + observe A_t.
        stage_start = _profile_mark(profile_decode)
        verify_input = dense_pending + draft
        verify_tensor = torch.tensor([verify_input], dtype=torch.long, device=device)
        if collect_due:
            collector.begin_verification(decode_rounds)
        verify_outputs = model(input_ids=verify_tensor, past_key_values=dense_pkv)
        if collect_due:
            collector.end_verification()
        verify_argmax = torch.argmax(verify_outputs.logits[0], dim=-1).tolist()
        if dense_pending:
            dense_predictions = [int(x) for x in verify_argmax]
        else:
            dense_predictions = [int(dense_next.item())] + [int(x) for x in verify_argmax]

        accept_len = 0
        while accept_len < len(draft) and draft[accept_len] == dense_predictions[accept_len]:
            accept_len += 1
        bonus_token = dense_predictions[accept_len]

        low_margin = False
        if verify_margin_threshold is not None:
            margin = _min_prediction_margin(
                verify_outputs.logits,
                range(accept_len + 1),
                has_dense_pending=bool(dense_pending),
            )
            if margin is not None:
                verify_margins.append(margin)
                low_margin = margin < verify_margin_threshold

        if _needs_sequential_fallback(
            verify_fallback,
            accept_len=accept_len,
            draft_len=len(draft),
            sequential_fallback_max_accept=sequential_fallback_max_accept,
            low_margin=low_margin,
        ):
            fallback_count += 1
            if low_margin:
                verify_margin_reruns += 1
            seq_accept_len, seq_bonus_token = _sequential_verify_draft(
                model,
                draft,
                dense_pending,
                dense_next,
                dense_pkv,
                dense_lengths,
                dense_cached_len,
            )
            if seq_accept_len > accept_len:
                fallback_accepted_extra += seq_accept_len - accept_len
            accept_len = seq_accept_len
            bonus_token = seq_bonus_token
        verify_time += _profile_mark(profile_decode) - stage_start
        accept_lengths.append(accept_len)
        accepted_total += accept_len

        # Step 3-4: update selection S_{t+1} = TopK(A_t) and refresh the sparse
        # cache's visual prefix in place (non-visual + generated KV untouched).
        query_positions = []
        if collector_version == "v2":
            positions = verification_query_positions(
                accept_len, len(dense_pending), propose_len, mode=query_mode
            )
            query_positions = positions
            A_t = collector.compute(positions, dense_pkv) if collect_due else None
        else:
            A_t = collector.latest_scores()
        candidate_state = policy_impl.update(A_t, state, k) if update_due else state
        if update_due:
            selection_update_time += float(candidate_state.update_time)
        if not update_due:
            skipped_selection_updates += 1
        changed_ratio = 1.0 - float(candidate_state.selection_overlap)
        accept_update = update_due and should_refresh_selection(
            candidate_state.round_id,
            changed_ratio,
            interval=1,
            min_changed_ratio=min_selection_change_ratio,
        )
        if accept_update:
            new_state = candidate_state
        else:
            # Preserve the cache/selection when the candidate is stale or only
            # marginally different.  Keep round accounting for diagnostics.
            new_state = SelectionState(
                indices=state.indices,
                k=state.k,
                round_id=candidate_state.round_id,
                selection_overlap=1.0,
                update_time=candidate_state.update_time,
            )
        refresh_time = 0.0
        cache_consistent = None
        if policy == "previous_verify_topk" and not torch.equal(state.indices, new_state.indices):
            if refresh_mode == "incremental":
                refresh_time = incremental_refresh_sparse_visual_kv(
                    sparse_pkv, dense_pkv, non_visual_positions, state.indices, new_state.indices, k
                )
            else:
                refresh_time = refresh_sparse_visual_kv(
                    sparse_pkv, dense_pkv, non_visual_positions, new_state.indices, k
                )
            # T1 invariant: the refreshed compact prompt must equal the canonical
            # dense KV at the selected positions. A mismatch means the routing
            # state and the cache contents drifted apart, which silently degrades
            # the draft without changing any verifier logit.
            if assert_selection_cache_consistency and consistency_checks < consistency_check_limit:
                consistency_checks += 1
                mismatches = verify_sparse_visual_consistency(
                    sparse_pkv, dense_pkv, non_visual_positions, new_state.indices, k
                )
                consistency_mismatches += mismatches
                cache_consistent = mismatches == 0
                if mismatches:
                    raise RuntimeError(
                        f"selection/cache drift at round {decode_rounds}: {mismatches} "
                        "compact slots disagree with the canonical dense cache"
                    )
        refresh_records.append(
            {
                "round_id": decode_rounds,
                "jaccard_old_new": float(new_state.selection_overlap),
                "changed_ratio": float(1.0 - new_state.selection_overlap),
                "changed_tokens": float(count_changed_tokens(state.indices, new_state.indices)),
                "refresh_time_ms": float(refresh_time * 1000.0),
                "update_applied": bool(accept_update),
                "cache_consistent": cache_consistent,
                "query_positions": query_positions,
                "query_mode_effective": query_mode if collector_version == "v2" else "all",
            }
        )
        state = new_state

        # Step 5: commit accepted draft + bonus (identical to static baseline).
        full_append = draft[:accept_len] + [bonus_token]
        eos_index = None if ignore_eos else _contains_eos(full_append, eos_token_id)
        if eos_index is not None:
            full_append = full_append[: eos_index + 1]

        to_append = full_append[:remaining]
        generated.extend(int(x) for x in to_append)

        reached_limit = len(generated) >= max_new_tokens
        reached_eos = bool(not ignore_eos and to_append and to_append[-1] == eos_token_id)
        if reached_limit or reached_eos:
            break

        stage_start = _profile_mark(profile_decode)
        _fill_cache_length(dense_lengths, dense_cached_len + len(dense_pending) + accept_len)
        _fill_cache_length(sparse_lengths, sparse_prev_len + accept_len)
        cache_adjust_time += _profile_mark(profile_decode) - stage_start

        dense_pending = [bonus_token]
        stage_start = _profile_mark(profile_decode)
        sparse_next = _append_sparse_tokens(
            model,
            [bonus_token],
            sparse_pkv,
            controller,
            start_position=context_len + accept_len,
        )
        bonus_time += _profile_mark(profile_decode) - stage_start

    collector.uninstall()

    torch.cuda.synchronize()
    end = time.time()
    out = torch.cat(
        [prompt_ids.to(device), torch.tensor([generated], dtype=torch.long, device=device)],
        dim=1,
    )
    mean_accept = sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0.0
    result = GenerateResult(
        output_ids=out,
        decoding_time=end - decode_start,
        inference_time=end - start,
        generate_len=len(generated),
        accepted_draft_tokens=accepted_total,
        proposed_draft_tokens=proposed_total,
        mean_accept_length=mean_accept,
        decode_rounds=decode_rounds,
        final_gamma=gamma,
        fallback_count=fallback_count,
        fallback_accepted_extra=fallback_accepted_extra,
        verify_margin_reruns=verify_margin_reruns,
        min_verify_margin=min(verify_margins) if verify_margins else 0.0,
        proposed_lengths=proposed_lengths,
        accept_lengths=accept_lengths,
        draft_time=draft_time,
        verify_time=verify_time,
        bonus_time=bonus_time,
        cache_adjust_time=cache_adjust_time,
        cache_init_time=cache_init_time,
        prefill_time=selection_prefill_time + dense_prefill_time,
        selection_prefill_time=selection_prefill_time,
        selection_time=selection_time,
        dense_prefill_time=dense_prefill_time,
        sparse_cache_time=sparse_cache_time,
    )

    refresh_ms = [r["refresh_time_ms"] for r in refresh_records]
    dynamic_stats = {
        "policy": policy,
        "collector_version": collector_version,
        "refresh_mode": refresh_mode,
        "query_mode": query_mode,
        "bootstrap_mode": bootstrap_mode,
        "bootstrap_coverage_ratio": bootstrap_coverage_ratio,
        "bootstrap_value_weight": bootstrap_value_weight,
        "bootstrap_window_tokens": bootstrap_window_tokens,
        "bootstrap_layer_stride": bootstrap_layer_stride,
        "total_collect_time_ms": float(collector.collect_time * 1000.0),
        "per_round": refresh_records,
        "mean_jaccard_old_new": float(sum(r["jaccard_old_new"] for r in refresh_records) / len(refresh_records))
        if refresh_records
        else 1.0,
        "mean_changed_ratio": float(sum(r["changed_ratio"] for r in refresh_records) / len(refresh_records))
        if refresh_records
        else 0.0,
        "mean_changed_tokens": float(sum(r["changed_tokens"] for r in refresh_records) / len(refresh_records))
        if refresh_records
        else 0.0,
        "total_refresh_time_ms": float(sum(refresh_ms)),
        "mean_refresh_time_ms": float(sum(refresh_ms) / len(refresh_ms)) if refresh_ms else 0.0,
        "total_selection_update_time_ms": float(selection_update_time * 1000.0),
        "skipped_selection_updates": skipped_selection_updates,
        # T3: the applied ratio is what actually reached the sparse cache. With
        # interval=1 it should be ~1.0; with interval=4 only ~0.25 of rounds are
        # even eligible, and hysteresis can suppress more (the A100 interval4 run
        # applied just 5/23). Reporting it prevents misreading a throttled run as
        # a full-strength dynamic run.
        "selection_updates_eligible": sum(
            1 for r in refresh_records if r.get("query_positions") is not None
        ),
        "selection_updates_applied": sum(
            1 for r in refresh_records if r.get("update_applied")
        ),
        "selection_update_applied_ratio": (
            sum(1 for r in refresh_records if r.get("update_applied")) / len(refresh_records)
            if refresh_records
            else 0.0
        ),
        "consistency_checks": consistency_checks,
        "consistency_mismatches": consistency_mismatches,
        "collector_time_synchronized": collector_version in {"v1", "v2"},
    }
    return result, selection, dynamic_stats
