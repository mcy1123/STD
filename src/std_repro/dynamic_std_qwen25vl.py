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
    StaticPolicy,
    VerificationCollectorV2,
    VerificationCollectorV3,
    verification_query_positions,
)
from std_repro.sparse_cache_refresh import (
    count_changed_tokens,
    incremental_refresh_sparse_visual_kv,
    refresh_sparse_visual_kv,
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
    collector_version: str = "v1",
    refresh_mode: str = "full",
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
    selection_output = model(input_ids=text_input_ids, past_key_values=selection_pkv, output_attentions=True)
    attentions = selection_output.attentions
    # 4. Dense branch (canonical) -> verifier KV + canonical next token.
    dense_output = model(input_ids=text_input_ids, past_key_values=dense_pkv, output_attentions=False)
    if dense_output.attentions is not None:
        raise RuntimeError("Correctness invariant violated: dense prefill returned attentions.")
    dense_next = _token_argmax(dense_output.logits)
    selection_prefill_time = _profile_mark(profile_prefill) - stage_start if profile_prefill else 0.0
    if attentions is None:
        raise RuntimeError("Selection prefill did not return attentions.")

    stage_start = _profile_mark(profile_prefill)
    selection = build_sparse_selection(
        attentions,
        prompt_ids,
        video_token_id,
        text_start,
        target_k_plus_text=target_k_plus_text,
        explicit_k=explicit_k,
        num_key_value_heads=model.config.num_key_value_heads,
    )
    del attentions
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
        collector.begin_verification(decode_rounds)
        verify_outputs = model(input_ids=verify_tensor, past_key_values=dense_pkv)
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
        if collector_version == "v2":
            positions = verification_query_positions(accept_len, len(dense_pending), propose_len)
            A_t = collector.compute(positions, dense_pkv)
        else:
            A_t = collector.latest_scores()
        new_state = policy_impl.update(A_t, state, k)
        refresh_time = 0.0
        if policy == "previous_verify_topk" and not torch.equal(state.indices, new_state.indices):
            stage_start = _profile_mark(profile_decode)
            if refresh_mode == "incremental":
                refresh_time = incremental_refresh_sparse_visual_kv(
                    sparse_pkv, dense_pkv, non_visual_positions, state.indices, new_state.indices, k
                )
            else:
                refresh_time = refresh_sparse_visual_kv(
                    sparse_pkv, dense_pkv, non_visual_positions, new_state.indices, k
                )
            refresh_time += _profile_mark(profile_decode) - stage_start
        refresh_records.append(
            {
                "round_id": decode_rounds,
                "jaccard_old_new": float(new_state.selection_overlap),
                "changed_ratio": float(1.0 - new_state.selection_overlap),
                "changed_tokens": float(count_changed_tokens(state.indices, new_state.indices)),
                "refresh_time_ms": float(refresh_time * 1000.0),
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
    }
    return result, selection, dynamic_stats
