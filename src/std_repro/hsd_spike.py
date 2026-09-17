"""Disposable nested speculative-decoding engine for the HSD feasibility spike.

The implementation intentionally favors observability over optimization.  A
small multimodal draft model proposes ``gamma`` tokens, the target model's
compact (static Top-K) cache scores them one at a time as the sparse stage, and
the canonical target cache performs the final batched dense verification.  The
sparse result is diagnostic only: dense verification remains authoritative.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch

from specvlm.kv_cache.kv_cache import initialize_past_key_values
from std_repro.std_qwen25vl import (
    GenerateResult,
    SparseDraftController,
    _fill_cache_length,
    _first_model_device,
    _split_video_text_inputs,
    _token_argmax,
    build_sparse_selection,
    compact_sparse_prompt_cache,
    copy_prompt_cache,
)


def greedy_prefix(draft_tokens: List[int], predictions: List[int]) -> Tuple[int, int]:
    """Return ``(accepted_prefix_length, correction/bonus)``.

    ``predictions`` contains one next-token prediction for every draft token
    plus the prediction after the final draft token.  Keeping this contract in
    a small pure helper prevents the common all-accepted off-by-one bug.
    """
    if len(predictions) != len(draft_tokens) + 1:
        raise ValueError("predictions must contain one bonus row after the draft")
    accepted = 0
    while accepted < len(draft_tokens) and draft_tokens[accepted] == predictions[accepted]:
        accepted += 1
    return accepted, int(predictions[accepted])


@torch.inference_mode()
def nested_draft_block(
    target_model,
    draft_model,
    draft_pkv,
    draft_lengths: torch.Tensor,
    draft_next: torch.Tensor,
    sparse_pkv,
    sparse_lengths: torch.Tensor,
    sparse_next: torch.Tensor,
    sparse_controller,
    *,
    context_len: int,
    block_len: int,
    inner_gamma: int,
) -> Tuple[List[int], torch.Tensor, torch.Tensor, Dict]:
    """Generate one outer block with a genuine D→Sparse-target cascade.

    This helper is intentionally multimodal-input agnostic so a tiny language
    model can exercise the cache/rollback invariant on CPU.  Both caches must
    already represent the same committed prefix; ``sparse_controller`` is
    enabled only while evaluating the sparse target branch.
    """
    if block_len < 1 or inner_gamma < 1:
        raise ValueError("block_len and inner_gamma must be positive")
    draft_base = int(draft_lengths[0].item())
    sparse_base = int(sparse_lengths[0].item())
    candidates: List[int] = []
    accepts: List[int] = []
    while len(candidates) < block_len:
        inner_len = min(inner_gamma, block_len - len(candidates))
        inner_context = context_len + len(candidates)
        dblock: List[int] = []
        d0 = time.time()
        for offset in range(inner_len):
            dblock.append(int(draft_next.item()))
            draft_next = _step(draft_model, draft_next, draft_base + len(candidates) + offset, draft_pkv)
        sparse_token = sparse_next
        predictions: List[int] = []
        for offset, token_id in enumerate(dblock):
            predictions.append(int(sparse_token.item()))
            sparse_token = _step(
                target_model,
                torch.tensor([[token_id]], dtype=torch.long, device=_first_model_device(target_model)),
                inner_context + offset,
                sparse_pkv,
                controller=sparse_controller,
            )
        predictions.append(int(sparse_token.item()))
        accepted, correction = greedy_prefix(dblock, predictions)
        accepts.append(accepted)
        candidates.extend(dblock[:accepted])
        include_correction = len(candidates) < block_len
        if include_correction:
            candidates.append(correction)
            _fill_cache_length(sparse_lengths, sparse_base + len(candidates) - 1)
            _fill_cache_length(draft_lengths, draft_base + len(candidates) - 1)
            sparse_next = _step(
                target_model,
                torch.tensor([[correction]], dtype=torch.long, device=_first_model_device(target_model)),
                context_len + len(candidates) - 1,
                sparse_pkv,
                controller=sparse_controller,
            )
            draft_next = _step(
                draft_model,
                torch.tensor([[correction]], dtype=torch.long, device=_first_model_device(draft_model)),
                context_len + len(candidates) - 1,
                draft_pkv,
            )
        else:
            _fill_cache_length(sparse_lengths, sparse_base + len(candidates))
            _fill_cache_length(draft_lengths, draft_base + len(candidates))
            sparse_next = sparse_token
    return candidates, draft_next, sparse_next, {
        "inner_rounds": len(accepts),
        "sparse_accept_len": sum(accepts),
        "sparse_accept_lengths": accepts,
    }


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _step(model, token: torch.Tensor, position: int, past, *, controller=None) -> torch.Tensor:
    if controller is not None:
        controller.set_enabled(True)
    try:
        position_ids = torch.tensor([[position]], dtype=torch.long, device=token.device)
        output = model(input_ids=token, past_key_values=past, position_ids=position_ids)
        return _token_argmax(output.logits)
    finally:
        if controller is not None:
            controller.set_enabled(False)


def _model_prefill(model, inputs, video_token_id: int, *, attentions: bool = False):
    past, _, lengths = initialize_past_key_values(model)
    prompt_ids, text_start, text_ids, video_inputs = _split_video_text_inputs(
        inputs, video_token_id, _first_model_device(model)
    )
    model(**video_inputs, past_key_values=past)
    output = model(input_ids=text_ids, past_key_values=past, output_attentions=attentions)
    return prompt_ids, text_start, past, lengths, _token_argmax(output.logits), output.attentions


@torch.inference_mode()
def hsd_generate_qwen25vl(
    target_model,
    draft_model,
    inputs: Dict[str, torch.Tensor],
    video_token_id: int,
    eos_token_id: int,
    max_new_tokens: int = 64,
    gamma: int = 9,
    inner_gamma: int = 3,
    target_k_plus_text: int = 1024,
    mode: str = "hsd",
    ignore_eos: bool = True,
    profile_decode: bool = True,
) -> Tuple[GenerateResult, Dict]:
    """Run ``small_dense`` or nested ``hsd`` and return observable stage stats."""
    if mode not in {"small_dense", "hsd"}:
        raise ValueError("mode must be 'small_dense' or 'hsd'")
    if gamma < 1 or inner_gamma < 1 or inner_gamma > gamma:
        raise ValueError("inner_gamma must be in [1, gamma]")
    _sync()
    start = time.time()
    target_device = _first_model_device(target_model)

    # Canonical target prefill.  HSD additionally builds a compact target cache
    # used by the diagnostic sparse stage; small_dense intentionally skips it.
    if mode == "hsd":
        prompt_ids, text_start, selection_pkv, _, target_next, attentions = _model_prefill(
            target_model, inputs, video_token_id, attentions=True
        )
        dense_pkv, _, dense_lengths = initialize_past_key_values(target_model)
        copy_prompt_cache(selection_pkv, dense_pkv, dense_lengths, text_start)
        dense_output = target_model(
            input_ids=prompt_ids[:, text_start:].to(target_device),
            past_key_values=dense_pkv,
            output_attentions=False,
        )
        dense_next = _token_argmax(dense_output.logits)
        # Seed the sparse branch from the canonical dense verifier.  The
        # attention-returning prefill above may use a different backend and
        # must never become an alternate target for lossless comparison.
        target_next = dense_next
        selection = build_sparse_selection(
            attentions, prompt_ids, video_token_id, text_start,
            target_k_plus_text=target_k_plus_text,
            num_key_value_heads=target_model.config.num_key_value_heads,
        )
        sparse_pkv, _, sparse_lengths = initialize_past_key_values(target_model)
        copy_prompt_cache(dense_pkv, sparse_pkv, sparse_lengths, int(prompt_ids.shape[1]))
        sparse_prompt_len = compact_sparse_prompt_cache(sparse_pkv, sparse_lengths, selection)
        try:
            from std_repro.sparse_verify_spike import BatchedSparseController
        except ImportError:
            BatchedSparseController = SparseDraftController
        controller = BatchedSparseController(target_model, selection, sparse_attn_mode="gqa_sdpa", use_compile=False)
        controller.install()
        del selection_pkv, attentions
        sparse_next = target_next
    else:
        prompt_ids, _, dense_pkv, dense_lengths, dense_next, _ = _model_prefill(
            target_model, inputs, video_token_id, attentions=False
        )
        target_next = dense_next
        sparse_pkv = sparse_lengths = controller = None
        sparse_prompt_len = 0

    # Draft model receives the same unmodified multimodal input and owns its
    # independent full KV cache.
    draft_prompt, _, draft_pkv, draft_lengths, draft_next, _ = _model_prefill(
        draft_model, inputs, video_token_id, attentions=False
    )
    if not torch.equal(prompt_ids.detach().cpu(), draft_prompt.detach().cpu()):
        raise ValueError("target and draft tokenized prompts differ")
    prompt_len = int(prompt_ids.shape[1])
    generated: List[int] = []
    accepted_total = 0
    proposed_total = 0
    rounds = 0
    accepts: List[int] = []
    draft_time = sparse_time = verify_time = bonus_time = 0.0
    sparse_matches = sparse_compared = 0
    sparse_accepts: List[int] = []

    _sync(); decode_start = time.time()
    while len(generated) < max_new_tokens:
        rounds += 1
        outer_remaining = min(gamma, max_new_tokens - len(generated))
        context_len = prompt_len + len(generated)
        dense_base = context_len
        sparse_base = sparse_prompt_len + len(generated)
        draft_base = context_len
        candidates: List[int] = []
        if mode == "hsd":
            # Build the outer candidate block from inner D->S blocks.  A
            # sparse correction is fed back to both caches before the next
            # inner block, so S—not D—defines the outer proposal stream.
            while len(candidates) < outer_remaining:
                inner_len = min(inner_gamma, outer_remaining - len(candidates))
                inner_context = context_len + len(candidates)
                inner_draft_base = draft_base + len(candidates)
                t0 = time.time(); draft_block: List[int] = []
                for offset in range(inner_len):
                    draft_block.append(int(draft_next.item()))
                    draft_next = _step(draft_model, draft_next, inner_draft_base + offset, draft_pkv)
                draft_time += time.time() - t0
                proposed_total += len(draft_block)
                sparse_token = sparse_next
                sparse_predictions: List[int] = []
                t0 = time.time()
                if len(draft_block) > 1:
                    # The experimental controller supplies a prefix-aware
                    # offset-causal mask.  Explicit logical positions preserve
                    # RoPE after the visual KV prefix has been compacted.
                    sparse_input = torch.tensor([draft_block], dtype=torch.long, device=target_device)
                    sparse_positions = torch.arange(
                        inner_context, inner_context + len(draft_block), dtype=torch.long, device=target_device
                    ).view(1, -1)
                    controller.set_enabled(True)
                    try:
                        sparse_output = target_model(
                            input_ids=sparse_input,
                            past_key_values=sparse_pkv,
                            position_ids=sparse_positions,
                        )
                    finally:
                        controller.set_enabled(False)
                    sparse_predictions.append(int(sparse_token.item()))
                    sparse_predictions.extend(torch.argmax(sparse_output.logits[0], dim=-1).tolist())
                    sparse_token = torch.argmax(sparse_output.logits[:, -1, :], dim=-1, keepdim=True)
                else:
                    for offset, token_id in enumerate(draft_block):
                        sparse_predictions.append(int(sparse_token.item()))
                        sparse_token = _step(
                            target_model, torch.tensor([[token_id]], dtype=torch.long, device=target_device),
                            inner_context + offset, sparse_pkv, controller=controller,
                        )
                sparse_time += time.time() - t0
                if len(draft_block) == 1:
                    sparse_predictions.append(int(sparse_token.item()))
                sparse_accept, sparse_bonus = greedy_prefix(draft_block, sparse_predictions)
                sparse_accepts.append(sparse_accept)
                sparse_compared += len(draft_block)
                sparse_matches += sum(a == b for a, b in zip(sparse_predictions, draft_block))
                candidates.extend(draft_block[:sparse_accept])
                correction_included = len(candidates) < outer_remaining
                if len(candidates) < outer_remaining:
                    candidates.append(sparse_bonus)
                    correction_included = True
                if len(candidates) > outer_remaining:
                    # A fully accepted final inner block may produce a bonus
                    # beyond the requested outer budget.  Do not verify or
                    # expose that token in this round.
                    candidates[:] = candidates[:outer_remaining]
                    correction_included = False
                if len(candidates) < outer_remaining:
                    # The sparse branch has cached the accepted D prefix, but
                    # not the S correction/bonus.  Roll back both branches to
                    # that prefix before feeding the authoritative S token.
                    prefix_len = len(candidates) - 1
                    _fill_cache_length(sparse_lengths, sparse_base + prefix_len)
                    _fill_cache_length(draft_lengths, draft_base + prefix_len)
                    sparse_next = _step(
                        target_model, torch.tensor([[sparse_bonus]], dtype=torch.long, device=target_device),
                        context_len + prefix_len, sparse_pkv, controller=controller,
                    )
                    draft_next = _step(
                        draft_model, torch.tensor([[sparse_bonus]], dtype=torch.long,
                                                  device=_first_model_device(draft_model)),
                        context_len + prefix_len, draft_pkv,
                    )
                else:
                    # If the correction was included in this outer block,
                    # materialize it now; otherwise the block is full of
                    # accepted D tokens and the existing cache is sufficient.
                    if correction_included:
                        prefix_len = len(candidates) - 1
                        _fill_cache_length(sparse_lengths, sparse_base + prefix_len)
                        _fill_cache_length(draft_lengths, draft_base + prefix_len)
                        sparse_next = _step(
                            target_model, torch.tensor([[sparse_bonus]], dtype=torch.long, device=target_device),
                            context_len + prefix_len, sparse_pkv, controller=controller,
                        )
                        draft_next = _step(
                            draft_model, torch.tensor([[sparse_bonus]], dtype=torch.long,
                                                      device=_first_model_device(draft_model)),
                            context_len + prefix_len, draft_pkv,
                        )
                    else:
                        _fill_cache_length(sparse_lengths, sparse_base + len(candidates))
                        _fill_cache_length(draft_lengths, draft_base + len(candidates))
                        sparse_next = sparse_token
        else:
            t0 = time.time()
            for offset in range(outer_remaining):
                candidates.append(int(draft_next.item()))
                draft_next = _step(draft_model, draft_next, draft_base + offset, draft_pkv)
            draft_time += time.time() - t0
            proposed_total += len(candidates)

        t0 = time.time()
        _fill_cache_length(dense_lengths, dense_base)
        verify_input = torch.tensor([candidates], dtype=torch.long, device=target_device)
        if controller is not None:
            controller.set_enabled(False)
        dense_output = target_model(input_ids=verify_input, past_key_values=dense_pkv)
        logits = dense_output.logits[0]
        dense_predictions = [int(dense_next.item())]
        # Each candidate row predicts the token following that candidate; the
        # final row is therefore the bonus when every candidate is accepted.
        dense_predictions.extend(torch.argmax(logits, dim=-1).tolist())
        accept, bonus = greedy_prefix(candidates, dense_predictions)
        verify_time += time.time() - t0
        accepts.append(accept); accepted_total += accept

        append = candidates[:accept] + [bonus]
        if not ignore_eos and eos_token_id in append:
            append = append[:append.index(eos_token_id) + 1]
        remaining = max_new_tokens - len(generated)
        append = append[:remaining]
        generated.extend(append)
        if len(generated) >= max_new_tokens or (not ignore_eos and append and append[-1] == eos_token_id):
            break

        # Roll back speculative writes, then commit the authoritative bonus to
        # all three caches so the next round starts from the same context.
        _fill_cache_length(dense_lengths, dense_base + accept)
        if mode == "hsd":
            _fill_cache_length(sparse_lengths, sparse_base + accept)
        _fill_cache_length(draft_lengths, draft_base + accept)
        t0 = time.time()
        bonus_tensor = torch.tensor([[bonus]], dtype=torch.long, device=target_device)
        dense_next = _step(target_model, bonus_tensor, context_len + accept, dense_pkv)
        bonus_time += time.time() - t0
        if mode == "hsd":
            sparse_next = _step(target_model, bonus_tensor, context_len + accept, sparse_pkv, controller=controller)
        draft_next = _step(draft_model, bonus_tensor.to(_first_model_device(draft_model)), context_len + accept, draft_pkv)
        target_next = sparse_next if mode == "hsd" else dense_next

    _sync(); end = time.time()
    output = torch.cat([prompt_ids.to(target_device), torch.tensor([generated], dtype=torch.long, device=target_device)], dim=1)
    result = GenerateResult(
        output, end - decode_start, end - start, len(generated), accepted_total,
        proposed_total, (sum(accepts) / len(accepts)) if accepts else 0.0, rounds,
        gamma, draft_time=draft_time, verify_time=verify_time, bonus_time=bonus_time,
        prefill_time=decode_start - start,
    )
    stats = {
        "mode": mode, "gamma": gamma, "inner_gamma": inner_gamma,
        "sparse_accept_lengths": sparse_accepts,
        "sparse_token_match_rate": sparse_matches / sparse_compared if sparse_compared else 0.0,
        "sparse_accept_mean": sum(sparse_accepts) / len(sparse_accepts) if sparse_accepts else 0.0,
        "sparse_time": sparse_time,
        "dense_verify_time": verify_time,
        "draft_time": draft_time,
        "bonus_time": bonus_time,
        "profile_decode": bool(profile_decode),
    }
    return result, stats
