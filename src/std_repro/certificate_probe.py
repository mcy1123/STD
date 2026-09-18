"""CSV-0 probe: is a certified sparse verification layer feasible?

The three-level funnel was falsified because it inserts a full model forward that
merely re-orders candidates for the dense verifier (see
``docs/superpowers/plans/2026-09-19-hsd-three-level-verification.md``).  The only
version with a real mechanism is *certified* sparse verification: the middle level
runs once, and for rounds where it can certify that its own prediction equals the
dense one, the dense pass is skipped entirely.

This module answers the two questions that decide whether that is worth building:

  A. **Correctness ceiling.**  Sparse (static top-K) and dense attention are run
     over the *same* draft block, so their next-token argmax can be compared
     position by position.  The fraction of rounds where every position agrees is
     the ceiling on the certificate rate -- no rule computable from the sparse
     pass alone can exceed it without emitting wrong tokens.
  B. **Whether a cheap statistic can find the agreeing rounds.**  Top-1 logit
     margin and predictive entropy are available for free from the sparse pass.
     The probe reports coverage/precision curves over those thresholds, because a
     certificate is only useful if it fires on rounds that really do agree.

Both are cheap: one sparse batched pass plus one dense pass per round, no draft
model, no engine.  Only a round whose certificate fires may skip the dense pass,
so the metric that feeds ``scripts/analysis/hsd_feasibility.py`` is the
round-level certificate rate at a reportable precision.
"""

from __future__ import annotations

import time
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# Certificate statistics computable from the sparse pass alone.  "min" over the
# block is the round-level statistic: a round is certified only if every position
# in it clears the threshold.
CERTIFICATE_STATISTICS = ("margin", "entropy")

# The model stack cannot be imported on every interpreter this repository is
# developed on (see PROGRESS §16.3), and the certificate math needs only torch.
# These thin wrappers resolve the heavy symbols on first use so the pure helpers
# stay importable and unit-testable in isolation.  A module-level ``__getattr__``
# would not help here: PEP 562 only covers attribute access on the module object,
# not global lookups from inside the module's own functions.
def _std(name: str):
    import std_repro.std_qwen25vl as std

    return getattr(std, name)


def _draft_tokens(*args, **kwargs):
    return _std("_draft_tokens")(*args, **kwargs)


def _fill_cache_length(*args, **kwargs):
    return _std("_fill_cache_length")(*args, **kwargs)


def _first_model_device(*args, **kwargs):
    return _std("_first_model_device")(*args, **kwargs)


def _sparse_aware_attention_forward(*args, **kwargs):
    return _std("_sparse_aware_attention_forward")(*args, **kwargs)


def _token_argmax(*args, **kwargs):
    return _std("_token_argmax")(*args, **kwargs)


def offset_causal_mask(*args, **kwargs):
    from std_repro.sparse_verify_spike import offset_causal_mask as impl

    return impl(*args, **kwargs)


def position_records(
    sparse_logits: torch.Tensor,
    dense_logits: torch.Tensor,
) -> List[Dict[str, float]]:
    """Per-position agreement and free certificate statistics.

    ``sparse_logits[i]`` and ``dense_logits[i]`` are the next-token distributions
    after the same prefix plus ``draft[:i+1]``, so comparing their argmax at
    position ``i`` is exactly the question a sparse verifier asks.
    """
    if sparse_logits.shape != dense_logits.shape:
        raise ValueError("sparse and dense logits must have the same shape")
    if sparse_logits.ndim != 2:
        raise ValueError("expected [block, vocab] logits")
    sparse = sparse_logits.float()
    dense = dense_logits.float()
    s_vals, s_idx = torch.topk(sparse, k=2, dim=-1)
    d_vals, d_idx = torch.topk(dense, k=2, dim=-1)
    log_probs = torch.log_softmax(sparse, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    agree = (s_idx[:, 0] == d_idx[:, 0]).tolist()
    return [
        {
            "position": index,
            "agree": bool(agree[index]),
            "sparse_top1": int(s_idx[index, 0].item()),
            "dense_top1": int(d_idx[index, 0].item()),
            "margin": float((s_vals[index, 0] - s_vals[index, 1]).item()),
            "dense_margin": float((d_vals[index, 0] - d_vals[index, 1]).item()),
            "entropy": float(entropy[index].item()),
        }
        for index in range(sparse.shape[0])
    ]


def round_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collapse a round's positions into the values the certificate decision uses."""
    if not records:
        raise ValueError("cannot summarize an empty round")
    return {
        "positions": len(records),
        "all_agree": all(bool(record["agree"]) for record in records),
        "agreements": sum(1 for record in records if record["agree"]),
        "min_margin": min(float(record["margin"]) for record in records),
        "mean_margin": sum(float(record["margin"]) for record in records) / len(records),
        "max_entropy": max(float(record["entropy"]) for record in records),
        "mean_entropy": sum(float(record["entropy"]) for record in records) / len(records),
    }


def default_thresholds(rounds: Sequence[Dict[str, Any]], count: int = 24) -> List[float]:
    """Threshold grid spanning the observed margin range (quantile spaced)."""
    values = sorted(float(round["min_margin"]) for round in rounds)
    if not values:
        return []
    if values[0] == values[-1]:
        return [values[0]]
    thresholds = []
    for index in range(count):
        position = index * (len(values) - 1) / (count - 1)
        low = int(position)
        high = min(low + 1, len(values) - 1)
        weight = position - low
        thresholds.append(values[low] * (1 - weight) + values[high] * weight)
    return sorted(set(thresholds))


def certificate_curve(
    rounds: Sequence[Dict[str, Any]],
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    """Coverage/precision of "certify a round when min sparse margin >= tau".

    ``precision`` is the fraction of certified rounds whose every position really
    did agree with dense; ``coverage`` is the fraction of all rounds certified.
    A certificate that skips dense verification must keep precision at 1.0 to
    stay lossless, which is why the ceiling column is reported alongside.
    """
    if not rounds:
        raise ValueError("cannot build a certificate curve without rounds")
    total = len(rounds)
    ceiling = sum(1 for round in rounds if round["all_agree"]) / total
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        certified = [round for round in rounds if float(round["min_margin"]) >= float(threshold)]
        fired = len(certified)
        correct = sum(1 for round in certified if round["all_agree"])
        rows.append(
            {
                "threshold": float(threshold),
                "certified_rounds": fired,
                "coverage": fired / total,
                "precision": (correct / fired) if fired else 0.0,
                "wrong_skips": fired - correct,
                "ceiling": ceiling,
            }
        )
    return rows


def select_operating_point(
    curve: Sequence[Dict[str, Any]],
    min_precision: float = 1.0,
) -> Optional[Dict[str, Any]]:
    """Highest-coverage point whose precision still clears ``min_precision``."""
    eligible = [row for row in curve if row["precision"] >= min_precision]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (row["coverage"], row["threshold"]))


def _cached_mask_sparse_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    """Attention wrapper that reuses one mask across all layers of a forward.

    Identical to ``sparse_verify_spike._batched_sparse_aware_attention_forward``
    except that the additive causal mask is taken from the controller instead of
    being rebuilt for every one of the model's attention layers.  The mask depends
    only on ``(q_len, kv_len, dtype)``, all of which are constant within a single
    forward, so rebuilding it per layer is pure overhead.
    """
    controller = getattr(self, "_std_sparse_controller", None)
    original = getattr(self, "_std_original_forward", None)
    if controller is None or not controller.enabled:
        if original is None:
            raise RuntimeError("ProbeSparseController requires an installed original forward")
        return original(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
    if hidden_states.shape[1] == 1:
        return _sparse_aware_attention_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
    return controller.batched_attention(
        self, hidden_states, past_key_value, position_embeddings, cached_mask=controller.cached_mask
    )


def _build_mask(q_len: int, kv_len: int, dtype: torch.dtype, device: torch.device):
    """Additive offset-causal mask in the query dtype (shared with the spike)."""
    mask = offset_causal_mask(q_len, kv_len, device)
    return mask.to(dtype=dtype) if mask is not None else None


class ProbeSparseController:
    """Sparse controller with a switchable, instrumented batched attention path.

    ``cached_mask=False`` reproduces the existing per-layer mask construction so
    the probe can time both variants in the same process and assert that they
    produce identical logits.  This mirrors ``BatchedSparseController``'s
    interface but deliberately does not inherit from it, so importing this module
    does not pull in the model stack.
    """

    def __init__(self, model, selection, sparse_attn_mode: str = "gqa_sdpa", use_compile: bool = False):
        if sparse_attn_mode not in {"repeat_sdpa", "gqa_sdpa", "triton_gqa"}:
            raise ValueError(f"Unsupported sparse_attn_mode={sparse_attn_mode!r}.")
        self.model = model
        self.selection = selection
        self.sparse_attn_mode = sparse_attn_mode
        self.enabled = False
        self._installed = False
        self.compiled_forward = None
        self.qwen_mod = None
        self.cached_mask = False
        self._prepared_mask: Optional[torch.Tensor] = None
        self._prepared_key: Optional[Tuple[int, int]] = None
        self.mask_hits = 0
        self.mask_misses = 0
        self.mask_build_seconds = 0.0

    def install(self) -> None:
        if self._installed:
            return
        from specvlm.models import modeling_qwen2_5_vl as qwen_mod

        self.qwen_mod = qwen_mod
        for layer in self.model.model.layers:
            attn = layer.self_attn
            if getattr(attn, "_std_original_forward", None) is None:
                attn._std_original_forward = attn.forward
            attn.forward = types.MethodType(_cached_mask_sparse_attention_forward, attn)
            attn._std_sparse_controller = self
        self._installed = True

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def reset_mask_stats(self) -> None:
        self.mask_hits = 0
        self.mask_misses = 0
        self.mask_build_seconds = 0.0

    def prepare_mask(self, q_len: int, kv_len: int, dtype: torch.dtype, device: torch.device) -> None:
        """Build the additive causal mask once for the next forward."""
        start = time.perf_counter()
        self._prepared_mask = _build_mask(q_len, kv_len, dtype, device)
        self._prepared_key = (q_len, kv_len)
        self.mask_build_seconds += time.perf_counter() - start

    def take_mask(self, q_len: int, kv_len: int, dtype: torch.dtype, device: torch.device):
        if self._prepared_key == (q_len, kv_len):
            self.mask_hits += 1
            return self._prepared_mask
        self.mask_misses += 1
        start = time.perf_counter()
        mask = _build_mask(q_len, kv_len, dtype, device)
        self.mask_build_seconds += time.perf_counter() - start
        return mask

    def batched_attention(self, attn, hidden_states, past_key_value, position_embeddings, *, cached_mask: bool):
        """Shared q/k/v + RoPE + SDPA body for the q_len>1 sparse path."""
        qwen_mod = self.qwen_mod
        bsz, q_len, _ = hidden_states.size()
        query_states = attn.q_proj(hidden_states)
        key_states = attn.k_proj(hidden_states)
        value_states = attn.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
        if position_embeddings is None:
            raise ValueError("position_embeddings are required for Qwen2.5-VL attention")
        cos, sin = position_embeddings
        query_states, key_states = qwen_mod.apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, attn.rope_scaling["mrope_section"]
        )
        if past_key_value is not None:
            key_states = past_key_value[0].cat(key_states, dim=2)
            value_states = past_key_value[1].cat(value_states, dim=2)
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        kv_len = int(key_states.shape[-2])
        if q_len == 1:
            mask = None
        elif cached_mask:
            mask = self.take_mask(q_len, kv_len, query_states.dtype, query_states.device)
        else:
            start = time.perf_counter()
            mask = _build_mask(q_len, kv_len, query_states.dtype, query_states.device)
            self.mask_build_seconds += time.perf_counter() - start
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, attn.hidden_size)
        return attn.o_proj(attn_output), None, None


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _model_prefill(model, inputs, video_token_id: int, *, attentions: bool = False):
    from specvlm.kv_cache.kv_cache import initialize_past_key_values
    from std_repro.std_qwen25vl import _split_video_text_inputs

    past, _, lengths = initialize_past_key_values(model)
    prompt_ids, text_start, text_ids, video_inputs = _split_video_text_inputs(
        inputs, video_token_id, _first_model_device(model)
    )
    model(**video_inputs, past_key_values=past)
    output = model(input_ids=text_ids, past_key_values=past, output_attentions=attentions)
    return prompt_ids, text_start, past, lengths, _token_argmax(output.logits), output.attentions


@torch.inference_mode()
def certificate_probe_qwen25vl(
    model,
    inputs: Dict[str, torch.Tensor],
    video_token_id: int,
    eos_token_id: int,
    max_new_tokens: int = 64,
    gamma: int = 9,
    target_k_plus_text: int = 1024,
    sparse_attn_mode: str = "gqa_sdpa",
    ignore_eos: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Run one paired sparse/dense decode and return per-round certificate data.

    The decode itself mirrors the parallel-verification branch of
    ``std_generate_qwen25vl`` so the emitted sequence must equal greedy AR exactly;
    the driver asserts that.  The extra work is the instrumented sparse batched
    pass, which is run once per round in both mask variants over the same draft
    block, and the dense pass whose logits define ground truth.
    """
    from specvlm.kv_cache.kv_cache import initialize_past_key_values
    from std_repro.std_qwen25vl import (
        GenerateResult,
        build_sparse_selection,
        compact_sparse_prompt_cache,
        copy_prompt_cache,
    )

    if gamma < 2:
        raise ValueError("the certificate probe needs gamma >= 2 to form a block")
    _sync()
    start = time.time()
    target_device = _first_model_device(model)

    prompt_ids, text_start, selection_pkv, _, _, attentions = _model_prefill(
        model, inputs, video_token_id, attentions=True
    )
    dense_pkv, _, dense_lengths = initialize_past_key_values(model)
    copy_prompt_cache(selection_pkv, dense_pkv, dense_lengths, text_start)
    dense_output = model(
        input_ids=prompt_ids[:, text_start:].to(target_device),
        past_key_values=dense_pkv,
        output_attentions=False,
    )
    dense_next = _token_argmax(dense_output.logits)

    selection = build_sparse_selection(
        attentions, prompt_ids, video_token_id, text_start,
        target_k_plus_text=target_k_plus_text,
        num_key_value_heads=model.config.num_key_value_heads,
    )
    sparse_pkv, _, sparse_lengths = initialize_past_key_values(model)
    copy_prompt_cache(dense_pkv, sparse_pkv, sparse_lengths, int(prompt_ids.shape[1]))
    sparse_prompt_len = compact_sparse_prompt_cache(sparse_pkv, sparse_lengths, selection)
    controller = ProbeSparseController(model, selection, sparse_attn_mode=sparse_attn_mode, use_compile=False)
    controller.install()
    sparse_next = dense_next
    del selection_pkv, attentions

    prompt_len = int(prompt_ids.shape[1])
    generated: List[int] = []
    rounds: List[Dict[str, Any]] = []
    draft_seconds = dense_seconds = sparse_seconds = cached_seconds = bonus_seconds = dense_bonus_seconds = 0.0
    logit_mismatches = 0
    logit_delta = 0.0
    per_layer_mask_builds = 0.0
    mask_prepare_seconds = 0.0
    cached_mask_hits = cached_mask_misses = 0
    _sync()
    decode_start = time.time()

    while len(generated) < max_new_tokens:
        block_len = min(gamma, max_new_tokens - len(generated))
        if block_len < 2:
            break
        context_len = prompt_len + len(generated)
        dense_base = context_len
        sparse_base = sparse_prompt_len + len(generated)

        # 1) Autoregressive sparse draft, then roll the sparse cache back so the
        #    same block can be replayed in one batched pass.
        t0 = time.perf_counter()
        _fill_cache_length(sparse_lengths, sparse_base)
        draft = _draft_tokens(model, sparse_next, block_len, sparse_pkv, controller, context_len)
        _fill_cache_length(sparse_lengths, sparse_base)
        _sync()
        draft_seconds += time.perf_counter() - t0

        draft_tensor = torch.tensor([draft], dtype=torch.long, device=target_device)
        positions = torch.arange(context_len, context_len + block_len, dtype=torch.long, device=target_device).view(1, -1)

        # 2) Sparse batched verification -- the existing behaviour, where every
        #    attention layer rebuilds the causal mask for the same (q_len, kv_len).
        controller.cached_mask = False
        controller.reset_mask_stats()
        controller.set_enabled(True)
        _sync()
        t0 = time.perf_counter()
        try:
            sparse_out = model(input_ids=draft_tensor, past_key_values=sparse_pkv, position_ids=positions)
        finally:
            controller.set_enabled(False)
            _fill_cache_length(sparse_lengths, sparse_base)
        _sync()
        sparse_seconds += time.perf_counter() - t0
        per_layer_mask_builds += controller.mask_build_seconds
        sparse_logits = sparse_out.logits[0].detach().clone()

        # 3) Same block with one mask built per forward and reused by every layer.
        #    After the forward appends the block, attention sees
        #    ``sparse_base + block_len`` keys, and the mask must be built for that
        #    width in the query dtype (logits are frequently fp32 even for an fp16
        #    model, so they are not a valid dtype source for the mask).  The mask
        #    build is timed separately so the report can give both the forward-only
        #    figure and the all-inclusive cost of the variant.
        query_dtype = model.model.embed_tokens.weight.dtype
        controller.cached_mask = True
        controller.reset_mask_stats()
        controller.set_enabled(True)
        _sync()
        t0 = time.perf_counter()
        controller.prepare_mask(block_len, sparse_base + block_len, query_dtype, target_device)
        _sync()
        mask_prepare_seconds += time.perf_counter() - t0
        _sync()
        t0 = time.perf_counter()
        try:
            cached_out = model(input_ids=draft_tensor, past_key_values=sparse_pkv, position_ids=positions)
        finally:
            controller.set_enabled(False)
            _fill_cache_length(sparse_lengths, sparse_base)
            controller.cached_mask = False
        _sync()
        cached_seconds += time.perf_counter() - t0
        cached_mask_hits += controller.mask_hits
        cached_mask_misses += controller.mask_misses
        cached_logits = cached_out.logits[0].detach()
        if not torch.equal(sparse_logits, cached_logits):
            logit_mismatches += 1
        logit_delta = max(logit_delta, float((sparse_logits - cached_logits).abs().max().item()))
        del sparse_out, cached_out

        # 4) Dense verification -- authoritative, and the ground truth for the
        #    certificate comparison.
        _fill_cache_length(dense_lengths, dense_base)
        _sync()
        t0 = time.perf_counter()
        dense_out = model(input_ids=draft_tensor, past_key_values=dense_pkv)
        _sync()
        dense_seconds += time.perf_counter() - t0
        dense_logits = dense_out.logits[0].detach()

        records = position_records(sparse_logits, dense_logits)
        summary = round_summary(records)
        summary["round"] = len(rounds)
        summary["block_len"] = block_len
        summary["positions_detail"] = records
        rounds.append(summary)

        # 5) Commit exactly like std_generate_qwen25vl's parallel branch.
        dense_predictions = [int(dense_next.item())]
        dense_predictions.extend(torch.argmax(dense_logits, dim=-1).tolist())
        accept = 0
        while accept < block_len and draft[accept] == dense_predictions[accept]:
            accept += 1
        bonus = int(dense_predictions[accept])
        append = draft[:accept] + [bonus]
        if not ignore_eos and eos_token_id in append:
            append = append[: append.index(eos_token_id) + 1]
        append = append[: max_new_tokens - len(generated)]
        generated.extend(append)
        if len(generated) >= max_new_tokens or (not ignore_eos and append and append[-1] == eos_token_id):
            break

        _fill_cache_length(dense_lengths, dense_base + accept)
        _fill_cache_length(sparse_lengths, sparse_base + accept)
        bonus_tensor = torch.tensor([[bonus]], dtype=torch.long, device=target_device)
        bonus_position = torch.tensor([[context_len + accept]], dtype=torch.long, device=target_device)
        _sync()
        t0 = time.perf_counter()
        dense_next = _token_argmax(model(
            input_ids=bonus_tensor, past_key_values=dense_pkv, position_ids=bonus_position,
        ).logits)
        _sync()
        dense_bonus_seconds += time.perf_counter() - t0
        sparse_next = _token_argmax(model(
            input_ids=bonus_tensor, past_key_values=sparse_pkv, position_ids=bonus_position,
        ).logits)
        _sync()
        bonus_seconds += time.perf_counter() - t0
        del dense_out, sparse_logits, dense_logits, cached_logits

    # Tail: when fewer than two tokens remain there is no block to verify, so the
    # remaining budget is filled with plain single-token dense steps.  Without
    # this the probe would stop up to one token short of the AR reference and look
    # like a correctness failure.
    while len(generated) < max_new_tokens:
        token = int(dense_next.item())
        generated.append(token)
        if not ignore_eos and token == eos_token_id:
            break
        if len(generated) >= max_new_tokens:
            break
        position = prompt_len + len(generated) - 1
        _fill_cache_length(dense_lengths, position)
        dense_next = _token_argmax(model(
            input_ids=torch.tensor([[token]], dtype=torch.long, device=target_device),
            past_key_values=dense_pkv,
            position_ids=torch.tensor([[position]], dtype=torch.long, device=target_device),
        ).logits)

    _sync()
    end = time.time()
    output = torch.cat(
        [prompt_ids.to(target_device), torch.tensor([generated], dtype=torch.long, device=target_device)], dim=1
    )
    result = GenerateResult(
        output, end - decode_start, end - start, len(generated),
        sum(round["agreements"] for round in rounds),
        sum(round["positions"] for round in rounds),
        (sum(round["agreements"] for round in rounds) / len(rounds)) if rounds else 0.0,
        len(rounds), gamma,
        draft_time=draft_seconds, verify_time=dense_seconds, prefill_time=decode_start - start,
    )
    stats = {
        "gamma": gamma,
        "rounds": rounds,
        "draft_seconds": draft_seconds,
        "sparse_pass_seconds": sparse_seconds,
        "cached_sparse_pass_seconds": cached_seconds,
        "dense_pass_seconds": dense_seconds,
        "bonus_seconds": bonus_seconds,
        "dense_bonus_seconds": dense_bonus_seconds,
        "mask_prepare_seconds": mask_prepare_seconds,
        "per_layer_mask_build_seconds": per_layer_mask_builds,
        "per_layer_mask_builds": len(rounds) * _num_layers(model),
        "cached_mask_hits": cached_mask_hits,
        "cached_mask_misses": cached_mask_misses,
        "mask_build_microbench_seconds": measure_mask_build(
            gamma, sparse_prompt_len + gamma, model.model.embed_tokens.weight.dtype, target_device
        ),
        "logit_mismatch_rounds": logit_mismatches,
        "max_logit_delta": logit_delta,
        "sparse_prompt_len": sparse_prompt_len,
        "prompt_len": prompt_len,
    }
    return result, stats


def _num_layers(model) -> int:
    return len(model.model.layers)


def measure_mask_build(
    q_len: int,
    kv_len: int,
    dtype: torch.dtype,
    device: torch.device,
    repeats: int = 20,
) -> float:
    """Wall time to construct one offset-causal mask (no sync; launch cost only).

    Diagnostic only: the authoritative comparison between the two batched-pass
    variants is the synchronised end-to-end forward time.
    """
    start = time.perf_counter()
    for _ in range(repeats):
        _build_mask(q_len, kv_len, dtype, device)
    return (time.perf_counter() - start) / repeats


def aggregate(stats_by_sample: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool per-round certificate data across samples and build the curves."""
    rounds: List[Dict[str, Any]] = []
    for stats in stats_by_sample:
        for round_row in stats["rounds"]:
            rounds.append({**round_row, "sample_id": stats.get("sample_id")})
    if not rounds:
        raise ValueError("no rounds to aggregate")
    total_positions = sum(round_row["positions"] for round_row in rounds)
    total_agreements = sum(round_row["agreements"] for round_row in rounds)
    thresholds = default_thresholds(rounds)
    curve = certificate_curve(rounds, thresholds)
    return {
        "rounds": len(rounds),
        "positions": total_positions,
        "position_agreement": total_agreements / total_positions,
        "round_ceiling": sum(1 for round_row in rounds if round_row["all_agree"]) / len(rounds),
        "curve": curve,
        "strict_operating_point": select_operating_point(curve, min_precision=1.0),
        "risk_operating_point": select_operating_point(curve, min_precision=0.95),
        "timing": {
            key: sum(float(stats.get(key, 0.0)) for stats in stats_by_sample)
            for key in (
                "draft_seconds",
                "sparse_pass_seconds",
                "cached_sparse_pass_seconds",
                "dense_pass_seconds",
                "bonus_seconds",
                "dense_bonus_seconds",
                "mask_prepare_seconds",
                "per_layer_mask_build_seconds",
                "mask_build_microbench_seconds",
            )
        },
        "cached_mask_hits": sum(int(stats.get("cached_mask_hits", 0)) for stats in stats_by_sample),
        "cached_mask_misses": sum(int(stats.get("cached_mask_misses", 0)) for stats in stats_by_sample),
        "per_layer_mask_builds": sum(int(stats.get("per_layer_mask_builds", 0)) for stats in stats_by_sample),
        "logit_mismatch_rounds": sum(int(stats.get("logit_mismatch_rounds", 0)) for stats in stats_by_sample),
        "max_logit_delta": max(
            (float(stats.get("max_logit_delta", 0.0)) for stats in stats_by_sample), default=0.0
        ),
    }
