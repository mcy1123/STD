"""Experimental batched sparse-verifier attention.

This module is deliberately a sidecar for feasibility experiments.  It does
not alter :mod:`std_repro.std_qwen25vl`: the existing q_len=1 sparse draft path
is delegated verbatim, while q_len>1 calls use the already-compacted prefix
and an offset causal mask.  The latter is the primitive needed to prototype a
Sparse verifier over a whole speculative block without future leakage.
"""

from __future__ import annotations

import types
from typing import Optional, Tuple

import torch

from specvlm.models import modeling_qwen2_5_vl as qwen_mod
from std_repro.std_qwen25vl import (
    SparseDraftController,
    _sparse_aware_attention_forward,
)


def offset_causal_mask(q_len: int, kv_len: int, device: torch.device) -> Optional[torch.Tensor]:
    """Return an additive causal mask for ``q_len`` appended queries.

    ``kv_len - q_len`` keys are cached before the query block (the compact
    visual prefix plus generated tail).  Query row ``i`` may therefore read
    through key ``kv_len - q_len + i``.  A single-token query has the existing
    fast path and needs no explicit mask.
    """

    if q_len < 1 or kv_len < q_len:
        raise ValueError(f"expected 1 <= q_len <= kv_len, got q_len={q_len}, kv_len={kv_len}")
    if q_len == 1:
        return None
    past_len = kv_len - q_len
    key_positions = torch.arange(kv_len, device=device)
    query_limits = past_len + torch.arange(q_len, device=device)
    blocked = key_positions.unsqueeze(0) > query_limits.unsqueeze(1)
    mask = torch.zeros((1, 1, q_len, kv_len), device=device, dtype=torch.float32)
    return mask.masked_fill(blocked.view(1, 1, q_len, kv_len), float("-inf"))


def _offset_causal_gqa_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    """Run GQA SDPA with an offset causal mask for an appended query block."""

    q_len = int(query_states.shape[-2])
    kv_len = int(key_states.shape[-2])
    if kv_len < q_len:
        raise ValueError("key/value sequence cannot be shorter than query sequence")
    mask = offset_causal_mask(q_len, kv_len, query_states.device)
    if mask is not None:
        mask = mask.to(dtype=query_states.dtype)
    return torch.nn.functional.scaled_dot_product_attention(
        query_states.contiguous(),
        key_states.contiguous(),
        value_states.contiguous(),
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=True,
    )


def _batched_sparse_aware_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    """Attention wrapper used by :class:`BatchedSparseController`.

    The q_len=1 branch is the original STD implementation, byte-for-byte in
    behavior.  For a block, the sparse cache is already compacted, so gathering
    by original visual positions would be incorrect; we instead attend to its
    compact prefix and generated tail in-place with the offset mask.
    """

    controller = getattr(self, "_std_sparse_controller", None)
    original = getattr(self, "_std_original_forward", None)
    if controller is None or not controller.enabled:
        if original is None:
            raise RuntimeError("BatchedSparseController requires an installed original forward")
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

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        raise ValueError("position_embeddings are required for Qwen2.5-VL attention")
    cos, sin = position_embeddings
    query_states, key_states = qwen_mod.apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )
    if past_key_value is not None:
        key_states = past_key_value[0].cat(key_states, dim=2)
        value_states = past_key_value[1].cat(value_states, dim=2)

    attn_output = _offset_causal_gqa_attention(query_states, key_states, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
    return self.o_proj(attn_output), None, None


class BatchedSparseController(SparseDraftController):
    """Experimental controller supporting q_len>1 sparse verification calls."""

    def install(self) -> None:
        if self._installed:
            return
        for layer in self.model.model.layers:
            attn = layer.self_attn
            if getattr(attn, "_std_original_forward", None) is None:
                attn._std_original_forward = attn.forward
            attn.forward = types.MethodType(_batched_sparse_aware_attention_forward, attn)
            attn._std_sparse_controller = self
        self._installed = True

