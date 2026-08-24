"""Physical sparse-cache visual-KV refresh for Verification-Guided STD.

The sparse cache is compacted by ``compact_sparse_prompt_cache`` into a per-head
*sorted absolute-position* layout: for each head the compact prompt is
``sorted(non_visual_positions ∪ topk[head])`` (visual and non-visual slots are
interleaved, not a contiguous visual prefix). Because the sparse draft attends to
the whole compacted prefix with ``is_causal=False`` and RoPE is baked into the
keys, only the *content* of each slot matters, not its order.

Dynamic routing changes the selected visual slots (S_t -> S_{t+1}). This module
rebuilds only the compact *prompt* segment (``[0:compact_len]``) in place,
gathering ``sorted(non_visual ∪ new_topk[head])`` from the canonical dense cache
(which keeps full visual KV at original positions). The generated-KV segment
(``[compact_len:]``) is never touched, truncated, or rebuilt; position ids are
unchanged.
"""

from __future__ import annotations

import time
from typing import List

import torch


def refresh_sparse_visual_kv(
    sparse_past_key_values,
    dense_past_key_values,
    non_visual_positions: torch.Tensor,
    new_topk: torch.Tensor,
    k: int,
) -> float:
    """Rebuild the sparse cache's compact prompt segment for a new visual top-K.

    Args:
        sparse_past_key_values: the compacted sparse cache (list per layer of
            ``[KVCache_k, KVCache_v]``).
        dense_past_key_values: the canonical dense cache holding full visual KV.
        non_visual_positions: ``[N]`` CPU long tensor of non-visual prompt
            positions (identical for every layer and head).
        new_topk: ``[num_layers, kv_heads, k]`` sorted absolute visual positions.
        k: number of selected visual KV slots (unchanged).

    Returns:
        elapsed wall-clock seconds (CUDA-synchronized) spent refreshing.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    compact_len = int(non_visual_positions.numel()) + k
    non_visual = non_visual_positions

    for layer_idx, layer_cache in enumerate(sparse_past_key_values):
        topk = new_topk[layer_idx]                          # [kv_heads, k]
        dense_layer = dense_past_key_values[layer_idx]
        for cache, dense_cache in zip(layer_cache, dense_layer):
            data = cache.data                               # [batch, kv_heads, max_len, head_dim]
            ddata = dense_cache.data
            bsz, kv_heads, _, head_dim = data.shape
            indices = []
            for head in range(kv_heads):
                idx = torch.cat([non_visual, topk[head]], dim=0)
                idx = torch.unique(idx, sorted=True).to(ddata.device)
                if int(idx.numel()) != compact_len:
                    raise RuntimeError("Refreshed sparse KV produced an unexpected length.")
                indices.append(idx)
            index = torch.stack(indices, dim=0)
            index = index.view(1, kv_heads, compact_len, 1).expand(bsz, kv_heads, compact_len, head_dim)
            compacted = ddata.gather(2, index).contiguous()
            data[:, :, :compact_len, :].copy_(compacted)

    torch.cuda.synchronize()
    return time.perf_counter() - t0


def count_changed_tokens(old_topk: torch.Tensor, new_topk: torch.Tensor) -> float:
    """Mean number of visual tokens replaced per (layer, head) when going
    from ``old_topk`` to ``new_topk`` (both ``[num_layers, kv_heads, k]``)."""
    if old_topk.shape != new_topk.shape:
        raise ValueError(f"top-K shape mismatch: {old_topk.shape} vs {new_topk.shape}.")
    L, H, K = old_topk.shape
    total = 0
    for l in range(L):
        for h in range(H):
            so = set(int(x) for x in old_topk[l, h].tolist())
            sn = set(int(x) for x in new_topk[l, h].tolist())
            total += K - len(so & sn)
    return total / (L * H) if L * H else 0.0


def incremental_refresh_sparse_visual_kv(
    sparse_past_key_values,
    dense_past_key_values,
    non_visual_positions: torch.Tensor,
    old_topk: torch.Tensor,
    new_topk: torch.Tensor,
    k: int,
) -> float:
    """Refresh only the changed visual slots of the sparse cache in place.

    Unlike ``refresh_sparse_visual_kv`` (which rebuilds the whole compact prompt
    segment ``sorted(non_visual ∪ new_topk)``), this overwrites just the slots of
    the removed visual tokens with the KV of the added visual tokens. It is valid
    because the sparse draft attends to the whole compact prefix with
    ``is_causal=False`` and RoPE is baked into the keys, so slot *order* is
    irrelevant: the resulting compact prompt still contains exactly
    ``non_visual ∪ new_topk`` (as a set), merely reordered.

    Args:
        sparse_past_key_values: the compacted sparse cache.
        dense_past_key_values: the canonical dense cache holding full visual KV.
        non_visual_positions: ``[N]`` CPU long tensor (sorted).
        old_topk / new_topk: ``[num_layers, kv_heads, k]`` sorted absolute visual
            positions (CPU long), the current and next selection.
        k: number of selected visual KV slots (unchanged).

    Returns:
        elapsed wall-clock seconds (CUDA-synchronized) spent refreshing.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    non_visual = non_visual_positions

    for layer_idx, layer_cache in enumerate(sparse_past_key_values):
        old_t = old_topk[layer_idx]                        # [kv_heads, k] sorted
        new_t = new_topk[layer_idx]                        # [kv_heads, k] sorted
        dense_layer = dense_past_key_values[layer_idx]

        # Vectorized set difference (both sorted, batched searchsorted): removed
        # = old_t \ new_t, added = new_t \ old_t, per head.
        K = int(old_t.shape[1])
        pos = torch.searchsorted(new_t, old_t)
        in_new = (pos < K) & (new_t.gather(1, pos.clamp(max=K - 1)) == old_t)
        removed_mask = ~in_new
        pos2 = torch.searchsorted(old_t, new_t)
        in_old = (pos2 < K) & (old_t.gather(1, pos2.clamp(max=K - 1)) == new_t)
        added_mask = ~in_old

        m_total = int(removed_mask.sum())
        if m_total == 0:
            continue

        # Flatten changed slots across heads (row-major, so removed[i] pairs
        # with added[i] of the same head).
        idx = torch.nonzero(removed_mask, as_tuple=False)   # [M, 2] -> (head, rank)
        head_idx = idx[:, 0]                                # [M]
        rank = idx[:, 1]                                    # [M]
        removed_val = old_t[removed_mask]                   # [M]
        added_val = new_t[added_mask]                       # [M]
        # Physical slot = (# non_visual < r) + rank.
        phys = torch.searchsorted(non_visual, removed_val) + rank

        for cache, dense_cache in zip(layer_cache, dense_layer):
            data = cache.data                               # [bsz, kv_heads, max_len, hd]
            ddata = dense_cache.data
            h_idx = head_idx.to(ddata.device)               # [M]
            p_idx = phys.to(ddata.device)                   # [M]
            a_idx = added_val.to(ddata.device)              # [M]
            added_kv = ddata[:, h_idx, a_idx, :]            # [bsz, M, head_dim]
            data[:, h_idx, p_idx, :] = added_kv

    torch.cuda.synchronize()
    return time.perf_counter() - t0
