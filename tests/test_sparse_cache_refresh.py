"""CPU regressions for physical sparse-prompt KV refresh.

These exercise the real gather/copy logic.  Only CUDA synchronization is
disabled: the refresh routines otherwise work on ordinary CPU tensors.
"""

from dataclasses import dataclass

import pytest
import torch

from std_repro.sparse_cache_refresh import (
    incremental_refresh_sparse_visual_kv,
    refresh_sparse_visual_kv,
    verify_sparse_visual_consistency,
)


@dataclass
class _Cache:
    data: torch.Tensor
    current_length: torch.Tensor


@pytest.fixture(autouse=True)
def _cpu_synchronization(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


def _caches(initial_topk, non_visual):
    """Unique KV vectors make wrong-head, wrong-layer and wrong-slot copies visible."""
    layers, heads, k = initial_topk.shape
    batch, capacity, dim = 2, 14, 3
    compact_len = len(non_visual) + k
    dense, sparse = [], []
    for layer in range(layers):
        dense_layer, sparse_layer = [], []
        for kv in range(2):
            data = torch.arange(batch * heads * capacity * dim).reshape(batch, heads, capacity, dim)
            data = data + layer * 100_000 + kv * 10_000
            sparse_data = -data.clone() - 1
            for head in range(heads):
                positions = sorted(non_visual.tolist() + initial_topk[layer, head].tolist())
                sparse_data[:, head, :compact_len] = data[:, head, positions]
            dense_layer.append(_Cache(data, torch.tensor(capacity)))
            sparse_layer.append(_Cache(sparse_data, torch.tensor(compact_len + 3)))
        dense.append(dense_layer)
        sparse.append(sparse_layer)
    return dense, sparse


def _clone_caches(caches):
    return [[_Cache(cache.data.clone(), cache.current_length.clone()) for cache in layer] for layer in caches]


def _tail_snapshot(caches, compact_len):
    return [[cache.data[:, :, compact_len:].clone() for cache in layer] for layer in caches]


def _assert_tail_unchanged(caches, snapshot, compact_len):
    for layer, old_layer in zip(caches, snapshot):
        for cache, old_tail in zip(layer, old_layer):
            assert torch.equal(cache.data[:, :, compact_len:], old_tail)
            assert int(cache.current_length) == compact_len + 3


def _assert_prompt_sets(caches, dense, non_visual, topk):
    """Compare content, not order: incremental refresh may reorder visual slots."""
    compact_len = len(non_visual) + topk.shape[-1]
    for layer_idx, layer in enumerate(caches):
        for kv, cache in enumerate(layer):
            for batch in range(cache.data.shape[0]):
                for head in range(cache.data.shape[1]):
                    positions = sorted(non_visual.tolist() + topk[layer_idx, head].tolist())
                    expected = dense[layer_idx][kv].data[batch, head, positions]
                    actual = cache.data[batch, head, :compact_len]
                    assert sorted(map(tuple, actual.tolist())) == sorted(map(tuple, expected.tolist()))


def _fixed_slot_snapshot(caches, topk, non_visual):
    fixed = []
    for layer_idx, layer in enumerate(caches):
        layer_fixed = []
        for cache in layer:
            per_head = []
            for head in range(cache.data.shape[1]):
                positions = sorted(non_visual.tolist() + topk[layer_idx, head].tolist())
                slots = [positions.index(position) for position in non_visual.tolist()]
                per_head.append((slots, cache.data[:, head, slots].clone()))
            layer_fixed.append(per_head)
        fixed.append(layer_fixed)
    return fixed


def _assert_fixed_slots_unchanged(caches, snapshot):
    for layer, old_layer in zip(caches, snapshot):
        for cache, per_head in zip(layer, old_layer):
            for head, (slots, expected) in enumerate(per_head):
                assert torch.equal(cache.data[:, head, slots], expected)


def _selections():
    non_visual = torch.tensor([0, 4, 8])
    initial = torch.tensor([[[1, 3, 7], [2, 5, 7]], [[1, 5, 6], [2, 3, 6]]])
    # Head 0/layer 0 deliberately moves token 6 into token 1's physical slot.
    first = torch.tensor([[[3, 6, 7], [1, 5, 6]], [[2, 5, 7], [1, 3, 7]]])
    # Removing token 6 now requires its actual slot, not its sorted rank.
    second = torch.tensor([[[2, 3, 7], [2, 3, 6]], [[1, 3, 7], [2, 5, 6]]])
    return non_visual, initial, first, second


def test_full_refresh_replaces_prompt_content_and_preserves_generated_tail():
    non_visual, initial, first, _ = _selections()
    dense, sparse = _caches(initial, non_visual)
    tail = _tail_snapshot(sparse, len(non_visual) + initial.shape[-1])

    refresh_sparse_visual_kv(sparse, dense, non_visual, first, first.shape[-1])

    _assert_prompt_sets(sparse, dense, non_visual, first)
    _assert_tail_unchanged(sparse, tail, len(non_visual) + first.shape[-1])


def test_incremental_refresh_matches_full_prompt_set_and_preserves_fixed_data():
    non_visual, initial, first, _ = _selections()
    dense, sparse = _caches(initial, non_visual)
    full = _clone_caches(sparse)
    compact_len = len(non_visual) + initial.shape[-1]
    tail = _tail_snapshot(sparse, compact_len)
    fixed = _fixed_slot_snapshot(sparse, initial, non_visual)

    refresh_sparse_visual_kv(full, dense, non_visual, first, first.shape[-1])
    incremental_refresh_sparse_visual_kv(sparse, dense, non_visual, initial, first, first.shape[-1])

    _assert_prompt_sets(full, dense, non_visual, first)
    _assert_prompt_sets(sparse, dense, non_visual, first)
    _assert_fixed_slots_unchanged(sparse, fixed)
    _assert_tail_unchanged(sparse, tail, compact_len)


def test_incremental_refresh_tracks_physical_slots_across_multiple_rounds():
    non_visual, initial, first, second = _selections()
    dense, sparse = _caches(initial, non_visual)
    full = _clone_caches(sparse)
    compact_len = len(non_visual) + initial.shape[-1]
    tail = _tail_snapshot(sparse, compact_len)
    fixed = _fixed_slot_snapshot(sparse, initial, non_visual)
    previous = initial

    for following in (first, second, initial, first):
        refresh_sparse_visual_kv(full, dense, non_visual, following, following.shape[-1])
        incremental_refresh_sparse_visual_kv(
            sparse, dense, non_visual, previous, following, following.shape[-1]
        )
        _assert_prompt_sets(full, dense, non_visual, following)
        _assert_prompt_sets(sparse, dense, non_visual, following)
        _assert_fixed_slots_unchanged(sparse, fixed)
        _assert_tail_unchanged(sparse, tail, compact_len)
        previous = following


def test_full_refresh_resets_layout_before_a_later_incremental_refresh():
    non_visual, initial, first, second = _selections()
    dense, sparse = _caches(initial, non_visual)
    compact_len = len(non_visual) + initial.shape[-1]
    tail = _tail_snapshot(sparse, compact_len)

    incremental_refresh_sparse_visual_kv(sparse, dense, non_visual, initial, first, first.shape[-1])
    refresh_sparse_visual_kv(sparse, dense, non_visual, second, second.shape[-1])
    fixed = _fixed_slot_snapshot(sparse, second, non_visual)
    incremental_refresh_sparse_visual_kv(sparse, dense, non_visual, second, initial, initial.shape[-1])

    _assert_prompt_sets(sparse, dense, non_visual, initial)
    _assert_fixed_slots_unchanged(sparse, fixed)
    _assert_tail_unchanged(sparse, tail, compact_len)


# ---- T1: selection/cache consistency invariant ----------------------------


def test_verify_consistency_accepts_a_freshly_built_cache():
    non_visual, initial, _, _ = _selections()
    dense, sparse = _caches(initial, non_visual)

    assert verify_sparse_visual_consistency(
        sparse, dense, non_visual, initial, initial.shape[-1]
    ) == 0


def test_verify_consistency_detects_a_single_corrupted_slot():
    non_visual, initial, _, _ = _selections()
    dense, sparse = _caches(initial, non_visual)
    sparse[0][0].data[0, 0, 0, 0] += 1

    assert verify_sparse_visual_consistency(
        sparse, dense, non_visual, initial, initial.shape[-1]
    ) == 1


def test_verify_consistency_detects_a_stale_selection():
    """Cache still holds `initial` while the routing state claims `first`."""
    non_visual, initial, first, _ = _selections()
    dense, sparse = _caches(initial, non_visual)

    assert verify_sparse_visual_consistency(
        sparse, dense, non_visual, first, first.shape[-1]
    ) > 0


def test_verify_consistency_passes_after_a_full_refresh():
    non_visual, initial, first, _ = _selections()
    dense, sparse = _caches(initial, non_visual)
    refresh_sparse_visual_kv(sparse, dense, non_visual, first, first.shape[-1])

    assert verify_sparse_visual_consistency(
        sparse, dense, non_visual, first, first.shape[-1]
    ) == 0


def test_verify_consistency_passes_after_an_incremental_refresh():
    non_visual, initial, first, _ = _selections()
    dense, sparse = _caches(initial, non_visual)
    incremental_refresh_sparse_visual_kv(
        sparse, dense, non_visual, initial, first, first.shape[-1]
    )

    assert verify_sparse_visual_consistency(
        sparse, dense, non_visual, first, first.shape[-1]
    ) == 0


def test_verify_consistency_rejects_a_head_count_mismatch():
    non_visual, initial, _, _ = _selections()
    dense, sparse = _caches(initial, non_visual)
    fewer_heads = initial[:, :1, :]

    with pytest.raises(ValueError):
        verify_sparse_visual_consistency(
            sparse, dense, non_visual, fewer_heads, initial.shape[-1]
        )
