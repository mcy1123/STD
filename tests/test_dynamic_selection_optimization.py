import torch

from std_repro.dynamic_selection import (
    should_refresh_selection,
    topk_indices,
    topk_intersection_counts,
    attention_free_visual_topk,
    verification_query_positions,
)


def test_topk_indices_returns_same_set_as_full_sort():
    scores = torch.tensor([[0.1, 0.9, 0.3, 0.8, 0.2]])
    actual = topk_indices(scores, 2)
    assert set(actual[0].tolist()) == {1, 3}


def test_should_refresh_selection_respects_interval_and_change_threshold():
    assert should_refresh_selection(round_id=2, changed_ratio=0.25, interval=2, min_changed_ratio=0.2)
    assert not should_refresh_selection(round_id=1, changed_ratio=0.25, interval=2, min_changed_ratio=0.2)
    assert not should_refresh_selection(round_id=2, changed_ratio=0.19, interval=2, min_changed_ratio=0.2)


def test_should_refresh_selection_rejects_invalid_parameters():
    for kwargs in (
        {"round_id": 0, "changed_ratio": 0.1, "interval": 0, "min_changed_ratio": 0.0},
        {"round_id": 0, "changed_ratio": 0.1, "interval": 1, "min_changed_ratio": -0.1},
        {"round_id": 0, "changed_ratio": 1.1, "interval": 1, "min_changed_ratio": 0.0},
    ):
        try:
            should_refresh_selection(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid selection refresh parameters were accepted")


def test_topk_intersection_counts_is_vectorized_and_order_independent():
    a = torch.tensor([[[5, 1, 3], [2, 4, 8]]])
    b = torch.tensor([[[3, 6, 7], [8, 4, 9]]])
    assert torch.equal(topk_intersection_counts(a, b), torch.tensor([[1, 2]]))


def test_verification_query_positions_supports_collect_two_query_policy():
    assert verification_query_positions(accept_len=3, pending_len=2, propose_len=5, mode="two") == [0, 4]
    assert verification_query_positions(accept_len=0, pending_len=2, propose_len=5, mode="two") == [0]
    assert verification_query_positions(accept_len=3, pending_len=0, propose_len=5, mode="two") == [0, 2]
    assert verification_query_positions(accept_len=3, pending_len=2, propose_len=5, mode="three") == [0, 4, 5]


def test_attention_free_visual_topk_combines_core_and_uniform_coverage():
    # [layers, kv_heads, visual_len, head_dim], with an obvious outlier at 5.
    keys = torch.zeros(2, 2, 8, 4)
    values = torch.ones_like(keys)
    keys[:, :, 5, :] = 10.0
    values[:, :, 7, :] = 8.0
    selected = attention_free_visual_topk(keys, values, k=4, coverage_ratio=0.5)
    assert selected.shape == (2, 4)
    assert 5 in selected[0].tolist()
    assert 7 in selected[0].tolist()
    # Coverage contributes deterministic endpoints for this short sequence.
    assert 0 in selected[0].tolist()
    assert 4 in selected[0].tolist()
    assert all(len(set(row.tolist())) == 4 for row in selected)


def test_attention_free_bootstrap_uses_local_not_global_frame_centroid():
    # A large between-frame offset must not hide the first frame's outlier.
    keys = torch.tensor([0., 0., 0., 3., 100., 100., 100., 100.]).reshape(1, 1, 8, 1)
    selected = attention_free_visual_topk(
        keys, torch.ones_like(keys), k=1, coverage_ratio=0,
        value_weight=0, window_size=4,
    )
    assert selected.tolist() == [[3]]


def test_attention_free_bootstrap_handles_partial_last_window():
    keys = torch.zeros(1, 1, 9, 2)
    selected = attention_free_visual_topk(
        keys, keys, k=9, coverage_ratio=0.25, window_size=4,
    )
    assert selected.tolist() == [list(range(9))]
