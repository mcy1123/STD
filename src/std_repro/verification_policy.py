"""Small, hardware-independent verification fallback policies."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch


def positional_token_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
    """Measure positional token disagreement, counting any length difference."""

    reference_flat = reference.flatten()
    candidate_flat = candidate.flatten()
    reference_len = int(reference_flat.numel())
    candidate_len = int(candidate_flat.numel())
    compared = min(reference_len, candidate_len)
    positional_mismatches = int(
        (reference_flat[:compared] != candidate_flat[:compared]).sum().item()
    )
    mismatch_count = positional_mismatches + abs(reference_len - candidate_len)
    denominator = max(reference_len, candidate_len)
    return {
        "mismatch_token_count": mismatch_count,
        "token_level_agreement": 1.0 - mismatch_count / denominator if denominator else 1.0,
    }


def min_prediction_margin(
    logits: torch.Tensor,
    prediction_indices: Sequence[int],
    has_dense_pending: bool,
) -> Optional[float]:
    """Return the smallest top-1/top-2 margin among committed predictions."""

    logit_indices = []
    for prediction_idx in prediction_indices:
        logit_idx = prediction_idx if has_dense_pending else prediction_idx - 1
        if 0 <= logit_idx < logits.shape[1]:
            logit_indices.append(logit_idx)
    if not logit_indices:
        return None
    selected = logits[0, logit_indices, :].float()
    top2 = torch.topk(selected, k=2, dim=-1).values
    return float((top2[:, 0] - top2[:, 1]).min().item())


def needs_sequential_fallback(
    mode: str,
    *,
    accept_len: int,
    draft_len: int,
    sequential_fallback_max_accept: int,
    low_margin: bool,
) -> bool:
    """Return whether a parallel verification round needs exact replay."""

    return (
        mode == "sequential_guard"
        or (mode == "sequential_on_low_margin" and low_margin)
        or (mode == "sequential_on_reject" and accept_len < draft_len)
        or (
            mode == "sequential_on_low_accept"
            and accept_len < draft_len
            and accept_len <= sequential_fallback_max_accept
        )
    )
