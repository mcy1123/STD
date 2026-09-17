"""Selection policies and the runtime verification-score collector for the
Verification-Guided STD MVP (Fixed-K Previous Verification Guided Top-K Selection).

This module contains only *read-only* machinery: it selects which visual KV
tokens the sparse draft attends to, and it observes the dense verifier's visual
attention. It never changes logits, KV contents, or the attention backend, and it
never touches the speculative acceptance loop.

Two policies are provided:

  * ``StaticPolicy``       — S_next = S_initial (the frozen static STD baseline).
  * ``PreviousVerifyTopKPolicy`` — S_next = TopK(current dense-verification
    visual relevance A_t), with no smoothing / EMA / predictive routing.

The runtime collector mirrors the Oracle Study collector (``scripts/analysis/
attention_trace.py``) but keeps the *per-layer* ``[num_layers, kv_heads,
visual_len]`` relevance, because the sparse cache stores a per-layer, per-head
top-K selection.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class SelectionState:
    """Current visual selection plus per-round bookkeeping."""

    # Stacked per-layer top-K absolute visual positions: [num_layers, kv_heads, k].
    indices: torch.Tensor
    k: int
    round_id: int
    # Jaccard(S_prev, S_this) averaged over layers and KV heads (0..1).
    selection_overlap: float
    # Wall-clock time spent computing this selection update, in seconds.
    update_time: float


def topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the K largest scores (ordering within the result is unspecified)."""
    if k < 1 or k > scores.shape[-1]:
        raise ValueError(f"k must be in [1, {scores.shape[-1]}], got {k}.")
    # Full argsort is O(N log N) even though routing only needs K entries.
    # ``sorted=False`` avoids an unnecessary ordering pass; callers that need
    # stable absolute positions sort the selected K indices afterward.
    return torch.topk(scores, k, dim=-1, largest=True, sorted=False).indices


def should_refresh_selection(
    round_id: int,
    changed_ratio: float,
    interval: int = 1,
    min_changed_ratio: float = 0.0,
) -> bool:
    """Return whether a candidate routing update should reach the sparse cache.

    ``round_id`` is one-based. ``changed_ratio`` is Jaccard distance (1 minus
    set overlap), not the fraction of K tokens replaced. Tiny Top-K changes
    are ignored so they do not trigger a costly cache rewrite.
    """
    if round_id < 1:
        raise ValueError("round_id must be positive")
    if interval < 1:
        raise ValueError("interval must be positive")
    if not 0.0 <= changed_ratio <= 1.0:
        raise ValueError("changed_ratio must be between 0 and 1")
    if not 0.0 <= min_changed_ratio <= 1.0:
        raise ValueError("min_changed_ratio must be between 0 and 1")
    return round_id % interval == 0 and changed_ratio >= min_changed_ratio


def topk_jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean Jaccard over layers and KV heads of two [L, H, k] top-K selections."""
    intersections = topk_intersection_counts(a, b).float()
    unions = 2 * a.shape[-1] - intersections
    return float((intersections / unions.clamp_min(1)).mean()) if intersections.numel() else 0.0


def topk_intersection_counts(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-layer/head intersection sizes without Python sets or token loops.

    Top-K selections contain unique indices, but need not arrive sorted.
    """
    if a.shape != b.shape:
        raise ValueError(f"Selection shape mismatch: {a.shape} vs {b.shape}.")
    if a.ndim != 3:
        raise ValueError("Selections must have shape [layers, heads, k].")
    k = a.shape[-1]
    if k == 0:
        return torch.zeros(a.shape[:-1], dtype=torch.long, device=a.device)
    sorted_a = torch.sort(a, dim=-1).values.contiguous()
    sorted_b = torch.sort(b, dim=-1).values.contiguous()
    positions = torch.searchsorted(sorted_b, sorted_a)
    matches = (positions < k) & (sorted_b.gather(-1, positions.clamp(max=k - 1)) == sorted_a)
    return matches.sum(dim=-1)


def attention_free_visual_topk(
    keys: torch.Tensor,
    values: torch.Tensor,
    k: int,
    coverage_ratio: float = 0.25,
    value_weight: float = 0.25,
    window_size: int = 0,
) -> torch.Tensor:
    """Select a shared visual top-K from cached K/V without attention scores.

    ``keys`` and ``values`` have shape ``[layers, kv_heads, visual_len, dim]``.
    A windowed centroid residual and value norm are z-normalized per
    layer/head, averaged across layers, and combined with deterministic
    uniform coverage.  The returned indices are ``[kv_heads, k]`` and sorted.
    This is a bootstrap heuristic only; it is deliberately independent of the
    model implementation so it can be tested on CPU and compared against the
    attention-selected bootstrap.
    """
    if keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("keys and values must have equal shape [layers, heads, visual_len, dim].")
    layers, heads, visual_len, _ = keys.shape
    if not 1 <= k <= visual_len:
        raise ValueError(f"k must be in [1, {visual_len}], got {k}.")
    if not 0.0 <= coverage_ratio <= 1.0:
        raise ValueError("coverage_ratio must be between 0 and 1.")
    if value_weight < 0:
        raise ValueError("value_weight must be non-negative.")
    if window_size < 0:
        raise ValueError("window_size must be non-negative (0 means global).")

    kf = keys.float()
    vf = values.float()
    window_size = min(window_size or visual_len, visual_len)
    full_length = (visual_len // window_size) * window_size
    blocks = kf[:, :, :full_length].reshape(layers, heads, -1, window_size, keys.shape[-1])
    residual = torch.linalg.vector_norm(blocks - blocks.mean(dim=3, keepdim=True), dim=-1).flatten(2)
    if full_length < visual_len:
        tail = kf[:, :, full_length:]
        residual = torch.cat([
            residual, torch.linalg.vector_norm(tail - tail.mean(dim=2, keepdim=True), dim=-1)
        ], dim=2)
    value_norm = torch.linalg.vector_norm(vf, dim=-1)

    def _zscore(x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean(dim=-1, keepdim=True)) / x.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(1e-6)

    scores = (_zscore(residual) + value_weight * _zscore(value_norm)).mean(dim=0)
    coverage_k = min(k, int(round(k * coverage_ratio)))
    coverage = torch.div(
        torch.arange(coverage_k, device=keys.device) * visual_len,
        max(coverage_k, 1),
        rounding_mode="floor",
    ).long() if coverage_k else torch.empty(0, dtype=torch.long, device=keys.device)
    selected_rows = []
    for head in range(heads):
        mask = torch.ones(visual_len, dtype=torch.bool, device=keys.device)
        if coverage.numel():
            mask[coverage] = False
        remaining = k - coverage.numel()
        candidates = torch.nonzero(mask, as_tuple=False).flatten()
        core = candidates[torch.topk(scores[head, candidates], remaining, sorted=False).indices] if remaining else candidates[:0]
        selected_rows.append(torch.sort(torch.cat([coverage, core])).values)
    return torch.stack(selected_rows, dim=0)


def stack_topk(topk_positions: List[torch.Tensor]) -> torch.Tensor:
    """Stack a per-layer ``List[[kv_heads, k]]`` into ``[num_layers, kv_heads, k]``."""
    return torch.stack([t.to(torch.long).cpu() for t in topk_positions], dim=0)


class SelectionPolicy(ABC):
    """Interface for choosing the sparse draft's visual KV selection."""

    @abstractmethod
    def initialize(self, initial_topk: List[torch.Tensor], k: int) -> SelectionState:
        """Build the initial selection from the static prefill top-K (S_0)."""

    @abstractmethod
    def update(
        self,
        verification_scores: torch.Tensor,
        state: SelectionState,
        k: int,
    ) -> SelectionState:
        """Update the selection given one round's dense-verification relevance A_t.

        ``verification_scores`` is ``[num_layers, kv_heads, visual_len]`` on CPU.
        """


class StaticPolicy(SelectionPolicy):
    """Keep S_0 unchanged for every round (frozen static STD)."""

    def initialize(self, initial_topk: List[torch.Tensor], k: int) -> SelectionState:
        indices = stack_topk(initial_topk)
        return SelectionState(indices=indices, k=k, round_id=0, selection_overlap=1.0, update_time=0.0)

    def update(self, verification_scores, state, k) -> SelectionState:
        state.round_id += 1
        state.update_time = 0.0
        state.selection_overlap = 1.0
        return state


class PreviousVerifyTopKPolicy(SelectionPolicy):
    """S_next = TopK(A_t), where A_t is the previous round's dense verification.

    Fixed K, no smoothing. ``visual_positions`` maps the relevance columns
    (relative visual index 0..visual_len-1) to absolute prompt positions.
    """

    def __init__(self, visual_positions: torch.Tensor):
        self.visual_positions = visual_positions.to(torch.long).cpu()

    def initialize(self, initial_topk: List[torch.Tensor], k: int) -> SelectionState:
        indices = stack_topk(initial_topk)
        return SelectionState(indices=indices, k=k, round_id=0, selection_overlap=1.0, update_time=0.0)

    def update(self, verification_scores, state, k) -> SelectionState:
        t0 = time.perf_counter()
        # verification_scores: [num_layers, kv_heads, visual_len] (cpu).
        rel = topk_indices(verification_scores, k)               # [L, H, k]
        abs_idx = self.visual_positions[rel]                      # [L, H, k]
        abs_idx, _ = torch.sort(abs_idx, dim=-1)
        jacc = topk_jaccard(state.indices, abs_idx)
        elapsed = time.perf_counter() - t0
        return SelectionState(
            indices=abs_idx,
            k=k,
            round_id=state.round_id + 1,
            selection_overlap=jacc,
            update_time=elapsed,
        )


class RuntimeVerificationCollector:
    """Read-only per-round dense-verification visual relevance collector.

    Installs ``_std_trace_hook`` on every attention module (a plain attribute
    lookup that is absent during baseline runs) and captures, per round, the
    per-layer visual-only softmax attention mass:

        A_t : [num_layers, kv_heads, visual_len]

    It never materializes a full attention matrix and never keeps per-layer
    per-round tensors longer than one round, so memory stays
    O(num_layers * kv_heads * visual_len) per round.
    """

    def __init__(self, model, visual_positions: torch.Tensor):
        self.model = model
        self.visual_positions = visual_positions.to(torch.long).cpu()
        self.visual_len = int(visual_positions.numel())
        self.num_kv_heads = model.config.num_key_value_heads
        self.rounds: List[torch.Tensor] = []
        self._active = False
        self._layer_scores: Dict[int, torch.Tensor] = {}
        self._vis_pos_gpu: Optional[torch.Tensor] = None
        # Wall-clock seconds spent inside the hook (GEMM + softmax + .cpu()).
        self.collect_time = 0.0

    def install(self) -> None:
        for layer in self.model.model.layers:
            layer.self_attn._std_trace_hook = self._hook

    def uninstall(self) -> None:
        for layer in self.model.model.layers:
            if hasattr(layer.self_attn, "_std_trace_hook"):
                del layer.self_attn._std_trace_hook

    def begin_verification(self, round_id: int) -> None:
        self._active = True
        self._layer_scores = {}

    def end_verification(self) -> None:
        self._active = False
        if not self._layer_scores:
            return
        order = sorted(self._layer_scores)
        stacked = torch.stack([self._layer_scores[i] for i in order], dim=0)
        self.rounds.append(stacked)
        self._layer_scores = {}

    def latest_scores(self) -> Optional[torch.Tensor]:
        return self.rounds[-1] if self.rounds else None

    def _hook(self, layer_idx: int, query_states: torch.Tensor, key_states: torch.Tensor) -> None:
        if not self._active:
            return
        t0 = time.perf_counter()
        with torch.no_grad():
            bsz, num_heads, q_len, hd = query_states.shape
            num_kv_heads = key_states.shape[1]
            num_groups = num_heads // num_kv_heads
            # Average query heads within each GQA group -> [bsz, kv_heads, q_len, hd].
            q_kv = query_states.float().reshape(bsz, num_kv_heads, num_groups, q_len, hd).mean(dim=2)
            if self._vis_pos_gpu is None or self._vis_pos_gpu.device != key_states.device:
                self._vis_pos_gpu = self.visual_positions.to(key_states.device)
            k_vis = key_states[:, :, self._vis_pos_gpu, :].float()
            logits = (q_kv @ k_vis.transpose(-2, -1)) * (hd ** -0.5)
            w = logits.softmax(dim=-1)
            self._layer_scores[layer_idx] = w.sum(dim=2).squeeze(0).cpu()
        self.collect_time += time.perf_counter() - t0


def verification_query_positions(
    accept_len: int,
    pending_len: int,
    propose_len: int,
    mode: str = "three",
) -> List[int]:
    """Pick representative verification query positions.

    The dense verification forward runs ``verify_input = dense_pending + draft``
    of length ``L = pending_len + propose_len``. Output position ``i`` predicts the
    greedy target for ``verify_input[i]``. V2's ``two`` mode keeps the first
    available query and the query that predicts the bonus token; ``three`` is a
    legacy control that additionally samples one lookahead query:

      * ``first_valid``    — position 0 (the first speculative prediction);
      * ``accept_boundary``— position ``pending_len + accept_len - 1`` (the query
        whose input is the last accepted draft token, and whose output is the
        bonus token), present iff ``accept_len >= 1``;
      * ``legacy_lookahead`` — position ``pending_len + accept_len`` in three-query
        mode, clamped to the last query.

    Positions are deduplicated and clamped to ``[0, L)``.
    """
    if mode not in {"two", "three"}:
        raise ValueError("mode must be 'two' or 'three'.")
    L = pending_len + propose_len
    candidates: List[int] = [0]
    if mode == "two":
        # The bonus token is predicted by the query after the last accepted
        # input. With pending context and zero accepted drafts, q0 is already
        # that bonus query; with no pending context, q0 is the only available
        # proxy because the first token came from dense_next.
        if accept_len >= 1:
            candidates.append(pending_len + accept_len - 1)
        elif pending_len > 0:
            candidates.append(0)
    else:
        # Keep the legacy three-query control, including its lookahead query
        # after the rejection boundary, for apples-to-apples ablations.
        if accept_len >= 1:
            candidates.append(pending_len + accept_len - 1)
        candidates.append(min(pending_len + accept_len, L - 1))
    seen = set()
    out: List[int] = []
    for p in candidates:
        if 0 <= p < L and p not in seen:
            seen.add(p)
            out.append(p)
    return out


class VerificationCollectorV2:
    """Two-phase, 3-query visual relevance collector.

    V1 (`RuntimeVerificationCollector`) computes the visual softmax relevance for
    *all* query positions synchronously inside the forward hook, forcing a
    per-layer GPU->CPU sync on the decode critical path. V2 instead:

      1. the forward hook only *caches* ``query_states`` (post-RoPE) on GPU with
         no GEMM and no ``.cpu()``, so it never blocks the dense verification;
      2. after the acceptance loop knows ``accept_len``, ``compute()`` gathers the
         visual keys from the canonical dense cache, restricts attention to the
         three decision-relevant query positions, and performs one batched
         GEMM/softmax followed by a single ``.cpu()``.

    Output relevance has the same shape as V1: ``[num_layers, kv_heads,
    visual_len]``.
    """

    def __init__(self, model, visual_positions: torch.Tensor):
        self.model = model
        self.visual_positions = visual_positions.to(torch.long).cpu()
        self.visual_len = int(visual_positions.numel())
        self.num_kv_heads = model.config.num_key_value_heads
        self._active = False
        self._query_cache: Dict[int, torch.Tensor] = {}
        self._latest: Optional[torch.Tensor] = None
        self.collect_time = 0.0

    def install(self) -> None:
        for layer in self.model.model.layers:
            layer.self_attn._std_trace_hook = self._hook

    def uninstall(self) -> None:
        for layer in self.model.model.layers:
            if hasattr(layer.self_attn, "_std_trace_hook"):
                del layer.self_attn._std_trace_hook

    def begin_verification(self, round_id: int) -> None:
        self._active = True
        self._query_cache = {}

    def end_verification(self) -> None:
        self._active = False

    def _hook(self, layer_idx: int, query_states: torch.Tensor, key_states: torch.Tensor) -> None:
        if not self._active:
            return
        self._query_cache[layer_idx] = query_states.detach()

    def compute(self, query_positions: List[int], dense_past_key_values) -> Optional[torch.Tensor]:
        """Compute ``[num_layers, kv_heads, visual_len]`` relevance for the given
        verification query positions, gathering visual keys from the dense cache."""
        if not query_positions:
            return None
        t0 = time.perf_counter()
        q_pos = torch.tensor(query_positions, dtype=torch.long)
        device = dense_past_key_values[0][0].data.device
        q_pos_gpu = q_pos.to(device)
        vis_gpu = self.visual_positions.to(device)
        gpu_scores = []
        for layer_idx, layer_cache in enumerate(dense_past_key_values):
            q = self._query_cache.get(layer_idx)
            if q is None:
                continue
            bsz, num_heads, q_len, hd = q.shape
            num_groups = num_heads // self.num_kv_heads
            q_sel = q[:, :, q_pos_gpu, :].contiguous()                    # [bsz, heads, nq, hd]
            q_kv = q_sel.float().reshape(bsz, self.num_kv_heads, num_groups, -1, hd).mean(dim=2)
            ddata = layer_cache[0].data                                   # [bsz, kv_heads, len, hd]
            k_vis = ddata[:, :, vis_gpu, :].float()                       # [bsz, kv_heads, visual_len, hd]
            logits = (q_kv @ k_vis.transpose(-2, -1)) * (hd ** -0.5)
            w = logits.softmax(dim=-1)
            gpu_scores.append(w.sum(dim=2).squeeze(0))                    # [kv_heads, visual_len]
        if not gpu_scores:
            self.collect_time += time.perf_counter() - t0
            return None
        stacked = torch.stack(gpu_scores, dim=0).cpu()                    # single .cpu()
        self.collect_time += time.perf_counter() - t0
        self._latest = stacked
        return stacked

    def latest_scores(self) -> Optional[torch.Tensor]:
        return self._latest


class VerificationCollectorV3:
    """Fused, all-query visual relevance collector (no per-layer GPU->CPU sync).

    V1 computes the visual softmax relevance for *all* query positions inside
    the forward hook and immediately calls ``.cpu()`` on every layer, forcing a
    per-layer GPU->CPU sync that serializes the decode loop on the CPU. V2 defers
    the GEMM to a post-verification ``compute()`` pass, which removes the sync but
    turns the collector into a serial GPU pass that no longer overlaps the dense
    verification forward (net slower).

    V3 keeps V1's in-hook GEMM (so the relevance kernels are queued on the same
    stream as the verification attention and overlap it) but drops the per-layer
    ``.cpu()``: ``_hook`` accumulates per-layer GPU tensors, and ``end_verification``
    performs a single ``torch.stack(...).cpu()`` sync. The CPU therefore blocks only
    once per round, at the very end, instead of once per layer.

    Output relevance is identical in shape to V1/V2: ``[num_layers, kv_heads,
    visual_len]`` (summed over all query positions).
    """

    def __init__(self, model, visual_positions: torch.Tensor):
        self.model = model
        self.visual_positions = visual_positions.to(torch.long).cpu()
        self.visual_len = int(visual_positions.numel())
        self.num_kv_heads = model.config.num_key_value_heads
        self.rounds: List[torch.Tensor] = []
        self._active = False
        self._layer_scores: Dict[int, torch.Tensor] = {}
        self._vis_pos_gpu: Optional[torch.Tensor] = None
        # Wall-clock seconds spent on CPU launch overhead inside the hook (no sync).
        self.collect_time = 0.0

    def install(self) -> None:
        for layer in self.model.model.layers:
            layer.self_attn._std_trace_hook = self._hook

    def uninstall(self) -> None:
        for layer in self.model.model.layers:
            if hasattr(layer.self_attn, "_std_trace_hook"):
                del layer.self_attn._std_trace_hook

    def begin_verification(self, round_id: int) -> None:
        self._active = True
        self._layer_scores = {}

    def end_verification(self) -> None:
        self._active = False
        if not self._layer_scores:
            return
        order = sorted(self._layer_scores)
        stacked = torch.stack([self._layer_scores[i] for i in order], dim=0).cpu()  # single sync
        self.rounds.append(stacked)
        self._layer_scores = {}

    def latest_scores(self) -> Optional[torch.Tensor]:
        return self.rounds[-1] if self.rounds else None

    def _hook(self, layer_idx: int, query_states: torch.Tensor, key_states: torch.Tensor) -> None:
        if not self._active:
            return
        t0 = time.perf_counter()
        with torch.no_grad():
            bsz, num_heads, q_len, hd = query_states.shape
            num_kv_heads = key_states.shape[1]
            num_groups = num_heads // num_kv_heads
            q_kv = query_states.float().reshape(bsz, num_kv_heads, num_groups, q_len, hd).mean(dim=2)
            if self._vis_pos_gpu is None or self._vis_pos_gpu.device != key_states.device:
                self._vis_pos_gpu = self.visual_positions.to(key_states.device)
            k_vis = key_states[:, :, self._vis_pos_gpu, :].float()
            logits = (q_kv @ k_vis.transpose(-2, -1)) * (hd ** -0.5)
            w = logits.softmax(dim=-1)
            self._layer_scores[layer_idx] = w.sum(dim=2).squeeze(0)  # stays on GPU
        self.collect_time += time.perf_counter() - t0
