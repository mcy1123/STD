"""Read-only attention-trace collector for STD dense verification.

Captures visual attention relevance during canonical dense prefill (A_0) and
per-round dense verification (A_t) without altering logits / KV / correctness.
It relies on the `_std_trace_hook` injection point in `modeling_qwen2_5_vl.py`
(a plain attribute lookup that is absent during baseline runs).

The collector reduces query/key states to a per-round `[num_kv_heads, visual_len]`
visual-only relevance score (visual softmax attention mass, layer-mean, summed
over query positions). It never materializes a full
`[num_heads, q_len, full_context]` attention matrix and never keeps per-layer
per-round tensors, so memory stays O(num_kv_heads * visual_len) per round.

With `record_per_query=True` the *verification rounds* keep the query axis
(`[num_kv_heads, q_len, visual_len]`) so that query-subset collectors
(all / two / three-query) can be compared offline. The prefill trace is always
summed over queries: its q_len is the whole prompt, so a per-query copy would be
orders of magnitude larger and is not needed by any analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class RoundTrace:
    """One dense-verification round's aggregated visual relevance."""

    round_id: int
    # Cross-layer mean relevance (cpu fp32). Shape is [num_kv_heads, visual_len]
    # by default, or [num_kv_heads, query_len, visual_len] when the collector
    # was constructed with record_per_query=True.
    visual_scores: torch.Tensor
    query_len: int = 0
    per_query: bool = False


class AttentionTraceCollector:
    """Injects a read-only hook into every attention module.

    `begin_prefill`/`end_prefill` capture the static A_0 relevance (canonical
    dense text prefill). `begin_verification`/`end_verification` capture each
    decode round's A_t relevance.
    """

    def __init__(self, model, visual_positions: torch.Tensor, record_per_query: bool = False):
        self.model = model
        # CPU long tensor: actual KV indices of visual tokens in the dense cache.
        self.visual_positions = visual_positions
        self.visual_len = int(visual_positions.numel())
        self.num_kv_heads = model.config.num_key_value_heads
        self.record_per_query = bool(record_per_query)
        self.prefill_scores: Optional[torch.Tensor] = None
        self.rounds: List[RoundTrace] = []
        self._active = False
        self._label = None
        self._layer_scores: Dict[int, torch.Tensor] = {}
        self._query_len = 0
        self._vis_pos_gpu: Optional[torch.Tensor] = None

    def install(self) -> None:
        for layer in self.model.model.layers:
            layer.self_attn._std_trace_hook = self._hook

    def uninstall(self) -> None:
        for layer in self.model.model.layers:
            if hasattr(layer.self_attn, "_std_trace_hook"):
                del layer.self_attn._std_trace_hook

    def begin_prefill(self) -> None:
        self._begin("prefill")

    def end_prefill(self) -> None:
        self._end(target="prefill")

    def begin_verification(self, round_id: int) -> None:
        self._begin(round_id)

    def end_verification(self) -> None:
        self._end(target="round")

    def _begin(self, label) -> None:
        self._active = True
        self._label = label
        self._layer_scores = {}
        self._query_len = 0

    def _end(self, target: str) -> None:
        self._active = False
        if not self._layer_scores:
            return
        # Prefill keeps the query axis summed (q_len is the whole prompt);
        # only verification rounds may retain it.
        keep_query = self.record_per_query and target != "prefill"
        # [num_layers, kv_heads, (q_len,) visual_len] -> layer-mean.
        stacked = torch.stack(list(self._layer_scores.values()), dim=0)
        visual_scores = stacked.mean(dim=0)
        if target == "prefill":
            self.prefill_scores = visual_scores
        else:
            self.rounds.append(
                RoundTrace(
                    round_id=self._label,
                    visual_scores=visual_scores,
                    query_len=self._query_len,
                    per_query=keep_query,
                )
            )
        self._layer_scores = {}

    def _hook(self, layer_idx: int, query_states: torch.Tensor, key_states: torch.Tensor) -> None:
        if not self._active:
            return
        with torch.no_grad():
            bsz, num_heads, q_len, hd = query_states.shape
            num_kv_heads = key_states.shape[1]
            num_groups = num_heads // num_kv_heads
            self._query_len = q_len
            # Average query heads within each GQA group -> [bsz, kv_heads, q_len, hd].
            q_kv = query_states.float().view(bsz, num_kv_heads, num_groups, q_len, hd).mean(dim=2)
            if self._vis_pos_gpu is None or self._vis_pos_gpu.device != key_states.device:
                self._vis_pos_gpu = self.visual_positions.to(key_states.device)
            # Visual keys only -> [bsz, kv_heads, visual_len, hd].
            k_vis = key_states.float()[:, :, self._vis_pos_gpu, :]
            # Raw logits then visual-only softmax (monotone w.r.t. full softmax,
            # preserving Top-K ordering).
            logits = (q_kv @ k_vis.transpose(-2, -1)) * (hd ** -0.5)
            w = logits.softmax(dim=-1)
            # Default: sum attention mass across query positions
            # -> [bsz, kv_heads, visual_len]. Per-query: keep the query axis
            # -> [bsz, kv_heads, q_len, visual_len].
            if self.record_per_query and self._label != "prefill":
                self._layer_scores[layer_idx] = w.squeeze(0).cpu()
            else:
                self._layer_scores[layer_idx] = w.sum(dim=2).squeeze(0).cpu()
