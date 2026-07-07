"""Sparse-to-Dense decoding for the SpecVLM Qwen2.5-VL fork.

This module keeps the implementation deliberately close to the paper:
one target model, two KV caches, sparse top-K visual KV access for drafting,
and dense full-attention verification.
"""

from __future__ import annotations

import math
import sys
import time
import types
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


SPECVLM_ROOT = Path("/home/mcy/projects/SpecVLM")
if str(SPECVLM_ROOT) not in sys.path:
    sys.path.insert(0, str(SPECVLM_ROOT))

from kv_cache.kv_cache import initialize_past_key_values  # noqa: E402
from models import modeling_qwen2_5_vl as qwen_mod  # noqa: E402
from std_repro.triton_attention import fused_gqa_attention  # noqa: E402
from utils.utils import get_last_video_idx  # noqa: E402


@dataclass
class SparseSelection:
    """Per-layer sparse visual KV choices."""

    topk_positions: List[torch.Tensor]
    non_visual_positions: torch.Tensor
    prompt_len: int
    visual_len: int
    text_len: int
    k: int


@dataclass
class GenerateResult:
    output_ids: torch.Tensor
    decoding_time: float
    inference_time: float
    generate_len: int
    accepted_draft_tokens: int = 0
    proposed_draft_tokens: int = 0
    mean_accept_length: float = 0.0
    decode_rounds: int = 0
    final_gamma: int = 0
    fallback_count: int = 0
    fallback_accepted_extra: int = 0
    verify_margin_reruns: int = 0
    min_verify_margin: float = 0.0
    gamma_history: List[int] = field(default_factory=list)
    proposed_lengths: List[int] = field(default_factory=list)
    accept_lengths: List[int] = field(default_factory=list)
    draft_time: float = 0.0
    verify_time: float = 0.0
    bonus_time: float = 0.0
    cache_adjust_time: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        if self.proposed_draft_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.proposed_draft_tokens


def clone_tensor_dict(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Clone tensor values so prefill helpers can slice without mutating callers."""

    cloned = {}
    for key, value in inputs.items():
        cloned[key] = value.clone() if torch.is_tensor(value) else value
    return cloned


def _first_model_device(model) -> torch.device:
    return model.model.embed_tokens.weight.device


def _move_inputs(inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _fill_cache_length(current_length_data: torch.Tensor, length: int) -> None:
    current_length_data.fill_(int(length))


def _token_argmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def _sdpa_backend_context(backend: str):
    if backend in {"default", "math_on_full_accept"}:
        return nullcontext()
    if backend == "math":
        return torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH])
    raise ValueError(f"Unsupported SDPA backend: {backend!r}.")


def _profile_mark(enabled: bool) -> float:
    if enabled:
        torch.cuda.synchronize()
    return time.time()


def _min_prediction_margin(
    logits: torch.Tensor,
    prediction_indices: Sequence[int],
    has_dense_pending: bool,
) -> Optional[float]:
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


def prefill_prompt(
    model,
    inputs: Dict[str, torch.Tensor],
    past_key_values,
    video_token_id: int,
    output_attentions: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, ...]], int]:
    """Prefill video-prefix then text suffix, matching SpecVLM's Qwen path."""

    local_inputs = clone_tensor_dict(inputs)
    device = _first_model_device(model)
    local_inputs = _move_inputs(local_inputs, device)

    input_ids = local_inputs["input_ids"]
    last_video_idx = get_last_video_idx(input_ids[0], video_token_id)
    if last_video_idx is None or last_video_idx < 0:
        raise ValueError("No video token found in the prompt; STD visual KV selection requires video input.")

    text_input_ids = input_ids[:, last_video_idx + 1 :].clone()
    video_inputs = dict(local_inputs)
    video_inputs["input_ids"] = input_ids[:, : last_video_idx + 1]
    if "attention_mask" in video_inputs:
        video_inputs["attention_mask"] = video_inputs["attention_mask"][:, : last_video_idx + 1]

    model(**video_inputs, past_key_values=past_key_values)

    text_kwargs = {
        "input_ids": text_input_ids,
        "past_key_values": past_key_values,
        "output_attentions": output_attentions,
    }

    output = model(**text_kwargs)
    next_token = _token_argmax(output.logits)
    return input_ids.clone(), next_token, output.attentions if output_attentions else None, last_video_idx + 1


def build_sparse_selection(
    attentions: Sequence[torch.Tensor],
    full_input_ids: torch.Tensor,
    video_token_id: int,
    text_start: int,
    target_k_plus_text: int = 1024,
    explicit_k: Optional[int] = None,
    num_key_value_heads: int = 4,
) -> SparseSelection:
    """Select top-K visual cache positions from text-to-video attention."""

    prompt_ids = full_input_ids[0]
    visual_positions = torch.nonzero(prompt_ids == video_token_id, as_tuple=False).flatten().cpu()
    if visual_positions.numel() == 0:
        raise ValueError("Cannot build sparse selection without visual/video token positions.")

    non_visual_positions = torch.nonzero(prompt_ids != video_token_id, as_tuple=False).flatten().cpu()
    prompt_len = int(prompt_ids.numel())
    text_len = max(1, prompt_len - int(text_start))
    k = int(explicit_k) if explicit_k is not None else max(1, int(target_k_plus_text) - text_len)
    k = min(k, int(visual_positions.numel()))

    layer_positions: List[torch.Tensor] = []
    for layer_attn in attentions:
        # [batch, query_heads, text_len, prompt_len]
        attn = layer_attn.detach().float().cpu()[0]
        query_heads = attn.shape[0]
        heads_per_kv = query_heads // num_key_value_heads
        per_kv = []
        for kv_head in range(num_key_value_heads):
            h0 = kv_head * heads_per_kv
            h1 = (kv_head + 1) * heads_per_kv
            grouped = attn[h0:h1, :, visual_positions]
            scores = grouped.sum(dim=0).mean(dim=0)
            top_local = torch.topk(scores, k=k, largest=True).indices
            per_kv.append(torch.sort(visual_positions[top_local]).values)
        layer_positions.append(torch.stack(per_kv, dim=0))

    return SparseSelection(
        topk_positions=layer_positions,
        non_visual_positions=non_visual_positions,
        prompt_len=prompt_len,
        visual_len=int(visual_positions.numel()),
        text_len=text_len,
        k=k,
    )


class SparseDraftController:
    """Monkey-patch Qwen2.5-VL attention modules with sparse draft behavior."""

    def __init__(self, model, selection: SparseSelection, sparse_attn_mode: str = "gqa_sdpa",
                 use_compile: bool = False):
        if sparse_attn_mode not in {"repeat_sdpa", "gqa_sdpa", "triton_gqa"}:
            raise ValueError(f"Unsupported sparse_attn_mode={sparse_attn_mode!r}.")
        self.model = model
        self.selection = selection
        self.sparse_attn_mode = sparse_attn_mode
        self.enabled = False
        self._installed = False
        self.compiled_forward = None
        if use_compile:
            self.compiled_forward = _build_compiled_sparse_step(model)

    def install(self) -> None:
        if self._installed:
            return
        for layer in self.model.model.layers:
            attn = layer.self_attn
            if getattr(attn, "_std_original_forward", None) is None:
                attn._std_original_forward = attn.forward
                attn.forward = types.MethodType(_sparse_aware_attention_forward, attn)
            attn._std_sparse_controller = self
        self._installed = True

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled


def compact_sparse_prompt_cache(
    past_key_values,
    current_length_data: torch.Tensor,
    selection: SparseSelection,
) -> int:
    """Physically compact sparse prompt KV caches to the selected KV set.

    The selected order can differ per KV head. For autoregressive q_len=1 sparse
    drafting this is valid because we attend to the whole compacted prefix without
    a causal mask; RoPE is already baked into keys. Generated tokens are appended
    after the compact prompt and use explicit original position_ids.
    """

    compact_len = int(selection.non_visual_positions.numel() + selection.k)
    for layer_idx, layer_cache in enumerate(past_key_values):
        topk = selection.topk_positions[layer_idx]
        for cache in layer_cache:
            data = cache.data
            bsz, kv_heads, _, head_dim = data.shape
            indices = []
            non_visual = selection.non_visual_positions
            for head in range(kv_heads):
                idx = torch.cat([non_visual, topk[head]], dim=0)
                idx = torch.unique(idx, sorted=True).to(data.device)
                if int(idx.numel()) != compact_len:
                    raise RuntimeError("Compacted sparse KV produced an unexpected length.")
                indices.append(idx)
            index = torch.stack(indices, dim=0)
            index = index.view(1, kv_heads, compact_len, 1).expand(bsz, kv_heads, compact_len, head_dim)
            compacted = data.gather(2, index).contiguous()
            data[:, :, :compact_len, :].copy_(compacted)
            cache.current_length.fill_(compact_len)
    current_length_data.fill_(compact_len)
    return compact_len


def copy_prompt_cache(
    source_past_key_values,
    target_past_key_values,
    target_length_data: torch.Tensor,
    prompt_len: int,
) -> None:
    """Copy a full prompt KV cache into another cache before sparse compaction."""

    for source_layer, target_layer in zip(source_past_key_values, target_past_key_values):
        for source_cache, target_cache in zip(source_layer, target_layer):
            source_len = int(source_cache.current_length.item())
            if source_len < prompt_len:
                raise RuntimeError(
                    f"Cannot copy prompt cache: source length {source_len} is shorter than prompt length {prompt_len}."
                )
            src = source_cache.data.narrow(2, 0, prompt_len)
            dst = target_cache.data.narrow(2, 0, prompt_len)
            dst.copy_(src, non_blocking=True)
            target_cache.current_length.fill_(prompt_len)
    target_length_data.fill_(prompt_len)


def _gather_sparse_kv(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    controller: SparseDraftController,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather full non-visual prompt KV, selected visual KV, and all generated KV."""

    bsz, kv_heads, total_len, head_dim = key_states.shape
    selection = controller.selection
    generated_start = min(selection.prompt_len, total_len)
    generated_positions = torch.arange(generated_start, total_len, dtype=torch.long)
    non_visual = selection.non_visual_positions[selection.non_visual_positions < total_len]
    topk = selection.topk_positions[layer_idx]

    gathered_indices = []
    for head in range(kv_heads):
        head_topk = topk[head][topk[head] < total_len]
        idx = torch.cat([non_visual, head_topk, generated_positions], dim=0)
        idx = torch.unique(idx, sorted=True).to(key_states.device)
        gathered_indices.append(idx)

    max_len = max(int(idx.numel()) for idx in gathered_indices)
    if any(int(idx.numel()) != max_len for idx in gathered_indices):
        raise RuntimeError("Sparse KV gather produced uneven per-head lengths.")

    index = torch.stack(gathered_indices, dim=0)
    index = index.view(1, kv_heads, max_len, 1).expand(bsz, kv_heads, max_len, head_dim)
    return key_states.gather(2, index), value_states.gather(2, index)


def _sparse_aware_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    """Instance-level replacement for Qwen2_5_VLSdpaAttention.forward."""

    controller = getattr(self, "_std_sparse_controller", None)
    if controller is None or not controller.enabled:
        return self._std_original_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()
    if q_len != 1:
        raise RuntimeError("Sparse draft attention currently expects autoregressive q_len=1 calls.")

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = qwen_mod.apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        key_states = past_key_value[0].cat(key_states, dim=2)
        value_states = past_key_value[1].cat(value_states, dim=2)

    query_states = query_states.contiguous()
    if controller.sparse_attn_mode == "triton_gqa":
        if bsz != 1:
            raise RuntimeError("triton_gqa sparse attention currently supports batch size 1 only.")
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        attn_output = fused_gqa_attention(query_states, key_states, value_states)
    elif controller.sparse_attn_mode == "gqa_sdpa":
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
            enable_gqa=True,
        )
    else:
        key_states = qwen_mod.repeat_kv(key_states, self.num_key_value_groups)
        value_states = qwen_mod.repeat_kv(value_states, self.num_key_value_groups)

        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, None


def _forward_tokens(model, tokens: torch.Tensor, past_key_values, sparse_controller: Optional[SparseDraftController]) -> torch.Tensor:
    if sparse_controller is not None:
        sparse_controller.set_enabled(True)
        if sparse_controller.compiled_forward is not None:
            try:
                outputs = sparse_controller.compiled_forward(tokens, past_key_values)
            finally:
                sparse_controller.set_enabled(False)
            return outputs.logits
    try:
        outputs = model(input_ids=tokens, past_key_values=past_key_values)
    finally:
        if sparse_controller is not None:
            sparse_controller.set_enabled(False)
    return outputs.logits


def _build_compiled_sparse_step(model):
    """Build a torch.compile'd version of a single-token sparse forward step."""

    def _sparse_step(token: torch.Tensor, past_key_values, position_ids: torch.Tensor):
        return model(input_ids=token, past_key_values=past_key_values, position_ids=position_ids)

    # mode="reduce-overhead" uses CUDA graphs for the compiled region,
    # which is ideal for the repetitive single-token decode pattern.
    return torch.compile(_sparse_step, dynamic=True, mode="reduce-overhead")


def _append_dense_tokens(model, tokens: Sequence[int], past_key_values) -> torch.Tensor:
    device = _first_model_device(model)
    token_tensor = torch.tensor([list(tokens)], dtype=torch.long, device=device)
    outputs = model(input_ids=token_tensor, past_key_values=past_key_values)
    return _token_argmax(outputs.logits)


def _append_sparse_tokens(
    model,
    tokens: Sequence[int],
    past_key_values,
    sparse_controller: SparseDraftController,
    start_position: int,
) -> torch.Tensor:
    device = _first_model_device(model)
    next_token = None
    sparse_controller.set_enabled(True)
    try:
        for offset, token in enumerate(tokens):
            token_tensor = torch.tensor([[int(token)]], dtype=torch.long, device=device)
            position_ids = torch.tensor([[start_position + offset]], dtype=torch.long, device=device)
            outputs = model(input_ids=token_tensor, past_key_values=past_key_values, position_ids=position_ids)
            next_token = _token_argmax(outputs.logits)
    finally:
        sparse_controller.set_enabled(False)
    if next_token is None:
        raise ValueError("Cannot append an empty sparse token sequence.")
    return next_token


def _contains_eos(tokens: Sequence[int], eos_token_id: int) -> Optional[int]:
    for idx, token in enumerate(tokens):
        if int(token) == eos_token_id:
            return idx
    return None


def _sequential_verify_draft(
    model,
    draft: Sequence[int],
    dense_pending: Sequence[int],
    dense_next: torch.Tensor,
    dense_pkv,
    dense_lengths: torch.Tensor,
    dense_cached_len: int,
) -> Tuple[int, int]:
    """Sequentially verify draft tokens from a rolled-back dense cache."""

    device = _first_model_device(model)
    _fill_cache_length(dense_lengths, dense_cached_len)
    if dense_pending:
        next_dense_token = _append_dense_tokens(model, dense_pending, dense_pkv)
    else:
        next_dense_token = dense_next

    accept_len = 0
    while accept_len < len(draft) and int(draft[accept_len]) == int(next_dense_token.item()):
        token_tensor = torch.tensor([[int(draft[accept_len])]], dtype=torch.long, device=device)
        outputs = model(input_ids=token_tensor, past_key_values=dense_pkv)
        next_dense_token = _token_argmax(outputs.logits)
        accept_len += 1
    return accept_len, int(next_dense_token.item())


def _draft_tokens(
    model,
    first_token: torch.Tensor,
    gamma: int,
    past_key_values,
    sparse_controller: SparseDraftController,
    start_position: int,
) -> List[int]:
    token = first_token.to(_first_model_device(model))
    drafted: List[int] = []
    sparse_controller.set_enabled(True)
    try:
        for offset in range(gamma):
            drafted.append(int(token.item()))
            position_ids = torch.tensor([[start_position + offset]], dtype=torch.long, device=token.device)
            outputs = model(input_ids=token, past_key_values=past_key_values, position_ids=position_ids)
            token = _token_argmax(outputs.logits)
    finally:
        sparse_controller.set_enabled(False)
    return drafted


@torch.inference_mode()
def ar_generate_qwen25vl(
    model,
    inputs: Dict[str, torch.Tensor],
    video_token_id: int,
    eos_token_id: int,
    max_new_tokens: int = 256,
    output_attentions: bool = False,
) -> GenerateResult:
    """Vanilla greedy decoding with the same two-stage prefill as STD.

    When output_attentions=True, the AR prefill uses the same attention path as
    the STD attention-selection prefill, ensuring identical KV cache values and
    thus exact token equality between AR and STD from a single shared prefill.
    """

    torch.cuda.synchronize()
    start = time.time()
    past_key_values, _, current_length_data = initialize_past_key_values(model)
    prompt_ids, next_token, _, _ = prefill_prompt(model, inputs, past_key_values, video_token_id, output_attentions)

    generated: List[int] = []
    torch.cuda.synchronize()
    decode_start = time.time()
    device = _first_model_device(model)
    while len(generated) < max_new_tokens:
        token_id = int(next_token.item())
        generated.append(token_id)
        if token_id == eos_token_id:
            break
        outputs = model(input_ids=next_token.to(device), past_key_values=past_key_values)
        next_token = _token_argmax(outputs.logits)

    _fill_cache_length(current_length_data, prompt_ids.shape[1] + len(generated))
    torch.cuda.synchronize()
    end = time.time()
    out = torch.cat(
        [prompt_ids.to(device), torch.tensor([generated], dtype=torch.long, device=device)],
        dim=1,
    )
    return GenerateResult(out, end - decode_start, end - start, len(generated))


@torch.inference_mode()
def std_generate_qwen25vl(
    model,
    inputs: Dict[str, torch.Tensor],
    video_token_id: int,
    eos_token_id: int,
    max_new_tokens: int = 256,
    gamma: int = 9,
    target_k_plus_text: int = 1024,
    explicit_k: Optional[int] = None,
    verify_mode: str = "parallel",
    verify_fallback: str = "none",
    sequential_fallback_max_accept: int = 1,
    profile_decode: bool = False,
    sparse_attn_mode: str = "gqa_sdpa",
    adaptive_gamma_min: Optional[int] = None,
    adaptive_gamma_mode: str = "accept_len",
    reuse_dense_prefill: bool = False,
    copy_sparse_prefill: bool = True,
    verify_attn_backend: str = "default",
    verify_margin_threshold: Optional[float] = None,
    use_compile: bool = False,
) -> Tuple[GenerateResult, SparseSelection]:
    """Sparse-to-Dense greedy decoding."""

    if verify_mode not in {"parallel", "sequential"}:
        raise ValueError(f"Unsupported verify_mode={verify_mode!r}; expected 'parallel' or 'sequential'.")
    if verify_fallback not in {"none", "sequential_on_reject", "sequential_on_low_accept", "sequential_guard"}:
        raise ValueError(
            "verify_fallback must be 'none', 'sequential_on_reject', "
            "'sequential_on_low_accept', or 'sequential_guard'."
        )
    if sequential_fallback_max_accept < 0:
        raise ValueError("sequential_fallback_max_accept must be non-negative.")
    if adaptive_gamma_min is not None and not (1 <= adaptive_gamma_min <= gamma):
        raise ValueError("adaptive_gamma_min must be between 1 and gamma.")
    if adaptive_gamma_mode not in {"accept_len", "conservative"}:
        raise ValueError("adaptive_gamma_mode must be 'accept_len' or 'conservative'.")
    if verify_attn_backend not in {"default", "math", "math_on_full_accept"}:
        raise ValueError("verify_attn_backend must be 'default', 'math', or 'math_on_full_accept'.")
    if verify_margin_threshold is not None and verify_margin_threshold < 0:
        raise ValueError("verify_margin_threshold must be non-negative.")

    torch.cuda.synchronize()
    start = time.time()

    selection_pkv, _, selection_lengths = initialize_past_key_values(model)
    if reuse_dense_prefill:
        dense_pkv = selection_pkv
        dense_lengths = selection_lengths
    else:
        dense_pkv, _, dense_lengths = initialize_past_key_values(model)
    sparse_pkv, _, sparse_lengths = initialize_past_key_values(model)

    prompt_ids, dense_next, attentions, text_start = prefill_prompt(
        model, inputs, selection_pkv, video_token_id, output_attentions=True
    )
    if attentions is None:
        raise RuntimeError("Dense prefill did not return attentions.")

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
    controller = SparseDraftController(model, selection, sparse_attn_mode=sparse_attn_mode, use_compile=use_compile)
    controller.install()

    if not reuse_dense_prefill:
        prompt_ids, dense_next, _, _ = prefill_prompt(model, inputs, dense_pkv, video_token_id, output_attentions=False)
    if copy_sparse_prefill:
        sparse_next = dense_next.clone()
        copy_prompt_cache(dense_pkv, sparse_pkv, sparse_lengths, int(prompt_ids.shape[1]))
    else:
        _, sparse_next, _, _ = prefill_prompt(model, inputs, sparse_pkv, video_token_id, output_attentions=False)
    sparse_prompt_len = compact_sparse_prompt_cache(sparse_pkv, sparse_lengths, selection)

    generated: List[int] = []
    accepted_total = 0
    proposed_total = 0
    accept_lengths: List[int] = []
    proposed_lengths: List[int] = []
    gamma_history: List[int] = []
    dense_pending: List[int] = []
    draft_time = 0.0
    verify_time = 0.0
    bonus_time = 0.0
    cache_adjust_time = 0.0
    decode_rounds = 0
    current_gamma = gamma
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
        gamma_history.append(current_gamma)
        propose_len = min(current_gamma, remaining)
        context_len = prompt_ids.shape[1] + len(generated)
        dense_cached_len = context_len - len(dense_pending)
        sparse_prev_len = sparse_prompt_len + len(generated)

        if verify_mode == "sequential" and dense_pending:
            stage_start = _profile_mark(profile_decode)
            _fill_cache_length(dense_lengths, dense_cached_len)
            dense_next = _append_dense_tokens(model, dense_pending, dense_pkv)
            dense_pending = []
            dense_cached_len = context_len
            cache_adjust_time += _profile_mark(profile_decode) - stage_start

        stage_start = _profile_mark(profile_decode)
        draft = _draft_tokens(model, sparse_next, propose_len, sparse_pkv, controller, start_position=context_len)
        draft_time += _profile_mark(profile_decode) - stage_start
        proposed_total += len(draft)
        proposed_lengths.append(len(draft))

        stage_start = _profile_mark(profile_decode)
        if verify_mode == "parallel":
            verify_input = dense_pending + draft
            verify_tensor = torch.tensor([verify_input], dtype=torch.long, device=device)
            with _sdpa_backend_context(verify_attn_backend):
                verify_outputs = model(input_ids=verify_tensor, past_key_values=dense_pkv)
            verify_argmax = torch.argmax(verify_outputs.logits[0], dim=-1).tolist()
            if dense_pending:
                dense_predictions = [int(x) for x in verify_argmax]
            else:
                dense_predictions = [int(dense_next.item())] + [int(x) for x in verify_argmax]

            accept_len = 0
            while accept_len < len(draft) and draft[accept_len] == dense_predictions[accept_len]:
                accept_len += 1
            bonus_token = dense_predictions[accept_len]
            if verify_margin_threshold is not None:
                margin = _min_prediction_margin(
                    verify_outputs.logits,
                    range(accept_len + 1),
                    has_dense_pending=bool(dense_pending),
                )
                if margin is not None:
                    verify_margins.append(margin)
                if margin is not None and margin < verify_margin_threshold:
                    verify_margin_reruns += 1
                    _fill_cache_length(dense_lengths, dense_cached_len)
                    with _sdpa_backend_context("math"):
                        verify_outputs = model(input_ids=verify_tensor, past_key_values=dense_pkv)
                    verify_argmax = torch.argmax(verify_outputs.logits[0], dim=-1).tolist()
                    if dense_pending:
                        dense_predictions = [int(x) for x in verify_argmax]
                    else:
                        dense_predictions = [int(dense_next.item())] + [int(x) for x in verify_argmax]

                    math_accept_len = 0
                    while math_accept_len < len(draft) and draft[math_accept_len] == dense_predictions[math_accept_len]:
                        math_accept_len += 1
                    if math_accept_len > accept_len:
                        fallback_accepted_extra += math_accept_len - accept_len
                    accept_len = math_accept_len
                    bonus_token = dense_predictions[accept_len]
            if verify_attn_backend == "math_on_full_accept" and accept_len == len(draft):
                fallback_count += 1
                _fill_cache_length(dense_lengths, dense_cached_len)
                with _sdpa_backend_context("math"):
                    verify_outputs = model(input_ids=verify_tensor, past_key_values=dense_pkv)
                verify_argmax = torch.argmax(verify_outputs.logits[0], dim=-1).tolist()
                if dense_pending:
                    dense_predictions = [int(x) for x in verify_argmax]
                else:
                    dense_predictions = [int(dense_next.item())] + [int(x) for x in verify_argmax]

                math_accept_len = 0
                while math_accept_len < len(draft) and draft[math_accept_len] == dense_predictions[math_accept_len]:
                    math_accept_len += 1
                if math_accept_len > accept_len:
                    fallback_accepted_extra += math_accept_len - accept_len
                accept_len = math_accept_len
                bonus_token = dense_predictions[accept_len]
            needs_sequential_fallback = (
                verify_fallback == "sequential_guard"
                or (verify_fallback == "sequential_on_reject" and accept_len < len(draft))
                or (
                    verify_fallback == "sequential_on_low_accept"
                    and accept_len < len(draft)
                    and accept_len <= sequential_fallback_max_accept
                )
            )
            if needs_sequential_fallback:
                fallback_count += 1
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
        else:
            accept_len = 0
            next_dense_token = dense_next
            while accept_len < len(draft) and int(draft[accept_len]) == int(next_dense_token.item()):
                token_tensor = torch.tensor([[int(draft[accept_len])]], dtype=torch.long, device=device)
                outputs = model(input_ids=token_tensor, past_key_values=dense_pkv)
                next_dense_token = _token_argmax(outputs.logits)
                accept_len += 1
            bonus_token = int(next_dense_token.item())
        verify_time += _profile_mark(profile_decode) - stage_start
        accept_lengths.append(accept_len)
        accepted_total += accept_len
        if adaptive_gamma_min is not None:
            if accept_len < propose_len:
                if adaptive_gamma_mode == "accept_len":
                    current_gamma = max(adaptive_gamma_min, accept_len if accept_len > 0 else adaptive_gamma_min)
                else:
                    accept_ratio = accept_len / propose_len if propose_len else 0.0
                    if accept_len == 0:
                        current_gamma = max(adaptive_gamma_min, current_gamma - 2)
                    elif accept_ratio < 0.5:
                        current_gamma = max(adaptive_gamma_min, current_gamma - 1)
            elif current_gamma < gamma:
                current_gamma = min(gamma, current_gamma + 1)

        full_append = draft[:accept_len] + [bonus_token]
        eos_index = _contains_eos(full_append, eos_token_id)
        if eos_index is not None:
            full_append = full_append[: eos_index + 1]

        to_append = full_append[:remaining]
        generated.extend(int(x) for x in to_append)

        reached_limit = len(generated) >= max_new_tokens
        reached_eos = bool(to_append and to_append[-1] == eos_token_id)
        if reached_limit or reached_eos:
            break

        stage_start = _profile_mark(profile_decode)
        # Verification wrote dense_pending plus draft-token KVs. Keep the
        # previous pending bonus and the accepted draft prefix; the new bonus is
        # intentionally left pending and will be cached by the next verification.
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
        final_gamma=current_gamma,
        fallback_count=fallback_count,
        fallback_accepted_extra=fallback_accepted_extra,
        verify_margin_reruns=verify_margin_reruns,
        min_verify_margin=min(verify_margins) if verify_margins else 0.0,
        gamma_history=gamma_history,
        proposed_lengths=proposed_lengths,
        accept_lengths=accept_lengths,
        draft_time=draft_time,
        verify_time=verify_time,
        bonus_time=bonus_time,
        cache_adjust_time=cache_adjust_time,
    )
    return result, selection


def generated_suffix(output_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    return output_ids[:, prompt_len:]


def tokens_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and bool(torch.equal(a.detach().cpu(), b.detach().cpu()))
