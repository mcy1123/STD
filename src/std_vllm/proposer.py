"""Custom speculative-decoding proposers for vLLM, used by the STD workstream.

Registered with vLLM via::

    speculative_config={
        "method": "custom_class",
        "model": "std_vllm.proposer.<ClassName>",
        "num_speculative_tokens": k,
    }

Contract (vLLM v0.22.0, `vllm/v1/spec_decode/custom_class_proposer.py`): the class
is constructed with the `VllmConfig` and must expose a callable `propose`. That is
the *entire* interface -- no `load_model`, no `dummy_run`, no attention-backend
hooks are invoked for this method (each of those call sites in
`v1/worker/gpu_model_runner.py` is guarded by an isinstance check against the
built-in proposer types).

`propose` is called once per decoding step with CPU-side state::

    propose(
        sampled_token_ids: list[list[int]],   # accepted tokens from the previous step
        num_tokens_no_spec: np.ndarray,       # (num_reqs,) valid length per request
        token_ids_cpu: np.ndarray,            # (max_num_reqs, max_model_len) token ids
        slot_mappings=None,                   # unused by CPU proposers
    ) -> list[list[int]]

Returns one draft-token list per request. Lists may be empty (skip speculation for
that request) and may be shorter than `num_speculative_tokens`; the framework books
`len(draft_token_ids)` directly (`gpu_model_runner.py:2148`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["BaseProposer", "EmptyProposer", "RepeatProposer", "SparseContextDraftProposer"]


class BaseProposer:
    """Shared plumbing: config access, request filtering, call accounting."""

    def __init__(self, vllm_config: Any) -> None:
        self.vllm_config = vllm_config
        spec = vllm_config.speculative_config
        assert spec is not None
        self.k = spec.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        # Self-observability: lets us prove the proposer was actually invoked and
        # how often it declined to draft.
        self.num_calls = 0
        self.num_requests = 0
        self.num_declined = 0

    def eligible(self, i: int, sampled_token_ids: list[list[int]],
                 num_tokens_no_spec: np.ndarray) -> bool:
        """Mirror the two guards NgramProposer applies.

        1. A request whose previous step produced nothing accepted is skipped.
        2. A request already at max_model_len cannot take more tokens.
        """
        if not len(sampled_token_ids[i]):
            return False
        if int(num_tokens_no_spec[i]) >= self.max_model_len:
            return False
        return True

    def context(self, i: int, num_tokens_no_spec: np.ndarray,
                token_ids_cpu: np.ndarray) -> np.ndarray:
        """The request's real token ids. MUST slice by num_tokens_no_spec: the
        rest of the row is padding. NOTE: `token_ids_cpu` is a numpy *view* of a
        torch tensor -- never write to it."""
        n = int(num_tokens_no_spec[i])
        return token_ids_cpu[i, :n]

    def propose(self, sampled_token_ids: list[list[int]],
                num_tokens_no_spec: np.ndarray,
                token_ids_cpu: np.ndarray,
                slot_mappings: Any = None) -> list[list[int]]:
        raise NotImplementedError


class EmptyProposer(BaseProposer):
    """Phase 3 dummy: never drafts anything.

    Purpose is *not* speed. It proves the custom_class plumbing connects end to end
    and that with zero speculative tokens the engine reproduces the eager baseline
    bit-for-bit.
    """

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu,
                slot_mappings=None):
        self.num_calls += 1
        self.num_requests += len(sampled_token_ids)
        self.num_declined += len(sampled_token_ids)
        return [[] for _ in sampled_token_ids]


class RepeatProposer(BaseProposer):
    """Phase 3 dummy #2: draft the last accepted token, repeated k times.

    This is deliberately a *bad* predictor. Its job is to exercise the verification
    path with a non-empty draft -- proving drafts really reach the target model and
    that acceptance is counted -- while keeping correctness (greedy output must stay
    identical, since the target model still decides every token).
    """

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu,
                slot_mappings=None):
        self.num_calls += 1
        self.num_requests += len(sampled_token_ids)
        drafts: list[list[int]] = []
        for i, sampled in enumerate(sampled_token_ids):
            if not self.eligible(i, sampled_token_ids, num_tokens_no_spec):
                drafts.append([])
                self.num_declined += 1
                continue
            last = int(sampled[-1])
            drafts.append([last] * self.k)
        return drafts


class SparseContextDraftProposer(BaseProposer):
    """Token-level analogue of STD's sparse draft.

    WHY THIS IS NOT STD's MECHANISM -- read before interpreting any number
    ---------------------------------------------------------------------
    STD (Sparse-to-Dense) drafts from a *sparse KV view*: it takes the prefill
    attention tensors `[batch, q_heads, text_len, prompt_len]`, sums them per
    KV head, and keeps the top-K **visual** positions per layer; the draft step
    then attends only to (non-visual positions + those top-K visual positions).
    See `src/std_repro/std_qwen25vl.py::build_sparse_selection`.

    vLLM v0.22.0's `custom_class` proposer cannot express that. It is constructed
    with `VllmConfig` alone and called with CPU token ids only:

        propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings)

    There is no handle to the model, the KV cache, the attention groups or the
    prefill attention weights -- those live on `GPUModelRunner` and are never
    passed (`vllm/v1/worker/gpu_model_runner.py:4778`). v0.22.0 also has no
    GPU-speculator registry to hook instead. So the *sparse-KV* half of STD is
    unreachable from this interface.

    What IS reachable is STD's other stated principle -- **draft from a sparse
    view of the context, verify densely** -- at the *token* level. This proposer
    implements that: it matches the trailing n-gram against a deliberately
    *sparse* context window and copies what followed. `STD_DRAFT_WINDOW` is the
    sparsity knob (how many recent tokens the draft is allowed to see), so a
    window sweep measures whether restricting the draft's view helps or hurts.
    Compare with `NgramProposer`, which looks at the whole context.

    Config comes from the environment because `speculative_config` has no field
    for proposer-specific parameters (unknown keys are rejected by vLLM's config).

        STD_DRAFT_WINDOW  how many trailing context tokens the draft may read.
                          0 or negative = full context (no sparsity).
        STD_DRAFT_NGRAM   suffix match length (default 3).
    """

    def __init__(self, vllm_config) -> None:
        super().__init__(vllm_config)
        self.window = int(os.environ.get("STD_DRAFT_WINDOW", "0"))
        self.ngram = int(os.environ.get("STD_DRAFT_NGRAM", "3"))
        # Observability for the write-up.
        self.num_matched = 0
        self.num_missed = 0
        import atexit
        atexit.register(self.dump_stats)

    def _draft_one(self, ctx: "np.ndarray") -> list[int]:
        n = int(ctx.size)
        if n < self.ngram + 1:
            return []
        # The sparse view: only the last `window` tokens are searchable.
        lo = 0
        if self.window > 0:
            lo = max(0, n - self.window)
        # Never match the pattern against itself: the last valid start is n-ngram-1.
        pat = ctx[n - self.ngram:n]
        last = n - self.ngram - 1
        for start in range(last, lo - 1, -1):
            if last < lo:
                break
            if np.array_equal(ctx[start:start + self.ngram], pat):
                out = ctx[start + self.ngram:start + self.ngram + self.k]
                if out.size:
                    self.num_matched += 1
                    return [int(t) for t in out]
        self.num_missed += 1
        return []

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu,
                slot_mappings=None):
        self.num_calls += 1
        self.num_requests += len(sampled_token_ids)
        drafts: list[list[int]] = []
        for i, sampled in enumerate(sampled_token_ids):
            if not self.eligible(i, sampled_token_ids, num_tokens_no_spec):
                drafts.append([])
                self.num_declined += 1
                continue
            drafts.append(self._draft_one(
                self.context(i, num_tokens_no_spec, token_ids_cpu)))
        # Dump periodically, NOT via atexit: vLLM terminates the EngineCore process
        # with SIGTERM, so atexit handlers never run. A periodic (cheap) write is
        # the only way proposer-side counters survive.
        if self.num_calls % 25 == 0:
            self.dump_stats()
        return drafts

    def dump_stats(self) -> None:
        """Write proposer-side counters where the benchmark can find them.

        vLLM only reports drafts it *counted*; we also need to know whether the
        proposer found a match at all, otherwise "0% acceptance" is ambiguous
        between "emitted good drafts that were rejected" and "emitted nothing".
        """
        import json
        path = os.environ.get("STD_PROPOSER_STATS")
        if not path:
            return
        try:
            Path(path).write_text(json.dumps({
                "proposer": type(self).__name__,
                "window": self.window, "ngram": self.ngram, "k": self.k,
                "num_calls": self.num_calls,
                "num_requests": self.num_requests,
                "num_declined": self.num_declined,
                "num_matched": self.num_matched,
                "num_missed": self.num_missed,
            }, indent=2))
        except Exception:
            pass
