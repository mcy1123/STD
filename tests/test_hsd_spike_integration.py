"""Tiny CPU integration checks for the experimental nested draft block."""
from __future__ import annotations

import importlib
import sys
import unittest
from types import SimpleNamespace

import torch


try:
    _hsd = importlib.import_module("std_repro.hsd_spike")
    _import_error = None
except Exception as exc:  # Turn an absent helper into a useful RED assertion.
    _hsd = None
    _import_error = exc


@unittest.skipIf(
    sys.version_info < (3, 9) and _import_error is not None,
    "Qwen2.5-VL fork requires Python >= 3.9 (unsupported interpreter)",
)
class HsdSpikeIntegrationTests(unittest.TestCase):
    @staticmethod
    def _tiny_lm(seed: int):
        torch.manual_seed(seed)
        from specvlm.models.configuration_qwen2_5_vl import Qwen2_5_VLConfig
        from specvlm.models.modeling_qwen2_5_vl import Qwen2_5_VLModel

        config = Qwen2_5_VLConfig(
            vocab_size=31, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=128,
            rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        )
        # This repository's preallocated KVCache is consumed by its SDPA
        # attention fork (the eager Transformers path expects DynamicCache).
        config._attn_implementation = "sdpa"

        class TinyLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = config
                self.model = Qwen2_5_VLModel(config)
                self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

            @property
            def dtype(self):
                return self.model.embed_tokens.weight.dtype

            def forward(self, **kwargs):
                output = self.model(**kwargs)
                return SimpleNamespace(logits=self.lm_head(output.last_hidden_state))

        return TinyLM().eval()

    @staticmethod
    def _cache(model, prompt):
        from specvlm.kv_cache.kv_cache import initialize_past_key_values

        past, _, lengths = initialize_past_key_values(model)
        output = model(input_ids=prompt, past_key_values=past)
        return past, lengths, torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)

    def test_nested_block_matches_sparse_autoregressive_tokens(self):
        self.assertIsNotNone(_hsd, "hsd_spike helper is missing: %r" % (_import_error,))
        if not hasattr(_hsd, "nested_draft_block"):
            self.fail("nested_draft_block helper is not implemented")
        from std_repro.std_qwen25vl import SparseSelection, SparseDraftController
        from specvlm.kv_cache.kv_cache import initialize_past_key_values

        target = self._tiny_lm(3)
        # Deliberately use a different draft seed so at least one inner block
        # exercises sparse rejection/correction and cache rollback.
        draft = self._tiny_lm(4)
        prompt = torch.tensor([[4, 7, 2, 9]], dtype=torch.long)
        # Every prompt position is retained as the compact visual prefix.  This
        # keeps the fixture independent of the multimodal processor while still
        # exercising the real KVCache and sparse controller.
        layers = target.config.num_hidden_layers
        topk = [torch.arange(prompt.shape[1]).view(1, -1) for _ in range(layers)]
        selection = SparseSelection(topk, torch.empty(0, dtype=torch.long), prompt.shape[1],
                                    prompt.shape[1], 1, prompt.shape[1])
        sparse_controller = SparseDraftController(target, selection, sparse_attn_mode="gqa_sdpa")
        sparse_controller.install()
        sparse_pkv, sparse_lengths, sparse_next = self._cache(target, prompt)
        draft_pkv, draft_lengths, draft_next = self._cache(draft, prompt)

        # Reference S-AR sequence from an independent cache.
        ref_pkv, ref_lengths, ref_next = self._cache(target, prompt)
        reference = []
        for index in range(5):
            reference.append(int(ref_next.item()))
            position = prompt.shape[1] + index
            ref_next = target(input_ids=ref_next,
                              past_key_values=ref_pkv,
                              position_ids=torch.tensor([[position]])).logits[:, -1].argmax(-1, keepdim=True)

        candidates, draft_next_out, sparse_next_out, stats = _hsd.nested_draft_block(
            target, draft, draft_pkv, draft_lengths, draft_next,
            sparse_pkv, sparse_lengths, sparse_next, sparse_controller,
            context_len=prompt.shape[1], block_len=5, inner_gamma=2,
        )
        self.assertEqual([int(x) for x in candidates], reference)
        self.assertEqual(len(candidates), 5)
        self.assertIn("sparse_accept_len", stats)
        self.assertGreaterEqual(int(stats["sparse_accept_len"]), 0)
        # The helper must leave both speculative caches at exactly the
        # committed block boundary; stale rejected suffixes would make the
        # next outer target verification observe the wrong context.
        self.assertEqual(int(draft_lengths[0].item()), prompt.shape[1] + len(candidates))
        self.assertEqual(int(sparse_lengths[0].item()), prompt.shape[1] + len(candidates))

        # A second block starts after the first block's accepted prefix and
        # correction.  Comparing it with the continued S-AR stream catches a
        # rollback that leaves one rejected KV (or bonus token) in either cache.
        reference_next = []
        for index in range(3):
            reference_next.append(int(ref_next.item()))
            position = prompt.shape[1] + 5 + index
            ref_next = target(input_ids=ref_next, past_key_values=ref_pkv,
                              position_ids=torch.tensor([[position]])).logits[:, -1].argmax(-1, keepdim=True)
        next_candidates, _, _, _ = _hsd.nested_draft_block(
            target, draft, draft_pkv, draft_lengths, draft_next_out,
            sparse_pkv, sparse_lengths, sparse_next_out, sparse_controller,
            context_len=prompt.shape[1] + 5, block_len=3, inner_gamma=2,
        )
        self.assertEqual([int(x) for x in next_candidates], reference_next)


if __name__ == "__main__":
    unittest.main()
