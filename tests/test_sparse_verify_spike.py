"""Tests for the experimental batched sparse-verifier attention spike.

These tests intentionally exercise the attention primitive directly.  The full
Qwen model is not needed to establish the causal-mask invariant, and keeping
the fixture small makes the test runnable on CPU as well as A100.
"""

import importlib
import sys
import unittest

import torch


try:
    _spike = importlib.import_module("std_repro.sparse_verify_spike")
    _import_error = None
except Exception as exc:  # Keep the RED test as an assertion, not collection error.
    _spike = None
    _import_error = exc


@unittest.skipIf(
    sys.version_info < (3, 9) and _import_error is not None,
    "Qwen2.5-VL fork requires Python >= 3.9 (unsupported interpreter)",
)
class SparseVerifySpikeTests(unittest.TestCase):
    def require_spike(self):
        self.assertIsNotNone(
            _spike,
            "experimental sparse verifier module is missing or cannot import: %r" % (_import_error,),
        )

    def test_offset_mask_is_causal_with_a_compact_prefix_and_tail(self):
        self.require_spike()
        mask = _spike.offset_causal_mask(3, 7, torch.device("cpu"))
        self.assertEqual(tuple(mask.shape), (1, 1, 3, 7))
        # Three newly queried positions follow four cached (compact-prefix +
        # generated-tail) positions.  A row can see all four past positions
        # and itself, but not a future query key.
        self.assertTrue(torch.all(mask[0, 0, 0, :5] == 0))
        self.assertTrue(torch.isneginf(mask[0, 0, 0, 5:]).all())
        self.assertTrue(torch.all(mask[0, 0, 1, :6] == 0))
        self.assertTrue(torch.isneginf(mask[0, 0, 1, 6:]).all())
        self.assertTrue(torch.all(mask[0, 0, 2, :7] == 0))

    def test_offset_mask_allows_q_len_one_fast_path(self):
        self.require_spike()
        self.assertIsNone(_spike.offset_causal_mask(1, 4, torch.device("cpu")))

    def test_batched_attention_matches_sequential_causal_attention(self):
        self.require_spike()
        torch.manual_seed(7)
        q = torch.randn(1, 4, 3, 8)
        k = torch.randn(1, 2, 7, 8)
        v = torch.randn(1, 2, 7, 8)
        batched = _spike._offset_causal_gqa_attention(q, k, v)
        offset = k.shape[-2] - q.shape[-2]
        rows = []
        for i in range(q.shape[-2]):
            one = _spike._offset_causal_gqa_attention(
                q[:, :, i : i + 1], k[:, :, : offset + i + 1], v[:, :, : offset + i + 1]
            )
            rows.append(one)
        sequential = torch.cat(rows, dim=-2)
        self.assertTrue(torch.allclose(batched, sequential, atol=1e-5, rtol=1e-5))

    def test_batched_attention_has_no_future_leakage(self):
        self.require_spike()
        torch.manual_seed(11)
        q = torch.randn(1, 2, 2, 4)
        k = torch.randn(1, 1, 5, 4)
        v = torch.randn(1, 1, 5, 4)
        baseline = _spike._offset_causal_gqa_attention(q, k, v)
        changed = v.clone()
        changed[:, :, -1] += 10000
        altered = _spike._offset_causal_gqa_attention(q, k, changed)
        # kv_len-q_len == 3.  Key 4 is future for row 0 but current for row 1.
        self.assertTrue(torch.allclose(baseline[:, :, :1], altered[:, :, :1], atol=1e-5, rtol=1e-5))
        self.assertFalse(torch.allclose(baseline[:, :, 1:], altered[:, :, 1:], atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
