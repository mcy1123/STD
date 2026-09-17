"""Unit tests for the trace-collector changes needed by the L1 mechanism study.

Covers:
  * `AttentionTraceCollector(record_per_query=...)` end-to-end contract through
    its real `_hook`: verification rounds may keep the query axis, the prefill
    trace must stay summed (its q_len is the whole prompt);
  * `collect_traces.iter_videomme_samples` delegating to the benchmark's seed-42
    video iterator, and the new CLI surface.

`collect_traces` pulls in `benchmark_std` (imports `av`, absent here) and
`std_repro.std_qwen25vl` (evaluates PEP 585 annotations, unavailable on the
local Python 3.8), so both are stubbed before import.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts" / "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class _Cfg:
    num_key_value_heads = 2


class _Model:
    config = _Cfg()


NUM_HEADS = 4
KV_HEADS = 2
Q_LEN = 5
VISUAL_LEN = 6
KV_LEN = 8
HEAD_DIM = 4


def _collector(record_per_query: bool):
    from attention_trace import AttentionTraceCollector

    return AttentionTraceCollector(
        _Model(), torch.arange(VISUAL_LEN), record_per_query=record_per_query
    )


def _run_hook(collector, label, layers: int = 2) -> None:
    """Drive the real `_hook` for `layers` synthetic attention layers."""
    collector._active = True
    collector._label = label
    collector._query_len = Q_LEN
    for layer in range(layers):
        # Deterministic but non-degenerate values so a wrong reduction shows up.
        q = torch.full((1, NUM_HEADS, Q_LEN, HEAD_DIM), 0.5 + 0.1 * layer)
        k = torch.full((1, KV_HEADS, KV_LEN, HEAD_DIM), 0.25 + 0.1 * layer)
        collector._hook(layer, q, k)


class TestPerQueryContract:
    def test_per_query_round_keeps_query_axis(self):
        collector = _collector(record_per_query=True)
        _run_hook(collector, label=0)
        collector._end(target="round")

        assert len(collector.rounds) == 1
        trace = collector.rounds[0]
        assert trace.per_query is True
        assert trace.query_len == Q_LEN
        assert trace.visual_scores.shape == (KV_HEADS, Q_LEN, VISUAL_LEN)
        assert torch.isfinite(trace.visual_scores).all()

    def test_prefill_stays_summed_even_when_recording_per_query(self):
        collector = _collector(record_per_query=True)
        _run_hook(collector, label="prefill", layers=1)
        collector._end(target="prefill")

        assert collector.rounds == []
        assert collector.prefill_scores.shape == (KV_HEADS, VISUAL_LEN)

    def test_default_mode_sums_over_queries(self):
        collector = _collector(record_per_query=False)
        _run_hook(collector, label=0, layers=1)
        collector._end(target="round")

        trace = collector.rounds[0]
        assert trace.per_query is False
        assert trace.visual_scores.shape == (KV_HEADS, VISUAL_LEN)

    def test_summed_mass_equals_per_query_mass(self):
        """The summed trace must be exactly the per-query trace summed over q."""
        per_query = _collector(record_per_query=True)
        _run_hook(per_query, label=0, layers=2)
        per_query._end(target="round")

        summed = _collector(record_per_query=False)
        _run_hook(summed, label=0, layers=2)
        summed._end(target="round")

        assert torch.allclose(
            per_query.rounds[0].visual_scores.sum(dim=1), summed.rounds[0].visual_scores
        )

    def test_rounds_with_different_query_lengths_are_independent(self):
        collector = _collector(record_per_query=True)
        _run_hook(collector, label=0)
        collector._end(target="round")

        collector._active = True
        collector._label = 1
        collector._query_len = Q_LEN + 3
        q = torch.full((1, NUM_HEADS, Q_LEN + 3, HEAD_DIM), 0.5)
        k = torch.full((1, KV_HEADS, KV_LEN, HEAD_DIM), 0.25)
        collector._hook(0, q, k)
        collector._end(target="round")

        assert [r.visual_scores.shape for r in collector.rounds] == [
            (KV_HEADS, Q_LEN, VISUAL_LEN),
            (KV_HEADS, Q_LEN + 3, VISUAL_LEN),
        ]
        # Stacking is impossible by design; collect_traces stores a list instead.
        with pytest.raises(RuntimeError):
            torch.stack([r.visual_scores for r in collector.rounds])


class TestVideoMmeIterator:
    @staticmethod
    def _import_collect_traces(monkeypatch):
        """Import collect_traces with its heavy dependencies stubbed out."""
        bench = types.ModuleType("benchmark_std")
        bench.VIDEO_TOKEN_ID = 151656
        bench.load_qwen_model = lambda *a, **k: (None, None)
        bench.make_qwen_video_inputs = lambda *a, **k: {}
        calls = []

        def fake_iter(dataset_name_or_path, split, limit, video_root, prompt_style="direct"):
            calls.append(
                {
                    "dataset_name_or_path": dataset_name_or_path,
                    "split": split,
                    "limit": limit,
                    "video_root": video_root,
                    "prompt_style": prompt_style,
                }
            )
            yield {
                "sample_id": "s1",
                "video_path": "/tmp/s1.mp4",
                "question": "q",
                "duration": None,
            }

        bench.iter_generic_hf_video = fake_iter
        monkeypatch.setitem(sys.modules, "benchmark_std", bench)

        std = types.ModuleType("std_repro.std_qwen25vl")
        std.std_generate_qwen25vl = lambda *a, **k: None
        std.set_trace_collector = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, "std_repro.std_qwen25vl", std)

        # collect_traces installs the STD streaming reader (also an `av` user).
        stream = types.ModuleType("std_repro.streaming_video")
        stream.install_streaming_video_reader = lambda: None
        monkeypatch.setitem(sys.modules, "std_repro.streaming_video", stream)

        import std_repro

        monkeypatch.setattr(std_repro, "std_qwen25vl", std, raising=False)
        monkeypatch.setattr(std_repro, "streaming_video", stream, raising=False)

        if "collect_traces" in sys.modules:
            return importlib.reload(sys.modules["collect_traces"]), calls
        import collect_traces

        return collect_traces, calls

    def test_delegates_to_benchmark_iterator_with_cot_style(self, monkeypatch):
        collect_traces, calls = self._import_collect_traces(monkeypatch)

        samples = list(
            collect_traces.iter_videomme_samples(
                "/data/Video-MME", "/data/Video-MME/videos", 20, prompt_style="cot"
            )
        )

        assert len(samples) == 1
        assert samples[0]["sample_id"] == "s1"
        assert calls == [
            {
                "dataset_name_or_path": "/data/Video-MME",
                "split": "test",
                "limit": 20,
                "video_root": "/data/Video-MME/videos",
                "prompt_style": "cot",
            }
        ]

    def test_cli_exposes_video_mme_and_per_query(self, monkeypatch):
        collect_traces, _ = self._import_collect_traces(monkeypatch)

        args = collect_traces.build_parser().parse_args(
            [
                "--dataset",
                "Video-MME",
                "--data-path",
                "/d",
                "--video-root",
                "/d/videos",
                "--record-per-query",
                "--limit",
                "20",
            ]
        )

        assert args.dataset == "Video-MME"
        assert args.data_path == "/d"
        assert args.video_root == "/d/videos"
        assert args.record_per_query is True
        # Defaults must stay compatible with the existing VDC/MLVU callers.
        assert args.data_dir == "/mnt/local2/mcy/datasets/VideoDetailCaption"
        assert args.prompt_style == "cot"

    def test_legacy_vdc_invocation_still_parses(self, monkeypatch):
        collect_traces, _ = self._import_collect_traces(monkeypatch)

        args = collect_traces.build_parser().parse_args(
            ["--dataset", "VideoDetailCaption", "--data-dir", "/vdc", "--limit", "10"]
        )

        assert args.dataset == "VideoDetailCaption"
        assert args.data_dir == "/vdc"
        assert args.record_per_query is False
