from __future__ import annotations

import numpy as np
import pytest
import torch

av = pytest.importorskip("av")
vision_process = pytest.importorskip("qwen_vl_utils.vision_process")

from std_repro.streaming_video import (
    BACKEND_NAME,
    install_streaming_video_reader,
    read_video_streaming,
)


@pytest.fixture
def tiny_video(tmp_path):
    path = tmp_path / "tiny video.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for index in range(16):
            pixels = np.full((24, 32, 3), index * 13, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


@pytest.mark.parametrize("sampling", [{"nframes": 6}, {"fps": 3.0}])
def test_streaming_matches_existing_torchvision_sampling(tiny_video, sampling):
    # benchmark_std uses an unescaped local ``file://`` URI, which is also
    # what torchvision's legacy PyAV wrapper accepts.
    ele = {"video": f"file://{tiny_video}", **sampling}
    expected, expected_fps = vision_process._read_video_torchvision(ele)
    actual, actual_fps = read_video_streaming(ele)
    assert actual.dtype == torch.uint8
    assert torch.equal(actual, expected)
    assert actual_fps == pytest.approx(expected_fps)


def test_missing_frame_metadata_uses_counting_pass(tmp_path):
    path = tmp_path / "unknown-count.mkv"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("ffv1", rate=8)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for index in range(12):
            frame = av.VideoFrame.from_ndarray(
                np.full((24, 32, 3), index * 15, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    with av.open(str(path)) as container:
        assert container.streams.video[0].frames == 0
    ele = {"video": str(path), "nframes": 4}
    expected, expected_fps = vision_process._read_video_torchvision(ele)
    actual, actual_fps = read_video_streaming(ele)
    assert torch.equal(actual, expected)
    assert actual_fps == pytest.approx(expected_fps)


def test_selected_frame_budget_fails_before_rgb_materialization(tiny_video, monkeypatch):
    def must_not_allocate(*args, **kwargs):
        raise AssertionError("No output tensor should be allocated above the budget")

    monkeypatch.setattr(torch, "empty", must_not_allocate)
    with pytest.raises(MemoryError, match="Selected RGB frames"):
        read_video_streaming({"video": str(tiny_video), "nframes": 4}, max_selected_bytes=100)


def test_trimmed_video_is_rejected_explicitly(tiny_video):
    with pytest.raises(ValueError, match="whole videos"):
        read_video_streaming({"video": str(tiny_video), "video_start": 0.5})


def test_installer_forces_streaming_and_blocks_torchvision_fallback(monkeypatch):
    monkeypatch.setattr(vision_process, "VIDEO_READER_BACKENDS", dict(vision_process.VIDEO_READER_BACKENDS))
    monkeypatch.setattr(vision_process, "FORCE_QWENVL_VIDEO_READER", None)
    monkeypatch.setenv("FORCE_QWENVL_VIDEO_READER", "torchvision")
    install_streaming_video_reader()
    try:
        assert vision_process.get_video_reader_backend() == BACKEND_NAME
        with pytest.raises(RuntimeError, match="fallback is disabled"):
            vision_process.fetch_video({"video": "/does/not/exist.mp4"})
    finally:
        vision_process.get_video_reader_backend.cache_clear()
