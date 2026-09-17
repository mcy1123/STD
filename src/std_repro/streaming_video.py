"""Memory-bounded full-video sampling for the qwen-vl-utils video pipeline.

The standard torchvision backend materializes every decoded frame before
sampling. This backend keeps only qwen's uniformly selected RGB frames. The
normal qwen resize and processor still run afterward, unchanged.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

import av
import torch


DEFAULT_MAX_SELECTED_BYTES = 1024**3
BACKEND_NAME = "std_streaming_pyav"


def _local_video_path(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("The streaming video reader requires a local video path.")
    if value.startswith("file://"):
        parsed = urlparse(value)
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError("Remote file:// hosts are not supported by the streaming reader.")
        value = unquote(parsed.path)
    elif "://" in value:
        raise ValueError("The streaming video reader only supports local files.")
    if not Path(value).is_file():
        raise FileNotFoundError(value)
    return value


def _video_stream(container):
    if not container.streams.video:
        raise ValueError("The input file has no video stream.")
    stream = container.streams.video[0]
    # Keep FFmpeg's in-flight frame buffers bounded as well as our RGB output.
    stream.thread_count = 1
    return stream


def _count_frames(path: str) -> int:
    with av.open(path) as container:
        stream = _video_stream(container)
        return sum(1 for _ in container.decode(stream))


def _sample_frames(path: str, total_frames: int, nframes: int, max_bytes: int):
    indices = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    next_selected = 0
    selected = None
    frame_count = 0
    with av.open(path) as container:
        stream = _video_stream(container)
        for index, frame in enumerate(container.decode(stream)):
            frame_count += 1
            if next_selected >= nframes or index != indices[next_selected]:
                continue
            shape = (nframes, 3, frame.height, frame.width)
            selected_bytes = math.prod(shape)
            if selected_bytes > max_bytes:
                raise MemoryError(
                    f"Selected RGB frames need {selected_bytes / 1024**3:.2f} GiB, "
                    f"above the streaming reader limit of {max_bytes / 1024**3:.2f} GiB; "
                    "reduce the requested frame count or use lower-resolution source videos."
                )
            if selected is None:
                selected = torch.empty(shape, dtype=torch.uint8)
            elif tuple(selected.shape) != shape:
                raise ValueError("Video dimensions change within the selected frames.")
            rgb = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
            # The loop also handles duplicate indices if sampling rules change.
            while next_selected < nframes and indices[next_selected] == index:
                selected[next_selected].copy_(rgb)
                next_selected += 1
    return selected, frame_count, next_selected


def read_video_streaming(
    ele: dict, *, max_selected_bytes: int = DEFAULT_MAX_SELECTED_BYTES
) -> tuple[torch.Tensor, float]:
    """Return qwen-compatible ``(T,C,H,W)`` uint8 RGB and effective FPS.

    Only whole videos are accepted. Frame indices and sampled FPS match the
    torchvision qwen backend for normally decoded videos. Positive container
    frame counts are used initially, then checked against the actual decoded
    count; absent or inaccurate metadata triggers an additional bounded pass.
    """
    from qwen_vl_utils.vision_process import smart_nframes

    if ele.get("video_start", 0.0) != 0.0 or ele.get("video_end") is not None:
        raise ValueError("The streaming reader currently supports whole videos only.")
    if max_selected_bytes <= 0:
        raise ValueError("max_selected_bytes must be positive.")
    path = _local_video_path(ele["video"])
    with av.open(path) as container:
        stream = _video_stream(container)
        total_frames = int(stream.frames or 0)
        video_fps = float(stream.average_rate or 0)
    if not math.isfinite(video_fps) or video_fps <= 0:
        raise ValueError("Video metadata does not provide a positive average frame rate.")
    if total_frames <= 0:
        total_frames = _count_frames(path)
    if total_frames <= 0:
        raise ValueError("The video contains no decodable frames.")

    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    video, actual_frames, selected_count = _sample_frames(
        path, total_frames, nframes, max_selected_bytes
    )
    if actual_frames != total_frames:
        # Discard the old selection before allocating another. No full-video
        # tensor is retained, even when container metadata is inaccurate.
        del video
        total_frames = actual_frames
        if total_frames <= 0:
            raise ValueError("The video contains no decodable frames.")
        nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        video, checked_frames, selected_count = _sample_frames(
            path, total_frames, nframes, max_selected_bytes
        )
        if checked_frames != total_frames:
            raise RuntimeError("Video frame count changed between decoding passes.")
    if video is None or selected_count != nframes:
        raise RuntimeError(f"Decoded {selected_count} selected frames, expected {nframes}.")
    return video, nframes / total_frames * video_fps


def _disabled_torchvision_fallback(_ele: dict):
    raise RuntimeError(
        "Streaming video decoding failed. Full-video torchvision fallback is disabled "
        "to prevent unbounded host-memory use; inspect the preceding reader error."
    )


def install_streaming_video_reader() -> None:
    """Opt this process into bounded decoding, including on reader failure.

    qwen-vl-utils 0.0.10 catches backend exceptions and calls its registered
    torchvision reader. Replace that fallback with an explicit error so memory
    limits and malformed inputs cannot silently enable full-file loading.
    """
    from qwen_vl_utils import vision_process

    vision_process.VIDEO_READER_BACKENDS[BACKEND_NAME] = read_video_streaming
    vision_process.VIDEO_READER_BACKENDS["torchvision"] = _disabled_torchvision_fallback
    vision_process.FORCE_QWENVL_VIDEO_READER = BACKEND_NAME
    os.environ["FORCE_QWENVL_VIDEO_READER"] = BACKEND_NAME
    vision_process.get_video_reader_backend.cache_clear()
