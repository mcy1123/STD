# EAGLE-3 video probe — why the local draft is not runnable out of the box

> 2026-09-23. Attempted: run the **pretrained** EAGLE-3 draft
> (`/home/mcy/local2/models/qwen2.5-vl-7b-eagle3-sgl`, 1 layer, 1.8 GB) with
> **sglang 0.5.2rc2** against Qwen2.5-VL-7B on Video-MME clips, to measure accept
> length at 64 vs 128 frames. This is the cheapest training-free route to a
> video speculative-decoding number, because the draft was trained by someone
> else (SpecForge, 30k ALLaVA-4V samples).
>
> **Result: not runnable as-is.** Six launch attempts, four distinct blockers.
> No accept-length number was obtained. The blockers themselves are the finding.

## 1. The measured blockers

| # | launch config | where it died |
|---|---|---|
| 1 | README command (`--cuda-graph-max-bs 1`) | CUDA-graph capture: dynamo fake-tensor `view` error, mrope |
| 2 | `--disable-cuda-graph` | after the video prefill: verify gets **0 tokens** → `view(0, -1, 128)` |
| 3 | default CUDA graphs | graph capture bs=8: non-contiguous `view` in mrope |
| 4 | + `view`→`reshape` patch | capture now **passes** (all 8 batch sizes); then `mrope_positions[:, :1].copy_(...24...)` broadcast error |
| 5 | patch + `--disable-cuda-graph` | back to `reshape(0, -1, 128)` — so the zero-token verify is a **logic** bug, not a contiguity bug |
| 6 | — | stopped; sglang source restored from backup |

The patch in #4 was applied to a backup-and-restore sandbox only: one line,
`query/key.view(...)` → `.reshape(...)` in `rotary_embedding.py`, reverted
immediately afterwards (verified back to 2 `view`, 0 `reshape`). No user job was
touched, and no model file was modified.

## 2. Two structural findings that do **not** depend on the crashes

### 2.1 The draft's context window is 8,192 tokens — sglang refuses the pair

```
ValueError: Warning: Target model's context_length (128000) is greater than the
derived context_length (8192). ... To allow overriding this maximum, set the env
var SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
```

| | tokens | vs draft's 8,192 |
|---|---:|---|
| 128-frame Video-MME prompt | **15,948** | **1.95× over** |
| 64-frame Video-MME prompt | **7,884** | 0.96× — just fits |

`Qwen2.5-VL-7B/32B-Instruct` declare `max_position_embeddings = 128000`; the draft
declares 8,192. The env var removes the guard, not the capability gap: RoPE beyond
the trained range degrades acceptance, and measuring exactly that degradation was
the point of the 64-vs-128 experiment. **So the experiment is still worth running
— it just has to be run knowingly.**

### 2.2 The local-file video path is broken in this sglang build

`sglang/srt/multimodal/processors/base_processor.py`:

* `estimate_frames` (L276) **does** understand a `video:` prefix:
  `if isinstance(image, str) and image.startswith("video:")`
* `_load_single_item` (L309) then passes the string **verbatim** to
  `load_video`, which (`srt/utils.py:741`) only handles `bytes`, `http(s)://`,
  `data:` and bare existing paths — anything else falls through to
  `pybase64.b64decode`.

Consequences, both observed:

| what the client sends | what happens |
|---|---|
| bare path | loads, but `estimate_frames` counts it as **1** frame → *"More image data items provided than corresponding tokens found in the prompt"* → the 0-token verify crash |
| `video:/abs/path` | `Incorrect padding` (base64 decode of a path string) |

**The supported video inputs for Qwen2.5-VL in this build are base64
`data:video/mp4;base64,...` or an `http(s)://` URL.** Local paths are only proven
for images.

## 3. Next attempts, in order of cost

1. **Serve the clip over HTTP** — `python -m http.server` in the clip directory,
   pass `http://127.0.0.1:PORT/clip.mp4`. `load_video` handles http and the frame
   counting takes the documented path. Cheapest next try.
2. **base64 `data:` URL** — a 64-frame 2 fps re-encode of a 1-minute source is
   under 1 MB, so this is practical; `load_video` handles `data:`.
3. **Cap the frames at 64** (7,884 tokens) until the draft's context limit is
   addressed; 128 frames needs a draft trained or extended past 8k.
4. If EAGLE-3 in sglang keeps failing, fall back to the **32B-target / 7B-draft**
   route (ParallelVLM's own pairing; both models are already local). It needs no
   EAGLE machinery — only tensor-parallel sharding of the 32B and a batch-capable
   runner. Predicted `c ≈ 1.95` after TP sharding → ≈1.45× (see PROGRESS §20).

## 4. Artifacts

* `scripts/analysis/eagle3_video_bench.py` — the client, ready to run: slices N
  frames with PyAV (no ffmpeg on this box) and posts
  `{"type": "video_url", "video_url": {"url": ...}}` to
  `/v1/chat/completions`, reporting latency, completion tokens and
  `spec_accept_length`.
* Fix list for a real run: `--disable-cuda-graph` is required at this sglang
  revision (or the one-line `reshape` patch plus `--cuda-graph-max-bs >= 24`);
  `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` is required for any prompt over
  8,192 tokens.
