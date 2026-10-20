# 阶段 1：Qwen2.5-VL 纯视频推理跑通

> 状态：**✅ 完成（GO）**
> 日期：2026-09-26
> 环境：`vllm-std` / vLLM `0.22.0+cu129` / torch `2.11.0+cu129` / 主机 `A6000` GPU3

---

## 1.1 我做了什么

用 vLLM 加载 **Qwen2.5-VL-7B-Instruct**，喂**本地 mp4 视频**，greedy decode 生成文本。
这一步的产物是**后面所有阶段的 bit-exact 基准**：后续 dummy / STD / batch 的所有配置，都要和这里的 token ids 逐位对齐。

---

## 1.2 命令

```bash
source scripts/vllm-std/env.sh
GPU=$(scripts/vllm-std/pick_gpu.sh | awk '{print $1}')
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/vllm-std/01_baseline_video.py \
    --num-frames 32 --max-tokens 128 --runs 3 --warmup 1 \
    --gpu-mem-util 0.80 --tag baseline_f32
```

脚本：`scripts/vllm-std/01_baseline_video.py`
日志：`results/vllm-std/logs/01_baseline_f32.log`
原始记录：`results/vllm-std/01_baseline_video.jsonl`（只追加，不覆盖）

---

## 1.3 结果（硬数字）

| 项 | 值 |
|---|---|
| video | `datasets/Video-MME/videos/data/0ay2Qy3wBe8.mp4` |
| num_frames | 32 |
| **prompt tokens** | **19,166** |
| **output tokens** | **128** |
| **耗时（3 次）** | **5.1653 s ± 0.0530 s**（min/max 见 jsonl） |
| **bit-exact（3 次重复）** | ✅ **True** |
| finish_reason | `stop`（自然结束，非截断） |
| 模型显存 | 15.67 GiB |
| KV cache | 12.93 GiB / **242,112 tokens** |
| 引擎初始化 | 169.5 s（首次，含 torch.compile 23.2 s + CUDA graph 捕获 14 s）；缓存命中后 41–60 s |

### 输出可读性（原文）

> The video begins with a title card that reads "BROOKE'S SOLO" with the subtitle "DANCE STYLE: ACROBATIC." The scene transitions to a stage where a young girl, dressed in a colorful leotard, performs an acrobatic routine. The background is vibrant, featuring a large screen with dynamic, fiery visuals

画面内容（片头卡、字幕、舞台、小女孩、杂技动作）与描述一致 → **不是「看起来跑通」，是内容对得上**。

### 首 token ids（供阶段 3/4 对齐用）

```
[785, 2766, 12033, 448, 264, 2265, 3701, 429, 15804, 330, 33, 1285, 46, 3390, 13272, 5627]
```

---

## 1.4 frame sampling / fps / 分辨率 行为验证

Phase 1 要求「确认 frame sampling、fps、分辨率参数行为正常」。做了独立实验（`scripts/vllm-std/01b_frame_sampling.py`）：

### num_frames：**正常，线性**

| num_frames | prompt tokens |
|---:|---:|
| 4 | 2,420 |
| 16 | 9,596 |
| 32 | 19,164 |

约 **599 tokens/帧**，线性扩展 → `num_frames` 真实生效。

### fps：**有坑，被构造函数默认值静默封顶** ⚠

| 配置 | prompt tokens |
|---|---:|
| `{"fps": 1}` | 19,164 |
| `{"fps": 4}` | 19,164 ← **与 fps=1 完全相同** |
| `{"fps": 4, "num_frames": -1}` | **151,920** |

**根因**：`vllm/multimodal/media/video.py` 的 `VideoMediaIO.__init__` 签名是

```python
def __init__(self, image_io, *, num_frames: int = 32, **kwargs)
```

而运行期合并逻辑只在**两侧都来自 kwargs** 时才互斥：

```python
# video.py:32-39
if "num_frames" in runtime_kwargs and "fps" not in runtime_kwargs:
    merged.pop("fps", None)
elif "fps" in runtime_kwargs and "num_frames" not in runtime_kwargs:
    merged.pop("num_frames", None)
```

`num_frames=32` 是**形参默认值**，不在 `merged` 里，所以只传 `fps` 时**弹不掉它**，采样仍被 32 帧封顶。必须显式传 `num_frames=-1`（表示不封顶）才能让 fps 真正决定帧数：fps=4 时得到 151,920 tokens ≈ 253 帧。

> 这条对后面阶段有直接影响：**做帧数消融时必须显式写 `num_frames`，不要指望 `fps`**，否则会静默测到同一个 32 帧配置。

---

## 1.5 和 baseline 比，输出是否一致？

本阶段 baseline **就是**基准，所以比的是「重复运行之间是否一致」：
3 次 timed run + 1 次 probe + 1 次 warmup，**token ids 逐位相同**（`bit_exact_across_runs: true`）。

---

## 1.6 失败记录（未掩盖）

| # | 失败 | 根因 | 处置 |
|---|---|---|---|
| 1 | 引擎起不来：`#error "CUDA versions below 12 are not supported."`（flashinfer JIT） | `/usr/local/cuda` 指向 **nvcc 11.8**，flashinfer 首用时要 JIT 编译采样 kernel，其内置 CCCL 拒绝 CUDA<12 | `env.sh` 里把 `CUDA_HOME/PATH` 指向 **`/usr/local/cuda-12.6`** |
| 2 | `AssertionError: Expected code to be unreachable, but got: '/'` | 用 `llm.generate(multi_modal_data={"video": "<path>"})` 传**裸路径字符串**；`MultiModalDataParser._parse_video_data` 只接受已解码的 ndarray/tensor/PIL，遇到 str 会**逐字符**迭代 | 改用**chat 路径**：`llm.chat` + `{"type":"video_url","video_url":{"url":"file://..."}}`，并在 LLM 构造时给 `allowed_local_media_path` |
| 3 | `KeyError: 'num_output_tokens'` | 脚本自身 bug（顶层字段没赋值） | 修脚本 |
| 4 | `fps=1` 与 `fps=4` 结果完全相同 | 见 §1.4，构造函数默认 `num_frames=32` 封顶 | 记录为**平台行为**，后续实验显式指定 num_frames |
| 5 | `01b` 中第 2/3 个配置 `EngineCore failed to start`：`Free memory on device cuda:0 (9.73/47.53 GiB) is less than desired GPU memory utilization (0.8, 38.02 GiB)` | **别人在这期间抢占了 GPU3**（共享机） | 非代码 bug；脚本已记录 `gpu_before/after`，并在每次运行前用 `pick_gpu.sh` 重新选卡 |

---

## 1.7 是否进入下一阶段？

**是。** 依据：

1. 模型输出可读，且**内容与视频对得上**；
2. 3 次重复 **bit-exact**；
3. 帧采样参数行为已验证（且发现并定位了一个 fps 封顶陷阱）；
4. 耗时统计已具备 mean/std。

**→ GO，进入阶段 2（投机解码接口分析）。**
