# 阶段 2.1：路线 B 最小验证（2.1a–2.1e）

> 状态：**⛔ 2.1a 未完成 —— 被机器资源阻塞，非技术失败**
> 日期：2026-09-26
> 前置：`22-route-decision.md`（阶段 2.0，走路线 B）
> 环境：vLLM `0.22.0+cu129` / 驱动 550.163.01 / 主机 `A6000`

---

## 2.1a：钩子捕获 `pixel_values_videos`

### 这一步验证什么

路线 B 需要 HF 重跑视觉前缀，那就必须拿到 `pixel_values_videos` /
`video_grid_thw` / `second_per_grid_ts`。它们**不在 `propose()` 的参数里**，只能靠钩子从
vLLM 的模型 forward 上截获。

**钩点**（读源码确认，非猜测）：`vllm/model_executor/models/qwen2_5_vl.py:1927`

```python
    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs):     # ← 视频张量在 **kwargs 里
```

脚本：`scripts/vllm-std/P21a_capture_video_inputs.py`
（钩 `Qwen2_5_VLForConditionalGeneration.forward`；只记录张量引用与 `torch.Size`，
**无 `.cpu()` / `.tolist()` / host sync**，满足 trace-safe 前提）

### 结果：**5 次尝试全部失败，全部死在 GPU 显存，与钩子无关**

| # | GPU | 配置 | 失败点 |
|---|---|---|---|
| 1 | 7 (A6000) | 4 帧 / util 0.34 | 启动时该卡仅剩 15.54 GB（需 16.16） |
| 2 | 4 (4090) | 4 帧 / util 0.65 | 模型已加载（15.25 GB），**视觉编码器再要 1.02 GB 时 OOM**（仅剩 809 MB） |
| 3 | 7 (A6000) | 4 帧 / 自动 util | 探测到 37,257 MiB → 引擎启动时只剩 26.98 GB |
| 4 | 2 (A6000) | 4 帧 / util 0.36 | 探测到 23,792 MiB → **6 秒内掉到 6.28 GB** |
| 5 | 等待 ≥30 GB 窗口 | 4 帧 | 等了 10 分钟，最好只有 16,743 MiB，**始终没出现窗口** |

**逐字错误**（第 2 次，最有信息量的一次）：

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 1.02 GiB. GPU 0 has a total
capacity of 23.64 GiB of which 809.12 MiB is free. ... this process has 15.25 GiB
memory in use.
```

**即：模型权重加载成功，卡在视觉编码器。**

### 根因：不是设计问题，是本机此刻没有资源

- 7B bf16 权重本身 **15.67 GB**（阶段 0 实测）。
- 加视觉编码器 activations（~1 GB）+ KV，**vLLM 单侧最低约 17.5 GB**。
- 本机当前最好的一张卡只剩 **16.7 GB**（A6000）/ **16.4 GB**（4090），**全部 A6000 利用率 100%**。
- 波动极端：实测到 **23.8 GB → 6.28 GB / 6 秒**、**37.3 GB → 27.0 GB / 数秒**。

### 顺带产物：`scripts/vllm-std/wait_and_run.sh`

为解决「探测到窗口，等真正跑起来窗口已经关了」的问题而写：
**在同一次 shell 调用里完成探测与启动**，无间隙；并要求调用方给一个**固定的保守显存预算**
（而不是按探测峰值缩放 —— 按峰值缩放在第 3 次尝试时请求了 33.5 GiB，而卡已被邻居压到 27 GiB）。

后续只要卡有空闲，这条命令可直接复用：

```bash
bash scripts/vllm-std/wait_and_run.sh 30000 3600 \
    P21a_capture_video_inputs.py --num-frames 32 --max-model-len 32768
```

---

## 2.1b/c/d/e：**未开始**

按硬性规则 1「前一步不通过，不许跳下一步」，2.1a 未通过 → 2.1b–2.1e **一律未开始**。

**且 2.1b 的门槛比 2.1a 更高**：它与 2.1a 的区别是**同时**驻留 vLLM(15.67 GB) + HF(15.5 GB)
≈ **31.2 GB 固定占用**，再加两侧 KV 与 activation，峰值 ≈ **37 GB**（阶段 2.0 §2.0.2 的算术估算）。
**本机当前没有任何一张卡有 37 GB 空闲**（最大 16.7 GB）。

> 换句话说：**即使现在给一张完全空闲的 48 GB A6000，2.1a/2.1b 才有资格开始跑**；
> 而当前状态下连 2.1a 都进不去。

---

## 2.1 阶段验收

| 步骤 | 状态 | 判据 | 结论 |
|---|---|---|---|
| 2.1a 捕获 video inputs | ⛔ **未完成** | 拿到 `pixel_values_videos` | **无数据**（5 次尝试全因显存失败） |
| 2.1b vLLM+HF 共存 | ⛔ 未开始 | 共存且余量 ≥5 GB | — |
| 2.1c 输入对齐断言 | ⛔ 未开始 | `len(hf_input_ids)==num_prompt_tokens` | — |
| 2.1d proposer 骨架 | ⛔ 未开始 | 空 proposer 跑通 | — |
| 2.1e 单 token 接受率 | ⛔ 未开始 | 接受率 > 0 | — |

### ⚠️ 必须避免的错误结论

- ❌ **不能**说「路线 B 不可行」—— 失败是**资源可用性**，不是设计缺陷。钩点、
  可复用函数、显存算术（阶段 2.0）都还没被实测挑战过。
- ❌ **不能**说「接受率是 0」—— 接受率**没有测过**，不是 0。
- ✅ 正确表述：**「2.1a 因本机无 ≥17.5 GB 空闲显存而未能执行；2.1b 需要 ≈37 GB，
  当前机器给不出。」**

### 转路线 A？**不转。**

任务书规定的转 A 条件是「2.1a **失败**（拿不到视频输入 / 形状无法重建）」。
本次是**根本没跑起来**，**不构成 2.1a 的技术性失败**，因此**不触发转 A**。
若因资源问题就转 A，等于用一个未验证的失败去换另一条更重的路线 —— 没有依据。

---

## 2.1 一句话总结

**vLLM 上单 token 真 STD 的接受率 = 未测（不是 0）**；
2.1a 的钩子验证做了 5 次，**全部死在 GPU 显存**（模型能加载、视觉编码器 OOM；
本机最好只剩 16.7 GB，而 vLLM 单侧最低要 ~17.5 GB，2.1b 要 ~37 GB）。
差异原因无从谈起 —— **没有数据**。

---

## 2.1a 定稿：**PASS**（2026-09-26 10:33，GPU7，32 帧）

```
verdict      : PASS
num_frames   : 32
prompt_tokens: 19166
captured_keys: ['pixel_values_videos', 'second_per_grid_ts', 'video_grid_thw']
```

| 张量 | 形状 | dtype | device |
|---|---|---|---|
| `pixel_values_videos` | **`[131072, 1176]`** | bfloat16 | cuda:0 |
| `video_grid_thw` | `[1, 3]` | int64 | cpu |
| `second_per_grid_ts` | `[1]` | bfloat16 | cpu |

`1176 = 3 ch × 2 temp × 14 × 14` —— 即 vLLM 视觉塔直接消费的**已切分 patch 张量**，
正是 HF `Qwen2_5_VLForConditionalGeneration.forward` 也期望的输入格式，**可以原样喂回**。

### ★ 决定性发现：正确的钩点是 `embed_multimodal`，**不是** `forward`

**第一版钩 `forward`，结果是 `captured_keys: []`**，且 `call_log` 显示：

```
n_input_ids: None | kwargs: []
```

**`input_ids=None` 且 kwargs 为空** —— 说明：

> **vLLM v1 把视觉塔单独跑**（启动日志：`Encoder cache will be initialized with a budget of
> 32768 tokens`），把算好的 `inputs_embeds` 交给 LLM。**原始像素根本不经过 `forward`。**

正确钩点：`Qwen2_5_VLForConditionalGeneration.embed_multimodal(**kwargs)`
（`qwen2_5_vl.py:1595`），它分发到 `_process_video_input`（`:1432`），
后者在 `:1442` 取出 `pixel_values_videos`、在 `:1451` 送进 `self.visual`。

**修正后立刻 PASS。** 这条经验对后续所有阶段有效：
**在 vLLM v1 里找多模态输入，要钩 `embed_multimodal`，不要钩 `forward`。**

### 2.1a 的显存配置（实测，非估算）

| 项 | 值 |
|---|---|
| 帧数 / max_model_len | 32 / 32768 |
| **可用 `gpu_mem_util`** | **0.560**（在 37 GB 空闲窗口上） |
| 失败过的值 | 0.360 → KV 预算 **-3.3 GiB**；0.430 → KV 预算 **0.03 GiB**（需 1.75） |

**教训**：32 帧时 vLLM 为编码器缓存 + activations 保留约 **5.4 GB**，
不是按「权重 + KV」两段式能算出来的。**0.430 差 1.7 GB 就是不行。**

### 顺带修掉的两个自己的 bug

1. **失败分类漏了关键串**：`No available memory for the cache blocks` 与
   `larger than the available KV cache memory` 未被识别为显存类失败，
   导致驱动把「预算不够」误判成「技术失败」并退出。
2. **stale-log bug 复发**：失败判定 grep 的是**整个追加日志**，
   匹配到上一轮的旧错误串。已改为只检查本次尝试新增的字节区间
   （与 `run_bench.sh` 早先的修法一致）。

---

## 2.1b + 2.1c 定稿：**双双 PASS**（2026-09-26，GPU2 满卡窗口）

### 2.1b 共存显存 —— PASS

| 项 | 值 |
|---|---|
| 帧数 / max_model_len / util | 8 / 8192 / **0.400** |
| vLLM 加载后空闲 | 30,050 MiB |
| HF(15.45 GiB) 加载后空闲 | 14,586 MiB |
| **余量** | **14.36 GiB**（门槛 5 GiB）✅ |
| HF 加载耗时 | 5.71 s |

**结论：显存不是阻塞点。** 8 帧配置下两份 7B 权重（15.67 + 15.45 GiB）共存后仍余 14.36 GiB。
（依赖坑：`device_map` 需要 `accelerate`；已改为 `.to("cuda")`，**不引入额外依赖**。）

### 2.1c 输入对齐 —— PASS ★ 本阶段最有价值的结果

```
vllm_prompt_tokens     : 4814
captured_video_grid_thw: [4, 52, 92]
captured_pixel_shape   : [19136, 1176]
hf_next_token          : 785
vllm_first_token       : 785
next_token_match       : True
```

**用 vLLM 的 `prompt_token_ids` + 捕获的像素张量喂给 HF，HF 的 greedy 下一个 token = 785，
与 vLLM 实际产出的第一个 token **完全一致**。**

这是**预测接受率的最强单项证据**：两边在同一上下文上给出同一个 argmax，
说明跨框架数值差异在这个位置上没有造成分歧。

### ★ 又一个「捕获第一次调用」的坑（已修）

第一版 2.1c 报假失败：

```
captured video_grid_thw: [1, 256, 256] → HF 算出 16384 features
prompt 里的 video tokens: 4784
ValueError: Videos features and video tokens do not match
```

**根因**：vLLM 启动时先用**最大特征尺寸**做一次 encoder profiling
（启动日志原文：`profiled with 1 video items of the maximum feature size`）。
`mm_call_shapes` 记录证实了这点 —— **只有两次调用**：

```
call #1: pixel_values_videos [65536, 1176]   ← warmup（最大尺寸）
call #2: pixel_values_videos [19136, 1176]   ← 真实请求
```

我捕获了**第一次**（warmup），于是得到一个**幻影不匹配**。
改为捕获**最后一次**后立即 PASS。

> **这是同一类 bug 的第三次出现**（前面在 `03b` 探针、`P1` profiling 里都踩过）。
> **规训：在这个代码库里，任何「捕获形状/张量」的钩子都必须捕获最后一次调用，
> 或按与 prompt 长度匹配来挑选，绝不能取第一次。**

### 一个值得记录的细节：8 帧 → grid T=4

`video_grid_thw = [4, 52, 92]`，**T=4 而不是 8** —— vLLM 的时间维做了合并
（8 帧 → 4 个时间位置）。HF 用同一份 grid 张量，因此天然一致。

### 阶段 2.1 汇总

| 步骤 | 状态 | 结论 |
|---|---|---|
| 2.1a 捕获 video inputs | ✅ PASS | 钩 `embed_multimodal`（非 `forward`）；三个张量齐全 |
| 2.1b vLLM+HF 共存 | ✅ PASS | 余量 **14.36 GiB**，显存非阻塞 |
| 2.1c 输入对齐 | ✅ PASS | **HF 与 vLLM 的下一个 token 一致（785）** |
| 2.1d proposer 骨架 | ⏳ 未开始 | 两门前置已通过，可以开工 |
| 2.1e 单 token 接受率 | ⏳ 未开始 | — |
