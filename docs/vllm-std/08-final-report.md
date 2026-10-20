# 阶段 8：总结交付

> 日期：2026-09-26
> 工作目录：`/home/mcy/Projects/SpecVLM/STD`
> 主机：`A6000`（6×RTX A6000 48GB + 2×RTX 4090，驱动 550.163.01）

---

## 8.1 一句话总结

**Qwen2.5-VL 的视频推理已在 vLLM 上完整跑通并量化；投机解码接口已打通并实测；
但 STD 的稀疏草稿无法装入该接口，且 token 级替代方案在本任务上接受率为 0、净变慢 7.7%。**

三个目标逐一交代：

| # | 任务书目标 | 结果 |
|---|---|---|
| 1 | Qwen2.5-VL 在 vLLM 上跑通纯视频推理 | ✅ **成功**（32 帧 19,166 prompt tokens，输出内容与视频对得上） |
| 2 | 分析投机解码接口，找到绕开「draft_model 不支持多模态」的注入点 | ✅ **成功**（`custom_class`；并实测确认真正的死点是 M-RoPE 而非多模态） |
| 3 | 分阶段实现 STD（稀疏草稿 + 密集验证） | ❌ **未达成**（接口无法承载稀疏 KV 草稿；token 级替代方案为负结果，已刻画） |

---

## 8.2 是否成功

| 阶段 | 判定 | 关键数字 |
|---|---|---|
| 0 环境 | ✅ GO | vLLM 0.22.0+cu129 在驱动 550 上跑通 |
| 1 纯视频基线 | ✅ GO | 32 帧 → 19,166 tokens；128 tokens **5.1653 ± 0.0530 s**；3 次 bit-exact |
| 2 接口分析 | ✅ GO | 注入点定位到 `custom_class_proposer.py` + `gpu_model_runner.py:4778` |
| 3 Dummy Proposer | ✅ GO（标准已修正） | 接口通；**发现参考答案自身不稳定** |
| 4 STD 单 token | ⚠️ 部分 | **架构性阻塞**：接口无 model/KV/attention 句柄 |
| 5 STD 多 token | ⚠️ 负结果 | 接受率 **0.0**；`k` 无收益；窗口曲线受测量混淆 |
| 6 batch 扫描 | ⚠️ 部分 | b1–b8 完成；b16/b32 因吞吐崩塌中止；**无真实稀疏收益** |
| 7 对照归因 | ✅ GO | 交错配对 **0.9227 ± 0.03×**（慢 7.7%） |
| 8 总结 | ✅ 完成 | 本文 |

---

## 8.3 是否 bit-exact

**本任务的 bit-exact 要求在一台真实机器上不可满足，原因是参考答案本身不稳定。**

### 证据链

1. `RepeatProposer` 发 381 组 / 1524 个草稿，**接受 0 个**（所有 token 都由 target 自己选），
   输出却与 eager baseline 在 **token 62** 分歧，且两次独立复现完全一致。
2. 定位到该位置是**精确平局**：
   - 启用 CUDA graph：`' featuring'` -0.666198 vs `' with'` -0.791198
   - `--enforce-eager`：两者**都是 -0.726219，margin = 0.000000**
3. **决定性对照**：两次**都不带投机解码**的 eager 运行，唯一差别是是否用 CUDA graph，
   **同样在 token 62 分歧**（`16445` vs `448`）。

### 结论

`temperature=0` 的 greedy argmax 在**平局位置**由浮点噪声（kernel 选择、CUDA graph 重放、
batch 形状）决定。**任何实现都无法对一个自身不稳定的参考做到 bit-exact。**

margins 画像（70 个位置）：margin<0.25 有 **3 个（4.3%）**，margin<0.5 有 **9 个（12.9%）**。

### 修正后的判定标准（已写入阶段 3 文档，后续沿用）

报三元组 `(完全一致位置数, 平局位置分歧数, 稳定位置分歧数)` + margin 画像；
**只有第三项 > 0 才是真正的实现缺陷**。本工作流未发现稳定位置分歧。

---

## 8.4 加速比

### 权威数字：**0.9227 ± 0.0300×（慢 7.7%）**

交错配对 A/B（`07_paired.json`，n=2）：

| 配对 | eager | spec (k=4, W=64) | 加速比 | 接受率 |
|---|---:|---:|---:|---:|
| 1 | 3.7429 s | 4.1520 s | 0.9015× | 0.0 |
| 2 | 3.9276 s | 4.1612 s | 0.9439× | 0.0 |

### 全部配置对照

| 配置 | vs eager | 接受率 |
|---|---:|---:|
| eager baseline | 1.000× | — |
| custom_class / 零草稿（框架税） | **0.954×** | — |
| custom_class / 草稿全被拒 | 0.695× | 0.0 |
| custom_class / 稀疏 k=1 | 0.950× | 0.0 |
| custom_class / 稀疏 k=4 | 0.968× | 0.0 |
| custom_class / 稀疏 k=8 | 0.964× | 0.0 |
| **交错配对（权威）** | **0.9227×** | 0.0 |

> 阶段 5 曾出现 1.260× 的表观加速。逐运行 GPU 取证显示那一行跑在**独占空卡**上，
> 其余跑在 25% 占用的卡上 → **测量假象**。交错配对复测同一配置得 0.90–0.94×。

---

## 8.5 根因

| # | 现象 | 根因 | 层级 |
|---|---|---|---|
| 1 | 投机的「加速」不存在 | **接受率恒为 0**：token 级 n-gram/稀疏上下文草稿不匹配自由形式的视频描述文本 | 算法 |
| 2 | 开投机即慢 4.8% | `custom_class` 强制关闭异步调度（vLLM 日志明示）+ 验证路径固定开销 | 框架 |
| 3 | 无法实现 STD 稀疏 KV 草稿 | `custom_class` 是**纯 CPU / 纯 token 级**接口：构造只给 `VllmConfig`，`propose` 只收 token id，**拿不到 model / KV cache / attention 权重** | 架构 |
| 4 | batch 曲线形状不可用 | 共享机争用 + `Running: 1 reqs` 串行调度；吞吐 b1→b2 崩塌 5.5×（31.03 → 5.59 tok/s） | 机器 |
| 5 | b16/b32 未完成 | 引擎自报生成吞吐跌到 2.0–4.3 token/s，b32 预计 >40 min/臂 | 机器 |
| 6 | bit-exact 无法成立 | **精确平局** + 浮点噪声；eager-vs-eager 在 token 62 同样分歧 | 数值 |

---

## 8.6 交付物清单

### 文档（`docs/vllm-std/`）

| 文件 | 内容 |
|---|---|
| `00-environment.md` | 环境勘察；**vLLM 版本决策的完整取证**（为何必须 cu129 而非 PyPI 默认 cu13） |
| `01-baseline-video.md` | 纯视频基线；frame sampling / fps 行为验证 |
| `02-spec-decode-interface.md` | 投机解码接口分析；注入点全链路；报错边界；最小 patch 清单 |
| `03-dummy-proposer.md` | Dummy Proposer；**bit-exact 不可能性的完整证明** |
| `04-std-1token.md` | STD 单 token；**架构性阻塞**；token 级替代方案 |
| `05-std-multitoken.md` | 多 token + 参数扫描；测量混淆的定位 |
| `06-batch-sweep.md` | batch 扫描；b16/b32 中止根因 |
| `07-attribution.md` | 对照与归因；负结果的可复现刻画 |
| `08-final-report.md` | 本文 |

### 脚本（`scripts/vllm-std/`）

| 脚本 | 用途 |
|---|---|
| `env.sh` | 统一环境（去死代理、CUDA 12.6、PYTHONPATH） |
| `00_setup_env.sh` | 一键重建环境 + 固定源码树 |
| `gpu.sh` / `pick_gpu.sh` / `run_bench.sh` | 远端/本机 SSH 封装、按需选卡、抗争用重试 |
| `01_baseline_video.py` / `01b_frame_sampling.py` | 阶段 1 |
| `03_bench_spec.py` | **通用投机基准**（阶段 3–6 共用） |
| `03b_divergence_probe.py` / `03c_logit_margin.py` | 分歧定位 + logits 对比 |
| `compare_runs.py` | bit-exact 对比（含首个分歧 token） |
| `05_sweep.sh` / `06_batch_sweep.sh` / `summarize_batch_sweep.py` | 阶段 5/6 |
| `07_paired_ab.sh` / `07_paired_analyze.py` | **交错配对测量**（阶段 7 权威） |
| `07_draft_model_probe.py` | `draft_model` 失败路径取证 |

### 代码（`src/std_vllm/`）

`proposer.py`：`BaseProposer` / `EmptyProposer` / `RepeatProposer` / `SparseContextDraftProposer`

### 数据（`results/vllm-std/`）

`spec_bench.jsonl`（全部原始记录）、`01_baseline_video.jsonl`、`batch_sweep.csv`、
`07_paired.json`、`07_draft_model_probe.json`、`03_bitexact.json`、
`env/pip-freeze-vllm-std.txt`、`logs/`（全部原始日志）

---

## 8.7 下一步建议（按优先级）

### P0：把 STD 的稀疏 KV 草稿装进正确的注入点

`custom_class` 已证伪。真正可行的是**新增一个 GPU 侧 proposer，继承 `LLMBaseProposer`**
（它持有 `self.model`、能建 KV cache、有 `dummy_run` / `prepare_inputs`）。改动面 4 处，
清单见 `04-std-1token.md` §4.5。

**第一件必须实测的事（最大未知数）**：vLLM 0.22.0 默认注意力后端
（FLASH_ATTN / FLASHINFER）**不返回 attention 权重**，而 `build_sparse_selection` 依赖
prefill 注意力矩阵。要么强制走能返回权重的后端（`TRITON_ATTN` / eager SDPA，需实测性能代价），
要么改用不依赖注意力的稀疏判据。**先验证这一条，再决定要不要投工程量。**

### P1：换一台独占的 A6000 重做计时

本机 6 张卡被他人作业占满，导致：b16/b32 不可测、batch 曲线形状由调度决定、
阶段 5 出现假加速。**独占卡是拿到可信 batch 曲线的前提**，不是优化项。

### P2：若要在 token 级继续，换任务而不是换参数

token 级草稿（n-gram / 后缀 / 稀疏上下文）在**自由形式视频描述**上接受率为 0，
这是任务性质决定的，扫参数救不回来。若要验证 token 级机制，应选
**摘要 / 抽取 / 复制型**任务（prompt 与输出大量重合），例如 Video-MME 的
「找出视频中出现的文字」类问题。

### P3：缩短 `draft_model` 的 M-RoPE 阻塞

`_raise_if_mrope` 是一行 `if` 的硬阻塞（`llm_base_proposer.py:309`）。
若能在只读 KV 的验证语义下正确支持 M-RoPE 位置推进，`draft_model` 模式对
Qwen2.5-VL 即可用。但注意：**这不是 STD**，它解决的是另一条路。

---

## 8.8 诚实声明

- 本次**未实现** STD 的稀疏 KV 草稿。原因已用代码级证据定位（接口无法访问 model/KV/attention），
  并给出了最小 patch 路线与首要待验证风险。**没有把未做的事写成结论。**
- 阶段 6 的 batch 曲线**不完整**（b1–b8），b16/b32 中止并记录了根因与日志，**未用外推数字填充**。
- 阶段 5 的表观加速已被**自我否证**（交错配对 + GPU 取证），保留原始日志以便复核。
- 所有性能数字均来自**同机、同环境、同输入、同采样参数**，预热后重复；
  但**跨时刻不可比**这一本机特性已单独记录，并促成了阶段 7 的方法学修正。
