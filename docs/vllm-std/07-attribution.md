# 阶段 7：对照与归因

> 状态：**✅ 完成（结论为负，但每个结论都有对照实验与硬数字）**
> 日期：2026-09-26
> 产物：`results/vllm-std/07_paired.json`、`results/vllm-std/07_draft_model_probe.json`

---

## 7.1 核心问题的答案

> **「加速到底来自哪里？是草稿长度、接受率、batch，还是框架实现？」**

**答：没有加速。** 在本次全部实测配置中，投机解码**没有产生任何净加速**；
用交错配对测量时它稳定地**变慢 7.7%**。

「加速」表象的三个来源，逐一被排掉：

| 候选来源 | 判决 | 证据 |
|---|---|---|
| 草稿长度 `k` | ❌ 无关 | `k=1..8` 耗时无趋势（5.24–5.43 s），接受率恒 0（阶段 5） |
| 接受率 | ❌ 恒为 0 | 所有配置 `accept_rate = 0.0` |
| batch | ❌ 无法归因 | batch 曲线形状由共享机调度决定（吞吐 b1→b2 崩塌 5.5×）（阶段 6） |
| **框架实现** | ✅ **唯一真实效应，且为负** | 交错配对 **0.9227 ± 0.03×**；空草稿对照 **0.954×** |

---

## 7.2 对照矩阵（全部同机、同视频、同帧数、同采样参数）

| # | 配置 | 耗时 | vs eager | 接受率 | 结论 |
|---|---|---|---:|---:|---|
| 1 | **eager baseline**（无投机） | 5.2305 ± 0.0406 s | 1.000× | — | 基准 |
| 2 | **`draft_model` 模式** | **起不来** | — | — | 见 §7.3，`NotImplementedError` (M-RoPE) |
| 3 | **custom_class / EmptyProposer**（零草稿） | 5.4807 ± 0.0327 s | **0.954×** | — | **框架税 = 4.8%** |
| 4 | **custom_class / RepeatProposer**（草稿全被拒） | 7.5295 ± 0.1534 s | **0.695×** | 0.0 | 白干越多的验证越慢 |
| 5 | **custom_class / SparseContext k=1** | 5.5080 ± 0.0834 s | 0.950× | 0.0 | — |
| 6 | **custom_class / SparseContext k=4 W=256** | 5.4038 ± 0.0420 s | 0.968× | 0.0 | — |
| 7 | **custom_class / SparseContext k=8 W=256** | 5.4231 ± 0.0368 s | 0.964× | 0.0 | — |
| 8 | **交错配对 A/B（k=4 W=64，n=2）** | — | **0.9227 ± 0.03×** | 0.0 | **权威数字** |

---

## 7.3 对照 2：`draft_model` 失败路径（阶段 2 预测被实测确认）

Phase 2 §2.6 predicted: the blocker is **M-RoPE**, not multimodality, and it lives in
`LLMBaseProposer.__init__` — which `custom_class` never runs.

`scripts/vllm-std/07_draft_model_probe.py` 用真实的 `method="draft_model"` + Qwen2.5-VL 跑了一遍：

```json
{
  "outcome": "FAILED",
  "target_uses_mrope": true,
  "draft_uses_mrope": true,
  "target_rope_scaling": {"type": "default", "mrope_section": [16, 24, 24], "rope_type": "default"},
  "confirmed_root_cause": "NotImplementedError: Speculative Decoding with draft models or parallel drafting does not support M-RoPE yet",
  "phase2_prediction_confirmed": true
}
```

**逐字引擎错误**（`results/vllm-std/logs/07_draft_model_probe2.log`）：

```
NotImplementedError: Speculative Decoding with draft models or parallel drafting
does not support M-RoPE yet
```

要点三条：

1. **不是**死在「多模态」上。多模态在 v0.22.0 里只是一条 warning（`_warn_if_multimodal`）。
2. **是**死在 **M-RoPE** 的 `NotImplementedError`（`_raise_if_mrope`，`llm_base_proposer.py:309`）。
   Qwen2.5-VL 的 `rope_scaling` 含 `mrope_section` → `uses_mrope=True` → 直接抛错。
3. 这道闸门长在 `LLMBaseProposer.__init__`（`llm_base_proposer.py:198`）。
   `custom_class` **从不实例化该基类** → 闸门永不执行 → **多模态 target 可以配自定义 Proposer**。

> 附带踩坑记录：vLLM 的 `_uses_mrope` 读的是**归一化后**的 `config.rope_parameters`，
> 而 transformers 的 `AutoConfig` 暴露的是 `rope_scaling`。
> 在原始 HF config 上调 `uses_mrope()` 会得到 **假阴性 False**（探针第一版就被骗过）。
> 修复后同时检查两个属性名 → `True`。这条已写进探针注释，避免复现者重踩。

---

## 7.4 对照 8：交错配对测量（本阶段最重要的方法学产物）

### 问题

阶段 5 出现了「窗口 64 → 4.15 s（**1.260×** 快于基准）」这类数字。
核对每次运行记录的 GPU 状态后发现：

| 运行 | 该卡空闲显存 | 该卡利用率 |
|---|---:|---:|
| `p5_w256_k*`, `p5_w0_k4`, `p5_w64_k4` | 43,759 MiB | **24–26%（有邻居）** |
| `p5_w1024_k4` | **48,648 MiB** | **0%（独占空卡）** |

→ **跨时刻的单点计时不可比。** 这些「加速」是**测量假象**。

### 方法

交替 A,B,A,B 运行（`scripts/vllm-std/07_paired_ab.sh`），
使时间漂移成为**共模误差**，在配对差值中抵消。

### 结果

| 配对 | eager | spec (k=4, W=64) | 差值 | 加速比 | 接受率 |
|---|---:|---:|---:|---:|---:|
| 1 | 3.7429 s | 4.1520 s | +0.4091 s | **0.9015×** | 0.0 |
| 2 | 3.9276 s | 4.1612 s | +0.2336 s | **0.9439×** | 0.0 |

**配对结果：`0.9227 ± 0.0300×`（n=2），配对差值 +0.3214 ± 0.1241 s。**

### 这条结果同时完成了两件事

1. **否证阶段 5 的 1.260×**：同一个配置（k=4, W=64），交错配对实测 **0.90–0.94×**。
2. **给出可信的归因**：投机解码在本任务上**稳定慢 ~7.7%**，两次配对方向一致（都 < 1）。

---

## 7.5 负结果的可复现刻画

任务书要求「如果负结果，必须给出可复现的负面刻画」。完整配方：

```bash
# 环境
bash scripts/vllm-std/00_setup_env.sh          # vLLM 0.22.0+cu129, torch 2.11.0+cu129
source scripts/vllm-std/env.sh

# 1) 基线（阶段 1/3）
bash scripts/vllm-std/run_bench.sh results/vllm-std/logs/rep_base.log \
  03_bench_spec.py --num-frames 32 --max-tokens 128 --runs 3 --warmup 1 --tag rep_base

# 2) 交错配对（阶段 7）—— 这是唯一可信的计时方式
bash scripts/vllm-std/07_paired_ab.sh 2
$PY scripts/vllm-std/07_paired_analyze.py

# 3) draft_model 边界（阶段 7）
MIN_FREE_MIB=38000 bash scripts/vllm-std/run_bench.sh \
  results/vllm-std/logs/rep_probe.log 07_draft_model_probe.py --max-model-len 8192
```

**预期复现到的三个数**：

| 观测量 | 预期值 | 允许波动 |
|---|---|---|
| 空草稿框架税（EmptyProposer k=4） | **≈ 0.95×** | ±0.03 |
| 交错配对加速比（SparseContext k=4 W=64） | **≈ 0.92×** | ±0.05 |
| 接受率（所有 token 级草稿配置） | **0.00** | 严格为 0 |
| `draft_model` + Qwen2.5-VL | `NotImplementedError` (M-RoPE) | 逐字一致 |

---

## 7.6 结论汇总（每条都有对照）

| # | 结论 | 对照实验 | 硬数字 |
|---|---|---|---|
| 1 | 多模态 target **可以**配自定义 Proposer，无需补丁 | `draft_model` 实测死于 M-RoPE；`custom_class` 成功运行 | 381 组草稿到达验证层 |
| 2 | 开投机有 **4.8% 固定框架税** | EmptyProposer（零草稿）vs eager | 0.954× |
| 3 | token 级稀疏草稿在本任务上**接受率为 0** | 4 种 k × 4 种窗口 | `accept_rate = 0.0` |
| 4 | 草稿长度**无收益** | k=1,2,4,8 | 耗时 5.24–5.43 s，无趋势 |
| 5 | batch 曲线**不可用于归因** | 逐运行 GPU 状态取证 | 吞吐 b1→b2 崩塌 5.5× |
| 6 | 交错配对下投机**慢 7.7%** | 配对 A/B（n=2） | **0.9227 ± 0.03×** |
| 7 | `temperature=0` **不保证唯一答案** | eager(graph) vs eager(no-graph) | token 62 分歧；logprob 精确相等 |
| 8 | `custom_class` **无法承载 STD 稀疏 KV 草稿** | 接口签名 + 模块清单 | 无 model/KV/attn 句柄 |

---

## 7.7 EAGLE-3 对照（按任务书要求说明，未执行）

任务书规定「EAGLE-3 只作为备选，不作为主线；没有对应 draft head 权重时不要浪费时间」。

本仓库 §20（`docs/eagle3-video-probe-findings.md`）已记录：本地**有**一个训好的 EAGLE-3 draft
（`qwen2.5-vl-7b-eagle3-sgl`），但在 **sglang** 上 6 次启动全部崩溃，且该 draft 上下文仅 8192 token。
**那是 sglang 路线，与本次 vLLM 路线不共享引擎。**

在 **vLLM** 上跑 EAGLE-3 需要 draft head 与 vLLM 的 `eagle3` 权重重排格式匹配。
本次**未执行**，理由：① 主线（custom_class）已被证明无法承载稀疏草稿，先解决注入点才有意义；
② 即使 EAGLE-3 在 vLLM 上跑通，它也只能证明「有 EAGLE head 时能加速」，**不能回答 STD 的问题**。
**不把未做的事写成结论。**

---

## 7.8 验收

| 判据（任务书） | 结果 |
|---|---|
| 对照：baseline / draft_model 失败路径 / 自定义 Proposer | ✅ 三者齐备（§7.2 表 1/2/3-8） |
| 回答「加速来自哪里」 | ✅ **没有加速**；唯一致因是框架税，且为负 |
| 负结果给出可复现刻画 | ✅ §7.5 给出命令、预期值与波动范围 |
| 每个结论都有对照实验和硬数字 | ✅ §7.6 八条逐一对应 |

---

## 7.9 【线 A 更正】独占卡复测：`0.9227×` 是**悲观偏差**，修正为 `0.9718×` ★

> 日期：2026-09-26（后续）
> 触发：用户指出「共享卡既然能制造 1.260× 的**假加速**，同样可能在压低 0.9227×」
> 证据：`results/vllm-std/A1_paired_exclusive.json`、`results/vllm-std/logs/lineA_driver.log`

### 做法

发现 **GPU0 当时无任何 compute app（独占）**，于是在 `STD_FORCE_GPU=0` 下重跑配对 A/B，
与 §7.4 的共享卡测量使用完全相同的主机、脚本、输入与采样参数。

### 结果

| 配对 | eager | spec (k=4, W=64) | 加速比 | 接受率 |
|---|---:|---:|---:|---:|
| 1 | 4.0173 s | 4.1050 s | 0.9786× | 0.0 |
| 2 | 4.0824 s | 4.2306 s | 0.9650× | 0.0 |

**独占卡：`0.9718 ± 0.0096×`（n=2）**
共享卡（§7.4）：`0.9227 ± 0.0300×`（n=2）

### 结论：用户的怀疑成立，但方向是「差别不大，仍为负」

1. **共享卡确实压低了数字**：−7.7% → **−2.8%**。差距约 5 个百分点，
   即争用环境**放大**了 proposer 的劣势（多进程/多核争抢对 CPU 侧 proposer 更不利）。
2. **但定性结论不变**：`0.9718× < 1`，两次配对方向一致，**仍然没有加速**。
   所以 §7.1「没有加速」的结论**不需要改写**，只需要修正幅度。
3. **对外使用的数字应改为 `0.9718 ± 0.0096×`**（独占卡），并在方法节说明共享卡会引入约 5pp 的悲观偏差。
4. **`1.260×` 假加速与 `0.9227×` 假减速是同一现象的两面**：单点计时在共享机上**双向**不可信。
   这反过来加强了 §C.4 的方法学论点。

### 未完成（已记录，未外推）

| 项 | 状态 |
|---|---|
| 第 3 对配对 | ❌ GPU0 被他人作业抢占（free 48658 → 9334 MiB，util 0% → 90%），A3 起不来 |
| batch b1–b8 独占曲线 | ❌ 同一原因未取得；脚本 `lineA_exclusive.sh` 已就绪，待独占卡可复用 |
| b16/b32 独占复测 | ❌ 同上 |

**未用外推数字填充这三项。** 复现命令：

```bash
STD_FORCE_GPU=<独占A6000> bash scripts/vllm-std/lineA_exclusive.sh
```
