# 实验硬件归属与结果分档（STD 项目）

> **为什么必须有这份文档**：本项目的结论长期按"时间线"而非"硬件"归档，导致一个曾被误读的事实——
> **唯一一次 dynamic 正结果在 A6000/4090 上、唯一一次负结果在 A100 上，而且两者之间同时有 4 个变量不同。**
> 本文按硬件把每条结果钉死，并附 2026-09-17 新提取的逐轮诊断。
>
> 配套：`docs/superpowers/plans/2026-09-17-dynamic-routing-thorough-analysis.md`（彻底分析方案）

## 0. 判定依据

| 硬件标识 | 出现的文件 | 判定 |
|---|---|---|
| `"host": "gpu23"` | 仅 `results/a100_dynamic_*`（16 个） | **A100 远端** |
| `"hardware": ["NVIDIA RTX A6000", ...]` | 仅 `results/adaptive_gamma_*`（8 个） | **A6000 本机** |
| 无任何硬件字段 | `dynamic_std_mvp`、`routing_*`、`adaptive_k_offline`、`adaptive_gamma_offline` | 按时间线推断（见下） |

**时间线定性**：A100 部署完成于 **2026-09-05**，首批 A100 结果是 **09-09**。因此 **8 月的全部实验只可能在 A6000/4090 本机**上执行——这与 `adaptive_gamma_runtime`（8/25）自带 `hardware: [A6000, 4090]` 的显式记录一致。

---

## 1. A6000 / RTX 4090（本机，2026-08-21 ~ 09-05）

| 实验 | 数据集 | 配置 | 结果 |
|---|---|---|---|
| Oracle Study Phase 1（8/21） | VDC 10 + MLVU 10 | 128f, K=996, 615 轮 | **GO**：recall Static 0.5678 → Previous **0.7667**；accept proxy 5.70→6.67 |
| **Dynamic MVP（8/21）** | **VDC 10** | 128f / **256 tok**, γ=9, K=996, **v1 + full** | **static 4.732（acc 0.531）→ dynamic 6.395（0.719），Δ=+1.66**；但 decode **158.74s → 162.71s（0.975×）**；token_match 5/10 |
| Adaptive-K offline（8/24） | 现有 trace | 无 GPU | **NO-GO**（证据不足） |
| Adaptive-γ offline（8/24） | 现有 trace | 无 GPU | GO（仅 `proxy`），acceptance +15.32% |
| Adaptive-γ runtime（8/25） | VDC 1 | 128f/256tok | **CORRECTNESS NO-GO**；γ 切换 25/54（46.3%） |
| Adaptive-γ capped9（8/25） | VDC 1 | γ∈{3,5,7,9}, q_len=10 | **CORRECTNESS NO-GO**；padding 槽位 **0.71% → 35.26%**；adaptive accept 0.639 但 mean accepted len 3.51 < static 5.12，rounds 57 > 42 |
| fixed-q sweep（8/25） | VDC 2 已知 mismatch 样本 | q=10..14 | **FIXED-Q NO-GO**（无任何 q 对所有样本 AR-exact，q*=none） |
| Verification fallback（9/05） | VDC 1（`v_-6dz6tBH77I`） | 96f/256tok | **存疑**：`fallback=none` static match=true；`sequential_on_low_margin` static match=**false**，fallback_count 11，verify 3.41→6.16s |

> **A6000 侧从未测过 static-vs-AR 的同口径加速比。** 「1.15–1.24×」是 A100 的数字，不要记到这一档。

---

## 2. A100 80GB / gpu23（远端，2026-09-09）

`benchmark_a100_dynamic.py`，Video-MME 可用子集 seed-42，128 帧，repeats=2，warmup 16 tok，K+text=1024，γ=9，**static 与 dynamic 在同一 run 内同张量对比**。

| 实验 | 样本 | 生成长度 | 结果 |
|---|---|---|---|
| Dynamic VG-Lite `v2_i1_final` | 3 | 128 | static accept **0.823** → dynamic **0.762**；static decode 34.39s → dynamic 41.78s（**0.823×**）；exact 6/6 |
| Dynamic `q2_profile3` | 3 | 128 | static 0.823 → dynamic 0.762；0.827× |
| Dynamic `interval4_profile` | 3 | 128 | static 0.823 → dynamic 0.777；0.885× |
| Dynamic `bootstrap_windowed`（attention_free） | 3 | 128 | static 0.823 → dynamic **0.731**；0.808× |
| Dynamic `v2_i1_t256` | 3 | 256 | static 0.857 → dynamic 0.797；0.794× |
| `f128_t128_r2`（v1/v3） | 3 | 128 | static 0.823 → dynamic 0.751；0.68× |
| **static vs AR（同 run 内）** | 3 | 128 | decode **1.15–1.24×**；端到端 1.03–1.08× |
| HSD spike | 3 | 32f/64tok | AR 1.640s、static 1.897s、small dense 5.967s、**HSD 5.227s**；HSD 比 static 慢 **2.76×**；exact 6/6 |

**A100 运行使用的代码 = 当前代码**（2026-09-17 逐文件 sha256 与 manifest 比对，6/6 `SAME`）。即负结果**不是**因为旧版缺 slot-map 修复。

---

## 3. 纯离线模拟（无 GPU）

- `results/adaptive_k_offline/` — Adaptive-K budget 模拟 → NO-GO。
- `results/adaptive_gamma_offline/` — γ 成本模型 → GO（全部为 `proxy` / `optimistic_estimate`）。

---

## 4. 【新】逐轮选择诊断（2026-09-17 从 `dynamic_stats.per_round` 提取）

此前只报告了均值，从未看逐轮行为。这是本次最重要的新证据。

### 4.1 interval=1（`v2_i1_final`，每轮都更新）

| 样本 | rounds | static acc | dyn acc | Δ | Jaccard(S_t,S_{t+1}) | 每轮替换 token（K=996） | collect | refresh |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 050-1 | 19 | 0.623 | 0.667 | **+0.044** | 0.536 | 301.7（**30.3%**） | 141ms | 415ms |
| 717-1 | 15 | 0.983 | 0.857 | **−0.126** | 0.524 | 298.4（30.0%） | 191ms | 392ms |
| 496-3 | 17 | 0.864 | 0.762 | **−0.102** | 0.510 | 320.3（32.2%） | 107ms | 417ms |

- `update_applied = 19/19`、`15/15`、`17/17`：**interval=1 时确实每轮都在改选择**。
- `mean_jaccard ≈ 0.51–0.54`：相邻两轮有近 **一半** 的 visual KV 不同。
- `attention_free` 与 `two-query` 变体的 jaccard 同样在 0.48–0.53，替换量 295–358。

### 4.2 interval=4（`interval4_profile`，节流）

`update_applied` 仅 **5/23、3/14、3/15**——大部分轮次被 interval/滞回拦下，**干预强度约为 interval=1 的 1/4**。

### 4.3 核心：收益与 static 的"剩余空间"单调相关

| 档位 | static accept | Δaccept（dynamic − static） |
|---|---:|---:|
| VDC 10 样本均值 | 0.531 | **+0.188** |
| Video-MME `050-1` | 0.623 | **+0.044** |
| Video-MME `496-3` | 0.864 | **−0.102** |
| Video-MME `717-1` | 0.983 | **−0.126** |

**假设 H_headroom**：动态更新每轮替换约 30% 的 visual KV。当 static 选择已接近最优（accept 高）时，这个替换是**纯噪声**，只会损害草稿；当 static 离最优很远（accept 低）时，重选带来真实增益。

> ⚠️ **证据强度**：`n=3`（+1 个聚合点）。方向高度一致，但**远未证明**。这正是彻底分析方案要检验的核心命题。
> 若成立，"dynamic 有用吗"就不是二元问题，而是 **"在 static 有多少剩余空间时才启用 dynamic"** 的条件问题——这本身是一个可发表的正结果。

---

## 5. 已知混淆（写结论时必须声明）

唯一正结果与唯一负结果之间**同时**变了 4 件事：

| | 正结果（+1.66） | 负结果 |
|---|---|---|
| 硬件 | A6000 / 4090 | A100 |
| 数据集 | VDC | Video-MME |
| 生成长度 | 256 token | 128 token |
| collector / refresh | **v1 + full** | **v2 + incremental** |

因此「dynamic 行不行」**至今没有被干净地回答过一次**。

## 6. 由此得到的两个直接推论

1. **STD 在任何硬件上都没有拿到过 >1× 的 dynamic 加速。** 唯一 >1× 的是 **static vs AR**，且只在 A100 上测过。
2. **不要用均值下结论。** A100 上 n=3 的均值 −0.061 完全掩盖了 +0.044 / −0.126 / −0.102 的分化；后续分析必须**逐样本配对 + 分层**。

---

## 7. 【2026-09-19】跨硬件不可比的直接证据（同一配置、同 10 样本）

**这不是推断，是同一条命令在两台机器上的实测对照。**

配置完全相同：`benchmark_a100_dynamic.py`，Video-MME 可用子集 seed-42 前 10 个样本（**顺序逐位相同**）、
128 帧 / 128 tokens、γ=9、K+text=1024、`cache-len 20480`、`collector=v2`、`refresh=incremental`、
`interval=1`、`min_change=0.00`、`fallback=none`，`--assert-equal-s0 --assert-consistency` 均开。

| 运行 | 硬件 | 日期 | 文件 |
|---|---|---|---|
| **R1** | A6000 本机（`host: A6000`, physical GPU 0） | 09-18 | `results/l2_screening/R1_incr_three_att.jsonl` |
| **A / B / C / D** | A100（`host: gpu23`, physical GPU 1） | 09-19 | `results/l2_screening/e1e6_20260919/` |

**static 接受率（static 与 collector / interval 无关，所以这本该是一个常数）：**

| 样本 | A6000（R1） | A100（A/B/C/D 四方一致） | Δ |
|---|---:|---:|---:|
| 050-1 | 0.6627 | 0.6229 | **−0.0398** |
| 717-1 | 0.9829 | 0.9829 | 0.0000 |
| 496-3 | 0.7958 | 0.8636 | **+0.0678** |
| 754-1 | 0.5072 | 0.4953 | −0.0119 |
| 154-3 | 0.5408 | 0.6918 | **+0.1510** |
| 445-2 | 0.8309 | 0.7902 | −0.0407 |
| 102-2 | 0.8496 | 0.8906 | **+0.0410** |
| 647-1 | 0.9664 | 0.9664 | 0.0000 |
| 504-2 | 0.9664 | 0.8571 | **−0.1093** |
| 599-1 | 1.0000 | 1.0000 | 0.0000 |
| **均值** | **0.8103** | **0.8161** | **+0.0058** |

- **逐样本最大差 0.151（154-3），平均绝对差 0.046**；而**均值只差 0.006**。
  → 同一份代码、同一条命令、同一批样本，换一张卡就能把单个样本的 static 接受率挪动 15 个百分点。
- A100 侧 **A/B/C/D 四次独立进程调用给出逐位相同的 static**，所以这不是随机性，是**硬件确定的数值差异**
  （Qwen 的自定义 attention 与 batched verification 的 reduction 顺序不同）。
- **A100 上这批 run 全部 10/10 exact vs AR**；同一配置在 A6000 上是 **8/10**（`496-3`、`754-1` 不符）。
  所以连"是否 bit-exact"都是硬件属性。

**两条硬规则（写结论前必须遵守）：**

1. **不同硬件上的 run 不能配对。** 把 A6000 的 R1 当作 A100 的 B/C/D 的基线，会把配置效应和硬件效应混在一起——
   而且因为均值只差 0.006，混进去之后**看不出来**。
2. **`docs/experiment-provenance.md` 的"硬件分档"必须下沉到"同一硬件同一 session"这一级。**
   基线要在同一台机器、同一批进程里现测；跨 session 的同硬件结果也要先验证 static 是否逐位相同再配对。
