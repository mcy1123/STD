# Dynamic Routing 彻底分析方案

**状态**：设计完成，未执行。
**前置记录**：`docs/experiment-provenance.md`（硬件分档 + 逐轮诊断）
**相关**：`PROGRESS.md` §14（A100 负结果）、§16（旧消融矩阵，本文取代其判读框架）

---

## 0. 为什么要重做

现有证据无法回答"dynamic 有没有用"，因为**每一次对比都同时变了多个变量**：

- 正结果（+1.66 accept）= A6000 + VDC + 256 tok + `v1/full`
- 负结果（−0.061）= A100 + Video-MME + 128 tok + `v2/incremental`

而 2026-09-17 的逐轮诊断给出了新的关键线索（`experiment-provenance.md` §4）：**A100 上 n=3 的均值 −0.061 掩盖了 +0.044 / −0.126 / −0.102 的分化**，且分化方向与 static 的剩余空间单调相关。

---

## 1. 核心重构：把二元问题改成条件问题

**不要再问**："dynamic routing 能不能提升接受率？"（答案必然是"看情况"，无法证伪）

**要问**：
> 在什么可测量的条件下，把 static `S_0` 换成 `TopK(A_{t-1})` 会提升接受率？
> 能否**先验地**从可测量特征预测这个增益？

### H_headroom（主假设）

> 动态更新每轮替换约 30% 的 visual KV（实测 Jaccard≈0.52，changed_tokens≈300/996）。
> 当 static 选择已接近最优时，该替换是纯噪声；当 static 远离最优时，重选带来真实增益。

支撑（n=3，方向一致但未证明）：

| 档位 | static accept | Δaccept |
|---|---:|---:|
| VDC 均值 | 0.531 | +0.188 |
| Video-MME 050-1 | 0.623 | +0.044 |
| Video-MME 496-3 | 0.864 | −0.102 |
| Video-MME 717-1 | 0.983 | −0.126 |

### 候选预测特征（全部可测，用于拟合/分层）

| 特征 | 含义 | 来源 |
|---|---|---|
| `static_accept` / `static_acc_len` | 剩余空间代理 | 每轮 verify 实测 |
| `attn_topk_mass` | verifier 注意力在 `S_0` 上的质量占比 | trace |
| `attn_concentration` | softmax 熵 / top-1 占比（越尖越可预测） | trace |
| `recall(S_0, S_t*)` | 静态选择相对 oracle 的召回 | trace |
| `drift_jaccard(S_0, S_t*)` | 漂移速度 | trace |
| `churn_jaccard(S_t, S_{t+1})` | 相邻轮选择稳定性 | 已有 `per_round` |
| `visual_len` / `text_len` / `frame_num` / `prompt_len` | 规模 | manifest |

**产出目标**：一个 `Δaccept ≈ f(特征)` 的可检验关系 + 一个启用判据（例如 `static_accept < τ` 时启用动态）。

---

## 2. 必须消除的 7 个混淆

| # | 混淆 | 现状 | 处理 |
|---|---|---|---|
| C1 | 硬件（A6000 vs A100） | 正负结果分属不同卡 | 主实验**固定 A100**；A6000 只做机制/离线 |
| C2 | 数据集（VDC vs Video-MME） | 从未在同一数据集上做干净对比 | 主实验用 **Video-MME**（标准集），VDC 作为低-headroom 对照 |
| C3 | 生成长度（256 vs 128） | 不同 | 固定 **256**（decode 占比更高，信号更足） |
| C4 | collector / refresh / query / bootstrap | 默认值曾被从 `v1/full` 改成 `v2/incremental` | **全部显式传参**，禁止依赖默认值 |
| C5 | S_0 是否等于 static | `attention` bootstrap 时才相等，但**未被断言** | 加 **equal-S_0 断言**（见 L0） |
| C6 | verify fallback | A100 上 6/6 exact 依赖它，但它本身正确性存疑 | 主实验 **fallback=none**；fallback 单列 A/B |
| C7 | 样本量与种子 | n=1/3/10，单种子 | **n≥20，3 seeds**，逐样本配对 |

---

## 3. 分层方法

### L0 — 仪表与不变量（任何实验之前必须通过）

> 目的：排除"我们以为在测 dynamic，其实没测到"这类错误。历史已经发生过一次（默认值漂移）。

- **T1 选择/cache 一致性断言**：每轮结束后，从 dense cache 重算 compact 布局，与 `state.indices` + `_std_visual_slot_map` 实际内容逐位比对。加 `--assert-selection-cache-consistency` 开关，主实验强制开启。
- **T2 equal-S_0 断言**：`bootstrap=attention` 时，断言 dynamic 的初始 `S_0` 与同 run 的 `static` 选择**逐位相同**。
- **T3 更新确实发生**：记录并断言 `update_applied` 的比例符合配置预期（interval=1 应≈100%；interval=4 应≈25%）。**A100 的 interval4 run 只有 5/23，说明节流比预期强得多，必须显式报告。**
- **T4 计时修复**：部分 run 出现 `total_collect_time_ms=0.0`；修好后组件计时才可用于瓶颈归因。
- **T5 单一配置路径**：所有参数写进 manifest 并回读校验；移除会静默改变行为的默认值。

**验收**：T1–T5 全部有测试且通过，否则不进入 L1。

### L1 — 离线机制分析（最便宜、最先做、不改变生成）

> 目的：直接回答"**在 Video-MME 上，`TopK(A_{t-1})` 到底有没有信号？**"
> **这件事从未做过**——Oracle Study（8/21）只覆盖了 VDC + MLVU。

**L1.1 在 Video-MME 上重做 Oracle Study**
- 采集 trace：20 样本，128f/256tok，γ=9，K=996，记录每轮 per-layer visual-only softmax attention mass。
- 计算 Static / Previous / Oracle 的 **recall** 与 **accept-proxy**。
- **按 `static_accept` 分箱报告**（低/中/高三档），检验 H_headroom。

**L1.2 collector 估计质量消融（纯离线，用同一份 attention）**
- 用同一轮 attention 分别按 **all-query（V1）/ two-query / three-query** 估计 top-K，比较各自对 oracle 的 recall。
- 直接判定：**two-query 是否就是 churn（Jaccard 0.52）的来源**。若 two-query recall 显著低于 all-query，则负结果的一部分是估计器问题，而非算法问题。

**L1.3 漂移与 churn 的结构分析**
- 逐轮画 `recall(S_0,S_t*)`、`Jaccard(S_t,S_{t+1})`、`static_accept` 的轨迹，看 churn 是"跟踪真实漂移"还是"纯噪声"。
- 判据：若 `S_t*` 本身轮间稳定（oracle 的 Jaccard 高）而我们的 `S_t` 轮间抖动大 → 是**估计噪声**。

**L1 产出**：一份机制报告 + H_headroom 的初步验证 + 是否需要先修 collector 的结论。**不需要正确性门槛，没有翻车风险。**

### L2 — 受控在线消融（等 S_0、单变量、逐样本配对）

主实验固定：Video-MME，20 样本，256 tok，γ=9，K+text=1024，`collector=v1`（全 query 控制）+ 显式 `refresh`，`fallback=none`，3 seeds。

| run | bootstrap | refresh | query | interval | min_change | 检验 |
|---|---|---|---|---|---|---|
| **R0** | attention | full | three | 1 | 0.0 | **纯控制**（= static 的 S_0，每轮全量更新） |
| R1 | attention | **incremental** | three | 1 | 0.0 | C4：incremental slot 顺序 |
| R2 | attention | full | **two** | 1 | 0.0 | collector 估计（与 L1.2 互证） |
| R3 | attention | full | three | 1 | **0.05** | 滞回 |
| R4 | attention | full | three | **4** | 0.0 | 更新频率 |
| R5 | **attention_free** | full | three | 1 | 0.0 | bootstrap 起点 |

**报告口径（强制）**：
- 逐样本 `Δaccept = dynamic − static`（配对），给 **bootstrap 95% CI**；
- **按 `static_accept` 分箱**报告 Δaccept（这是 H_headroom 的直接检验）；
- 同时报告 `update_applied` 比例、churn、changed_tokens；
- exact vs AR 逐 trial 报告，fallback 关闭时若 exact 掉，**单列**而不混入性能结论。

### L3 — wall-clock

仅在 L2 中 **accept ≥ static** 的分层/配置上测。报告 synchronized decode + 端到端，并给出组件分解（draft/verify/collect/refresh）。

---

## 4. 预注册判读规则（先说好，避免事后解释）

| 观察 | 结论 | 行动 |
|---|---|---|
| L1 中 `recall(Previous) ≈ recall(Static)` | 该数据集上**信号不存在** | 放弃在该数据集用动态；转条件化/换数据集 |
| L1 有 recall 优势，但 L2 的 Δaccept ≤ 0 | **应用层缺陷** | 回 L0，重点查 T1（选择/cache 一致性） |
| L2 的 Δaccept 与 `static_accept` 单调负相关 | **接受 H_headroom** | 采纳"headroom-gated 动态"：`static_accept < τ` 时启用——**这本身是正结果** |
| L2 各分层 Δaccept 均 ≤ 0，且 L1 recall 也无优势 | **真实负结果** | 正式归档，停止投入 |
| L2 中某单轴（如 R1 vs R0）差异显著 | **定位到实现缺陷** | 修该轴后重跑该轴 |

**统计纪律**：不使用未分层的均值；n≥20；3 seeds；配对检验；任何"提升"必须给 CI 且排除 0。

---

## 5. 成本与排期

| 阶段 | GPU | 规模 | 预估 |
|---|---|---|---|
| L0 | 0 | 纯代码 + 单测 | 0.5–1 天 |
| L1.1 trace 采集 | 1 卡 | 20 样本 × 1 次 | 1–2 h |
| L1.2/L1.3 分析 | 0 | 离线 | 0.5 天 |
| L2 筛选（n=10, 1 seed, 仅 R0/R1/R2） | 1 卡 | 3 run | ~6–9 h |
| L2 正式（n=20, 3 seeds, 6 run） | 1 卡 | 18 run | 多日（排队） |
| L3 | 1 卡 | 仅胜出配置 | 数 h |

**第一步（最小可判定）**：**L1.1 —— 在 Video-MME 上采一次 trace 并算 recall**。
不改算法、不跑消融、无正确性风险，且能直接判定"信号是否存在"。**在 GPU 被其他项目占用的当下，这是最该先排期的实验。**

---

## 6. 代码改动清单

| 项 | 文件 | 说明 |
|---|---|---|
| T1 一致性断言 | `src/std_repro/dynamic_std_qwen25vl.py` | 每轮校验 selection vs 实际 cache 布局 |
| T2 equal-S_0 断言 | 同上 + `scripts/benchmark_a100_dynamic.py` | 与 static 分支逐位比对 |
| T3 更新比例上报 | `dynamic_std_qwen25vl.py` | `update_applied` 已在 `per_round`，补聚合上报 |
| T4 计时修复 | `dynamic_std_qwen25vl.py` | collect/refresh 计时归零问题 |
| T5 显式配置 | `scripts/benchmark_a100_dynamic.py`、`run_a100_ablation.sh` | 禁止依赖默认值；manifest 回读校验 |
| L1 trace 采集 | `scripts/analysis/collect_traces.py`（扩展）+ 新 `analyze_headroom.py` | 记录 per-layer attention + 逐轮 accept |
| L2 分层报告 | `scripts/summarize_a100_dynamic.py` | 增加"逐样本配对 + 按 static_accept 分箱"输出 |

> 旧方案 `docs/superpowers/plans/2026-09-17-dynamic-std-ablation-rerun.md` 的 R1–R6 矩阵仍然可用，
> 但**判读框架由本文取代**：先过 L0，先做 L1，再谈 L2；且 L2 必须分层报告。
