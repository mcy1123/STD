# STD 项目进展文档

> 本文档记录 SpecVLM/STD 仓库（复现 *Sparse-to-Dense: A Free Lunch for Lossless Acceleration of Video Understanding in LLMs*）的完整研究进展，聚焦 **Verification-Guided Dynamic Routing（验证引导的动态视觉路由）** 这条主线，并附静态 STD 复现背景。

---

## 先看这个：现在是什么情况（大白话版）

> 技术细节在后面各节；这一节只讲"我们在干嘛、现在到哪了"。最后有一行术语对照。

### 我们在做什么

目标是**让视频大模型回答问题时跑得更快，而且答案一个字都不变**。

做法是"先猜后查"：先用一个简化版模型快速猜出接下来几个词，再让完整版模型一次性把这些猜测全检查一遍。**猜对的越多，需要完整版模型跑的轮数越少，就越省时间。**

- **静态做法**：每次都固定看视频里同一批画面块。这是论文的做法，我们已经复现，能跑。
- **动态做法**：每猜完一轮，看看完整版模型"刚才把注意力放在哪些画面块上"，下一轮改看那些。**这是我们自己加的研究点。**

### 判断好不好，只看一个数

**接受率**：这一轮猜的词里，完整版模型认可了几个。认可越多越快。

所以"动态做法有没有用" = "改看别的画面块之后，认可变多了还是变少了"。

### 最重要的新结论：我们过去问错了问题

以前一直问"动态做法到底有没有用"，结论忽正忽负，谁也说服不了谁。**现在搞清楚了：真正决定成败的，是"静态做法自己已经做得有多好"。**

- 静态做法**本来就猜得很准时**（接受率接近满分），改看别的画面块只会**帮倒忙**——因为没有可改进的空间，改动就是纯捣乱。
- 静态做法**猜得不准时**，改看别的画面块能**明显帮上忙**。

Video-MME 上 10 个样本的实测（静态接受率 → 改成动态后的变化）：

| 静态做得怎样 | 样本 | 改成动态后 |
|---|---|---|
| 差 0.507 | 754-1 | **+0.191（明显变好）** |
| 差 0.541 | 154-3 | **+0.106（变好）** |
| 一般 0.663 | 050-1 | +0.049（略好） |
| 一般 0.796 | 496-3 | 0（没变化） |
| 好 0.831 | 445-2 | **−0.184（明显变差）** |
| 好 0.850 | 102-2 | −0.162（变差） |
| 好 0.966 | 504-2 | −0.089（变差） |
| 好 0.966 | 647-1 | −0.031（略差） |
| 好 0.983 | 717-1 | **−0.187（明显变差）** |
| 满分 1.000 | 599-1 | 0（没变化） |

**十个样本平均下来是 −0.031**，看起来"没用甚至有害"。**但这个平均数是骗人的**：它把"做得好时大亏"和"做得差时大赚"混在一起抵消掉了。两个数之间的相关性是 **−0.74**（静态越接近满分，动态越吃亏）。

**所以正确的说法不是"动态没用"，而是"动态要在静态还差得远的时候才有用"。**

### 顺便排除掉的两种解释

之前怀疑过两个原因，现在都排除了：

1. **不是缓存更新方式的锅。** 我们担心过一种"局部更新"方式会打乱内部顺序而变差。这次特意用了**完全重建**这种最干净的方式，**结果一模一样**（还是"静态好就亏、静态差就赚"）。所以这个解释不成立。
2. **不是"只取两个位置"的锅。** 我们担心过只从 2 个位置取信息不够准。实测它确实损失约 18% 的精度，**但它几乎不影响"前后两轮选择忽上忽下"这件事**，所以它也不是主因。

### 还欠一件事（重要，别当已完成）

上表只跑完了**完全重建**这一种配置。**"两种缓存更新方式的直接对比"还没跑完**（代号 R1，被本机显卡占用卡住了，现已按你的要求停下）。

所以"排除第 1 条"目前的依据是**"最干净的配置下问题依旧存在"这个推理**，不是两种方式的直接对比。这一点我明确写进了文档，没有当成已经完成。

### 实验的可靠性

这次做了以前没有的严格自查，两项都通过：
- 动态做法的**起始选择和静态逐位完全相同**（13/13 全通过）；
- 缓存内容**逐项校验**（0 处不一致）。

所以上表那些差异**不是代码写错造成的**。

### 顺带的两个观察

- 静态 STD 在本机 A6000 上，解码比原始模型快 **1.40×**（87.9s → 62.8s）；此前在 A100 上只有 1.15–1.24×。这个差距值得单独查。
- 本机显卡被别人占用得非常厉害，跑一次要反复抢卡，这是目前实验推进慢的主要原因。

> **术语对照**：本文下面的章节里把"静态还有多少可改进空间"叫做 **headroom**（余量），"接受率提升"叫 **Δaccept**，"完全重建/局部更新"叫 **full rebuild / incremental refresh**。

---


## 0. 项目目标

- **静态 STD 复现**：Qwen2.5-VL-7B-Instruct 上的 Sparse-to-Dense 投机解码（稀疏 draft + 稠密 verify），已可用（见 `README.md` / `H100_REPRODUCTION.md`）。
- **动态路由研究线**：把静态 STD 中固定的 visual top-K 选择（S_0）升级为**每轮根据稠密 verify 的 visual attention 信号动态选择**（S_t = TopK(A_{t-1})），目标是在不改变正确性（greedy 与 AR token 完全一致）的前提下提升接受率（mean accept），从而加速 decode。

---

## 1. 时间线总览

| 日期 | 阶段 | 结论 |
|---|---|---|
| 2026-08-21 | Oracle Study Phase 1 | **GO** — visual attention 可预测下一块所需 visual KV |
| 2026-08-21 | Dynamic STD MVP（Fixed-K Previous Top-K） | **GO** — VDC accept +1.66，correctness 保持 |
| 2026-08-24 | Mismatch 根因因果验证 | prefill 已统一(Scheme B)；残余 5/10 根因是 batched verification |
| 2026-08-24 | Dynamic STD wall-clock 优化 | collector/refresh 已到极限，瓶颈是 GPU-bound verify forward（0.98x） |
| 2026-08-24 | Adaptive-K Phase 1（offline） | **NO-GO** — 只证明 budget 会变，未证明同 budget 下 acceptance 更好 |
| 2026-08-24 | Adaptive-γ Phase 1（offline cost model） | **GO（仅 proxy/optimistic）** — acceptance controller 估计 +15.3% |
| 2026-08-25 | Adaptive-γ runtime MVP + capped9 + fixed-q sweep | **CORRECTNESS NO-GO / FIXED-Q NO-GO** — 正确性门槛未过 |
| 2026-09-05 | Verification fallback 诊断 | ⚠️ 单样本上 fallback 反而引入 mismatch，且 verify ~2× |
| 2026-09-05 | ECNU Phase-8 部署（plan/design） | 事实已在 gpu23 跑通（见 §13） |
| 2026-09-09 | Dynamic STD-VG Lite on A100/Video-MME | **❌ 负结果** — static accept 0.823 > dynamic 0.731–0.777，decode 也更慢 |
| 2026-09-09 | HSD（嵌套投机）spike | **❌ 负结果** — 比 static STD 慢 2.76× |
| 2026-09-17 | 工程：A100 免密访问 + 消融驱动 | 已配好，见 `docs/a100-access.md`、`scripts/a100.sh` |
| 2026-09-18 | **L1 机制分析（首次在 Video-MME）** | **信号存在**：Previous recall +8.23pp；**H_headroom 成立**；**H3 否证** |
| 2026-09-18 | **L2 筛选 R0（full rebuild）** | **H_headroom 确认**（corr −0.74）；**H1 排除**；均值 −0.031 掩盖 ±0.19 |
| 2026-09-18 | L0 仪表 T1/T2/T3 | equal-S₀ **13/13 通过**、cache 一致性 **0 处不一致** |
| 待办 | L2 筛选 R1（incremental 直接对照） | **未完成** — 本机显卡争用；按用户要求暂停 |

---

## 2. Oracle Study Phase 1（2026-08-21）— GO

**目标**：判断「稠密 verification 的 visual attention 信号能否预测下一 speculative block 所需的 visual KV」，作为是否进入动态路由实现的门槛。

**方法**：20 样本（VideoDetailCaption 10 + MLVU 10，frame=128，615 轮验证，K=996），从 dense verification 前向中采集 per-layer visual-only softmax attention mass，计算 Static / Previous / Oracle 三种选择的 recall 与 accept-proxy。

**结果**：

| 指标 | Static | Previous | Oracle |
|---|---|---|---|
| Recall vs Oracle | 0.5678 | **0.7667** (+19.89pp) | 1.0 |
| Attention mass | 0.3704 | 0.4337 | 0.4665 |
| Accept proxy（mean accept） | 5.70 | 6.67 | **7.17** (+1.48) |

- Q1 静态漂移成立：Jaccard(S_0, S_t*) ~0.58 → 0.23（静态选择随 decode 漂移）。
- Q2 局部可预测成立：Jaccard(S_{t-1}*, S_t*) ~0.74。
- Q3 Oracle 提升成立：accept proxy +1.48 ≥ 0.5 阈值。

**结论**：**GO**。Previous-verify 信号显著优于静态且接近 Oracle，进入 Phase 2 动态路由实现。

**产物**：`scripts/analysis/{attention_trace.py, collect_traces.py, analyze_routing.py}`；`results/routing_traces/`、`results/routing_analysis/`。

---

## 3. Dynamic STD MVP（2026-08-21）— GO

**目标**：用最简单策略（Fixed-K Previous Verification Guided Top-K）实现动态路由，验证 accept 提升。

**实现**（不改 `std_qwen25vl.py` 静态基线）：

| 文件 | 职责 |
|---|---|
| `src/std_repro/dynamic_selection.py` | SelectionState / SelectionPolicy / StaticPolicy / PreviousVerifyTopKPolicy / RuntimeVerificationCollector（V1） |
| `src/std_repro/sparse_cache_refresh.py` | refresh_sparse_visual_kv / count_changed_tokens |
| `src/std_repro/dynamic_std_qwen25vl.py` | 动态主循环（复用 std_qwen25vl 辅助函数） |
| `scripts/benchmark_dynamic_std.py` | AR / Static / Dynamic 三方对比 |

**结果**（PreviousVerifyTopK vs Static，K 固定）：

| 数据集 | mean accept | decode |
|---|---|---|
| VDC 10（128f / 256tok） | 4.73 → 6.40（**+1.66**） | 0.97x |
| MLVU 10（128f / 64tok） | 5.93 → 6.34（+0.41） | 0.79x |

- **correctness**：Dynamic 与 Static 逐样本完全一致（0 disagree）；dense verifier 是独立 canonical cache（Scheme B），动态只改 accept 不改输出。
- refresh ~36ms/round，VDC 总 decode 12.56s。

**关键实现坑**：`compact_sparse_prompt_cache` 是**绝对位置排序混合布局**（`unique(sorted(non_visual ∪ topk))`，visual 与 non-visual 交错），不是 `[non_visual | visual]` 分段。refresh 必须按此布局从 dense cache 重建 compact prompt 段（等价 re-compact），只动 prompt 段、不动 generated 段。初版误按分段布局写错位置导致 accept 暴跌（7.12 → 3.06）。

**结论**：**GO**。accept 提升成立，decode 未加速是 collector/refresh 开销未优化所致（见 §5），非算法信号问题。

---

## 4. Mismatch 根因因果验证（2026-08-24）

**目标**：以因果验证方式确认「SpecVLM 自定义 `scaled_dot_product_attention` 中的 FP16→BF16 QK、FP16 softmax 是否是 greedy exactness mismatch 的主要根因」，严格区分「代码可疑 / 实验证实根因 / 次要误差源」。

**方法**：forensic 实验在原始 5 个 VDC mismatch 样本上对比 AR / STD_parallel / STD_sequential / STD_math 四种路径。

**证据分级结论**：

1. **Confirmed**：SpecVLM 自定义 attention（`modeling_qwen2_5_vl.py:966`）在 `output_attentions=True` 时走 custom 路径，`False` 走 canonical flash，二者 prefill logits max diff ~1.5，KV 从 layer 1 起分叉。
2. **Confirmed（ablation）**：prefill 分叉中 **FP16→BF16 QK cast 占 ~96%**，FP16 softmax 占 ~0%。
3. **Confirmed**：当前 `std_qwen25vl.py` 已是 **Scheme B**（dense verifier 分支 `output_attentions=False`，line 697），custom attention **不再污染 verifier**。
4. **Confirmed（残余根因）**：当前 5/10 VDC mismatch 的残余根因是 **batched q_len=γ+1 verification 数值差异**：`verify_mode="sequential"`（q_len=1）5/5 与 AR 完全一致，`verify_mode="parallel"`（batched flash）5/5 分叉，`verify_attn_backend="math"`（batched fp32）4/5 仍分叉。分叉位置每样本固定（确定性）。

**五个明确问题的回答**：

1. AR 与 STD 在 prefill 完成时**曾经**产生数值分叉（custom vs canonical attention），但该路径已从 dense verifier 移除。
2. FP16→BF16 QK 是 prefill 分叉的**主要贡献源（~96%）**。
3. FP16 softmax **不是**主要贡献源（~0%）。
4. 统一 prefill 后**不恢复 exact**——因为残余 mismatch 来自 batched verification，非 prefill。
5. 统一 prefill 后，batched verification 与 sequential AR **仍存在 token-level mismatch**（5/10 样本）。

**最终结论**：当前 5/10 mismatch 的根因是 **batched q_len=γ+1 verification 的 reduction-order 数值差异**（~2-4e-02 logit 扰动翻转 near-tie logits），不是 SpecVLM 遗留 attention 路径，也不是 speculative verification state machine。唯一 bit-exact 路径是 `verify_mode="sequential"`。

**阶段性产物**：因果验证曾使用 `diagnose_*` / `forensic_*` / `verify_scheme_b.py` 等一次性脚本；结论固化后已于 2026-08-24 清理这些脚本。

---

## 5. Dynamic STD wall-clock 优化（2026-08-24）

**目标**：不引入新算法，优化 collector 与 sparse KV refresh 的 wall-clock 开销，把 Dynamic STD 从 ~0.97x 提到 >1x。

### Task 1 — Collector 优化：V3 fused collector（已实现）

- 新增 `VerificationCollectorV3`（`dynamic_selection.py`）：保留 V1 的 in-hook 全 query GEMM（与 verify 的 attention 内核同流重叠），去掉 per-layer `.cpu()`，改为 `end_verification` 时单次 `torch.stack(...).cpu()`。
- 接入 `collector_version="v3"`（`dynamic_std_qwen25vl.py`）。

| 指标 | V1 (mvp) | V3 (opt) |
|---|---|---|
| mean_accept | 6.40 | 6.40（**bit-identical**） |
| accept_rate | 0.719 | 0.719 |
| token_match | 5/10 | 5/10 |
| collect_time | 1432ms | **176ms（8.1×）** |
| decoding_time | 16.27s | 16.18s（**几乎不变**） |

**结论**：V3 把 collector 的 CPU 侧开销降 8 倍，accept/ranking 完全一致；但 wall-clock 几乎不动。原因是 decode 为 GPU-bound 的 dense verify forward，V1 的 per-layer `.cpu()` 同步本就藏在 GPU 忙碌期后，CPU 不是瓶颈。

### Task 2 — Sparse KV refresh：incremental 严格更差（保留 full rebuild）

用 `probe_optimize.py` 隔离 full vs incremental（同 collector 下对比）：

| sample | v3_full accept | v3_incr accept | 变化 |
|---|---|---|---|
| v_-6dz6tBH77I | 7.03 | 7.03 | 持平 |
| v_-D1gdv_gQyw | 6.56 | **6.34** | 下降 |
| v_-IMXSEIabMM | 6.56 | **5.95** | 下降 |

**根因**：`incremental_refresh_sparse_visual_kv` 原位覆盖 removed→added，产出**非排序** slot 顺序；full rebuild 产出规范 `sorted(non_visual ∪ topk)`。稀疏 draft 的 `is_causal=False` softmax 是 FP 求和顺序敏感的，slot 顺序不同 → 稀疏注意力数值微扰 → near-tie logits 翻转 → accept 下降。且 refresh 时间无一致加速。

**结论**：incremental refresh **既不等价（降 accept）又不加速**，弃用，默认 full rebuild。

### Task 3 — 回归（VDC 10 samples, 256 tokens, gamma=9, K+text=1024）

| config | match | mean_accept | accept_rate | decoding | speedup vs static |
|---|---|---|---|---|---|
| static | 5/10 | 4.73 | 0.531 | 15.87s | 1.000× |
| mvp（v1 + full） | 5/10 | 6.40 | 0.719 | 16.27s | 0.976× |
| opt（v3 + full） | 5/10 | 6.40 | 0.719 | 16.18s | **0.981×** |

### 最终结论

**Dynamic STD 无法仅靠 collector/refresh 优化把 decode 提到 >1×。**

- accept 大幅提升（4.73 → 6.40，+35%，意味着 verify 轮数少 ~26%），但 wall-clock 反而略降（0.981×）。
- 根因：**瓶颈是 GPU-bound 的 dense verify forward**。动态 routing 的收益被两类 GPU 开销抵消：
  1. collector 的 GEMM 与 verify 在**同一条 CUDA stream 上串行**（V3 只消除 CPU sync，无法消除 GPU GEMM 串行）；
  2. refresh 的 gather/scatter 是**强制 `torch.cuda.synchronize()` 的完全串行 pass**（~1290ms）。
- 这是预设的「acceptance 提升但 speed 不提升」场景，按指令不再加复杂度。

**正确性事实**：三种 STD 变体 match 都是 5/10 且完全一致；那 5 个 mismatch 是已知的 batched verification 数值问题，与 routing 无关。

---

## 6. 关键指标汇总

| 指标 | 值 |
|---|---|
| Oracle accept proxy 提升 | Static 5.70 → Oracle 7.17（+1.48） |
| Previous recall 提升 | +19.89pp（0.5678 → 0.7667） |
| Dynamic MVP accept 提升（VDC） | 4.73 → 6.40（+1.66） |
| Dynamic 正确性 | 与 Static 逐样本一致（0 disagree） |
| V3 collector 提速 | collect_time 1432 → 176ms（8.1×） |
| Dynamic 最终 decode speedup | 0.981×（未达 >1×） |
| 残余 mismatch 根因 | batched q_len=γ+1 verification（非 prefill / 非 state machine） |
| Adaptive-K offline | NO-GO（证据不足） |
| Adaptive-γ offline（估计） | acceptance +15.32%、cost_aware +63.57%（`optimistic_estimate`） |
| Adaptive-γ runtime | CORRECTNESS NO-GO（fixed vs adaptive 后缀不同） |
| Adaptive-γ capped9 padding 占比 | 0.71% → 35.26% |
| Adaptive-γ fixed-q sweep（q=10..14） | FIXED-Q NO-GO（minimal shared AR-exact q* = none） |
| **Dynamic VG-Lite accept（A100/Video-MME，3 样本）** | **static 0.823 → dynamic 0.731–0.777（劣化）** |
| **Dynamic VG-Lite decode（A100）** | static 32.6–34.4s → dynamic 37.9–41.8s（0.79–0.89×） |
| HSD spike vs static STD | 慢 2.76×（decode）／2.32×（inference） |

---

## 7. 代码 / 产物清单

**核心实现**

- `src/std_repro/dynamic_selection.py` — SelectionPolicy 接口；StaticPolicy；PreviousVerifyTopKPolicy；RuntimeVerificationCollector（V1）；VerificationCollectorV2（可配置 two/three-query 延迟）；VerificationCollectorV3（fused）；`attention_free_visual_topk`（无注意力 bootstrap）；`should_refresh_selection`（interval / hysteresis 调度）。
- `src/std_repro/dynamic_std_qwen25vl.py` — 动态 decode 主循环（`collector_version ∈ {v1,v2,v3}`、`refresh_mode ∈ {full,incremental}`、`query_mode ∈ {two,three}`、`bootstrap_mode ∈ {attention,attention_free}`、`selection_update_interval`、`min_selection_change_ratio`）。
- `src/std_repro/sparse_cache_refresh.py` — refresh_sparse_visual_kv（full rebuild）；incremental_refresh_sparse_visual_kv（带 `_std_visual_slot_map` 的 slot 追踪；性能/accept 仍不划算，默认不用）。
- `src/std_repro/verification_policy.py` — `positional_token_metrics` / `min_prediction_margin` / `needs_sequential_fallback`。
- `src/std_repro/hsd_spike.py` — HSD 嵌套投机可行性 spike（已判定负结果，见 §15）。
- `src/std_repro/sparse_verify_spike.py` — q_len>1 offset-causal 稀疏验证原语（未接入 benchmark，见 §16）。
- `src/std_repro/streaming_video.py` — PyAV 内存有界流式抽帧（A100 benchmark 已使用）。
- `src/std_repro/std_qwen25vl.py` — 静态 STD 基线（Scheme B；`verify_mode` / `verify_attn_backend` / `verify_fallback`）。
- `src/specvlm/models/modeling_qwen2_5_vl.py` — SpecVLM 自定义 attention（line 966）；`_std_trace_hook`（line 1065）。

**分析 / 基准脚本**

- `scripts/analysis/{attention_trace.py, collect_traces.py, analyze_routing.py}` — Oracle Study。
- `scripts/analysis/simulate_adaptive_k.py` — Adaptive-K offline 模拟。
- `scripts/benchmark_dynamic_std.py` — 本地 AR / Static / Dynamic 三列对比（VDC）。
- `scripts/benchmark_a100_dynamic.py` — A100/Video-MME 动态对比（streaming 抽帧、组件计时、配对 JSONL）。
- `scripts/summarize_a100_dynamic.py` — 由 JSONL 生成 `_summary.json` 与 `_report.md`。
- `scripts/benchmark_a100_hsd_spike.py` — HSD 四路对比。
- 一次性 `diagnose_*` / `forensic_*` / `probe_optimize.py` / `profile_std_breakdown.py` / `regress_std_correctness.py` / `verify_scheme_b.py` 已在结论固化后清理；核心 benchmark、trace collector 与实验结果保留。

**产出数据**

- `results/routing_analysis/`、`results/routing_analysis_mlvu/` — Oracle Study 分析 CSV。
- `results/dynamic_std_mvp/vdc10_3col_v3.jsonl` — 最终三列回归。
- `results/adaptive_k_offline/`、`results/adaptive_gamma_offline/`、`results/adaptive_gamma_runtime*/`、`results/adaptive_gamma_fixed_q_sweep*/` — offline 模拟与 γ 运行时/sweep。
- `results/correctness_fallback_a6000/` — fallback A/B 诊断。
- `results/a100_dynamic_20260909_*.jsonl|_summary.json|_report.md` — VG-Lite A100 全量配对结果。
- `results/a100_hsd_spike_report.md` — HSD 报告；原始 `a100_hsd_spike_20260909{,_l3}.jsonl` 已于 2026-09-17 从远端拉回本地 `results/`。
- 注：`results/` 已被 `.gitignore` 忽略，仅作本地/远端审计产物。

---

## 8. 未决问题 / 下一步

> **2026-09-17 更新**：第 2 条（Dynamic STD >1×）的前提已被 §14 的 A100 负结果动摇。**在 Video-MME 上 dynamic 连 acceptance 都低于 static**，因此当前第一优先级不是"如何加速"，而是"dynamic 的 accept 优势是否真实存在、在什么条件下存在"。
> **同日补充**：硬件分档（`docs/experiment-provenance.md`）显示正结果在 **A6000**、负结果在 **A100**，且两者间同时有 4 个变量不同；逐轮诊断又显示 **Δaccept 与 static 剩余空间单调相关**（H_headroom）。因此判读框架升级为 `docs/superpowers/plans/2026-09-17-dynamic-routing-thorough-analysis.md`（L0 仪表 → L1 离线机制 → L2 分层受控 → L3 wall-clock）。

1. **【最高优先级】在 Video-MME 上做机制分析（L1）**：VDC 正结果（4.73→6.40）与 A100/Video-MME 负结果（0.823→0.731–0.777）方向相反，且从未在 Video-MME 上做过 Oracle Study。先采 trace 算 Static/Previous/Oracle 的 recall，**按 static 剩余空间分箱**——这一步不改算法、无正确性风险，决定后续是否值得投入（§14、§16、`experiment-provenance.md`）。
2. **Dynamic STD >1× 的可行路径**（在 accept 优势被确认后才成立，均需另行决策）：
   - 降低 verify 的 q_len 或批量化（改 dense verification）；
   - 把 collector GEMM / refresh 放到独立 CUDA stream 与 verify 重叠；
   - 降低 visual KV 规模（降低 verify forward 本身）。
3. **`verify_fallback` 的正确性未定**：它被用作 A100 上 6/6 exact 的保证，但 §12 的单样本显示它可能**引入** mismatch 并带来 ~2× verify 开销。需 ≥10 样本 A/B 后才可继续依赖。
4. **残余 greedy mismatch**：根因是 batched q_len=γ+1 verification 的 reduction-order 数值差异。唯一 bit-exact 路径是 `verify_mode="sequential"`。
5. **`sparse_verify_spike`（q_len>1 稀疏验证）尚未接入任何评测**，是唯一可能同时改善中间验证成本与 HSD 结构性开销的现成原语（§16）。
6. **Adaptive-K / Adaptive-γ 已停在正确性门槛**：runtime Adaptive-γ 为 CORRECTNESS NO-GO，fixed-q sweep 也 NO-GO；重新开启需先解决 exactness。
7. **下一算法阶段**（若继续）：EMA / predictive routing（当前明确不做）。

---

## 9. Adaptive-K Phase 1：Offline Budget Simulation（2026-08-24）— NO-GO

**目标**：只用已有 verification trace 判断 feedback 是否足以调节 visual KV budget，并评估是否值得进入 runtime Adaptive-K；未运行新 decoding，未修改 decoder / sparse cache / correctness path。

**数据**：VideoDetailCaption 10 samples，128 frames，256 max tokens，gamma=9，共 456 verification rounds。已有 trace 全部来自 visual K=996（`K+text=1024`），因此只有该 recorded trajectory 的 acceptance 是实测值；其他 K 和 Adaptive-K acceptance 均只能是 attention-mass proxy。

**Controller K 分布**：

| controller | mean K | median | min / max | CV | change fraction |
|---|---:|---:|---:|---:|---:|
| Attention rho=0.80 | 5429.5 | 5629 | 996 / 6963 | 0.209 | 1.000 |
| Attention rho=0.90 | 7641.7 | 8192 | 996 / 8192 | 0.174 | 0.462 |
| Attention rho=0.95 | 7894.3 | 8192 | 996 / 8192 | 0.146 | 0.092 |
| Acceptance Feedback | 4608.2 | 4096 | 512 / 8192 | 0.693 | 0.567 |
| Hybrid rho=0.80 | 6892.0 | 8192 | 996 / 8192 | 0.260 | 0.527 |
| Hybrid rho=0.90 | 7779.7 | 8192 | 996 / 8192 | 0.164 | 0.294 |
| Hybrid rho=0.95 | 7924.9 | 8192 | 996 / 8192 | 0.143 | 0.085 |

**结果**：

1. **Q1 — budget 是否变化：YES（但高 rho 饱和）**。Attention rho=0.80 与 Acceptance Feedback 有明显逐轮变化；rho=0.95 的 Attention/Hybrid 很快饱和到 8192（median=8192、IQR=0），不构成有用的细粒度自适应。
2. **Q2 — 是否优于 fixed K：INSUFFICIENT EVIDENCE / NO-GO**。按既有 benchmark 的 sample-macro 口径，Recorded K=996 的实测 mean accepted length=4.732、accept rate=0.531。最有利的 proxy 是 Attention rho=0.80：mean K=5429.5、proxy accepted length=6.724，与 static K=8192 的 proxy 6.769 接近且 K 低 33.7%；但单 K trace 不能把这个反事实结果当作实测 Pareto 优势。
3. **Q3 — wall-clock：POSSIBLE BUT UNVERIFIED**。上述 proxy 暗示 draft visual attention cost 可能下降，但历史结果显示 collector/refresh overhead 和 dense verification 是关键瓶颈；本阶段不能声称 wall-clock speedup。

**最终结论**：**NO-GO**。当前只证明 verification feedback 能产生变化的 K budget，未证明 Adaptive-K 在相同平均 budget 下提高实测 acceptance。按阶段门槛停止 runtime Adaptive-K；只有在另行授权多 K measured trace / static sweep 后才应重新评估。

**产物**：`src/std_repro/adaptive_k_offline.py`、`scripts/analysis/simulate_adaptive_k.py`、`tests/test_adaptive_k_offline.py`、`tests/test_simulate_adaptive_k_cli.py`；本地报告位于 `results/adaptive_k_offline/`。

---

## 10. Adaptive-γ Phase 1：Offline Cost-Model Simulation（2026-08-24）— GO（仅 proxy）

**目标**：只用已有 gamma=9 trace 与组件计时，判断"逐轮调节 γ"是否值得进入 runtime；新 decoding 运行数 = 0。

**证据分级**：`measured`（记录的 γ=9 acceptance / 组件时间）、`replayed`（同轮内 γ≤9 前缀裁剪，不重建 generation context）、`proxy`（反事实轨迹、γ>9 外推、线性成本模型）、`optimistic_estimate`（零 controller overhead 的组件吞吐）。

**结果**：

| controller | mean γ | 接受长度 | accept rate | 估计 tokens/s | 相对效率 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| fixed γ=9 | 9.000 | 4.732 | 0.520 | 17.53 | +0.00% | measured |
| acceptance（primary） | 6.561 | 3.910 | 0.579 | 20.22 | +15.32% | proxy |
| acceptance（conservative） | 6.386 | 3.727 | 0.570 | 20.05 | +14.38% | proxy |
| cost_aware | 3.000 | 2.247 | 0.743 | 28.67 | +63.57% | replayed |

- acceptance controller 是**唯一真正逐轮变化**的 controller（switch freq 0.368，无反转）；cost_aware 每轮都选 γ=3，等价于另一个 fixed-depth，其 +63.57% 不构成"动态"证据。
- break-even controller overhead 估计：acceptance 0.0362 s/round；cost_aware 0.0715 s/round。
- γ>9 的收益完全依赖未测的 tail `proxy`。

**最终结论**：**GO（离线研究决策）**，但所有吞吐均为 `optimistic_estimate`，**不构成 runtime speedup 声明**。GO 只授权另行设计 runtime 实验。

**产物**：`src/std_repro/adaptive_k_offline.py` 的 γ 扩展、`results/adaptive_gamma_offline/`（含 figure1/figure2）。

---

## 11. Adaptive-γ Runtime（2026-08-25）— CORRECTNESS NO-GO

在 RTX A6000 / 4090 上跑 fixed γ=9 与 frozen Acceptance Adaptive γ∈{3,5,7,9}（及 capped9、fixed-q10）的配对比较。

| 实验 | 配置 | 结果 |
|---|---|---|
| `adaptive_gamma_runtime` | γ∈{3..13}, q_len=γ+1 | **CORRECTNESS NO-GO** — fixed 与 adaptive 后缀不同 |
| `adaptive_gamma_runtime_capped9` | γ∈{3,5,7,9}, physical q_len=10 | **CORRECTNESS NO-GO** — AR/Static/Adaptive exact gate FAIL |
| `adaptive_gamma_fixed_q_sweep` | q=10..14，2 个已知 mismatch 样本 | **FIXED-Q NO-GO** — 无任何 q 能对所有已知样本 AR-exact（minimal shared q* = none） |

关键观察：
- γ 抖动导致 physical q_len 频繁变化，**padding 槽位从 0.71% 暴涨到 35.26%**（capped9，201/570），验证工作量被严重浪费。
- capped9 上 adaptive 的 accept rate 反而更高（0.639 vs 0.572），但 mean accepted length 更低（3.51 vs 5.12），rounds 更多（57 vs 42）——**γ 变小并没有转化为更快的 wall-clock**。
- 所有计时都标注为"仅审计、不可用于性能结论"（correctness gate 未过）。

**最终结论**：Adaptive-γ 在**正确性门槛**上终止，从未进入可比较的性能阶段。

**产物**：`results/adaptive_gamma_runtime/`、`results/adaptive_gamma_runtime_capped9/`、`results/adaptive_gamma_fixed_q_sweep*/`。

---

## 12. Verification Fallback 诊断（2026-09-05）— 存疑

**背景**：§4 已定位残余 mismatch 来自 batched verification。`src/std_repro/verification_policy.py` 提供 `sequential_on_low_margin` 等回退策略，在 near-tie 时改走 q_len=1 的精确路径。9 月的 A100 实验把 `verify_fallback=sequential_on_low_margin` 作为 6/6 exact 的保证。

**A6000 单样本（VDC `v_-6dz6tBH77I`, 96 帧, 256 tokens）对照**：

| 配置 | static_token_match | mvp/opt_token_match | static fallback_count | static verify 时间 |
|---|---|---:|---:|---:|
| `verify_fallback=none` | **true** | true | 0 | 3.41s |
| `verify_fallback=sequential_on_low_margin` | **false** | true | 11 | 6.16s |

- 打开 fallback 后该样本的 static 反而 **mismatch**（true→false），且 verify 时间近乎翻倍。
- Dynamic（mvp/opt）在两种配置下均为 match。

**结论（存疑，非定论）**：样本量=1，不能据此否定 fallback；但说明"回退即保正确"的假设**尚未被验证**，且回退有显著 verify 开销。继续在 A100 上默认开启 fallback 之前，必须先做 ≥10 样本 A/B（见 §16）。

**产物**：`results/correctness_fallback_a6000/`（注意 `vdc1_margin01.jsonl` 为空文件）、`tests/test_verification_fallback.py`。

---

## 13. ECNU Phase-8 部署（2026-09-05）— 事实完成

**目标**：在 ECNU Phase-8 集群 `login2` 上准备代码、隔离 conda 环境、Qwen2.5-VL-7B-Instruct 与 Video-MME chunk 01，为后续 A100 实验做准备。

**设计要点**（`docs/superpowers/specs/2026-09-05-ecnu-std-deployment-design.md`）：

- 代码 `/public/home/xlwang/mcy/Project/STD`；环境 `/public/home/xlwang/mcy/conda_envs/specvlm`（Python 3.10 / torch 2.6.0+cu124 / transformers 4.48.0）；资产 `/public/home/xlwang/mcy/STD_assets/{models,datasets,cache,results}`。
- 只下载 Video-MME chunk 01；不在部署阶段连 GPU 节点。

**状态**：**事实已完成**。§14 所有 A100 运行的 manifest 均指向 `STD_assets` 下的模型与数据路径，证明环境/模型/数据已跑通；但 `.spec-workflow/` 的 plan 复选框与 approvals 未更新，**部署记录与代码库脱节**。执行脚本/记录待补。

**访问方式**：连接拓扑、远端路径、GPU 规约与故障排查已固化为 `docs/a100-access.md`，配套可复用脚本 `scripts/a100.sh`（`status` / `login` / `gpu` / `sync` / `ablation` / `e2e` / `push` / `pull` / `close`）。**已配置免密**：`~/.ssh/config` 提供别名 `a100`（login2 = `59.78.189.133:2323`）与 `a100-gpu`（gpu23 = `10.11.200.23`，经 ProxyJump），公钥 `~/.ssh/id_ed25519` 已装入集群；`/public/home` 为共享家目录，故 login2 与 gpu23 同时免密。日常流程为 `a100.sh e2e <stage>`（同步代码 → 查卡 → 跑消融 → 拉回报告）。gpu23 上 **GPU0 常驻 vLLM，只用 GPU1**。两个产物均不含任何凭证。


---

## 14. Dynamic STD-VG Lite on A100 / Video-MME（2026-09-09）— ❌ 负结果 ★

**目标**：按 `docs/superpowers/plans/2026-09-09-dynamic-std-vg-lite.md` 实现并评测 verifier-guided dynamic STD 的低风险部分（two-query collector、selection update interval/hysteresis、attention-free bootstrap），在保持 exact 的前提下寻找 wall-clock 收益。

**实现**（commit 前为工作区改动，见 §16 的入库说明）：

- `verification_query_positions(..., mode="two"|"three")`：two = 首 query + bonus query；three = 旧控制组（含 lookahead）。
- `should_refresh_selection(round_id, changed_ratio, interval, min_changed_ratio)`：控制更新频率与滞回。
- `attention_free_visual_topk(keys, values, ...)`：无注意力 bootstrap（windowed centroid residual + value norm + coverage）。
- 主循环新增 `selection_update_interval` / `min_selection_change_ratio` / `query_mode` / `bootstrap_mode`，并把默认值从 **v1+full 改为 v2+incremental**。
- 修复 refresh 子计时重复计数（报告中的 `refresh_timing_note`）。

**结果**（gpu23 物理 GPU1，Video-MME 3 样本 seed-42，128 帧，128/256 tokens，repeats=2，**从 `*_profile.jsonl` 的 `measure` 阶段重新聚合**）：

| 实验 | static accept | dynamic accept | static decode | dynamic decode | vs static | exact |
|---|---:|---:|---:|---:|---:|---:|
| q2_profile3（query=two, interval=1） | **0.823** | 0.762 | 33.69s | 40.75s | 0.827× | 6/6 |
| interval4_profile（i=4, attention） | **0.823** | 0.777 | 33.49s | 37.85s | 0.885× | 6/6 |
| bootstrap_windowed（attention_free） | **0.823** | 0.731 | 32.95s | 40.78s | 0.808× | 6/6 |
| v2_i1_final | **0.823** | 0.762 | 34.39s | 41.78s | 0.823× | 6/6 |
| v2_i1_t256 | **0.857** | 0.797 | 63.20s | 79.61s | 0.794× | 6/6 |
| f128_t128_r2（v1/v3） | **0.823** | 0.751 | 32.57s | 47.8/48.3s | 0.68/0.68× | 6/6 |

**⚠️ 硬件归属（2026-09-17 补记，详见 `docs/experiment-provenance.md`）**：本节的**负结果全部来自 A100/gpu23**；而 §3 的 **dynamic 正结果（+1.66）来自 A6000/4090 本机**。两者之间同时变了 **4 个变量**（硬件 A6000→A100、数据集 VDC→Video-MME、生成长度 256→128、collector/refresh `v1+full`→`v2+incremental`），因此「dynamic 行不行」**至今没有被干净地回答过一次**。A100 运行所用代码与当前代码逐文件 sha256 一致（6/6 SAME），故负结果**不是**旧版缺 slot-map 修复所致。

**核心发现**：

1. **static 的 acceptance 全面高于所有 dynamic 变体**，decode 也全面更慢。这与 §3 的 VDC 结果（static 0.531 → dynamic 0.719）**方向相反**。
2. 6/6 exact 达成，但依赖 `verify_fallback=sequential_on_low_margin`——而 §12 显示该 fallback 本身正确性存疑。
3. 组件计时显示 dynamic 的额外开销落在 **draft+verify**（static draft 10.15s/verify 5.51s vs dynamic 10.91s/5.71s），而非 selection/collect（~0.027s）。
4. 该数据集的 **static accept 本身就很高（0.823）**，与 VDC（0.531）不在同一区间——这可能是"动态收益消失"的根因，也可能是 3 样本方差。

**⚠️ 逐轮诊断（2026-09-17 新提取，此前只报均值）**：从 `dynamic_stats.per_round` 还原出每轮真实行为：

| 样本 | rounds | static acc | dyn acc | **Δ** | Jaccard(S_t,S_{t+1}) | 每轮替换 token（K=996） | update_applied |
|---|---:|---:|---:|---:|---:|---:|---:|
| 050-1 | 19 | 0.623 | 0.667 | **+0.044** | 0.536 | 301.7（30%） | 19/19 |
| 496-3 | 17 | 0.864 | 0.762 | **−0.102** | 0.510 | 320.3（32%） | 17/17 |
| 717-1 | 15 | 0.983 | 0.857 | **−0.126** | 0.524 | 298.4（30%） | 15/15 |

- interval=1 时**确实每轮都在改选择**，每轮替换约 **30%** 的 visual KV，相邻轮 Jaccard≈0.52。
- interval=4 时 `update_applied` 仅 **5/23、3/14、3/15**——节流强度远超预期。
- **Δaccept 与 static 的"剩余空间"单调相关**（VDC 0.531→+0.188；050-1 0.623→+0.044；496-3 0.864→−0.102；717-1 0.983→−0.126）。
- **n=3 的均值 −0.061 完全掩盖了这个分化。** 这是 §16.2 新方案的核心命题 **H_headroom**。

**候选解释（尚未被实验区分）**：
- **(H_headroom，主假设)** static 接近最优时，每轮 30% 的 KV 替换是纯噪声；static 远离最优时才有增益；
- (H1) **incremental refresh** 的 slot 顺序回归（§5 Task 2 已在 VDC 证明会降 accept），而 A100 默认开 incremental；
- (H2) **attention-free bootstrap** 产生劣于 static 的 S_0（attention_free 变体确实最差）；
- (H3) **v2 two-query collector** 的 query 选择劣于 three；
- (H4) Video-MME/CoT/128 帧下 `TopK(A_{t-1})` 信号本身失效（真实负结果）；
- (H5) 3 样本方差。

**结论**：在 `limit=3` 上 **declared negative**。**但判读框架已由 `docs/superpowers/plans/2026-09-17-dynamic-routing-thorough-analysis.md` 取代**：先过仪表不变量（L0）、先在 Video-MME 上做机制分析（L1），再做分层受控消融（L2），且**禁止用未分层均值下结论**。

**产物**：`results/a100_dynamic_20260909_*.jsonl|_summary.json|_report.md`（`results/` 已被 gitignore，未入库）。

---

## 15. HSD（嵌套投机）Spike（2026-09-09）— ❌ 负结果

**目标**：验证 D（3B draft）→ Sparse（7B compact cache 逐个打分）→ Dense（7B canonical batched verify）的嵌套级联是否可用且更快。

**配置**：Qwen2.5-VL-7B target + Qwen2.5-VL-3B draft，Video-MME `050-1`/`496-3`/`717-1`，32 帧，64 tokens，repeats=2。

| method | decode（s, mean） | inference（s, mean） | exact vs AR |
|---|---:|---:|---:|
| AR（7B dense） | 1.640 | 3.508 | reference |
| static STD（7B sparse→dense） | **1.897** | 3.822 | 6/6 |
| small dense（3B→7B dense） | 5.967 | 9.509 | 6/6 |
| HSD spike（3B→7B sparse→7B dense） | 5.227 | 8.874 | 6/6 |

- HSD 比 static STD **慢 2.76×（decode）/ 2.32×（inference）**，比 AR 慢 3.19× / 2.53×。
- 全部 6/6 exact，说明嵌套验证路径**功能正确**，但**结构上不划算**：3B draft + 中间稀疏验证 = 两次额外模型前向；当前稀疏验证仍有 per-block launch/cache 管理开销。

**结论**：负结果，不建议继续此形态。若要做快，需要真正更便宜的中间验证器（独立 slim 模型/子网络）和/或重叠调度，而不是简单插入稀疏注意力。

**产物**：`src/std_repro/hsd_spike.py`、`scripts/benchmark_a100_hsd_spike.py`、`tests/test_hsd_spike*.py`、`results/a100_hsd_spike_report.md`（原始 `a100_hsd_spike_20260909{,_l3}.jsonl` 已拉回本地 `results/`）。

---

## 16. 在研原语与彻底分析方案（2026-09-17）

### 16.1 在研原语（未接入评测）

- `src/std_repro/sparse_verify_spike.py` — q_len>1 的 offset-causal 稀疏验证（整块验证，避免未来泄漏）。**只有单元测试，未接入任何 benchmark。** 这是唯一可能同时改善"中间验证成本"与 HSD 结构性开销的现成 primitive，优先级高。
- `src/std_repro/streaming_video.py` — PyAV 流式抽帧，内存有界（替代 torchvision 全量解码），已被 A100 benchmark 使用。

### 16.2 消融复跑方案 → 已升级为「彻底分析方案」

**动机**：§14 的负结果可能是实现回归（H1–H3）、也可能是真实负结果（H4），而 `limit=3` 无法区分（H5）；更关键的是 §14 新提取的逐轮诊断显示 **Δaccept 与 static 剩余空间单调相关**（H_headroom），而均值掩盖了它。

**当前方案（取代旧消融矩阵的判读框架）**：
`docs/superpowers/plans/2026-09-17-dynamic-routing-thorough-analysis.md`

核心变化：

1. **问题重构**：不再问"dynamic 有没有用"（二元、不可证伪），而是问"**在什么可测条件下有用，能否先验预测**"——产出 `Δaccept ≈ f(特征)` 与启用判据。
2. **分层执行**：**L0 仪表不变量**（选择/cache 一致性、equal-S_0、更新比例、计时修复、禁止默认值漂移）→ **L1 离线机制分析**（在 **Video-MME** 上重做 Oracle Study——此前只在 VDC+MLVU 做过；并用同一份 attention 做 all/two/three-query 估计质量消融）→ **L2 分层受控在线消融**（等 S_0、单变量、n≥20、3 seeds、逐样本配对 + 按 `static_accept` 分箱）→ **L3 wall-clock**。
3. **预注册判读**：给出 5 条观察→结论→行动的对照表，禁止事后解释；**禁止用未分层均值下结论**。
4. **执行状态（2026-09-17）**：方案已定稿并**锁定 A100/gpu23 物理 GPU1**，但**当前两张卡均被占用**（GPU0 vLLM、GPU1 他人作业），故暂停执行、等待放卡。
5. **执行前置（阶段 A）**：G1–G4、G6 **已完成并入库**（Video-MME 采集、per-query trace、分箱分析、collector 消融、分层汇总）。G5（L0 仪表 T1–T5）留待 L2 前完成。
6. **L1 已执行并通过门槛（2026-09-18）**：因 A100 两张卡均被占用，按方案 §8 把 L1 提前到本机 A6000/GPU7。Video-MME 20 样本 / **442 轮**：`recall(Previous)−recall(Static) = **+8.23pp**`（≥+5pp），且**分箱单调**（低 headroom **+14.31pp** → 高 headroom **+5.95pp**，pearson −0.255）→ **H_headroom CONSISTENT，G-L1 通过，PROCEED to L2**。collector 消融显示 **H3 被否证**（two-query churn 0.566 vs all 0.585；churn 是信号固有漂移），但 two-query 保真度仅 **0.815**。详见方案 §10。
7. **L2 消融范围已收窄**：只需覆盖 **H1（refresh 机制）+ headroom 分层**；H2/H3 已弱化。仍按计划在 **A100** 上做（墙钟 + 正确性）。

旧的 `docs/superpowers/plans/2026-09-17-dynamic-std-ablation-rerun.md` + `scripts/run_a100_ablation.sh` 仍可用作 L2 的执行载体，但**判读以新方案为准**。

**硬件分档记录**：`docs/experiment-provenance.md`（A6000 vs A100 逐条结果 + 逐轮诊断 + 7 个混淆）。

### 16.3 工程状态（2026-09-17）

- §14/§15 的实现此前**未入库**；本次已把核心代码、脚本与测试提交（见 git log）。
- `.spec-workflow/`（第三方 spec 工具模板，644 行样板）已加入 `.gitignore`，不入库。
- 本地 `python3` 为 3.8，`tests/test_hsd_spike.py` 会因 `src/specvlm/models/modeling_rope_utils.py:97` 的 PEP 585 `tuple[...]` 报 3 个失败；项目 pin 的是 Python 3.10，**非代码缺陷**。带 `PYTHONPATH=src` 时其余 58 passed / 6 skipped。
- **A100 免密访问**已配好（`ssh a100` / `ssh a100-gpu`），见 `docs/a100-access.md` 与 `scripts/a100.sh`。

---

## 17. 动态路由：2026-09-18 的机制结论（L1 + L2-R0）

> 大白话解释见文档开头的「先看这个」。本节只放数据和判据。
> 完整方案与判读规则：`docs/superpowers/plans/2026-09-17-dynamic-routing-thorough-analysis.md`（§10 = L1，§11 = R0）。
> 硬件分档：`docs/experiment-provenance.md`。

### 17.1 L1 — 首次在 Video-MME 上做 Oracle Study

20 样本（seed-42，与 A100 实验同序）、128 帧、256 tokens、γ=9、K+text=1024、CoT，共 **442 轮验证**。

| selector | mean recall | Δ vs Static | attention mass |
|---|---:|---:|---:|
| Static | 0.6465 | — | 0.3713 |
| **Previous** | **0.7288** | **+8.23 pp** | 0.3920 |
| EMA λ=0.5 | 0.7294 | +8.29 pp | 0.3942 |
| Oracle | 1.0000 | — | 0.4311 |

- **信号存在**：`TopK(A_{t-1})` 在 Video-MME 上确实比静态选择更接近 oracle。→ **H4（该数据集无信号）否证。**

**按每轮实测 static 接受长度分箱**（等分，低余量在前）：

| bin | n | accept | static recall | prev recall | Δ |
|---|---:|---|---:|---:|---:|
| 0（低余量） | 147 | 0–8 | 0.5930 | 0.7361 | **+14.31 pp** |
| 1 | 147 | 8–9 | 0.6484 | 0.7345 | +8.61 pp |
| 2（高余量） | 148 | 9 | 0.6539 | 0.7134 | **+5.95 pp** |

pearson **−0.255**、spearman **−0.258** → **H_headroom CONSISTENT**。

**collector 估计质量消融**（同一份 attention，比较不同取位置方案）：

| estimator | fidelity vs all | predictive | churn Jaccard |
|---|---:|---:|---:|
| all（参照） | 1.0000 | 0.7302 | 0.5846 |
| **two** | **0.8153** | 0.7004 | 0.5660 |
| three | 0.8242 | 0.7081 | 0.5726 |

- two-query 保真度 0.815（损失约 18%），但 **churn 几乎不变**（0.585 vs 0.566）→ **H3（子采样是 churn 来源）否证**；churn 是 Video-MME 上 attention 信号本身的漂移。

### 17.2 L2 筛选 R0 — full rebuild（最干净配置）

10 样本、128 帧、128 tokens、`refresh=full`、`query=three`、`bootstrap=attention`、`interval=1`、`min_change=0.0`、`fallback=none`，强制 `--assert-equal-s0 --assert-consistency`。

**仪表不变量（G5）首次实证通过**：`equal_s0` **13/13 True**；`consistency_mismatches` **全 0**；更新 **16–19/全部轮次**；`token_equal` **8/10**（fallback 关闭下）。

| 样本 | static | dynamic | Δ |
|---|---:|---:|---:|
| 754-1 | 0.507 | 0.698 | **+0.191** |
| 154-3 | 0.541 | 0.647 | **+0.106** |
| 050-1 | 0.663 | 0.712 | +0.049 |
| 496-3 | 0.796 | 0.796 | 0.000 |
| 445-2 | 0.831 | 0.647 | **−0.184** |
| 102-2 | 0.850 | 0.688 | −0.162 |
| 504-2 | 0.966 | 0.877 | −0.089 |
| 647-1 | 0.966 | 0.935 | −0.031 |
| 717-1 | 0.983 | 0.796 | **−0.187** |
| 599-1 | 1.000 | 1.000 | 0.000 |

均值 Δ = **−0.0308**、中位数 −0.0157、范围 **[−0.187, +0.191]**、**corr(static_accept, Δ) = −0.740**、低余量组 **+0.115**。

**结论**：
1. **H_headroom 在 full rebuild 下依然成立** → **H1（incremental slot 顺序是负结果主因）排除**。
2. 均值第三次掩盖 ±0.19 的分化 → 判读必须分层（`summarize_a100_dynamic.py` 已强制输出配对 + 分箱）。
3. 附带：本机 A6000 上 **static vs AR decode = 62.8s vs 87.9s ≈ 1.40×**（A100 上为 1.15–1.24×），差异待查。

### 17.3 未完成 / 未决

| 项 | 状态 |
|---|---|
| **R1（incremental 直接对照）** | **未跑**。H1 的排除目前基于"R0 下问题依旧"的推理，**非直接对照**。本机显卡全部被争用（无卡同时满足空闲 + 34.3 GiB 显存），按用户要求暂停 |
| G5 的 T4（计时归零） | 已核实为**误判**：来自 warmup 行，measure 阶段计时正常，无需修复 |
| G5 的 T5（显式配置） | 已由 `run_a100_ablation.sh` 强制传入断言开关 |
| A6000 vs A100 的 static 加速差异 | 新观察到 1.40× vs 1.15–1.24×，未归因 |
| DSPark r3 | 已完成：在线 `spec_accept_length` 1.6936（基线 1.4209，+19.2%），但**仍未测墙钟** |

### 17.4 本次新增的代码与脚本

- `scripts/analysis/analyze_collector_ablation.py`（新）— all/two/three-query 估计质量与 churn 消融。
- `scripts/analysis/collect_traces.py` — 支持 Video-MME + `--record-per-query`。
- `scripts/analysis/attention_trace.py` — `record_per_query`（verification 轮保留 query 维；prefill 仍求和）。
- `scripts/analysis/analyze_routing.py` — 按 static accept 分箱 + G-L1 门槛。
- `src/std_repro/sparse_cache_refresh.py` — `verify_sparse_visual_consistency`（T1，**布局无关**）。
- `src/std_repro/dynamic_std_qwen25vl.py` — T1 接入 + T3 更新比例 + `pending_lengths`。
- `scripts/benchmark_a100_dynamic.py` — `--assert-equal-s0` / `--assert-consistency` / `--allow-shared-gpu` / `--min-free-gib` / `timing_valid`。
- `scripts/summarize_a100_dynamic.py` — 逐样本配对 + 分箱 + `timing_valid` 时 speedup 渲染为 `n/a`。
- 测试：新增 40+ 用例（`test_attention_trace_per_query.py`、`test_analyze_routing_headroom.py`、`test_analyze_collector_ablation.py`、`test_summarize_headroom.py`、`test_sparse_cache_refresh.py` 扩充）。全量 **103 passed / 6 skipped**（3 个失败为本地 Python 3.8 的 PEP 585 既有问题）。


