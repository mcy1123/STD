# STD 项目进展文档

> 本文档记录 SpecVLM/STD 仓库（复现 *Sparse-to-Dense: A Free Lunch for Lossless Acceleration of Video Understanding in LLMs*）的完整研究进展，聚焦 **Verification-Guided Dynamic Routing（验证引导的动态视觉路由）** 这条主线，并附静态 STD 复现背景。

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

---

## 7. 代码 / 产物清单

**核心实现**

- `src/std_repro/dynamic_selection.py` — SelectionPolicy 接口；StaticPolicy；PreviousVerifyTopKPolicy；RuntimeVerificationCollector（V1）；VerificationCollectorV2（3-query 延迟）；VerificationCollectorV3（fused）。
- `src/std_repro/dynamic_std_qwen25vl.py` — 动态 decode 主循环（`collector_version ∈ {v1,v2,v3}`，`refresh_mode ∈ {full,incremental}`）。
- `src/std_repro/sparse_cache_refresh.py` — refresh_sparse_visual_kv（full rebuild）；incremental_refresh_sparse_visual_kv（弃用）。
- `src/std_repro/std_qwen25vl.py` — 静态 STD 基线（Scheme B；`verify_mode` / `verify_attn_backend` / `verify_fallback`）。
- `src/specvlm/models/modeling_qwen2_5_vl.py` — SpecVLM 自定义 attention（line 966）；`_std_trace_hook`（line 1065）。

**分析 / 基准脚本**

- `scripts/analysis/{attention_trace.py, collect_traces.py, analyze_routing.py}` — Oracle Study。
- `scripts/benchmark_dynamic_std.py` — AR / Static / Dynamic 三列对比。
- 一次性 `diagnose_*` / `forensic_*` / `probe_optimize.py` / `profile_std_breakdown.py` / `regress_std_correctness.py` / `verify_scheme_b.py` 已在结论固化后清理；核心 benchmark、trace collector 与实验结果保留。

**产出数据**

- `results/routing_analysis/`、`results/routing_analysis_mlvu/` — Oracle Study 分析 CSV。
- `results/dynamic_std_mvp/vdc10_3col_v3.jsonl` — 最终三列回归。
- `results/dynamic_std_mvp/vdc10_3col.jsonl`、`vdc10.jsonl`、`smoke_mlvu*.jsonl` — 早期基准。

---

## 8. 未决问题 / 下一步

1. **Dynamic STD >1× 的可行路径**（均超出「不做新算法」范围，需另行决策）：
   - 降低 verify 的 q_len 或批量化（改 dense verification）；
   - 把 collector GEMM / refresh 放到独立 CUDA stream 与 verify 重叠；
   - 降低 visual KV 规模（降低 verify forward 本身）。
2. **残余 5/10 greedy mismatch**：根因已定位为 batched q_len=γ+1 verification 的 reduction-order 数值差异。若要 token-level exact，需 `verify_mode="sequential"`（最慢）或在 near-tie 时回退（`verify_margin_threshold` / `verify_fallback`，`std_qwen25vl.py` 已备参数）。
3. **下一算法阶段**（若继续）：EMA / predictive routing（当前明确不做）。

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
