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

### 这件事已经收尾了（2026-09-19 补记）

上表是**完全重建**（R0）的结果。**对照配置 R1（局部更新）其实在 09-18 14:43 就跑完了**，只是当时没有回写文档——数据一直在 `results/l2_screening/R1_incr_three_att.jsonl`（112 行完整、`R1_EXIT=0`、`timing_valid=true`）。

补算之后：**局部更新和完全重建的结果几乎逐样本相同**——10 个样本里 **7 个完全一致**，其余差异 ≤0.018；平均 **−0.0305 vs −0.0308**，相关性 **−0.728 vs −0.740**，低余量组都是 **+0.115**。所以"排除第 1 条"现在是**两种方式的直接对照**，不再只是推理。

附带一个好消息：**局部更新每轮只花 22 毫秒，是完全重建（71 毫秒）的三分之一**，而且没有长尾（最慢一轮 29 毫秒 vs 598 毫秒）。它本来就是代码里的默认（`--refresh-mode` 默认 `incremental`），**这次终于有了数据支撑**（此前 §5 Task 2 在 VDC 上得出的是相反结论）。

但**端到端仍然没赢**：static 328 毫秒/轮 vs dynamic 403 毫秒/轮（1.22× 慢）。**搬运这条腿基本解决了，剩下的过路费里最大的一块变成了"没被计时器覆盖的 host 侧开销"（约 34 毫秒/轮）。**

### 实验的可靠性

这次做了以前没有的严格自查，两项都通过：
- 动态做法的**起始选择和静态逐位完全相同**（13/13 全通过）；
- 缓存内容**逐项校验**（0 处不一致）。

所以上表那些差异**不是代码写错造成的**。

### 顺带的两个观察

- 静态 STD 在本机 A6000 上，解码比原始模型快 **1.40×**（87.9s → 62.8s）；此前在 A100 上只有 1.15–1.24×。这个差距值得单独查。
- 本机显卡被别人占用得非常厉害，跑一次要反复抢卡，这是目前实验推进慢的主要原因。

### 接下来打算做什么（两条腿一起走）

**先看一张账单**——动态做法每轮要多花 82 毫秒，占一轮总时间的 **22%**：

| 花的什么钱 | 每轮 | 说明 |
|---|---:|---|
| **打分**（读注意力、算打分表） | **11 ms** | **很便宜**，只占 1.9% |
| **搬运**（把新挑的画面块写进缓存） | **71 ms** | **主要开销**，占 11.7% |

所以：**打分很便宜，贵的是"搬运"。** 而这次多花的 38 秒里，**几乎全是纯开销，没有换回任何速度**。

**第一条腿：把"搬运"做便宜。** 现在搬一次要 71 毫秒，但只搬了 57 MB——折算下来速度只有 **0.8 GB/s**，比显卡真实带宽（~2000 GB/s）**慢了两千多倍**。说明瓶颈不是"搬得多"，而是"**拆成几百个小动作 + 每次都要强制等 GPU 做完**"。三级做法，从便宜到贵：

1. **只改配置**：隔几轮才换一次、差别不够大就不换（这次为了做最干净的对照，我把这两个闸门**全关了**，跑的是最贵配置）——**不用写代码**；
2. **去掉强制等待、把零碎动作合并**；
3. **彻底改**：不搬数据，只改"看哪几块"的索引——要搬的东西立刻少 **128 倍**。

**第二条腿：把"怎么挑画面块"做聪明一点。** 按顺序试：

1. **用更全的信息打分**——现在为了省时间只读 2 个位置的注意力，白白丢了 **18%** 的准确度；而打分总共才占 1.9%，改回全读几乎不增加成本；
2. **只在静态表现差的时候才开动态**；
3. **平滑更新**，不要每轮猛换 30%。

**一个必须记住的上限**：就算"挑画面块"挑到完美（用理想答案），也只能省大约 **16% 的轮数**。所以这是个**中等收益**的活——必须"挑得更准"和"换得更便宜"两条腿一起走，单独做哪一条都很难看到整体变快。

### 2026-09-19 更新：这两条腿都试过了，结果有点出乎意料

在 A100 上把两个"开关"分别打开、再一起打开（10 样本 × 4 配置，各约 51 分钟）：

| 试的东西 | 单独打开的效果 |
|---|---|
| **少换一点**（隔 4 轮才换一次 + 差别不够大就不换） | **+0.031 ~ +0.045（明显变好）**，而且搬运降到约 **1/5** |
| **打分读全一点**（2 个位置 → 全部位置） | **−0.003（基本没用）** |
| **两个一起打开** | **Δ ≈ 0（从"亏"变成"不亏"）**，而且轮数最少 |

两个反直觉的结论：

1. **"少换一点"比"打得更准"值钱得多。** 把打分读全（准确度从 0.815 提到 ~1.0）几乎不影响接受率——说明换来换去的噪声主要来自**信号本身在漂移**，不是"读得不够准"。所以第 2 条腿的第 1 项（读全）不是重点。
2. **两个开关有正交互作用。** 打分更全之后，"这一轮值不值得换"判断更准，节流才敢跳。单独用哪一个都看不到效果。

**但现在也只是"不亏"，还不是"赚"。** 剩下的问题很集中：节流之后**搬运只剩 16 毫秒/轮，可每轮的总过路费仍有上百毫秒**——也就是说**钱花在别处了**，而那部分目前连计时器都没覆盖。所以下一步不是继续调"挑画面块"，而是**先把每轮到底花在哪测清楚**。

另外一个必须记住的教训（这次差点踩坑）：**动态的基线必须和它跑在同一张卡上现测。** 同一份代码、同一批样本，换一张卡就能把单个样本的静态接受率挪动 **15 个百分点**，而**均值只动 0.6 个百分点**——拿别处的旧结果当基线，错误会藏得看不出来。

### 2026-09-21 更新：把生成长度从 128 提到 512，结果好了一截

之前**所有**加速比都是在 **128 个输出 token** 下测的，太短了。因为"动态"的收益要一轮一轮累积，而它的额外开销是**每轮都要付的固定成本**，所以生成越短，动态越吃亏。这次在本机 A6000 上一张空出来的卡（计时可信）把输出提到 **512** 重测（10 样本）：

| 只看解码 | 128 token | 512 token |
|---|---|---|
| 静态比原始模型快多少 | 1.55× | **1.71 – 1.87×** |
| 动态比静态 | 0.82×（慢 18%） | **0.91×（慢 9%）** |

两件事：

1. **静态复现的成绩明显变好**，已经贴近论文写的 up to 1.94×。所以之前的数字偏低，**有一部分是"生成太短"造成的**，不是实现有缺陷。
2. **动态的差距同步缩小，但仍然是负的。** 而且 512 下静态自己已经做得更好（接受率均值 0.82 → **0.91**），可改进的空间更小——这与"只有静态差的时候动态才有用"完全一致。

顺便确认了一件工程上的事：**"开跑时卡是空的"不等于"整段运行期间机器是安静的"。** 这次三档虽然都是卡空着时启动的，但"全打分"那一档明显被别的作业拖慢（同样配置，预处理阶段慢了 **30%**）。所以引用墙钟数时，除了看计时标记，还要看"预处理绝对时间"这个旁证。详见 `docs/experiment-provenance.md` §9.3。

### 2026-09-22 更新：最有用的一条结论 ——「多给一点预算」居然几乎免费

这一轮做了一件之前一直没做的事：**把选多少画面块（预算 K）当成变量扫了一遍**（64 帧、6 样本）。
结果有点反直觉：

| 保留比例 | 猜对的比例（接受率） | 需要跑几轮 | 每轮花的时间 | 总解码时间 |
|---|---|---|---|---|
| 6.12%（很小） | 0.776 | 101 | 524 ms | 53.0 s |
| 10.05% | 0.827 | 94 | 536 ms | 50.4 s |
| 12.38%（≈我们现在用的） | 0.853 | 91 | 537 ms | 48.9 s |
| **20.10%（很大）** | **0.912** | **86** | **535 ms** | **46.0 s（−13%）** |

**两件事同时成立**：给更多画面块 → 猜得更准 → 轮数变少；**但每轮并没有变慢**
（"猜"这一步要跑整个 7B 模型，多看几百个画面块几乎不花时间：k 涨 3.3 倍，每轮只贵 2.4%）。

**所以："我们现在用的预算太小了。"** 把 `--k-plus-text` 从 1024 提到约 3400（保留率 ~20%），
**一行代码都不用改**，静态线解码就能快约 13%。这是目前所有工作里**唯一"零改动、正收益"**的结论，
优先级高于我们调研的那五篇论文方法。

⚠️ 两点保留：这个 −13% 是在**和别人共用显卡**时测的（时间不准，需要空闲卡复验）；
但**"轮数从 101 降到 86"是确定性数字**，"每轮几乎不变贵"也有分量计时佐证。

> **术语对照**：本文下面的章节里把"静态还有多少可改进空间"叫做 **headroom**（余量），"接受率提升"叫 **Δaccept**，"完全重建/局部更新"叫 **full rebuild / incremental refresh**，"配置"叫 **A/B/C/D**（A = 基线，B = 只开节流，C = 只开读全，D = 两个都开）。

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
| 2026-09-18 | **L2 筛选 R1（incremental 直接对照）** | ✅ **已完成**（09-18 14:43，`timing_valid=true`）— **H1 直接排除**；refresh 70.9 → **22.0 ms/轮**（3.2×、无长尾），但 dynamic 仍 **0.819×** |
| 2026-09-19 | **E1/E6 单变量矩阵（A100 2×2）** | **E1（节流）是主效应 +0.031/+0.045；E6（全打分）单独无效 −0.003；两者叠加 D 把总体惩罚清零（Δ −0.0008）**，10/10 exact。详见 §17.8 |
| 2026-09-21 | **512 输出 token（本机 A6000 / GPU 2）** | **`timing_valid=true` 首次拿到可信墙钟**：static vs AR decode **1.711–1.873×**（128 tok 时仅 1.554×）；dynamic vs static 最好 **0.910×**（B=E1），仍未 >1×。详见 §17.9 |

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
1. **H_headroom 在 full rebuild 下依然成立** → **H1（incremental slot 顺序是负结果主因）排除**；**R1 直接对照已完成并独立确认该排除**（见 §17.6）。
2. 均值第三次掩盖 ±0.19 的分化 → 判读必须分层（`summarize_a100_dynamic.py` 已强制输出配对 + 分箱）。
3. 附带：本机 A6000 上 **static vs AR decode = 62.8s vs 87.9s ≈ 1.40×**（A100 上为 1.15–1.24×），差异待查。

### 17.3 未完成 / 未决

| 项 | 状态 |
|---|---|
| **R1（incremental 直接对照）** | ✅ **已完成**（2026-09-18 14:43，`timing_valid=true`）。**H1 直接排除**：mean Δ = −0.0305（R0 −0.0308）、corr = −0.728（R0 −0.740）、**7/10 样本逐位一致**。详见 §17.6 |
| 本机 GPU（2026-09-19） | `nvidia-smi` 报 "couldn't communicate with the NVIDIA driver"（内核模块仍在、4×A6000 + 4090 在位）→ **本机暂时无法跑实验** |
| A100/gpu23 GPU1（2026-09-19） | 37.9/81.9 GiB 已占但**利用率 0%**，剩 ~44 GiB → 加 `--allow-shared-gpu --min-free-gib 26` 即可跑（GPU0 是他人 vLLM，不碰） |
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

### 17.5 为什么没赢：开销账单 + 指标错配（2026-09-18 新增）

**开销账单**（R0，10 样本 × 128 tokens）：

| 项目 | 耗时 | 占比 | 每轮 |
|---|---:|---:|---:|
| **打分**（读注意力、算打分表） | 1.87 s | **1.9 %** | **11 ms** |
| **搬运**（把新选中的画面块写进缓存） | 11.84 s | **11.7 %** | **71 ms** |
| 调试校验 T1（正式跑没有） | ≈17.3 s | — | — |
| static 基线 | 62.8 s（168 轮） | | 374 ms/轮 |

- **每轮过路费 82 ms = 单轮成本的 22 %** → 动态必须**减少 22 % 以上轮数**才回本。
- 一次搬运 **57 MB / 71 ms → 有效带宽仅 0.8 GB/s**，比 A100 的 ~2000 GB/s **低约 2500 倍**。
  → **瓶颈不是带宽，而是 224 次 `torch.unique` + 56 次 gather/copy 的碎片化发射 + 2 次强制 `cuda.synchronize()`。**
- 去掉搬运后过路费 **82 → 11 ms/轮（22 % → 3 %）**，准入门槛降约 7 倍。
- 本批轮数几乎没变（168 → 167）：**多花的 38 s 几乎全是纯开销**。

**指标错配（最关键的一条）**：

| 选法 | 召回率 | 每轮接受词数（proxy） |
|---|---:|---:|
| Static | 0.6465（**理想的 65 %**） | 7.45 |
| Previous（我们的动态） | 0.7288 | 7.87（+0.42） |
| Oracle（上限） | 1.0000 | 8.65（**+1.20**） |

- 静态召回率仅 0.647 却接受率高达 0.83 → **召回率很敏感、接受率很迟钝**（很多不同的 KV 集合会产生同样的 greedy 输出）。
- oracle 上限只有 **+1.20 词/轮（≈16 % 少轮）**，我们拿到 +0.42 → **转化率 35 %**。
- **结论：动态 KV 选择不是伪命题（静态只拿到理想的 65 %）；问题在于预测器太弱（信息过期快 Jaccard 0.58、估计有噪声 fidelity 0.815、每轮硬换 30 %），且接受率这条通道本身很迟钝。**

### 17.6 L2 筛选 R1 — incremental 直接对照（09-18 已跑，09-19 补算）

**数据**：`results/l2_screening/R1_incr_three_att.jsonl`（112 行、`complete`、`timing_valid=true`），
报告 `R1_incr_three_att_report.md` + `R1_incr_three_att_summary.json`（2026-09-19 用 `summarize_a100_dynamic.py` 补出）。
`wait_r1.log` 显示前 13 次尝试被抢卡（rc=1），第 14 次在 GPU0 成功（`R1_EXIT=0`）——**不是未完成，是未整理**。

**对照的干净性**：与 R0 之间只差 `refresh_mode: full → incremental`（`cache_len` 40960 vs 20480 仅影响预分配），
且 `dynamic_std_qwen25vl.py` / `dynamic_selection.py` / `sparse_cache_refresh.py` 的 sha256 **与 R0 逐字节相同**。

**正确性侧（H1 的直接检验）**：

| 指标 | R0（full） | R1（incremental） |
|---|---:|---:|
| mean static accept | 0.8103 | **0.8103**（逐样本相同 → 仪表自检 ✓） |
| mean Δaccept | −0.0308 | **−0.0305** |
| `corr(static_accept, Δ)` | −0.740 | **−0.728** |
| 低余量组 Δ | +0.032 | +0.032 |
| 逐样本差异 | — | **7/10 逐位一致**，其余 3 个 \|Δ\| ≤ 0.018 |
| T1 `consistency_mismatches` | 0 / 30 | 0 / 30 |
| exact（static / dynamic） | 8/10 / 8/10 | 8/10 / 8/10（同为 `496-3`、`754-1`，**static 本身即不符**） |

→ **H1 排除由逻辑论证升级为直接对照**；headroom 结构（corr ≈ −0.73）在两种 refresh 下稳定复现。

**经济性侧（incremental 明显更好）**：

| 每轮 refresh | mean | p50 | p90 | p99 | max | 10 样本总计 |
|---|---:|---:|---:|---:|---:|---:|
| full（R0） | 70.9 ms | 16.9 | 274.0 | 562.8 | 598.1 | 11.84 s |
| **incremental（R1）** | **22.0 ms** | 21.8 | 23.4 | 26.1 | 29.3 | **3.68 s** |

**但端到端仍然没赢**（R1，`timing_valid=true`）：

| 每轮 | static | dynamic | 差 |
|---|---:|---:|---:|
| decode / 轮 | **328.0 ms**（168 轮） | **403.1 ms**（167 轮） | **+75.1 ms（+22.9%）** |
| 其中 refresh | 0 | 22.0 | +22.0 |
| 其中 selection update | ~0.6 | ~10.1 | +9.5 |
| draft + verify + bonus | — | — | +8.9 |
| **未被 profile 覆盖的 host 侧** | — | — | **≈ +34** |

→ 两条结论：
1. **incremental 作为默认配置现在有数据支撑**（代码里本来就是默认）：同等 accept、refresh 便宜 **3.2×**、长尾消除（max 598 → 29 ms）；§5 Task 2 在 VDC 上的相反结论属小样本，不适用于 Video-MME。
2. **"搬运"这条腿基本解决（71 → 22 ms/轮），过路费剩下最大的一块变成未被计时器覆盖的 host 侧开销（≈34 ms/轮）**——E4 的靶子应当据此改写。
   轮数几乎没变（167 vs 168），多花的 12.2 s 仍全部是纯开销。

### 17.7 下一阶段实验计划

完整版见方案 §12.4。**E1 与 E6 已于 2026-09-19 执行完毕（§17.8）**，剩余按性价比排序：

| # | 实验 | 为什么 | 成本 | 状态 |
|---|---|---|---|---|
| **E1** | **打开节流闸门**（`interval=4` + 阈值 0.05） | R0 把闸门**全关了**，跑的是最贵配置；打开即降本 | **零代码** | ✅ **+0.031/+0.045，refresh 降到 1/5** |
| **E6** | **用更全的打分**（`--dynamic-collector v3` = all-query） | 打分只占 1.9 %，却因只读 2 个位置损失 18 % 保真度 | **零代码** | ✅ **单独 −0.003（无效）；与 E1 叠加 +0.011** |
| **E4′** | **消掉未被计时的 host 侧开销**（先补计时再优化） | E1 后 refresh 只剩 ~16 ms/轮，**过路费却仍有 ~160 ms/轮**；§17.6 已确认 ≈34 ms/轮是黑盒 | 中 | ⏭ **优先级升到第一** |
| **E3** | **headroom-gated 启用** | 静态差才开，是 §17.2 结论的直接推论 | 小 | ⏭ 次之 |
| **E2** | **运行时实现 EMA 平滑** | 离线 recall 打平但更平滑（搬运更少）；**离线测过、运行时从未实现** | 小 | 待定 |
| **E5** | **索引化 KV（PagedAttention 式）** | 每轮搬运 **57 MB → 0.45 MB（↓128×）** | 大（需新核） | 后续 |
| **E7** | **换判别标准**（注意力 × value 范数等） | E6 已证"打分更准"补不回接受率，说明该换判别量而非换精度 | 中 | 后续 |

**天花板提醒**：即使完美预测也只有约 **16 % 的轮数节省**，属**中等收益**，必须与开销削减叠加才有端到端意义。

> **E4 的靶子已因 R1 改变（2026-09-19）**：incremental 已把 refresh 从 **70.9 → 22.0 ms/轮**（§17.6），
> "把 refresh 压到 <15 ms"的增量收益变小；**现在真正的大头是未被 profile 覆盖的 ≈34 ms/轮 host 侧开销**
> （Python 记账、digest、断言、主循环本身）。E4 因此改为：**先给动态主循环补上覆盖 host 侧的计时，定位后再消掉它。**

> **E1 与 E6 已执行完毕（2026-09-19），见 §17.8。** 两句结论：
> **E1（节流）是主效应**（Δ +0.031 / +0.045），**E6（all-query）单独几乎无效**（Δ −0.003），
> 两者叠加（配置 D）把总体惩罚清零（Δ **−0.0008**）。**下一步只剩 E3 + E4′。**

---

## 17.8 E1 / E6 单变量矩阵（A100，2026-09-19）★

> **零代码实验。** E1 是配置（节流闸门），E6 原以为要改代码，核对后发现 **`--dynamic-collector v3` 就是 all-query**
> （`dynamic_std_qwen25vl.py:440` 里 v3 的 `query_mode_effective = "all"`，in-hook GEMM、不强制同步），
> 所以两者都能用开关完成，做成一个单变量矩阵。

**执行**：gpu23 物理 GPU1，Video-MME seed-42 前 10 样本、128 帧 / 128 tokens、γ=9、K+text=1024、
`cache-len 20480`、`refresh=incremental`、`bootstrap=attention`、`fallback=none`，
强制 `--assert-equal-s0 --assert-consistency`。每个配置 **10 样本、约 51 分钟**。
产物：`results/l2_screening/e1e6_20260919/`（4 个 JSONL + 报告 + `comparison.md`）。

### 17.8.1 基线必须在同一台机器上现测（这次差点踩坑）

R1（§17.6）是在**本机 A6000** 上跑的，我原本打算直接拿它当配置 A。核对后发现
**同一份代码、同一条命令、同一批样本，static 接受率在 A6000 与 A100 上不同**——
单样本最大差 **0.151**（`154-3`：0.541 → 0.692）、另一个方向 **−0.109**（`504-2`：0.966 → 0.857），
而**均值只差 0.006**（0.8103 → 0.8161）。

这正是最危险的情形：把 R1 当基线，硬件效应会被摊进每一个配置的 Δ 里，**从均值上完全看不出来**。
所以补跑了 A100 上的 A。完整证据与规则见 **`docs/experiment-provenance.md` §7**。

**A100 侧自校验**：A/B/C/D 四次独立进程调用（跨 3.5 小时）给出**逐位相同的 static**，所以这批配对比对是干净的。

### 17.8.2 2×2 结果

| 配置 | collector | interval | **Δaccept** | corr(static,Δ) | 低余量组 Δ | 轮数 | 每轮改选比例 | refresh ms/轮 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **A** 基线 | v2 | 1 | **−0.0429** | −0.670 | +0.077 | 169 | 0.467 | 86.5 |
| **B** = E1 节流 | v2 | 4 | **−0.0121** | −0.391 | +0.037 | 164 | 0.136 | 15.7 |
| **C** = E6 全打分 | v3 | 1 | **−0.0454** | −0.715 | +0.101 | 169 | 0.436 | 70.6 |
| **D** = E1+E6 | v3 | 4 | **−0.0008** | −0.609 | **+0.117** | **161** | 0.125 | 18.6 |

**四个单变量效应**（同一硬件、同一 static，D↔B 与 D↔C 是干净对照）：

| 效应 | 大小 |
|---|---:|
| **E1（节流）在 v2 下** | **+0.0308** |
| **E1（节流）在 v3 下** | **+0.0446** |
| **E6（all-query）在 interval=1 下** | **−0.0025**（≈0） |
| **E6（all-query）在 interval=4 下** | **+0.0113** |
| **交互作用**（两条路径算出一致） | **+0.0138** |

**逐样本 Δ**（static 由高到低）：

| 样本 | static | A 基线 | B = E1 | C = E6 | D = E1+E6 |
|---|---:|---:|---:|---:|---:|
| 599-1 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 717-1 | 0.983 | −0.126 | −0.040 | −0.170 | −0.126 |
| 647-1 | 0.966 | −0.024 | −0.031 | −0.031 | **+0.025** |
| 102-2 | 0.891 | −0.203 | −0.170 | −0.242 | −0.199 |
| 496-3 | 0.864 | −0.102 | −0.006 | −0.152 | −0.006 |
| 504-2 | 0.857 | −0.061 | **+0.070** | −0.061 | 0.000 |
| 445-2 | 0.790 | −0.143 | −0.053 | −0.101 | −0.053 |
| 154-3 | 0.692 | +0.011 | +0.086 | +0.004 | **+0.178** |
| 050-1 | 0.623 | +0.044 | −0.074 | **+0.124** | −0.048 |
| 754-1 | 0.495 | +0.175 | +0.098 | +0.175 | **+0.221** |

### 17.8.3 判读

1. **E1（改选更少）是主效应。** 在两个 collector 下都为正（+0.031 / +0.045），并把每轮改选比例
   从 **0.467 砍到 0.136（1/3.5）**、refresh 从 **86.5 降到 15.7 ms/轮（约 1/5）**。
   → **"少改一点"比"打得更准"值钱得多**，这是本次最重要的一条。
2. **E6（打分更全）单独几乎无效（−0.0025）。** 把 two/three-query 换成 all-query，保真度从 0.815 提到 ~1.0，
   但**接受率没动**。→ 与 L1 的 **H3 否证**一致：churn 主要来自 **attention 信号本身的漂移**，不是估计误差；
   所以"把估计做准"补不回接受率。
3. **正交互作用（+0.0138）：E1 与 E6 互相成就。** 打分更全 → "这一轮是否值得改"判断更准 → 节流才敢跳。
   单独用任一项都看不到。
4. **D 把总体惩罚清零（−0.0008）且轮数最少（161 vs 基线 169），同时保住 headroom 梯度**（corr −0.609、低余量 +0.117）。
   token 口径下 D 是**唯一为正**的（Δaccept_len **+0.047**，基线 −0.375）。
5. **正确性无回归**：四个 run 全部 **10/10 exact vs AR**、T1 **30/30**。
6. **但这仍是"不亏"，不是"赚"。** 没有任何配置让 Δ 显著为正。
   要变成正收益，只剩两条没做的腿：**E3（headroom 门控）**与 **E4′（host 侧开销）**。
7. ⚠️ **`timing_valid=false`，本批墙钟数字不可用**：四个"本应完全相同"的 run 里，static decode 在
   **284–338 ms/轮**之间波动，证明计时被共租户污染。**可用的**是改变比例、轮数、refresh 计数这类结构性量；
   "端到端快多少"仍要等一张空闲卡（本机 A6000 驱动恢复后最合适）。

**新暴露的问题（下一轮必须处理）**：E1 之后 refresh 只剩 ~16 ms/轮，**但 D 的过路费仍高达 +162.9 ms/轮**
（即使这个数被争用污染，量级也远大于 refresh）。→ 与 §17.6 的结论合流：
**host 侧开销（未被 profile 覆盖的部分）现在是唯一的大头，E4′ 的优先级升到 E3 之前。**

---

## 17.9 512 输出 token（本机 A6000 / GPU 2，2026-09-21）★

> **这是第一条 `timing_valid=true` 的 dynamic 墙钟证据。** 此前所有动态墙钟都来自
> 共租户污染的 run（`timing_valid=false`），只能看结构性量、不能看"快多少"。

### 17.9.1 动机：128 token 太短，动态被系统性低估

动态路由的收益**跨轮累积**，而 selection/refresh 的过路费是**每轮固定支出**。
生成越短，累积越少、固定支出占比越高 → 动态越吃亏。此前主线的
decode 加速比**全部**在 128 token 下测得，因此必须把输出长度提上去复测。

**新增驱动**：`scripts/run_dynamic_matrix_local.sh`（本地版，与远端版逻辑一致，
但**等 GPU 真正 idle**而不是只等显存，因此产出 `timing_valid=true`）。

**配置**：`benchmark_a100_dynamic.py`，本机 `host=A6000`、`physical_gpu=2`，
Video-MME 可用子集 seed-42 前 **10 样本**、**128 帧 / 512 tokens**、γ=9、K+text=1024、
`cache-len 20480`、`refresh=incremental`、`repeats=1`、`--assert-equal-s0 --assert-consistency`。
产物：`results/l2_screening/local512/`（含 `comparison.md`）。

### 17.9.2 结果（A/B/C 三档，全部 `timing_valid=true`）

| 配置 | collector | interval | min_change | **static 接受率均值** | Δaccept | corr(static,Δ) | 低余量 Δ | 动态轮数 | 改选比例 | refresh ms/轮 | 过路费 ms/轮 | **decode static vs AR** | **decode dynamic vs static** | exact |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **A** 基线 | v2 | 1 | 0.00 | 0.9115 | −0.0178 | −0.604 | +0.0271 | 572 | 0.417 | 22.3 | +40.4 | **1.711×** / 1.749× | **0.887×** / 0.886× | 8/10 |
| **B** = E1 节流 | v2 | 4 | 0.05 | 0.9115 | −0.0171 | −0.403 | +0.0049 | 572 | **0.136** | **6.0** | **+28.9** | **1.796×** / 1.812× | **0.910×** / 0.910× | 8/10 |
| **C** = E6 全打分 | v3 | 1 | 0.00 | 0.9115 | −0.0128 | −0.643 | +0.0413 | 569 | 0.368 | 37.9 | +80.8 | **1.873×** / 1.876× | **0.827×** / 0.831× | 8/10 |
| **D** = E1+E6 | v3 | 4 | 0.05 | — | — | — | — | — | — | — | — | — | **未测** | — |

（加速比写 `ratio-of-sums / mean-of-ratios`。**AR 10/10 exact**；static/dynamic 8/10 差在
`496-3`、`754-1`，**static 本身即不符**，与 routing 无关。T1 **30/30**。峰值显存三级同为 **33.22 GiB**。）

### 17.9.3 与 128 token 的对照

| 量 | 128 tok（R1, GPU 0） | 512 tok（A/B/C, GPU 2） |
|---|---|---|
| **decode static vs AR** | 1.554× / 1.621× | **1.711–1.873×** / 1.749–1.876× |
| **decode dynamic vs static**（最好档） | 0.819× / 0.816× | **0.910× / 0.910×（B）** |
| static accept 均值 | 0.8103 | **0.9115** |
| Δaccept 均值 | −0.0305 | **−0.013 ~ −0.018** |
| 过路费 | +75.1 ms/轮 | **+28.9 ~ +80.8 ms/轮** |

**判读：**

1. **输出拉长显著改善 static 复现：1.554× → 1.71–1.87×**，逼近论文的 up to 1.94×。
   → 此前偏低的数字**有一部分是生成长度造成的**，不是实现缺陷。这是本次最重要的收获。
2. **dynamic 的劣势同步收窄但仍为负：0.819× → 最好 0.910×**（慢 18% → 慢 9%）。
   **仍然没有任何一档 >1×。**
3. **E1（节流）在 512 下仍是最好的动态配置**（0.910×、refresh 6.0 ms/轮、改选比例 0.136），
   **与 §17.8 在 128 token 上的结论一致，且跨生成长度稳定复现**。
4. **E6（全打分）在 512 下更差（0.827×），但该 run 被机器争用污染**（见 17.9.4），
   B 档最可信，C 档墙钟要打折扣（结论仍与 §17.8 的"E6 单独无效"一致）。
5. **headroom 梯度仍在**（低余量组 Δ 为正、高余量组为负，corr −0.40 ~ −0.64），
   但 512 下 static 接受率整体升到 **0.9115**（128 时 0.8103）→ **余量变小，动态的潜在收益空间更小**。
   这与 H_headroom 一致：这次是"空间本身被压缩"。
6. **D 未测**：起跑时 GPU 2 被 `ccl` 与**本账号 3 × `opd_train_vlm.py`** 占用，
   8 张卡全忙且 4090 只有 24 GB（放不下 33.2 GiB），等不到空闲卡。

### 17.9.4 ⚠️ 新增口径漏洞：`timing_valid` 只覆盖"起跑瞬间"

**三档都在 GPU 2 空闲时启动、`timing_valid` 全为 `true`，但 C 明显仍被机器级争用污染：**

| 量 | A | B | C |
|---|---:|---:|---:|
| prefill（10 样本合计） | 315.8 s | 317.6 s | **409.6 s** |
| AR decode | 350.0 s | 349.8 s | **430.4 s** |
| `Selection/top-k` | 0.110 s | 0.100 s | **1.330 s** |

C 的 prefill 比 A/B 高 **30%**，而配置与样本完全相同 → 不是代码，是**别的 GPU 上的负载**
（邻居作业吃满机器带宽/功耗）。**`timing_valid` 的判据只是"启动时本卡 `used ≤ 256 MiB` 且 `util == 0`"，
它不保证整段运行期间机器安静。**

**记录规则补充**：引用墙钟时必须同时给出 **prefill 与 AR decode 的绝对时间**作为旁证。
AR 是所有配置共用的分母，**它偏慢就说明整段被拖慢**。
（完整版见 `docs/experiment-provenance.md` §9。）

### 17.9.5 硬件可行性（2026-09-21 实测）

- **本机 A6000 是当时唯一能拿到可信墙钟的地方**：6×A6000（48 GB）+ 2×4090（24 GB）。
  **峰值 33.22 GiB，512 与 128 token 相同** → 输出长度不是显存约束；但 33.2 GiB > 24 GB，**塞不进 4090**。
- **A100/gpu23 当时不可用**：GPU 0 常驻 vLLM（79.5 GB / 100%）；GPU 1 被他人 vLLM + 另一进程占用，
  普通启动被拒、加 `--allow-shared-gpu` 则 `timing_valid=false`（等于回到不可用状态）。
- 模型从 `/mnt/local2`（机械盘 ext4）加载 **约 4.5 分钟/次**，排期要计入。

### 17.9.6 下一步

1. **补 D**：等一张空闲 A6000 跑 `D_v3_i4`，凑齐 512 的 2×2（命令已备好，
   `CONFIGS_OVERRIDE="D_v3_i4|v3|4|0.05" scripts/run_dynamic_matrix_local.sh local512`）。
2. **E4′ 仍是主线**：E1 之后 refresh 只剩 6.0 ms/轮，**过路费仍有 +28.9 ms/轮**，
   host 侧开销依旧是唯一大头；512 下它相对每轮成本已被摊薄，但绝对值没变。
3. **E3（headroom 门控）**：512 的证据进一步支持它——static 接受率已经 0.91，
   整体收益空间很小，**只有低余量样本才有得赚**，所以门控比"全局开动态"更合理。

---

## 18. 廉价 KV 筛选：五种方法的调研与执行计划（2026-09-22）★

> **完整方案**：`docs/superpowers/plans/2026-09-22-cheap-kv-selection-methods.md`
> **论文与代码索引**：`papers/README.md`（5 篇论文 PDF 已下载；MM-ShiftKV 官方代码存于 `third_party/MM-ShiftKV/`）

### 18.1 出发点需要修正

用户提出的前提是"**STD 的 top-K 选择每个 decoding step 都要做，开销持续产生**"。
**这个描述只对 dynamic 研究线成立，对 STD 不成立**：static STD **本来就是在 prefill 做一次选择**
（`build_sparse_selection`），而且已经在 512 token 下拿到 **1.711–1.873× vs AR**。
所以 MM-ShiftKV / FlashCache 的"prefill-only"属性对 STD**不是新能力**，它们真正的价值是另外两件事：

1. **提高那"一次性 prefill 选择"的质量** → 抬高 static 接受率（现 0.9115），**这是唯一能真正减少轮数的杠杆**；
2. **免掉 `output_attentions=True`** → 现在为了拿注意力矩阵被迫走 eager 自定义 attention，既慢又扰动数值、破坏位精确性。

### 18.2 门控结论（用现有数据算出，未新增实验）★

把 dynamic 的 refresh 与 selection 时间**全额扣除**，得到"搬运与选择零成本"的乐观上界：

| 运行 | static (s) | dynamic (s) | refresh (s) | select (s) | **oracle (s)** | **C/A 上界** | B/A 现状 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 tok R1 | 55.11 | 67.32 | 3.68 | 1.69 | 61.94 | **0.890×** | 0.819× |
| 512 tok A | 204.59 | 230.58 | 12.77 | 5.96 | 211.85 | **0.966×** | 0.887× |
| 512 tok B (E1) | 194.75 | 214.03 | 3.45 | 1.95 | 208.63 | **0.933×** | 0.910× |
| 512 tok C (E6) | 229.72 | 277.74 | 21.58 | 5.72 | 250.44 | **0.917×** | 0.827× |

**即便把 refresh 与 selection 全变成 0，dynamic 仍然比 static 慢 3–11%。**
→ **"把每轮筛选做便宜"这条路在单请求下封顶 <1.0×，不可能靠它翻盘。**
扣除后仍残留 **+12.7 ~ +40.9 ms/轮**（host 侧 + draft/verify 差异），即 **E4′ 的靶子**。
（注：该上界是保守的——它没扣掉未计时的 host 侧开销。）

**结论：主线转向"提高一次性 prefill 选择的质量"（提高接受率 → 减少轮数），并同时推进 E4′。**

### 18.2b 门控结论二：选择器的剩余空间也只有 ~7pp ★（2026-09-22 新增）

用**已有 trace**（`results/routing_traces_videomme/`，20 条）离线算出：
`prefill_scores`（层均值）作 static 选择分数，`round_scores`（每轮 `[kv_heads, q_len, visual_len]`）
作**解码期真实注意力**，在 STD 预算（retention 6.09%，k=967）下算覆盖率。
脚本：`scripts/analysis/selection_headroom.py`；产物：`results/routing_analysis_videomme/selection_headroom.md`。

| 量 | 值 |
|---|---:|
| `coverage_static`（现有 prefill 选择覆盖多少解码期注意力质量） | **0.3770** |
| `coverage_oracle`（**能预知未来**的完美选择器） | **0.4484** |
| **绝对空间（完美解码感知选择器的全部上限）** | **+0.0714（7.1pp）** |
| static / 随机 | 6.19× |
| **static / oracle** | **84.1%** |

**两个关键子结果：**

1. **覆盖率随解码不降反升**：早期轮 0.3629 → 晚期轮 **0.3903**（+2.7pp）。
   → **static 在解码后期并没有变差**，而"每轮重选"的立论前提正是它会漂移。
   （与 `drift_jaccard.csv` 不矛盾：选择**集合**在漂移，但**覆盖的质量没掉**——再次印证"召回率敏感、接受率迟钝"。）
2. **预算扫描**：static/oracle 在所有预算下都高（3%→80.6%、6.09%→84.1%、10%→86.3%、20%→89.6%、30%→91.9%）。
   → **瓶颈既不在选择器，也不在预算。**

**判读：即便给一个能预知未来的完美 prefill 选择器，覆盖率也只能从 37.7% 提到 44.8%（+7.1pp / 相对 15.9%）。
现有 static 已拿到 oracle 可达质量的 84.1% → 选择器不是主要瓶颈。**
MM-ShiftKV / FlashCache-Dev / HybridKV 这些"更好的 prefill 打分器"的收益上界就是这个 7.1pp，
而它们都是**代理估计**（看不到未来），实际只能吃到其中一部分。

> ⚠️ **粒度限制**：trace 的分数是**层均值**（`attention_trace.py:101`），所以这是**跨层共享选择**的数字，
> 不是 STD 真实的 per-(layer,head) 选择。按 Jensen 不等式，per-layer 的 oracle 会**更高**，
> 真实空间**可能比 7.1pp 更大** —— 需要一次 per-layer trace 才能定论（P1-a/P2-a 的第一个动作）。

**两个门控合起来**：把筛选做便宜（封顶 <1.0×）与把筛选做准（空间 ~7pp，且已吃掉 84%）**都被实测封住了上限**。

### 18.2c 门控结论三：覆盖率↔接受率是**关联不是因果**，并由此找到当前最该做的实验 ★

**（1）看起来有相关性**：把 trace 的 per-sample `coverage_static` 与实测 static 接受率配对（n=10，
trace 的 20 个样本包含 benchmark 用的那 10 个），`r(coverage, accept)` 在所有预算下都是正的：

| retention | mean coverage | r(coverage, accept) |
|---:|---:|---:|
| 3.00% | 0.2723 | **+0.720** |
| 6.09% | 0.3880 | **+0.752** |
| 20.00% | 0.6395 | **+0.806** |

**（2）但这是样本难度混淆**：

| 检验 | r |
|---|---:|
| `r(coverage@3%, coverage@6%)` | **+0.994** |
| `r(coverage@6%, coverage@20%)` | **+0.981** |

**一个样本在 6% 预算下的覆盖率几乎完全决定它在 20% 下的覆盖率** → coverage 基本是
**"这个样本注意力有多集中"的样本级属性**。它与接受率的相关只是"注意力集中的样本恰好也是
draft 本来就能猜准的简单样本"。**所以不能用它论证"提高覆盖率会提高接受率"。**

接受率侧的**唯一因果上界**仍来自 Oracle Study：完美选择只值 **+1.20 词/轮（≈+16% 相对，≈14% 少轮）**，
而动态实现只拿到 **35%**。
（另：`dynamic_stats.per_round` **不记录逐轮接受率**，所以这个因果问题**无法离线回答**，必须端到端 A/B。）

**（3）由此得到当前性价比最高的实验：静态线 K 扫描（G0.5）**

| K（retention） | k | coverage_static |
|---|---:|---:|
| 6.09%（现状） | 967 | 0.3770 |
| **10%** | **1588** | **0.4726** ← **已超过 6% 预算下的 oracle（0.4484）** |
| 20% | 3175 | 0.6287 |

**把预算从 6.09% 提到 10%，覆盖率涨 +9.6pp —— 比"在 6% 预算下做到完美选择"（+7.1pp）还多。**
而且它**不用移植任何论文代码、不用 per-layer trace、不改选择逻辑**
（`build_sparse_selection` 已有 `explicit_k`/`target_k_plus_text`，只改命令行）。

**判据**：接受率是否随 K 上升。
- 上升且每轮成本涨得更慢 → 存在净收益最优 K，**静态线直接获益**（本计划最有希望的正面结果）；
- 不动 → "覆盖率→接受率"被证伪为纯关联，**选择器方向整体止损**，资源转入 E4′ 与单轮 verify 效率。

**成本**：3 档 × 10 样本 × 512 token ≈ 3–4.5 小时。

**修正后的杠杆排序**：**① G0.5 静态线 K 扫描** → ② E4′（host 侧开销）→ ③ 单轮 verify 效率 →
④ P1/P2/P3（选择器，上界已被量化）。

### 18.2d 离线准备已全部完成（2026-09-22）★

阶段 0 的四个门控（G0.1/G0.3/G0.4）用现有数据就把方向收敛了；此后三轮**全部 GPU 被占**，
于是把所有"有卡就能立刻出数"的离线件做完并测试：

| 类别 | 文件 | CPU 测试 |
|---|---|---|
| **P1 内核**（MM-ShiftKV） | `src/std_repro/mmshift_proxy.py` | 15 |
| **P2 内核**（HybridKV） | `src/std_repro/hybridkv_budget.py` | 18 |
| **P3 内核**（FlashCache） | `src/std_repro/flashcache_dev.py` | 26 |
| **P1-a/P2-a 采集** | `scripts/analysis/attention_trace_per_layer.py` | 15 |
| **P1-a/P2-a 分析** | `scripts/analysis/analyze_per_layer_selection.py` | 15 |
| G0.3/G0.4 门控 | `scripts/analysis/selection_headroom.py` | — |
| LAUNCH G0.5 | `scripts/run_dynamic_matrix_local.sh`（新增第 5 字段"附加参数"） | — |

**合计 89 个新 CPU 测试全部通过**；全量套件 293 passed（4 个既有失败是环境性的：
cpu 测试环境的 transformers 5.10.1 vs specvlm 的 4.48.0，与本次改动无关）。

**因此现在缺的只有硬件**：G0.5（3–4.5 h）、P1-a/P2-a（一次采集 30–40 min/样本）、P3-a（1 天）
全部是"一条命令就能跑"的状态，命令都写在
`docs/superpowers/plans/2026-09-22-cheap-kv-selection-methods.md` 里。

### 18.2e 【代码核实】异构 per-head 预算是死路；冻结 static 头才是便宜的那半边 ★

计划 §5.2 原来假设"现有 `idx = cat([non_visual, topk[head]])` 已支持每头不同长度"，
**这个假设是错的**，逐处读码后确认变长预算会硬撞三堵墙：

| 位置 | 事实 |
|---|---|
| `std_qwen25vl.py:278,289–290,297` | `compact_len = non_visual + selection.k` 是**单一全局标量**，每头都断言相等，`current_length_data.fill_(compact_len)` 只填一次 |
| `std_qwen25vl.py:349–351` | `_gather_sparse_kv` **显式 raise**：`"Sparse KV gather produced uneven per-head lengths."` |
| `std_qwen25vl.py:405–410` | draft 走 `attn_mask=None, is_causal=False` → **没有 mask 能遮住 padding**；用 padding 凑齐会让填充 KV 参与 softmax，**破坏 lossless** |

→ **HybridKV Eq.5 与 FlashCache DBA 的异构预算成本从"中等"上调为"大"，且有破坏位精确性的风险，降级。**

**但 P2 值钱的那一半根本不需要改预算。** HybridKV 的收益可拆成两半：
"冻结 static 头（省 refresh）"＝ **不需要改布局**；"按 S 重摊预算"＝ 需要（被上面挡住）。

**为什么冻结免费**：`sparse_cache_refresh.py:188–190` 的 `incremental_refresh_sparse_visual_kv`
**本来就按 (layer, head) 算 changed mask**，且 `m_total == 0` 时**整层 `continue`**。
某个 head 选择不变 → 它的 scatter 工作自然归零，**refresh 开销随 dynamic head 占比下降，现存代码一行不用改。**

**已实现**：`src/std_repro/dynamic_selection.py` 新增 `FrozenHeadTopKPolicy` 与 `freeze_mask_from_sparsity(·, θ)`；
`tests/test_frozen_head_policy.py` **9 个测试**，含两条等价性断言 ——
全 `False` mask ≡ `PreviousVerifyTopKPolicy`、全 `True` mask ≡ `StaticPolicy`（逐位相同）。

**这是一个"先读码、省下一轮 GPU 试错"的例子**：如果直接上 GPU 试 Eq.5，会在布局改造上撞墙并浪费卡时。

**接线已完成（2026-09-22）**：`SparseSelection.head_sparsity`（默认 `None`）→
`build_sparse_selection(record_head_sparsity=...)`（用同一张已物化张量顺手算 `S_{l,h}`）→
`dynamic_std_generate_qwen25vl(freeze_threshold=...)` 构造 `FrozenHeadTopKPolicy` →
`benchmark_a100_dynamic.py --freeze-threshold`。
**默认关闭，baseline 逐位不变**；`tests/test_build_sparse_selection_sparsity.py` **6 个测试**，
其中最关键的是**非侵入性断言**：开/关该 hook，`topk_positions` 逐位相同。
`dynamic_stats` 会记录 `freeze_threshold` / `frozen_head_count` / `total_head_count`，便于判读 `f_d`。

### 18.2f 【第二个更正】FlashCache 的 DBA 也被同一套墙挡住；只有 Dev 判据可用

上一节确认了"异构 per-head 预算"被三堵墙挡住。**本轮进一步核实：FlashCache 的 DBA 也一样中招。**
DBA 是**逐层给不同预算**，而 `compact_sparse_prompt_cache` 只有**一个全局 `compact_len`** 并逐头断言相等 →
**逐层差异同样 `RuntimeError`**。

所以"**DBA 与'用什么分数'正交、可以随时叠加**"这个判断**只对了一半**：
它与「用什么分数」正交，但**不与「统一长度」正交**。→ **DBA 降级**，与 HybridKV Eq.5 同命运。

**仍然可用的是 Dev 判据本身**：它只是"换一个打分函数"，**K 保持 967 不变**，
因此完全不碰布局、也不碰 `SparseSelection` 的长度语义。
代价是它的收益只来自"选得更准"，上界仍受 G0.3 的 **7.1pp** 约束。

**把结论变成可执行证据**：`tests/test_uniform_k_constraint.py`（**6 个测试**）用**行为断言**钉住了三堵墙：
① 逐头变长 → `compact_sparse_prompt_cache` 抛 `unexpected length`；
② 逐层变长 → 同样抛；③ 逐头变长 → `_gather_sparse_kv` 抛 `uneven per-head lengths`；
④ `attn_mask=None` 的 SDPA 在加 padding 后输出确实改变（数值验证，不是注释）。
将来若有人改了布局，这些测试会失败，**强制更新计划而不是让它悄悄失效**。

> 这两轮（§18.2e/18.2f）合计**推翻了两条写在计划里的成本假设**，靠的全是读码 + CPU 测试，
> **没有花一分钟 GPU**。在一个连续 7 轮拿不到卡的时段，这是能产生的最有价值的东西。

---

## 18.3 【第一批实测】64 帧可跑通共享卡；HybridKV 的 θ=0.90 在 STD 上是空操作 ★

前 7 轮拿不到空闲卡。本轮换了个思路：**峰值显存随帧数下降**——实测 128 帧峰值 33.22 GiB，
而 **64 帧只需 21.00 GiB**，于是可以在**正在被别人占用（util 99%）**的 A6000 上以
`--allow-shared-gpu` 跑（`timing_valid=false`，**墙钟不可用，但接受率与所有结构性量仍然有效**）。

### 18.3.1 HybridKV 的 `S_{l,h}` 分布（P2-a 的第一个实测数字，2 样本 × 64 帧）

`dynamic_stats.head_sparsity_summary` 现在记录整条分布（**纯 CPU 后处理，零额外 GPU 开销**）：

| 样本 | S 均值 | S 范围 | θ=0.90 | θ=0.80 | θ=0.70 | θ=0.60 |
|---|---:|---|---:|---:|---:|---:|
| 050-1 | 0.779 | 0.589–0.923 | **0.9%**（1/112） | 45.5% | 82.1% | 98.2% |
| 717-1 | 0.788 | 0.558–0.958 | **8.9%**（10/112） | 50.0% | 83.0% | 96.4% |

**HybridKV 论文推荐的 θ=0.90 在 STD 上几乎冻结不了任何头**（1/112、10/112）。
要有意义地冻结，阈值得降到 **θ≈0.80**（约一半）或 **θ≈0.70**（约 5/6）。
计划里"θ 必须重新标定"的预判得到实测确认。

### 18.3.2 P2 的价值上界（解析，用实测冻结比例 + 实测每轮成本）

| θ | 冻结 f | refresh ms/轮 | 过路费 | dynamic ms/轮 | static/dynamic |
|---:|---:|---:|---:|---:|---:|
| 0.90 | 0.049 | 21.2 | 39.3 | 402.0 | 0.902 |
| 0.80 | 0.478 | 11.7 | 29.8 | 392.5 | 0.924 |
| 0.70 | 0.825 | 3.9 | 22.0 | 384.7 | 0.943 |
| 0.60 | 0.973 | 0.6 | 18.7 | 381.4 | 0.951 |
| **冻结全部（= static）** | 1.0 | 0 | 18.1 | — | **0.952（仍 < 1.0）** |

**→ P2 的冻结路径无法让 dynamic 翻盘：最好 0.952×，而那已经是"退化成 static"。**
原因是过路费里**只有 refresh 可冻结**，`draft/verify/host ≈ 18 ms/轮` 一动不动。

**这是第三条独立指向同一结论的证据**：① G0.1 廉价筛选 ≤0.979×；② G0.3 更好的选择器只值 +7.1pp；③ 本节冻结 ≤0.952×。

### 18.3.3 ★ 冻结扫描完成：完美的剂量-响应，HybridKV 是本轮唯一"实测有效"的方法

`results/freeze_sweep64/`（6 样本 × 64 帧，`timing_valid=false` → 墙钟不可用，**接受率与结构性量有效**）；
可复现分析：`scripts/analysis/analyze_freeze_sweep.py` → `results/freeze_sweep64/report.md`：

| 配置 | θ | 冻结头数 | dynamic 占比 | static 接受率 | dynamic 接受率 | **Δ(dyn−static)** | dynamic 接受长度 | **每轮 changed tokens** | **jaccard** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `F_none` | — | — | 1.000 | 0.8525 | 0.7953 | **−0.0573** | 6.956 | 278.8 | 0.562 |
| `F_080` | 0.80 | 51 | 0.545 | 0.8525 | 0.8110 | **−0.0415** | 7.116 | 133.2 | 0.785 |
| `F_070` | 0.70 | 92 | 0.179 | 0.8525 | **0.8354** | **−0.0171** | **7.345** | **42.5** | **0.931** |

**完美单调的剂量-响应。** 冻结越多头，dynamic 在**每一个轴上都变好**：

- **改写量**：278.8 → 133.2 → 42.5（θ=0.70 时 **−85%**）
- **选择稳定性 jaccard**：0.562 → 0.785 → **0.931**
- **dynamic 接受率**：0.7953 → 0.8110 → **0.8354**（单调逼近 static 的 0.8525）
- **Δ(dyn−static)**：−0.0573 → −0.0415 → **−0.0171**（惩罚缩小 **70%**）

**三条结论：**

1. **仪器自检通过**：三档的 static 接受率**逐位相同（0.8525）**——冻结只作用于 dynamic 侧。
2. **机制成立且超预期**：计划原本只预期"省成本、接受率中性"；实测是**省成本 + 接受率变好**，
   说明被冻住的头原本贡献的主要是**漂移噪声**，重选它们纯属帮倒忙。
3. **但仍不能翻盘**：static 始终领先，且 Δ 的极限是 0（冻结全部 = static）。
   这与 §18.3.2 的解析上界（≤0.952×）**完全一致**——**两条独立路径给出同一结论**。

✅ **正确性（已更正，2026-09-22）**：本批 64 帧运行的 **static 与 dynamic 都是 5/6 exact vs AR**，
**且在冻结三档、K 三档共 6 个配置中完全一致**——冻结与预算改动都**不影响位精确性**。
（此前本文档曾写"static 0/6 exact"，那是**读数错误**：`token_equal` 在 `kind: comparison` 记录上，
不在 `kind: trial` 记录上；从 trial 取该键会得到 `None` 而被误计为"不精确"。已核实并更正。）

**这是本次调研第一个、也是唯一一个"实测有效"的方法**：HybridKV 的冻结确实可测地改善了 dynamic 的
接受率与改写量（虽然不足以让它超过 static）。

### 18.3.4 一条重要的方法论副产品

**"没有空闲卡"不等于"不能出数"。** 只要把配置降到 **64 帧**（峰值 21 GiB），
就能在别人占用的卡上拿到**接受率与全部结构性量**；对"方向判定"类问题（如 θ 该定多少、冻结是否真的减少改写）
这些量已经足够。**代价是墙钟不可用**，所以任何"快多少倍"的结论仍必须等空闲卡。

---

## 18.4 【第二轮实测】预算几乎免费：加大 K 可省 ~13% decode ★★

G0.5 静态线 K 扫描完成（6 样本 × 64 帧，`results/ksweep64/`；复现 `scripts/analysis/analyze_k_sweep.py`）。
`timing_valid=false` → **ms/轮 仅供参考；接受率与轮数是确定性量，有效**。

| retention | k | **static 接受率** | 接受长度 | **轮数** | ms/轮 | 其中 draft | decode 总时长 | **相对 6.12%** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6.12% | 478 | 0.7755 | 6.823 | 101 | 524.5 | 403.5 | 53.0 s | — |
| 10.05% | 785 | 0.8265 | 7.261 | 94 | 536.5 | 413.1 | 50.4 s | **−4.8%** |
| 12.38% | 967 | 0.8525 | 7.513 | 91 | 537.3 | 413.8 | 48.9 s | **−7.7%** |
| **20.10%** | 1570 | **0.9123** | **8.001** | **86** | 535.5 | 413.2 | **46.0 s** | **−13.1%** |

### 三条结论

1. **接受率随预算单调上升**（0.7755 → 0.9123）。
   → **§18.2c 里"覆盖率↔接受率可能只是样本难度混淆"的疑虑被否证**：
   **同一实验内**提高预算确实提高接受率——这是**因果**证据，不是跨样本相关。
   （F2-a 的关键判据因此有了正面答案。）
2. **轮数单调下降**：101 → **86（−15%）**，确定性量。
3. **★ 每轮成本几乎不随 k 变化**：k 涨 **3.3×**（478 → 1570），而 **draft 只涨 +2.4%**（403.5 → 413.2 ms/轮），
   verify 几乎不动。**原因**：sparse draft 每步要跑**整个 7B 模型**，对 k 个 KV 的注意力只占极小一部分。
   → **k 几乎是免费的。**

### 行动结论

**本项目默认的 `k-plus-text=1024`（128 帧下 = 6.09% 保留）严重欠预算。**
把预算提到 ~20% 保留，**不改一行代码**（只改 `--k-plus-text`），能让 static 线 decode **快约 13%**。
这是整个调研里**唯一的"零代码、正收益"改动**，优先级高于全部五个论文方法。

⚠️ **口径**：`timing_valid=false`，**−13.1% 这个数需要在空闲卡上复验**；
但**轮数 −15% 是确定性的**，"每轮成本与 k 几乎无关"也有 draft 分量跨档恒定作为结构性支撑。

**下一步（唯一需要的动作）**：空闲 A6000 + **128 帧** + 512 token，
`--k-plus-text ∈ {1024, 1750, 3400}`（6.09% / 10% / 20%）复验并确认曲线是否继续不饱和。

> **修正后的杠杆排序**：**① 加大 K（零代码，实测 ~13%）** → ② E4′（host 侧开销）→ ③ HybridKV 冻结
> （实测有效但只能缩小 dynamic 的惩罚，不能翻盘）→ ④ 选择器类方法（上界 +7.1pp）。

---

## 18.5 【第三批实测】FlashCache 的 Dev 判据被否证：选择集与随机无法区分 ★

P3-a 完成（3 样本 × 64 帧，V=7812，k≈950）。
脚本：`scripts/analysis/flashcache_dev_recall.py`；产物：`results/flashcache_dev/recall.md`。

| 样本 | k | γ | **Jaccard(Dev, attention)** | **dev_ms** |
|---|---:|---:|---:|---:|
| 050-1 | 967 | 0.1 / 0.2 / 0.3 | 0.064 / 0.064 / 0.063 | 4440 / 3872 / 3930 |
| 717-1 | 924 | 0.1 / 0.2 / 0.3 | 0.073 / 0.071 / 0.070 | 3877 / 3986 / 4102 |
| 496-3 | 951 | 0.1 / 0.2 / 0.3 | 0.067 / 0.070 / 0.073 | 4047 / 4164 / 4166 |

**随机基线**：两个**独立随机** size-k 子集的期望 Jaccard = `k/(2V−k)` = 967/14657 = **0.0660**。
**实测 0.063–0.073 = 随机基线的 0.97×–1.11×** → **Dev 的选择与 attention 的选择统计上无法区分。**

**两条预注册门槛都没过，而且都不是"接近"而是相差一个数量级：**

| 门槛 | 要求 | 实测 | 判定 |
|---|---|---|---|
| 一致性 | Jaccard ≥ ~0.6 | **0.063–0.073（≈随机）** | ❌ |
| 成本 | < 100 ms | **3872–4440 ms** | ❌ **超 40×** |

**→ P3 的 Dev 判据被否证，不必再做端到端 A/B。** DBA 那半边早已被 §5.2 的三堵墙挡住。
**FlashCache 这一篇至此完全出局。**

**为什么**：Dev 度量**缓存 K/V 自身的高频残差能量**，与"文本 query 关注哪些视觉 token"是**正交信号**。
论文 Fig.2 只证明了"删掉高 Dev 的 token 掉点更快"这一弱单调性，
**从未做过同预算下 Dev vs attention 的 head-to-head**——这次替它补做了，答案是不可互相替代。

**一个有用的副产品**：仓库已有的免注意力启发式 `attention_free_visual_topk` 与 attention 选择的
Jaccard 只有 **0.091**，同样很低——这为它被标注为"弱 bootstrap 启发式"提供了量化依据。

### 五篇方法至此的实测/判定状态

| 方法 | 状态 | 关键数字 |
|---|---|---|
| **HybridKV** | ✅ **实测有效**（但不足以翻盘） | 冻结 92/112 头 → 改写 −85%、接受率 0.795→0.835、Δ 惩罚 −70% |
| **FlashCache** | ❌ **实测否证** | Jaccard 0.068 ≈ 随机；耗时超门槛 40×；DBA 被布局墙挡住 |
| **FastCache** | ❌ **分析排除** | 单请求下调度数学上退化为零收益；压缩器需重训且仅 5× |
| **SimiCache** | ⛔ **阻塞** | 论文付费墙，无开放版本 |
| **MM-ShiftKV** | ✅ **门槛已测（per-layer）** | 选择器上界 **+13.8pp**（G0.3 的层均值低估了一半）；内核 15 个 CPU 测试通过 |

---

## 18.6 【第四批实测】per-layer 数据推翻了两条假设：headroom 翻倍，HybridKV 的分类判据是反的

P1-a / P2-a 完成（3 样本 × 64 帧，28 层 × 4 head = 112 单元，retention 12.38%）：
采集 `scripts/analysis/attention_trace_per_layer.py` → `results/routing_traces_per_layer/`；
分析 `scripts/analysis/analyze_per_layer_selection.py`。

### （1）per-layer headroom 是层均值估计的**两倍** —— 修正 §18.2b（G0.3）

| 样本 | coverage_static | coverage_oracle | **gap** |
|---|---:|---:|---:|
| 050-1 | 0.6157 | 0.7809 | **+0.1652** |
| 496-3 | 0.6219 | 0.7516 | **+0.1296** |
| 717-1 | 0.6659 | 0.7860 | **+0.1201** |
| **均值** | **0.6345** | **0.7728** | **+0.1383** |

§18.2b 用**层均值** trace 得到 gap **+7.1pp**、static/oracle **84.1%**；
**per-layer 实测 gap = +13.8pp、static/oracle = 82.1%**。
→ **"更好的 prefill 选择器"的真实上界约 +13.8pp，是层均值估计的两倍**；
G0.3 当时就声明"A per-layer trace is needed for the decisive number"，现在补上了，**数字翻倍**。
这**提高了 MM-ShiftKV（P1）的期望值**。

### （2）HybridKV 的**分类判据**在本数据上是反的

| θ | f_d（dynamic 占比） |
|---:|---|
| 0.90 | **0.79 – 0.90**（几乎全是 dynamic） |
| 0.80 | 0.36 – 0.42 |
| 0.70 | **0.09 – 0.11** |

**`pearson(S, churn)` = +0.255 / +0.300 / +0.221 —— 三个样本全为**正**。**
计划里写的是"HybridKV 假设**负**相关（高 S = 稳定头）"；**实测是高 S 的头反而更不稳定**。

**这与 §18.3.3 的正面结果不矛盾**：冻结高 S 头之所以有效，
**恰恰因为高 S 头正是 churn 更大的那些头**，冻掉它们去掉的漂移更多。

> **修正后的表述**：**"冻结/节流"这个机制有效（P2-b 已证）；
> 但"用 `S_{l,h}` 挑该冻结哪些头"这条规则在本数据上是反的（P2-a 已证）。**
> HybridKV 可用的部分是机制本身，不是它的头分类判据。

---

## 18.7 【第五批实测】MM-ShiftKV 的方差放大被否证：每个 γ 都比 attention 差，γ=10 最差 ★

P1-a 完成（3 样本 × 64 帧，V=7812，128 proxies/32 组/τ=0.95）：
脚本 `scripts/analysis/mmshift_proxy_recall.py`；产物 `results/mmshift_proxy/recall.md`。
指标 = **对解码期真实注意力质量的覆盖率**（括号内 = 与 attention 选择的 Jaccard）：

| 样本 | **attention cov** | oracle cov（上界） | γ=1.0 | γ=5.0 | **γ=10.0（论文推荐）** | γ=20.0 |
|---|---:|---:|---:|---:|---:|---:|
| 050-1 | **0.6144** | 0.7140 | 0.5814 (0.58) | 0.4838 (0.41) | **0.3962 (0.26)** | 0.4967 (0.40) |
| 717-1 | **0.6526** | 0.7168 | 0.6165 (0.56) | 0.4831 (0.37) | **0.4364 (0.27)** | 0.5052 (0.34) |
| 496-3 | **0.6112** | 0.6844 | 0.5800 (0.59) | 0.4814 (0.42) | **0.3886 (0.26)** | 0.4779 (0.37) |

**三条结论：**

1. **预注册门槛不通过**：没有任何 γ 超过 attention 选择；最好的 γ=1.0 也低约 **3.3pp**。
2. **方差放大有害，且论文推荐的 γ=10 最差**（比不放大低约 **20pp**，Jaccard 从 0.58 掉到 0.26）。
   → **直接否证论文核心前提在视频/STD 工作点上的适用性**：
   论文说"decode query 方差比 prefill 大 13–22×，所以要放大"；**实测放大会破坏选择质量**。
   这正是 §18.2 的 G0.2 预警的情形——那个比值是**图像**上测的，视频上不成立。
3. γ=1（只做分组投票、不放大）≈ attention 但不超过——合理，因为它与 attention 选择同源。

**⚠️ 保真度声明**：本实验在 **query 空间**放大（hook 只暴露 post-RoPE query/key，不暴露 hidden_states），
**不含论文的合成未来位置**。所以这**不是忠实复现**，而是"方差放大这个成分在视频上是否成立"的检验——
对该成分结论为负。hidden 空间版本不排除有差异，但**单调变差 + γ=10 最差**已足以否证其推荐取值。

### ✅ 五篇方法的最终判定（本轮完成）

| 方法 | 判定 | 关键实测 |
|---|---|---|
| **HybridKV** | **机制有效、判据反了** | 冻结 92/112 头 → 改写 −85%、接受率 +0.040、Δ 惩罚 −70%；但 `pearson(S,churn)=+0.25`（论文假设负） |
| **FlashCache** | **否证** | Dev 选择 Jaccard 0.068（随机基线 0.066）；耗时 3872–4440 ms（门槛 100 ms） |
| **MM-ShiftKV** | **否证** | 代理覆盖 0.581/0.617/0.580 < attention 0.614/0.653/0.611；γ=10 最差（−20pp） |
| **SimiCache** | **阻塞** | 论文付费墙（IEEE TMM，无开放版本） |
| **FastCache** | **分析排除** | 单请求下调度退化为零收益；压缩器需重训且仅 5× |

> **而整个调研里唯一"实测正收益"的，不是这五篇里的任何一篇，而是把 KV 预算调大**（§18.4，−13% decode）。

---

## 18.8 【最终确认】128 帧复验：预算效应成立，甜点在 ~10.7% 保留 ★★

`results/ksweep128/`（6 样本 × **128 帧 × 512 token**，即项目真实工作点；`timing_valid=false`）：

| retention | k | **static 接受率** | 接受长度 | **轮数** | ms/轮 | 其中 draft | decode 总时长 | **相对 6.09%** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6.09%（现状默认） | 967 | 0.8653 | 7.757 | 354 | 462.8 | 327.6 | 163.8 s | — |
| **10.66%** | 1693 | **0.9093** | 8.146 | 338 | 455.6 | 322.8 | 154.0 s | **−6.0%** |
| 21.06% | 3343 | **0.9463** | 8.489 | **325** | 468.8 | 334.6 | **152.4 s** | **−7.0%** |

**四条结论：**

1. **接受率单调上升**（0.8653 → 0.9463，+0.081）。**这彻底否证了 §18.2c 的疑虑**——
   在同一实验内改预算确实提高接受率，是**因果**关系，不是跨样本相关。
2. **轮数单调下降**：354 → **325（−8.2%）**，确定性量。
3. **每轮成本仍与 k 无关**：k 涨 **3.5×**，draft 只从 327.6 到 334.6 ms/轮（+2.1%，中间点反而更低），
   在 ±3% 争用噪声内**无趋势**。

   **✅ 配对成本探针（`scripts/analysis/sparse_attn_cost_probe.py`，同进程交替以抵消争用）**：
   k 涨 3.46×，唯一依赖 k 的稀疏注意力只涨 **1.79×**、绝对值 **0.027 ms**（占一轮 draft 的 0.01%）。
   **更稳健的物理解释**：draft 每步读 ~15 GB 模型权重（~20 ms @768 GB/s），
   而选中的 KV 只有 `k×2 KB` = **2.0 MB（k=967）→ 6.9 MB（k=3343）**，
   即 **≤0.05% 的权重读取量** → "每轮成本与 k 无关"是**结构性**的，不是噪声。
4. **饱和很快**：10.66% → 21.06%（k 再翻倍）只再换 **−1.0pp**。
   → **甜点 ~10.7% 保留**：拿到几乎全部收益，预算只用一半。

**→ 落地改动（零代码）**：默认 `--k-plus-text` 从 **1024 → ~1750**，decode 约 **−6%**。

**逐样本配对统计（n=6）**：6.09%→10.66% 每样本省轮 [3,6,2,2,0,3]（**5/6 同向**）、接受率 **+0.0440（无一样本为负）**；
6.09%→21.06% 省轮 [3,11,4,3,1,7]（**6/6 同向**，符号检验 p≈0.016）、接受率 **+0.0810**；
**10.66%→21.06% 只值 −1.1%（饱和）**。

**安全性验证（离线）**：三档的 **exact 均为 4/6**、**峰值显存均为 33.22 GiB**、`k+text ≤ 1750 ≪ cache-len 20480`
→ **加大预算在正确性、显存、缓存容量三项上零代价**。

⚠️ `timing_valid=false`，那两个百分比需在**空闲卡**复验；但**轮数 −8.2% 是确定性的**。

### 18.3 五种方法的定位与优先级

| 序 | 方法 | 可迁移点 | 需 attention 矩阵 | 官方代码 | 判定 |
|---|---|---|---|---|---|
| **1** | **MM-ShiftKV**（ACL 26 Findings） | variance-expanded query proxy + 分组投票替代 prefill 打分器；**免 `output_attentions`** | ❌ | ✅（Qwen2.5-VL 路径可用） | **主线** |
| **2** | **HybridKV**（ACL 26） | per-(layer,head) `S_{l,h}` + Eq.(5) 非均匀预算；STD **已物化**所需张量 → 零新增采集 | ❌ | ❌ | **做，低成本** |
| **3** | **FlashCache**（CVPR 26） | **DBA 逐层非均匀预算**（+5.0/+5.2 分，与打分器正交）；Dev 判据需先验 recall | ❌ | ❌ | **做 DBA** |
| **4** | **SimiCache**（IEEE TMM 26） | 相似度两步聚类 + 加权合并 | 推测❌ | ❌ | **阻塞：论文付费墙** |
| **5** | **FastCache**（arXiv 2503.08461） | **不作方法来源**；仅贡献 §18.2 门控实验的设计范式 | — | ❌ | **只取门控思路** |

**明确排除**：FastCache 的三块机制（可训练 MLP 压缩需重训且 5× 远弱于我们的 16.4%、单请求下调度无收益、预算分配不存在）；
HybridKV 的 CPU offload + 8-token chunk 检索（比 STD 单 token 粒度粗，128 帧下传输量大）。

**关键限制**：五篇里 **MM-ShiftKV 零视频实验**（全是单图，输入 642–4830 token）；
**FlashCache 视频只有 FAVOR-Bench 一项且 ρ=0.05 已退化**，而 STD 工作点 6.09% 更激进。

**走查清单**：阶段 0.1 已完成；下一步 P1-a（MM-ShiftKV 离线 coverage 对照 + 视频 prefill→decode 方差比）、
P2-a（`S_{l,h}` 分布与 `N_stat/N_dyna`）、P3-a（Dev recall + A6000 DCT 耗时）。

### 18.4 为什么 HybridKV 是性价比最高的移植点

STD 的 `build_sparse_selection` 拿到的 `[q_heads, text_len, prompt_len]` 注意力张量
**就是** HybridKV Eq.(2) 的 `S_text`，而且这笔 prefill 成本 static 路径**本来就在付**。
算 `S_{l,h}` 只是对既有张量多做一次 `topk(...).sum(-1).mean()` 后处理 → **零 decode 开销、零新增采集、不触碰 verifier 数值**。

更关键的是它**直接命中 dynamic 的真实瓶颈**：STD 的 dynamic 亏损里 **refresh 主导（≈34 ms/轮）**，
而优化后的采集只占 ≈4.7 ms/轮。所以"只对部分 head 采集"收益有限，
**必须同时"只对 dynamic head 做 refresh"**，节省才随 dynamic head 占比线性缩放。
（论文**从不报告 `N_stat/N_dyna`**，这正是 P2-a 要测的第一个数。）






---

## 19. W1：decode-only 优化三件套（2026-09-22 落地，待 GPU 验证）★

> 完整方案与判读方法：**[`docs/w1-decode-knobs.md`](docs/w1-decode-knobs.md)**
> 代码：`src/std_repro/std_qwen25vl.py`、`src/std_repro/dynamic_std_qwen25vl.py`、`scripts/benchmark_a100_dynamic.py`
> 测试：`tests/test_w1_decode_knobs.py`（43 条 CPU 回归，全绿；原有 324 条也全绿）

### 19.1 为什么是这三件，而不是继续调 KV 选择

local512_A 的 decode 账（10 样本 / 128 帧 / 512 token）：

| 组成 | 时间 | 占 decode | 每轮付几次 |
|---|---:|---:|---:|
| 稀疏 draft 前向 | 146.8 s | 71.8% | **9 次** |
| bonus 稀疏前向 | 16.3 s | 8.0% | 1 次 |
| dense verify | 41.3 s | 20.2% | 1 次 |
| selection / cache copy | 0.4 s | 0.2% | 一次 |

`c_s = 28.5 ms/步` 占每轮 **80%**，而整个稀疏 attention 只值 **0.83 ms（2.9%）**。
结论：**继续调"选哪些 KV"不可能改善 decode**；能动的只有"每步多贵"和"每 token 跑几步"。

### 19.2 三个开关

| flag | 改什么 | 是否保输出 |
|---|---|---|
| `--compile-draft` | 单 token 稀疏步套 `torch.compile(mode="reduce-overhead")`（CUDA graph） | ✅ 只产草稿，接受与否由 dense 决定 |
| `--draft-layers N` | 稀疏 draft 只跑前 N 层，再走模型自己的 final norm + LM head（早退） | ✅ 同上，draft 变弱只会降接受率 |
| `--fair-dense-verify` | **只对 batch 验证**（q_len>1）用 `enable_gqa` 取代 `repeat_kv` | ⚠️ **不保输出**：改了 verifier 数值，必须看 `Exact` 列 |

`--fair-dense-verify-mode`：`mask`（默认，语义与 stock 完全一致，只去掉 7× KV 物化）／`causal`（再丢掉 mask 改用 `is_causal=True`，才可能走融合 kernel，**但只有在 SDPA 把 causal 对齐到右下角时才等价**）。

**关键：AR 基线不受影响。** E3b 的 `--fair-dense-attention` 打的是 `q_len==1`（正是基线本身）；这里是 `q_len>1`，且补丁在 verify 之外处于 inert 状态，prefill 与 selection 分支碰不到。

### 19.3 顺手修掉的两个真 bug

1. **`use_compile` 是死代码**：`_build_compiled_sparse_step` 的闭包收 3 个参数，`_forward_tokens` 只传 2 个（缺 `position_ids`），且 `grep use_compile scripts/` 为空——从来没有任何调用点打开过它。
2. **`_draft_tokens` / `_append_sparse_tokens` 根本不走 compiled 路径**，所以即便打开也只影响 bonus 那 1/10 的前向。

现在：compiled 步走预分配缓冲（CUDA graph 需要稳定地址），且**失败即软退**——编译失败会回滚 sparse cache 写游标、永久禁用、返回 `None` 让调用方回落到 eager（`KVCache.cat` 先写后进位，所以回滚游标就是 decode 主循环本来就在用的回滚方式）。

### 19.4 怎么跑（两步）

```bash
# 第一步：先在这张卡上判定 causal 是否等价（5 分钟）
python scripts/analysis/verify_attention_variants.py --gpu 3 \
    --q-len 10 --kv-len 16460 --output results/w1_decode/verify_variants.md

# 第二步：等空闲 A6000，跑一因子一变的矩阵
GPU=3 LIMIT=10 bash scripts/run_w1_matrix_local.sh w1_decode_20260923
```

`verify_attention_variants.py` 同时给出**数值**（与 math 后端右下角 mask 参考实现的最大偏差）和**成本**（每步 ms），并且**偏差超容差时拒绝推荐 `causal`**。这是离线推不出来的部分，必须实测。

### 19.5 预期量级

```
decode ms/token = (S·c_s + c_v) / n        S = 10, c_d = 68.36 ms, n ≈ 8.95
decode ratio    = c_d · n / (S·c_s + c_v)
```

| 配置 | 预期 |
|---|---|
| `W1_compile` | `draft_time` −10~15% → 比值 **+9%**；接受率与 `Exact` **必须不变** |
| `W2…W5_layers{24,16,12,8}` | `draft_time` 随 N/28 缩放；**接受率才是观测量** |
| `W6_fairmask` | `verify_time` 下降；`Exact` 可能变动 |
| `W7_faircausal` | 同上但更大，**仅在第一步通过后** |

`draft_layers=16` 若接受率守住，比值可达 **≈2.8×**。

### 19.6 W1 解决不了什么（必须同时声明）

* **`c = 1/ρ = 2.40×` 的天花板不变**：draft 与 target 是同一个 7B，两边都付 ~22 ms/步的权重重读。W1 让 `c_s` 变小、比值向天花板靠，但抬不高天花板。
* **公平基线的归因不变**：与同样用 `enable_gqa` 的 AR 相比，诚实比值仍 ≈1.05×。
* **诚实稀疏收益仍是 1.10 ms/步**，W1 一点没碰。

### 19.7 验证状态

* 代码与 43 + 9 条 CPU 回归已完成；原有 324 条测试无回归（合计 **376 passed**）。
* **GPU 已执行**（2026-09-22 20:15 起，A6000 物理 GPU1，`timing_valid=true`）。三个开关的实测结论见 §19.8。

### 19.8 【实测】三个开关的判决（2026-09-22，A6000 GPU1，空闲卡）★

#### 19.8.1 `--fair-dense-verify` 是**负结果，不要用**

`scripts/analysis/verify_attention_variants.py --gpu 1 --q-len 10 --kv-len 16460`
（28 步/样本、20 reps × 4 轮交错、fp16；产物 `results/w1_decode/verify_variants.md` + `.json`）：

| 变体 | 与 math 参考等价？ | ms/验证步 | vs stock |
|---|---|---:|---:|
| `stock_repeat_mask`（现役） | ✅ 1.9e-06 | **43.36** | **1.000×** |
| `gqa_mask`（`--fair-dense-verify-mode mask`） | ✅ **0.0**（逐位相等） | 99.34 | **2.291×** |
| `gqa_boolmask` | ✅ **0.0** | 98.09 | 2.262× |
| `gqa_causal`（`--fair-dense-verify-mode causal`） | ❌ **0.325** | 98.05 | 2.261× |

两条结论，**方向都和跑之前的估算相反**：

1. **去掉 `repeat_kv` 让验证慢 2.3×，而不是快。** 验证必须带 causal mask，而**任何 mask 都会把 SDPA 踢出融合 kernel**；`enable_gqa` 随后落到 math 后端。这同时解释了 E1 里 `dense_gqa_mask`（113 ms）为什么是最差变体。q_len=10 时，省下的 6.6 GB 物化远小于失去的 kernel。
2. **`causal` 不等价，禁止使用。** 0.325 的偏差直接证明：在这套 torch/GPU 上，`q_len < kv_len` 时 SDPA 的 `is_causal` **不是**右下角对齐。闸门拦下了一个会静默算错的配置——这个模式留在代码里作为**已记录的负结果**，不是推荐项。

#### 19.8.2 `--compile-draft` 可用，编译一次性 ~50 s

`results/w1_decode/smoke_compile_layers16.jsonl`（1 样本 / 128 token）：

* `draft_compiled: true` 且 `draft_compile_note` **为空** —— `torch.compile` 成功 trace 了自定义 `KVCache`，CUDA graph 捕获成功，**没有触发软退**。
* 编译成本全部落在 warmup（12 轮 `draft_time` 52.3 s），measure 阶段已消失（121 轮 17.8 s）。

#### 19.8.3 `--draft-layers` 会打穿接受率：早退不是免费的

同一次 smoke，`--draft-layers 16`（28 层里的前 16 层）：

| | accepted length | 接受率 | 128 token 的解码轮数 |
|---|---:|---:|---:|
| 基线（28 层，已记录） | 8.16 | 0.87 | ~16 |
| `draft_layers=16` | 0.058 | **0.007** | **121** |

16 层草稿基本等于随机 → static decode（28.2 s）**比 AR（8.6 s）慢 3.3×**。
**第 17–28 层几乎承载了全部 next-token 信号**，砍层不是免费的 `c_s` 收益。`layers=24/26` 正在测，用来标出崩塌从哪一层开始。


#### 19.8.4 【端到端 A/B】compile 一点没快；layers=26 只砍 2 层就崩 ★

`results/w1_decode/w1_decode_20260922/`（3 样本 / 128 帧 / 512 token / 同卡同时段 / `timing_valid=true`；
对比表见同目录 `comparison.md`，工具 `scripts/analysis/compare_w1_configs.py`）：

| 运行 | W1 开关 | static/AR decode | `c_s` (ms/稀疏步) | 接受长度 n | 轮数 | exact |
|---|---|---:|---:|---:|---:|---:|
| `W0_base` | — | **1.855×** | **26.95** | 9.253 | 166 | 2/3 |
| `W1_compile` | `--compile-draft` | 1.849× | **27.04** | 9.253 | 166 | 2/3 |
| `W2_layers26` | `--draft-layers 26` | **0.358×** | 24.70 | **1.664** | **923** | 2/3 |

（`W0_base` 复现了已记录的 local512/E3a 水平：1.855× vs 记录值 1.711–1.873×，且样本 050-1 的
static 接受率 0.8716 与 09-21 记录逐位相同 → 环境可复现。）

**三条判决，全部为负：**

1. **`--compile-draft`：跑通了，但一点没快（±0.3%）。** `draft_compiled: true` 在 3/3 static
   measure trial 成立、`draft_compile_note` 为空，接受率/接受长度/轮数/exact **逐位不变**——
   完全符合"只改草稿、不改输出"的论证。但 `c_s` 26.95 → 27.04 ms。
   → **跑之前的估算错了**：那 ~7 ms 不是 kernel launch / Python 开销，28.5 ms 里有 ~27 ms 就是
   在流式读 15.45 GiB 权重；batch=1 单 token 前向是 **~100% DRAM-bound**，没有 launch 开销可捞。
2. **`--draft-layers`：`c_s` 严格按层数线性下降，但接受率崩盘。** 26/28 预测
   `c_s = 26.95 × 26/28 = 25.02 ms`，实测 **24.70 ms（差 1.3%）** → **步长就是"读多少字节权重"，
   没有隐藏常数**。代价是接受长度 9.253 → **1.664**、轮数 166 → **923**，比值塌到 **0.358×**
   （static 比 AR 慢 2.8×）。16 层更极端（0.058）。
3. **`--fair-dense-verify`：见 19.8.1**，慢 2.3× 且 `causal` 算错。

**合起来的结论比单个负结果有价值**：三个"不动权重字节"的开关全都无效——这**证明了**（不再是猜测）
decode 步就是权重读取本身。

#### 19.8.5 【修正】唯一剩下的杠杆：**只量化 draft**，它不花任何正确性预算 ★★

`c_s ≈ 1.74 ms / GiB`（26.95 ms ÷ 15.45 GiB），所以只有两条路能降 `c_s`：少层数（已证死）或
**每个权重少读字节**。

⚠️ **必须修正本项目此前的一个说法**："量化会破坏 bit-exact"——那只在**验证器**被量化时成立。
**只量化 draft 不破坏任何东西**：接受与否仍由 **fp16** dense verifier 的 argmax 决定，草稿变差
只会降低接受率。这和 `--compile-draft`／`--draft-layers` 的保输出论证**完全同一个**。

按实测的 `c_d = 68.0`、`c_v = 69.35`、`n = 9.253` 外推：

| draft 精度 | draft 权重 | `c_s` (ms) | 预测 static/AR decode |
|---|---:|---:|---:|
| fp16（现状） | 15.45 GiB | 26.95 | **1.855×**（实测） |
| int8 | 7.7 GiB | ≈ 14 | **≈ 3.0×** |
| int4 | 3.9 GiB | ≈ 8 | **≈ 4.2×** |
| 草稿免费 | 0 | 0 | 9.1×（渐近 `c_d·n/c_v`） |

**这条路能越过 `c = c_d/c_s = 2.5×` 的天花板**，因为 draft 不再和 target 读同样的字节。
两个必须实测、不能假设的点：① 量化草稿会提出更差的 token（`n` 会掉）；② 过了 int4，
`c_v`（69 ms/轮）成为主项，曲线变平。

**→ 这就是 W4 的第一件事，也是本项目第一条有量化预测、且不牺牲 exactness 的加速路径。**

---

## 20. EAGLE-3 免训练视频 SD：可行性实测（2026-09-23）★

> 完整记录：**[`docs/eagle3-video-probe-findings.md`](docs/eagle3-video-probe-findings.md)**
> 客户端（可用）：`scripts/analysis/eagle3_video_bench.py`

### 20.1 出发点：本地其实已经有零件

| 资产 | 路径 | 说明 |
|---|---|---|
| 大 target | `Qwen2.5-VL-32B-Instruct`（63 GB，64 层） | ParallelVLM 的 target |
| **预训练 EAGLE-3 draft** | `qwen2.5-vl-7b-eagle3-sgl`（1.8 GB，**1 层**） | 别人用 SpecForge 训好的，**对我们是免训练** |
| 推理引擎 | sglang 0.5.2rc2（`sglang-eagle3` 环境） | 原生支持 EAGLE3 |

那个 draft 的 README 自报：**accept length 3.219、e2e 1.5×**（MMStar，图像）。

### 20.2 结论：**开箱即用不成立**。6 次启动，4 类不同崩溃

| # | 配置 | 死在哪 |
|---|---|---|
| 1 | README 原命令（`--cuda-graph-max-bs 1`） | graph 捕获：dynamo fake-tensor `view` 错 |
| 2 | `--disable-cuda-graph` | 视频 prefill 后 verify 拿到 **0 token** → `view(0,-1,128)` |
| 3 | 默认 graph | 捕获 bs=8：mrope `view` 非连续错 |
| 4 | + `view`→`reshape` 补丁 | 捕获**全过**（8 个 bs）；接着 `mrope_positions` 广播错（graph bs=1 缓冲 vs 24 个 draft token） |
| 5 | 补丁 + 禁 graph | 又回到 `reshape(0,-1,128)` → **0-token 是逻辑 bug，不是连续性 bug** |
| 6 | — | 停止，**sglang 源码已从备份还原**（验证：2 处 `view`、0 处 `reshape`） |

### 20.3 两条与崩溃无关的结构性发现

**① draft 的上下文只有 8192 token，sglang 直接拒绝配对**（运行时原文）：

```
ValueError: Target model's context_length (128000) is greater than the derived context_length (8192)
```

| prompt | token 数 | vs 8192 |
|---|---:|---|
| 128 帧 | **15,948** | **超 1.95×** |
| 64 帧 | **7,884** | 0.96×，刚好塞下 |

`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` 只去掉守卫、去不掉能力差距 —— 而"超窗口后接受率掉多少"正是 64 vs 128 实验要测的东西。

**② 这个 sglang 版本的本地视频输入路径是坏的**：`estimate_frames`（base_processor.py:276）认 `video:` 前缀，但 `_load_single_item`（:309）把字符串**原样**交给 `load_video`（utils.py:741），后者只认 bytes / http / `data:` / 裸路径。于是：裸路径→被当成 1 帧→"More image data items than tokens"→0-token 崩溃；带前缀→base64 "Incorrect padding"。**这一版对 Qwen2.5-VL 视频只支持 base64 `data:` 或 http URL。**

### 20.4 下一步（按成本排序）

1. **把切片过 HTTP 服务**（`python -m http.server`）再传 http URL —— 最便宜的下一次尝试
2. 或 base64 `data:` URL（64 帧 2 fps 重编码 <1 MB，可行）
3. 帧数先卡在 **64**（7,884 token）；128 帧要先解决 draft 的 8k 窗口
4. EAGLE-3 若在 sglang 里持续失败，退回 **32B target / 7B draft**（ParallelVLM 那一对，两个模型都在本地）：只需 TP=2 + runner 支持 batch，不需要 EAGLE 机制。按一阶模型 TP 分片后 **c ≈ 1.95 → ≈1.45×**

> 全程未改动任何模型文件；未触碰你正在跑的任务；sglang 源码改动已还原。

---

## 21. vLLM 上的 Qwen2.5-VL 视频推理与 STD 投机解码（2026-09-26）★

> 完整分阶段记录：**[`docs/vllm-std/`](docs/vllm-std/)**（`00-environment.md` … `07-attribution.md`）
> 脚本：`scripts/vllm-std/`；原始日志与数据：`results/vllm-std/`
> **性质**：与 §1–§20 的 SpecVLM/SGLang 路线**并行**的一条新主线，目标是**在 vLLM 上**复现 STD。

### 21.0 任务书三处前提的修正（先读）

| 任务书假设 | 实测 | 处置 |
|---|---|---|
| 需要找 GPU 机器（暗指远程） | **本机 `A6000` 就是 A6000 环境**（6×A6000 48G + 2×4090，驱动 550.163.01） | 主战场定本机；远端 gpu23 无外网、驱动更旧（535），弃用 |
| `git clone` vLLM | `github.com` git clone **超时**；`codeload.github.com` 可用 | release tarball + `git init` 固定版本 |
| `draft_model` 跑多模态会 `NotImplementedError` | v0.22.0 里多模态**只是 warning**（`_warn_if_multimodal`）；硬错是 **M-RoPE**（`_raise_if_mrope`） | 阶段 2/7 实测确认 |

### 21.1 最关键的环境结论：为什么必须用 `+cu129` 而不是 PyPI 默认 wheel ★★

要拿到 `speculative_config.method="custom_class"` 这个注入点，**vLLM 必须 ≥ 0.22.0**（0.21.0 还没有）。但：

- vLLM ≥ 0.22.0 锁 `torch==2.11.0`，**PyPI 上 torch 2.11.0 是 cu13 构建**（依赖 `nvidia-*-cu13`）；
- 本机驱动 **550.163.01 = CUDA 12.4**，**加载不了 CUDA 13 运行时**。

这不是推测，是实测：既有环境 `vllm-omni-dev-fast`（vllm 0.28.0）`import vllm` 直接报

```
ImportError: libcudart.so.13: cannot open shared object file: No such file or directory
```

**解法**：CUDA **12.9** 属 CUDA 12.x，受 *minor version compatibility* 覆盖（驱动 ≥525.60.13 即可），而 vLLM 与 PyTorch 都发布了非默认 CUDA 的 wheel：

```
vllm-0.22.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl   (wheels.vllm.ai)
torch-2.11.0+cu129-cp311-cp311-manylinux_2_28_x86_64.whl (download.pytorch.org)
```

→ **拿到真实（未打补丁）的 `custom_class` 注入点，且能在驱动 550 上跑。** 实测通过（§21.2）。

> 附注：`wheels.vllm.ai/<ver>/cuNNN/vllm/` **不是 PEP 503 index**（只是目录列表），`pip install vllm==0.22.0+cu129 --extra-index-url` 会报 `No matching distribution found`，必须直链取 wheel。

### 21.2 阶段进展

| 阶段 | 状态 | 结论 | 证据 |
|---|---|---|---|
| 0 环境勘察 | ✅ GO | vLLM 0.22.0+cu129 / torch 2.11.0+cu129 在驱动 550 上跑通 | `docs/vllm-std/00-environment.md` |
| 1 纯视频基线 | ✅ GO | 32 帧→19,166 prompt tokens，128 tokens 用时 **5.1653 s ± 0.0530 s**（3 次），**bit-exact** | `docs/vllm-std/01-baseline-video.md` |
| 2 接口分析 | ✅ GO | **多模态 target 能配自定义 Proposer，且无需补丁**：`custom_class` 不加载 draft model，`_warn_if_multimodal`/`_raise_if_mrope` 两道闸门都长在 `LLMBaseProposer.__init__` 里，永不执行 | `docs/vllm-std/02-spec-decode-interface.md` |
| 3 Dummy Proposer | ✅ GO（标准已修正） | 接口接通（381 组草稿到达验证层）；**发现关键反例：bit-exact 在本机不可能成立** | `docs/vllm-std/03-dummy-proposer.md` |
| 4 STD 单 token | ⚠️ 部分 | **架构性阻塞**：`custom_class` 是纯 CPU/token 级接口，拿不到 model/KV/attention → 装不下 STD 稀疏 KV 草稿；已给最小 patch 路线 | `docs/vllm-std/04-std-1token.md` |
| 5 STD 多 token | ⚠️ 负结果 | 接受率恒 **0.0**；`k=1..8` 无收益；窗口曲线受测量混淆（已定位） | `docs/vllm-std/05-std-multitoken.md` |
| 6 batch 扫描 | ⚠️ 部分 | b1–b8 完成，**无真实稀疏收益**；b16/b32 因吞吐崩塌（2–4 tok/s）中止，有日志根因 | `docs/vllm-std/06-batch-sweep.md` + `results/vllm-std/batch_sweep.csv` |
| 7 对照归因 | ✅ GO | **交错配对 0.9227 ± 0.03×（慢 7.7%）**；`draft_model` 实测死于 M-RoPE（阶段 2 预测确认） | `docs/vllm-std/07-attribution.md` |
| 8 总结交付 | ✅ 完成 | 汇总报告 + 复现配方 + 下一步 | `docs/vllm-std/08-final-report.md` |

### 21.7 阶段汇总表（任务书要求格式）

| 阶段 | 状态 | 命令 | 证据路径 | 结论 | 阻塞 | 下一步 |
|---|---|---|---|---|---|---|
| 0 环境 | ✅ | `bash scripts/vllm-std/00_setup_env.sh` | `docs/vllm-std/00-environment.md` | vLLM 0.22.0+cu129 在驱动 550 跑通 | 无 | 阶段1 |
| 1 纯视频基线 | ✅ | `run_bench.sh … 01_baseline_video.py --num-frames 32 --max-tokens 128` | `docs/vllm-std/01-baseline-video.md`；`results/vllm-std/01_baseline_video.jsonl` | 19,166 prompt tok；**5.1653±0.0530 s**；3 次 bit-exact | 无 | 阶段2 |
| 2 接口分析 | ✅ | 只读源码 | `docs/vllm-std/02-spec-decode-interface.md` | 多模态 target **可**配 custom_class，无需补丁；真正死点是 M-RoPE | 无 | 阶段3 |
| 3 Dummy Proposer | ✅ | `run_bench.sh … 03_bench_spec.py --proposer …EmptyProposer/RepeatProposer` | `docs/vllm-std/03-dummy-proposer.md`；`03_bitexact.json` | 接口通；**bit-exact 不可满足（平局）**；框架税 4.8% | 无 | 阶段4 |
| 4 STD 单 token | ⚠️ | `… --proposer …SparseContextDraftProposer --num-spec-tokens 1` | `docs/vllm-std/04-std-1token.md` | 接受率 0.0；**接口无法承载 STD 稀疏 KV 草稿** | 架构：无 model/KV/attn 句柄 | 阶段5 |
| 5 STD 多 token | ⚠️ | `bash scripts/vllm-std/05_sweep.sh` | `docs/vllm-std/05-std-multitoken.md` | `k` 无收益；接受率 0.0；窗口曲线被混淆 | 草稿不匹配自由文本 | 阶段6 |
| 6 batch 扫描 | ⚠️ | `bash scripts/vllm-std/06_batch_sweep.sh` | `docs/vllm-std/06-batch-sweep.md`；`batch_sweep.csv` | 无真实稀疏收益；框架税支配 | **共享机**：b16/b32 吞吐崩塌 | 阶段7 |
| 7 对照归因 | ✅ | `bash scripts/vllm-std/07_paired_ab.sh 2` | `docs/vllm-std/07-attribution.md`；`07_paired.json` | **0.9227±0.03×（慢7.7%）**；加速来自框架税且为负 | 无 | 阶段8 |
| 8 总结 | ✅ | — | `docs/vllm-std/08-final-report.md` | 目标1/2 成功；目标3 未达成（根因清晰） | — | P0: GPU 侧 proposer |

### 21.8 数值速查（全部同机同输入，预热后重复）

| 配置 | 耗时 | vs eager | 接受率 |
|---|---:|---:|---:|
| eager baseline | 5.2305 ± 0.0406 s | 1.000× | — |
| 空草稿（框架税） | 5.4807 ± 0.0327 s | **0.954×** | — |
| 草稿全被拒 | 7.5295 ± 0.1534 s | 0.695× | 0.0 |
| 稀疏 k=1 | 5.5080 ± 0.0834 s | 0.950× | 0.0 |
| 稀疏 k=4 W=256 | 5.4038 ± 0.0420 s | 0.968× | 0.0 |
| 稀疏 k=8 W=256 | 5.4231 ± 0.0368 s | 0.964× | 0.0 |
| **交错配对（权威）** | — | **0.9227 ± 0.03×** | **0.0** |

### 21.9 三条必须记住的教训

1. **参考答案可能不稳定**：两次 eager 运行（仅差 CUDA graph）在同一 token 分歧，
   该位置 top-1/top-2 logprob **精确相等**。`temperature=0` 不保证唯一答案。
   → bit-exact 必须按 `(一致, 平局分歧, 稳定分歧)` 三元组报告。
2. **共享机的单点计时不可信**：阶段 5 出现过 1.260× 假加速，取证发现那一次跑在独占空卡上；
   交错配对复测同一配置得 0.90–0.94×。**跨时刻必须配对测量。**
3. **`custom_class` 不是万能注入点**：它是 CPU/token 级接口，
   适合 n-gram 类草稿；任何需要 model/KV/attention 的机制（含 STD）都必须走 GPU 侧 proposer。

### 21.5 ★ 阶段 3 的核心发现：`temperature=0` 不保证唯一答案，bit-exact 标准必须修正

实测（`docs/vllm-std/03-dummy-proposer.md` §3.4）：

- `RepeatProposer` 发 381 组 / 1524 个草稿，**接受 0 个**（所有 token 都由 target 自己选），
  输出**仍与 eager baseline 不一致**，且两次独立复现**分歧点完全相同**：token 62，`16445 → 448`。
- 定位到该位置是**精确平局**：
  - 启用 CUDA graph：`' featuring'` -0.666198 vs `' with'` -0.791198（margin 0.125000）
  - `--enforce-eager`：`' with'` 与 `' featuring'` **都是 -0.726219，margin = 0.000000**
- **决定性对照**：两次**都不带投机解码**的 eager 运行，唯一差别是是否用 CUDA graph，
  **同样在 token 62 分歧**（`16445` vs `448`）。

**→ 参考答案自身不稳定。在平局位置，argmax 由浮点噪声决定，任何实现都无法做到 bit-exact。**
这不是 proposer 的 bug，也不是 vLLM 的记账 bug，而是「greedy = argmax」在有限精度下的固有缺陷。

margins 画像（70 个位置）：margin<0.25 有 3 个（4.3%），margin<0.5 有 9 个（12.9%），margin<1.0 有 17 个（24.3%）。

**方法论修正（后续阶段强制）**：bit-exact 结论一律报三元组
`(完全一致位置数, 平局位置分歧数, 稳定位置分歧数)` + margin 画像；**只有第三项 > 0 才是真缺陷**。

### 21.6 框架税（阶段 3 已量出的硬数字）

| 配置 | 耗时 | vs baseline |
|---|---:|---:|
| eager baseline | 5.2305 ± 0.0406 s | 1.000× |
| 开投机但**零草稿**（EmptyProposer） | 5.4807 ± 0.0327 s | **0.954×** |
| 开投机、草稿**全被拒**（RepeatProposer） | 7.5295 ± 0.1534 s | **0.695×** |

开投机本身就有 **4.8% 固定开销**（疑与「custom_class 强制关闭异步调度」有关，阶段 7 归因）。
**阶段 4–6 的任何收益都必须先跨过这 4.8% 的地板。**

### 21.3 阶段 1 的两个可复用发现

1. **视频必须走 chat 路径**：`llm.generate(multi_modal_data={"video": "<path>"})` 传裸路径会 `AssertionError: ... got: '/'`（parser 把字符串**逐字符**迭代）。正解是 `llm.chat` + `{"type":"video_url","video_url":{"url":"file://..."}}` + LLM 构造时 `allowed_local_media_path`。
2. **`fps` 会被静默封顶**：`VideoMediaIO.__init__(..., num_frames: int = 32)` 是形参默认值，只传 `fps` 时合并逻辑弹不掉它，采样仍被 32 帧封顶（实测 `fps=1` 与 `fps=4` 都得 19,164 tokens）。显式 `num_frames=-1` 才放行（fps=4 → 151,920 tokens）。**做帧数消融必须显式写 `num_frames`。**

### 21.4 环境事实（供复现）

- 独立 conda env `vllm-std`（Python 3.11.16）；源码树 `third_party/vllm` 分支 `feat/std-proposer`，基线 tag `v0.22.0-pristine`（commit `028c47b`）。
- **必须** `CUDA_HOME=/usr/local/cuda-12.6`：`/usr/local/cuda` 指向 nvcc **11.8**，flashinfer 首用 JIT 会因 `#error "CUDA versions below 12 are not supported."` 直接让引擎起不来。
- 本机是**共享机**，空闲卡每分钟都在变；脚本用 `scripts/vllm-std/pick_gpu.sh` 选卡并记录 `gpu_before/after`。**绝不 kill 他人进程。**

---

## 22. 分叉决策执行：三条并行线（2026-09-26）

> 文档：`docs/vllm-std/09-lineB-attention-probe.md`（线 B 判定）、
> `docs/vllm-std/10-negative-result-skeleton.md`（线 C 骨架）
> 线 A 更正已并入 `docs/vllm-std/07-attribution.md` §7.9

### 22.1 线 B（go/no-go）：✅ **能拿到 → STD 在 vLLM 上有路**

**不需要给 vLLM 打补丁。** `Attention.forward`
（`vllm/model_executor/layers/attention/attention.py:437`）是** Python 方法**，
在派发融合 kernel **之前**就持有真实 Q/K/V。

实测（`results/vllm-std/B_attn_weight_probe.json`）：

```
prompt_tokens      : 2420
chosen q / kv      : 2420 / 2420
key_is_full_prompt : True                 <-- K 覆盖完整 prompt
statistic_shape    : [4, 2420]            <-- == build_sparse_selection 的输出形状
mass_check         : 1792.0 / 1792.0      <-- 精确吻合（64 text行 × 7组头 × 4 KV头）
hook_calls         : prefill=84, decode=196
```

**三重校验**：① 钩子在 prefill/decode 两条路径都触发；② K 覆盖完整 prompt；
③ softmax 重建的统计量**质量校验精确通过**——说明「后端不返回权重」**不构成障碍**。

**3 个硬性前提**（均实测踩到并解决）：
1. 钩子跑在 EngineCore 子进程 → 需 `VLLM_ENABLE_V1_MULTIPROCESSING=0`（或 IPC 回传）；
2. chunked prefill 会让 K 只剩当前块 → 需 `enable_chunked_prefill=False`（或累积 K）；
3. 钩子必须 **trace-safe**（torch.compile 会 trace 穿过它——这一点是被一次 `.tolist()` 崩溃反向证明的）。

**1 个必须提前解决的问题**：朴素实现的 selection 开销为单层 157 ms（64 text行 × 2420 kv）；
但理论量仅 ~8.8 GFLOP/全部层 → **是实现开销而非基本成本**，必须融合化。
**P0 验收已定死：selection 开销 ≤ prefill 的 10%，否则止损转 EAGLE-3。**

### 22.2 线 A（独占卡复测）：⚠️ `0.9227×` 是悲观偏差，修正为 **`0.9718 ± 0.0096×`**

| 环境 | 配对加速比 |
|---|---:|
| 共享卡（原 §7.4） | 0.9227 ± 0.0300× |
| **独占 A6000（GPU0）** | **0.9718 ± 0.0096×** |

- **用户「共享卡可能在压低 0.9227×」的怀疑成立**：−7.7% → −2.8%。
- **但定性结论不变**：仍 < 1，两次配对方向一致，**没有加速**。
- **对外数字应用 0.9718×**，并说明共享卡引入约 5pp 悲观偏差。
- **未完成**：第 3 对配对、batch b1–b8 独占曲线、b16/b32 独占复测
  —— GPU0 独占窗口被他人抢占后关闭（free 48658→9334 MiB）。**未外推填充。**

### 22.3 线 C（负面刻画）：⚠️ 三条「硬料」中**两条不存在、一条方向相反**

核查 `PROGRESS.md` 原文后：

| 用户表述 | 实际 | 判定 |
|---|---|---|
| 「batch=1 的 DRAM 带宽上限（92% 峰值，双流 1.004×）」 | 「92% 峰值」「双流 1.004×」**搜不到** | ❌ 数字不存在 |
| 「DRAM 带宽上限」这一定性 | §19 原文：57 MB/71 ms → **0.8 GB/s**，比 A100 峰值低 **~2500×** → **瓶颈不是带宽，是碎片化发射 + 强制同步** | ❌ **方向相反** |
| 「SGLang vs vLLM 框架税对照（0.9227× 跨框架自洽）」 | `0.9227×` 是 vLLM 单框架实测；全仓库唯一 sglang 数字是 **1.45×**，且是**外推预测非实测** | ❌ 跨框架自洽不存在 |

**处置**：骨架只采用**已核实**的 4 条材料（① 参考答案不稳定 ② 框架税为正
③ 接口能力≠算法需求 ④ 解码是权重读取而非带宽受限，来自仓库 §19）。
`10-negative-result-skeleton.md` §C.1 完整记录了这次纠正。

**写之前仍缺**：独占卡 batch 曲线、≥3 对配对、融合后的 selection 成本、**一个接受率 > 0 的配置**、跨框架对照。
→ 材料足以支撑 **characterization/negative-result 短文**；**不足以**支撑「STD 在 vLLM 上是否有效」。
**这是两篇文章，不要混。**

---

## 23. P0 决策阶段：selection 开销拆解（2026-09-26）

> 文档：`docs/vllm-std/11-p0-environment.md`、`12-157ms-breakdown.md`、`13-p0-gonogo.md`、`16-characterization-draft.md`
> 数据：`results/vllm-std/profile_32frame.json`、`profile_single_layer.json`

### 23.1 ★ 核心结论：**157 ms 是测量假象，原命题被推翻**

原探针用 `perf_counter + torch.cuda.synchronize()` 在**引擎内部**计时，
那次 sync 把**引擎自己排队的 GPU 工作**一并等掉了。同形状在 GPU 空闲上下文中用 **CUDA event** 复测：

| 形状（64 text 行，单层） | 原报告 | 复测 |
|---|---:|---:|
| kv = 2,420 | **156.6 ms** | **0.65 ms** |
| kv = 19,164（真实 32 帧） | — | **2.92 ms** |

**≈240× 的假象。**

### 23.2 A / B / C 拆解（真实工况 32 帧，kv=19,164，5 次重复）

```
variant       A_gpu    A_std   B_host     wall   A/wall      ×28   /prefill
core          3.775    1.854    0.036    3.811    99.1%    105.7     11.7%
core+book     3.170    0.005    0.037    3.207    98.8%     88.8      9.8%
naive         2.918    0.011    0.032    2.950    98.9%     81.7      9.1%   <-- 采用
```

| 块 | 时间 | 占比 |
|---|---:|---:|
| **A：GPU kernel** | **2.918 ms** | **98.9%** |
| **B：CPU 调度/同步/launch** | **0.032 ms** | **1.1%** |
| **C：内存拷贝** | **0.000 ms** | **0.0%** |
| unknown | 0 | 0% |

**三块合计 100%**（验收要求 ≥90%）。**逐项排查结论：没有任何 host 侧坏习惯是主导项。**

**prefill 本身 = 901 ms**（cache-cold）。**selection / prefill = 9.1%（naive）～11.7%（core）。**

### 23.3 判定：**NO-GO（对"融合"这条工程路线）**，但理由与规则设想相反

- **A 占比 98.9% > 60%** → 按规则字面 **NO-GO**，不启动阶段 3。
- **但因果与规则设想相反**：规则以为「A 高 = selection 重 = 救不了」；
  实测是「**A 高 = 已经没有 host 开销可省 = 融合无意义**」，**而绝对开销本来就已达标**。
- **融合天花板收益 ≤1.1%**（B 的占比）。**不建议投入一个 CUDA kernel 的工程量。**
- ⚠️ **`×28` 是外推**（只实测 `layers.0`）。未当实测数字使用。

### 23.4 判定对项目的意义：**阻塞点不在 selection 成本**

selection = 81.7 ms/prompt ≈ prefill 的 9.1%、端到端（基线 5.23 s）的 **1.6%**。
**即使一分钱不优化也已在预算内。** 真正的阻塞点回到既有硬数字：

| 阻塞点 | 数字 | 出处 |
|---|---|---|
| 草稿接受率恒为 0 | `accept_rate = 0.0`（所有 k、所有窗口） | `05-std-multitoken.md` |
| 投机固定框架税 | 零草稿 **0.954×** | `03-dummy-proposer.md` |
| 参考序列在平局处不稳定 | token 62 精确平局；eager-vs-eager 同样分歧 | 同上 §3.4 |

### 23.5 阶段 0 的失败记录（未掩盖）

| # | 事件 | 根因 | 处置 |
|---|---|---|---|
| 1 | profiler 报 `A_kernel=0.003 ms` | `device_time_total` 对小 kernel 严重低报 | 改用 CUDA event；profiler 仅用于 op 清单 |
| 2 | 抓到 8192 warmup 桶 / 256 pass 而非真实请求 | 钩子取首/取最小 | 改为「取 ≤ prompt_tokens 的最大 prefill」 |
| 3 | **误杀线 A 正在运行的引擎** | 只用属主+进程名判断，而我同时有两个作业 | 已记录；新规则：**必须用启动时间与作业时间窗交叉比对**；线 A 自动重试未受影响 |
| 4 | 阶段 0 曾出现 41GB EngineCore 残留 | driver 被 SIGTERM 只杀 shell，python 子进程存活 | `run_bench.sh` 加 `trap cleanup EXIT INT TERM` 终止子进程；脚本内 `try/finally shutdown()` |

### 23.6 并行线与待决

- **线 A**：独占卡 batch 扫描运行中（GPU2 独占，`p6x_*` tag）→ `results/vllm-std/batch_sweep_exclusive.csv`
- **线 C**：characterization 短文草稿完成（`16-characterization-draft.md`），只用已核实材料，
  已更正「DRAM 带宽上限」为「碎片化发射+强制同步」。
- **待用户决策**：路 A（做接受率探针，不做融合，约半天）vs 路 B（直接转 EAGLE-3）。

---

## 24. P0 路 A：接受率前置探针（2026-09-26）

> 文档：`docs/vllm-std/14-acceptance-probe.md`
> 数据：`results/vllm-std/P2_acceptance_probe.json`、日志 `logs/P2_acceptance2.log`
> 脚本：`scripts/vllm-std/P2_acceptance_probe.py`

用户选择路 A（做接受率探针，不做融合）。**结果找到了 `accept_rate = 0.0` 的机制性原因。**

### 24.1 ★ 核心发现：STD 的评分判据与「预测下一个 token」错位

`build_sparse_selection` 用 `grouped.sum(dim=(1,2))` —— **对所有 query 位置求和**。
但 token n+1 只取决于**最后一个 query 行**。

4 帧 / prompt=2422 / 视频 token=2392 / text 行=32 / **8 层**：

```
    K |  STD all-query mass_recall |  last-query oracle |  std overlap
   16 |             0.0824 +/- 0.0430 |             0.4668 |       0.2578
   32 |             0.1083 +/- 0.0405 |             0.5595 |       0.2695
   64 |             0.1461 +/- 0.0405 |             0.6520 |       0.2676
  128 |             0.1761 +/- 0.0463 |             0.7363 |       0.2939
  256 |             0.2464 +/- 0.0771 |             0.8133 |       0.3398
  512 |             0.3086 +/- 0.0891 |             0.8841 |       0.4199
```

- **同样 K 下，STD 判据保住的「最后一行注意力质量」只有 last-query 判据的 ~1/3**
  （K=256：**24.6% vs 81.3%**，差 3.3×）。
- **STD 集合与 last-query oracle 的重合率仅 26%–42%。**
- 即 STD 把「对整段文本重要」当成了「对下一个 token 重要」。

### 24.2 结论边界（不夸大）

本探针测的是**前置条件（注意力质量保真度）**，**不是接受率本身**。
严格表述：**「有强证据表明接受率会接近 0，但接受率仍未实测」**。
把它变成实测仍需一次稀疏前向 —— 这一点**没有被掩盖**。

### 24.3 与 §19 的张力（已标注，不强行调和）

`PROGRESS.md` §19 记录静态 STD 在部分样本上接受率 0.966–1.000，与本次恒 0 **不是同一个东西**：
§19 是**完整 HF 实现 + 自建稀疏 KV 前向**；本次 `SparseContextDraftProposer` 是
**token 级 n-gram 类比，根本没用稀疏 KV**。
**故本探针不能解释 §19 的 0.966，也不能否证它。** 要判定需在本视频上真跑一次 STD 稀疏前向。

### 24.4 下一步（需用户定夺，不擅自启动）

- **路 A1**：先修判据（全 query 求和 → 最后一行/最后 K 行），重测 §24.1 的表（约 1 小时）。
  若 K=256 保真度从 24.6% 升到 ~80%，说明**问题在判据而非稀疏性**，值得再投一次稀疏前向。
- **路 A2**：直接转阶段 4（EAGLE-3 + 负面刻画），并把「判据错位」作为材料 ③ 的强化版写进短文。

### 24.5 线 A（独占卡 batch 扫描）

运行中（GPU2 独占，tag `p6x_*`）。已完成 b1/b2 的 eager+spec。
产物待其结束后生成 `results/vllm-std/batch_sweep_exclusive.csv`。

---

## 25. 判定 §19 判据 → A1 分支决策（2026-09-26）

> 文档：`docs/vllm-std/17-section19-criterion.md`、`18-a1-decision.md`
> 方法：**只读代码**，逐条给出文件路径 + 行号 + 原文

### 25.1 先更正一处我自己的错误归属 ★

任务书问「**§19** 那次接受率 0.966–1.000 的静态 STD」。**§19 里没有这些数字。**

| 位置 | 实际内容 |
|---|---|
| `PROGRESS.md:1514` `## 19. W1：decode-only 优化三件套` | `c_s`/draft-layers/compile 三个 decode 开关，**无接受率表** |
| `PROGRESS.md:642` `## 17. 动态路由` | **0.507–1.000 的 static 接受率表在这里**（`:690-698`） |

错误归属是我在 `14-acceptance-probe.md` §3′.5 写的，**已在该文件内更正并标记作废**。

### 25.2 判据答案：**是（全 query 求和）**

`src/std_repro/std_qwen25vl.py`，函数 `build_sparse_selection`（def L185）：

```python
# L219-243
grouped = attn.reshape(num_key_value_heads, heads_per_kv,
                       attn.shape[-2],   # text_len（全部 text query 行）
                       attn.shape[-1])
# L220 注释：[batch, query_heads, text_len, prompt_len]
scores = grouped.sum(dim=(1, 2), dtype=torch.float32)      # L239 ★
```

`dim=(1,2)` = 对 **heads_per_kv** 与 **text_len（全部 query 位置）** 求和
→ **与探针假设逐字相同**。

**链条**：`:239` ← 调用点 `:1224`（在 `std_generate_qwen25vl`，def `:1118`）
← `scripts/benchmark_a100_dynamic.py:330`（`method=="static"` 臂）← §17 表的 static 列。

**且是真稀疏前向**：`SparseDraftController`（`:268`，注释「Monkey-patch ... sparse draft behavior」）、
`sparse_attn_mode ∈ {repeat_sdpa, gqa_sdpa, triton_gqa}`（`:273`）、`compact_sparse_prompt_cache`（`:302`）。
**且接受率对密集验证测**：逐 token 前缀比对（`:1033-1039`）、`acceptance_rate`（`:96-99`）。

### 25.3 ★ 判定：**分支 A → A1 不执行**，探针假设被证伪

| | 我的探针 | §17 静态 STD |
|---|---|---|
| 判据 | `grouped.sum(dim=(1,2))` | **同一条，逐字相同** |
| 真稀疏前向 | ❌ 否 | ✅ 是 |
| 测的是 | **前置条件**（最后一行质量保真度） | **接受率本身** |
| 结果 | K=256 时 24.6% → **推测**接受率≈0 | **实测 0.507–1.000，均值 ≈0.81** |

**直接观测量胜。推测被否证。A1 不执行**（阶段 3 不触发，不产出 `19-a1-result.md` / `a1_table.csv`）。

**探针错在哪**：它隐含假设「只有最后一个 query 行重要」。`build_sparse_selection`
对全部 text 行求和（text_len 通常仅 15–30 token），实践中够用。
**是代理指标构造有误，不是 STD 判据有误。**

**必须同时说清**：接受率**并不总是高**（范围 0.507–1.000）。正确表述是
「部分样本达 0.966–1.000，整体均值 ≈0.81」，而**不是**「永远接近满分」——但仍足以证伪「≈0」。

### 25.4 那「token 级接受率恒为 0」怎么解释

§17 是**真稀疏 KV 前向**（接受率高）；本次 `SparseContextDraftProposer` 是
**token 级 n-gram 类比**（接受率 0）。两者是**两个不同的东西**，不矛盾。
**问题不在判据，在「token 级 n-gram 类比 vs 真稀疏前向」的差异。**

下一步（不在本任务内，需另行决策）：做**真稀疏前向**测接受率，或转 EAGLE-3。**A1 路线已关闭。**

### 25.5 阶段汇总

| 阶段 | 状态 | 命令 | 证据路径 | 结论 | 阻塞 | 下一步 |
|---|---|---|---|---|---|---|
| 0/1 定位判据 | ✅ | 只读代码（grep/read） | `17-section19-criterion.md` | **是**：全 query 求和，逐字相同 | 无 | 阶段2 |
| 2 分支决策 | ✅ | — | `18-a1-decision.md` | **分支 A：A1 不执行** | 无 | 真稀疏前向 or EAGLE-3 |
| 3 A1 | ⛔ 未触发 | — | — | 判据无需修复 | — | — |
| 4 短文更新 | ✅ | — | `16-characterization-draft.md` §C.6 | 改为「**代理指标失效**」的方法论结论 | 无 | — |

**一句话总结**：§17（任务书写作 §19）静态 STD 用的**就是** `grouped.sum(dim=(1,2))` 全 query 求和，
且实测接受率 0.507–1.000（均值 ≈0.81），**故 A1 不执行** —— 被证伪的是我的代理指标，不是 STD 的判据。

---

## 26. 线 A 独占卡 batch 扫描：按用户要求停机（2026-09-26）

> 用户指示：「vllm 服务不要一直占着，跑完就去掉」
> 产物：`results/vllm-std/batch_sweep_exclusive.csv`（**9 行，未跑完**）

### 26.1 停机与释放

按依赖顺序显式 PID 停机（不用进程名，避免误杀）：

```
kill 3869601  (lineA_batch_exclusive.sh 驱动，先停以免启动 b32)
kill 1709510  (03_bench_spec.py)
kill 1709469 / 1709468  (run_bench.sh / timeout 包装)
kill 1715775  (VLLM::EngineCore，父进程死后成为孤儿，ppid=1)
```

**验证**：
- 我的 vLLM 相关进程 **NONE**
- **GPU2 完全释放**：`23 MiB used / 48648 MiB free / 0% util`
- 其余 compute app（其他用户与并行的 `fin_research` 作业）**未触碰**

### 26.2 已获得的数据（8 帧，64 token，3 次重复）

| batch | eager | spec | vs eager | 接受率 |
|---:|---:|---:|---:|---:|
| 1 | 2.0402 ± 0.0289 | 2.2662 ± 0.0361 | 0.9003× | — |
| 2 | 23.0942 ± 0.5475 | 23.5957 ± 0.7772 | 0.9787× | — |
| 4 | 32.4090 ± 0.2698 | 30.5461 ± 0.5375 | **1.0610×** | 0.0 |
| 8 | 80.1532 ± 2.4681 | 75.0915 ± 2.5103 | **1.0674×** | 0.0 |
| 16 | 244.0839 ± 1.0117 | ⛔ 未完成 | — | — |
| 32 | ⛔ 未跑 | ⛔ 未跑 | — | — |

### 26.3 两个重要发现（即使未跑完也成立）

**① b16 在独占卡上「可跑完」，共享卡上不可跑完。**
- 共享卡（早先）：b16 引擎自报吞吐跌到 **2–4 tok/s**，>15 分钟未完成 → 中止。
- 独占卡（本次）：b16 eager **244.08 s 完成**。
- → 早先的「b16 不可测」是**共享机争用**造成的，不是 b16 本身不可测。

**② b1→b2 的 11× 时间跳变在独占卡上复现 → 它不是争用噪声。**
```
b1 eager  2.04 s
b2 eager 23.09 s     <-- 11.3× jump for 2x batch
```
batch 翻倍而时间涨 11 倍，在**独占卡**上依然出现 → 这是 **vLLM 自身的行为**
（疑与 chunked prefill 调度 / 多模态编码器串行化有关），**值得单独查**。
**同一现象此前被归因为「共享机调度」，该归因现已被本次数据推翻。**

**③ b4/b8 出现 >1×（1.061 / 1.067），但接受率为 0.0。**
接受率为 0 意味着投机路径**只做额外功**，物理上不可能稳定快 6%。
→ **这两个 >1× 仍应视为测量假象**（跨时刻比较；即使 GPU 独占，
CPU/内存带宽/功耗仍与并行的 `fin_research` 作业共享）。
与 `docs/vllm-std/05-std-multitoken.md` §5.3 的结论一致：**跨时刻单点计时不可比。**

### 26.4 未完成项（不掩盖、不外推）

| 项 | 状态 |
|---|---|
| `p6x_spec_b16` | ⛔ 中止时正在运行（已等 4 分钟未完成） |
| b32（eager + spec） | ⛔ 未启动（停机以免继续占卡） |
| b1–b16 完整曲线 | ⚠️ eager 完整，spec 缺 b16 |

**复现命令**（GPU 空闲时）：
```bash
STD_FORCE_GPU=<独占A6000> bash scripts/vllm-std/lineA_batch_exclusive.sh
```

---

## 27. 真 STD 进 vLLM：阶段 0/1（2026-09-26）

> 文档：`docs/vllm-std/20-user-std-implementation.md`、`21-vllm-mapping.md`
> 方法：**只读代码**，每个结论附文件路径 + 行号 + 原文

### 27.1 阶段 0：真 STD 的机制（一句话）

> **用 prefill 注意力按 KV head 选出 top-K 视觉位置 → 把 KV cache 物理压缩成
> `非视觉位置 ∪ 该 head 的 top-K 视觉位置` → 让同一个 target 模型在这个缩短的 cache 上
> 做 γ 步自回归 greedy 前向 → 每步 argmax 就是一个草稿 token。**

**草稿来自模型前向，不是复制上下文** —— 这是它与 vLLM 上 n-gram 类比的根本区别。

| 问题 | 答案 | 证据（路径:行号） |
|---|---|---|
| 判据 | **全 query 求和**，逐层、逐 KV head | `std_qwen25vl.py:239` `grouped.sum(dim=(1,2))`；形状 L220/L233 |
| 稀疏前向 | **物理压缩 KV**（非 mask），**`attn_mask=None` / `is_causal=False`**，q_len=1，全 28 层 | `:302-335`（`compact_sparse_prompt_cache`，逐 head L324）、`:450-452`、`:290` |
| 草稿 | **γ 步自回归，greedy 温度=0**；`drafted[0]` 来自密集前向 | `:1042-1059`、`:117-118` |
| 接受率配置 | Video-MME / **128 帧** / 256 tok / **γ=9** / **K+text=1024** / CoT / 20 样本 | `PROGRESS.md:648`、`benchmark_a100_dynamic.py:31-38` |
| 接受率数值 | **0.507–1.000，均值 ≈0.81** | `PROGRESS.md:690-698`（**§17，非 §19**） |
| 框架/入口 | **本地 HF transformers**；static 臂在 `benchmark_a100_dynamic.py:330` | sglang 只用于 EAGLE-3，与本 STD 无关 |

**接受率统计口径**（必须是实测）：逐 token 前缀比对 `:1033-1039`；
`acceptance_rate = accepted/proposed` `:96-99`。**验证方是密集模型**，与 vLLM 可直接对照。

### 27.2 ★ 阶段 1：`propose` 拿不到 model 的绕过机制（已验证）

**`vllm/v1/worker/gpu_model_runner.py:5060-5063`**：

```python
if hasattr(self, "drafter"):
    logger.info_once("Loading drafter model...")
    if hasattr(self.drafter, "load_model"):
        self.drafter.load_model(self.model)      # L5063 ★ 把 target 模型交给 proposer
```

- **无 method 类型守卫** → 对 `custom_class` 同样生效。
- `NgramProposer` 也定义了 `load_model` 但为 no-op（`ngram_proposer.py:164-166`，
  注释 `# No model to load.`）→ 正是 `hasattr` 会通过的原因。

**完整可行链路**：构造时装 `Attention.forward` 钩子 → `load_model(model)` 拿到模型
→ dense prefill 时钩子抓每层 Q/K/V → `propose()` 在**同进程**内读缓存、自建压缩 KV、
手写单 token decode → 返回草稿 → **vLLM dense 验证给出接受率**。

### 27.3 ⚠️ 最大工程量（阶段 2 的核心，尚未开工）

用户的稀疏前向能跑，是因为它**自己拥有整个 decode 循环**（HF `past_key_values` 是
Python 对象，可原地改写 L332-333；可自由调 `model(...)`）。**vLLM 里三件事都不成立**：
KV 是分页张量且 proposer 拿不到；单步前向要走 runner 全流程；attention 已是
`Attention.forward(query,key,value)`（Q/K/V 为投影后张量）。

**路线 1（推荐）**：proposer 内自建 mini-decode —— 用 `load_model` 拿到的子模块
（`embed_tokens` / 每层 `q/k/v/o_proj` / `mlp` / norms）手写单 token step，
配私有压缩 KV。风险：vLLM 用 fused `QKVParallelLinear` 需拆权重（TP=1 可简化）、
**M-RoPE 必须复现**（用户 L428-430）。
**路线 2（不推荐）**：另加载 HF 权重做草稿 → 需第二份 16 GB 权重，且更接近 draft_model 模式，**不是 STD**。

**硬约束**：**绝不原地改写 vLLM 的 paged KV**（用户实现是原地改写 L332-333，
在 vLLM 里会破坏 dense 验证，而 dense 验证是接受率的唯一裁判）。
私有 KV 内存代价：32 帧 / 28 层 × 2 方向 ≈ **1.1 GB**，可接受。

### 27.4 阶段汇总

| 阶段 | 状态 | 命令 | 证据路径 | 结论 | 阻塞 | 下一步 |
|---|---|---|---|---|---|---|
| 0 读懂真 STD | ✅ | 只读 `std_qwen25vl.py` | `20-user-std-implementation.md` | 判据=L239 全 query 求和；稀疏=物理压缩 KV 无 mask；草稿=γ 步 greedy 自回归 | 无 | 阶段1 |
| 1 vLLM 映射 | ✅ | 只读 vLLM 源码 | `21-vllm-mapping.md` | **`load_model` 钩子可拿到 target 模型**（`:5063`，无类型守卫）；8 步映射表齐备 | 无 | 阶段2 |
| 2 GPU proposer | ⏳ **未开工** | — | — | 需手写 mini-decode（路线 1） | **工程量**：fused QKV 拆权重 + M-RoPE 复现 | 见 §27.3 |
| 3 多 token | ⏳ | — | — | — | 依赖阶段2 | — |
| 4 端到端计时 | ⏳ | — | — | — | 依赖阶段3 | — |
| 5 归因 | ⏳ | — | — | — | 依赖阶段4 | — |
| 支线 §19 判据 | ✅ | 只读代码 | `17-section19-criterion.md` | 全 query 求和，逐字相同 | 无 | — |
| 支线 b1→b2 断崖 | ⏳ | — | `26-b1b2-cliff.md` 未建 | — | 需 GPU 跑 enforce_eager 对照 | — |

---

## 28. 阶段 2.0：路线决策 —— 走路线 B（2026-09-26）

> 文档：`docs/vllm-std/22-route-decision.md`

### 28.1 决策

# **路线 B（HF draft + vLLM verify）**

**理由**：路线 B 能**复用已被验证过、能拿到 0.81 接受率的原实现函数**，
而路线 A 要拆 fused QKV + 复现 M-RoPE，**且没有已验证的实现可复用**。

### 28.2 四门槛（静态核验，全部通过）

| 门槛 | 结论 | 关键证据 |
|---|---|---|
| 1 显存 | ✅ | **`prefill_prompt:160-182` 是两段式 prefill**：视频前缀（L172，不要注意力）→ **只有 text 后缀**（L180，`output_attentions=True`）。故 attention 是 `[heads, text_len, prompt_len]` 而非全 prompt²。32 帧下 ≈1.5 GB，非原先担心的 9 GB。合计 ≈**37 GB / 48 GB** |
| 2 输入对齐 | ✅ | `pixel_values_videos` 是 vLLM 模型 forward 的 kwargs（`qwen2_5_vl.py:1352`），**可钩**；token id **不需要重建**（`propose()` 已给 `token_ids_cpu`） |
| 3 tokenize | ✅ | 同一份 tokenizer；已实测 HF 模板渲染正确（video id 151656）；展开数由 `grid_thw` 决定，捕获即一致 |
| 4 权重 | ✅ | 独立 `from_pretrained` FP16，不从 vLLM 导出 |

### 28.3 工作量为什么真的小（关键证据）

原实现**四个核心函数都是独立可调用的**，非焊死在生成循环里：

| 函数 | 位置 | 复用 |
|---|---|---|
| `prefill_prompt` | `:160` | ✅ 直接调 |
| `build_sparse_selection` | `:185` | ✅ 直接调 |
| `compact_sparse_prompt_cache` | `:302` | ✅ 直接调 |
| **`_draft_tokens`** | `:1042` | ✅ **直接调**——已是独立函数，参数 = (model, first_token, gamma, past_key_values, sparse_controller, start_position)，全部可由 proposer 备齐 |

### 28.4 ⚠️ 证据边界（必须说清）

**只读代码 + 静态算术，未在任何 GPU 上实跑路线 B 的任何一步。**
- 37 GB 是**算术估算**，未实测
- 「钩模型 forward 能拿到 pixel_values_videos」**未实跑验证**
- 「视频展开数一致」**未在运行时断言过**

**故措辞是「走路线 B」，不是「路线 B 已被证明可行」。**
阶段 2.1 第一个动作 = 最小验证（加载 HF + 捕获 video inputs + 断言
`len(hf_input_ids) == vLLM num_prompt_tokens`）；**断言失败立即转路线 A，不硬撑。**

### 28.5 阶段汇总

| 阶段 | 状态 | 命令 | 证据路径 | 结论 | 阻塞 | 下一步 |
|---|---|---|---|---|---|---|
| 2.0 路线决策 | ✅ | 只读代码 + 静态核算 | `22-route-decision.md` | **走路线 B**；四门槛静态通过 | 无 | 2.1 最小验证 |
| 2.1 HF draft proposer | ⏳ **未开工** | — | — | — | **未实测 GPU** | 最小验证→实现 |
| 2.2 接受率归因 | ⏳ | — | — | — | 依赖 2.1 | — |

---

## 29. 阶段 2.1：路线 B 最小验证 —— ⛔ 被机器资源阻塞（2026-09-26）

> 文档：`docs/vllm-std/22-gpu-proposer-1token.md`
> 脚本：`scripts/vllm-std/P21a_capture_video_inputs.py`、`wait_and_run.sh`

### 29.1 2.1a：钩子捕获 `pixel_values_videos` —— **未完成（5 次尝试全败于显存）**

钩点已读源码确认（`qwen2_5_vl.py:1927`，视频张量在 `**kwargs`），脚本已就绪。

| # | GPU | 失败点 |
|---|---|---|
| 1 | 7 (A6000) | 启动时该卡剩 15.54 GB（需 16.16） |
| 2 | 4 (4090) | **模型已加载（15.25 GB），视觉编码器再要 1.02 GB 时 OOM** |
| 3 | 7 | 探测 37,257 MiB → 启动时只剩 26.98 GB |
| 4 | 2 | 探测 23,792 MiB → **6 秒内掉到 6.28 GB** |
| 5 | 等 ≥30 GB 窗口 | 等 10 分钟，最好仅 16,743 MiB，**始终无窗口** |

**根因**：7B bf16 权重 **15.67 GB** + 视觉编码器 ~1 GB + KV → **vLLM 单侧最低 ~17.5 GB**；
本机当前最好 **16.7 GB**（A6000）/ 16.4 GB（4090），**全部 A6000 100% 利用率**。
波动极端（实测 23.8→6.28 GB / 6 秒）。

**2.1b 门槛更高**：需 vLLM(15.67) + HF(15.5) ≈ **31.2 GB 固定 + KV ≈ 37 GB 峰值**，
**本机无任何卡有 37 GB 空闲**。

### 29.2 2.1b–2.1e 一律未开始（遵守硬性规则 1）

| 步骤 | 状态 |
|---|---|
| 2.1a | ⛔ 未完成 |
| 2.1b vLLM+HF 共存 | ⛔ 未开始 |
| 2.1c 输入对齐断言 | ⛔ 未开始 |
| 2.1d proposer 骨架 | ⛔ 未开始 |
| 2.1e 单 token 接受率 | ⛔ 未开始 |

### 29.3 ⚠️ 必须避免的错误结论 + 为什么不转 A

- ❌ **不能说「路线 B 不可行」**：失败是**资源可用性**，不是设计缺陷。
- ❌ **不能说「接受率是 0」**：接受率**没测过**，不是 0。
- ✅ 正确表述：**「2.1a 因本机无 ≥17.5 GB 空闲显存未能执行；2.1b 需 ≈37 GB，当前机器给不出。」**
- **不转路线 A**：任务书的转 A 条件是「2.1a 失败（拿不到视频输入）」，
  本次是**根本没跑起来**，**不构成技术性失败**，
  不触发转 A。用未验证的失败去换一条更重的路线没有依据。

### 29.4 顺带产物

`scripts/vllm-std/wait_and_run.sh`：在**同一次 shell 调用内**完成「探测窗口 → 立即启动」，
消除探测与启动之间的间隙（本机 6 秒内可掉 17 GB）；并要求固定保守显存预算，
**不按探测峰值缩放**（按峰值缩放曾请求 33.5 GiB，而卡已被邻居压到 27 GiB）。

复现（卡空闲时）：
```bash
bash scripts/vllm-std/wait_and_run.sh 30000 3600 \
    P21a_capture_video_inputs.py --num-frames 32 --max-model-len 32768
```

### 29.5 阶段汇总

| 阶段 | 状态 | 命令 | 证据路径 | 结论 | 阻塞 | 下一步 |
|---|---|---|---|---|---|---|
| 2.1a | ⛔ | `wait_and_run.sh 30000 … P21a_…py` | `22-gpu-proposer-1token.md`、`logs/P21a*.log` | 5 次尝试全败于显存 | **本机无 ≥17.5 GB 空闲卡** | 等窗口重跑 |
| 2.1b–e | ⛔ | — | — | 未开始 | 同上（2.1b 需 ~37 GB） | 依赖 2.1a |
| 2.2 归因 | ⛔ | — | — | 无数据 | — | — |

### 29.6 用户选 (a)：挂长等待重跑 2.1a（2026-09-26 19:44 起）

驱动：`scripts/vllm-std/P21a_driver.sh`（后台，deadline 3 小时）
日志：`results/vllm-std/logs/P21a_driver.log`

**策略**（针对本机波动设计）：
- 要求**大余量**而非"刚好够"—— 实测过 6 秒掉 17 GB，刚好够的窗口在权重落卡前就没了
- 前 45 分钟要求 ≥30 GB；之后降到 ≥22 GB（避免一个固定阈值永远等不到）
- 窗口 ≥32 GB → 32 帧 / max_model_len 32768（任务书规格）
  窗口 22–32 GB → 8 帧 / max_model_len 8192（够过 2.1a 的钩子门槛）
- 固定 `gpu_mem_util=0.360`（不按探测峰值缩放，那是第 3 次尝试失败的教训）
- **只在显存类失败时继续等**；出现非显存失败立即停止并留待检查
  （真正的技术性失败是信息，不该被无限重试掩盖）

**注意**：驱动会**独占**该卡运行 vLLM；若你需要用卡，先 `pkill -f P21a_driver`（或用明确 PID）。

---

## 30. 阶段 2.1a：**PASS** —— 正确钩点是 `embed_multimodal`（2026-09-26 10:33）

> 文档：`docs/vllm-std/22-gpu-proposer-1token.md`（2.1a 定稿）
> 数据：`results/vllm-std/P21a_video_inputs.json`

### 30.1 结果

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

`1176 = 3ch × 2temp × 14 × 14` → vLLM 视觉塔直接消费的**已切分 patch 张量**，
与 HF `forward` 期望的格式一致，**可原样喂回**。

### 30.2 ★ 决定性发现：钩 `forward` 是错的，要钩 `embed_multimodal`

第一版钩 `forward` → `captured_keys: []`，`call_log` 显示 `n_input_ids: None | kwargs: []`。

**根因**：**vLLM v1 把视觉塔单独跑**（启动日志 `Encoder cache ... budget of 32768 tokens`），
把算好的 `inputs_embeds` 交给 LLM，**原始像素根本不经过 `forward`**。

**正确钩点**：`Qwen2_5_VLForConditionalGeneration.embed_multimodal`（`qwen2_5_vl.py:1595`）
→ `_process_video_input`（`:1432`，取像素 `:1442`，喂视觉塔 `:1451`）。**修正后立即 PASS。**

> **对后续所有阶段有效**：在 vLLM v1 里找多模态输入，**钩 `embed_multimodal`，不要钩 `forward`**。

### 30.3 实测显存配置（非估算）

| 项 | 值 |
|---|---|
| 帧数 / max_model_len | 32 / 32768 |
| **可用 `gpu_mem_util`** | **0.560**（37 GB 空闲窗口） |
| 失败值 | 0.360 → KV **-3.3 GiB**；0.430 → KV **0.03 GiB**（需 1.75） |

**教训**：32 帧时 vLLM 为编码器缓存 + activations 保留约 **5.4 GB**，
不是「权重 + KV」两段式能算出来的；**0.430 差 1.7 GB 就是不行**。

### 30.4 顺带修掉的两个自己的 bug

1. 失败分类漏串：`No available memory for the cache blocks` /
   `larger than the available KV cache memory` 未被识别为显存类失败，
   导致驱动把「预算不够」误判为「技术失败」并退出。
2. **stale-log bug 复发**：失败判定 grep 整个追加日志，匹配到上一轮旧错误。
   已改为只检查本次尝试新增的字节区间。

### 30.5 阶段汇总

| 阶段 | 状态 | 命令 | 证据路径 | 结论 | 阻塞 | 下一步 |
|---|---|---|---|---|---|---|
| 2.1a 捕获 video inputs | ✅ **PASS** | `bash scripts/vllm-std/P21a_driver.sh` | `22-gpu-proposer-1token.md`、`P21a_video_inputs.json` | 钩 `embed_multimodal` 可拿到全部三个张量；32 帧需 `gpu_mem_util=0.560` | 无 | 2.1b |
| 2.1b vLLM+HF 共存 | ⏳ | — | — | 需 ≈37 GB 峰值（vLLM 21.1 + HF 15.5） | **本机窗口** | 依赖窗口 |
| 2.1c 输入对齐 | ⏳ | — | — | — | 依赖 2.1b | — |
| 2.1d/e proposer + 接受率 | ⏳ | — | — | — | 依赖 2.1c | — |

---

## 31. 阶段 2.1b + 2.1c：**双双 PASS**（2026-09-26）

> 文档：`docs/vllm-std/22-gpu-proposer-1token.md`
> 数据：`results/vllm-std/P21b_coexist.json`
> 脚本：`scripts/vllm-std/P21b_coexist_and_align.py`、`P21b_driver.sh`

### 31.1 2.1b 共存显存 —— ✅ PASS

| 项 | 值 |
|---|---|
| 8 帧 / max_model_len 8192 / util **0.400** | vLLM 加载后空闲 30,050 MiB |
| HF 15.45 GiB 加载后空闲 | 14,586 MiB |
| **余量** | **14.36 GiB**（门槛 5 GiB）✅ |

**显存不是阻塞点。** 依赖坑：`device_map` 需 `accelerate` → 改为 `.to("cuda")`，不加依赖。

### 31.2 2.1c 输入对齐 —— ✅ PASS ★ 最有价值的结果

```
vllm_prompt_tokens: 4814 | grid_thw: [4, 52, 92] | pixel: [19136, 1176]
hf_next_token: 785 | vllm_first_token: 785 | next_token_match: TRUE
```

**用 vLLM 的 token ids + 捕获像素喂 HF，HF 的 greedy 下一 token 与 vLLM 实际产出一致（785）。**
这是**预测接受率的最强单项证据**：两边在同一上下文上给出同一 argmax。

### 31.3 ★ 「捕获第一次调用」的坑（第三次出现，已修）

第一版报假失败：`grid [1,256,256]` → 16384 features ≠ prompt 里 4784 video tokens。
**根因**：vLLM 启动时先用**最大特征尺寸**做 encoder profiling。
`mm_call_shapes` 证实只有两次调用：`#1 [65536,1176]`（warmup）、`#2 [19136,1176]`（真实）。
改捕获**最后一次**后立即 PASS。

> **规训**：本代码库中任何捕获形状/张量的钩子**必须取最后一次调用**，
> 或按 prompt 长度匹配挑选，**绝不能取第一次**。（同类 bug 已在 03b 探针、P1 profiling 出现）

### 31.4 细节：8 帧 → grid T=4

`video_grid_thw=[4,52,92]`，T=4 而非 8 —— vLLM 时间维有合并。HF 用同一 grid 张量，天然一致。

### 31.5 阶段汇总

| 阶段 | 状态 | 证据路径 | 结论 | 下一步 |
|---|---|---|---|---|
| 2.1a | ✅ PASS | `P21a_video_inputs.json` | 钩 `embed_multimodal` 可拿到全部张量 | — |
| 2.1b | ✅ PASS | `P21b_coexist.json` | 余量 14.36 GiB，显存非阻塞 | — |
| 2.1c | ✅ PASS | 同上 | HF 与 vLLM 下一 token 一致（785） | **2.1d** |
| 2.1d proposer 骨架 | ⏳ | — | 前置齐备，可开工 | 写 proposer |
| 2.1e 接受率 | ⏳ | — | — | 依赖 2.1d |
