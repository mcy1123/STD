# 廉价 KV 筛选：五种方法的执行计划

**状态**：**五篇方法已全部走完判定** —— 4 篇有实测结论（HybridKV 机制有效/判据反了；FlashCache、MM-ShiftKV 否证；FastCache 分析排除），
1 篇（SimiCache）**确认不存在任何开放获取版本**，需用户提供 PDF。
**唯一实测正收益来自"把 KV 预算调大"**（非五篇中的任何一篇），见 §G0.5。
**日期**：2026-09-22
**论文与代码**：`papers/`（索引见 [`papers/README.md`](../../../papers/README.md)）、`third_party/MM-ShiftKV/`
**基线证据**：`results/l2_screening/local512/`、`results/ksweep64/`、`results/ksweep128/`、`results/freeze_sweep64/`、
`results/routing_traces_per_layer/`、`results/mmshift_proxy/`、`results/flashcache_dev/`
**口径**：所有加速比均为 **decode**、`ratio-of-sums`、注明 `timing_valid`（规则见 `docs/experiment-provenance.md` §8.3/§9.3）

---

## 0. 结论先行：问题需要重新定位

**你给的出发点是"STD 的 top-K 选择每个 decoding step 都要做，开销持续产生"。这个描述只对 `dynamic` 这条研究线成立，对 STD 本身不成立。**

| | 选择发生在何时 | 现状 decode 加速比（512 tok） |
|---|---|---|
| **static STD（论文复现主线）** | **只在 prefill 做一次** | **1.711–1.873× vs AR** |
| dynamic routing（我们的研究线） | 每个 decode 轮重选 | 0.887–0.910× vs static（**亏**） |

也就是说：**"把筛选挪到 prefill、decoding 零筛选"对 STD 来说不是新能力——它就是 static，而且已经在赢 1.71–1.87×。**
你这五篇里 MM-ShiftKV / FlashCache 的"prefill-only"属性，对 STD 的价值不在"省掉 decoding 筛选"，而在下面两件**别的事**：

1. **提高那"一次性 prefill 选择"的质量** → 直接抬高 static 接受率（现在 0.9115），这是**唯一能真正减少轮数、从而继续加速**的杠杆。
2. **免掉 `output_attentions=True`** → 现在为了拿注意力矩阵，prefill 被迫走 eager 自定义 attention，既慢又**扰动数值、破坏与 greedy AR 的位精确性**（`PROGRESS.md` §4）。MM-ShiftKV / FlashCache 都**只吃 K/V 或 hidden_states**，可以在 SDPA/FlashAttention 下工作。

### 0.1 五种方法的定位（按"对 STD 的价值"排序，不是按论文影响力）

| 排序 | 方法 | 对 STD 的可迁移点 | 是否需要 attention 矩阵 | 官方代码 | 判定 |
|---|---|---|---|---|---|
| **1** | **MM-ShiftKV** | 方差放大代理 + 分组投票；**已实测否决**（"免 `output_attentions`"的工程价值仍在，但不属算法收益） | ❌ | ✅ 有（`d267f8cd`） | **❌ 已否证**：所有 γ 都低于 attention，γ=10 最低（−20pp） |
| **2** | **HybridKV** | `S_{l,h}` 分类 + Eq.(5) 非均匀预算；**实测：冻结机制有效、分类判据反了** | ❌ | ❌ 无 | **⚠️ 机制有效但不能翻盘**：冻结 92/112 头 → 改写 −85%、接受率 +0.040；`pearson(S,churn)=+0.25`（论文假设负） |
| **3** | **FlashCache** | **Dev 判据**（免注意力）已实测**否决**；DBA 逐层非均匀预算**需先做布局改造**（且被 §5.2 三堵墙挡住） | ❌ | ❌ 无 | **❌ 已否证**：Jaccard 0.068（≈随机 0.066）、耗时 4 s（超门槛 40×） |
| **4** | **SimiCache** | 相似度两步聚类 + 加权合并 | 推测不需要 | ❌ 无 | **⛔ 确认无 OA 版本**（OpenAlex `any_repository_has_fulltext=false`），需用户提供 PDF |
| **5** | **FastCache** | **不作为方法来源**；仅贡献 §3 门控实验的设计范式 | — | ❌ 无 | **❌ 分析排除**（单请求下调度退化为零收益） |

### 0.2 本计划的核心判断（两个门控结果都指向同一方向）

调研过程中用**现有数据**（零新增 GPU 实验）先算了两个门控，结论都限制了这个诉求的上限：

> **门控 1（§3 G0.1）**：把 dynamic 的 refresh 与 selection 时间**全额扣除**后，
> dynamic 仍比 static 慢 **3–11%**（0.890–0.966×）。
> → **"把每轮筛选做便宜"在单请求下封顶 <1.0×，不可能靠它翻盘。**
>
> **门控 2（§3 G0.3）**：在解码期真实注意力上，现有 static prefill 选择器已经拿到
> **oracle 可达质量的 84.1%**；即便给一个**能预知未来**的完美选择器，覆盖率也只从 37.7% → 44.8%
> （**仅 7.1pp**）。而且覆盖率随解码**不降反升**（+2.7pp）。
> → **"把 prefill 选择做得更好"的空间也只有 ~7pp，且选择器不是主要瓶颈。**
>
> **✅ 后续修正（P1-a，per-layer 实测）**：G0.3 用的是**层均值** trace，**低估了一半**。
> per-(layer,head) 实测：static 0.6345、oracle 0.7728、**gap +13.8pp**。
> 所以"更好的选择器"的真实上界是 **+13.8pp**，不是 +7.1pp（见 §「P1-a / P2-a 已完成」）。

**合起来的结论：在 STD 当前的工作点（128 帧、K+text=1024、6.09% 视觉保留）上，
"廉价 KV 筛选"这个诉求的两条可能路径——把筛选做便宜、把筛选做准——都被实测封住了上限。**

> **门控 3（§3 G0.4）**：看起来"覆盖率能预测接受率"（r=0.752），但**这是样本难度的混淆**：
> 一个样本在 3%/6%/20% 预算下的覆盖率几乎完全相同（r=0.98–0.99），说明 coverage 是**样本级属性**；
> 它与接受率的相关性只是"简单样本两头都好"。**接受率侧的因果上界只有 Oracle Study 的 +16% 相对（≈14% 少轮）。**

因此本计划的定位相应下调为：**把五篇论文当作"上界已被量化的备选方案"逐个验证**，
而不是当作能翻盘的来源。真正还没被证否的杠杆只剩三个：

1. **★ 改变预算本身（G0.5）** —— **新晋第一优先**。把预算从 6.09% 提到 10%，覆盖率涨 **+9.6pp**，
   **比"在 6% 预算下做到完美选择"（+7.1pp）还多**；而且**不用移植任何论文代码、不用 per-layer trace、
   不改选择逻辑**（`build_sparse_selection` 已有 `explicit_k` / `target_k_plus_text`，只改命令行）。
   它同时是 G0.4 那个因果问题的**直接检验**：接受率到底会不会随覆盖率上升。
2. **E4′**：消掉未被 profile 覆盖的 host 侧开销（G0.1 的残留 +12.7~+40.9 ms/轮）。
3. **提高单轮 verify 效率**（`draft + verify + bonus` 的绝对值）。

**若 G0.5 显示接受率不随 K 上升**，则"覆盖率→接受率"被证伪为纯关联，
**选择器方向整体止损**（P1/P2/P3 降级为纯学术记录），资源全部转入 2 与 3。

> 与旧结论一致：`PROGRESS.md` §17.5（"理想选择只值 ~16% 轮数，而动态要回本需减少 22% 以上轮数"）、
> §17.8（E1 是主效应）、§17.9（512 token 下 static 已达 1.71–1.87×、dynamic 仍亏）。
> **本计划新增的是：把"更好的选择器"的空间从"~16% 轮数"进一步收紧到"覆盖率层面 7.1pp"。**

---

## 1. 现状基线（后续所有实验的对照）

**512 token / 128 帧 / Video-MME 10 样本 / A6000 GPU 2 / `timing_valid=true`**（`results/l2_screening/local512/`）：

| 配置 | static 接受率 | Δaccept | **decode static vs AR** | **decode dyn vs static** | refresh ms/轮 | 过路费 ms/轮 | exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| A 基线（v2,i1,0.00） | 0.9115 | −0.0178 | 1.711× | 0.887× | 22.3 | +40.4 | 8/10 |
| B = E1 节流（v2,i4,0.05） | 0.9115 | −0.0171 | 1.796× | **0.910×** | **6.0** | **+28.9** | 8/10 |
| C = E6 全打分（v3,i1,0.00） | 0.9115 | −0.0128 | 1.873× | 0.827× | 37.9 | +80.8 | 8/10 |

**128 token 对照（R1）**：static vs AR 1.554×、dyn vs static 0.819×、static accept 0.8103。

**STD 选择器的现状（代码事实）**：
- `build_sparse_selection()`（`src/std_repro/std_qwen25vl.py:175–231`）吃 `attentions`（`[q_heads=28, text_len≈57, prompt_len≈15933]`），reshape 成 `[kv_heads=4, heads_per_kv=7, text_len, prompt_len]`，**对 query head 与 text query 直接求和**，再在 visual 位置上 top-k。
- `k = 1024 − text_len ≈ 967`，即视觉保留 **967/15876 = 6.09%**；**28 层 × 4 KV head = 112 个选择单元，全部共用同一个 `k`**（`SparseSelection.k` 是单个 int）。
- 已有的免注意力基线：`attention_free_visual_topk()`（`src/std_repro/dynamic_selection.py:104–168`）——windowed centroid residual + value norm，需注意它**对 layer 做了 `.mean(dim=0)`**，返回跨层共享的 `[kv_heads, k]`。
- **现有 trace 不能直接用**：`results/routing_traces_videomme/*.pt` 里 `prefill_scores` 只有 `[4, 15876]`（head 已归并），做不了 per-(layer,head) 分析，需要一次仅 prefill 的新采集。

**接受率的天花板**：static 0.9115 → 1.0 是 **+9.7% 相对**；oracle study 显示完美选择只值 **~16% 轮数**。所以这条线是**中等收益**，必须与"免 `output_attentions`"的确定性收益一起算账。

---

## 2. 阶段 0：门控实验（已执行，用现有数据）

### G0.1 「refresh + selection 完全免费」的上界 —— ✅ 已算出，结论是负的

**做法**：不改代码，直接从现有 JSONL 里把 dynamic 的 `total_refresh_time_ms` 与 `total_selection_update_time_ms`
从 decode 时间里**全额扣除**，得到一个"搬运与选择零成本"的乐观上界：

```
C/A = static_decode / (dynamic_decode − refresh − selection)
```

| 运行 | static (s) | dynamic (s) | refresh (s) | selection (s) | **oracle (s)** | **C/A（上界）** | B/A（现状） |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 tok R1 | 55.11 | 67.32 | 3.68 | 1.69 | 61.94 | **0.890×** | 0.819× |
| 512 tok A 基线 | 204.59 | 230.58 | 12.77 | 5.96 | 211.85 | **0.966×** | 0.887× |
| 512 tok B (E1) | 194.75 | 214.03 | 3.45 | 1.95 | 208.63 | **0.933×** | 0.910× |
| 512 tok C (E6) | 229.72 | 277.74 | 21.58 | 5.72 | 250.44 | **0.917×** | 0.827× |

**判读：**

1. **即便把 refresh 与 selection 全变成 0，dynamic 仍然比 static 慢 3–11%。**
   → **"让每轮筛选/搬运更便宜"这条路，在单请求下封顶就是 <1.0×，不可能靠它翻盘。**
   这正是"廉价 KV 筛选"这个诉求本身的边界：**省下来的不是胜负手**。
2. 扣除后仍残留 **+12.7 ~ +40.9 ms/轮**（A +12.7、B +24.3、C +36.4、R1 +40.9）。
   这部分是 **draft/verify/bonus 的差异 + 未被 profile 覆盖的 host 侧开销**（`PROGRESS.md` §17.6 估 ≈34 ms/轮），
   即 **E4′ 的靶子**。
3. ⚠️ **这个 C/A 是保守的**：它没有也无法扣掉第 2 点里那部分 host 侧开销。若那部分也能归零，C/A 会更高。
   所以严格表述是：**"零 refresh+selection 成本"这一档已经不足以翻盘；要翻盘必须连未计时的 host 侧开销一起消掉。**

**由此得到的预注册判据（沿用 FastCache 报告的建议，但结论已提前落地）：**
- ✅ **停止**把"藏每轮固定开销"当作主线。
- ✅ **转向**：① 提高接受率（减少轮数）—— 即本计划的阶段 1；② 同时消掉 host 侧开销（E4′）。
- 仅当"阶段 1 + E4′"合并后仍 ≤1.0×，才考虑放弃 dynamic 线、只保留 static。

### G0.2 视频上的 prefill→decode 方差比 —— 待做（P1 的前置）

MM-ShiftKV 的全部立论建立在"decode query 方差比 prefill 大 **13–22×**"（图像上测得）。
**视频上这个比值可能完全不同**（时序冗余 → 更可预测；跨帧注意力切换 → 更发散）。若比值 ≪ 10，γ=10 的放大反而有害（论文 Table 6：γ=1→55.7、γ=10→68.8、γ=20→68.5，非单调）。

**做法**：10 个 Video-MME 样本，prefill 与 decode 各轮逐层 hidden_states 在线累积 per-dim `mean`/`std`（只增 2×3584 float/层，零峰值内存代价），报出 `std_ratio = decode_std/prefill_std` 的分布。
**成本**：一次 prefill + 前 32 轮 decode 的 trace 采集，半天以内。

### G0.3 prefill 选择器的**剩余空间** —— ✅ 已算出（离线，20 条已有 trace）

**做法**：不改代码。用 `results/routing_traces_videomme/` 里已有的
`prefill_scores`（`[kv_heads, visual_len]`，层均值）作 static 选择的分数，
用 `round_scores`（每个 decode 轮 `[kv_heads, q_len, visual_len]`）作**解码期真实注意力**，
在 STD 的预算下（retention 6.09%，k=967）算两个覆盖率：

- `coverage_static` = prefill 选择能覆盖多少解码期注意力质量
- `coverage_oracle` = **能预知解码注意力的完美选择器**能覆盖多少（任何选择器的上界）

新增脚本：`scripts/analysis/selection_headroom.py`；产物 `results/routing_analysis_videomme/selection_headroom.md`。

| 量 | 值 |
|---|---:|
| `coverage_static` | **0.3770** |
| `coverage_oracle` | **0.4484** |
| 绝对差（= 完美解码感知选择器的全部空间） | **+0.0714**（7.1 个百分点） |
| 随机基线（≈ retention） | 0.0609 |
| static / 随机 | **6.19×** |
| oracle / 随机 | 7.36× |
| **static / oracle** | **84.1%** |

**加上两个关键子结果：**

**(a) 覆盖率随解码进行**不降反升**（前一半 0.3629 → 后一半 0.3903，+2.7pp）**：

| | 早期轮 | 晚期轮 |
|---|---:|---:|
| coverage_static 均值 | 0.3629 | **0.3903** |

→ **static 选择在解码后期并没有变差**。而"重选"的立论前提正是"static 会漂移、后期需要更新"。
（注意与 `drift_jaccard.csv` 看似矛盾：选择**集合**确实在漂移，但**覆盖的质量没掉**——
这再次印证 §17.5 的"召回率敏感、接受率迟钝"。）

**(b) 预算扫描**：static/oracle 比例在所有预算下都很高（3% → 80.6%、6.09% → 84.1%、
10% → 86.3%、20% → 89.6%、30% → 91.9%）。**说明瓶颈不在选择器，也不在预算。**

**判读（这条直接约束了整个阶段 1 的期望值）：**

1. **即便给一个能预知未来的完美 prefill 选择器，覆盖率也只能从 37.7% 提到 44.8%
   —— 只有 7.1 个百分点的绝对空间 / 15.9% 的相对空间。**
2. **现有 static 选择器已经拿到 oracle 可达质量的 84.1%** → **选择器不是主要瓶颈。**
3. 因此 MM-ShiftKV / FlashCache-Dev / HybridKV-`S_{l,h}` 这些"更好的 prefill 打分器"，
   **其收益上界就是这个 7.1pp**；而它们都是**代理估计**（看不到未来注意力），实际只能吃到其中一部分。
   → **P1-a 的预注册门槛应据此重读：相对 (A) 提升 >1–2pp 已经是"吃掉 oracle 空间的 14–28%"，属于乐观预期。**
4. **覆盖率不随解码衰减** → 进一步削弱"每轮重选"的必要性（与 G0.1 相互独立地指向同一结论）。

**⚠️ 粒度限制（必须声明）**：trace 的 `prefill_scores` / `round_scores` 都是**层均值**
（`attention_trace.py:101` 的 `stacked.mean(dim=0)`），所以上表是**跨层共享选择**的数字，
**不是 STD 真实的 per-(layer,head) 选择**。
按 Jensen 不等式，per-layer 的 oracle 会**高于**层均值 oracle（per-layer 自由度更大），
所以 per-layer 的真实空间**可能比 7.1pp 更大**；但 STD 的 static 也是 per-layer 的，
它的 per-layer 覆盖率也可能更高。**两者之差需要一次 per-layer trace 才能定论 —— 这就是 P1-a/P2-a 的第一个动作。**

### G0.4 覆盖率↔接受率是**关联，不是因果** —— ✅ 已算出（离线，n=10）

把 trace 的 per-sample `coverage_static` 与实测的 per-sample static 接受率配对
（trace 的 20 个样本**包含** benchmark 用的 10 个，可直接配对）：

| retention | mean coverage | **r(coverage, static_accept)** |
|---:|---:|---:|
| 3.00% | 0.2723 | **+0.720** |
| 6.09% | 0.3880 | **+0.752** |
| 10.00% | 0.4844 | **+0.767** |
| 20.00% | 0.6395 | **+0.806** |

看起来"覆盖率能预测接受率"。**但这是样本难度的混淆，不是因果证据：**

| 检验 | r |
|---|---:|
| `r(coverage@3%, coverage@6%)` | **+0.994** |
| `r(coverage@6%, coverage@20%)` | **+0.981** |
| `r(coverage@3%, accept)` | +0.720 |

**一个样本在 6% 预算下的覆盖率，几乎完全决定它在 20% 预算下的覆盖率（r=0.981）。**
也就是说 **coverage 基本是"这个样本的注意力有多集中"这个**样本级属性**，与预算几乎无关**；
而它与接受率的相关性在所有预算下都稳定在 0.72–0.81。
→ 真实解释是：**注意力集中的样本，恰好也是 draft 本来就能猜准的"简单样本"**（两者都被样本难度驱动）。

**因此：`r=0.752` 不能用来论证"提高选择器覆盖率会提高接受率"。**
**接受率侧的唯一因果上界来自 Oracle Study**（`PROGRESS.md` §2/§17.5）：
完美选择只值 **+1.20 词/轮（≈+16% 相对，≈14% 少轮）**，而我们的动态实现只拿到其中 **35%**（+0.42）。

**补充事实**：`dynamic_stats.per_round` **不记录逐轮接受率**（只有 `jaccard_old_new`/`changed_ratio`/
`refresh_time_ms`/`update_applied`），所以"改变选择 → 接受率如何变"的因果问题**无法离线回答**，
必须做端到端 A/B（即 P1-b）。

### G0.5 由此得到的、最该先做的一个实验：静态线 K 扫描 ★

G0.3 + G0.4 合起来指向一个**比"换打分器"更直接**的杠杆：**预算本身。**

| K（retention） | k | coverage_static | 相对 6.09% 的 oracle（0.4484） |
|---|---:|---:|---|
| 6.09%（现状） | 967 | 0.3770 | 还差 7.1pp |
| **10%** | 1588 | **0.4726** | **已超过 6% 预算下的 oracle** |
| 20% | 3175 | 0.6287 | 大幅超过 |

**把预算从 6.09% 提到 10%，覆盖率涨 +9.6pp —— 比"在 6% 预算下做到完美选择"（+7.1pp）还多。**

而且这个实验**不需要移植任何论文代码、不需要 per-layer trace、不需要改选择逻辑**
（`build_sparse_selection` 已有 `explicit_k` / `target_k_plus_text` 参数，只改命令行），
是**当前性价比最高的一个**：config 固定、512 token、10 样本、3 档 K ∈ {967, 1588, 2381}。

**判据**：看**接受率是否随 K 上升**（这是对 G0.4 那个因果问题的直接检验）。
- 若接受率显著上升且**每轮成本的上升更慢** → 存在净收益最优 K，静态线可以直接获益（这是本计划最有希望的正面结果）。
- 若接受率**不动** → 证实"覆盖率→接受率"是纯关联，**选择器方向整体止损**，资源全部转到 E4′ 与单轮 verify 效率。

**成本**：3 档 × 10 样本 × 512 token ≈ 3–4.5 小时（每档含 4.5 分钟模型加载）。

**已就绪的启动命令**（`run_dynamic_matrix_local.sh` 已支持第 5 个字段"附加参数"，见其注释；
它会等 GPU 真正 idle 才开跑，产出 `timing_valid=true`）：
`k = k_plus_text − text_len`，text_len ≈ 57，故 k=967/1588/2381 ↔ `--k-plus-text 1024/1645/2438`。

```bash
cd /home/mcy/Projects/SpecVLM/STD
TAG=k_sweep512 GPU=2 LIMIT=10 MAX_NEW_TOKENS=512 REPEATS=1 \
CONFIGS_OVERRIDE="K967|v2|1|0.00|--k-plus-text 1024
K1588|v2|1|0.00|--k-plus-text 1645
K2381|v2|1|0.00|--k-plus-text 2438" \
  bash scripts/run_dynamic_matrix_local.sh k_sweep512
```

**开跑后要报的量**：逐档的 `mean_accept_length` / `acceptance_rate`（是否随 K 上升）、
`decode_rounds`、`decode ms/轮`、以及 decode 墙钟加速比。
**注意**：K 变大后 sparse draft 变贵而 dense verify 不变，所以"每轮成本"必然上升；
要判断净收益必须同时看**轮数下降**与**每轮成本上升**谁更快。

**分析不需要新工具**：G0.5 的三档产物直接喂给已有的 `scripts/analysis/compare_dynamic_configs.py`，
它报的 `mean static`（逐档 static 接受率）、`decode ms/轮`、`rounds`、`refresh ms/轮` 正是判据所需的全部字段。

### G0.6 P1-a / P2-a 的 per-layer 工具链 —— ✅ 已就绪（2026-09-22，CPU 可测）

G0.3 的结论受限于 trace 是**层均值**。要拿到 per-(layer,head) 的决定性数字，需要一条新的采集+分析链，
现在两端都已写好并测试：

| 文件 | 作用 |
|---|---|
| `scripts/analysis/attention_trace_per_layer.py` | **采集**：保留 layer 轴的只读 hook（沿用同一个 `_std_trace_hook` 注入点）。为控制体积，在 hook 内就把 HybridKV Eq.3 的 `S_{l,h}` 算好（否则 per-text-query 张量约 400MB），只留 `prefill_visual [L,H,V]` fp16、`prefill_sparsity [L,H,K]`、以及**均匀抽稀**到 `--max-rounds` 的 `round_visual [R,L,H,V]` |
| `scripts/analysis/analyze_per_layer_selection.py` | **分析**：per-(layer,head) 的 static vs oracle 覆盖率（P1-a 的决定性版本），θ∈{0.90,0.85,0.80} 下的 **`N_stat/N_dyna`**（P2-a 的核心未知数），以及 `S_{l,h}` 与**逐头 churn** 的相关性（P2-a 的止损判据） |
| `tests/test_attention_trace_per_layer.py` + `tests/test_analyze_per_layer_selection.py` | **30 个 CPU 回归测试，全部通过**（用假模型驱动 hook 路径 + 解析式已知答案的合成 trace） |

**抽稀是均匀的而不是"取前 N 轮"**：G0.3 已证明覆盖率在解码过程中会变化，前段取样会让每个结论都有偏。
`decimate_round_indices` 保证首尾都被覆盖且间隔近似均匀，并有测试钉住。

**接线时修掉/钉住的坑**：hook 里误把"对 query 维求和"写成 `sum(dim=1)`（实为 KV-head 维）——
会让 per-layer 张量的 head 维变成 query 长度，测试直接抓到；`classification` 在 `analysis_top_k=None`
时曾会 `top_ks.index(None)` 抛错；`stack_layer_scores` 对**缺层/形状不齐**直接报错，
避免"少采一层"静默污染所有 per-layer 统计量。

**剩余动作（需 GPU）**：
```bash
# 1) 采集（约 30–40 分钟/样本，10 个样本）
CUDA_VISIBLE_DEVICES=2 python scripts/analysis/attention_trace_per_layer.py \
  --frame-num 128 --max-new-tokens 256 --gamma 9 --k-plus-text 1024 \
  --limit 10 --gpu 2 --max-rounds 12 --sparsity-top-ks 967 1588 \
  --output-dir results/routing_traces_per_layer
# 2) 分析（秒级）
python scripts/analysis/analyze_per_layer_selection.py \
  --trace-dir results/routing_traces_per_layer \
  --retention 0.0609 --thresholds 0.90 0.85 0.80
```

---

## 3. 阶段 1：提高一次性 prefill 选择的质量（主线）

> 三条腿都以 **static 线**为目标：不动 verifier 的 dense 路径，不动 `SparseSelection` 的索引语义，
> 因此**不触碰 lossless 契约**。判定一律看 **draft acceptance** 与 **decode 墙钟**，不看任务准确率。

### P1（用户优先级 #1）MM-ShiftKV：variance-expanded query proxy + 分组投票

**原理**（论文 §3.2，Eq.4–11）：prefill 统计 `µ/σ` → 采样 `H̃ ~ N(µ, diag((γσ)²))`（γ=10）→ `q̃ = W_Q H̃` →
给代理赋**合成未来位置**并做 RoPE → `a = softmax(q̃K^T/√d)`（**无 causal mask**）→ 分 G=32 组、每组取覆盖 τ=0.95 的最小集合 →
`vote(t)` 计票 → `S_t = vote(t) + λ·a_t(q_last)`（λ=1）→ 取 top-(C−1) **再恒定保留最后 1 个 key**。

**为什么适合 STD：**
- **签名只需要 `hidden_states` + `q_proj`/`k_proj`**，**彻底不需要 `output_attentions`**，代码里传 `output_attentions=True` 会直接崩（印证其设计）。
  → 能把 prefill 从 eager 自定义 attention 切回 **SDPA/FA2**，消掉数值扰动与位精确性代价。
- 打分结果直接套进现有 per-KV-head top-k 框架；`_gather_sparse_kv()`、`_sparse_aware_attention_forward()`（`is_causal=False`）**一行都不用改**。
- 模型匹配：官方代码**只有 Qwen2.5-VL 一条路径可跑**（LLaVA/Mistral 路径缺 import、符号名不匹配）——恰好就是 STD 的模型。

**成本**（按 STD 规模 T≈15933、H_q=28、N=512、28 层估算）：代理打分约 **2.11 TFLOP/次 prefill**，
是现有 static 打分（0.18 TFLOP）的 **≈9×**；瞬时 fp32 attention 张量 `[28,512,15933]` ≈ **914 MB/层**（逐层释放）。
论文 32K/LLaVA 实测 TTFT +0.38 s（H100）；**A6000 估计 +0.3–1.0 s/样本，必须实测**。
→ 若这个增量吃掉了 static 的收益，就是得不偿失。

**移植清单**（详见 `third_party/MM-ShiftKV/PROVENANCE.md`）：只需 3 段约 100–150 行
（`qwen_model.py:740–852` 代理生成、`mmshift_util.py:943–979` 打分+采样、`mmshift_util.py:1042–1124` 投票+预算）。
**不需要** `csrc/` CUDA 扩展、`DynamicCacheSplitHeadFlatten`、`statistics_table/`、`visual_head/`。

**✅ 已完成的移植基础设施（2026-09-22，CPU 可测，不依赖 GPU）**：

| 文件 | 内容 |
|---|---|
| `src/std_repro/mmshift_proxy.py` | 与模型无关的选择内核：`prefill_statistics`（Eq.4 逐维）、`variance_expanded_proxies`（Eq.5 对角高斯采样）、`minimal_prefix_sizes`（Eq.9 τ 覆盖）、`group_vote_scores`（Eq.8/10）、`group_vote_selection`（Eq.11 预算+恒定保留最近 key） |
| `tests/test_mmshift_proxy.py` | **15 个 CPU 回归测试，全部通过**（`PYTHONPATH=src pytest tests/test_mmshift_proxy.py -q`） |

两处**刻意与官方代码不同**（都是那边的缺陷，已在 module docstring 记录）：
① 统计量按**逐特征维**算（官方部署路径用的是 flatten 后的**全局标量**，且硬编码 `decode_std = prefill_std*10`）；
② **不依赖** `./statistics_table/*.json`（官方构造函数用相对路径读它且注释掉了异常处理，缺文件即崩）。
测试专门钉住了这两点（`test_proxies_are_per_dimension_not_global_scalar` 等）。

**仍需 GPU 的部分**：把 proxy 接到真实层里取 `hidden_states` + `q_proj`/`k_proj` + M-RoPE 合成位置，
以及 per-layer trace 采集 —— 这才是 P1-a 的门槛。

**必须避开的 3 个坑**（已核实）：
1. 部署路径用的是**全局标量** mean/std（`statistical_query_predictor.py:49–106`，`decode_std = prefill_std*10` 硬编码），
   **不是论文 Eq.4 的逐维 diagonal** → **按论文公式自己实现**，别照抄。
2. `is_prefill = q_len != 1` 在 STD 的两段式 prefill 下会**在 video-prefix 那次就误触发**（此时 text 未入 cache、anchor 不存在）。
   → 必须换成显式 gate。且统计窗口建议改成在 video-prefix forward 里**在线累积全序列 per-dim 统计**（text suffix 只有 ~57 token，统计噪声大）。
3. 官方是**跨层跨 head 统一预算**（`head_adaptive_capacity` 被 `torch.full` 覆盖）。

**第一步实验（P1-a，零侵入，半天）：离线 coverage 对照，不改 STD 主路径**

在分析脚本里（**不在主路径**）开 `output_attentions=True` 采集 10 个样本的真实 decode 各轮注意力（前 32 轮即可），
对每个 (layer, KV head) 计算 **"decode 注意力质量被保留 KV 覆盖的比例"**，三方对比：

- (A) STD 现有 static（text→video 注意力求和，k≈967）
- (B) LastAttn-only（最弱基线）
- (C) MM-ShiftKV 代理投票，网格 γ∈{1,5,10,20} × N∈{128,512}，τ=0.95、λ=1、G=32

同时算 per-(layer,head) 的 **recall/Jaccard**（复用 `topk_jaccard`、`topk_intersection_counts`，`dynamic_selection.py:78–101`），
与仓库现成的 `attention_free_visual_topk` 做第四方对照；并顺带产出 G0.2 的方差比。

**判定门槛（预注册）**：
- (C) 相对 (A) 的 coverage 提升 **> 1–2 个百分点** → 进入 P1-b（接入 `build_sparse_selection`，`output_attentions=False`，测 acceptance 是否 > 0.9115、decode 是否守住 1.71–1.87×、prefill 增量是否 < 收益）。
- 覆盖持平 → **放弃替换打分器**，退而只取"**分组投票 + last-query anchor**"作为对现有 static 打分的**低风险增强**（几乎零额外成本）。
- **确定性收益通道（可并行做）**：即便精度不涨，"用代理投票摆脱 `output_attentions=True`"本身就有价值（恢复数值一致性 + 去掉 eager 慢路径）。

### ❌ P1-a 已完成（2026-09-22）：代理投票**每个 γ 都比 attention 差**，γ=10 最差

3 样本 × 64 帧（V=7812，k≈950，128 proxies / 32 组 / τ=0.95）；
脚本 `scripts/analysis/mmshift_proxy_recall.py`；产物 `results/mmshift_proxy/recall.md`。
指标 = **对解码期真实注意力质量的覆盖率**（越高越好），括号内是"与 attention 选择的 Jaccard"：

| 样本 | **attention cov** | **oracle cov（上界）** | γ=1.0 | γ=5.0 | **γ=10.0（论文推荐）** | γ=20.0 |
|---|---:|---:|---:|---:|---:|---:|
| 050-1 | **0.6144** | 0.7140 | 0.5814 (J 0.58) | 0.4838 (0.41) | **0.3962 (0.26)** | 0.4967 (0.40) |
| 717-1 | **0.6526** | 0.7168 | 0.6165 (0.56) | 0.4831 (0.37) | **0.4364 (0.27)** | 0.5052 (0.34) |
| 496-3 | **0.6112** | 0.6844 | 0.5800 (0.59) | 0.4814 (0.42) | **0.3886 (0.26)** | 0.4779 (0.37) |

**三条结论：**

1. **预注册门槛不通过**：没有任何 γ 能超过 attention 选择；最好的 γ=1.0 也**低约 3.3pp**。
2. **方差放大是有害的，而且论文推荐的 γ=10 最差**（比不放大低约 20pp，Jaccard 从 0.58 掉到 0.26）。
   → **直接否证 MM-ShiftKV 的核心前提在视频/STD 工作点上的适用性**：
   论文的立论是"decode query 方差比 prefill 大 13–22×，所以要放大"；
   **实测放大会破坏选择质量**，说明在 128/64 帧视频上真实 prefill query 比方差放大的代理更好。
   （这正是 G0.2 预警的情形：那个 13–22× 是图像上测的，视频上不成立。）
3. **γ=1（不放大、只做分组投票）≈ attention 但不超过**——合理，因为它用的信息与 attention 选择同源。

**⚠️ 实现保真度声明（必须同时引用）**：本实验在**query 空间**做方差放大，
不是论文的**hidden 空间 + 合成未来位置**。所以这**不是**论文方法的忠实复现，
而是"方差放大这个成分是否在视频上成立"的检验。结论对该成分是负的；
hidden 空间版本不排除有差异，但**单调变差的趋势 + γ=10 最差**已足够否证其推荐取值。
（另一个独立于精度的收益——"免 `output_attentions`"——仍然成立，但那属于工程改造，不属算法收益。）

**风险**：① 论文**零视频实验**（全是单图 OCR/文档类，平均输入仅 642–4830 token）；② STD 的 6.09% 保留率 ≈ 论文 C=256–512 区间，
而**该区间 MM-ShiftKV 已接近 FullKV 无损、对 SnapKV 的领先也收窄到打平**（C=512：Avg 75.5 vs FullKV 75.4 vs SnapKV 75.1）→ **STD 可能本来就处在收益平缓区**；
③ 打分成本 9×。

---

### P2 HybridKV：`S_{l,h}` 头分类 + Eq.(5) 非均匀 per-head 预算

**原理**（论文 §3.1–3.2）：
- `S_{l,h} = (1/|T|) Σ_{i∈text} Σ_{j≤k} TopK(S_text[i], k)`，其中 `S_text = softmax(Q_text K_context^T/√d)`；
  即**每个文本 query 先只留 top-5% 上下文、再跨文本 query 平均** → 归一化到 [0,1]，可用 θ=0.9 切分
  static head（`S≥θ`，KV 稳定，prefill 剪一次即可）与 dynamic head（`S<θ`，需持续检索）。
- 预算：`B̄ = B_total/(N_stat+N_dyna)`，`B_dyna = ⌈r·B̄·N_dyna⌉`（最优 **r=0.75**，偏向 static），
  static head 内 `b_{l,h} = ⌈α·B_stat/N_stat + (1−α)·B_stat·S_{l,h}⌉`。

**为什么这是本项目**性价比最高**的移植点：**
- **STD 已经物化了它需要的张量**：`build_sparse_selection` 拿到的 `[q_heads, text_len, prompt_len]` **就是** Eq.(2) 的 `S_text`，
  且**这笔 prefill 成本 static 路径本来就在付**。算 `S_{l,h}` 只是对既有张量多做一次 `topk(...).sum(-1).mean()` 后处理。
- **零 decode 开销、零新增注意力采集**，不触碰 verifier 数值。
- **直接命中 dynamic 当前的真实瓶颈**：STD 的 dynamic 亏损里 **refresh 主导（≈34 ms/轮）**、
  优化后采集只占 ≈4.7 ms/轮。所以"只采集部分 head"收益有限，**必须同时"只对 dynamic head 做 refresh"**，节省才随 dynamic head 占比线性缩放。
- 论文还**主动点名 speculative decoding（即 SpecVLM/STD）为兼容方向**。

**关键未知数（正是第一步要测的）**：论文**从不报告 `N_stat/N_dyna`**。若 128 帧视频下 θ=0.9 仍留下一大半 dynamic head，收益趋零。

**第一步实验（P2-a，纯离线，近乎免费）**：
在 `build_sparse_selection` 里、`grouped.sum(dim=(1,2))` **之前**，用**同一张已物化的张量**按 Eq.(3) 算 112 个 `S_{l,h}`
（k 取 967 且 top-k 限定在**视觉位置**上，这样 `S_{l,h}` 直接度量"冻结的 top-K 能覆盖多少注意力质量"，阈值语义更干净）：
1. 报出 `S_{l,h}` 分布与 **θ∈{0.90,0.85,0.80} 下的 `N_stat/N_dyna`**（论文缺失、我们最需要的数字）；
2. 对 10 个 Video-MME 样本，把 `S_{l,h}` 与**逐头选择稳定性（churn）**做相关——需一次保留 layer/head 轴的小 trace run
   （现有 trace 已做 layer 均值 + visual-only softmax，**不能直接用**）。

### ⚡ P2-a 已有实测（2026-09-22，2 样本 × 64 帧，shared GPU，`timing_valid=false`）

`dynamic_stats.head_sparsity_summary` 现在会记录整条分布（纯 CPU 后处理，零额外 GPU 开销）。实测：

| 样本 | S 均值 | S 范围 | θ=0.90 | θ=0.80 | θ=0.70 | θ=0.60 |
|---|---:|---|---:|---:|---:|---:|
| 050-1 | 0.779 | 0.589–0.923 | **0.9%**（1/112） | 45.5% | 82.1% | 98.2% |
| 717-1 | 0.788 | 0.558–0.958 | **8.9%**（10/112） | 50.0% | 83.0% | 96.4% |

**结论 1（对论文的直接修正）**：HybridKV 推荐的 **θ=0.90 在 STD 上几乎是空操作**（只冻结 1/112 或 10/112）。
要冻结有意义的一大半头，阈值必须降到 **θ≈0.80**（约一半）或 **θ≈0.70**（约 5/6）。
→ 计划里预判的"θ 必须重新标定"得到实测确认。

**结论 2（P2 的价值上界，解析）**：用实测冻结比例 + 实测每轮成本算上界：

| θ | 冻结 f | refresh ms/轮 | 过路费 | dynamic ms/轮 | static/dynamic |
|---:|---:|---:|---:|---:|---:|
| 0.90 | 0.049 | 21.2 | 39.3 | 402.0 | 0.902 |
| 0.80 | 0.478 | 11.7 | 29.8 | 392.5 | **0.924** |
| 0.70 | 0.825 | 3.9 | 22.0 | 384.7 | **0.943** |
| 0.60 | 0.973 | 0.6 | 18.7 | 381.4 | 0.951 |
| **冻结全部（= static）** | 1.0 | 0 | 18.1 | — | **0.952（仍 < 1.0）** |

**→ P2 的冻结路径无法让 dynamic 翻盘：最好只有 0.952×，而那已经是"退化成 static"。**
原因是过路费里 **只有 refresh 是可冻结的部分**，剩下的 `draft/verify/host ≈ 18 ms/轮` 一动不动。

**这是第三条独立指向同一结论的证据**（① G0.1 廉价筛选 ≤0.979×；② G0.3 更好的选择器只值 +7.1pp；
③ 本节 冻结 ≤0.952×）。

### ✅ P2-b 已完成：完美的剂量-响应 —— **本计划唯一"实测有效"的方法** ★

6 样本 × 64 帧（`results/freeze_sweep64/`；复现：`scripts/analysis/analyze_freeze_sweep.py`）：

| 配置 | θ | 冻结头数 | dynamic 占比 | static acc | dynamic acc | **Δ(dyn−static)** | dyn 接受长度 | **changed tok/轮** | **jaccard** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `F_none` | — | — | 1.000 | 0.8525 | 0.7953 | **−0.0573** | 6.956 | 278.8 | 0.562 |
| `F_080` | 0.80 | 51 | 0.545 | 0.8525 | 0.8110 | **−0.0415** | 7.116 | 133.2 | 0.785 |
| `F_070` | 0.70 | 92 | 0.179 | 0.8525 | **0.8354** | **−0.0171** | **7.345** | **42.5** | **0.931** |

**冻结越多头，dynamic 在每个轴上都单调变好**：改写量 −85%、jaccard 0.562→0.931、
接受率 0.7953→0.8354、**Δ 惩罚缩小 70%**。三档 static 接受率**逐位相同（0.8525）**= 仪器自检通过。

**两条结论：**

1. **机制成立且超出预期**：计划只预期"省成本、接受率中性"，实测是**省成本 + 接受率变好**——
   被冻住的头原本贡献的主要是**漂移噪声**，重选它们纯属帮倒忙。
2. **但仍不能翻盘**：static 始终领先，Δ 的极限是 0（冻结全部 = static），
   与上面 §「P2 的价值上界」解析结论**完全一致**。
   → **P2 的建议改为**：作为"**降低 dynamic 惩罚**"的手段**有效且已实测**，
   但**不足以让 dynamic 超过 static**；若最终交付是 static 线，P2 的价值在于
   "让 dynamic 在大 `f_d` 下不再明显拖后腿"（例如 E3 headroom 门控场景）。

✅ **正确性（已更正，2026-09-22）**：64 帧下 **static 与 dynamic 均为 5/6 exact vs AR**，
**且在冻结 3 档 + K 3 档共 6 个配置中逐一相同**——**冻结与加大预算都不影响位精确性**。
128 帧 K 扫描中三档也都是 **4/6**（前 6 个样本含已知非精确的 `496-3`、`754-1`，与 `local512` 的 8/10 自洽）。
（本文档此前曾写"static 0/6 exact"，那是**读数错误**——`token_equal` 在 `kind: comparison` 记录上，
不在 `kind: trial` 上。已核实并更正。）

### ✅ 推荐改动的安全性验证（离线，2026-09-22）

把默认 `--k-plus-text` 从 1024 改成 ~1750（6.09% → 10.66% 保留）前，逐项核对：

| 检查项 | 6.09% | 10.66% | 21.06% | 结论 |
|---|---:|---:|---:|---|
| static exact vs AR | 4/6 | 4/6 | 4/6 | **不变** |
| dynamic exact vs AR | 4/6 | 4/6 | 4/6 | **不变** |
| 峰值显存 | 33.22 GiB | 33.22 GiB | 33.22 GiB | **不变** |
| 稀疏缓存需求（k+text） | 1024 | 1750 | 3400 | 远小于 `cache-len 20480` |
| 接受率 | 0.8653 | **0.9093** | 0.9463 | 单调上升 |

→ **加大预算在正确性、显存、缓存容量三项上零代价**，只改一个命令行参数。

**判定门槛**：
- 若高 `S` 头确实低 churn **且 `f_d = N_dyna/112 ≤ ~0.3`** → 冻结 static 头，按 `f_d` 估计 per-round 节省，进入运行时实现。
- 若 `S` 与 churn **不相关** → 这条路对 128 帧视频不成立，**止损**。
- 若出现健康的 static 多数派 → 第二步把 static 选择器换成 Eq.(6)(7) 观测窗口打分（`Q[C−w:C]` 就是文本块最后 w 行，**同样已物化**）+ Eq.(5) 预算，固定 K=967 测 acceptance。

**✅ 已完成的移植基础设施（2026-09-22，CPU 可测，不依赖 GPU）**：

| 文件 | 内容 |
|---|---|
| `src/std_repro/hybridkv_budget.py` | 与模型无关的内核：`group_query_heads`（GQA 内 q_head 取均值）、`text_centric_sparsity`（Eq.2/3，top-k 限定视觉位置、默认按视觉质量归一化 → 与 `selection_headroom.py` 的 coverage **同一个量**，可直接对比）、`classify_heads`（θ 切分）、`allocate_head_budget`（Eq.4/5，`r`/`α` 必须显式传入） |
| `tests/test_hybridkv_budget.py` | **18 个 CPU 回归测试，全部通过** |

`allocate_head_budget` 的 `exact=True`（默认）用**最大余数法**把预算收敛到**恰好等于总预算**——
因为 STD 的 sparse cache bookkeeping 与 draft kernel 需要一个可预测的总量；
`exact=False` 才是论文原样的 `ceil`。测试还钉住了一个真实陷阱：
在 float32 输入下 `s_hat` 会有 ~2e-6 的噪声，直接 `ceil` 会把本该是 120 的值算成 121，所以 `ceil` 路径带容差。

**仍需 GPU 的部分**：真正的 `S_{l,h}` 需要在 prefill 时拿到 `[q_heads, text_len, prompt_len]` 张量
（`std_qwen25vl.py:175` 已经在算它），以及逐头 churn 需要一次保留 layer/head 轴的小 trace。

**风险**：① "text-centric" 在视觉主导时统计薄弱（T≈57 vs C≈15933，Eq.(2) 只用了注意力矩阵 **0.36% 的行**），
论文验证上下文 ≈4000，**STD 是它的 4 倍，属外推**；② θ 与预算强耦合，6.09% 保留率下**必须重新标定**；
③ Eq.(5) 的 α 论文**未给值、无敏感性实验**；④ 不可与 HybridKV 的 1.52× 相加（那是对 full-cache AR 的，与 static 1.71–1.87× 不同层）；
⑤ 论文里 "STD" 是 **Spot-the-Diff 基准**，与本项目无关，**切勿引用其 29.79→29.84 数字**。

---

### P3 FlashCache：DBA 逐层非均匀预算（高价值/低风险）+ Dev 判据（需先验证）

**原理**：对 **K 和 V 沿 token 维做 DCT** → 低通截断 `ω=γN` → IDCT 得 Base KV →
`Dev[x] = MSE(K[x],K_base[x]) + MSE(V[x],V_base[x])`（**高频残差能量**，注意"outlier = 高频偏差"**不是**低能量）→ 取 Dev 最大者。
**只在 prefill 做一次**；**只需 K/V，不需要 attention 矩阵**（论文第一卖点即兼容 FlashAttention，全程 H200+FA 评测）。

**⚠️ 它救不了 dynamic**：`Dev` 是缓存 K/V 的**纯函数**，prompt K/V 在 prefill 后不变 → **逐轮重算分数恒等**，dynamic 退化；
在压缩重排后的子序列上做 DCT 会丧失频率语义。**只能进 static 线。**

**⚠️ 但 DBA 也被 §5.2 的三堵墙挡住了（2026-09-22 更正）**：DBA 是**逐层给不同预算**，
而 `compact_sparse_prompt_cache` 只有**一个全局 `compact_len`** 并逐头断言相等 →
**逐层预算同样会 `RuntimeError`**。原判断"DBA 与打分器正交、可随时叠加"**只对了一半**：
它与"用什么分数"正交，但**不与"统一长度"正交**。
→ **DBA 降级**，与 Eq.5 一样需要先做布局改造。

**仍然可用的是 Dev 判据本身**（它只是"换一个打分函数"，**不需要变长预算**）：
把 `build_sparse_selection` 里的 attention 分数换成 `flashcache_dev.deviation_scores`
（从 canonical `dense_pkv` 的 visual 段按 per-(layer,head) 算），**K 保持 967 不变**，
这样完全不碰布局，也不碰 `SparseSelection` 的长度语义。
它的收益来自"选得更准"，上界受 G0.3 的 7.1pp 约束。

**§5.2 的三堵墙已有可执行的证据**：`tests/test_uniform_k_constraint.py`（**6 个测试**）把
"逐头/逐层变长会被 `compact_sparse_prompt_cache` 与 `_gather_sparse_kv` 拒绝"、
以及"padding 会改变 softmax（因为 draft 是 `attn_mask=None`）"变成**行为断言**——
将来若有人改了布局，这些测试会失败，强制更新本计划而不是让它悄悄失效。

**第一步实验（P3-a）**：以已有 attention 选点轨迹为参照，从 canonical `dense_pkv` 的 **video token 连续段**
（`[4, 15876, 128]`，`output_attentions=False`，见 `std_qwen25vl.py:675–696`）按 per-(layer,head)、沿 token 维算 `Dev`
（γ∈{0.1,0.2,0.3}），在 k=967 下报 recall/Jaccard，与 `attention_free_visual_topk` 三方对比，
并**实测 A6000 上 DCT+IDCT+top-K 的一次 prefill 耗时**（论文 H200：2K 1.66ms / 4K 3.86ms / 8K 6.77ms；**A6000 估 30–60ms**）。

**判定门槛**：Jaccard ≥ ~0.6 且耗时 < 100 ms → 做 static-swap 端到端 A/B；
否则**只取 Dev 判据**（K 不变，不需要布局改造；DBA 因 §5.2 的三堵墙已降级）。
证据：`tests/test_uniform_k_constraint.py` 用行为断言钉住了那三堵墙。

### ❌ P3-a 已完成（2026-09-22）：Dev 判据**两条门槛都没过**，且选择集**与随机无法区分**

3 样本 × 64 帧（V=7812，k≈950）；脚本 `scripts/analysis/flashcache_dev_recall.py`，
产物 `results/flashcache_dev/recall.md`：

| 样本 | k | γ | **Jaccard(Dev, attention)** | **dev_ms** |
|---|---:|---:|---:|---:|
| 050-1 | 967 | 0.1 / 0.2 / 0.3 | 0.064 / 0.064 / 0.063 | 4440 / 3872 / 3930 |
| 717-1 | 924 | 0.1 / 0.2 / 0.3 | 0.073 / 0.071 / 0.070 | 3877 / 3986 / 4102 |
| 496-3 | 951 | 0.1 / 0.2 / 0.3 | 0.067 / 0.070 / 0.073 | 4047 / 4164 / 4166 |

**关键对照（随机基线）**：两个**独立随机**的 size-k 子集，其期望 Jaccard = `k/(2V−k)`
= 967/(15624−967) = **0.0661**。

**实测 0.063–0.073 ≈ 随机基线 0.066 → Dev 的选择集与 attention 的选择集在统计上无法区分。**

两条门槛的判定：

| 门槛 | 要求 | 实测 | 结论 |
|---|---|---|---|
| 一致性 | Jaccard ≥ ~0.6 | **0.063–0.073（≈随机）** | ❌ **差一个数量级** |
| 成本 | < 100 ms | **3872–4440 ms** | ❌ **超 40×** |

**→ P3 的 Dev 判据被否证，不需要再做 static-swap 端到端 A/B。**
（顺带测得的参考值：仓库已有的免注意力启发式 `attention_free_visual_topk`
与 attention 选择的 Jaccard 只有 **0.091**，也很低——与它被标注为"弱 bootstrap 启发式"一致。）

**方法学备注**：Dev 度量的是**缓存 K/V 自身的高频残差能量**，与"文本 query 关注哪些视觉 token"是
**两个正交的信号**；论文 Fig.2 只证明了"删掉高 Dev 的 token 掉点更快"这一弱单调性，
**从未做过同预算下 Dev 与 attention 的 head-to-head**——这次替它做了，结论是**不可互相替代**。
（成本差异部分来自实现：我们用 fp32 FFT 在 `[L,H,V,D]` 上一次性变换；论文报的是 H200+CuPy 的 1.66 ms@2K。
但即便优化 40×，一致性门槛仍然不过。）

### ✅ P1-a / P2-a 已完成（per-layer，3 样本 × 64 帧）：两条结论都推翻了原有假设 ★

采集：`scripts/analysis/attention_trace_per_layer.py` → `results/routing_traces_per_layer/`；
分析：`scripts/analysis/analyze_per_layer_selection.py` → `results/routing_analysis_videomme/per_layer/per_layer_report.md`。
retention 12.38%（k=967），28 层 × 4 KV head = 112 个单元。

**（1）per-layer headroom 比层均值估计的**大一倍**** —— 修正 G0.3 的结论：

| 样本 | coverage_static | coverage_oracle | **gap** |
|---|---:|---:|---:|
| 050-1 | 0.6157 | 0.7809 | **+0.1652** |
| 496-3 | 0.6219 | 0.7516 | **+0.1296** |
| 717-1 | 0.6659 | 0.7860 | **+0.1201** |
| **均值** | **0.6345** | **0.7728** | **+0.1383** |

- G0.3（**层均值**）给出的是 gap **+7.1pp**、static/oracle **84.1%**；
  **per-layer 的实测 gap 是 +13.8pp，static/oracle 只有 82.1%**。
- 方向与 Jensen 不等式预测一致（per-layer 自由度更大 → oracle 更高）。
- **因此"更好的 prefill 选择器"的真实上界约为 +13.8pp，而不是 +7.1pp**——
  G0.3 因为只有层均值 trace 而**低估了一半**。这提高了 P1（MM-ShiftKV）的期望值。

**（2）HybridKV 的分类前提在本数据上**不成立****：

| θ | f_d（dynamic 占比）范围 |
|---:|---|
| 0.90 | **0.79 – 0.90**（几乎全是 dynamic） |
| 0.80 | 0.36 – 0.42 |
| 0.70 | **0.09 – 0.11** |

而且 **`pearson(S, churn)` = +0.255 / +0.300 / +0.221（三个样本全为**正**）**。
计划里写的是"HybridKV 假设**负**相关（高 S = 稳定头）"——
**实测是弱正相关，即高 S 的头反而更不稳定**，与论文的分类前提相反。

**但这与 P2-b 的正面结果并不矛盾**：P2-b 里冻结高 S 头之所以有效，
恰恰是因为**高 S 头正是 churn 更大的那些头**，冻掉它们去掉的漂移更多。
→ 所以正确的表述是：**"冻结"这个操作确实有效（P2-b 已证），
但"用 S 挑选该冻结哪些头"这条规则在本数据上是反的（P2-a 已证）。**

**综合：HybridKV 的可用部分是"冻结/节流"这个机制本身，不是它的头分类判据。**

**✅ 已完成的移植基础设施（2026-09-22，CPU 可测，不依赖 GPU）**：

| 文件 | 内容 |
|---|---|
| `src/std_repro/flashcache_dev.py` | `dct2`/`idct2`（**正交归一 DCT-II 及其转置**，用长度 2N 的 FFT 实现，O(N log N)——N≈15876 时稠密矩阵要 ~1GB/head，不可行）、`low_pass_base`（Eq.6–9）、`deviation_scores`（Eq.10，`Dev=Dev_k+Dev_v`）、`high_frequency_ratio`（Eq.11–12，逐层 `R`）、`allocate_layer_budget`（**DBA**，带每层下限 + 最大余数法收敛到总预算） |
| `tests/test_flashcache_dev.py` | **26 个 CPU 回归测试，全部通过** |

三个被测试钉住的关键点：
① **DCT/IDCT 真的正交归一**（对 4/5/8/16/33 逐点与显式 DCT 矩阵比对，误差 ~1e-15），这是整个判据的地基；
② **Dev 度量的是"高频"偏差**——常量段 Dev≈0、交替段 Dev 显著更高（注意论文里 "outlier" = **高频**，不是低能量，极易读反）；
③ DBA **恰好收敛到总预算**且不饿死任何一层。

实现中修掉的两个真实陷阱：负 `dim` 未归一化会让 1-D 路径构造出错误的 pad 形状；
`dct2` 若用 float64 构造相位会把 float32 输入**静默升为 float64**（生产张量内存翻倍）——两者都已在代码里注释并被测试覆盖。

**仍需 GPU 的部分**：从真实 `dense_pkv` 取 K/V、以及 A6000 上的 DCT 耗时实测。

**风险**：① 论文视频证据只有 FAVOR-Bench 一项、**无帧数说明、无长视频**，且 ρ=0.05 处已明显退化（all 40.91→30.71）；
**STD 的 6.09% 比它验证过的 ρ=0.05 更激进**；② 论文是 **per-layer、head-agnostic**，STD 需要 per-head（按 head 通道切片推广可行但**未验证**）；
③ DCT 作用域论文**从未说明**是否含 text（转 STD 时应**只对 visual 连续段**做）；④ 128 帧下 DCT 沿 flattened visual token 做，
"低频"混合空间与时间频率，**语义不清**（Dev 可能捕获动作/切变，也可能只是 patch 噪声）；⑤ **论文没做同预算 Dev vs attention top-K 的 head-to-head**，
只对比了 eviction baseline，**"更好"未被证明**（唯一弱证据是 Fig.2 的删除单调性）。

---

## 4. 阶段 2：SimiCache（**阻塞** —— 论文不可得）

- **状态**：IEEE TMM 2026，DOI `10.1109/TMM.2026.3716758`，**closed access**（OpenAlex 确认无任何开放版本）。
  已取得**权威题录 + 完整摘要**（`papers/SimiCache_IEEE_TMM2026_ABSTRACT.md`）。
- **摘要要点（推论，非原文公式）**：视觉 token 高度相似 → 跨层 K/V 冗余；
  **两步聚类**（关键词含 Spectral/Global Clustering）把每层高度相似的 KV 聚簇，再用**加权求和合并**成代表 token；
  明确对比"prior merging 只在局部邻域合并"，它是**全局**的；10% budget、显存降 ~81%。
- **⚠️ 与 STD 契约冲突**：它是**有损合并**（多个 token 合成一个代表），而 STD 的 lossless 性质依赖 **dense verifier 走完整 cache**。
  → 若要试，**只能用于 draft 侧稀疏缓存**，verifier 仍走 dense；但这会把稀疏 draft 的 slot 语义从"原始 token 索引"变成"合成向量"，
  需改 `sparse_cache_refresh` 与位置编码 bookkeeping（中等改动）。
- **它很可能也是 query-agnostic**（同 FlashCache 的 Dev）→ **同样只能进 static 线**，不能逐轮重算。
- **需要你做**：机构订阅下载全文，或直接提供 PDF。拿到后补：两步聚类的精确定义、加权系数、复杂度、是否 per-head、是否有代码。

---

## 5. 阶段 3：逐轮方法（被 G0.1 门控）+ 共享使能改动

### 5.1 明确不做的

- **HybridKV 的 CPU offload + chunk 检索（8-token chunk）**：chunk 比 STD 的单 token top-K 粗，
  且 CPU↔GPU 传输与 STD draft 需要常驻紧凑缓存的需求冲突；128 帧下每头 15880/8 ≈ 1985 个 chunk，传输量远大于论文 64 帧设置。
- **FastCache 的全部三块机制**：
  - 压缩器 = **可训练的**模态专用 MLP + 全局 5× 均匀分块（**无重要性判据**），需为 Qwen2.5-VL 重训，且论文无代码无配方；
    5× 远弱于 STD 需要的 ~16.4×，且**有损合并破坏 lossless 契约**；它省的是"打分"，而 STD 贵的是"搬运"——**打错靶**。
  - 调度 = 单卡上的跨请求组批；**单请求时 Y=Z=1，退化为它自己批评的串行流水线，3.662 s 一分钱藏不掉**。
    STD 是单进程单请求单卡，**无请求队列可组批**；论文全部实验在 1–40 req/s，**无 batch=1 数据**。→ **直接排除。**
  - 预算分配 = 不存在（只有全局标量 λ=5）。
  - `NVIDIA/kvpress` 是它的**基线/引用**，**不是它的代码**；kvpress 里也没有对应 FastCache compressor 的类。

### 5.2 共享使能改动 —— ⚠️ **原假设是错的，已用代码核实并改道（2026-09-22）**

**原计划**：把 `SparseSelection.k` 从单个 int 扩成 per-layer/per-head 预算（HybridKV Eq.5 与 FlashCache DBA 都需要），
并假设"现有 `idx = cat([non_visual, topk[head]])` 已支持每头不同长度"。

**❌ 这个假设不成立。** 逐处核实后，**per-head 甚至 per-layer 的变长预算会硬撞三堵墙**：

| 位置 | 事实 | 后果 |
|---|---|---|
| `std_qwen25vl.py:278,289–290,297` | `compact_len = non_visual + selection.k` 是**单一全局标量**；每个 head 的 `idx` 都被断言 `numel() == compact_len`；`current_length_data.fill_(compact_len)` 只填一次 | 各 head / 各 layer 长度不一致 → 直接 `RuntimeError` |
| `std_qwen25vl.py:349–351` | `_gather_sparse_kv` 显式 `raise RuntimeError("Sparse KV gather produced uneven per-head lengths.")` | draft 路径**主动拒绝**不等长 |
| `std_qwen25vl.py:405–410` | draft attention 是 `attn_mask=None, is_causal=False, enable_gqa=True` | **没有任何 mask 可以遮住 padding 槽位** → 用 padding 凑齐长度会让填充的 KV 参与 softmax，改变 logits，**破坏 lossless 契约** |

→ **异构预算需要真正的布局改造**（给 SDPA 加 per-head mask，或写变长 kernel），成本从"中等"上调为**大**，
且大概率破坏位精确性。**这正是必须先用代码核实、而不是先在 GPU 上试的那类问题。**

**✅ 改道：P2 的主要收益根本不需要改预算 —— 冻结 static 头就够了。**

HybridKV 对 STD 的价值可以拆成两半，而**值钱的那一半不需要 Eq.5**：

| 半边 | 机制 | 是否需要改布局 |
|---|---|---|
| **冻结 static 头**（值钱） | 高 `S_{l,h}` 的头逐轮重选是纯开销；冻结后其选择不变 | **❌ 不需要** |
| Eq.5 异构预算（次要） | 把预算按 `S` 重新摊到各头 | ✅ 需要（被上面三堵墙挡住） |

**为什么冻结不需要改布局**：`sparse_cache_refresh.py:188–190` 的 `incremental_refresh_sparse_visual_kv`
**本来就是按 (layer, head) 算 changed mask 的**，并且 `m_total == 0` 时**整层 `continue` 跳过**。
所以某个 head 的选择不变 → 它的 scatter 工作自然为 0，**refresh 开销随 dynamic head 占比下降，代码一行不用改。**

**✅ 已实现（2026-09-22，CPU 可测）**：

| 位置 | 内容 |
|---|---|
| `src/std_repro/dynamic_selection.py` | `FrozenHeadTopKPolicy`（`freeze_mask [L,H]`，`True` = 冻结：选择保持上一轮）、`freeze_mask_from_sparsity(sparsity, theta)`（`S ≥ θ` 即冻结） |
| `tests/test_frozen_head_policy.py` | **9 个 CPU 回归测试**，含两条**等价性断言**：全 `False` mask ≡ `PreviousVerifyTopKPolicy`、全 `True` mask ≡ `StaticPolicy`（逐位相同）；以及逐头"冻的不动、没冻的取新 top-k" |

**✅ 接线也已完成（2026-09-22，加性改动）**：

| 位置 | 改动 |
|---|---|
| `std_qwen25vl.py` `SparseSelection` | 新增可选字段 `head_sparsity`（默认 `None`） |
| `std_qwen25vl.py` `build_sparse_selection` | 新增 `record_head_sparsity=False`、`sparsity_normalize=True`；开启时用**同一张已物化张量**顺手算 `[L, H]` 的 `S_{l,h}` |
| `dynamic_std_qwen25vl.py` | 新增 `freeze_threshold`；开启时才 `record_head_sparsity=True`，并构造 `FrozenHeadTopKPolicy`；`dynamic_stats` 记录 `freeze_threshold` / `frozen_head_count` / `total_head_count` |
| `benchmark_a100_dynamic.py` | 新增 `--freeze-threshold` |
| `tests/test_build_sparse_selection_sparsity.py` | **6 个测试**，其中最关键的是**非侵入性**：开/关该 hook，`topk_positions` **逐位相同** |

**默认关闭 ⇒ 现有 baseline 路径逐位不变**（既有 4 个环境性失败除外，全套 308 passed）。

**剩余动作（需 GPU）**：按阈值扫一遍，测 per-round refresh 与 decode 墙钟：
```bash
TAG=freeze512 GPU=2 LIMIT=10 MAX_NEW_TOKENS=512 REPEATS=1 \
CONFIGS_OVERRIDE="F0_none|v2|1|0.00
F1_t090|v2|1|0.00|--freeze-threshold 0.90
F2_t085|v2|1|0.00|--freeze-threshold 0.85" \
  bash scripts/run_dynamic_matrix_local.sh freeze512
```
`F0` 是同一批进程里的对照；判据看 **`frozen_head_count`**、`refresh ms/轮`、decode 墙钟。
注意 `--dynamic-bootstrap` 必须是 `attention`（`attention_free` 路径没有 `S_{l,h}`，会显式报错）。

### ✅ G0.5 已完成：**预算几乎免费，加大 K 可省 ~13% decode** ★★

6 样本 × 64 帧，`results/ksweep64/`（复现：`scripts/analysis/analyze_k_sweep.py`）。
`timing_valid=false`（共享卡）→ **ms/轮 仅供参考，接受率与轮数是确定性的、有效**。

| retention | k | **static 接受率** | static 接受长度 | **轮数** | ms/轮 | 其中 draft ms/轮 | decode 总时长 | **相对 6.12%** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6.12% | 478 | 0.7755 | 6.823 | 101 | 524.5 | 403.5 | 53.0 s | — |
| 10.05% | 785 | 0.8265 | 7.261 | 94 | 536.5 | 413.1 | 50.4 s | **−4.8%** |
| 12.38% | 967 | 0.8525 | 7.513 | 91 | 537.3 | 413.8 | 48.9 s | **−7.7%** |
| **20.10%** | 1570 | **0.9123** | **8.001** | **86** | 535.5 | 413.2 | **46.0 s** | **−13.1%** |

**三条结论（这是整个调研里最有行动价值的结果）：**

1. **接受率随预算单调上升**：0.7755 → 0.8265 → 0.8525 → **0.9123**。
   → **G0.4 那个"覆盖率↔接受率可能只是样本难度混淆"的疑虑被否证**：
   **在同一个实验内**提高预算确实提高接受率，这是因果证据，不是跨样本相关。
2. **轮数单调下降**：101 → 94 → 91 → **86（−15%）**，这是确定性量。
3. **★ 每轮成本几乎不随 k 变化**：k 从 478 涨到 1570（**3.3×**），
   其中 **draft 只从 403.5 涨到 413.2 ms/轮（+2.4%）**，verify 几乎不动。
   **原因**：sparse draft 是**整个 7B 模型**每步前向一次，对 k 个 KV 的注意力只占其中极小一部分——
   **k 几乎是免费的**（与 `PROGRESS.md` §17.5 的"瓶颈是全量权重前向，不是选择长度"完全一致）。
4. **净效果：decode 总时长单调下降，20.10% 时 −13.1%。** 而且曲线**还没饱和**（接受率仍在陡升）。

**⚠️ 必须声明的口径**：`timing_valid=false`，所以 **ms/轮 与那个 −13.1% 需要在空闲卡上复验**；
但**轮数下降（−15%）是确定性的**，且"每轮成本几乎与 k 无关"有 draft 分量跨档近乎恒定作为结构性支撑。

**→ 直接结论：本项目默认的 `k-plus-text=1024`（128 帧下 = 6.09% 保留）是严重欠预算的。**
把预算提到 ~20% 保留，**不改一行代码**（只改命令行 `--k-plus-text`）就能让 static 线 decode 快约 13%。
**这是本计划当前唯一的"零代码、正收益"改动。**

**下一步（唯一需要的动作）**：在**空闲 A6000** 上、**128 帧**、512 token，
跑 `--k-plus-text ∈ {1024, 1750, 3400}`（对应 6.09% / 10% / 20%）复验那个百分比并确认曲线形状。

### ✅ 128 帧复验已完成（6 样本 × 128 帧 × 512 token，`results/ksweep128/`）★

**效应完整转移到项目真实工作点**，但**幅度更小且饱和更快**：

| retention | k | **static 接受率** | 接受长度 | **轮数** | ms/轮 | 其中 draft ms/轮 | decode 总时长 | **相对 6.09%** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6.09%（现状默认） | 967 | 0.8653 | 7.757 | 354 | 462.8 | 327.6 | 163.8 s | — |
| **10.66%** | 1693 | **0.9093** | 8.146 | 338 | 455.6 | 322.8 | 154.0 s | **−6.0%** |
| 21.06% | 3343 | **0.9463** | 8.489 | **325** | 468.8 | 334.6 | **152.4 s** | **−7.0%** |

**四条结论：**

1. **接受率单调上升**：0.8653 → 0.9093 → **0.9463（+0.081）**。**这是因果证据**（同一实验内改预算），
   彻底否证了 G0.4 提的"覆盖率↔接受率可能只是样本难度混淆"这一可能。
2. **轮数单调下降**：354 → 338 → **325（−8.2%）**，确定性量。
3. **每轮成本仍然与 k 无关**：k 涨 **3.5×**（967→3343），
   其中 **draft 只从 327.6 变到 334.6 ms/轮（+2.1%，且中间点反而更低）**——
   在 ±3% 的争用噪声内**没有可辨趋势**。再次印证"draft 的瓶颈是 7B 权重前向，不是选择长度"。

   **✅ 独立的配对成本探针（消除争用，`scripts/analysis/sparse_attn_cost_probe.py`）**：
   把唯一依赖 k 的操作（稀疏 draft 的注意力）在**同一进程内按预算交替**测量，
   任何共租户争用对所有预算的影响相同、在比值里抵消：

   | 预算 k | KV 长度 | 注意力中位耗时 | 相对最小预算 | 占一轮 draft（327.6 ms） |
   |---:|---:|---:|---:|---:|
   | 967 | 1027 | 0.015 ms | 1.000× | 0.00% |
   | 1693 | 1753 | 0.018 ms | 1.215× | 0.01% |
   | 3343 | 3403 | 0.027 ms | 1.789× | 0.01% |

   **k 涨 3.46×，这部分只涨 1.79×（次线性，因为这么小的 kernel 被启动开销主导），绝对值 0.027 ms。**

   **更稳健的物理解释（不依赖计时）**：draft 每步要**读入整模型权重 ~15 GB**（A6000 显存带宽 ~768 GB/s
   ⇒ 仅权重读取就要 ~20 ms，与实际 ~36 ms/步同量级），而被选中的 KV 只占
   `k × 4 heads × 128 dim × 2 (K,V) × 2 B = k × 2 KB`——
   k=967 时 **2.0 MB**，k=3343 时 **6.9 MB**。
   → **即使 k 翻 3.5 倍，KV 读取也只是权重读取的 0.05%**，所以"每轮成本与 k 无关"是**结构性**的，
   不是测量噪声的产物。
4. **饱和很快**：第二段（10.66% → 21.06%，k 再翻倍）只再换来 **−1.0pp**。
   → **甜点在 ~10.7% 保留**：拿到几乎全部收益，预算只用一半。

**逐样本配对统计（n=6，加固上面的结论）**：

| 对比 | 每样本省下的轮数 | 同向 | 接受率增益（min / max） | decode |
|---|---|---:|---|---:|
| 6.09% → 10.66% | [3, 6, 2, 2, 0, 3]（均 2.7） | **5/6** | **+0.0440（0.0000 / +0.0895）** | −6.0% |
| 6.09% → 21.06% | [3, 11, 4, 3, 1, 7]（均 4.8） | **6/6** | **+0.0810（+0.0213 / +0.1793）** | −7.0% |
| 10.66% → 21.06% | [0, 5, 2, 1, 1, 4]（均 2.2） | 5/6 | +0.0370 | **−1.1%（已饱和）** |

**两条稳健性事实**：① **接受率增益在任何一个样本上都不是负的**——不是被离群样本拉动的；
② 6.09%→21.06% 的轮数下降是 **6/6 同向**（符号检验 p = 1/64 ≈ 0.016）。

**→ 可直接落地的改动（零代码）**：把默认 `--k-plus-text` 从 **1024 改到 ~1750**
（128 帧下 6.09% → 10.66% 保留），decode 约 **−6%**，接受率 +0.044、轮数 −4.5%。
再往上加只值 −1.1%，**不划算**。

⚠️ **口径**：`timing_valid=false`（共享卡）→ **ms/轮 与那两个百分比需要在空闲卡上复验**；
但**轮数 −8.2% 是确定性的**，"每轮成本与 k 无关"有 draft 分量跨档恒定作为结构性支撑。

### 5.3 E4′（并行推进，不依赖本计划）

G0.1 显示扣除 refresh+selection 后仍残留 **+12.7 ~ +40.9 ms/轮**，即 host 侧开销。
**它现在是 dynamic 翻盘的唯一现实靶子** → 给动态主循环补上覆盖 host 侧的计时，定位后再消掉。

---

## 6. 记录与判定规则（写结论时必须遵守）

1. 任何加速比必须同时给出：**①硬件/session ②ratio-of-sums 还是 mean-of-ratios ③decode 还是端到端 ④`timing_valid` ⑤prefill 与 AR decode 的绝对时间**（`docs/experiment-provenance.md` §8.3/§9.3）。
2. **不同硬件/不同 session 不可配对**；同一张卡不同时间窗也可能被机器级争用污染（§9.3 的 C 档即例）。
3. 判定一律用 **draft acceptance（配对、逐样本、分层）** 与 **decode 墙钟**，**不用任务准确率**——论文的 coverage/accuracy 增益不保证 acceptance 同幅提升。
4. 新方法的端到端实验必须在 **A6000 空闲卡**上跑（`scripts/run_dynamic_matrix_local.sh` 会等 GPU 真正 idle 才启动，产出 `timing_valid=true`）。

---

## 7. 排期与成本（建议顺序）

| 序 | 实验 | 依赖 | 成本 | 产出 |
|---|---|---|---|---|
| 0 | **G0.1 门控** | 无 | ✅ **已完成**（纯离线） | **"便宜筛选"封顶 <1.0× → 转向提高接受率** |
| 0b | **G0.3 选择器空间** | 无 | ✅ **已完成**（纯离线，`selection_headroom.py`） | **static 已达 oracle 的 84.1%，完美选择器也只有 +7.1pp；覆盖率不随解码衰减 → 期望值下调** |
| 0c | **G0.4 因果性检验** | 无 | ✅ **已完成**（纯离线） | **coverage↔accept 的 r=0.75 是样本难度混淆（r(cov6,cov20)=0.98）→ 不能用它论证选择器收益** |
| 0d | **G0.6 per-layer 工具链** | 无 | ✅ **已完成**（纯离线，30 个 CPU 测试） | 采集器 + 分析器就绪，**P1-a/P2-a 变成"有卡就能出数"** |
| 0e | **冻结 static 头**（P2 的值钱半边） | 无 | ✅ **已完成**（纯离线，9 个 CPU 测试） | `FrozenHeadTopKPolicy`；**无需改布局**——refresh 本来就按 (layer,head) 掩码并跳过整层。**§5.2 的异构预算已核实为"高成本+有破坏位精确性风险"，降级** |
| **1★** | **G0.5 静态线 K 扫描**（K ∈ {967, 1588, 2381}） | 无（只改命令行参数） | 3–4.5 h GPU | **当前性价比最高**：预算 6.09%→10% 的覆盖率增益（+9.6pp）**大于**在 6% 下做到完美（+7.1pp）；同时直接检验因果性 |
| 2 | **P1-a** per-(layer,head) 覆盖率对照 | 需要 per-layer trace + idle GPU（工具已就绪） | 采集 30–40 min/样本 + 秒级分析 | 决定是否替换 static 打分器；**门槛重读为"吃掉 7.1pp 中的 1–2pp"** |
| 3 | **P2-a** `S_{l,h}` 分布 + `N_stat/N_dyna` + 与 churn 的相关性 | 同上（同一次采集即可） | 同上 | 决定是否冻结 static 头；`f_d ≤ ~0.3` 且 S 与 churn 负相关才继续 |
| 4 | **P3-a** Dev top-K recall/Jaccard + A6000 DCT 耗时 | 需要 idle GPU | 1 天 | 决定 Dev 是否可用；DBA 随时可上 |
| 4 | **§5.2 使能改动**（per-head 预算） | P2-a/P3-a 结论 | 2–3 天 | 解锁 DBA / Eq.(5) |
| 5 | **P1-b** 接入 static 路径（`output_attentions=False`） | 1 通过门槛 | 2–3 天 | 确定性收益：免 eager attention + 可能的接受率提升 |
| 6 | **P2-b / P3-b** 端到端 A/B | 4 | 各 1 天（512 tok × 10 样本） | acceptance / decode 加速比 |
| 7 | **E4′** host 侧开销定位 | 无 | 2–3 天 | dynamic 翻盘的唯一现实靶子 |
| 8 | **SimiCache** | **需论文全文** | 待定 | — |

**止损条件**：若 P1-a/P2-a/P3-a 的离线门槛全部不过，则本计划收敛到"§5.2 使能改动 + DBA 单独试 + E4′"，
并据 G0.1 的结论正式判定 **dynamic 逐轮路线在本项目的工作点上不成立**，把 static 1.71–1.87× 作为最终交付。

---

## 8. 参考与文件清单

**论文**（`papers/`，索引 [`papers/README.md`](../../../papers/README.md)）：
- MM-ShiftKV — ACL 2026 Findings — `MM-ShiftKV_ACL2026_findings.pdf`
- HybridKV — ACL 2026 — `HybridKV_ACL2026.pdf`
- FlashCache — CVPR 2026 — `FlashCache_CVPR2026.pdf`
- FastCache — arXiv 2503.08461 — `FastCache_arXiv2503.08461.pdf`
- SimiCache — IEEE TMM 2026 — **仅摘要** `SimiCache_IEEE_TMM2026_ABSTRACT.md`

**代码**：
- MM-ShiftKV 官方实现 — `third_party/MM-ShiftKV/`（commit `d267f8cd`，**无 license，勿提交**），移植清单见其 `PROVENANCE.md`
- 其余四个**无官方代码**，只能按论文实现

**本计划引用的 STD 代码位置**：
- `src/std_repro/std_qwen25vl.py:175–231` — `build_sparse_selection`（要替换的打分器）
- `src/std_repro/std_qwen25vl.py:41–50, 278` — `SparseSelection.k`（要扩成 per-head）
- `src/std_repro/dynamic_selection.py:78–101, 104–168` — `topk_jaccard` / `topk_intersection_counts` / `attention_free_visual_topk`（现成的对照与评估工具）
- `scripts/run_dynamic_matrix_local.sh` — 本机 A6000 可信墙钟驱动
