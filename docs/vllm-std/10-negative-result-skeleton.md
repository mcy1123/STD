# 线 C：负面刻画（骨架）

> 日期：2026-09-26
> 说明：先按用户要求成文，**但其中两条被引用的「硬料」经核查与仓库实际内容不符**，见 §C.1。
> 本文只使用**已核实**的材料。

---

## C.1 ⚠️ 先纠正三条「硬料」中的两条

用户提出的三块料，逐条核对 `PROGRESS.md` / `docs/` 原文：

| # | 用户表述 | 仓库实际 | 判定 |
|---|---|---|---|
| 1 | 「batch=1 的 DRAM 带宽上限（**92% 峰值**，**双流 1.004×**）」 | 仓库里**搜不到**「92% 峰值」或「双流 1.004×」。相关的 1.000× 出现在两处不相干的地方：`PROGRESS.md:318`（static 基线行）、`:1602`（kernel 开关对比 `stock_repeat_mask`） | ❌ **数字不存在** |
| 2 | 「DRAM 带宽上限」这个**定性结论** | **仓库结论恰好相反**。`PROGRESS.md:743-744` 原文：<br>「一次搬运 **57 MB / 71 ms → 有效带宽仅 0.8 GB/s**，比 A100 的 ~2000 GB/s **低约 2500 倍**。」<br>「→ **瓶颈不是带宽，而是 224 次 `torch.unique` + 56 次 gather/copy 的碎片化发射 + 2 次强制 `cuda.synchronize()`**。」 | ❌ **方向相反**：不是带宽受限，是**发射碎片化 + 强制同步** |
| 3 | 「SGLang vs vLLM 的框架税对照（0.9227× **跨框架自洽**）」 | `0.9227×` 是**本次 vLLM 单框架**实测（`results/vllm-std/07_paired.json`）。全仓库唯一的 sglang 加速数字是 **1.45×**，且它是 `PROGRESS.md:1736` 的**外推预测（「按一阶模型 TP 分片后 c ≈ 1.95 → ≈1.45×」），不是实测** | ❌ **不存在跨框架自洽** |

**处置**：不把这三条写进骨架。下面只用**已核实**的材料，其中 3 条来自本次工作（有原始日志），
1 条来自仓库既有实测（§19）。

> 这不是抠字眼：如果按原表述成文，第 1、3 条会被审稿人直接判为数据不存在，
> 第 2 条会被判为结论反向。**先修正，再写。**

---

## C.2 可用于成文的核实材料（4 条）

### 材料 ①：参考答案在 `temperature=0` 下不稳定 ★ 最强

**来源**：本次实测，`docs/vllm-std/03-dummy-proposer.md` §3.4、`results/vllm-std/03d_*.json`

- `RepeatProposer` 发 381 组 / 1524 个草稿、**接受 0 个**，输出仍与 eager 分歧，两次独立复现分歧点**完全一致**（token 62：`16445 ↔ 448`）。
- 该处 logprob **精确相等**：`' with'` 与 `' featuring'` 在 `--enforce-eager` 下同为 **-0.726219**（margin 0.000000）。
- **决定性对照**：两次**都不含投机解码**的 eager 运行，唯一差别是 CUDA graph，**同样在 token 62 分歧**。
- margins 画像（70 位置）：margin<0.25 有 3 个（4.3%），<0.5 有 9 个（12.9%）。

**可写成的论点**：*"Bit-exactness is not a well-posed acceptance criterion for greedy decoding:
we show a reproducible case where the reference implementation disagrees with itself, and
quantify the density of such tie positions."*
这是一个**与具体系统无关的方法论结论**，最容易被接受，也最容易被复现。

### 材料 ②：投机解码的固定框架税为正、且可测

**来源**：本次实测，`docs/vllm-std/03-dummy-proposer.md` §3.3

| 配置 | vs eager |
|---|---:|
| 开投机但**零草稿** | **0.954×** |
| 草稿全被拒 | 0.695× |

且 vLLM 日志明示：`Async scheduling not supported with custom_class-based speculative decoding
and will be disabled.` —— 提供了一个**可检验的机制假设**（框架税部分来自异步调度被关闭）。

### 材料 ③：接口能力 ≠ 算法需求（架构性负结果）

**来源**：本次实测，`docs/vllm-std/04-std-1token.md` §4.2、`09-lineB-attention-probe.md`

- vLLM 0.22.0 的 `custom_class` 注入点构造只给 `VllmConfig`、`propose` 只收 CPU token id，
  **拿不到 model / KV / attention** → STD 的稀疏 KV 草稿不可表达。
- 但同一版本的 `Attention.forward` 是 Python 方法，**可以**用纯钩子拿到 prefill Q/K/V，
  重建出的 `[nkv, prompt_len]` 统计量与 `build_sparse_selection` 一致，**质量校验精确通过**
  （64 text 行 × 7 组头 × 4 KV 头 = 1792 = 实测 `acc.sum()`）。
- **前提 3 条**：进程内（或 IPC）、关 chunked prefill（或累积 K）、钩子必须 trace-safe
  （torch.compile 会 trace 穿过它——这一点是被一次 `.tolist()` 崩溃反向证明的）。
- **成本未定**：朴素实现单层 157 ms（64×2420）；理论量仅 ~8.8 GFLOP/全部层，
  说明是**实现开销而非基本成本**，但必须融合化，否则净收益为负。

**可写成的论点**：*"A 'custom proposer' hook that looks general can still be structurally
incapable of hosting a given algorithm; and the capability that is missing from the abstraction
may be reachable one level down."*

### 材料 ④：解码成本是权重读取，不是 DRAM 带宽 ★ 来自仓库既有实测

**来源**：`PROGRESS.md` §19.8、`docs/w1-decode-knobs.md`

- 三个「不动权重字节」的开关**全部无效** → 证明 decode 步就是权重读取本身。
- 一次搬运 57 MB / 71 ms → **有效带宽 0.8 GB/s**，比 A100 峰值低 **~2500×**
  → 瓶颈是**碎片化发射 + 强制同步**，不是带宽。
- `c_s ≈ 1.74 ms/GiB`；由 `c_d=68.0, c_v=69.35, n=9.253` 外推：只量化 draft（不牺牲 exactness）
  是唯一能越过 `c = c_d/c_s = 2.5×` 天花板的路径。

**注意**：材料 ④ 的**方向与用户原表述相反**（不是带宽上限，是发射/同步受限）。
写成论文时这是一个**更强的**结论，因为它排除了一个显而易见的解释。

---

## C.3 骨架（只基于 §C.2）

**暂定标题**：
*When Speculative Decoding Does Not Pay: Bit-Exactness, Framework Tax, and Interface Limits in
Video-LLM Serving*

**摘要主线**：在单卡 A6000 的真实服务栈上系统测量视频 LLM 的投机解码，
给出三个**可复现的负面/方法论结果**：(i) greedy 的 bit-exact 判据不成立；
(ii) 投机路径存在可测的正框架税；(iii) 主流框架的自定义 proposer 抽象在结构上装不下
稀疏-KV 类算法，而缺失能力在低一层可达但需付实现成本。

| 节 | 内容 | 证据 |
|---|---|---|
| 1 引言 | 「投机解码能不能加速视频 LLM」这一问被默认答案为是；我们给出否的定量刻画 | — |
| 2 方法 | 配对 A/B 计时协议（对抗共享机漂移）；三元组 bit-exact 判据 | §C.4 |
| 3 bit-exact 不成立 | token 62 精确平局 + eager-vs-eager 对照 + margins 画像 | 材料 ① |
| 4 框架税 | 零草稿 0.954×；草稿全拒 0.695×；异步调度被关的日志证据 | 材料 ② |
| 5 接口边界 | custom_class 拿不到 model/KV/attention；注意力在低一层可达及其 3 个前提与成本 | 材料 ③ |
| 6 成本归属 | decode 是权重读取不是带宽；0.8 GB/s vs 2000 GB/s；量化 draft 是唯一不牺牲 exactness 的路 | 材料 ④ |
| 7 讨论/局限 | 单机单框架；接受率恒 0 的任务性质；selection 成本未融合化 | — |

---

## C.4 方法学贡献（应单独成节，且是本文最可迁移的部分）

1. **配对 A/B 协议**：共享机上单点计时会产生**双向**假象——
   我们实测到 `1.260×` 的**假加速**（那一次跑在独占空卡），
   也实测到 `0.9227×` 的**假减速**（共享卡争用放大 proposer 劣势）。
   交错配对把漂移变成共模误差。**这条对任何共享集群上的系统测量都适用。**
2. **bit-exact 三元组判据**：`(完全一致, 平局分歧, 稳定分歧)` + margin 画像。
3. **反例优先**：证明参考不稳定，比证明自己与参考一致**更有价值**——
   因为后者在平局位置是运气。

---

## C.5 写之前还缺什么（诚实清单）

| 缺口 | 状态 |
|---|---|
| 独占卡上的 batch b1–b8 曲线 | ❌ 未取得（GPU0 独占窗口被他人抢走后关闭） |
| 独占卡上 ≥3 对配对 | ⚠️ 只拿到 2 对（0.9718 ± 0.0096×） |
| selection 融合后的真实开销 | ❌ 未测（P0 的验收项） |
| 一个接受率 > 0 的配置 | ❌ 本次全部为 0 |
| 跨框架对照 | ❌ 不存在（见 §C.1 第 3 条） |

**结论**：材料 ①②③④ 足以支撑一篇**characterization / negative result** 短文；
但**不足以**支撑「STD 在 vLLM 上有效/无效」的结论——那需要 P0 的接受率与 selection 成本。
两者是**两篇文章**，不要混。
