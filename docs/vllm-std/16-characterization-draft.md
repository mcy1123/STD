# 线 C：characterization 短文（草稿）

> 状态：**草稿（不占卡，不受 P0 阻塞）**
> 日期：2026-09-26
> 边界声明：**本文只刻画"视频 LLM 投机解码为什么在本环境下不划算"这一现象，
> 不回答"STD 在 vLLM 上是否有效"。后者需要一个接受率 > 0 的配置，目前不存在。**
> 前置：`docs/vllm-std/10-negative-result-skeleton.md`（含对三条误引材料的逐条纠正）

---

## C.0 一句话

在单张 A6000 的真实 vLLM 服务栈上系统测量视频 LLM 的投机解码，
给出**四个可复现的负面/方法论结果**；其中最强的一条是
**「greedy 解码的 bit-exact 判据本身不成立」**——因为参考答案会与自己分歧。

---

## C.1 四条已核实材料（每条可追溯到实验与日志）

### 材料 ①：`temperature=0` 下参考答案不稳定 ★ 最强

| 项 | 内容 |
|---|---|
| 实验 | `scripts/vllm-std/03b_divergence_probe.py`、`03c_logit_margin.py`、`03d_*` |
| 数据 | `results/vllm-std/03d_eager_graph.json`、`03d_eager_nograph.json`、`03c_logit_margin.json` |
| 文档 | `docs/vllm-std/03-dummy-proposer.md` §3.4 |

- 两次**都不含投机解码**的 eager 运行，唯一差别是 **CUDA graph**，在 **token 62** 分歧：`16445` vs `448`。
- 该位置 top-1/top-2 logprob 在 `--enforce-eager` 下**精确相等**：`' with'` 与 `' featuring'` 同为 **-0.726219**（margin `0.000000`）。
- 三次独立复现，分歧点与 token 对完全一致。
- margins 画像（70 个位置）：`margin<0.25` 有 **3 个（4.3%）**，`margin<0.5` 有 **9 个（12.9%）**，`margin<1.0` 有 **17 个（24.3%）**。

**论点**：*bit-exactness 不是 greedy 解码的良定义验收判据。*
我们给出一个可复现的反例，并量化这类平局位置的密度。
**这条与具体系统无关，最易复现、最易被接受。**

### 材料 ②：投机路径存在可测的正框架税

| 项 | 内容 |
|---|---|
| 实验 | `scripts/vllm-std/03_bench_spec.py` + `run_bench.sh` |
| 数据 | `results/vllm-std/spec_bench.jsonl`（tag `p3_empty_k4`、`p3_repeat_k4`） |
| 文档 | `docs/vllm-std/03-dummy-proposer.md` §3.3 |

| 配置 | vs eager |
|---|---:|
| 开投机但**零草稿** | **0.954×**（慢 4.8%） |
| 草稿**全被拒** | **0.695×**（慢 30.5%） |

机制线索（日志原文）：`Async scheduling not supported with custom_class-based speculative
decoding and will be disabled.` —— 给出一个**可检验的假设**：框架税部分来自异步调度被关闭。

### 材料 ③：接口能力 ≠ 算法需求（架构性结果）

| 项 | 内容 |
|---|---|
| 实验 | 读源码 + `scripts/vllm-std/B_probe_attn_weights.py` |
| 数据 | `results/vllm-std/B_attn_weight_probe.json` |
| 文档 | `docs/vllm-std/04-std-1token.md` §4.2、`09-lineB-attention-probe.md` |

- vLLM 0.22.0 的 `custom_class` proposer：构造只收 `VllmConfig`，`propose` 只收 CPU token id，
  **拿不到 model / KV cache / attention** → 稀疏-KV 类算法（STD）**不可表达**。
- 但同一版本的 `Attention.forward`（`attention.py:437`）是 **Python 方法**，
  纯钩子即可拿到 prefill Q/K/V；重建的 `[nkv, prompt_len]` 统计量与
  `build_sparse_selection` 一致，**质量校验精确通过**（64 text 行 × 7 组头 × 4 KV 头 = **1792** = 实测 `acc.sum()`）。
- 前提 3 条：进程内执行、关 chunked prefill、钩子 **trace-safe**
  （torch.compile 会 trace 穿过它——由一次 `.tolist()` 崩溃**反向证明**）。

**论点**：*看起来通用的「自定义 proposer」抽象，仍可能在结构上装不下某类算法；
而抽象层缺失的能力，往下一层往往可达。*

### 材料 ④：解码成本是权重读取，不是 DRAM 带宽 ★ 来自仓库既有实测

| 项 | 内容 |
|---|---|
| 来源 | `PROGRESS.md` §19.8、`docs/w1-decode-knobs.md` |
| 原文 | `PROGRESS.md:743-744` |

- 一次搬运 **57 MB / 71 ms → 有效带宽仅 0.8 GB/s**，比 A100 峰值（~2000 GB/s）**低约 2500×**。
- 原文结论：**「瓶颈不是带宽，而是 224 次 `torch.unique` + 56 次 gather/copy 的碎片化发射 + 2 次强制 `cuda.synchronize()`」**。
- 三个「不动权重字节」的开关**全部无效** → 证明 decode 步就是权重读取本身。

> ⚠️ **更正**：任务书描述这条时为「DRAM 带宽上限（92% 峰值）」。核查后
> **「92% 峰值」在仓库中不存在**，且定性方向**相反**。
> 写成论文时这是一个**更强**的结论，因为它排除了一个显而易见的解释。

---

## C.2 方法论贡献（最可迁移，建议单独成节）

### ① 配对 A/B 计时协议（共享机必须）

共享机上的单点计时会制造**双向**假象，我们都实测到了：

| 假象 | 实测值 | 真值 | 成因 |
|---|---:|---:|---|
| **假加速** | `1.260×` | `0.90–0.94×` | 那一次跑在**独占空卡**上 |
| **假减速** | `0.9227×` | **`0.9718×`** | 共享卡争用**放大** CPU 侧 proposer 的劣势 |

出处：`docs/vllm-std/05-std-multitoken.md` §5.3、`07-attribution.md` §7.4/§7.9、
`results/vllm-std/A1_paired_exclusive.json`、`07_paired.json`。

**做法**：交替 A,B,A,B 运行，使时间漂移成为**共模误差**，在配对差值中抵消。

### ② bit-exact 三元组判据

报 `(完全一致位置数, 平局位置分歧数, 稳定位置分歧数)` + margin 画像。
**只有第三项 > 0 才是真正的实现缺陷。**
出处：`docs/vllm-std/03-dummy-proposer.md` §3.4.5。

### ③ 反例优先

**证明参考不稳定，比证明自己与参考一致更有价值** —— 因为后者在平局位置是运气。

### ④ 测量假象的识别（P0 新增）

原探针报告单层 selection **157 ms**；在 GPU 空闲上下文用 CUDA event 复测同形状为 **0.65 ms**
（**≈240×**）。根因：`torch.cuda.synchronize()` 把**引擎遗留的 GPU 队列**一并等掉了。

| 形状 | 原报告 | 复测（事件计时） |
|---|---:|---:|
| kv=2420，64 text 行，单层 | 156.6 ms | **0.65 ms** |
| kv=19,164，64 text 行，单层 | — | **2.92 ms** |

出处：`docs/vllm-std/12-157ms-breakdown.md`、`results/vllm-std/profile_32frame.json`。

**教训**：在**引擎内部**用 `perf_counter + cuda.synchronize()` 给一小段算子计时，
会把引擎自身的排队时间算进来。**必须用 CUDA event，且确认 GPU 为空闲**。

---

## C.3 骨架

**暂定标题**：
*When Speculative Decoding Does Not Pay: Bit-Exactness, Framework Tax, and Interface Limits
in Video-LLM Serving*

| 节 | 内容 | 证据 |
|---|---|---|
| 1 引言 | 「投机解码能加速视频 LLM」被默认成立；本文给出定量反例 | — |
| 2 方法 | 配对 A/B 协议；bit-exact 三元组；引擎内计时的陷阱 | C.2 |
| 3 bit-exact 不成立 | token 62 精确平局；eager-vs-eager 对照；margins 画像 | 材料 ① |
| 4 框架税 | 零草稿 0.954×；全拒 0.695×；异步调度被关的日志 | 材料 ② |
| 5 接口边界 | custom_class 够不到 model/KV/attention；低一层可达 + 3 前提 | 材料 ③ |
| 6 成本归属 | decode 是权重读取非带宽；0.8 GB/s vs 2000 GB/s；量化 draft 是唯一不牺牲 exactness 的路 | 材料 ④ |
| 7 测量方法论 | 157ms→0.65ms 的 240× 假象；共享机双向偏差 | C.2 ④① |
| 8 局限 | 单机单框架；接受率恒 0 的任务性质；未做端到端 | C.4 |

---

## C.4 写之前仍缺什么（诚实清单）

| 缺口 | 状态 |
|---|---|
| 独占卡 batch b1–b8 曲线 | ⏳ 本阶段正在跑（`lineA_batch_exclusive.sh`，GPU2 独占） |
| ≥3 对独占卡配对 | ⚠️ 只有 2 对（`0.9718 ± 0.0096×`） |
| 28 层 selection 实测（现在是外推） | ❌ 未测 |
| **一个接受率 > 0 的配置** | ❌ 全部为 0 |
| 跨框架对照 | ❌ **不存在**（唯一 sglang 数字 1.45× 是外推预测，非实测，**不得当实测引用**） |

**结论**：材料 ①②③④ 足以支撑一篇 **characterization / negative-result 短文**。
**不足以**支撑「STD 在 vLLM 上有效/无效」——那是另一篇，且需要 §C.4 最后两行。

---

## C.5 引用规范（写作时强制）

1. **只用已核实数字。** 每条结论必须能指到具体 `results/vllm-std/` 文件与 tag。
2. **禁止把外推当实测。** 特别是 `sglang 1.45×`（`PROGRESS.md:1736` 是预测）与
   `A × 28`（只测了 layer 0）。
3. **禁止把本文与「STD 是否有效」混写。**

---

## C.6 【2026-09-26 补】「判据错位」段落的定稿口径（分支 A）

曾有一个中间结论（`14-acceptance-probe.md` §3′.3）主张
「STD 的全 query 求和判据与下一个 token 错位 → 接受率接近 0」。
**该结论已被更强的实测证据证伪，不得写入论文。** 过程与证据：

| 步骤 | 内容 |
|---|---|
| 代理指标 | 探针测「最后一行注意力质量保真度」，K=256 时 STD 判据只有 24.6% |
| 当时的**推测** | 接受率接近 0 |
| **直接观测量** | §17 静态 STD（**同一判据，逐字相同**）实测接受率 **0.507–1.000，均值 ≈0.81** |
| 判定 | **推测被否证**；`docs/vllm-std/18-a1-decision.md` 判分支 A |

**代码证据**（判据逐字相同）：`src/std_repro/std_qwen25vl.py:239`
（`grouped.sum(dim=(1,2))`，`grouped` 形状 `[nkv, heads_per_kv, text_len, prompt_len]`），
调用点 `:1224`（`std_generate_qwen25vl`，def `:1118`），
产出者 `scripts/benchmark_a100_dynamic.py:330`（`method == "static"` 臂）。

### 这段该写成什么（**方法论**结论，而不是 STD 结论）

**这是一个"代理指标失效"的案例，本身就是材料 ①/② 的同族：**

> 当一个系统级指标（接受率）可以被直接测量时，不要用低层代理指标
> （注意力质量保真度）去预测它 —— 我们构造的代理在 K=256 时报 24.6%，
> 而直接测量的接受率是 0.507–1.000。**代理与目标之间没有我们假设的映射。**

这条与本文 §C.2 的三个方法论贡献**同族**（都是「测量怎么做才对」）：
配对计时协议、bit-exact 三元组、**以及：代理指标必须用直接观测量校准**。

### 必须避免的写法

- ❌「STD 的选择判据有缺陷」—— 已被证伪。
- ❌「稀疏选择导致接受率为 0」—— 未知；接受率为 0 的是**token 级 n-gram 类比**，不是稀疏前向。
- ✅「我们构造的代理指标与真实接受率不符，因此它不能作为判据优劣的依据」。
