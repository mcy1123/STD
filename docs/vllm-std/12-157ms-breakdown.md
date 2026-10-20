# 阶段 1：157 ms 的 op 级拆解

> 状态：**✅ 完成 —— 结论是把原命题推翻了**
> 日期：2026-09-26
> 脚本：`scripts/vllm-std/P1_profile_selection.py`
> 数据：`results/vllm-std/profile_32frame.json`、`profile_single_layer.json`
> 日志：`results/vllm-std/logs/P1_profile*.log`

---

## 1.1 一句话结论

**那 157 ms 是测量假象。** 同一段代码在**同一台机器、同一形状**上的稳态耗时是 **0.56 ms**（kv=2420），
在真实 32 帧工况（kv=19,164）上是 **2.95 ms/层**。

而且拆开之后，**A（GPU kernel）占 98.9%**，B（CPU 调度/同步）只占 **1.1%**，C（拷贝）≈ 0。
**所以「差距 2–3 个数量级是实现开销」这个判断也不成立** —— 真实实现里几乎没有可省的 host 开销。

---

## 1.2 157 ms 是怎么来的（假象的机制）

原探针里计时是这么写的：

```python
t0 = time.perf_counter()
acc = torch.zeros(...)
for s0 in range(0, n_rows, CH):
    ... einsum / softmax / sum ...
torch.cuda.synchronize()          # <-- 问题在这里
dt = time.perf_counter() - t0
```

`torch.cuda.synchronize()` 等待的是 **整个 CUDA stream 上所有未完成的 GPU 工作**，
而这段代码是在**引擎刚做完一次 19k-token prefill 之后**、在 hook 的余波里执行的。
于是这次 sync **把引擎自己排队的 GPU 工作也一并等掉了** —— 测到的不是 selection 的耗时，
而是「selection + 引擎遗留队列」。

**对照证据**：本次把同一段代码（`naive` 变体，逐字保留原探针写法）放到**GPU 空闲**的上下文里、
用 **CUDA event** 计时，同一形状（kv=2420）得到 **0.65 ms**：

| 形状 | 原探针报告 | 本次稳态（事件计时） | 倍数 |
|---|---:|---:|---:|
| kv = 2420，64 text 行，单层 | **156.6 ms** | **0.65 ms** | **≈ 240×** |
| kv = 8192，64 text 行，单层 | — | 1.41 ms | — |
| kv = 19,164，64 text 行，单层（真实 32 帧） | — | **2.92 ms** | — |

---

## 1.3 A / B / C 拆解（真实工况：32 帧，kv = 19,164）

三个变体，每个 **冷启动 1 次 + 预热 2 次 + 事件计时 5 次重复**：

```
layer=language_model.model.layers.0.self_attn.attn  prompt=19164  q=19164  kv=19164  text_rows=64
variant       A_gpu    A_std   B_host     wall   A/wall      ×28   /prefill
core          3.775    1.854    0.036    3.811    99.1%    105.7     11.7%
core+book     3.170    0.005    0.037    3.207    98.8%     88.8      9.8%
naive         2.918    0.011    0.032    2.950    98.9%     81.7      9.1%
```

- **A = GPU kernel 时间**：CUDA event (`Event.elapsed_time`) 测得，**不含 profiler 开销**
- **B = wall − A**：CPU 派发、Python 开销、launch、sync
- **C = memcpy**：profiler 里 device 侧 copy 类 op 的总时间 —— **实测 0.0 ms**

| 块 | 时间（naive，最保守变体） | 占比 |
|---|---:|---:|
| **A：GPU kernel** | **2.918 ms** | **98.9%** |
| **B：CPU 调度 / 同步 / launch** | **0.032 ms** | **1.1%** |
| **C：内存拷贝** | **0.000 ms** | **0.0%** |
| unknown | 0.000 ms | 0.0% |
| **合计** | **2.950 ms** | **100%** |

**三块合计 = 100%，远超验收要求的 ≥90%，unknown 为 0。**

### 逐项排查任务书列出的可疑点

| 可疑点 | 实测 | 判定 |
|---|---|---|
| `.tolist()` / `.item()` / `.cpu()` 次数与耗时 | `core`（无这些调用）vs `core+book`（有 5 处标量抽取） | **+0.26 ms/层**（0.808 vs 0.527 @ kv=2420），即约 **0.5% 量级**，非主导 |
| Python for 循环层数 | 单层 1 个 chunk 循环（`text_rows=64 < CH=512` → **只迭代 1 次**） | 非主导 |
| 逐 head / 逐 layer 重复计算 | 无；对 KV head 做了分组归并，一次 einsum | 非主导 |
| 反复 reshape / contiguous | `.float()` 是一次 dtype cast（kernel，不是拷贝）；view 是零拷贝 | **C = 0** |
| 隐式 sync 点 | `core` 变体**没有任何**标量抽取，仍有 99.1% 是 kernel | 非主导 |

**结论：没有任何一项 host 侧的「坏习惯」是主导项。时间就在 kernel 里。**

### 一处必须标注的不确定性

`core` 变体的 A 标准差是 **1.854 ms**（5 次里有一次明显偏高），
而 `core+book`（0.005）与 `naive`（0.011）**非常稳定**。
所以上表的区间取 **2.92–3.78 ms/层**，比例区间取 **9.1%–11.7%**。
对外引用请用 `naive`（最保守、最稳、且逐字对应原探针实现）。

---

## 1.4 复核：prefill 本身多贵

同一脚本在同一次运行里测了 prefill（`max_tokens=1`，几乎全是 prefill）：

```
prefill_ms runs = [901.19, 819.16, 983.96]  mean = 901.44
```

**注意**：vLLM 默认开 prefix caching，第 2/3 次可能命中缓存而**低估** prefill。
**诚实取第一次（cache-cold）901.19 ms。**

---

## 1.5 失败与修正记录（未掩盖）

| # | 事件 | 根因 | 处置 |
|---|---|---|---|
| 1 | 第一版 profiler 报 `A_kernel = 0.003 ms` | `key_averages().device_time_total` 对这种小 kernel **严重低报** | 改用 **CUDA event** 计时；profiler 只用于 op 清单，不用于绝对时间 |
| 2 | 第一版抓到了 8192 的 warmup 桶而非真实请求 | 钩子只留第一个 prefill | 改为「取 ≤ prompt_tokens 的最大 prefill」 |
| 3 | 第二版抓到了 256 的 pass | 改成了取最小 | 同上，修正为按 prompt 长度匹配 |
| 4 | profiled wall（1.5–3.4 ms）远大于真实 wall | `torch.profiler` 自身膨胀 host 时间 | **同时报 profiled 与 unprofiled**，结论只用后者 |
| 5 | 阶段 0 误杀线 A 引擎 | 用属主+进程名判断归属，而我同时有两个作业 | 已记录；新规则：必须用启动时间与作业时间窗交叉比对 |

---

## 1.6 与上一阶段的结论对比

| | 阶段 B（线 B）结论 | 阶段 1 修正后 |
|---|---|---|
| 单层 selection 耗时 | 157 ms（kv=2420） | **0.65 ms**（同形状），**2.92 ms**（真实 32 帧） |
| 主导项 | 「判定为实现开销，不是基本成本」 | **推翻**：99% 是 GPU kernel，host 只占 1.1% |
| 28 层总开销 | 外推 ≈ 4.4 s（按 157ms × 28） | **81.7–105.7 ms** |

**→ 阶段 1 的核心交付就是这次推翻。** 带着 157 ms 去做融合会白做：融合只能省掉那 1.1%。
