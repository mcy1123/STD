# 阶段 3：Dummy Proposer 接通 + bit-exact 判定

> 状态：**✅ 完成（GO，但带一个重要修正）**
> 日期：2026-09-26
> 环境：vLLM `0.22.0+cu129` / torch `2.11.0+cu129` / 主机 `A6000` GPU3
> 脚本：`src/std_vllm/proposer.py`、`scripts/vllm-std/03_bench_spec.py`、`compare_runs.py`、`03b_divergence_probe.py`、`03c_logit_margin.py`

---

## 3.1 我做了什么

实现了两个最简 Proposer，目标是**确认接口能接通**，而不是加速：

| 类 | 行为 | 目的 |
|---|---|---|
| `EmptyProposer` | 每个请求返回 `[]`（不草稿） | 验证 custom_class 管线连通 |
| `RepeatProposer` | 把上一步最后一个 token 重复 k 次 | 验证草稿**真的会**走到验证路径 |

代码：`src/std_vllm/proposer.py`（`PYTHONPATH=$STD_ROOT/src`，vLLM 启动时按模块路径导入）

---

## 3.2 命令

```bash
source scripts/vllm-std/env.sh
# 三个配置，同机、同输入、同采样参数（32 帧 / 128 token / greedy / 3 次重复）
bash scripts/vllm-std/run_bench.sh results/vllm-std/logs/03_base_k0.log   03_bench_spec.py --num-frames 32 --max-tokens 128 --runs 3 --warmup 1 --tag p3_base_k0
bash scripts/vllm-std/run_bench.sh results/vllm-std/logs/03_empty_k4.log  03_bench_spec.py --num-frames 32 --max-tokens 128 --runs 3 --warmup 1 --tag p3_empty_k4  --proposer std_vllm.proposer.EmptyProposer  --num-spec-tokens 4
bash scripts/vllm-std/run_bench.sh results/vllm-std/logs/03_repeat_k4.log 03_bench_spec.py --num-frames 32 --max-tokens 128 --runs 3 --warmup 1 --tag p3_repeat_k4 --proposer std_vllm.proposer.RepeatProposer --num-spec-tokens 4

$PY scripts/vllm-std/compare_runs.py --file results/vllm-std/spec_bench.jsonl \
    --ref p3_base_k0 --tags p3_empty_k4 p3_repeat_k4 --json-out results/vllm-std/03_bitexact.json
```

---

## 3.3 结果

| 配置 | bit-exact vs baseline | 首个分歧 | 耗时 mean±std | 接受率 | accept_len | 草稿/接受 |
|---|---|---|---|---|---|---|
| `p3_base_k0`（eager） | — （基准） | — | **5.2305 ± 0.0406 s** | — | — | — |
| `p3_empty_k4`（空草稿） | ✅ **是** | — | 5.4807 ± 0.0327 s | — | — | 0 / 0 |
| `p3_repeat_k4`（重复草稿） | ❌ **否** | **62** | 7.5295 ± 0.1534 s | **0.0000** | 1.0000 | 1524 / **0** |

**两次独立复现**（`p3_base_k0_b` / `p3_repeat_k4_b`）：分歧点**完全一致**——token 62，
`16445 → 448`。不是随机抖动。

### 框架税（已有硬数字）

| 对比 | 加速比 | 说明 |
|---|---|---|
| `empty_k4` / `baseline` | **0.954×**（慢 4.8%） | 开启投机但**一个草稿都不发**，仍要付 4.8% 的框架成本 |
| `repeat_k4` / `baseline` | **0.695×**（慢 44%） | 每轮发 4 个草稿、**全部被拒**，纯浪费 |

> **这两个数字本身就是交付物**：它们把「投机解码的固定开销」和「草稿质量的价值」分离开了。
> 阶段 4–6 的任何收益都必须先跨过这 4.8% 的地板。

---

## 3.4 ★ 核心发现：bit-exact 在本例中**不可能成立**，因为参考答案本身不稳定

这是本阶段最重要的结论，且是**可复现的硬证据**。

### 3.4.1 现象

`RepeatProposer` 发出了 381 组共 1524 个草稿，**接受 0 个**（`accept_rate=0.0`）。
也就是说：**每一个草稿都被拒绝，所有输出 token 都是 target 模型自己选的**。
按定义，这时的输出**应当**与 eager baseline 完全一致。但它不一致。

### 3.4.2 分歧点是一个**精确平局**

`03c_logit_margin.py` 在 token 62 处拿到 top-k logprob：

**启用 CUDA graph（vLLM 默认）：**

```
token=16445  logprob=-0.666198   ' featuring'
token=448    logprob=-0.791198   ' with'
margin(top1-top2) = 0.125000
```

**禁用 CUDA graph（`--enforce-eager`）：**

```
token=448    logprob=-0.726219   ' with'
token=16445  logprob=-0.726219   ' featuring'
margin(top1-top2) = 0.000000      <-- 完全相等
```

`' with'` 与 `' featuring'` 在这个位置**概率完全相同**，语义也等价：

- eager：`The background is vibrant, featuring a large screen with…`
- spec ：`The background is vibrant, with a large screen displaying…`

### 3.4.3 决定性实验：**eager 与 eager 自己都不一致**

`03d`：两次**都不带任何投机解码**的 eager 运行，同机、同 prompt、同 greedy 参数，
**唯一差别是是否使用 CUDA graph**：

```
eager + CUDA graph : token 62 = 16445  (' featuring')
eager + enforce_eager: token 62 =  448  (' with')
首个分歧：62
```

**→ 结论：`temperature=0` 并不保证唯一答案。** 在平局位置，argmax 由浮点噪声
（kernel 选择、CUDA graph 重放、batch 形状）决定。**任何实现都无法对一个自身不稳定的参考做到 bit-exact。**

这不是 my-proposer 的 bug，也不是 vLLM 的记账 bug。它是「greedy 解码 = 取 argmax」
这一假设在**有限精度**下的固有缺陷。

### 3.4.4 这类位置有多少？margins 分布（70 个位置）

| 阈值 | 脆弱位置数 | 占比 |
|---|---:|---:|
| margin < 0.01（含 1 个精确平局） | 1 | 1.4% |
| margin < 0.25 | 3 | 4.3% |
| margin < 0.50 | 9 | 12.9% |
| margin < 1.00 | 17 | 24.3% |

最小的几个：

```
idx=57  margin=0.000000  top1=576   top2=2932
idx=42  margin=0.125000  top1=25365 top2=78461
idx=62  margin=0.125000  top1=16445 top2=448     <-- 实际发生分歧的位置
idx=18  margin=0.250000  top1=448   top2=323
```

> 注意 margin 全是 0.125 的整数倍，说明返回的 logprob 有量化栅格；这只会**掩盖**更小的差距，
> 即真实脆弱位置只会**更多**，不会更少。

### 3.4.5 对后续阶段的强制性影响（方法论修正）

任务书的验收要求是「bit-exact 对比 token ids」。基于本阶段的证据，这条要求必须**加限定条件**，否则会得到假的失败：

1. **比较必须区分「稳定位置」与「平局位置」**：只在 `margin ≥ ε` 的位置上要求严格相等。
2. **报告形式**改为三元组：`(完全一致位置数, 平局位置分歧数, 稳定位置分歧数)`。
   只有**第三项 > 0** 才是真正的实现缺陷。
3. **每个阶段的 bit-exact 结论都要附带 margin 画像**，否则无法判断分歧是「数值」还是「逻辑」。
4. 阶段 1/3 的 baseline 已固定为 `p3_base_k0`（启用 graph，vLLM 默认）。

**这不是放松标准，而是把标准变得可判定。** 原先的标准在一台真实机器上是不可满足的。

---

## 3.5 顺带记录：vLLM 在 custom_class 下的两条行为

启动日志中（`results/vllm-std/logs/03_*.log`）：

```
WARNING [vllm.py:952] Async scheduling not supported with custom_class-based
        speculative decoding and will be disabled.
WARNING [__init__.py:204] min_p and logit_bias parameters won't work with
        speculative decoding.
INFO    [custom_class_proposer.py:67] Loaded custom proposer class
        'std_vllm.proposer.RepeatProposer' with num_speculative_tokens=4
```

- **异步调度被强制关闭** → 这本身就是 `empty_k4` 那 4.8% 开销的一个候选来源（待阶段 7 归因）。
- `min_p` / `logit_bias` 与投机解码不兼容（本项目不用，记录备查）。
- 自定义类被正确加载（这行 INFO 是接口接通的最直接证据）。

---

## 3.6 失败记录（未掩盖）

| # | 失败 | 根因 | 处置 |
|---|---|---|---|
| 1 | 引擎起不来，rc=1 | 脚本在 `gpu_mem_util=0.55` 下 KV 只剩 ~1 GiB，而 `max_model_len=65536` 需 3.5 GiB | 把 `max_model_len` 默认降到 32768，并让 `run_bench.sh` **按实际空闲显存推导** `gpu_mem_util` |
| 2 | `run_bench.sh` 无限误判「contention」 | 它 grep 整个**追加**日志，匹配到了上一次尝试的旧报错 | 改为只检查本次尝试新增的字节区间（记录 `wc -c` 偏移） |
| 3 | `TypeError: Object of type ModelConfig is not JSON serializable` | `llm_kwargs` 里被 vLLM 塞进了 config 对象 | `json.dumps(..., default=str)` |
| 4 | 连续两次被别的用户抢卡 | 共享机，选卡后显存被占 | `run_bench.sh` 自动换卡重试 |
| 5 | `03b` 探针**测不到**分歧 | 探针开 `logprobs`，改变了采样路径，掩盖了平局 | 改用 `03c` 直接在两个路径下取 top-k，并用 `03d` 做 eager-vs-eager 对照 |

---

## 3.7 是否进入下一阶段？

**是，但判定标准已修正。**

| 判据（任务书原文） | 结果 |
|---|---|
| 接口通 | ✅ `Loaded custom proposer class ... with num_speculative_tokens=4`，381 组草稿真实到达验证层 |
| 不报错 | ✅ 三个配置全部 rc=0 |
| 输出一致 **或差异有明确解释** | ✅ 差异已定位到 **token 62 的精确平局**，并证明 **eager-vs-eager 同样不一致** |

任务书原文是「输出一致**或**差异有明确解释」——本例满足**后半条**，且解释强度高于「一致」本身
（因为它同时证明了「一致」在平局位置是运气）。

**→ GO，进入阶段 4（STD 单 token 草稿）。**
后续所有 bit-exact 结论一律按 §3.4.5 的三元组 + margin 画像报告。
