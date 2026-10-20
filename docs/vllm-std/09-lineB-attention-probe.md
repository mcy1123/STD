# 线 B：注意力权重可获取性探针（go/no-go 判定）

> 日期：2026-09-26
> 脚本：`scripts/vllm-std/B_probe_attn_weights.py`
> 原始证据：`results/vllm-std/B_attn_weight_probe.json`、`results/vllm-std/logs/B_attn_probe*.log`

---

## B.0 判定

# ✅ **能拿到。STD 在 vLLM 上有路。启动 P0。**

但附带 **3 个硬性前提** 和 **1 个必须提前解决的实现问题**（见 §B.5）。
不是「拿到权重就完事」，但也**不是**「后端不返回权重所以死路」。

---

## B.1 探针问的是什么

`build_sparse_selection`（`src/std_repro/std_qwen25vl.py:185`）需要 prefill 注意力：

```python
# 输入 attn: [batch, query_heads, text_len, prompt_len]（softmax 之后）
grouped = attn.reshape(nkv, heads_per_kv, q_len, kv_len)
scores  = grouped.sum(dim=(1, 2))        # -> [nkv, kv_len]
top_local = torch.topk(visual_scores, k=k, dim=-1, largest=True).indices
```

阶段 4 标注的首要风险是：**vLLM 的 FLASH_ATTN / FLASHINFER 默认不 materialize 注意力权重**。

**但融合 kernel 不是唯一来源。** `Attention.forward`
（`vllm/model_executor/layers/attention/attention.py:437`）是一个 **Python 方法**，
它在派发给后端**之前**就拿到了真实张量：

```python
def forward(self, query, key, value, output_shape=None) -> torch.Tensor:
    ...
    query = query.view(-1, self.num_heads, self.head_size)   # L483
```

所以问题变成：**这个 Python 帧在视频 prefill 时到底会不会执行？** 探针挂在它上面实测。

---

## B.2 实测结果（决定性）

```
prompt_tokens      : 2420
chosen q / kv      : 2420 / 2420
key_is_full_prompt : True                     <-- K 覆盖完整 prompt
statistic_shape    : [4, 2420]                <-- == build_sparse_selection 的输出形状
mass_check         : 1792.0 / 1792.0          <-- 精确吻合
statistic_seconds  : 0.1566                   <-- 单层、64 个 text 行
hook_calls         : {'n': 280, 'prefill': 84, 'decode': 196}
```

观测到的全部 prefill 形状：

```
q=(8192, 3584) k=(8192, 512)  x 28   # 初始化 warmup（桶大小）
q=( 256, 3584) k=( 256, 512)  x 28
q=(2420, 3584) k=(2420, 512)  x 28   # <-- 真实请求，28 层全部可见
```

捕获到的张量：`dtype=torch.bfloat16`，`num_heads=28`，`num_kv_heads=4`，
`heads_per_kv=7`，`head_size=128`，层名如
`language_model.model.layers.0.self_attn.attn`。

### 三重校验

1. **钩子确实在 prefill 与 decode 两条路径上都被调用**（84 / 196 次）。
2. **K 覆盖完整 prompt**（`key_is_full_prompt: True`，2420 = prompt 长度），
   因此 text→prompt 的完整注意力信号一次 prefill 就能拿到。
3. **重建出的注意力在数学上正确**：softmax 每行和为 1，叠加 64 个 text 行 × 7 个组头 × 4 个 KV 头
   = **1792**，实测 `acc.sum()` **恰好 1792.0**。说明「从 Q/K 重建 post-softmax 注意力」是准确的，
   不是近似。

### 关键澄清：不需要 softmax 矩阵本身

`build_sparse_selection` 对 query 维**也求和**，而 softmax 是按 (query, head) 行独立归一化的，
所以分母**不会**自动约掉——**必须真的形成概率**。探针就是这么做的：
分块算 `QK^T → 减行最大值 → exp → 归一化 → 对 query/组头求和`。
结论：**融合 kernel 不返回权重，并不妨碍我们拿到需要的统计量。**

---

## B.3 一条差点漏掉的证据（traceback 反而帮了忙）

探针第一版在钩子里写了 `.tolist()`，引擎起不来：

```
torch._dynamo.exc.Unsupported: Tensor.tolist() with non-integer tensor
  ...
  File ".../vllm/model_executor/models/qwen2.py", line 234, in forward
      attn_output = self.attn(q, k, v)
  File ".../B_probe_attn_weights.py", line 103, in hooked_forward
```

这个**失败**恰好证明了一件事：**torch.compile 会 trace 穿过这个 Python 方法**，
即 `Attention.forward` 在编译路径下也真实执行。

**由此得到一条硬约束**：这个钩子**必须是 trace-safe 的**——
不能有 host 同步（`.cpu()` / `.item()` / `.tolist()`）、不能有依赖数据的 Python 分支。
探针最终版只保留张量引用、形状全部留到 trace 区之外再读，即在 `enforce_eager` 下测量。

---

## B.4 两个工程陷阱（都实测踩到并解决）

### 陷阱 1：钩子跑在 EngineCore 子进程里，结果回不到父进程

第一版钩子 `captured` 列表永远是空的（`NO_HOOK`），但日志里明明有 `hooked_forward` 的栈。
原因：vLLM 默认多进程，EngineCore 在子进程中 fork，**append 发生在子进程的 COW 副本里**。

**解法**：`VLLM_ENABLE_V1_MULTIPROCESSING=0`（`vllm/v1/engine/llm_engine.py:148`），
让引擎在进程内运行。→ 真做 GPU proposer 时，这条必须一并解决
（要么进程内，要么走 IPC 回传）。

### 陷阱 2：chunked prefill 会让 K 只剩当前分块

未关闭 chunked prefill 时：`kv_len = 256 ≠ prompt_tokens = 2420`，`key_is_full_prompt: False`。
即 text 行看到的只是本块 K，**拿不到完整 prompt 的注意力**。

**解法**：`enable_chunked_prefill=False` → `key_is_full_prompt: True`，`kv_len = 2420`。
（替代方案是把各块的 K 累积起来，但那要额外保存 KV。）

---

## B.5 成本：实测值 vs 真实成本（必须分清）

**实测**：64 个 text 行 × 2420 kv，**单层 157 ms**（RTX 4090，朴素 chunked 参考实现）。

**但这不是基本成本，是实现开销。** 该统计量的理论计算量是

```
text_len × prompt_len × num_heads × head_dim × 2
```

对本任务 32 帧场景（prompt ≈ 19,166，text 行 ≈ 64）：

```
64 × 19166 × 28 × 128 × 2 ≈ 8.8 GFLOP
```

A6000 fp16 约 30 TFLOPS → **理论上约 0.3 ms**。实测 157 ms 全部是
Python/分块/小张量的调度开销（fp32、CH=512、逐层调用）。

**结论**：成本**不是**架构性障碍，是**实现质量**问题。
但 P0 必须把它做进一个融合 kernel，否则按朴素实现 ×28 层 ≈ 4.4 s/prefill，
会直接吃掉全部解码收益，让 STD 变成负收益。

> 这条必须写进 P0 的验收里：**selection 开销要压到 prefill 的个位数百分比**，
> 否则 STAGE-D 的净收益为负。

---

## B.6 判定与下一步

| 问题 | 答案 |
|---|---|
| prefill Q/K/V 能否拿到？ | ✅ 能，纯 Python 钩子，**不需要给 vLLM 打补丁** |
| 能否重建 STD 要的统计量？ | ✅ 能，形状 `[nkv, prompt_len]` 与 `build_sparse_selection` 一致，**数值精确校验通过** |
| 需要后端返回注意力权重吗？ | ❌ 不需要 |
| SSD 在 vLLM 上是否死路？ | ❌ **不是死路** |
| 有前提吗？ | ✅ 3 条：进程内（或 IPC）、关 chunked prefill（或累积 K）、钩子 trace-safe |
| 有风险吗？ | ✅ 1 条：selection 重建必须融合化，否则净收益为负 |

### → **启动 P0：最小 GPU proposer**

P0 的**验收标准**（先定死，避免又变成几周的开放工程）：

1. 能在真实 32 帧视频上跑出 `[nkv, prompt_len]` 的 top-K 稀疏选择；
2. **selection 开销 ≤ prefill 的 10%**（否则不继续）；
3. 与 dense 验证的输出按阶段 3 的三元组标准对比（平局分歧不算缺陷）；
4. 实测接受率 > 0（当前 token 级方案恒为 0，这是「稀疏草稿有没有用」的第一道真实读数）。

若第 2 条不达标（融合后仍 >10%），**止损**：判定 STD-on-vLLM 的 selection 成本不可摊销，
转 EAGLE-3 与负面刻画。
