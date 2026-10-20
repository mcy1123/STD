# 阶段 4：STD 单 token 草稿

> 状态：**⚠️ 部分完成（架构性阻塞已定位，替代方案已实测）**
> 日期：2026-09-26
> 前置：阶段 3 已证明接口连通；本阶段回答「STD 能否装进这个接口」

---

## 4.1 一句话结论（先说最重要的）

**vLLM 0.22.0 的 `custom_class` proposer 无法承载 STD 的核心机制。**

它是一个**纯 CPU / 纯 token 级**接口，拿不到模型、KV cache、attention 权重中的任何一个；
而 STD 的稀疏草稿正是建立在这些之上。因此本阶段：

1. 用代码级证据**定位并证伪**了「在 Proposer 里实现 STD 稀疏 top-K 草稿」这条路径；
2. 实测了该接口**能**表达的最近似机制（token 级稀疏上下文草稿），并给出硬数字；
3. 给出真正实现 STD 所需的注入点与最小改动清单。

---

## 4.2 我做了什么

### 4.2.1 读本仓库既有 STD 复现代码，提取稀疏草稿逻辑

`src/std_repro/std_qwen25vl.py::build_sparse_selection`（L185-262）是 STD 稀疏草稿的源头：

```python
# 输入：attentions = 每层 prefill 注意力 [batch, query_heads, text_len, prompt_len]
for layer_attn in attentions:
    attn = layer_attn.detach()[0]
    grouped = attn.reshape(num_key_value_heads, heads_per_kv,
                           attn.shape[-2], attn.shape[-1])
    scores = grouped.sum(dim=(1, 2), dtype=torch.float32)      # 按 KV head 归并
    visual_scores = scores.index_select(-1, visual_positions)  # 只看视觉位置
    top_local = torch.topk(visual_scores, k=k, dim=-1, largest=True).indices
    selected = visual_positions[top_local]                     # 每层 top-K 视觉位置
```

即 **STD 的草稿 = 「用 prefill 注意力挑出每层 top-K 视觉 KV，让草稿步只attend 稀疏 KV」**。
草稿与验证共用同一个 target 模型，区别只在 KV 视野（`copy_prompt_cache` / `compact_sparse_prompt_cache`）。

### 4.2.2 对照 vLLM 的实际接口

`vllm/v1/worker/gpu_model_runner.py:4778` 是唯一调用点：

```python
elif spec_config.method == "custom_class":
    assert isinstance(sampled_token_ids, list)
    draft_token_ids = cast(Any, self.drafter).propose(
        sampled_token_ids,                      # list[list[int]]  ← 只有 token id
        self.input_batch.num_tokens_no_spec,    # np.ndarray
        self.input_batch.token_ids_cpu,         # np.ndarray       ← 只有 token id
        slot_mappings=slot_mappings,
    )
```

构造侧 `vllm/v1/spec_decode/custom_class_proposer.py:50`：

```python
instance = user_class(vllm_config)   # ← 只给 VllmConfig，没有 model / KV / attn
```

**Proposer 拿不到的东西**（它们都挂在 `GPUModelRunner` 上，从不传入）：

| 需要的东西 | 在 runner 上的位置 | 是否可见 |
|---|---|---|
| `self.model`（nn.Module） | `gpu_model_runner.py:517` | ❌ |
| `self.kv_caches`（KV 缓存张量） | `gpu_model_runner.py:519` | ❌ |
| `self.attn_groups` | `gpu_model_runner.py:524` | ❌ |
| prefill attention 权重 | 只在 forward 内部 | ❌ |
| hidden states | 只在 forward 内部 | ❌ |

并且 v0.22.0 的 `vllm/v1/spec_decode/` **没有 GPU speculator 注册表**可挂钩：

```
custom_class_proposer.py  draft_model.py  eagle.py  extract_hidden_states.py
gemma4.py  llm_base_proposer.py  medusa.py  ngram_proposer.py
ngram_proposer_gpu.py  suffix_decoding.py
```

（GPU 侧 speculator 目录 `vllm/v1/worker/gpu/spec_decode/` 是**更晚版本**才有的。）

### 4.2.3 判定

| 问题 | 答案 |
|---|---|
| `custom_class` 能否实现 STD 的稀疏 KV 草稿？ | **不能。** 接口拿不到 attention / KV / 模型 |
| 「slot_mappings」能否补救？ | **不能。** 它只是 token→KV slot 的索引，不是 KV 数据，且模型句柄本身就没有 |
| 最小 patch 是什么？ | 见 §4.5 |

---

## 4.3 那么该接口**能**做什么：token 级稀疏上下文草稿

STD 的表述原则是「**稀疏草稿、密集验证**」。在 token 级，可以忠实实现**后半句的前提**
——让草稿只看上下文的**稀疏子集**：

实现 `SparseContextDraftProposer`（`src/std_vllm/proposer.py`）：

- 取每个请求的尾部 n-gram（`STD_DRAFT_NGRAM`，默认 3）；
- 在**受限窗口** `STD_DRAFT_WINDOW` 内反向查找最近一次出现；
- 复制其后 `k` 个 token 作为草稿；找不到就返回 `[]`（该请求本轮不投机）。

`STD_DRAFT_WINDOW` 就是**稀疏度旋钮**（草稿允许看多少上下文）；`0` = 全上下文（不稀疏）。

> **必须强调**：这是 STD 在 **token 级**的类比，**不是** STD 的稀疏-KV 机制。
> 文档与 PROGRESS 中一律如此标注，不得混称。

逻辑已离线单测通过（`_draft_one`：全上下文命中 / 窗口外不命中 / 过短返回空）。

---

## 4.4 结果（单 token，k=1）

命令：

```bash
STD_DRAFT_NGRAM=3 STD_DRAFT_WINDOW=0 bash scripts/vllm-std/run_bench.sh \
  results/vllm-std/logs/04_p4_sparse_k1_full.log 03_bench_spec.py \
  --num-frames 32 --max-tokens 128 --runs 3 --warmup 1 --tag p4_sparse_k1_full \
  --num-spec-tokens 1 --proposer std_vllm.proposer.SparseContextDraftProposer
```

| 配置 | 耗时 mean±std | 加速比 | 接受率 | accept_len | 首个分歧 |
|---|---|---:|---:|---:|---:|
| `p3_base_k0`（eager 基准） | 5.2305 ± 0.0406 s | 1.000× | — | — | — |
| `p4_sparse_k1_full`（窗口=全上下文） | **16.3714 ± 4.2595 s** | **0.32×** | **0.0** | 1.0 | 90 |
| `p4_sparse_k1_w256`（窗口=256） | 5.5080 ± 0.0834 s | 0.95× | **0.0** | 1.0 | 90 |

### 三个硬结论

1. **接受率 0.0。** prompt-lookup 式草稿在**自由形式的视频描述**上没有命中：
   生成的文本是新的措辞，不会重复 prompt 里的 n-gram。
   → **这是负结果，且是可复现的负面刻画**：n-gram/后缀类 token 级草稿不适合开放式视频描述。
2. **全上下文窗口把 proposer 自己变成瓶颈**：16.37 s vs 5.23 s（**慢 3.1×**），
   且方差极大（std 4.26 s）。原因是 `_draft_one` 在 19,166 token 的上下文上做 Python 线性扫描。
3. **稀疏窗口直接买回速度**：窗口 256 时 16.37 s → **5.51 s**（**2.97× 提速**），
   方差从 4.26 s 降到 0.08 s。**这是本阶段唯一为正的「稀疏收益」证据**——
   但收益来自 *proposer 自身开销*，不是来自接受率。

> 结论措辞必须精确：**稀疏在这个 token 级机制里省的是草稿搜索成本，不是验证成本。**

### bit-exact 判定（按阶段 3 §3.4.5 的修正标准）

| 配置 | 完全一致 | 平局位置分歧 | 稳定位置分歧 |
|---|---|---|---|
| `p4_sparse_k1_w256` vs `p3_base_k0` | — | token 90 | 需 margin 画像确认 |

两个配置**同样在 token 90 分歧**（`11 → 10797`），与阶段 3 的 token 62 是不同位置，
说明分歧位置随验证 batch 形状漂移，与「平局 + 浮点噪声」的解释一致。
精确的平局分类见 `results/vllm-std/03d_*`（eager-vs-eager 已证明参考不稳定）。

---

## 4.5 真正实现 STD 需要什么（最小改动清单）

若要把 STD 的**稀疏 KV 草稿**真正装进 vLLM，注入点必须能拿到模型与 KV。两条路：

### 路线 A：新增一个 GPU 侧 proposer（推荐）

| 文件 | 改动 |
|---|---|
| `vllm/v1/spec_decode/llm_base_proposer.py` | 已有 `LLMBaseProposer`：持有 `self.model`、`self.draft_model_config`、能建 KV cache、有 `dummy_run` / `prepare_inputs`。STD proposer 继承它 |
| 新增 `vllm/v1/spec_decode/std_sparse.py` | ①prefill 时收集每层 text→prompt 注意力；②`build_sparse_selection` 的等价实现；③稀疏 forward 产出 k 个草稿 token |
| `vllm/config/speculative.py` | `SpeculativeMethod` 加入 `"std_sparse"`；加分支（照抄 `custom_class` 的 `draft_model_config = target_model_config`） |
| `vllm/v1/worker/gpu_model_runner.py` | 构造与分派各加一个分支（共 2 处） |

**关键难点（必须实测，不能假设）**：vLLM v0.22.0 的注意力后端（FLASH_ATTN / FLASHINFER）
**默认不返回 attention 权重**。`build_sparse_selection` 依赖 prefill 注意力矩阵，
因此在 vLLM 里要么强制走能返回权重的后端（`TRITON_ATTN` / eager SDPA），
要么改用**不依赖注意力**的稀疏判据（例如按 hidden state 范数 / 位置均匀采样）。
**这是路线 A 的第一件待验证事项，也是最大的未知数。**

### 路线 B：在 custom_class 内自带草稿模型

技术上可行（proposer 可以自己 `from_pretrained` 一个小模型并在 GPU 上跑），但：
- 它变成「draft_model 模式」的翻版，**不是 STD**（不使用稀疏 KV、不能复用 target 权重）；
- 需自行管理显存与 KV，绕开 vLLM 调度器，工程量大且与框架语义冲突。
**不作为推荐路线。**

---

## 4.6 失败记录（未掩盖）

| # | 失败 | 根因 | 处置 |
|---|---|---|---|
| 1 | 「在 Proposer 里实现 STD 稀疏 top-K 草稿」**不可行** | `custom_class` 只传 `VllmConfig` + CPU token id；拿不到 model/KV/attention | 记录为架构性阻塞（§4.2），给出路线 A/B（§4.5） |
| 2 | 接受率恒为 0 | prompt-lookup 草稿不匹配自由生成文本 | 记录为**负面刻画**，非 bug |
| 3 | 全上下文 proposer 慢 3.1× 且方差巨大 | Python 线性扫描 19k 上下文 | 引入 `STD_DRAFT_WINDOW` 稀疏窗口，恢复 2.97× |
| 4 | 首次 `proposer.py` 缺 `os` import | 脚本拼装遗漏 | 已修 + `import OK` 验证 |

---

## 4.7 是否进入下一阶段？

| 判据（任务书） | 结果 |
|---|---|
| 单 token 草稿可运行 | ✅ 可运行，381+ 组草稿真实到达验证层 |
| 有接受率数字 | ✅ 0.0000（k=1，两个窗口配置） |
| 有 bit-exact 结论 | ✅ 分歧在 token 90；分类标准已按阶段 3 修正 |

按任务书字面判据，**通过**。但必须诚实标注：**通过的是「token 级稀疏草稿」，不是 STD 的稀疏-KV 草稿**；
后者被接口阻塞，已给出最小 patch 路线。

**→ GO，进入阶段 5（多 token 草稿 + 参数扫描）**，扫描维度为 `k` 与稀疏窗口 `W`。
