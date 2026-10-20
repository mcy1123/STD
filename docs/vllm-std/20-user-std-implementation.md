# 阶段 0：读懂本仓库真正的 STD 实现

> 状态：**✅ 完成**
> 日期：2026-09-26
> 方法：只读代码，每个结论附**文件路径 + 行号 + 原文**
> 核心文件：`src/std_repro/std_qwen25vl.py`（1516 行）

---

## 0.1 一句话回答「用户的 STD 是怎么产生草稿的」

> **用 prefill 注意力按 KV head 选出 top-K 视觉位置 → 把 KV cache 物理压缩成
> `非视觉位置 ∪ 该 head 的 top-K 视觉位置` → 让同一个 target 模型在这个缩短的 cache 上
> 做 γ 步自回归 greedy 前向 → 每步的 argmax 就是一个草稿 token。**

**草稿来自模型前向，不是复制上下文。** 这是它与 vLLM 上 n-gram 类比的根本区别。

---

## 0.2 Q1：稀疏选择的判据是什么？

**答：全 query 求和（不是 last-query，也不是 last-K）。**

**文件**：`src/std_repro/std_qwen25vl.py`
**函数**：`build_sparse_selection`（def **L185**）

原文 **L219-243**：

```python
    for layer_attn in attentions:
        # [batch, query_heads, text_len, prompt_len]        # L220
        attn = layer_attn.detach()[0]
        visual_positions_device = visual_positions.to(attn.device)
        query_heads = attn.shape[0]
        ...
        heads_per_kv = query_heads // num_key_value_heads
        grouped = attn.reshape(                             # L233
            num_key_value_heads,
            heads_per_kv,
            attn.shape[-2],      # text_len  <- 全部 text query 行
            attn.shape[-1],      # prompt_len
        )
        scores = grouped.sum(dim=(1, 2), dtype=torch.float32)   # L239 ★
        visual_scores = scores.index_select(-1, visual_positions_device)
        top_local = torch.topk(visual_scores, k=k, dim=-1, largest=True).indices
        selected = visual_positions_device[top_local]
        layer_positions.append(torch.sort(selected, dim=-1).values.cpu())   # L243
```

**三点关键**：

1. `dim=(1,2)` 归约 **heads_per_kv** 与 **text_len（全部 query 位置）**。
   → 判据是「哪些视觉位置对**整段文本**重要」，不是「对下一个 token 重要」。
2. **逐层独立**：`for layer_attn in attentions`（L219），每层一套 `topk_positions`。
3. **逐 KV head 独立**：`top_local` 形状 `[nkv, k]`（L241），
   `layer_positions` 每层存 `[nkv, k]` 的索引（L243 的 `sort(selected)`）。
   后面 `compact_sparse_prompt_cache` 确实按 `topk[head]` 逐 head 取（见 Q2）。

**K 的算法**（L211-215）：

```python
    text_len = max(1, prompt_len - int(text_start))
    k = int(explicit_k) if explicit_k is not None else max(1, int(target_k_plus_text) - text_len)
    k = min(k, int(visual_positions.numel()))
```

即 **`K = target_k_plus_text - text_len`**；论文口径 `K + text = 1024`。

---

## 0.3 Q2：选中 KV 后，稀疏前向具体怎么跑？

### ① 不是加 mask，而是**物理压缩 KV cache**

**文件**：`src/std_repro/std_qwen25vl.py`
**函数**：`compact_sparse_prompt_cache`（def **L302**）

docstring 原文（L307-312）：

```
"""Physically compact sparse prompt KV caches to the selected KV set.

The selected order can differ per KV head. For autoregressive q_len=1 sparse
drafting this is valid because we attend to the whole compacted prefix without
a causal mask; RoPE is already baked into keys. Generated tokens are appended
after the compact prompt and use explicit original position_ids.
"""
```

实现（**L315-334**）：

```python
    compact_len = int(selection.non_visual_positions.numel() + selection.k)   # L315
    for layer_idx, layer_cache in enumerate(past_key_values):                 # L316
        topk = selection.topk_positions[layer_idx]
        for cache in layer_cache:
            data = cache.data
            bsz, kv_heads, _, head_dim = data.shape
            indices = []
            non_visual = selection.non_visual_positions
            for head in range(kv_heads):                                      # L323
                idx = torch.cat([non_visual, topk[head]], dim=0)              # L324 ★逐 head
                idx = torch.unique(idx, sorted=True).to(data.device)
                if int(idx.numel()) != compact_len:
                    raise RuntimeError("Compacted sparse KV produced an unexpected length.")
                indices.append(idx)
            index = torch.stack(indices, dim=0)
            index = index.view(1, kv_heads, compact_len, 1).expand(bsz, kv_heads, compact_len, head_dim)
            compacted = data.gather(2, index).contiguous()                    # L331
            data[:, :, :compact_len, :].copy_(compacted)                      # L332
            cache.current_length.fill_(compact_len)                           # L333
    current_length_data.fill_(compact_len)                                    # L334
```

**要点**：
- 保留集 = **全部非视觉位置** + **该 KV head 的 top-K 视觉位置**（L324）。
- **每个 KV head 的索引不同**（`topk[head]`，L324）→ 逐 head 压缩。
- 压缩后 cache 长度从 `prompt_len` 降到 `compact_len = len(non_visual) + K`（L315）
  → **这是真省时间的地方**：draft 步的 attention 只扫 `compact_len` 个 key，而不是整个 prompt。

### ② 压缩后的 attention：**完全没有 mask**

**文件**：`src/std_repro/std_qwen25vl.py`
**函数**：`_sparse_aware_attention_forward`（def **L391**）

原文 **L415-417**（只支持单 token 自回归）：

```python
    bsz, q_len, _ = hidden_states.size()
    if q_len != 1:
        raise RuntimeError("Sparse draft attention currently expects autoregressive q_len=1 calls.")
```

原文 **L443-454**（`gqa_sdpa` 分支，含 mask 参数）：

```python
    elif controller.sparse_attn_mode == "gqa_sdpa":
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,          # L450  <- 无 mask
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,         # L452  <- 非因果
            enable_gqa=True,         # L453
        )
```

**答：不用 mask。** `attn_mask=None` 且 `is_causal=False`（L450/L452）。
稀疏性 100% 由「KV cache 已被物理压缩」实现，而不是靠遮罩。
理由（L310-312）：q_len=1 的单查询可以看整个压缩前缀，无需因果遮罩，且 RoPE 已烘进 key。

### ③ 跑哪些层、哪些 head？

**层**：`SparseDraftController.install`（**L287-296**）**patch 所有层**：

```python
    def install(self) -> None:
        if self._installed:
            return
        for layer in self.model.model.layers:                       # L290  全部层
            attn = layer.self_attn
            if getattr(attn, "_std_original_forward", None) is None:
                attn._std_original_forward = attn.forward
                attn.forward = types.MethodType(_sparse_aware_attention_forward, attn)
            attn._std_sparse_controller = self
        self._installed = True
```

另有**可选的提前退出** `draft_layers`（`_draft_layer_window`，**L690**）：
只跑前 N 个 decoder block，`None` = 全栈（L282-283, L700-704）。
**这是 W1 的加速开关之一，不影响判据。**

**head**：全部 query head 参与，但 **KV 侧按 KV head 分组**（GQA）。
`gqa_sdpa` 用 `enable_gqa=True`（L453）；`repeat_sdpa` 用 `repeat_kv`（L456-457）；
`triton_gqa` 走 `fused_gqa_attention`（L442，batch=1 only，L438-439）。

### ④ 三种 sparse_attn_mode

**L273**（构造时校验）：

```python
        if sparse_attn_mode not in {"repeat_sdpa", "gqa_sdpa", "triton_gqa"}:
            raise ValueError(f"Unsupported sparse_attn_mode={sparse_attn_mode!r}.")
```

| mode | 实现 | 约束 |
|---|---|---|
| `repeat_sdpa` | `repeat_kv` + SDPA | 通用、最慢 |
| `gqa_sdpa` | SDPA `enable_gqa=True` | 需要 PyTorch ≥2.5 |
| `triton_gqa` | `fused_gqa_attention`（本仓库 `src/std_repro/triton_attention.py`） | **batch=1 only**（L438-439） |

---

## 0.4 Q3：草稿 token 怎么产生？单/多 token？温度？

**答：多 token、γ 步自回归、greedy（温度=0）。**

**文件**：`src/std_repro/std_qwen25vl.py`
**函数**：`_draft_tokens`（def **L1042**）

原文 **L1042-1059**：

```python
def _draft_tokens(
    model,
    first_token: torch.Tensor,
    gamma: int,
    past_key_values,
    sparse_controller: SparseDraftController,
    start_position: int,
) -> List[int]:
    token = first_token.to(_first_model_device(model))
    drafted: List[int] = []
    with sparse_draft_context(model, sparse_controller):          # L1054 打开稀疏
        for offset in range(gamma):                               # L1055 γ 步
            drafted.append(int(token.item()))
            position_ids = torch.tensor([[start_position + offset]], dtype=torch.long, device=token.device)
            outputs = _sparse_step_outputs(model, sparse_controller, token, past_key_values, position_ids)
            token = _token_argmax(outputs.logits)                 # L1059 下一步的输入
    return drafted
```

**温度**：`_token_argmax`（**L117-118**）：

```python
def _token_argmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
```

**纯 argmax，temperature = 0。**

**草稿构成**：`drafted[0] = first_token`（**密集模型已验证过的下一个 token**），
`drafted[1..γ-1]` 是稀疏前向自回归推出的。所以长度为 **γ**，
其中第 0 个必然被接受（它来自密集前向），**后 γ-1 个才是真正的投机赌注**。
→ 这解释了为什么 `accept_len` 的期望下限是 1。

---

## 0.5 Q4：接受率 60–80% 是在什么任务/数据/配置下测的？

**配置**（`PROGRESS.md:648` §17.1 原文）：

> 20 样本（seed-42，与 A100 实验同序）、**128 帧**、**256 tokens**、**γ=9**、**K+text=1024**、**CoT**，共 **442 轮验证**。

脚本默认值与之吻合（`scripts/benchmark_a100_dynamic.py`）：
`--frame-num 128`（L31）、`--max-new-tokens 128`（L32）、`--gamma 9`（L37）、`--k-plus-text 1024`（L38）。

**任务**：Video-MME 视频问答（CoT 提示）。

**接受率实测值**（`PROGRESS.md:690-698`，§17 的 static 列）：

| 样本 | static |
|---|---:|
| 754-1 | 0.507 |
| 154-3 | 0.541 |
| 050-1 | 0.663 |
| 496-3 | 0.796 |
| 445-2 | 0.831 |
| 102-2 | 0.850 |
| 504-2 | **0.966** |
| 647-1 | **0.966** |
| 717-1 | **0.983** |
| 599-1 | **1.000** |

**范围 0.507–1.000，均值 ≈0.81。** 任务书说的「60–80%」与之相符
（10 个里有 5 个落在 0.66–0.85 区间）。

> ⚠️ **注意章节归属**：这些数字在 **§17**（`PROGRESS.md:642`），**不在 §19**。
> §19（`PROGRESS.md:1514`）是 W1 decode 开关，无接受率表。详见 `17-section19-criterion.md`。

---

## 0.6 Q5：完整实现跑在哪个框架？入口脚本？

**框架**：**本地 HuggingFace transformers**，不是 SGLang。
- 模型：`Qwen2_5_VLForConditionalGeneration`（`src/specvlm/models/modeling_qwen2_5_vl.py` 为随仓库分发的实现）
- 双 KV cache：密集 verifier 一份 + 稀疏 draft 一份（`std_generate_qwen25vl` 内，L1213-1240）
- KV cache 容器：`src/specvlm/kv_cache/kv_cache.py`

**sglang 只出现在 EAGLE-3 路线**（`scripts/analysis/eagle3_video_bench.py`），**与本 STD 无关**。
**仓库里只有一套 STD 实现**：

| 实现 | 文件 | 判据 | 说明 |
|---|---|---|---|
| **静态 STD**（主线） | `src/std_repro/std_qwen25vl.py` | `grouped.sum(dim=(1,2))` @ L239 | 本次要对齐的目标 |
| 动态 STD | `src/std_repro/dynamic_std_qwen25vl.py` | `bootstrap_mode="attention"` 时**同一条**（`:258` 调用同函数）；否则用不依赖注意力的 `attention_free_visual_topk`（`dynamic_selection.py:104`） | 本任务**不需要** |

**入口脚本**：

| 脚本 | 用途 |
|---|---|
| `scripts/benchmark_std.py` | README 指定的主入口 |
| **`scripts/benchmark_a100_dynamic.py`** | **§17 那张表的产出者**；`:330` 是 `method=="static"` 臂 |
| `scripts/benchmark_dynamic_std.py` | 动态版 |
| `scripts/sweep_std_inprocess.py` | 一次加载模型做 K/γ 扫描 |

---

## 0.7 接受率的统计口径（必须是实测，不能是"理论上应该"）

**接受判定**（`std_qwen25vl.py` **L1033-1039**）：

```python
    accept_len = 0
    while accept_len < len(draft) and int(draft[accept_len]) == int(next_dense_token.item()):
        token_tensor = torch.tensor([[int(draft[accept_len])]], dtype=torch.long, device=device)
        ...
        accept_len += 1
    return accept_len, int(next_dense_token.item())
```

**逐 token 前缀匹配**：草稿第 i 个 token 与**密集模型**给出的第 i 个 token 相等则接受，遇到第一个不等就停。

**接受率定义**（**L96-99**）：

```python
    def acceptance_rate(self) -> float:
        ...
        return self.accepted_draft_tokens / self.proposed_draft_tokens
```

即 **接受草稿 token 数 / 提出的草稿 token 数**。

**这是标准 SD 语义，且验证方是密集模型** —— 与 vLLM 的密集验证可直接对照。

---

## 0.8 完整数据流（供阶段 1 映射用）

```
① prefill（返回 attention）            prefill_prompt L160
        ↓
② 密集 verifier 前向 → dense_next       L1218  _token_argmax(dense_output.logits)
        ↓
③ build_sparse_selection L185/L1224
     grouped.sum(dim=(1,2)) L239  → 每层 [nkv, k] 视觉索引
        ↓
④ compact_sparse_prompt_cache L302
     每 head: non_visual ∪ topk[head]  → KV cache 长度 prompt_len → compact_len
        ↓
⑤ _draft_tokens L1042：γ 步稀疏自回归
     _sparse_aware_attention_forward L391（无 mask，is_causal=False）
     _token_argmax L117 → 草稿 [d0..d_{γ-1}]
        ↓
⑥ 密集 verifier 逐 token 验证 → accept_len   L1033
        ↓
⑦ acceptance_rate = accepted / proposed      L96
```

---

## 0.9 阶段 0 验收

| 问题 | 答案 | 证据 |
|---|---|---|
| Q1 判据 | 全 query 求和，逐层逐 KV head | `std_qwen25vl.py:239`（L220/L233 佐证形状） |
| Q2 稀疏前向 | **物理压缩 KV**，**无 mask**，q_len=1，全 28 层（可选 `draft_layers` 提前退出） | `:302-335`、`:450-452`、`:290` |
| Q3 草稿 | γ 步自回归，**greedy 温度=0**，`drafted[0]` 来自密集前向 | `:1042-1059`、`:117-118` |
| Q4 接受率 | Video-MME / 128 帧 / 256 tok / γ=9 / K+text=1024 / CoT；**0.507–1.000，均值 ≈0.81** | `PROGRESS.md:648,690-698` |
| Q5 框架入口 | **本地 HF transformers**；`benchmark_a100_dynamic.py:330`（static 臂） | 见 §0.6 |

**没有「应该是」。每个结论都能点回具体行。**
