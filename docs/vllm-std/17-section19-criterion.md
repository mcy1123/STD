# 阶段 0/1：§19（实为 §17）静态 STD 的稀疏选择判据

> 状态：**✅ 有明确答案：是（全 query 求和）**
> 日期：2026-09-26
> 方法：只读代码，逐条给出文件路径 + 行号 + 原文

---

## 0.1 ⚠️ 首先更正一处我自己的错误归属

任务书问的是「**§19** 那次接受率 0.966–1.000 的静态 STD」。
**但 §19 里没有这些数字。**

| 位置 | 实际内容 |
|---|---|
| `PROGRESS.md:1514` `## 19. W1：decode-only 优化三件套` | 讲的是 `c_s` / draft layers / compile 三个 decode 开关，**无接受率表** |
| `PROGRESS.md:642` `## 17. 动态路由：2026-09-18 的机制结论` | **0.507–1.000 的 static 接受率表就在这里**（`PROGRESS.md:690-698`） |

原文（`PROGRESS.md:688-698`）：

```
| 样本 | static | dynamic | Δ |
| 754-1 | 0.507 | 0.698 | +0.191 |
...
| 504-2 | 0.966 | 0.877 | −0.089 |
| 647-1 | 0.966 | 0.935 | −0.031 |
| 717-1 | 0.983 | 0.796 | −0.187 |
| 599-1 | 1.000 | 1.000 | 0.000 |
```

**这个错误归属是我在此前 `14-acceptance-probe.md` §3′.5 里写下的，在此更正。**
（该表述已同步修正，见 §1.4。）

**但实质问题不变**：那批 static 接受率是哪个判据算出来的？下面是答案。

---

## 0.2 判据代码：确认是全 query 求和

**文件**：`src/std_repro/std_qwen25vl.py`
**函数**：`build_sparse_selection`（def 在 **L185**）

**原文 L219-243**（逐字）：

```python
    for layer_attn in attentions:
        # [batch, query_heads, text_len, prompt_len]          <- L220
        attn = layer_attn.detach()[0]
        visual_positions_device = visual_positions.to(attn.device)
        query_heads = attn.shape[0]
        if query_heads % num_key_value_heads != 0:
            raise ValueError(...)
        heads_per_kv = query_heads // num_key_value_heads
        grouped = attn.reshape(                                      # L233
            num_key_value_heads,
            heads_per_kv,
            attn.shape[-2],          # <- text_len（全部 text query 行）
            attn.shape[-1],          # <- prompt_len
        )
        scores = grouped.sum(dim=(1, 2), dtype=torch.float32)        # L239  ★
        visual_scores = scores.index_select(-1, visual_positions_device)
        top_local = torch.topk(visual_scores, k=k, dim=-1, largest=True).indices
        selected = visual_positions_device[top_local]
```

**判定**：

- `grouped` 的形状是 `[num_kv_heads, heads_per_kv, text_len, prompt_len]`（L233-238 + L220 注释三方一致）。
- `dim=(1, 2)` = 对 **`heads_per_kv`** 与 **`text_len`（全部 query 位置）** 求和（L239）。
- **这就是「全 query 求和」，与探针假设逐字一致。**

---

## 0.3 这条判据确实被静态 STD 使用，且是**真稀疏 KV 前向**

### 调用点在静态路径内

**文件**：`src/std_repro/std_qwen25vl.py`
**函数**：`std_generate_qwen25vl`（def 在 **L1118**）
**调用点**：**L1224**

```python
    selection = build_sparse_selection(          # L1224
        attentions,
        prompt_ids,
        video_token_id,
        text_start,
        target_k_plus_text=target_k_plus_text,
        explicit_k=explicit_k,
        num_key_value_heads=model.config.num_key_value_heads,
    )
```

### 是真稀疏前向，不是 token 级类比

同一个函数内，**双 cache 结构**清晰可见：

```python
    if dense_output.attentions is not None:      # L1213
        raise RuntimeError("Correctness invariant violated: dense verifier prefill "
                           "returned attentions (non-canonical attention path).")
    dense_next = _token_argmax(dense_output.logits)          # L1218  <- 密集验证
    ...
    if attentions is None:                       # L1220
        raise RuntimeError("Selection prefill did not return attentions.")
    selection = build_sparse_selection(...)      # L1224  <- 稀疏选择
    controller = SparseDraftController(          # L1235  <- 稀疏草稿
        model, selection, sparse_attn_mode=sparse_attn_mode,
        use_compile=use_compile, draft_layers=draft_layers)
```

`SparseDraftController`（def 在 **L268**）原文 L269：

```python
    """Monkey-patch Qwen2.5-VL attention modules with sparse draft behavior."""
```

且 `sparse_attn_mode` 只能是 `{"repeat_sdpa", "gqa_sdpa", "triton_gqa"}`（L273）——
**真的在改 attention 计算**，并带 `copy_prompt_cache` / `compact_sparse_prompt_cache`（L302-338）
做稀疏 KV 缓存。

### 接受率是**对密集验证**测的

**文件**：`src/std_repro/std_qwen25vl.py`

```python
    accept_len = 0                                                   # L1033
    while accept_len < len(draft) and int(draft[accept_len]) == int(next_dense_token.item()):
        token_tensor = torch.tensor([[int(draft[accept_len])]], ...)
        ...
        accept_len += 1                                              # L1038
    return accept_len, int(next_dense_token.item())
```

```python
    def acceptance_rate(self) -> float:                              # L96
        ...
        return self.accepted_draft_tokens / self.proposed_draft_tokens   # L99
```

`draft[accept_len]` 与 `next_dense_token` 逐 token 比对 → **前缀匹配式接受**，标准 SD 语义。

---

## 0.4 那批数字是哪个脚本产出的

**文件**：`scripts/benchmark_a100_dynamic.py` **L330**

```python
                        if method == "static":
                            result, selection = std_generate_qwen25vl(
                                model, inputs, VIDEO_TOKEN_ID, eos, **common)
                        else:
                            result, selection, dynamic_stats = dynamic_std_generate_qwen25vl(
                                ...)
```

**这正是 §17 那张 `static` vs `dynamic` 表的产出者**：
`static` 臂 = `std_generate_qwen25vl` = 用 L239 的全 query 求和判据。

---

## 0.5 交叉核验（阶段 1）：静态 vs 动态的判据差异

| 路径 | 判据 | 位置 |
|---|---|---|
| **静态 STD**（`std_generate_qwen25vl`） | `grouped.sum(dim=(1,2))` 全 query 求和 | `std_qwen25vl.py:239` |
| 动态 STD，`bootstrap_mode="attention"` | **同一个** `build_sparse_selection` | `dynamic_std_qwen25vl.py:258` |
| 动态 STD，`bootstrap_mode != "attention"` | **不同**：`attention_free_visual_topk`（不依赖 attention） | `dynamic_selection.py:104` |

**结论**：
- 静态臂与「attention bootstrap 的动态臂」**共用同一条判据**（逐字相同）。
- 只有「非 attention bootstrap」的动态臂换了判据，且那是**为省掉 prefill 注意力采集**而做的，
  不是为修正 query 维度。
- **全仓库没有任何地方把评分限制在「最后一行」或「最后 K 行」。**

---

## 0.6 阶段 0/1 的答案

| 问题 | 答案 | 证据 |
|---|---|---|
| §19（实为 §17）静态 STD 用的是全 query 求和吗？ | **是** | `std_qwen25vl.py:239`，调用点 `:1224`，产出者 `benchmark_a100_dynamic.py:330` |
| 是真稀疏 KV 前向吗？ | **是** | `SparseDraftController`（`:268`）、`sparse_attn_mode∈{repeat_sdpa,gqa_sdpa,triton_gqa}`（`:273`）、`compact_sparse_prompt_cache`（`:302`） |
| 接受率是对密集验证测的吗？ | **是** | 逐 token 前缀比对 `:1033-1039`；`acceptance_rate` `:96-99` |
| 判据是否逐字相同（vs 探针）？ | **逐字相同** | 两处都是 `sum over (heads_per_kv, text_len)` |
| 是否存在 last-query 变体？ | **不存在** | 全仓库 grep 无 |
