# 阶段 2.0：路线决策

> 状态：**✅ 决策：走路线 B（HF draft + vLLM verify）**
> 日期：2026-09-26
> 依据：只读代码 + 静态核算（**未做 GPU 实测，已在 §2.0.6 标注**）

---

## 2.0.1 决策

# 路线 B —— HF draft + vLLM verify

**理由一句话**：路线 B 能**复用已被验证过、能拿到 0.81 接受率的原实现函数**
（`prefill_prompt` / `build_sparse_selection` / `compact_sparse_prompt_cache` / `_draft_tokens`），
而路线 A 要重新推导全部数值路径、且**有"复现不出 logits"的实质风险**；
且路线 B 的四个可行性门槛经静态核验**全部通过**。

---

## 2.0.2 四个门槛

### 门槛 1：显存（A6000 48 GiB）—— ✅ **通过**

**关键发现**：`prefill_prompt`（`std_qwen25vl.py:160-182`）是**两段式 prefill**，
注意力矩阵**只覆盖 text 行，不是整个 prompt**：

```python
def prefill_prompt(model, inputs, past_key_values, video_token_id, output_attentions):
    input_ids, text_start, text_input_ids, video_inputs = _split_video_text_inputs(...)
    model(**video_inputs, past_key_values=past_key_values)        # L172 视频前缀，不要注意力
    text_kwargs = {
        "input_ids": text_input_ids,                              # L175 只有 text 后缀
        "past_key_values": past_key_values,
        "output_attentions": output_attentions,                    # L177
    }
    output = model(**text_kwargs)                                 # L180
```

所以 `attentions` 每层形状是 `[heads, text_len, prompt_len]`（与 L220 注释一致），
**不是** `[heads, prompt_len, prompt_len]`。这把我原先担心的 9 GB 降到：

| 项 | 计算 | 32 帧（prompt≈19164, text≈50） |
|---|---|---|
| attention（瞬态，28 层） | 28 × 28 heads × text_len × prompt_len × 2 B | 28×28×50×19164×2 = **1.5 GB** |
| HF KV（视频前缀+text） | 2 × 28 层 × 4 kv_head × prompt_len × 128 × 2 B | **1.1 GB** |
| HF 权重 | 7B FP16 | **15.5 GB** |
| vLLM target 权重 | 实测（阶段 0） | **15.67 GB** |
| vLLM KV | 由 `gpu_memory_utilization` 控制 | 压缩到 ~3–5 GB |
| **合计** | | **≈ 37 GB / 48 GB** ✅ |

**缓解措施**（原实现自己就用了）：`std_generate_qwen25vl:1251` 注释原文
「Release the selection cache now ... **This frees ~2.2 GiB** before the sparse cache is allocated」
—— 即用完 selection cache 立刻 `del`。proposer 里照做。

**结论**：可行，但必须把 vLLM 的 `gpu_memory_utilization` 压到 ~0.35–0.40，
且**从小帧数起步**（4 帧先跑通，再上 32 帧）。

### 门槛 2：输入对齐 —— ✅ **通过（需运行时验证）**

- HF 侧需要的是 `input_ids` + `pixel_values_videos` + `video_grid_thw`
  （`prefill_prompt` 的 `_split_video_text_inputs` 就是拆这个，`std_qwen25vl.py:135`）。
- 这些在 vLLM 侧是**模型 forward 的 kwargs**，可钩：
  `vllm/model_executor/models/qwen2_5_vl.py:1352`
  `pixel_values_videos = kwargs.pop("pixel_values_videos", None)`
  消费点 `:1442-1451`（`video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw_list)`）。
- **机制**：钩 `Qwen2_5_VLForConditionalGeneration.forward`，把
  `pixel_values_videos` / `video_grid_thw` 存进模块级缓存，供 proposer 取用。
- **token id 不需要重建**：`propose()` 已经提供 `token_ids_cpu`（展开后的完整 token 序列），
  与 vLLM 内部使用的完全一致。**这消除了最大的对齐风险。**

### 门槛 3：tokenize 一致性 —— ✅ **通过（部分已实测）**

- 两侧读**同一份 tokenizer 文件**（`/mnt/local/ldh/PhyLLM/Qwen2.5-VL-7B-Instruct`）。
- 已实测 HF 侧 chat template 渲染正确：
  ```
  '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>De'
  video token id = 151656, vision_start = 151652
  ```
- **视频 token 的展开数量由 `grid_thw` 决定**。只要 `pixel_values_videos` 与 `video_grid_thw`
  都从 vLLM 侧捕获（门槛 2），展开结果必然一致。
- **仍需运行时断言**：HF 展开后的 prompt 长度 == vLLM 的 `num_prompt_tokens`。
  **不通过就转路线 A。**

### 门槛 4：权重来源 —— ✅ **通过**

独立加载一份 HF 权重（`from_pretrained`，FP16）。
**不用从 vLLM 导出**（导出要处理 TP 切分/量化，得不偿失）。
代价：多 15.5 GB 显存，已在门槛 1 计入。

---

## 2.0.3 为什么路线 B 的工作量真的小（关键证据）

原实现里**四个核心函数都是独立可调用的**，不是焊死在生成循环里：

| 函数 | 签名 | 复用难度 |
|---|---|---|
| `prefill_prompt` | `std_qwen25vl.py:160` `(model, inputs, past_key_values, video_token_id, output_attentions)` | ✅ 直接调 |
| `build_sparse_selection` | `:185` `(attentions, full_input_ids, video_token_id, text_start, ...)` | ✅ 直接调 |
| `compact_sparse_prompt_cache` | `:302` `(past_key_values, current_length_data, selection)` | ✅ 直接调 |
| **`_draft_tokens`** | `:1042` `(model, first_token, gamma, past_key_values, sparse_controller, start_position)` | ✅ **直接调** |

`_draft_tokens` 尤其关键：它**已经**是一个独立函数，接收
`first_token`（= `propose()` 给的 `sampled_token_ids[-1]`，现成）、
`past_key_values`（proposer 自持）、`sparse_controller`、`start_position`（由 `num_tokens_no_spec` 推出）。
**proposer 只要能备齐这几个参数，就能直接复用用户的 draft 逻辑。**

对照路线 A：要拆 `QKVParallelLinear`、复现 M-RoPE、建私有 paged KV，
**且没有任何已验证的实现可复用** —— 一切数值正确性都要从零自证。

---

## 2.0.4 proposer 的状态机（实现蓝图）

由于 HF KV cache 必须跨 `propose()` 调用保持，proposer 是有状态的：

```
__init__(vllm_config)
   └ 装 Attention.forward 钩子（抓 Q/K/V）+ 装模型 forward 钩子（抓 pixel_values_videos）
   └ 建模块级 registry

load_model(model)                     ← gpu_model_runner.py:5063 调用
   └ 加载 HF 模型（FP16）到 GPU
   └ 保存 vLLM model 引用（备用）

propose(...)  第 1 次（该请求的 prefill 刚结束）
   ├ 从 registry 取 pixel_values_videos / grid_thw
   ├ initialize_past_key_values(hf_model)
   ├ prefill_prompt(hf_model, inputs, pkv, video_token_id, output_attentions=True)
   ├ build_sparse_selection(attentions, ...)      → SparseSelection
   ├ del attentions / selection cache（省显存）
   ├ SparseDraftController(hf_model, selection, sparse_ai_mode="gqa_sdpa").install()
   ├ compact_sparse_prompt_cache(pkv, lengths, selection)
   └ return []   # 第一轮不草稿，先建状态

propose(...)  后续每次
   ├ first_token = sampled_token_ids[i][-1]       ← vLLM 给的
   ├ 把新接受的 token 追加到 HF cache（含 position_ids）
   ├ draft = _draft_tokens(hf_model, first_token, gamma, pkv, controller, start_position)
   └ return draft                                  → vLLM dense 验证
```

**接受率由 vLLM 的 rejection sampler 统计**（`vllm:spec_decode_num_accepted_tokens`），
**不是自数** —— 满足规则 3。

---

## 2.0.5 已知风险（不掩盖）

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| 1 | **HF 与 vLLM 的 logits 不完全一致**（同权重、不同 kernel） | 即使草稿"正确"，也可能被 vLLM 拒绝；接受率**低于** 0.81 | 这正是**要测的东西**；若显著低，在阶段 2.2 归因为"跨框架数值差异" |
| 2 | 显存（37 GB 估算） | OOM | vLLM `gpu_mem_util` 压到 0.35；从 4 帧起步 |
| 3 | 视频 token 展开数不一致 | HF `input_ids` 与 vLLM 不对齐 → 草稿全错 | 运行时断言 `len(hf_input_ids) == num_prompt_tokens`，不过就止损 |
| 4 | HF cache 与 vLLM 已接受前缀不同步 | 草稿基于错误的上下文 | 每轮用 vLLM 的 `token_ids_cpu` 校正 |
| 5 | **每轮 `propose()` 的 HF 前向开销** | 可能比 vLLM 自己还慢 | 本阶段**不测加速**，只测接受率 |

---

## 2.0.6 ⚠️ 本次决策的证据边界（必须说清）

**我做了什么**：只读代码 + 静态核算显存。
**我没做什么**：**没有在任何 GPU 上实跑过路线 B 的任何一步。**
所以四个门槛的判定是**基于代码与静态算术的可行性判断，不是实测结论**。

具体未验证项：
- 门槛 1 的 37 GB 是**算术估算**，未实测（HF 权重实际占用、activations 峰值均未测）
- 门槛 2 的「钩模型 forward 能拿到 pixel_values_videos」**未实跑验证**
- 门槛 3 的「展开数一致」**未在运行时断言过**

**因此本决策的措辞是「走路线 B」，不是「路线 B 已被证明可行」。**
阶段 2.1 的**第一个动作**就是跑一个最小验证：加载 HF 模型 + 捕获 video inputs + 断言
`len(hf_input_ids) == vLLM num_prompt_tokens`。
**该断言失败 → 立即转路线 A**（task 规定的门槛，不做硬撑）。

---

## 2.0.7 阶段 2.0 验收

| 要求 | 结果 |
|---|---|
| 明确写出走 B 还是 A | ✅ **B**，无「两者都可」 |
| 理由有证据 | ✅ 四门槛逐条给代码行号；核心证据是 `prefill_prompt:160-182` 两段式 prefill 与 `_draft_tokens:1042` 可独立调用 |
| 不确定性标注清楚 | ✅ §2.0.6 明确列出三项未实测 |

**→ 进入阶段 2.1：HF draft proposer。**
