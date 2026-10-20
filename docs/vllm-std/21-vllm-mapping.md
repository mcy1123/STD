# 阶段 1：用户 STD → vLLM 注入点映射表

> 状态：**✅ 完成（映射清楚，并标注了一处重大工程量）**
> 日期：2026-09-26
> 依据：`docs/vllm-std/20-user-std-implementation.md` + 只读 vLLM 源码

---

## 1.1 先说结论：`propose` 拿不到 model 的限制**可以绕过**，且有一个现成钩子

任务书要求给出「具体机制，不要泛泛而谈」。**机制已找到并验证：**

**`vllm/v1/worker/gpu_model_runner.py:5060-5063`**

```python
                if hasattr(self, "drafter"):
                    logger.info_once("Loading drafter model...")
                    if hasattr(self.drafter, "load_model"):
                        self.drafter.load_model(self.model)      # L5063 ★
```

**只要自定义 proposer 定义一个 `load_model(self, model)` 方法，vLLM 就会把
「已加载完毕的 target 模型」交给它。**

- **无 method 类型守卫** —— 这段代码对所有 drafter 生效，包括 `custom_class`。
- 对照组：`NgramProposer` 也定义了 `load_model`，但是个 no-op
  （`vllm/v1/spec_decode/ngram_proposer.py:164-166`，注释原文 `# No model to load.`）。
  **这正是 `hasattr` 检查会通过的原因**，也证明这条路径就是给 proposer 拿模型用的。

**由此得到完整的可行链路：**

```
create_custom_proposer(vllm_config)              custom_class_proposer.py:50
        ↓  我的 proposer 构造：装 Attention.forward 钩子
（vLLM 加载 target 模型）
        ↓
self.drafter.load_model(self.model)              gpu_model_runner.py:5063
        ↓  ★ proposer 此刻拿到 model（nn.Module）★
（dense prefill 发生）
        ↓  钩子抓到每层 Q/K/V（线 B 已验证）
propose(sampled_token_ids, num_tokens, token_ids_cpu)   gpu_model_runner.py:4778
        ↓  在同一进程内读钩子缓存
返回 list[list[int]] 草稿
        ↓
vLLM 密集验证 → vLLM 自己的接受率指标
```

**三个前提**仍然必须（线 B 已确立）：`VLLM_ENABLE_V1_MULTIPROCESSING=0`（否则钩子在子进程）、
关 chunked prefill（否则 K 只有分块）、钩子 trace-safe。

---

## 1.2 逐步映射表

| # | 用户实现（`std_qwen25vl.py`） | vLLM 挂载点 | 状态 |
|---|---|---|---|
| ① | `prefill_prompt` L160 返回 `attentions` | 钩子 `Attention.forward`（`attention.py:437`）抓每层 Q/K/V | ✅ **线 B 已验证**：`key_is_full_prompt=True`，8 层可分析 |
| ② | 密集 verifier 前向 → `dense_next` L1218 | **不需要自己做**。vLLM 自己跑 dense prefill/decode，草稿由它验证 | ✅ 天然具备 |
| ③ | `build_sparse_selection` L185，判据 `grouped.sum(dim=(1,2))` L239 | proposer 内自己实现（我有 Q/K/V，判据就是几行 einsum+topk） | ✅ 可移植；阶段 1 复测 2.92 ms/层 |
| ④ | `compact_sparse_prompt_cache` L302：每 head `non_visual ∪ topk[head]` | proposer 内自建**私有**压缩 KV（不动 vLLM 的 paged cache） | ⚠️ 需自己维护，见 §1.4 |
| ⑤ | `_draft_tokens` L1042：γ 步稀疏自回归 | proposer 内自己驱动（见 §1.3） | ⚠️ **本任务最大工程量** |
| ⑥ | `_sparse_aware_attention_forward` L391：无 mask 的稀疏 attention | 自制单 token attention（SDPA + `enable_gqa=True`），**不需要 mask** | ✅ 已确认 `attn_mask=None, is_causal=False`（L450/L452） |
| ⑦ | `_token_argmax` L117：greedy | 同左，`torch.argmax(logits[:, -1, :], -1)` | ✅ 逐字照搬 |
| ⑧ | 接受率 `accepted/proposed` L96 | **vLLM 自己的指标** `vllm:spec_decode_num_accepted_tokens(_per_pos)` | ✅ 更可信：由 vLLM 的 rejection sampler 统计 |

---

## 1.3 最关键的不可直接映射项：稀疏草稿前向（第 ⑤ 步）

用户的实现能跑，是因为它**自己拥有整个 decode 循环**：
模型是 HF 的 `Qwen2_5_VLForConditionalGeneration`，`past_key_values` 是它自己传进去的对象，
`compact_sparse_prompt_cache`（L302）直接原地改写那个对象，然后再调模型。

**在 vLLM 里，这三件事都不成立：**

| 用户实现依赖 | vLLM 的现实 |
|---|---|
| `past_key_values` 是 Python 对象，可原地改写 | KV 是**分页张量**（paged），由 runner 管理，proposer 拿不到 |
| 可以自由调 `model(...)` 跑任意一步 | 单步前向要走 runner 的 `execute_model` 全流程 |
| attention 是 HF `Qwen2_5_VLAttention.forward` | attention 是 vLLM `Attention.forward(query,key,value)`，Q/K/V 已是投影后张量 |

### 两条可选路线（**都需要写代码，工程量不同**）

**路线 1（推荐，工程量中等）：proposer 内部自建 mini-decode**

- 从钩子拿到 prompt 的每层 K/V（原始张量，未分页）
- 按 `non_visual ∪ topk[head]` 逐 head 压缩成私有 cache（= 用户 L302 的等价物）
- 用 `load_model` 拿到的模型的**子模块**（`embed_tokens` / 每层 `self_attn` 的 q/k/v/o_proj / `mlp` / norms）
  手写一个单 token decode step：
  ```
  h = embed(token)
  for layer in model.layers:
      q,k,v = layer.self_attn.{q,k,v}_proj(h)   # vLLM 是 fused qkv_proj，需拆分
      attn = sdpa(q, compact_K[layer], compact_V[layer], enable_gqa=True, is_causal=False)
      h = h + o_proj(attn); h = h + mlp(norm(h))
  logits = lm_head(norm(h)); token = argmax(logits)
  ```
- **风险点**：vLLM 的 Qwen2.5-VL 层用 `QKVParallelLinear`（fused）+ `RowParallelLinear`，
  需要按 TP 切分规则拆权重；单卡 TP=1 时可简化为直接切片。
  **M-RoPE 位置**必须复现（用户实现 L428-430 用 `apply_multimodal_rotary_pos_emb`）。

**路线 2（工程量小，但保真度有损）：proposer 外部用 HF 模型做草稿**

- 在 proposer 里另加载一份 HF Qwen2.5-VL（或复用 `src/specvlm` 的实现）做稀疏草稿
- **缺点**：需要第二份权重（~16 GB），且「稀疏 KV」必须自己重建 —— 这更接近 draft_model 模式，
  **不是 STD**。
- **不推荐**，列出仅为完整性。

> **结论**：路线 1 是唯一忠实于用户 STD 的做法。
> 它是**本任务的主要工程量**，且是阶段 2 的核心内容。

---

## 1.4 KV cache：两套并存，互不干扰

| cache | 归属 | 用途 |
|---|---|---|
| vLLM paged KV | runner 管理 | **dense 验证**（vLLM 原生，不碰） |
| 私有压缩 KV | proposer 自持（Python 张量列表） | **稀疏草稿**（每层 `[1, nkv, compact_len, hd]`） |

**关键设计约束**：**绝不修改 vLLM 的 paged cache**。
用户实现里 `compact_sparse_prompt_cache` 是**原地改写**（L332-333），
在 vLLM 里这么做会**破坏 dense 验证**（而 dense 验证是接受率的唯一裁判）。
→ proposer 必须复制出私有 KV，`gather` 到新张量，不写回。

内存代价：prompt 每层 K/V `[1, nkv, prompt_len, hd]`。
32 帧 / 19164 token / nkv=4 / hd=128 / bf16 → 每层每方向 **19.6 MB**，28 层 × 2 = **1.1 GB**。
可接受（A6000 48 GB）。

---

## 1.5 不可映射 / 需要改判据的地方（诚实清单）

| 项 | 用户实现 | vLLM 方案 | 影响 |
|---|---|---|---|
| 注意力权重的来源 | HF `output_attentions=True`（eager SDPA 返回矩阵） | 钩子捕获 Q/K/V 后**自己重建** post-softmax 概率 | ✅ 线 B 已数值校验通过（mass check 精确） |
| `draft_layers` 提前退出（L690） | `model.model.layers[:N]` | 手写 decode 时可自由只跑前 N 层 | ✅ 更容易 |
| `triton_gqa`（batch=1 only, L438） | 仓库自带 Triton kernel | 可用，但阶段 2 先用 `gqa_sdpa` 保正确性 | ⚠️ 性能选项，后置 |
| 多模态视觉编码器 | HF 全流程 | **不需要重跑**：钩子已拿到语言层的 K/V，视觉编码已在 vLLM dense prefill 中完成 | ✅ 反而更简单 |
| 接受率统计 | 自己数 `accepted/proposed` | **用 vLLM 指标** | ✅ 更可信（第三方裁判） |

---

## 1.6 阶段 1 验收

| 要求 | 结果 |
|---|---|
| 映射表完整，每行有代码位置 | ✅ §1.2 八步，每步给出两侧文件行号 |
| 明确标注不可直接映射的部分 | ✅ §1.3（稀疏前向）、§1.5（五项） |
| `propose` 拿不到 model/KV 的绕过机制**具体** | ✅ `load_model` 钩子（`gpu_model_runner.py:5063`，已验证无类型守卫） |
| 不确定性标注清楚 | ✅ §1.3 两条路线、§1.4 内存代价、§1.5 风险点 |

---

## 1.7 阶段 2 的实现计划（尚未开工，待本阶段确认）

**最小可用 GPU proposer（单 token 草稿）**：

1. `StdGpuProposer(VllmConfig)`：
   - 构造时 `types.MethodType` 替换 `Attention.forward`，捕获每层 Q/K/V
   - 定义 `load_model(self, model)` → 保存 `self.model` 引用 ✅ 关键一步
2. prefill 结束后：`build_sparse_selection` 等价实现（判据 L239 逐字照搬）
3. 建私有压缩 KV（逐 head `non_visual ∪ topk[head]`）
4. 手写单 token decode step（路线 1），**γ=1**（只草稿 1 个 token，先跑通链路）
5. `propose()` 返回 `[draft]`，由 vLLM dense 验证
6. **读 vLLM 指标得接受率**，与用户实现 0.507–1.000 对照

**验收**：接受率 > 0 且 ≥ 50%。若 < 50%，先定位原因再进阶段 3。

**止损**：若手写 decode step 无法在合理成本内复现用户实现的 logits，
则判定为「vLLM 集成不可行」，记录根因，回退到路线 2 或转 EAGLE-3。
