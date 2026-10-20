# 阶段 2：vLLM 投机解码接口分析

> 状态：**✅ 完成（GO）**
> 日期：2026-09-26
> 代码基线：**vLLM v0.22.0**，已安装于 `site-packages/vllm`，只读镜像树 `third_party/vllm`
> 本文所有行号均指 v0.22.0 源码；引用路径可用 `third_party/vllm/<path>` 直接核对

---

## 2.1 一句话结论

**多模态 target 能配合自定义 Proposer，且不需要任何补丁。**

因为 `method="custom_class"` 这条路径**根本不加载 draft model**：配置阶段就把
`draft_model_config = target_model_config` 拱手让过，运行时用 `create_custom_proposer()`
直接导入用户类。而「多模态 draft 不支持」的两道闸门
（`_warn_if_multimodal` / `_raise_if_mrope`）**都长在 `LLMBaseProposer.__init__` 里**，
`custom_class` 从不实例化这个类 → **闸门根本不执行**。

---

## 2.2 v0.22.0 支持的全部投机解码方法

`vllm/config/speculative.py:59`

```python
SpeculativeMethod = Literal[
    "ngram", "medusa", "mlp_speculator", "draft_model", "suffix", "custom_class",
    EagleModelTypes,   # eagle / eagle3 / extract_hidden_states / mtp / dflash ...
    NgramGPUTypes,     # ngram_gpu
]
```

| method | 是否需要 draft 权重 | 多模态 target 可用？ | 备注 |
|---|---|---|---|
| `draft_model` | 需要（独立小模型） | ⚠ **M-RoPE 时硬失败** | 见 §2.6 |
| `eagle`/`eagle3` | 需要 draft head 权重 | 同上 | 本项目无对应权重，**不作为主线** |
| `ngram` / `ngram_gpu` | 不需要 | ✅ | 纯 CPU / GPU n-gram 匹配 |
| `suffix` | 不需要 | ✅ | 后缀树 |
| `medusa` / `mlp_speculator` | 需要额外头 | 同上 | |
| **`custom_class`** | **不需要** | ✅ **本方案主线** | 用户自备 `propose()` |

---

## 2.3 注入点全链路（这是本阶段的核心交付）

### ① 配置：怎么声明

`vllm/config/speculative.py`

```python
# L527-536 —— method 可以省略，vLLM 会自动识别
if (self.model is not None
        and "." in self.model
        and not self.model.startswith(("http://", "https://", "file://"))
        and "/" not in self.model):      # 不是 HF repo (org/model)
    self.method = "custom_class"         # 自动判定
```

```python
# L620-632 —— custom_class 分支：不加载 draft model
elif self.method == "custom_class":
    logger.warning_once("Using a custom class-based proposer backend. This is an "
                        "experimental feature and the proposer interface is "
                        "subject to breaking changes in future vLLM releases.")
    self.prompt_lookup_max = 0
    self.prompt_lookup_min = 0
    self.draft_model_config = self.target_model_config      # <-- 关键：不新增模型
    self.draft_parallel_config = self.target_parallel_config
```

**配置示例**（Python API）：

```python
from vllm import LLM
llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    speculative_config={
        "method": "custom_class",
        "model": "std_vllm.proposer.StdProposer",   # 模块路径.类名
        "num_speculative_tokens": 4,                # 草稿长度 k
    },
)
```

命令行等价形式：

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --speculative-config '{"method":"custom_class","model":"std_vllm.proposer.StdProposer","num_speculative_tokens":4}'
```

> 注意：`model` 字段在这里**不是模型名**，而是 **Python 类路径**。这也是 `custom_class` 自动识别规则的依据。

### ② 实例化

`vllm/config/speculative.py` → 运行时由 `vllm/v1/spec_decode/custom_class_proposer.py:12` 完成：

```python
def create_custom_proposer(vllm_config: VllmConfig):
    backend = spec_config.model                     # "pkg.mod.Cls"
    module_path, class_name = backend.rsplit(".", 1)
    module = importlib.import_module(module_path)
    user_class = getattr(module, class_name)
    instance = user_class(vllm_config)              # 构造函数必须收 VllmConfig
    if not callable(getattr(instance, "propose", None)):
        raise AttributeError(...)
    return instance
```

调用点：`vllm/v1/worker/gpu_model_runner.py:551`

```python
if self.speculative_config.method == "custom_class":
    self.drafter = create_custom_proposer(self.vllm_config)
```

**硬性要求**（三条，任一不满足即报错）：

| 要求 | 不满足时的报错 |
|---|---|
| `model` 必须含 `.` | `ValueError: Invalid custom proposer module path ...` |
| 模块可导入且类存在 | `ImportError` / `AttributeError: Module ... has no attribute ...` |
| 构造函数接受 `VllmConfig` | `RuntimeError: Failed to instantiate custom proposer class ... The class constructor must accept VllmConfig as argument.` |
| 实例有可调用的 `propose` | `AttributeError: ... must have a 'propose' method` / `... is not callable` |

### ③ 每步调用（真正的注入点）

`vllm/v1/worker/gpu_model_runner.py:4750` 定义、`:4778` 分派：

```python
elif spec_config.method == "custom_class":
    assert isinstance(sampled_token_ids, list)
    draft_token_ids = cast(Any, self.drafter).propose(
        sampled_token_ids,                      # list[list[int]]
        self.input_batch.num_tokens_no_spec,    # np.ndarray (num_reqs,)
        self.input_batch.token_ids_cpu,         # np.ndarray (max_num_reqs, max_model_len)
        slot_mappings=slot_mappings,            # dict | list[dict] | None —— CPU proposer 不用
    )
```

**因此需要实现的签名是：**

```python
class StdProposer:
    def __init__(self, vllm_config: VllmConfig) -> None: ...

    def propose(
        self,
        sampled_token_ids: list[list[int]],   # 上一步每个请求被接受的 token（被拒则为空 list）
        num_tokens_no_spec: np.ndarray,       # 每请求「不含草稿」的 token 数，shape (num_reqs,)
        token_ids_cpu: np.ndarray,            # 每请求完整 token 序列，shape (max_num_reqs, max_model_len)
        slot_mappings=None,                   # unused
    ) -> list[list[int]]:                     # 每请求一个草稿 token 列表
        ...
```

与 `NgramProposer.propose` 完全同构（`vllm/v1/spec_decode/ngram_proposer.py:131`）——后者是最佳参考实现。

### ④ 返回值怎么被消费

- 返回 `list[list[int]]`，**外层长度 = batch 内请求数**。
- 每个内层列表的长度**可以不同**，也**可以少于** `num_speculative_tokens`，甚至可以返回 `[]`（该请求本轮不做投机）。
  `gpu_model_runner.py:2148` 用的是 `draft_len = len(draft_token_ids)`，直接按实际长度记账。
  → **这一点对 STD 至关重要**：稀疏草稿天然是变长的，框架允许。
- 返回 `[]`（空列表）等价于「跳过投机解码」，也就是 dummy proposer 的合法形态。

---

## 2.4 关键输入字段的形状与语义

来源 `vllm/v1/worker/gpu_input_batch.py`：

```python
L133  self.token_ids_cpu_tensor = torch.zeros((max_num_reqs, max_model_len), dtype=torch.int32)
L139  self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()          # 是 torch 张量的 numpy 视图！
L148  self.num_tokens_no_spec_cpu_tensor = torch.zeros(max_num_reqs, dtype=torch.int32)
L154  self.num_tokens_no_spec = self.num_tokens_no_spec_cpu_tensor.numpy()
```

| 字段 | 类型 | 语义 |
|---|---|---|
| `token_ids_cpu[i, :num_tokens_no_spec[i]]` | `np.ndarray` int32 | 请求 i 的**完整** token 序列（prompt + 已生成） |
| `num_tokens_no_spec[i]` | `np.int32` | 请求 i 的**有效**长度（**必须用它切片**，其余是 padding） |
| `sampled_token_ids[i]` | `list[int]` | 上一步请求 i 被**接受**的 token；为空表示该请求本轮跳过 |

> ⚠ 坑：`token_ids_cpu` 是 **torch 张量的 numpy 视图**（非拷贝）。只读没问题，**绝对不要原地修改**，否则会污染 input batch。

参考 `NgramProposer.propose` 的两个前置过滤（应当照抄）：
1. `if not len(sampled_ids): continue` —— 上一步被拒的请求跳过；
2. `if num_tokens >= self.max_model_len: continue` —— 已达上限的请求跳过。

---

## 2.5 报错边界（实测/静态确认）

| 场景 | 结果 | 位置 |
|---|---|---|
| `method` 省略且 `model` 含 `.` | 自动判为 `custom_class` | `speculative.py:527` |
| `method="custom_class"` 但 `model=None` | `ValueError: method='custom_class' requires 'model' to contain the custom proposer module path` | `speculative.py:571` |
| 类路径不含 `.` | `ValueError: Invalid custom proposer module path` | `custom_class_proposer.py:29` |
| 构造函数不收 `VllmConfig` | `RuntimeError: ... The class constructor must accept VllmConfig as argument.` | `custom_class_proposer.py:52` |
| 无 `propose` | `AttributeError: ... must have a 'propose' method` | `custom_class_proposer.py:58` |
| 启动时给该 method 传了 draft 权重相关字段 | 无影响（`draft_model_config` 被指向 target） | `speculative.py:628` |

---

## 2.6 `draft_model` 路径为什么跑不了多模态（阶段 7 对照用）

任务书说「当前该路径会 `NotImplementedError`」。实测 v0.22.0 的真实情况**更微妙**，分两层：

**第一层：多模态 draft —— 只是 warning**

`vllm/v1/spec_decode/llm_base_proposer.py:301`

```python
def _warn_if_multimodal(self):
    if self.supports_mm_inputs:
        logger.warning("Speculative Decoding with draft models or parallel drafting "
                       "does not fully support multimodal models yet. "
                       "Proceeding with text-only speculative decoding.")
```

**第二层：M-RoPE —— 这才是硬错**

`vllm/v1/spec_decode/llm_base_proposer.py:309`

```python
def _raise_if_mrope(self):
    if self.draft_model_config.uses_mrope:
        raise NotImplementedError("Speculative Decoding with draft models or "
                                  "parallel drafting does not support M-RoPE yet")
```

两道闸门**都在 `LLMBaseProposer.__init__` 里被调用**（`llm_base_proposer.py:197-198`）：

```python
self._warn_if_multimodal()
self._raise_if_mrope()
```

而 **Qwen2.5-VL 正是 M-RoPE**。其 `config.json`：

```json
"rope_scaling": { "type": "mrope", "mrope_section": [16, 24, 24] }
```

`uses_mrope()`（`vllm/transformers_utils/config.py:512`）据此判定为 `True`。

> **所以：用 `draft_model` 配 Qwen2.5-VL，会死在 `_raise_if_mrope` 的 `NotImplementedError`（M-RoPE），而不是死在「多模态」本身。**
> 这个精确区分很重要：它意味着**只要换掉 proposer 实现、不实例化 `LLMBaseProposer`，同一个多模态 target 就能做投机解码**——这正是 `custom_class` 做到的。
> 阶段 7 会用真实运行验证这条边界。

---

## 2.7 如果 `custom_class` 不存在，最小 patch 是什么？

本机最终不需要（v0.22.0+cu129 已跑通）。但作为**备选方案的量化**记录在此：若被迫降到 vLLM 0.19.1（最后一个 CUDA-12 原生版本，torch 2.10/cu128），移植面是 **2 个文件、3 处改动**：

| 文件 | 改动 |
|---|---|
| `vllm/config/speculative.py` | ①`SpeculativeMethod` 加入 `"custom_class"`；②加 `elif self.method == "custom_class":` 分支，设 `draft_model_config = target_model_config`（照抄 v0.22.0 L620-632） |
| `vllm/v1/worker/gpu_model_runner.py` | ③`drafter` 构造处加 `if method == "custom_class": self.drafter = create_custom_proposer(...)`；④`propose_draft_token_ids` 加同名分派分支 |
| `vllm/v1/spec_decode/custom_class_proposer.py` | **新增**（73 行，直接移植 v0.22.0） |

风险：0.19.1 与 0.22.0 的 `gpu_model_runner` 结构差异较大（0.22.0 已把 proposer 拆到 `v1/worker/gpu/spec_decode/`），移植后需重新验证。

**已明确否决的替代路线**：升级驱动到 ≥580 以使用 cu13 wheel。需要 root，且本机 6 张卡正在跑他人作业，重载驱动会打断他人任务（违反硬性规则 8）。

---

## 2.8 对 STD 实现的直接含义

1. **稀疏草稿天然可表达**：`propose` 返回变长 `list[list[int]]`，框架按实际长度记账 → STD 的「只草稿 top-K 稀疏位置」不需要改框架。
2. **密集验证无需改动**：验证永远是 target 模型自己做，接受/拒绝由 vLLM 原生 rejection sampler 决定 → **bit-exact 是设计上就成立的**，不是碰运气。
3. **多模态不受限**：target 的 M-RoPE 位置由 `gpu_model_runner._init_mrope_positions` 自己维护，与 proposer 无关。
4. **注意 `slot_mappings`**：对 CPU proposer 无用（ngram 也标 `# unused`），STD 如果要做「基于 KV/attention 的稀疏选择」才需要它——**这是 STD 稀疏信号能否上车的分水岭**，留待阶段 4 判定。

---

## 2.9 验收

| 判据 | 结果 |
|---|---|
| 有具体文件路径 | ✅ 全部标注 `文件:行号` |
| 有类名 | ✅ `create_custom_proposer` / `StdProposer` / `NgramProposer` / `LLMBaseProposer` |
| 有配置示例 | ✅ Python API + CLI 两种 |
| 有报错边界 | ✅ §2.5 五条 + §2.6 draft_model 精确死因 |
| 明确回答「多模态 target 能否配合自定义 Proposer」 | ✅ **能**，且给出结构性理由 |

**→ GO，进入阶段 3（Dummy Proposer 接通）。**
