# STD 解码成本与加速比

> 2026-09-22 建立。记录 STD 的**基线口径、实测加速比、代价模型与场景分析**。

---

## 0. 基线口径

**dense 基线 = 模型原始实现**：`src/specvlm/models/modeling_qwen2_5_vl.py` 的
`Qwen2_5_VLSdpaAttention.forward`，即 `attn_implementation="sdpa"` 加载方式下的默认路径，
**batch = 1**（`verify_tensor = torch.tensor([verify_input])`，样本串行）。

**固定配置**：Qwen2.5-VL-7B · 128 帧 · 512 输出 token · γ=9 · `k_plus_text=1024` ·
`max_pixels=200704` · `--sparse-attn-mode gqa_sdpa`（默认）· A6000 空闲卡（`timing_valid=true`）。

---

## 1. 实测加速比

| 量 | 含义 | 值 |
|---|---|---:|
| `c_d` | dense AR 一步（`decoding_time / generate_len`） | 67.44 ms |
| `c_s` | 稀疏草稿一步（`draft_time / (rounds·γ)`） | 27.40 ms |
| `c_v` | dense 验证一遍（`verify_time / rounds`） | 68.81 ms |
| `c_bonus` | 稀疏 bonus 一步（`bonus_time / rounds`） | 27.25 ms |

### **加速比 = 1.82×**

（`c_d` 汇总 103.6 s vs `c_s` 侧汇总 56.9 s，3 样本 × 512 token，`timing_valid=true`）

**内部一致性**：`c_bonus`（27.25 ms）与 `c_s`（27.40 ms）由两条独立路径测得却几乎相等，
说明「一次稀疏前向 ≈ 27 ms」这个标定可信。

---

## 2. 代价模型

一轮 STD 的结构（`src/std_repro/std_qwen25vl.py`）：

```
γ 次稀疏草稿前向  +  1 次稀疏 bonus 前向  +  1 次 dense 验证前向
=> 产出 (accept_len + 1) 个 token
```

令 `A(α, γ) = Σ_{i=1..γ} α^i`（期望接受草稿数），`T = 1 + A`，`C = (γ+1)·c_s + c_v`：

```
S(γ, α) = T · c_d / C
```

| 输入 | 值 |
|---|---:|
| 实测 `mean_accept_length` | 8.278 / γ=9 |
| 位置接受率 | 0.9227 |
| 反解 α | 0.9832 |
| **预测 S** | **1.8251** |
| **实测 S** | **1.8202** |
| 误差 | **0.27%** |

模型误差 0.27%，可直接用于外推。

---

## 3. 100% 接受率的天花板

令 α=1（草稿全部被接受），每轮产出 γ+1 个 token：

```
S_100(γ) = (γ+1)·c_d / [(γ+1)·c_s + c_v]        γ→∞ 时 → c_d / c_s = 1/rho
```

以本配置标定（`rho = c_s/c_d = 0.406`）：

| γ | S at α=1 |
|---:|---:|
| 1 | 1.091× |
| 5 | 1.735× |
| **9（现用）** | **1.967×** |
| 15 | 2.127× |
| ∞ | **2.461×** |

**接受率这条轴已接近饱和**：现役 α=0.98 对应 1.82×，即使草稿 100% 被接受也只到 1.967×。

**真正的杠杆是 `rho = c_s/c_d`**：

| `rho` | S_100(γ=9) | 天花板 1/rho |
|---:|---:|---:|
| 0.20 | 4.55× | 5.00× |
| 0.30 | 3.03× | 3.33× |
| **0.41（现役）** | **2.24×** | **2.46×** |
| 0.60 | 1.52× | 1.67× |
| 0.80 | 1.14× | 1.25× |

即：**要让解码更快，应该去压 `c_s`（让稀疏草稿一步更便宜），而不是去提接受率。**

> 复现：`python scripts/analysis/spec_ceiling_sim.py --run results/attn_mode_ab/gqa_sdpa.jsonl`
> 报告见 `results/spec_ceiling_sim/report.md`。

---

## 4. batch 是比视频长度更强的杠杆

**实测事实**：`c_s ≈ 27 ms` 是一个 7B fp16 模型在 batch 1 下**流式读取权重的时间地板**
（`c_bonus` 与 `c_s` 独立测得的吻合证实了这一点）。KV 访问只是在这个地板上叠加。

- 权重流量（7B fp16 ≈ 16 GB）**每一步都要付一次，但与 batch 无关**；
- KV 流量**每序列都要付，随 batch 线性增长**。

所以每序列每步成本 ≈ `W/B + KV_per_seq`。**batch 越大，权重被摊得越薄，稀疏省下的那部分占比越高。**

对比原论文 `Sparse-to-Dense`（ACL 2025, Zhang et al.）§4 的实现细节：

| 项 | 论文 | 本仓库 |
|---|---|---|
| 模型 | LLaVA-OneVision-7B / Qwen2-VL-7B | Qwen2.5-VL-7B |
| **batch size** | **8** | **1** ⚠️ |
| `mt + K` | 1024 | 1024 ✅ |
| γ | 9 | 9 ✅ |
| 硬件 | A100-80GB | A6000 / A100 |
| 提示 | CoT | CoT ✅ |
| 时长动机 | **720 帧 = 141,120 视觉 token** | 128 帧 ≈ 15,876 |

论文报道（Table 1，batch=8）：MLVU 1.72×/1.94×，VideoMME-s 1.82×/1.71×、m 1.83×/1.62×、**l 1.59×/1.70×**。

**论文的 I/O 模型（§3.3）只数 KV 字节，不数模型权重**：

```
I/O_avg     = [γ(K + mt) + mv + mt] / (α·γ)
I/O_vanilla = mv + mt
成立条件：α > (K + mt)/(mv + mt) + γ⁻¹
```

该式是 per-request 记账，**没有 batch 项**，隐含假设「KV 是唯一瓶颈」。
这只在权重被 batch 摊薄之后才近似成立 —— 这是同一套方法在 batch=8 下能兑现 1.94×、
在 batch=1 下明显偏低的原因。

**两条操作结论：**

1. **要复现论文量级的加速，最小改动是把 batch 从 1 提到 8，而不是继续加帧数。**
   加帧数同时抬高视觉 token 处理与 prefill 成本，收益远不如提 batch。
2. 若继续做「动态路由 / 验证信号」这条线，**目标场景应定位为吞吐 / serving（大 batch + 长上下文）**，
   即论文与 Vegas 共同所在的区域；而不是当前 batch=1 的单视频低延迟。

---

## 5. 复现

```bash
cd /home/mcy/Projects/SpecVLM/STD

# 加速比（driver 自带空闲卡门控）
TAG=gqa_sdpa GPU=<IDLE> LIMIT=3 MAX_NEW_TOKENS=512 FRAME_NUM=128 REPEATS=1 \
OUT_DIR=$PWD/results/attn_mode_ab \
CONFIGS_OVERRIDE="gqa_sdpa|v2|1|0.00|--sparse-attn-mode=gqa_sdpa" \
bash scripts/run_dynamic_matrix_local.sh gqa_sdpa

# 代价模型 / 天花板
python scripts/analysis/spec_ceiling_sim.py \
  --run results/attn_mode_ab/gqa_sdpa.jsonl
```

> ⚠️ `CONFIGS_OVERRIDE` 的数组展开走 word-splitting，第 5 个字段**不能含空格**，
> 因此用 `--opt=value` 等号形式。

---

## 6. 原始数据

| 路径 | 内容 |
|---|---|
| `results/attn_mode_ab/gqa_sdpa.jsonl` | 加速比与代价模型的主标定（3 样本 / 512 token / 128 帧） |
| `results/attn_mode_ab/gqa_sdpa_report.md` | 上述 run 的汇总报告 |
| `results/spec_ceiling_sim/report.md` | 天花板报告 |
| `results/spec_ceiling_sim/calibration.json` | 标定结果（机器可读） |

（`results/` 已在 `.gitignore` 中，不入库。）
