# STD 复现 — 傻瓜指南

从头到尾，一个脚本搞定环境，几条命令出结果。

---

## 前置条件

- Linux 服务器（A100 80GB 或 H100 推荐）
- 已安装 conda
- HuggingFace 账号（用于下载模型和数据集）

---

## 第 1 步：克隆代码

```bash
git clone <this-repo-url> STD
cd STD
```

**注意**：本仓库已自包含 SpecVLM 依赖，无需额外 clone 任何仓库。

---

## 第 2 步：装环境（一次性，约 5 分钟）

```bash
bash scripts/setup_env.sh
```

这会创建 `specvlm` conda 环境并安装所有依赖。

验证：

```bash
conda run -n specvlm python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Count: {torch.cuda.device_count()}')
"
```

---

## 第 3 步：下载模型（一次性，约 16 GB）

```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir models/Qwen2.5-VL-7B-Instruct
```

---

## 第 4 步：下载 Video-MME 数据集（一次性）

```bash
# 下载 metadata + 第一批视频
conda run -n specvlm python scripts/prepare_std_datasets.py --dataset Video-MME

# 解压视频
conda run -n specvlm python scripts/extract_videomme_chunks.py --chunks 01

# 从 YouTube 补充下载（可选，需要 yt-dlp）
pip install yt-dlp
conda run -n specvlm python scripts/download_videomme_youtube.py \
  --duration short --limit 20
```

---

## 第 5 步：Smoke Test（验证链路）

```bash
SPECVLM_MAX_CACHE_LEN=40960 conda run -n specvlm python scripts/benchmark_std.py \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --dataset Video-MME \
  --data-path datasets/Video-MME \
  --video-root datasets/Video-MME/videos \
  --eval-num 3 \
  --frame-num 32 \
  --max-new-tokens 64 \
  --gamma 9 \
  --target-k-plus-text 1024 \
  --prompt-style cot \
  --gpu-ids 0
```

输出 JSONL 指标到 `results/std_qwen2_5_vl_7b/`。

---

## 第 6 步：跑论文配置

```bash
# 256 帧，论文 K+text=1024, gamma=9, CoT
SPECVLM_MAX_CACHE_LEN=163840 conda run -n specvlm python scripts/benchmark_std.py \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --dataset Video-MME \
  --data-path datasets/Video-MME \
  --video-root datasets/Video-MME/videos \
  --eval-num 30 \
  --frame-num 256 \
  --max-new-tokens 128 \
  --gamma 9 \
  --target-k-plus-text 1024 \
  --prompt-style cot \
  --sparse-attn-mode triton_gqa \
  --gpu-ids 0
```

---

## 第 7 步：查看结果

```bash
conda run -n specvlm python scripts/summarize_metrics.py \
  results/std_qwen2_5_vl_7b/*.jsonl
```

输出示例：

```text
samples: 30
token_equal: 30/30
speedup: 1.xxx
acceptance_rate: 0.xxx
mean_accept_length: x.xxx
```

**看 `speedup` 就行** — >1.0 说明 STD 比原版 AR 解码快。

---

## 参数速查

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--frame-num 256` | 32 | 视频帧数，越高稀疏收益越大 |
| `--target-k-plus-text 1024` | 1024 | 稀疏 KV 预算，越小越快但接受率越低 |
| `--gamma 9` | 9 | 每轮 draft 的 token 数 |
| `--max-new-tokens 128` | 64 | 最大生成 token 数 |
| `--sparse-attn-mode triton_gqa` | gqa_sdpa | H100/A100 推荐 triton_gqa |
| `--gpu-ids 0` | 0,1,2,3 | A100/H100 单卡即可 |

**如果 OOM**：降帧数 `--frame-num 128`。**如果显存充裕**：升到 384 或 512。

---

## 环境变量

- `SPECVLM_MAX_CACHE_LEN` — KV cache 预分配长度：
  - 32 帧 → `40960`
  - 256 帧 → `163840`
  - 384 帧 → `245760`

---

## 可选：MLVU 数据集

MLVU 需要手动申请 HuggingFace 访问权限：
1. `huggingface-cli login`
2. 访问 https://huggingface.co/datasets/MLVU/MVLU 点 "Access repository"

```bash
conda run -n specvlm python scripts/prepare_std_datasets.py --dataset MLVU
# 视频需要从 YouTube 手动下载
```
