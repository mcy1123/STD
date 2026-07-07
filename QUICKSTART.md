# STD 复现傻瓜指南

把代码拷到 A100 服务器上，按顺序执行下面几步，直接出结果。

---

## 第 0 步：拷代码

```bash
# 在本地机器上打包
tar czf std.tar.gz -C /home/mcy/projects Std/src Std/scripts
tar czf specvlm.tar.gz -C /home/mcy/projects SpecVLM --exclude=datasets --exclude=results --exclude=.git

# 传到 A100
scp std.tar.gz specvlm.tar.gz user@a100-server:/data/

# 在 A100 上解压
cd /data
tar xzf std.tar.gz
tar xzf specvlm.tar.gz
```

解压后的目录结构：

```text
/data/
├── Std/
│   ├── src/std_repro/
│   └── scripts/
└── SpecVLM/
    ├── models/
    ├── kv_cache/
    └── utils/
```

---

## 第 1 步：装环境（一次性）

```bash
conda create -n specvlm python=3.10 -y
conda activate specvlm

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.48.0 datasets accelerate "numpy<2.0"
pip install qwen-vl-utils==0.0.10 av==14.0.0 triton==3.2.0
pip install huggingface_hub
```

验证：

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.device_count())"
# 应该输出：NVIDIA A100-SXM4-80GB 和 GPU 数量
```

---

## 第 2 步：下载模型（一次性）

```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir /data/models/Qwen2.5-VL-7B-Instruct
```

约 16 GB，几分钟。

---

## 第 3 步：登录 HuggingFace + 申请 MLVU 访问（一次性）

```bash
huggingface-cli login
# 输入你的 HF token（去 https://huggingface.co/settings/tokens 创建）
```

然后浏览器打开 https://huggingface.co/datasets/MLVU/MVLU ，点 "Access repository" 申请访问，填写姓名、机构、用途（填 Research）。

---

## 第 4 步：下载 MLVU 数据集 + 视频（一次性）

```bash
cd /data/Std

# 下载 MLVU annotation（json 文件）
conda run -n specvlm python scripts/prepare_std_datasets.py --dataset MLVU

# 下载 MLVU 视频（从 YouTube，需要较长时间）
# MLVU 的视频 URL 在 json annotation 里，用 yt-dlp 下载
pip install yt-dlp

python -c "
import json, os, subprocess
from pathlib import Path

json_dir = Path('datasets/MLVU/MLVU/json')
video_dir = Path('datasets/MLVU/videos')
video_dir.mkdir(parents=True, exist_ok=True)

for json_file in sorted(json_dir.glob('*.json')):
    data = json.loads(json_file.read_text())
    for item in data:
        url = item.get('video') or item.get('video_path') or ''
        video_id = item.get('video_id', '')
        if not url or not video_id:
            continue
        out_path = video_dir / f'{video_id}.mp4'
        if out_path.exists():
            continue
        print(f'Downloading {video_id}: {url}')
        subprocess.run([
            'yt-dlp', '-f', 'best', '-o', str(out_path),
            '--no-playlist', url
        ], check=False)
"
```

如果下载慢或者部分视频失效，可以先跑有视频的部分看看结果，不用等全部下完。

---

## 第 5 步：跑实验

论文配置：`K+text=1024`，`γ=9`，CoT prompt，batch_size=8。
当前脚本内部的 STD 生成器仍是单 sample 解码，因此这里跑的是
batch_size=1 的论文超参配置；不要把它当作论文表 1 的 batch=8 结果。

```bash
cd /data/Std

SPECVLM_MAX_CACHE_LEN=163840 conda run -n specvlm python scripts/benchmark_std.py \
  --model-path /data/models/Qwen2.5-VL-7B-Instruct \
  --dataset MLVU \
  --data-path MLVU/MVLU \
  --video-root /data/Std/datasets/MLVU/videos \
  --eval-num 10 \
  --frame-num 256 \
  --max-new-tokens 128 \
  --gamma 9 \
  --target-k-plus-text 1024 \
  --prompt-style cot \
  --sparse-attn-mode triton_gqa \
  --gpu-ids 0 \
  --output results/mlvu_256f_kplus1024_g9.jsonl
```

---

## 第 6 步：看结果

```bash
conda run -n specvlm python scripts/summarize_metrics.py results/mlvu_256f_kplus1024_g9.jsonl
```

输出示例：

```text
samples: 10
speedup: 1.xxx
acceptance_rate: 0.xxx
mean_accept_length: x.xxx
ar_decoding_time: x.xxxs
std_decoding_time: x.xxxs
```

**看 `speedup` 就行**，>1.0 说明 STD 比原版 AR 解码快。

---

## 参数说明

论文核心就三个参数，想调可以调：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--target-k-plus-text 1024` | 1024 | 稀疏 KV 保留量，越小越快但接受率越低 |
| `--gamma 9` | 9 | 每轮 draft 的 token 数 |
| `--frame-num 256` | 256 | 视频采样帧数，MLVU 视频长建议 256+ |

如果 OOM，先降帧数：`--frame-num 128`。如果显存充裕，升到 384 或 512，加速比通常更好。
