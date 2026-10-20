# 阶段 0：环境勘察与基线

> 状态：**✅ 完成（GO）**
> 日期：2026-09-26
> 主机：`A6000`（ECNU 本机，非远程）

---

## 0.1 任务书前提的三处修正（先读这个）

任务书对环境的假设与本机实测不符。先把差异摆在最前面，后面所有决策都基于实测。

| # | 任务书假设 | 本机实测 | 影响 |
|---|---|---|---|
| 1 | 需要找一台 GPU 机器（暗指远程） | **本机就是 A6000 环境**：hostname `A6000`，6×RTX A6000 48GB + 2×RTX 4090 24GB，驱动 550.163.01 | 主战场定在本机，不走远程 |
| 2 | `git clone` vLLM 到 `third_party/vllm` | `github.com` 的 git 协议/HTTPS **超时**（`git clone` exit 124）；`codeload.github.com` 可用 | 改用 **release tarball + `git init`** 落地固定版本 |
| 3 | `draft_model` 跑多模态会 `NotImplementedError` | v0.22.0 里多模态**只是 warning**（`_warn_if_multimodal`）；真正的硬错是 **M-RoPE**（`_raise_if_mrope`）。Qwen2.5-VL 正是 M-RoPE | 阶段 2/7 用实测确认，而非引用假设 |

另：**远端 gpu23 不作为本次执行环境**。理由：① gpu23 无外网（`ping 8.8.8.8` → Network unreachable，DNS 失败），装不了依赖；② gpu23 驱动 535.154.05 比本机更旧，CUDA 13 问题只会更严重；③ 用户明确要求「最好用 a6000 的环境」。

---

## 0.2 主机与驱动（实测）

| 项 | 值 | 采集命令 |
|---|---|---|
| hostname | `A6000` | `hostname` |
| kernel | `Linux A6000 6.2.0-26-generic #26~22.04.1-Ubuntu SMP ... x86_64` | `uname -a` |
| CPU 核数 | 96 | `nproc` |
| 内存 | 503 GiB total / 385 GiB available | `free -g` |
| GPU | 0–3,6,7 = **RTX A6000 48GB (sm_86)**；4,5 = **RTX 4090 24GB (sm_89)** | `nvidia-smi --query-gpu=...` |
| 驱动 | **550.163.01**（= CUDA 12.4 上限） | `nvidia-smi --query-gpu=driver_version` |
| 系统 nvcc | 11.8（`/usr/local/cuda`）；另装 12.0 / 12.1 / 12.6 | `nvcc --version` |
| 本地模型 | `/mnt/local/ldh/PhyLLM/Qwen2.5-VL-7B-Instruct`（5 shard safetensors，完整） | 见 0.4 |
| 数据集 | `datasets/Video-MME` → `/mnt/local/mcy/home-offload/datasets/Video-MME`，30 个 mp4 | `find ... -name '*.mp4' \| wc -l` |

### 关键约束：CUDA 12.4 驱动

驱动 550.163.01 只支持到 **CUDA 12.4**。这**直接决定了 vLLM 版本的可选范围**——见 0.3，这是本阶段最重要的发现。

### GPU 占用（共享机，**不杀任何进程**）

采集时刻（`nvidia-smi`）：

```
index, name,                memory.total, memory.used, memory.free, util
0, NVIDIA RTX A6000,        49140 MiB,    44106 MiB,   4565 MiB,    100 %
1, NVIDIA RTX A6000,        49140 MiB,    42801 MiB,   5869 MiB,    100 %
2, NVIDIA RTX A6000,        49140 MiB,    23 MiB,      48648 MiB,   0 %   <-- 全空
3, NVIDIA RTX A6000,        49140 MiB,    36324 MiB,   12347 MiB,   98 %
4, NVIDIA GeForce RTX 4090, 24564 MiB,    7504 MiB,    16707 MiB,   27 %
5, NVIDIA GeForce RTX 4090, 24564 MiB,    17 MiB,      24195 MiB,   0 %   <-- 全空
6, NVIDIA RTX A6000,        49140 MiB,    27148 MiB,   21523 MiB,   100 %
7, NVIDIA RTX A6000,        49140 MiB,    20736 MiB,   27935 MiB,   99 %
```

- **GPU2 为全空 A6000（48.6 GiB）**，是本阶段首选。
- 但本机是共享机，空闲卡随时会被别人占。所有脚本把 GPU 做成参数，并在每次运行记录当时显存，避免把「邻卡干扰」误读成算法效果。**严格遵守：绝不 kill 他人进程，绝不动 GPU0/1/3/6/7 上的作业。**

---

## 0.3 vLLM 版本决策（本阶段核心，含失败证据）

目标：找到既有 **Qwen2.5-VL 视频**、又有 **`custom_class` 自定义 Proposer 注入点**、且**能在这块驱动 550 的卡上跑起来**的 vLLM。

### 0.3.1 逐版本取证

用 `tar tzf <codeload tarball> | grep custom_class_proposer.py` 直接验证注入点是否存在：

| vLLM | 有无 `spec_decode/custom_class_proposer.py` | torch pin | torch 的 CUDA |
|---|---|---|---|
| 0.12.0 | ❌ 无 | 2.9.0 | cu12 |
| 0.13.0 | ❌ 无 | 2.9.0 | cu12 |
| 0.15.0 | ❌ 无 | 2.9.1 | cu12 |
| 0.18.0 | ❌ 无 | 2.10.0 | cu12 |
| 0.19.1 | ❌ 无 | 2.10.0 | cu12 |
| 0.20.0 | ❌ 无 | **2.11.0** | **cu13** |
| 0.21.0 | ❌ 无 | 2.11.0 | cu13 |
| **0.22.0** | ✅ **有（首个）** | 2.11.0 | cu13 |
| 0.24.0 – 0.30.0 | ✅ 有 | 2.11 / 2.13 | cu13 |

`custom_class` 注入点是 **v0.22.0 才引入**的（0.21.0 还没有）。

### 0.3.2 冲突

- 注入点要求 vLLM ≥ 0.22.0。
- vLLM ≥ 0.22.0 锁 `torch==2.11.0`，而 **PyPI 上 torch 2.11.0 是 cu13 构建**（依赖 `nvidia-*-cu13`）。
- **cu13 在驱动 550 上加载不了。** 这不是推测，是实测：

  本机既有环境 `vllm-omni-dev-fast`（dist-info `vllm-0.28.0`）执行 `import vllm` 直接失败：

  ```
  File ".../vllm/platforms/cuda.py", line 23, in <module>
      import vllm._C_stable_libtorch
  ImportError: libcudart.so.13: cannot open shared object file: No such file or directory
  ```

  即：**任何走 PyPI 默认 wheel 的 vLLM ≥ 0.22.0，在这台机器上都起不来。**

### 0.3.3 解法：改用 cu129（CUDA 12.9）变体 wheel

关键事实：**CUDA 12.9 属于 CUDA 12.x，受「CUDA minor version compatibility」覆盖**——驱动 ≥ 525.60.13 即可运行任意 CUDA 12.x 应用。驱动 550 满足。而 cu13 不在覆盖范围内（需驱动 ≥ 580）。

而 vLLM 与 PyTorch 都发布了非默认 CUDA 的 wheel：

```
https://wheels.vllm.ai/0.22.0/cu129/vllm/
   -> vllm-0.22.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl        ✅ 存在

https://download.pytorch.org/whl/cu129/torch/
   -> torch-2.11.0+cu129-cp311-cp311-manylinux_2_28_x86_64.whl     ✅ 存在
```

**结论：vLLM 0.22.0+cu129 + torch 2.11.0+cu129，既能拿到真实的 `custom_class` 注入点，又能在驱动 550 上运行，且不需要给 vLLM 打补丁。**

### 0.3.4 备选方案（若 cu129 在本驱动上初始化失败）

只降一级：**vLLM 0.19.1**（torch 2.10.0 / cu12，最后一个 CUDA-12 原生版本）+ **移植 `custom_class` 钩子**。
移植面很小，共 2 个文件（依据 v0.22.0 源码）：
1. `vllm/config/speculative.py`：允许 `method="custom_class"`，并让 `draft_model_config = target_model_config`；
2. `vllm/v1/worker/gpu_model_runner.py`：`create_custom_proposer(...)` 实例化 + `propose(...)` 调用分支。

**已明确不采用**：升级 NVIDIA 驱动到 ≥580。原因：需要 root，且本机 6 张卡正在跑他人作业（GPU0/1/3/6/7 占用 21–44 GiB、利用率 98–100%），重载驱动会打断他人任务，违反硬性规则 8。

---

## 0.4 资产

### 模型

`/mnt/local/ldh/PhyLLM/Qwen2.5-VL-7B-Instruct`（只读引用，不复制、不修改）

```
model_type    : qwen2_5_vl
architectures : ['Qwen2_5_VLForConditionalGeneration']
text layers   : 28, hidden 3584
vision        : depth 32, hidden 1280
weights       : model-00001..00005-of-00005.safetensors (共 ~16.2 GB)
```

### 视频

`datasets/Video-MME/videos/data/*.mp4`，**30 个**片段（Video-MME chunk 01）。

---

## 0.5 环境隔离

| 项 | 值 |
|---|---|
| conda env | `vllm-std`（**新建**，Python 3.11.16） |
| 路径 | `/home/mcy/.conda/envs/vllm-std` |
| torch | `2.11.0+cu129`（CUDA 12.9 运行时） |
| vLLM | `0.22.0+cu129` |
| 源码树 | `third_party/vllm`，分支 `feat/std-proposer`，基线 tag `v0.22.0-pristine` |
| 基线 commit | `028c47b258f65568578ac28bb9638b2e5ef80517` |
| 源码 tarball sha256 | `72e40a471069d5285a1f57d2d056b18a10acd82f9fa5a403104affa2cfe23fd8` |
| wheel sha256 | `405319a7fde1af416eef6cbab0315d354d8b9c1a45b97097a9f1c3707bf05f74` |
| pip freeze | `results/vllm-std/env/pip-freeze-vllm-std.txt`（188 行） |
| GPU 清单 | `results/vllm-std/env/gpu-inventory.csv` |

**不使用**他人工况的既有环境（`vllm-omni-dev-fast` 等），仅作只读源码参考。

### 网络注记（会影响所有脚本）

Harness 导出的代理 `http_proxy=http://127.0.0.1:7904` 在本机是**死的**（`Connection refused`）。所有需要联网的命令必须 `unset` 代理，否则表现为「DNS 失败 / connection refused」的假故障。`scripts/vllm-std/env.sh` 已统一处理。

另有两条装机路径事实：
- `github.com` 的 git clone / raw 拉取**超时**；`codeload.github.com` 的 tarball **可用**。
- `wheels.vllm.ai` 与 `download.pytorch.org` **可用**，但 `wheels.vllm.ai/<ver>/cuNNN/vllm/` 只是目录列表，**不是 PEP 503 simple index**，必须按直链取 wheel。

---

## 0.6 复现命令

```bash
# 一键重建（幂等）：落地源码树 + 装 torch/vllm + 验收
bash scripts/vllm-std/00_setup_env.sh

# 单独验收
source scripts/vllm-std/env.sh
CUDA_VISIBLE_DEVICES=2 $PY -c \
  "import vllm,torch; print(vllm.__version__, torch.__version__, torch.cuda.get_device_name(0))"
```

---

## 0.7 验收（go/no-go）

| 判据 | 结果 |
|---|---|
| 能 `import vllm` | ✅ PASS |
| 能打印版本与 GPU 信息 | ✅ PASS |
| GPU 可见且驱动兼容 | ✅ PASS（cu129 在驱动 550 上加载成功） |

**判定：GO → 进入阶段 1。**

---

## 0.8 验收证据（实测输出）

导入与运行时（`CUDA_VISIBLE_DEVICES=2`，即那张空闲 A6000）：

```
vllm      : 0.22.0
torch     : 2.11.0+cu129 | cuda build: 12.9
devices   : 1
device0   : NVIDIA RTX A6000
capability: (8, 6)
matmul ok : True
```

这一条同时证明了本阶段最关键的不确定性：**CUDA 12.9 运行时在驱动 550.163.01 上真实可用**，且能在这张 sm_86 卡上跑通 kernel。若当初选了 PyPI 默认 wheel（cu13），此处会得到 `libcudart.so.13: cannot open shared object file`。

注入点在同一份已安装代码中确认存在：

```
vllm/v1/spec_decode/custom_class_proposer.py      ✅
vllm/v1/worker/gpu_model_runner.py:175   from vllm.v1.spec_decode.custom_class_proposer import create_custom_proposer
vllm/v1/worker/gpu_model_runner.py:551   if self.speculative_config.method == "custom_class":
vllm/v1/worker/gpu_model_runner.py:4778  elif spec_config.method == "custom_class":
vllm/config/speculative.py:65            "custom_class",
vllm/model_executor/models/qwen2_5_vl.py          ✅ (74 KB)
```

### 阶段 0 失败记录（未掩盖）

| 失败 | 根因 | 处置 |
|---|---|---|
| `git clone https://github.com/vllm-project/vllm.git` 超时（exit 124） | 本机到 github.com 的 TCP 被阻断/极慢 | 改用 `codeload.github.com` tarball + `git init` |
| `pip install vllm==0.22.0+cu129 --extra-index-url https://wheels.vllm.ai/0.22.0/cu129/vllm/` → `No matching distribution found` | 该 URL 是目录列表而非 PEP 503 index | 改为直链下载 wheel 再本地安装 |
| `pip install "/tmp/vllm-0.22.0+cu129.whl"` → `Invalid wheel filename (wrong number of parts)` | 保存 wheel 时丢了 tag 段 | 保留完整文件名 `...cp38-abi3-manylinux_2_28_x86_64.whl` |
| 既有环境 `vllm-omni-dev-fast` 的 `import vllm` 失败：`libcudart.so.13` 缺失 | 该 vLLM(0.28.0) 是 cu13 构建 | 不修它；本工作流改用 cu129 独立环境 |
| 首次 `nvidia-smi` 报 "couldn't communicate with the NVIDIA driver" | 瞬时驱动未就绪（后续稳定可用） | 复测即恢复；已在 §0.2 记录 |
