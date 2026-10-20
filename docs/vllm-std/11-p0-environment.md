# 阶段 0：P0 环境核验

> 状态：**✅ 通过（GO → 阶段 1）**
> 日期：2026-09-26
> 环境：vLLM `0.22.0+cu129` / torch `2.11.0+cu129` / 驱动 550.163.01 / A6000 (sm_86) / Python 3.11.16

---

## 0.1 三个硬前提：可复现 + 已脚本化

阶段 1 之前，这三个前提原本只存在于命令行里（`VLLM_ENABLE_V1_MULTIPROCESSING=0` 靠手打）。
现在它们**固化进 `scripts/vllm-std/B_probe_attn_weights.py` 本身**，不需要任何额外参数。

| # | 前提 | 脚本化方式 | 验证证据 |
|---|---|---|---|
| 1 | 进程内执行（否则钩子在 EngineCore 子进程，结果回不到父进程） | `os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING","0")`，放在 `import vllm` **之前**（`vllm.envs` 在导入期读取）——`B_probe_attn_weights.py:82` | 日志中 **`EngineCore pid=` 出现 0 次** → 确认无子进程 |
| 2 | 关闭 chunked prefill（否则 K 只剩当前块） | 改为**默认关闭**：`enable_chunked_prefill=args.chunked_prefill`（默认 False），`--chunked-prefill` 才开启——`B_probe_attn_weights.py:157` | `key_is_full_prompt: True`，`kv_len == prompt_tokens == 2420` |
| 3 | 钩子必须 trace-safe | 代码性质：钩子内**只保留张量引用**，无 `.cpu()` / `.tolist()` / `.item()` / 不读 `int(shape)`；全部分析移到 trace 区之外 | 阶段 B 中被一次 `.tolist()` 崩溃反向证明（`dynamo` trace 会穿过该 Python 方法） |

---

## 0.2 复现命令（无额外环境变量）

```bash
source scripts/vllm-std/env.sh
CUDA_VISIBLE_DEVICES=3 $PY scripts/vllm-std/B_probe_attn_weights.py \
    --num-frames 4 --max-tokens 8 --max-model-len 4096 --gpu-mem-util 0.85
```

日志：`results/vllm-std/logs/P0_verify_hook.log`
数据：`results/vllm-std/B_attn_weight_probe.json`

---

## 0.3 验收：三个数字与上次完全一致

| 量 | 上次（线 B） | 本次（阶段 0） | 判定 |
|---|---|---|---|
| `prompt_tokens` | 2420 | **2420** | ✅ 一致 |
| `statistic_shape` | `[4, 2420]` | **`[4, 2420]`** | ✅ 一致 |
| `mass_check` | 1792.0 / 1792.0 | **1792.0 / 1792.0** | ✅ 一致 |
| `key_is_full_prompt` | True | **True** | ✅ |
| `hook_calls` | 280 (prefill 84 / decode 196) | **280 (84 / 196)** | ✅ 一致（跨运行确定性） |
| `statistic_compute_seconds` | 0.1566 | 0.1114 | ⚠️ 单点值有波动，**阶段 1 会按规则取 3 次 mean/std** |

**判定：GO → 阶段 1。**

---

## 0.4 残留进程检查（含一次我自己的误杀，如实记录）

### 初始状态：干净

`ps -u mcy | grep EngineCore` → **无**。上一轮泄漏的 pid 1442533 已在上一轮清理。

### 我的这次误杀（失败记录）

在阶段 0 期间，我看到一个 `VLLM::EngineCore`（pid 3881732，占 41 GiB），
判定为泄漏并 kill 了它。**判断错了**：它其实是**同一时刻线 A 批量扫描正在运行的引擎**（在 GPU2 上）。

- **根因**：我的检查只用 `ps -u $(whoami)` + 进程名。但**我同时有两个作业在跑**
  （我的探针 + 线 A 扫描），两者属主相同、进程名相同 → 无法区分。
- **证据**：探针日志中 `EngineCore pid=` 出现 **0 次**，说明探针是**进程内**执行、**没有**子进程；
  因此那个 EngineCore 不属于探针。
- **影响**：只影响我自己的线 A 作业，**未触碰他人进程**；线 A 驱动存活并自动重试该配置。
- **处置**：
  1. `run_bench.sh` 增加 `trap cleanup EXIT INT TERM`，driver 被杀时一并终止 python 子进程
     ——这正是 pid 1442533 泄漏的成因；
  2. **新规则**：kill 任何 EngineCore 之前，必须先用**启动时间与当前作业的时间窗**交叉比对，
     不能只看属主+进程名。

### 给他人作业的安全声明

`nvidia-smi` 上的其他进程（`fin_research` 训练作业、用户 `zhr` 的作业）**全程未触碰**。

---

## 0.5 是否进入下一阶段

| 判据 | 结果 |
|---|---|
| 3 个前提可复现、可脚本化 | ✅ |
| 钩子复现成功，三个数字与上次一致 | ✅ |
| 残留进程已检查并报告 | ✅（含一次误杀，已记录根因与对策） |

**→ 进入阶段 1：157 ms op 级拆解。**
（约束：阶段 1 完成前**不写任何 proposer 业务代码**。）
