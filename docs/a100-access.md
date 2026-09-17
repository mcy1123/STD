# ECNU Phase-8 A100 访问手册

> 每次要连 A100 跑实验时，把本文档丢给 agent 或自己照做即可。
> 配套脚本：`scripts/a100.sh`（`bash scripts/a100.sh help`）。
> **状态（2026-09-17）：已配置免密 SSH，日常使用不再需要密码。**

## 0. 一句话

```bash
ssh a100-gpu                              # 直接从本机登进 gpu23
bash scripts/a100.sh status               # 看两张卡谁空
bash scripts/a100.sh e2e stage0           # 同步代码→查卡→跑实验→拉回报告
```

## 1. 连接拓扑

GPU 节点**不能从公网直达**，必须经登录节点跳转（现已用 `~/.ssh/config` 封装，你只需用别名）：

```
本机 ──ssh -p 2323──▶ login2 ──ProxyJump──▶ gpu23
```

| 项 | 值 |
|---|---|
| 登录节点别名 | **`ssh a100`** → `59.78.189.133:2323`，用户 `xlwang`（主机名 `login2`） |
| GPU 节点别名 | **`ssh a100-gpu`** → `10.11.200.23`（主机名 `gpu23`），经 `ProxyJump a100` |
| 认证 | **SSH 公钥**（`~/.ssh/id_ed25519`），已装好；login2 与 gpu23 都免密 |
| GPU | 2 × NVIDIA A100 80GB PCIe |
| 共享家目录 | `/public/home/xlwang` 在 login2 与 gpu23 之间**共享**，所以 `authorized_keys` 与代码在两台机器上一致 |

## 2. `~/.ssh/config` 片段（备份/重建用）

```sshconfig
Host a100
    HostName 59.78.189.133
    Port 2323
    User xlwang
    IdentityFile ~/.ssh/id_ed25519
    ControlMaster auto
    ControlPath ~/.ssh/cm-a100-%r@%h:%p
    ControlPersist 600

Host a100-gpu
    HostName 10.11.200.23
    User xlwang
    IdentityFile ~/.ssh/id_ed25519
    ProxyJump a100
    ControlMaster auto
    ControlPath ~/.ssh/cm-a100gpu-%r@%h:%p
    ControlPersist 600
```

> 注意：`ssh_config` 里"**首个命中的值生效**"，所以已有的 `Host *` 块要放在**前面**，本块追加在后即可。

## 3. 免密原理与恢复

- 本机公钥 `~/.ssh/id_ed25519.pub` 已写入集群的 `~/.ssh/authorized_keys`。
- 因为 `/public/home` 是共享家目录，同一份 `authorized_keys` 对 login2 和 gpu23 同时生效，所以两跳都免密。
- 若哪天免密失效，用密码重装一次即可（会提示输入一次密码）：

  ```bash
  ssh-copy-id -i ~/.ssh/id_ed25519.pub -p 2323 xlwang@59.78.189.133
  ```

- 验证：

  ```bash
  ssh -o BatchMode=yes a100 'hostname'          # 期望 login2
  ssh -o BatchMode=yes a100-gpu 'hostname'      # 期望 gpu23
  ```

## 4. GPU 使用规约（重要）

- **只用物理 GPU 1。**
- **GPU 0 常驻 vLLM**（`VLLM::EngineCore`，约 79GB，100% 利用率），**绝不触碰**。
- **GPU 1 可能被别人占用。** 截至 2026-09-17 21:xx，GPU 1 上有另一个作业：

  ```
  pid 47900  /public/home/xlwang/jyy/anaconda/envs/qwen25vl/bin/python  21352 MiB
  ```

  这不是你启动的进程，**不要 kill**。等它自行结束再跑实验。
- `scripts/benchmark_a100_dynamic.py` 与 `run_a100_ablation.sh` 自带空闲检查：目标卡不空闲会**直接拒绝启动**，这是预期保护，不要绕过。

查看状态：

```bash
bash scripts/a100.sh status
```

## 5. 远端关键路径

| 用途 | 路径 |
|---|---|
| 代码（实验实际工作副本） | `/public/home/xlwang/mcy/Project/STD-latest`（HEAD `40efa1e` + 工作区改动） |
| 代码（部署文档中的旧路径） | `/public/home/xlwang/mcy/Project/STD`（HEAD `36046dd`） |
| Python 环境 | `/public/home/xlwang/mcy/conda_envs/specvlm/bin/python` |
| 模型 | `/public/home/xlwang/mcy/STD_assets/models/Qwen2.5-VL-7B-Instruct` |
| 数据集 | `/public/home/xlwang/mcy/STD_assets/datasets/Video-MME`（视频在 `.../videos`） |
| 结果 | `/public/home/xlwang/mcy/STD_assets/results` |

远端仓库的 `origin` 也是 `https://github.com/mcy1123/STD.git`，但**日常同步用 `a100.sh sync`（rsync），不依赖 GitHub**。

## 6. `scripts/a100.sh` 用法

```bash
bash scripts/a100.sh help          # 全部子命令
bash scripts/a100.sh status        # gpu23 两卡状态 + 占用进程
bash scripts/a100.sh login [cmd]   # 在 login2 执行（无参数=交互 shell）
bash scripts/a100.sh gpu   [cmd]   # 在 gpu23 执行（无参数=交互 shell）
bash scripts/a100.sh sync  [paths] # rsync 本地代码到远端工作副本
bash scripts/a100.sh ablation stage1   # 跑消融矩阵
bash scripts/a100.sh e2e   stage1      # sync + 查卡 + 跑 + 拉回报告
bash scripts/a100.sh push <local> <remote>
bash scripts/a100.sh pull <remote> <local>
bash scripts/a100.sh close         # 关闭复用连接
```

默认行为：

- `sync` 默认同步 `src scripts tests docs PROGRESS.md README.md`（`--exclude __pycache__`）。
- `e2e` 把 `*_report.md` / `*_summary.json` 拉到本地 `results/a100_ablation/`。
- 所有目标可用环境变量覆盖：`A100_REPO`、`A100_ASSETS`、`A100_LOGIN_ALIAS`、`A100_GPU_ALIAS`、`A100_LOCAL_RESULTS` 等（见 `help`）。

## 7. 手动跑一条实验（不经 e2e）

```bash
bash scripts/a100.sh gpu 'cd /public/home/xlwang/mcy/Project/STD-latest && \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && \
  CUDA_VISIBLE_DEVICES=1 /public/home/xlwang/mcy/conda_envs/specvlm/bin/python \
  scripts/benchmark_a100_dynamic.py --gpu 1 --profile-components \
    --model-path /public/home/xlwang/mcy/STD_assets/models/Qwen2.5-VL-7B-Instruct \
    --data-path  /public/home/xlwang/mcy/STD_assets/datasets/Video-MME \
    --video-root /public/home/xlwang/mcy/STD_assets/datasets/Video-MME/videos \
    --output     /public/home/xlwang/mcy/STD_assets/results/my_run.jsonl \
    --frame-num 128 --max-new-tokens 128 --limit 10 --repeats 2 \
    --gamma 9 --k-plus-text 1024 \
    --dynamic-collector v2 --refresh-mode full \
    --dynamic-query-mode three --dynamic-bootstrap attention \
    --selection-update-interval 1 --min-selection-change-ratio 0.05 \
    --verify-fallback sequential_on_low_margin'
```

> 输出文件用 `open(..., "x")` 创建，**不会覆盖已有文件**；重复实验请换新文件名。
> 模型与数据都在本地盘，加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 避免联网。

## 8. 常见故障

| 现象 | 原因 / 处理 |
|---|---|
| `ssh a100` 报 `Could not resolve hostname` | `~/.ssh/config` 里的两个 Host 块丢了，按 §2 重建 |
| `Permission denied (publickey)` | 密钥没装或被清；按 §3 用 `ssh-copy-id` 重装 |
| `Connection refused` / 超时 | 端口必须是 **2323**（不是 22） |
| `a100-gpu` 报 `channel ... open failed` | login2 的 TCP 转发被限制；改用嵌套 `ssh a100 'ssh 10.11.200.23 ...'` |
| 实验拒绝启动（GPU not idle） | 预期保护：GPU1 被占用。`status` 看是谁，**不要 kill 别人的进程** |
| rsync 报 `command not found` | `-e` 用错；用脚本的 `sync`/`push`/`pull`，别手写 `-e "ssh a100"` |

## 9. ⚠️ 安全事项

- **密码仍以明文存在于本机 agent 会话记录中**：

  ```
  ~/.codex/sessions/**/*.jsonl
  ~/.codex/history.jsonl
  ```

  本次已把临时落盘的密码文件删除（`~/.ssh/a100_password`、`~/.ssh/a100_askpass.sh` 均已清理），但**记录里的那份还在**。建议**轮换该账号密码**，或至少清理/加密这些记录。
- 本手册与 `scripts/a100.sh` **均不含任何密码**，可安全提交。
- 本仓库有 GitHub remote，**永远不要把密码写进仓库里的任何文件**。
