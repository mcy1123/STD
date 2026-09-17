# ECNU Phase-8 A100 访问手册

> 每次要连 A100 跑实验时，把本文档丢给 agent 或自己照做即可。
> 配套脚本：`scripts/a100.sh`（`bash scripts/a100.sh help`）。

## 0. 一句话

```bash
bash scripts/a100.sh status        # 看 gpu23 两张卡谁空
bash scripts/a100.sh gpu 'nvidia-smi'   # 在 gpu23 上执行任意命令
```

## 1. 连接拓扑

本机**无法直达** GPU 节点，必须经过登录节点跳转：

```
本机 (Linux / Windows)
  │  ssh -p 2323 xlwang@59.78.189.133      ← login2，需要密码（或密钥）
  ▼
login2（登录节点，有外网）
  │  ssh 10.11.200.23                       ← gpu23，已配置免密（BatchMode 可用）
  ▼
gpu23（计算节点，2 × A100 80GB）
```

| 项 | 值 |
|---|---|
| 登录节点 | `59.78.189.133`，端口 **2323**，用户 `xlwang`（主机名 `login2`） |
| 计算节点 | `10.11.200.23`（主机名 `gpu23`），**只能从 login2 跳** |
| 认证方式 | login2 目前为**密码认证**（本机 `~/.ssh` 无该主机密钥；gpu23 从 login2 免密） |
| GPU | 2 × NVIDIA A100 80GB PCIe |

> 本机到 `59.78.189.133:2323` 的 TCP 连通性已实测可达。

## 2. GPU 使用规约（重要）

- **只用物理 GPU 1。**
- **GPU 0 常驻一个 vLLM 工作负载**（`VLLM::Worker_TP0`，约 50GB 显存），**绝不触碰、绝不 kill**。
- `scripts/benchmark_a100_dynamic.py` 与 `scripts/run_a100_ablation.sh` 自带空闲检查：目标卡利用率/占用不满足时会**直接拒绝启动**，这是预期行为，不要绕过。

查看两卡状态：

```bash
bash scripts/a100.sh status
```

## 3. 凭证管理

### 3.1 推荐：装一次密钥，之后永久免密

```bash
bash scripts/a100.sh setup-key          # 交互式输入一次密码
```

等价于 `ssh-copy-id -p 2323 xlwang@59.78.189.133`。装好后第 4 节所有命令**不再需要密码**，也是唯一能让自动化真正无人值守的方式。

### 3.2 备选：ControlMaster 连接复用（不落盘密码）

脚本默认行为。首次建立主连接时输一次密码，之后 **600 秒内**所有命令复用同一条 TCP 连接，不再提示：

```bash
bash scripts/a100.sh login 'hostname'   # 首次：提示输入密码
bash scripts/a100.sh gpu   'hostname'   # 复用，无提示
bash scripts/a100.sh close              # 主动关闭主连接
```

主连接 socket 默认在 `${TMPDIR:-/tmp}/a100-ssh-$USER/control`，可用 `A100_SOCK` 覆盖。

### 3.3 备选：密码文件 + sshpass（本机**尚未安装** sshpass）

```bash
sudo apt install sshpass
install -m600 /dev/null ~/.a100_password   # 把密码写进去，权限 600
export A100_PASSWORD_FILE=~/.a100_password
```

### 3.4 ⛔ 绝对不要做

- **不要把密码写进本仓库的任何文件。** 本仓库有 GitHub remote（`git@github.com:mcy1123/STD.git`），一旦 push 即泄露。
- 不要把密码写进 `~/.zshrc`、别名或可提交的脚本。

## 4. 远端关键路径

| 用途 | 路径 |
|---|---|
| 代码（实验实际使用的副本） | `/public/home/xlwang/mcy/Project/STD-latest` |
| 代码（部署文档中的路径） | `/public/home/xlwang/mcy/Project/STD` |
| Python 环境 | `/public/home/xlwang/mcy/conda_envs/specvlm/bin/python` |
| 模型 | `/public/home/xlwang/mcy/STD_assets/models/Qwen2.5-VL-7B-Instruct` |
| 数据集 | `/public/home/xlwang/mcy/STD_assets/datasets/Video-MME`（视频在 `.../videos`） |
| 结果 | `/public/home/xlwang/mcy/STD_assets/results` |

> `STD-latest` 与 `STD` 都存在。2026-09-09 的所有 A100 实验 manifest 指向的是 **`STD-latest`**；改代码前先确认哪一个是当前工作副本。

## 5. 跑实验的标准流程

```bash
# 1) 同步本地代码到远端（scp 经 login2，再落到 gpu23 或直接改 login2 上的副本）
bash scripts/a100.sh push src/std_repro/dynamic_selection.py /public/home/xlwang/mcy/Project/STD-latest/src/std_repro/

# 2) 确认 GPU1 空闲
bash scripts/a100.sh status

# 3) 在 gpu23 上跑（离线模式；模型与数据都在本地盘，不要联网）
bash scripts/a100.sh gpu 'cd /public/home/xlwang/mcy/Project/STD-latest && \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && \
  /public/home/xlwang/mcy/conda_envs/specvlm/bin/python \
  scripts/benchmark_a100_dynamic.py --gpu 1 --profile-components ...'

# 4) 把结果拉回本地
bash scripts/a100.sh pull /public/home/xlwang/mcy/STD_assets/results/xxx_report.md ./results/
```

消融实验矩阵直接用已入库的驱动脚本：

```bash
bash scripts/a100.sh gpu 'cd /public/home/xlwang/mcy/Project/STD-latest && bash scripts/run_a100_ablation.sh stage1'
```

## 6. 文件传输

```bash
bash scripts/a100.sh push <本地路径> <远端绝对路径>   # 本机 -> login2
bash scripts/a100.sh pull <远端绝对路径> <本地路径>   # login2 -> 本机
```

`push`/`pull` 走 login2。若要直达 gpu23，先 `push` 到 login2，再用
`bash scripts/a100.sh gpu 'scp -q /tmp/x 10.11.200.23:/目标/'`（历史实验就是这么做的）。

## 7. 常见故障

| 现象 | 原因 / 处理 |
|---|---|
| `Permission denied (publickey,...,password)` | 正常，说明还没装密钥；用 `setup-key`（§3.1）或让脚本提示输密码 |
| `Connection refused` / 超时 | 端口应为 **2323**（不是 22）；确认 `A100_PORT` |
| 脚本连上但命令卡住 | 主连接可能已过期（ControlPersist 600s），重跑即可；或 `bash scripts/a100.sh close` 后重连 |
| gpu23 上命令报 `Permission denied` | 说明 login2→gpu23 的免密失效；先 `bash scripts/a100.sh login 'ssh 10.11.200.23 hostname'` 手工确认 |
| 实验拒绝启动（GPU not idle） | 预期保护：目标卡被占用。等空闲或确认是否有人在用 GPU1 |

## 8. ⚠️ 安全警告：密码已明文泄露

排查连接方式时发现，**该账号的登录密码以明文形式存在于本机的 agent 会话记录中**：

```
~/.codex/sessions/**/*.jsonl
~/.codex/history.jsonl
```

这些文件同时包含主机、端口、账号与密码。建议按顺序处理：

1. **轮换该账号密码**（最彻底；上述记录无法保证已被清理）。
2. 清理或加密这些会话记录（`~/.codex/` 下相关文件）。
3. 之后改用 **SSH 密钥认证**（§3.1），从此不再有任何地方需要保存密码。
4. 若曾把 `~/.codex` 或本仓库同步到云端/网盘/其他机器，视同已泄露处理。

> 本手册与 `scripts/a100.sh` **均不含任何密码**，可安全提交。
