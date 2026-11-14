# dm-agent

[![CI](https://github.com/KDB1223/dm-code-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/KDB1223/dm-code-agent/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](src/dm_agent/__init__.py)

在本地用 ReAct 循环修改一个工作区，并把每一步追加写入 JSONL 会话日志，方便检查和回放。

网页聊天往往看不到工具轨迹。dm-agent 把轨迹留在项目旁边：计划、工具调用、观察结果、最终回答。内核足够短，可以当作课程实验来读。

- **ReAct 循环。** 计划，调用工具，读取观察，重复，直到最终回答或达到步数上限。
- **工作区工具。** 读、写、精确替换、列目录、搜索。Shell 会拒绝指向项目之外的路径。
- **课程工具。** `lint_python` 报告语法错误、裸 `except`、残留的 `breakpoint()`。`run_pytest` 返回结构化的通过/失败计数。
- **审计日志。** `.dm-agent/sessions/` 下的追加 JSONL，支持 `list`、`show`、`stats`、`changes`，以及不重新执行工具的干跑 `replay`。
- **离线评测。** 三个小题配有黄金检查。不需要 API key，`python -m evals.run` 应输出 `score 3/3`。

**Topics:** `coding-agent` `react` `auditability` `local-first` `llm` `education`

这是受「本地优先、可审计的编程代理」这一想法启发的**原创实现**（ReAct 循环、工作区工具、可检查的日志）。它不是 hwfengcs/DM-Code-Agent 的 fork，也没有复制该项目的代码。

English: [README.md](README.md). 课程实验：[docs/lab-zh.md](docs/lab-zh.md)（实验：可审计的本地 Coding Agent）。

## 相对最小基线多出来的部分

最小基线是循环、文件/搜索/shell 工具和 JSONL 日志。本仓库另外包含：

| 增加的内容 | 位置 |
| --- | --- |
| `lint_python` 与 `run_pytest` | `src/dm_agent/tools/` |
| 带 `check.py` 黄金检查的离线评测 | `evals/` |
| 中文说明与课程实验 | `README.zh-CN.md`、`docs/lab-zh.md` |
| 示例工作区、示例配置、CI | `examples/`、`.github/workflows/ci.yml` |

## 一分钟跑起来

不需要 API key：

```bash
git clone https://github.com/KDB1223/dm-code-agent.git
cd dm-code-agent
python -m pip install -e ".[dev]"
python -m evals.run
```

标准输出应与 `evals/expected_output.txt` 一致：

```text
fix-add        PASS
write-greet    PASS
clear-debug    PASS
score 3/3
```

查看某题的工具轨迹（标准输出是干跑转录，标准错误是一行 PASS/FAIL）：

```bash
python -m evals.run --transcript fix-add
```

### 配置 API 并改示例里的 bug

`examples/demo_workspace/math_utils.py` 里的 `add()` 写成了减法，测试期望的是加法。

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"
# 任何 OpenAI 兼容服务。用官方接口时可以不设。
# export OPENAI_BASE_URL="https://api.openai.com/v1"

python -m dm_agent run --workspace examples/demo_workspace \
  "$(cat examples/demo_workspace/TASK.txt)"
python -m dm_agent sessions list --workspace examples/demo_workspace
python -m dm_agent sessions show SESSION_ID --workspace examples/demo_workspace
python -m dm_agent sessions stats SESSION_ID --workspace examples/demo_workspace
python -m dm_agent sessions changes SESSION_ID --workspace examples/demo_workspace
python -m dm_agent sessions replay SESSION_ID --workspace examples/demo_workspace
```

`stats` 只读已有 JSONL，打印回合数、工具次数、成功的 `edit_file` / `write_file` 所记录的文件，以及会话是否以 `final` 结束。不需要 API key，也不会重新执行工具。加上 `--json` 时输出同一份摘要的 JSON。

`changes` 只读同一份 JSONL，按日志顺序打印每一次成功的 `edit_file` 和 `write_file`：路径、step、call id，以及记下的 `old`/`new` 或 `write_file` 内容。同一路径的多次成功修改都会列出。输出重建的是日志参数，不是当前工作区的 diff。不需要 API key，也不会调用模型或重新执行工具。加上 `--json` 时输出同一份列表的 JSON。

若 `dm-agent` 已在 `PATH` 上，可以把 `python -m dm_agent` 换成 `dm-agent`。

会话文件：

```text
examples/demo_workspace/.dm-agent/sessions/<session-id>.jsonl
```

`replay` 只根据日志重印转录，不会再次调用模型，也不会再改工作区。

配置示例：[`examples/config.example.toml`](examples/config.example.toml)。优先级从低到高：内置默认值、TOML 文件、环境变量、命令行。密钥请放在 `OPENAI_API_KEY`，不要写进仓库。会话日志只记录模型名和 base URL 的协议与主机，不记录密钥。

| 配置 | 环境变量 | 命令行 |
| --- | --- | --- |
| 密钥 | `OPENAI_API_KEY` | |
| 模型 | `OPENAI_MODEL` | `--model` |
| 接口地址 | `OPENAI_BASE_URL` | |
| 最大回合数 | `DM_AGENT_MAX_STEPS` | `--max-steps` |
| 工作区 | `DM_AGENT_WORKSPACE` | `--workspace` |
| 日志目录 | `DM_AGENT_LOG_DIR` | `--log-dir` |
| Shell / pytest 超时 | `DM_AGENT_SHELL_TIMEOUT` | |

未指定 `--config` 时，按顺序查找：`<workspace>/dm-agent.toml`、`./dm-agent.toml`、`$XDG_CONFIG_HOME/dm-agent/config.toml`（否则 `~/.config/dm-agent/config.toml`）。

`llm.provider = "openai"` 表示任何兼容 `POST {base_url}/chat/completions` 的服务（官方 API、本机 llama.cpp、vLLM 或网关）。没有实现 Anthropic 原生接口。

## 和黑盒聊天的差别

| | dm-agent | 常见托管聊天 |
| --- | --- | --- |
| 代码在哪跑 | 你的机器上，限定一个工作区 | 取决于产品自己的沙箱 |
| 留下什么 | 可 diff、可回放的 JSONL | 服务商保存的对话 |
| 工具轨迹 | 每一步计划、调用、观察都有 step id | 经常不可导出 |
| 教学 | 实验可以对照同一个 `call_id` | 中间步骤难以批改 |
| 安装 | `pip install`、一份 TOML；只有真跑模型才要密钥 | 账号和网页 |
| 离线检查 | `python -m evals.run` 不需要密钥 | 依赖在线服务 |

## 会话日志里要看什么

每一行是一个 JSON 对象，只追加、不改写。`kind` 包括 `session_start`、`user`、`thought`、`tool_call`、`tool_result`、`final`、`max_steps`、`error`。同一次工具调用和它的结果共享 `step` 与 `call_id`。

实验步骤、思考题和「应在 JSONL 里观察到什么」写在 [docs/lab-zh.md](docs/lab-zh.md)。

## 安全

文件工具和 `run_pytest` 会解析路径，拒绝工作区之外的目标，包括指向外部的符号链接。

`run_shell` 很强，但不是容器：它把工作目录和 `HOME` 固定在工作区，并在启动前拒绝文本里出现的越界绝对路径和 `..`。运行时拼出来的路径，以及访问网络，静态检查看不到。只在可信任务上使用 shell。更完整的说明见 [README.md](README.md) 的 Safety 一节。

## 开发

```bash
python -m pip install -e ".[dev]"
pytest
python -m evals.run
```

运行时依赖为零（Python 3.11+）。pytest 在 `dev` extra 里。许可证：[MIT](LICENSE)。
