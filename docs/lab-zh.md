# 实验：可审计的本地 Coding Agent

本实验用 dm-agent 观察一个编程代理的完整轨迹：模型想做什么、调用了哪个工具、工具返回了什么、最后如何收束。实现是原创的，想法上来自「本地优先、每一步都可审计」的 coding agent，**不是**任何现有仓库的 fork，也没有复制第三方源码。

## 实验目标

1. 在本机安装并运行一个 ReAct 编程代理，不依赖网页聊天窗口。
2. 读懂追加写入的 JSONL 会话：用 `step` 和 `call_id` 把一次工具调用和它的观察结果对上。
3. 用离线评测确认结果可复现：没有 API key 也能得到 `score 3/3`。
4. 说明工作区禁闭（path jail）挡住了什么，以及 shell 为什么仍然要小心使用。

## 你将修改或阅读的代码

| 路径 | 作用 |
| --- | --- |
| `src/dm_agent/agent.py` | ReAct 循环：计划 → 工具 → 观察 → 重复或给出最终答案 |
| `src/dm_agent/session.py` | 追加写入 `.dm-agent/sessions/*.jsonl` |
| `src/dm_agent/safety.py` | 文件路径与 shell 文本的工作区检查 |
| `src/dm_agent/tools/lint_python.py` | 课程新增工具：语法、裸 `except`、残留 `breakpoint()` |
| `src/dm_agent/tools/run_pytest.py` | 课程新增工具：在工作区内运行 pytest 并返回计数 |
| `evals/run.py` | 用脚本化模型跑三个小题，并用 `check.py` 做黄金检查 |

## 环境

需要 Python 3.11 或更新版本。在仓库根目录执行：

```bash
python -m pip install -e ".[dev]"
pytest
python -m evals.run
```

`pytest` 应全部通过。`python -m evals.run` 的标准输出应是：

```text
fix-add        PASS
write-greet    PASS
clear-debug    PASS
score 3/3
```

这一步不需要 `OPENAI_API_KEY`。模型回合写在每个任务的 `evals/tasks/*/task.json` 里，由 `evals/harness.py` 按顺序回放。黄金检查在同目录的 `check.py`。

## 步骤

### 1. 看一次真实轨迹

```bash
python -m evals.run --transcript fix-add
```

标准输出是干跑转录（dry replay）：工具没有被重新执行，打印的是日志里已经记下的观察结果。在输出里找出下面几类事件：

- `user`：任务原文
- `thought`：模型在调用工具前写下的计划
- `tool_call` / `tool_result`：同一步（同一个 `step`）里的调用和观察，用 `call_id` 配对
- `final`：最终回答

`fix-add` 应出现成功的 `read_file`、`edit_file` 和 `lint_python`。`write-greet` 还应出现成功的 `run_pytest`。`clear-debug` 会先看到一次失败的 `lint_python`（裸 `except` 和 `breakpoint()`），编辑之后再看到一次 `status: ok`。

### 2. 打开 JSONL 而不是只看转录

离线评测把会话写在临时目录里，进程结束就删掉。要留下可以反复打开的文件，在示例工作区上跑一次真实模型，或在实验报告里贴 `--transcript` 的输出。

配置好密钥后：

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"
# 可选：export OPENAI_BASE_URL="https://api.openai.com/v1"

dm-agent run --workspace examples/demo_workspace "$(cat examples/demo_workspace/TASK.txt)"
dm-agent sessions list --workspace examples/demo_workspace
dm-agent sessions show <id> --workspace examples/demo_workspace
dm-agent sessions stats <id> --workspace examples/demo_workspace
dm-agent sessions replay <id> --workspace examples/demo_workspace
```

日志路径：

```text
examples/demo_workspace/.dm-agent/sessions/<id>.jsonl
```

每一行是一个 JSON 对象，追加写入，不会回头改前面的行。字段顺序里始终有 `ts`、`session_id`、`seq`、`step`、`kind`。

建议用编辑器打开该文件，完成下面的对照：

1. 找到 `kind` 为 `tool_call` 且 `tool` 为 `edit_file` 的那一行，记下 `step` 和 `call_id`。
2. 找到 `call_id` 相同的 `tool_result`。确认它的 `step` 与调用相同，`ok` 为 true，`output` 写明替换了哪一处。
3. 找到随后的 `lint_python` 或 `run_pytest` 结果，确认最终回答不是在工具失败时结束的。

`dm-agent sessions replay` 只重印这些行。它不会再次改文件。可以先复制一份 `math_utils.py`，再执行 replay，确认副本没有变化。

`dm-agent sessions stats` 同样只读 JSONL。它打印模型回合数、各工具调用次数、成功写入或修改并已记录在日志里的文件路径，以及会话是否以 `final` 干净结束。不需要 API key，也不会重新执行工具。

### 3. 观察工作区禁闭

在 Python 里对一个临时目录调用（或直接阅读 `tests/test_safety.py`）：

- `read_file` 请求 `../` 或 `/etc/passwd` 时，日志里的 `tool_result` 应以 `Refused` 开头，且 `ok` 为 false。
- 工作区里的符号链接如果指向外部文件，解析后同样被拒绝。
- `run_shell` 的命令文本里如果出现工作区以外的绝对路径（包括写在 Python 字符串里的 `/etc/passwd`），命令不会启动。
- `cd` 到工作区子目录是允许的；`cd ..` 会在启动前被拒绝。

同时写进实验报告：shell **不是**容器。它把 `cwd` 和 `HOME` 固定在工作区，并做静态路径检查。运行时拼接出来的路径、以及访问网络，仍然可能发生。只在你信任的任务上使用 `run_shell`。

### 4. 自己改一处并让评测失败一次（可选）

把 `evals/tasks/01_fix_add/task.json` 里的 `new` 暂时改错（例如仍写成减法），运行 `python -m evals.run`。分数应变为 `score 2/3`，`fix-add` 为 `FAIL`，标准错误里能看到黄金检查的断言。改回后分数恢复为 `3/3`。不要把改错的版本提交。

## 思考题

1. 为什么干跑回放不能代替重新执行工具？什么时候回放是更合适的审计方式？
2. `clear-debug` 里第一次 `lint_python` 的 `ok` 为什么是 false？如果代理在那一次之后就给出最终答案，评测为什么仍然不会通过？
3. 把 API 密钥放进 `llm.api_key` 和放进 `OPENAI_API_KEY`，哪一个会写进会话日志？阅读 `session_start` 事件验证你的判断（日志里应出现模型名和主机名，不应出现密钥）。
4. 如果模型调用 `run_shell` 执行 `python3 -c 'open("/etc/passwd")'`，静态检查为什么能拒绝它？构造一个静态检查看不到、但仍然危险的命令，并说明这为什么意味着 shell 不能当成安全沙箱。

## 可以提交的材料

- `pytest` 与 `python -m evals.run` 的输出（应为 `score 3/3`）。
- 一份会话转录，并标出一对 `call_id` 相同的 `tool_call` 与 `tool_result`。
- 对思考题 2 和 4 的简短回答。
