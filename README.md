# dm-agent

[![CI](https://github.com/KDB1223/dm-code-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/KDB1223/dm-code-agent/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](src/dm_agent/__init__.py)

A local-first coding agent that edits one workspace with a ReAct loop and writes every step to an append-only JSONL log you can inspect and replay.

Hosted chat hides the tool trace. dm-agent keeps the trace on disk, next to the project: the plan, the tool call, the observation, and the final answer. The kernel is small enough to read in a lab.

- **ReAct loop.** Plan, call a tool, read the observation, repeat until a final answer or the step limit.
- **Workspace tools.** Read, write, edit, list, and search. The shell refuses commands that point outside the project.
- **Course tools.** `lint_python` reports syntax errors, bare `except`, and leftover `breakpoint()` calls. `run_pytest` returns a structured pass/fail summary.
- **Audit log.** Append-only JSONL under `.dm-agent/sessions/`, with `list`, `show`, and a dry `replay` that does not re-run tools.
- **Offline evals.** Three toy tasks with golden checks. `python -m evals.run` prints `score 3/3` without an API key.

**Topics:** `coding-agent` `react` `auditability` `local-first` `llm` `education`

This is an original implementation inspired by the idea of a local-first, auditable coding agent: a ReAct loop, workspace tools, and an inspectable log. It is not a fork of hwfengcs/DM-Code-Agent, and none of that project's code is included here.

中文说明：[README.zh-CN.md](README.zh-CN.md)。课程实验：[docs/lab-zh.md](docs/lab-zh.md)。

## What this adds beyond a minimal baseline

A minimal baseline is the loop, the file/search/shell tools, and the JSONL log. This repo also ships:

| Addition | Where |
| --- | --- |
| `lint_python` and `run_pytest` | `src/dm_agent/tools/` |
| Offline evals with golden `check.py` scripts | `evals/` |
| Chinese guide and a course lab | `README.zh-CN.md`, `docs/lab-zh.md` |
| Demo workspace, example config, CI | `examples/`, `.github/workflows/ci.yml` |

## Quickstart

From a clean checkout, about a minute, no API key:

```bash
git clone https://github.com/KDB1223/dm-code-agent.git
cd dm-code-agent
python -m pip install -e ".[dev]"
python -m evals.run
```

Stdout is the scoreboard (also checked in as `evals/expected_output.txt`):

```text
fix-add        PASS
write-greet    PASS
clear-debug    PASS
score 3/3
```

Print the tool trace for one task. Stdout is the dry transcript. A one-line `PASS` or `FAIL` goes to stderr:

```bash
python -m evals.run --transcript fix-add
```

### Live run on the demo bug

`examples/demo_workspace/math_utils.py` implements `add()` with subtraction. The test expects a sum.

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"
# Any OpenAI-compatible server. Omit this for api.openai.com.
# export OPENAI_BASE_URL="https://api.openai.com/v1"

python -m dm_agent run --workspace examples/demo_workspace \
  "$(cat examples/demo_workspace/TASK.txt)"
python -m dm_agent sessions list --workspace examples/demo_workspace
python -m dm_agent sessions replay SESSION_ID --workspace examples/demo_workspace
```

`pip install` also installs a `dm-agent` script. If it is not on `PATH`, use `python -m dm_agent`.

The session file is:

```text
examples/demo_workspace/.dm-agent/sessions/<session-id>.jsonl
```

`replay` reprints the transcript from that file. It does not call the model and does not edit the workspace again.

## Sample transcript

Recorded from `python -m evals.run --transcript fix-add`. The model turns are the script in `evals/tasks/01_fix_add/task.json`, so a second run follows the same tool sequence. The temporary workspace path changes; the events do not.

```text
# session 20261108T100749Z-3c6c4a
[2026-11-08T10:07:49.237Z] session_start model=scripted-harness workspace=/tmp/dm-eval-fix-add-md11xv6p/workspace

[2026-11-08T10:07:49.238Z] user
Fix add() in math_utils.py so it returns the sum of its arguments. Do not change the function signature.

[2026-11-08T10:07:49.239Z] step 1 thought
I will read math_utils.py before changing it.

[2026-11-08T10:07:49.239Z] step 1 tool_call read_file {"path": "math_utils.py"}

[2026-11-08T10:07:49.239Z] step 1 tool_result read_file ok
math_utils.py  lines 1-2 of 2
     1|def add(a, b):
     2|    return a - b

[2026-11-08T10:07:49.240Z] step 2 thought
The body subtracts. I will switch it to addition.

[2026-11-08T10:07:49.240Z] step 2 tool_call edit_file {"path": "math_utils.py", "old": "return a - b", "new": "return a + b"}

[2026-11-08T10:07:49.240Z] step 2 tool_result edit_file ok
Updated math_utils.py: replaced 1 occurrence(s)

[2026-11-08T10:07:49.241Z] step 3 thought
I will lint the file I just edited.

[2026-11-08T10:07:49.241Z] step 3 tool_call lint_python {"path": "math_utils.py"}

[2026-11-08T10:07:49.241Z] step 3 tool_result lint_python ok
status: ok
files: 1
issues: 0

[2026-11-08T10:07:49.241Z] step 4 final
add() now returns the sum of a and b. lint_python reported no issues.

# replay complete (tool results printed from the log; tools were not re-run)
```

## Compared with an opaque chat window

| | dm-agent | Typical hosted chat |
| --- | --- | --- |
| Where the code runs | On your machine, in one workspace | Whatever the product's sandbox does |
| What is stored | Append-only JSONL you can diff and replay | A transcript the vendor keeps |
| Tool trace | Every plan, call, and observation has a step id | Often hidden, or not downloadable as data |
| Teaching | A lab can point at one `call_id` and its result | Hard to grade the intermediate steps |
| Setup | `pip install`, a TOML file, an API key only for live runs | An account and a hosted UI |
| Offline check | `python -m evals.run` needs no key | Needs the service |

## Configure

Copy [`examples/config.example.toml`](examples/config.example.toml) to `dm-agent.toml` in the workspace, or pass `--config`.

Precedence, lowest to highest: built-in defaults, the TOML file, environment variables, CLI flags.

| Setting | Environment | CLI |
| --- | --- | --- |
| API key | `OPENAI_API_KEY` (preferred) | |
| Model | `OPENAI_MODEL` | `--model` |
| Base URL | `OPENAI_BASE_URL` | |
| Max model turns | `DM_AGENT_MAX_STEPS` | `--max-steps` |
| Workspace | `DM_AGENT_WORKSPACE` | `--workspace` |
| Session directory | `DM_AGENT_LOG_DIR` | `--log-dir` |
| Shell and pytest timeout | `DM_AGENT_SHELL_TIMEOUT` | |

`provider = "openai"` means any server that accepts `POST {base_url}/chat/completions` with bearer auth and the usual tools array. A local llama.cpp, vLLM, or a hosted gateway all use the same client. The native Anthropic API is not implemented.

The API key is never written to the session log. `session_start` records the model name and the base URL's scheme and host only.

File discovery order when `--config` is omitted: `<workspace>/dm-agent.toml`, `./dm-agent.toml`, then `$XDG_CONFIG_HOME/dm-agent/config.toml` (or `~/.config/dm-agent/config.toml`).

## Commands

```text
python -m dm_agent run "task" --workspace PATH
python -m dm_agent run - --workspace PATH          # task text from stdin
python -m dm_agent sessions list --workspace PATH
python -m dm_agent sessions show SESSION --workspace PATH
python -m dm_agent sessions replay SESSION --workspace PATH
python -m dm_agent sessions html SESSION --workspace PATH
```

`SESSION` may be the full id or a unique prefix. Exit codes: `0` finished, `1` runtime or log error, `2` bad config or usage, `3` stopped at `max_steps`. Progress lines go to stderr. The final answer goes to stdout.

## Tools

| Tool | Role |
| --- | --- |
| `read_file` | Numbered lines of a UTF-8 file. Optional `offset` and `limit`. |
| `write_file` | Create or replace a file, including parent directories inside the workspace. |
| `edit_file` | Exact string replace. Fails if `old` matches more than once unless `replace_all` is true. |
| `list_dir` | Directory listing. Symlinks that leave the workspace are labeled and not followed. |
| `search` | Regex or literal search. Skips `.git`, virtualenvs, and `.dm-agent`. |
| `run_shell` | Shell command with `cwd` and `HOME` set to the workspace. See Safety. |
| `lint_python` | Structured report: `syntax-error`, `bare-except`, `debug-call`. |
| `run_pytest` | `pytest` on a workspace path. Returns `status`, exit code, and passed/failed/error counts. |

`run_pytest` uses the current interpreter. Install pytest with the `dev` extra (`pip install -e ".[dev]"`). If pytest is missing, the tool says so.

## Session log

Each line is one JSON object. Lines are appended and `fsync`'d. A crash keeps every event that was already written.

```json
{"ts":"2026-11-08T10:07:49.240Z","session_id":"20261108T100749Z-3c6c4a","seq":6,"step":2,"kind":"tool_call","call_id":"call_2_1","tool":"edit_file","arguments":{"path":"math_utils.py","old":"return a - b","new":"return a + b"}}
```

`kind` is one of `session_start`, `user`, `thought`, `tool_call`, `tool_result`, `final`, `max_steps`, `error`. A `tool_call` and its `tool_result` share `step` and `call_id`.

`sessions show` prints every field. `sessions replay` prints the narrative transcript and states that tools were not re-run. `sessions html` writes a self-contained HTML file with the event text escaped.

## Safety

File tools resolve every path and refuse anything that is not inside the workspace, including a symlink whose target is outside. `run_pytest` uses the same check.

`run_shell` is more powerful and is not a container, seccomp profile, or user namespace:

- The process starts in the workspace. `HOME` is the workspace, so `~/file` stays inside. `~otheruser` is refused.
- Before the process starts, the command text is scanned. An absolute path must resolve inside the workspace. A `..` segment that resolves outside is refused. `file://` URLs are treated as paths. Ordinary `https://` URLs are not.
- Call programs by name (`python3`, `pytest`) so the command does not have to mention `/usr/bin/...`.
- Output is capped and the process group is killed on timeout.

A command can still reach the network, or build an outside path at runtime from pieces the static scan cannot see (`"/" + "etc/passwd"`). Run the shell only on workspaces and tasks you trust.

## Layout

```text
src/dm_agent/agent.py          ReAct loop
src/dm_agent/llm.py            OpenAI-compatible HTTP client (stdlib only)
src/dm_agent/session.py        Append-only JSONL
src/dm_agent/safety.py         Path jail and shell scan
src/dm_agent/tools/            read/write/edit/list/search/shell/lint/pytest
src/dm_agent/cli.py            dm-agent / python -m dm_agent
evals/                         Offline tasks, harness, golden checks
examples/demo_workspace/       add() bug for a live run
examples/config.example.toml
docs/lab-zh.md                 Course lab (Chinese)
```

There are no runtime dependencies. Python 3.11+ is enough. pytest is the only dev extra, and CI runs it on 3.11 and 3.12.

If the model does not return native tool calls, the loop also accepts a single JSON object `{"tool": "name", "arguments": {}}`, optionally inside a ` ```json ` fence. Any other assistant text is the final answer.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
python -m evals.run
```

See [CONTRIBUTING.md](CONTRIBUTING.md). The course write-up is [docs/lab-zh.md](docs/lab-zh.md).

## License

[MIT](LICENSE). Copyright 2026 dm-agent contributors.
