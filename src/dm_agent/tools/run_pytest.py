"""run_pytest: run pytest inside the workspace and return a structured summary.

The command is an argument vector, not a shell string, so the model cannot
smuggle extra shell syntax through this tool. The test path still has to
resolve inside the workspace. Pytest itself is optional: if the interpreter
cannot import it, the tool says so instead of pretending the tests passed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from dm_agent.safety import resolve_in_workspace
from dm_agent.tools.registry import Tool, ToolResult

_SUMMARY_RE = {
    "passed": re.compile(r"(\d+)\s+passed\b"),
    "failed": re.compile(r"(\d+)\s+failed\b"),
    "errors": re.compile(r"(\d+)\s+errors?\b"),
}


def run_pytest_tool(workspace: Path, *, timeout: float) -> Tool:
    root = workspace.resolve()
    default_timeout = timeout
    return Tool(
        name="run_pytest",
        description=(
            "Run pytest on a file or directory inside the workspace. "
            "Returns status (passed, failed, error, or timeout), the exit code, "
            "and passed/failed/error counts parsed from the pytest summary. "
            "Uses the current Python interpreter. Paths outside the workspace are refused. "
            "Optional keyword is forwarded to pytest -k. Timeout is in seconds, max 120."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Test file or directory relative to the workspace. Default is the root.",
                },
                "keyword": {
                    "type": "string",
                    "description": "Optional pytest -k expression.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds. Default comes from shell_timeout_seconds. Maximum 120.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=lambda args: run_pytest(root, args, default_timeout=default_timeout),
    )


def run_pytest(workspace: Path, args: dict[str, Any], *, default_timeout: float) -> ToolResult:
    """Execute pytest with cwd set to the workspace and return a structured report."""
    raw = args.get("path", ".")
    if raw is None:
        raw = "."
    if not isinstance(raw, str) or raw.strip() == "":
        return ToolResult(ok=False, output="path must be a string")
    keyword = args.get("keyword")
    if keyword is not None and not isinstance(keyword, str):
        return ToolResult(ok=False, output="keyword must be a string")
    if isinstance(keyword, str) and ("\n" in keyword or "\x00" in keyword or len(keyword) > 200):
        return ToolResult(ok=False, output="keyword is empty, too long, or contains a newline")
    timeout = _timeout(args.get("timeout", default_timeout), default_timeout)
    if isinstance(timeout, ToolResult):
        return timeout

    root = workspace.resolve()
    target = resolve_in_workspace(root, raw)
    if not target.exists():
        return ToolResult(ok=False, output=f"Path not found: {raw}")
    relative = target.relative_to(root)
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(relative),
        "-q",
        "--tb=short",
        "-p",
        "no:cacheprovider",
        "--rootdir",
        str(root),
        "--color=no",
        "-o",
        "addopts=",
    ]
    if keyword:
        command.extend(["-k", keyword])

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["NO_COLOR"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _text(exc.stdout)
        stderr = _text(exc.stderr)
        body = _format(
            status="timeout",
            exit_code=None,
            counts=(0, 0, 0),
            output=f"{stdout}\n{stderr}\nTimed out after {timeout:g}s and the process was killed",
        )
        return ToolResult(ok=False, output=body)

    combined = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0 and "No module named pytest" in combined:
        return ToolResult(
            ok=False,
            output=(
                "status: error\n"
                "exit_code: " + str(completed.returncode) + "\n"
                "passed: 0\nfailed: 0\nerrors: 0\n"
                "pytest is not installed for this interpreter. "
                "Install the dev extra: python -m pip install -e \".[dev]\""
            ),
        )
    counts = parse_pytest_counts(combined)
    status = "passed" if completed.returncode == 0 else "failed"
    return ToolResult(
        ok=completed.returncode == 0,
        output=_format(
            status=status,
            exit_code=completed.returncode,
            counts=counts,
            output=combined.strip(),
        ),
    )


def parse_pytest_counts(text: str) -> tuple[int, int, int]:
    """Pull passed/failed/error counts from the tail of pytest's report."""
    tail = "\n".join(text.splitlines()[-30:])
    values: list[int] = []
    for name in ("passed", "failed", "errors"):
        found = _SUMMARY_RE[name].findall(tail)
        values.append(int(found[-1]) if found else 0)
    return values[0], values[1], values[2]


def _format(
    *,
    status: str,
    exit_code: int | None,
    counts: tuple[int, int, int],
    output: str,
) -> str:
    passed, failed, errors = counts
    code = "" if exit_code is None else str(exit_code)
    body = output if output else "<empty>"
    return (
        f"status: {status}\n"
        f"exit_code: {code}\n"
        f"passed: {passed}\n"
        f"failed: {failed}\n"
        f"errors: {errors}\n"
        f"--- output ---\n"
        f"{body}"
    )


def _timeout(value: Any, default: float) -> float | ToolResult:
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ToolResult(ok=False, output="timeout must be a number")
    number = float(value)
    if number <= 0:
        return ToolResult(ok=False, output="timeout must be > 0")
    return min(number, 120.0)


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
