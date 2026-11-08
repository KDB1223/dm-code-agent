"""run_shell: a workspace-confined command runner.

The process starts with cwd and HOME set to the workspace. Before it starts,
``check_shell_command`` rejects absolute paths outside the workspace, ``..``
escapes, and ``~user`` paths. Output is capped and the process group is
killed on timeout or when the cap is hit.

This is not a container, seccomp profile, or user-namespace sandbox. A command
can still open the network or assemble an outside path at runtime from pieces
the static check does not see. Use it on workspaces and tasks you trust.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dm_agent.limits import MAX_TOOL_OUTPUT
from dm_agent.safety import check_shell_command
from dm_agent.tools.registry import Tool, ToolResult


@dataclass
class ShellOutcome:
    """Result of one confined command."""

    returncode: int | None
    output: str
    status: str  # "exited", "timeout", or "truncated"


def shell_tool(workspace: Path, *, timeout: float) -> Tool:
    root = workspace.resolve()
    default_timeout = timeout
    return Tool(
        name="run_shell",
        description=(
            "Run a shell command with the working directory and HOME set to the workspace root. "
            "Absolute paths must point inside the workspace. '..' segments that leave the "
            "workspace, and ~user paths, are refused before the command starts. "
            "Prefer relative paths. The shell is not a full sandbox: do not use it to "
            "read or write outside the workspace. Optional timeout is in seconds (max 120)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run inside the workspace.",
                },
                "timeout": {
                    "type": "number",
                    "description": (
                        "Timeout in seconds. Default comes from config (usually 30). Maximum 120."
                    ),
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        handler=lambda args: run_shell(root, args, default_timeout=default_timeout),
    )


def run_shell(workspace: Path, args: dict[str, Any], *, default_timeout: float) -> ToolResult:
    command = args.get("command")
    if not isinstance(command, str) or command.strip() == "":
        return ToolResult(ok=False, output="command is required")
    timeout = args.get("timeout", default_timeout)
    if timeout is None:
        timeout = default_timeout
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        return ToolResult(ok=False, output="timeout must be a number")
    timeout = float(timeout)
    if timeout <= 0:
        return ToolResult(ok=False, output="timeout must be > 0")
    timeout = min(timeout, 120.0)

    root = workspace.resolve()
    check_shell_command(command, root)
    outcome = execute_shell(
        command,
        cwd=str(root),
        env=_workspace_env(root),
        timeout=timeout,
        output_limit=MAX_TOOL_OUTPUT,
    )
    ok = outcome.status == "exited" and outcome.returncode == 0
    return ToolResult(ok=ok, output=outcome.output)


def execute_shell(
    command: str,
    *,
    cwd: str,
    env: dict[str, str],
    timeout: float,
    output_limit: int,
) -> ShellOutcome:
    """Run ``command`` and stop it on timeout or oversized output.

    stdout and stderr are each capped at ``output_limit`` bytes. The pipes
    are unbuffered so ``os.read`` observes what the process actually wrote.
    """
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        bufsize=0,
    )
    assert proc.stdout is not None and proc.stderr is not None
    stdout_buf = bytearray()
    stderr_buf = bytearray()
    streams = {
        proc.stdout.fileno(): stdout_buf,
        proc.stderr.fileno(): stderr_buf,
    }
    status = "exited"
    returncode: int | None = None
    deadline = time.monotonic() + timeout
    try:
        while streams:
            remain = deadline - time.monotonic()
            if remain <= 0:
                status = "timeout"
                _kill_group(proc)
                break
            try:
                ready, _, _ = select.select(list(streams), [], [], min(0.2, remain))
            except (ValueError, OSError):
                break
            if not ready:
                if proc.poll() is not None:
                    break
                continue
            if _absorb(ready, streams, output_limit):
                status = "truncated"
                _kill_group(proc)
                break
        if status == "exited":
            if _drain(proc.stdout, stdout_buf, output_limit) or _drain(proc.stderr, stderr_buf, output_limit):
                status = "truncated"
                _kill_group(proc)
        try:
            returncode: int | None = proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            returncode = proc.wait(timeout=2)
            if status == "exited":
                status = "timeout"
    finally:
        for pipe in (proc.stdout, proc.stderr):
            if pipe is None:
                continue
            try:
                pipe.close()
            except OSError:
                pass
        if proc.poll() is None:
            _kill_group(proc)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

    rendered = _render(
        returncode,
        stdout_buf.decode("utf-8", errors="replace"),
        stderr_buf.decode("utf-8", errors="replace"),
        status,
        timeout,
    )
    return ShellOutcome(returncode=returncode, output=rendered, status=status)


def _absorb(ready: list[int], streams: dict[int, bytearray], output_limit: int) -> bool:
    """Read ready pipes. Return True when a buffer hits ``output_limit``."""
    for fd in ready:
        buf = streams.get(fd)
        if buf is None:
            continue
        kind, chunk = _read_fd(fd)
        if kind in {"eof", "error"}:
            streams.pop(fd, None)
            continue
        if kind != "data":
            continue
        room = output_limit - len(buf)
        if room > 0:
            buf += chunk[:room]
        if len(chunk) > max(room, 0):
            return True
    return False


def _drain(pipe: Any, buf: bytearray, output_limit: int) -> bool:
    """Non-blocking read of bytes left after the child exits. True if capped."""
    if pipe is None:
        return False
    try:
        fd = pipe.fileno()
    except OSError:
        return False
    while len(buf) < output_limit:
        kind, chunk = _read_fd(fd, nonblocking=True)
        if kind != "data":
            return False
        room = output_limit - len(buf)
        buf += chunk[:room]
        if len(chunk) > room:
            return True
    return False


def _read_fd(fd: int, *, nonblocking: bool = False) -> tuple[str, bytes]:
    """Return ``(kind, chunk)`` where kind is data, eof, again, or error."""
    try:
        if nonblocking:
            os.set_blocking(fd, False)
        chunk = os.read(fd, 8192)
    except BlockingIOError:
        return "again", b""
    except OSError:
        return "error", b""
    finally:
        if nonblocking:
            try:
                os.set_blocking(fd, True)
            except OSError:
                pass
    if chunk == b"":
        return "eof", b""
    return "data", chunk


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except OSError:
            pass


def _render(
    returncode: int | None,
    stdout: str,
    stderr: str,
    status: str,
    timeout: float,
) -> str:
    parts: list[str] = []
    if status == "timeout":
        parts.append(f"timed out after {timeout:g}s and the process was killed")
    elif status == "truncated":
        parts.append("output exceeded the limit and the process was stopped")
    else:
        parts.append(f"exit: {returncode}")
    parts.append("--- stdout ---")
    parts.append(stdout if stdout else "<empty>")
    parts.append("--- stderr ---")
    parts.append(stderr if stderr else "<empty>")
    return "\n".join(parts)


def _workspace_env(workspace: Path) -> dict[str, str]:
    env = dict(os.environ)
    root = str(workspace)
    env["HOME"] = root
    env["PWD"] = root
    env["CDPATH"] = ""
    return env
