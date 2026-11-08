"""Workspace confinement for file paths and shell commands.

File tools resolve every path and refuse anything that is not inside the
workspace, including symlink targets. The shell jail is stricter than "cwd is
the workspace" but it is still not a sandbox: it pins HOME and the working
directory, then rejects commands whose text contains an absolute path outside
the workspace, a ``..`` segment that resolves outside it, or a ``~user`` path.
A command can still reach the network or build a path at runtime out of
pieces the static check cannot see. Treat the shell as powerful.
"""

from __future__ import annotations

import re
from pathlib import Path

from dm_agent.errors import PathJailError

# Characters that end a shell word for the purpose of the static scan.
_BOUNDARIES = set(" \t\n\r\"'`;&|<>()")

_URL_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9+.\-]*)://[^\s\"'`<>|]+")
# A slash that starts a path, not the slash inside ``foo/bar`` or ``~/note``.
_ABS_RE = re.compile(r"(?<![\w.+~-])/[^\s\"'`;&|<>()$]*")
_TILDE_RE = re.compile(r"(?:^|[\s\"'`=<>|(])(~[^\s\"'`;&|<>()$]*)")


def resolve_in_workspace(workspace: Path, raw: str) -> Path:
    """Resolve ``raw`` and require the result to stay inside ``workspace``.

    The returned path is absolute, with symlinks expanded. Callers should use
    it for IO so a symlink planted inside the workspace cannot point reads or
    writes somewhere else.
    """
    if not isinstance(raw, str) or raw.strip() == "":
        raise PathJailError("path is empty")
    if "\x00" in raw:
        raise PathJailError("path contains a NUL byte")
    if "\n" in raw or "\r" in raw:
        raise PathJailError("path contains a newline")

    root = workspace.resolve()
    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    if not is_within(root, resolved):
        raise PathJailError(f"path escapes the workspace: {raw}")
    return resolved


def is_within(root: Path, candidate: Path) -> bool:
    """Return True when ``candidate`` is ``root`` or a path under it."""
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def check_shell_command(command: str, workspace: Path) -> None:
    """Raise ``PathJailError`` when a command's text tries to leave ``workspace``.

    The check is static. It does not execute the command and it does not
    understand every shell expansion. See the module docstring.
    """
    if not isinstance(command, str) or command.strip() == "":
        raise PathJailError("command is empty")
    if "\x00" in command:
        raise PathJailError("command contains a NUL byte")
    if len(command) > 20_000:
        raise PathJailError("command is longer than 20000 characters")

    root = workspace.resolve()
    for token in _tilde_tokens(command):
        if token == "~" or token.startswith("~/") or token.startswith("~\\"):
            continue
        raise PathJailError(
            f"refusing tilde path that is not the workspace home: {token}"
        )

    scanned = _mask_nonfile_urls(command)
    for raw in _absolute_tokens(scanned):
        try:
            resolved = Path(raw).resolve()
        except OSError as exc:
            raise PathJailError(f"cannot resolve path: {raw}") from exc
        if not is_within(root, resolved):
            raise PathJailError(f"path escapes the workspace: {raw}")

    for raw in _dotdot_tokens(command):
        try:
            resolved = (root / raw).resolve()
        except OSError as exc:
            raise PathJailError(f"cannot resolve path: {raw}") from exc
        if not is_within(root, resolved):
            raise PathJailError(f"path escapes the workspace: {raw}")


def _mask_nonfile_urls(command: str) -> str:
    """Blank http(s) and similar URLs so their slashes are not paths.

    ``file://`` URLs are left intact because they name local files.
    Length is preserved so the rest of the command stays aligned for reading.
    """

    def repl(match: re.Match[str]) -> str:
        if match.group(1).lower() == "file":
            return match.group(0)
        return " " * len(match.group(0))

    return _URL_RE.sub(repl, command)


def _absolute_tokens(command: str) -> list[str]:
    found: list[str] = []
    for match in _ABS_RE.finditer(command):
        token = match.group(0).rstrip(".,:")
        if token == "":
            continue
        # A lone slash is the filesystem root. Longer tokens are paths.
        if token != "/" and not token.startswith("/"):
            continue
        found.append(token)
    return found


def _tilde_tokens(command: str) -> list[str]:
    return [match.group(1) for match in _TILDE_RE.finditer(command)]


def _dotdot_tokens(command: str) -> list[str]:
    found: list[str] = []
    seen: set[tuple[int, int]] = set()
    for match in re.finditer(r"\.\.", command):
        start = match.start()
        end = match.end()
        while start > 0 and command[start - 1] not in _BOUNDARIES:
            start -= 1
        while end < len(command) and command[end] not in _BOUNDARIES:
            end += 1
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        token = command[start:end].rstrip(".,:")
        if _has_dotdot_segment(token):
            found.append(token)
    return found


def _has_dotdot_segment(token: str) -> bool:
    parts = re.split(r"[/\\]", token)
    return ".." in parts
