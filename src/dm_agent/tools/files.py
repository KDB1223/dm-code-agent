"""read_file, write_file, edit_file, and list_dir."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from dm_agent.safety import resolve_in_workspace
from dm_agent.tools.registry import Tool, ToolResult

_SKIP_LIST_NAMES = {".git", "__pycache__"}


def file_tools(workspace: Path) -> list[Tool]:
    """Tools that read and write files inside ``workspace``."""
    root = workspace.resolve()
    return [
        Tool(
            name="read_file",
            description=(
                "Read a UTF-8 text file inside the workspace. "
                "Returns numbered lines. Use offset and limit for long files. "
                "Paths outside the workspace are refused."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace root, or an absolute path inside it.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-based line to start at. Default 1.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to return. Default 400.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=lambda args: read_file(root, args),
        ),
        Tool(
            name="write_file",
            description=(
                "Create or overwrite a UTF-8 text file inside the workspace. "
                "Parent directories are created when they stay inside the workspace. "
                "Paths outside the workspace are refused."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination path inside the workspace."},
                    "content": {"type": "string", "description": "Full new file contents."},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=lambda args: write_file(root, args),
        ),
        Tool(
            name="edit_file",
            description=(
                "Replace an exact string in a workspace file. "
                "old must match exactly once unless replace_all is true. "
                "Include enough surrounding context to make the match unique."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File to edit, inside the workspace."},
                    "old": {"type": "string", "description": "Exact text to replace."},
                    "new": {"type": "string", "description": "Replacement text."},
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence. Default false.",
                    },
                },
                "required": ["path", "old", "new"],
                "additionalProperties": False,
            },
            handler=lambda args: edit_file(root, args),
        ),
        Tool(
            name="list_dir",
            description=(
                "List entries in a workspace directory. "
                "Symlinks that point outside the workspace are labeled and not followed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory relative to the workspace. Default is the root.",
                    },
                    "max_entries": {
                        "type": "integer",
                        "description": "Maximum entries to return. Default 200.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=lambda args: list_dir(root, args),
        ),
    ]


def read_file(workspace: Path, args: dict[str, Any]) -> ToolResult:
    path = _require_str(args, "path")
    if isinstance(path, ToolResult):
        return path
    offset = _optional_int(args, "offset", default=1)
    limit = _optional_int(args, "limit", default=400)
    if isinstance(offset, ToolResult):
        return offset
    if isinstance(limit, ToolResult):
        return limit
    if offset < 1 or limit < 1:
        return ToolResult(ok=False, output="offset and limit must be >= 1")
    limit = min(limit, 2000)

    target = resolve_in_workspace(workspace, path)
    if not target.is_file():
        return ToolResult(ok=False, output=f"Not a file: {path}")
    size = target.stat().st_size
    if size > 2_000_000:
        return ToolResult(
            ok=False,
            output="File is larger than 2MB. Read a smaller range with offset and limit, or use search.",
        )
    data = target.read_bytes()
    if b"\x00" in data[:8192]:
        return ToolResult(ok=False, output=f"Refusing to read binary file: {path}")
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    total = len(lines)
    start = offset - 1
    if start > total:
        return ToolResult(ok=False, output=f"{path} has {total} lines; offset {offset} is past the end")
    chunk = lines[start : start + limit]
    if chunk:
        end = offset + len(chunk) - 1
        numbered = "\n".join(f"{start + index + 1:6d}|{line}" for index, line in enumerate(chunk))
        header = f"{path}  lines {offset}-{end} of {total}"
        body = header + "\n" + numbered
    else:
        body = f"{path}  lines 0-0 of 0\n<empty>"
    if start + len(chunk) < total:
        body += f"\n...[{total - (start + len(chunk))} more lines; raise offset or limit]"
    return ToolResult(ok=True, output=body)


def write_file(workspace: Path, args: dict[str, Any]) -> ToolResult:
    path = _require_str(args, "path")
    if isinstance(path, ToolResult):
        return path
    content = args.get("content")
    if not isinstance(content, str):
        return ToolResult(ok=False, output="content must be a string")
    target = resolve_in_workspace(workspace, path)
    if target.exists() and not target.is_file():
        return ToolResult(ok=False, output=f"Not a file: {path}")
    if target.parent.exists() and not target.parent.is_dir():
        return ToolResult(ok=False, output=f"Parent is not a directory: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # mkdir of a not-yet-existing chain is lexical; re-check the parent after create.
    resolve_in_workspace(workspace, str(target.parent))
    _atomic_write(target, content)
    return ToolResult(ok=True, output=f"Wrote {len(content.encode('utf-8'))} bytes to {path}")


def edit_file(workspace: Path, args: dict[str, Any]) -> ToolResult:
    path = _require_str(args, "path")
    if isinstance(path, ToolResult):
        return path
    old = args.get("old")
    new = args.get("new")
    if not isinstance(old, str) or not isinstance(new, str):
        return ToolResult(ok=False, output="old and new must be strings")
    if old == "":
        return ToolResult(ok=False, output="old must not be empty")
    replace_all = args.get("replace_all", False)
    if not isinstance(replace_all, bool):
        return ToolResult(ok=False, output="replace_all must be a boolean")

    target = resolve_in_workspace(workspace, path)
    if not target.is_file():
        return ToolResult(ok=False, output=f"Not a file: {path}")
    data = target.read_bytes()
    if b"\x00" in data[:8192]:
        return ToolResult(ok=False, output=f"Refusing to edit binary file: {path}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return ToolResult(ok=False, output=f"File is not valid UTF-8: {path}")
    count = text.count(old)
    if count == 0:
        return ToolResult(ok=False, output="old string was not found in the file")
    if count > 1 and not replace_all:
        return ToolResult(
            ok=False,
            output=(
                f"old string matched {count} times. "
                "Include more context so it is unique, or pass replace_all=true."
            ),
        )
    updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    if updated == text:
        return ToolResult(ok=True, output="No change; old and new are identical")
    _atomic_write(target, updated)
    replaced = count if replace_all else 1
    return ToolResult(ok=True, output=f"Updated {path}: replaced {replaced} occurrence(s)")


def list_dir(workspace: Path, args: dict[str, Any]) -> ToolResult:
    raw = args.get("path", ".")
    if raw is None:
        raw = "."
    if not isinstance(raw, str):
        return ToolResult(ok=False, output="path must be a string")
    max_entries = _optional_int(args, "max_entries", default=200)
    if isinstance(max_entries, ToolResult):
        return max_entries
    if max_entries < 1:
        return ToolResult(ok=False, output="max_entries must be >= 1")
    max_entries = min(max_entries, 1000)

    target = resolve_in_workspace(workspace, raw)
    if not target.is_dir():
        return ToolResult(ok=False, output=f"Not a directory: {raw}")

    entries = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    rows: list[str] = []
    for child in entries:
        if child.name in _SKIP_LIST_NAMES or child.name.startswith(".dm-tmp-"):
            continue
        if child.is_symlink():
            kind = "link"
            note = ""
            try:
                dest = child.resolve()
                dest.relative_to(workspace)
                note = f" -> {dest.name}"
            except (OSError, ValueError):
                note = " -> outside workspace"
            rows.append(f"{child.name}{note}  {kind}")
            continue
        if child.is_dir():
            rows.append(f"{child.name}/  dir")
        elif child.is_file():
            rows.append(f"{child.name}  file  {child.stat().st_size}B")
        else:
            rows.append(f"{child.name}  other")
    shown = rows[:max_entries]
    header = f"{raw}  {len(rows)} entries"
    body = header if not shown else header + "\n" + "\n".join(shown)
    if len(rows) > max_entries:
        body += f"\n...[{len(rows) - max_entries} more entries]"
    return ToolResult(ok=True, output=body)


def _atomic_write(path: Path, text: str) -> None:
    """Write text via a temp file in the same directory, then rename it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".dm-tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _require_str(args: dict[str, Any], key: str) -> str | ToolResult:
    value = args.get(key)
    if not isinstance(value, str) or value.strip() == "":
        return ToolResult(ok=False, output=f"{key} is required")
    return value


def _optional_int(args: dict[str, Any], key: str, *, default: int) -> int | ToolResult:
    if key not in args or args[key] is None:
        return default
    value = args[key]
    if isinstance(value, bool) or not isinstance(value, int):
        return ToolResult(ok=False, output=f"{key} must be an integer")
    return value
