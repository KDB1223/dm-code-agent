"""lint_python: a small structured checker for workspace Python files.

This is intentionally not a wrapper around ruff or flake8. It uses the
standard-library parser so a classroom machine with no extra packages still
gets a deterministic report. The checks are the ones that show up constantly
in short coding exercises: the file does not parse, a bare ``except`` swallows
every error, or a ``breakpoint()`` / ``pdb.set_trace()`` was left behind.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dm_agent.safety import is_within, resolve_in_workspace
from dm_agent.tools.registry import Tool, ToolResult

_SKIP_DIRS = {
    ".git",
    ".dm-agent",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".hg",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
}
_MAX_FILES = 200
_MAX_FILE_BYTES = 1_000_000


@dataclass(frozen=True)
class Issue:
    """One finding. Columns are 1-based, matching the read_file gutter."""

    path: str
    line: int
    col: int
    code: str
    message: str


def lint_python_tool(workspace: Path) -> Tool:
    root = workspace.resolve()
    return Tool(
        name="lint_python",
        description=(
            "Lint Python files inside the workspace and return a structured report. "
            "Checks: syntax-error, bare-except, and debug-call (breakpoint() or pdb.set_trace()). "
            "status is ok when there are no issues. Does not follow directory symlinks. "
            "Paths outside the workspace are refused."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Python file or directory relative to the workspace. Default is the root.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=lambda args: lint_python(root, args),
    )


def lint_python(workspace: Path, args: dict[str, Any]) -> ToolResult:
    """Lint ``path`` and return a structured, line-oriented report."""
    raw = args.get("path", ".")
    if raw is None:
        raw = "."
    if not isinstance(raw, str) or raw.strip() == "":
        return ToolResult(ok=False, output="path must be a string")
    root = workspace.resolve()
    target = resolve_in_workspace(root, raw)
    if not target.exists():
        return ToolResult(ok=False, output=f"Path not found: {raw}")
    if target.is_file() and target.suffix != ".py":
        return ToolResult(ok=False, output=f"Not a Python file: {raw}")

    files, truncated = _python_files(root, target)
    issues: list[Issue] = []
    parsed = 0
    for path in files:
        found, did_parse = _lint_file(root, path)
        if did_parse:
            parsed += 1
        issues.extend(found)
    output = _render(parsed, issues, truncated)
    return ToolResult(ok=not issues, output=output)


def _python_files(workspace: Path, target: Path) -> tuple[list[Path], bool]:
    if target.is_file():
        return [target], False
    found: list[Path] = []
    truncated = False
    for dirpath, dirnames, filenames in os.walk(target, followlinks=False):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in _SKIP_DIRS and not (Path(dirpath) / name).is_symlink()
        ]
        for name in sorted(filenames):
            if not name.endswith(".py") or name.startswith(".dm-tmp-"):
                continue
            path = Path(dirpath) / name
            if path.is_symlink():
                try:
                    dest = path.resolve()
                except OSError:
                    continue
                if not is_within(workspace, dest):
                    continue
            found.append(path)
            if len(found) >= _MAX_FILES:
                truncated = True
                return found, truncated
    return found, truncated


def _lint_file(workspace: Path, path: Path) -> tuple[list[Issue], bool]:
    rel = str(path.relative_to(workspace)) if is_within(workspace, path) else path.name
    read_path = path
    if path.is_symlink():
        try:
            read_path = path.resolve()
        except OSError:
            return [Issue(rel, 1, 1, "unreadable", "cannot resolve symlink")], False
    try:
        size = read_path.stat().st_size
    except OSError:
        return [Issue(rel, 1, 1, "unreadable", "cannot stat file")], False
    if size > _MAX_FILE_BYTES:
        return [Issue(rel, 1, 1, "unreadable", "file is larger than 1MB")], False
    try:
        data = read_path.read_bytes()
    except OSError:
        return [Issue(rel, 1, 1, "unreadable", "cannot read file")], False
    if b"\x00" in data[:8192]:
        return [Issue(rel, 1, 1, "unreadable", "binary file")], False
    try:
        source = data.decode("utf-8")
    except UnicodeDecodeError:
        return [Issue(rel, 1, 1, "unreadable", "file is not valid UTF-8")], False
    return _lint_source(rel, source), True


def _lint_source(rel: str, source: str) -> list[Issue]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [
            Issue(
                rel,
                exc.lineno or 1,
                exc.offset or 1,
                "syntax-error",
                exc.msg or "syntax error",
            )
        ]
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append(
                Issue(
                    rel,
                    node.lineno or 1,
                    (node.col_offset or 0) + 1,
                    "bare-except",
                    "bare except swallows all exceptions",
                )
            )
        elif isinstance(node, ast.Call) and _is_debug_call(node):
            issues.append(
                Issue(
                    rel,
                    node.lineno or 1,
                    (node.col_offset or 0) + 1,
                    "debug-call",
                    "breakpoint() or pdb.set_trace() left in source",
                )
            )
    return issues


def _is_debug_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name) and func.id == "breakpoint":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "set_trace":
        value = func.value
        if isinstance(value, ast.Name) and value.id == "pdb":
            return True
    return False


def _render(files: int, issues: list[Issue], truncated: bool) -> str:
    ordered = sorted(issues, key=lambda item: (item.path, item.line, item.col, item.code))
    lines = [
        f"status: {'ok' if not ordered else 'fail'}",
        f"files: {files}",
        f"issues: {len(ordered)}",
    ]
    if truncated:
        lines.append(f"note: stopped after {_MAX_FILES} files")
    for issue in ordered:
        lines.append(f"{issue.path}:{issue.line}:{issue.col}: {issue.code}: {issue.message}")
    if files == 0 and not ordered:
        lines.append("No Python files found.")
    return "\n".join(lines)
