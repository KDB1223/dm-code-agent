"""grep-style search over workspace text files."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from dm_agent.safety import is_within, resolve_in_workspace
from dm_agent.tools.registry import Tool, ToolResult

SKIP_DIRS = {
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
_MAX_FILE_BYTES = 1_000_000


def search_tool(workspace: Path) -> Tool:
    root = workspace.resolve()
    return Tool(
        name="search",
        description=(
            "Search workspace text files for a regular expression (or a literal string). "
            "Skips VCS directories, virtualenvs, and the session log directory. "
            "Does not follow directory symlinks or read files that point outside the workspace."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Python regular expression, unless literal is true.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search. Default is the workspace root.",
                },
                "glob": {
                    "type": "string",
                    "description": "Optional fnmatch filter, for example '*.py'. Matched against the filename and the relative path.",
                },
                "literal": {
                    "type": "boolean",
                    "description": "Treat pattern as plain text. Default false.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum matching lines. Default 50.",
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
        handler=lambda args: search(root, args),
    )


def search(workspace: Path, args: dict[str, Any]) -> ToolResult:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or pattern == "":
        return ToolResult(ok=False, output="pattern is required")
    literal = args.get("literal", False)
    if not isinstance(literal, bool):
        return ToolResult(ok=False, output="literal must be a boolean")
    glob = args.get("glob")
    if glob is not None and not isinstance(glob, str):
        return ToolResult(ok=False, output="glob must be a string")
    max_results = args.get("max_results", 50)
    if max_results is None:
        max_results = 50
    if isinstance(max_results, bool) or not isinstance(max_results, int):
        return ToolResult(ok=False, output="max_results must be an integer")
    if max_results < 1:
        return ToolResult(ok=False, output="max_results must be >= 1")
    max_results = min(max_results, 200)

    raw_path = args.get("path", ".")
    if raw_path is None:
        raw_path = "."
    if not isinstance(raw_path, str):
        return ToolResult(ok=False, output="path must be a string")
    root = resolve_in_workspace(workspace, raw_path)
    if not root.exists():
        return ToolResult(ok=False, output=f"Path not found: {raw_path}")

    try:
        compiled = re.compile(re.escape(pattern) if literal else pattern)
    except re.error as exc:
        return ToolResult(ok=False, output=f"Bad pattern: {exc}")

    matches: list[str] = []
    files_seen = 0
    truncated = False

    def consider(file_path: Path) -> None:
        nonlocal files_seen, truncated
        if truncated:
            return
        if file_path.is_symlink():
            try:
                dest = file_path.resolve()
            except OSError:
                return
            if not is_within(workspace, dest) or not dest.is_file():
                return
            file_path = dest
        elif not file_path.is_file():
            return
        rel = str(file_path.relative_to(workspace))
        if glob and not (fnmatch.fnmatch(file_path.name, glob) or fnmatch.fnmatch(rel, glob)):
            return
        try:
            size = file_path.stat().st_size
        except OSError:
            return
        if size > _MAX_FILE_BYTES:
            return
        files_seen += 1
        try:
            data = file_path.read_bytes()
        except OSError:
            return
        if b"\x00" in data[:8192]:
            return
        text = data.decode("utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if compiled.search(line):
                snippet = line if len(line) <= 400 else line[:400] + "..."
                matches.append(f"{rel}:{lineno}: {snippet}")
                if len(matches) >= max_results:
                    truncated = True
                    return

    if root.is_file() or (root.is_symlink() and root.resolve().is_file()):
        consider(root)
    else:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [
                name
                for name in dirnames
                if name not in SKIP_DIRS and not (Path(dirpath) / name).is_symlink()
            ]
            for name in filenames:
                consider(Path(dirpath) / name)
                if truncated:
                    break
            if truncated:
                break

    if not matches:
        return ToolResult(ok=True, output=f"No matches for {pattern!r} ({files_seen} files searched)")
    header = f"{len(matches)} match(es)"
    if truncated:
        header += f" (stopped at max_results={max_results})"
    return ToolResult(ok=True, output=header + "\n" + "\n".join(matches))
