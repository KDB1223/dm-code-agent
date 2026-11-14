"""Append-only JSONL session logs.

Each line is one event. Files are only opened for append after creation, so
a crash keeps every event that was already fsync'd. Replay reads these lines
back and does not re-run tools.
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dm_agent.errors import AgentError

LOG_VERSION = 1


def utc_now() -> str:
    """ISO-8601 timestamp in UTC with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_session_id() -> str:
    """Sortable session id: UTC timestamp plus a short random suffix."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


class SessionLog:
    """One append-only JSONL file."""

    def __init__(self, path: Path, session_id: str) -> None:
        self.path = path
        self.session_id = session_id
        self.seq = 0

    @classmethod
    def create(cls, directory: Path) -> SessionLog:
        """Reserve a new session file under ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        for _ in range(8):
            session_id = new_session_id()
            path = directory / f"{session_id}.jsonl"
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                continue
            os.close(fd)
            return cls(path, session_id)
        raise AgentError("could not allocate a unique session id")

    def append(self, *, kind: str, step: int, **fields: Any) -> dict[str, Any]:
        """Append one event and fsync it.

        ``fields`` must be JSON-serializable. Insertion order is preserved so
        a reader sees ``ts``, ``session_id``, ``seq``, ``step``, and ``kind``
        before the payload.
        """
        self.seq += 1
        record: dict[str, Any] = {
            "ts": utc_now(),
            "session_id": self.session_id,
            "seq": self.seq,
            "step": step,
            "kind": kind,
        }
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record


def load_events(path: Path) -> list[dict[str, Any]]:
    """Read every event from a session file."""
    if not path.is_file():
        raise AgentError(f"session file not found: {path}")
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise AgentError(f"{path}:{lineno}: invalid JSONL ({exc})") from exc
            if not isinstance(item, dict):
                raise AgentError(f"{path}:{lineno}: expected a JSON object")
            events.append(item)
    return events


def iter_session_files(log_dir: Path) -> list[Path]:
    """Return session files, newest id first."""
    if not log_dir.is_dir():
        return []
    return sorted(log_dir.glob("*.jsonl"), key=lambda path: path.name, reverse=True)


def resolve_session_file(log_dir: Path, token: str) -> Path:
    """Find a session by full id or a unique prefix."""
    if not token or token.strip() == "" or "/" in token or token.startswith("."):
        raise AgentError(f"invalid session id: {token!r}")
    exact = log_dir / f"{token}.jsonl"
    if exact.is_file():
        return exact
    matches = [path for path in iter_session_files(log_dir) if path.stem.startswith(token)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise AgentError(f"session not found: {token}")
    names = ", ".join(path.stem for path in matches)
    raise AgentError(f"session id is ambiguous: {names}")


# Model steps. ``session_start`` and ``user`` are not turns.
_TURN_KINDS = frozenset({"thought", "tool_call", "tool_result", "final", "max_steps", "error"})
_TERMINAL_KINDS = frozenset({"final", "max_steps", "error"})
# The log has no separate "files touched" field. These tools record a path
# and change a file when the paired result is ok.
_FILE_EDIT_TOOLS = frozenset({"edit_file", "write_file"})


def _first_session_id(events: list[dict[str, Any]], fallback: str = "") -> str:
    """First non-empty string ``session_id`` in ``events``, or ``fallback``."""
    for event in events:
        raw_id = event.get("session_id")
        if isinstance(raw_id, str) and raw_id:
            return raw_id
    return fallback


def _tool_result_ok(events: list[dict[str, Any]]) -> dict[str, bool]:
    """Map each ``call_id`` to whether its last ``tool_result`` has ``ok`` true."""
    result_ok: dict[str, bool] = {}
    for event in events:
        if event.get("kind") != "tool_result":
            continue
        call_id = event.get("call_id")
        if isinstance(call_id, str) and call_id:
            result_ok[call_id] = event.get("ok") is True
    return result_ok


def _event_step(event: dict[str, Any]) -> int | None:
    """Integer ``step`` from an event. Bools and other types are missing."""
    step = event.get("step")
    if isinstance(step, int) and not isinstance(step, bool):
        return step
    return None


def session_stats(events: list[dict[str, Any]], *, session_id: str = "") -> dict[str, Any]:
    """Summarize turns, tool counts, edited files, and a clean ending.

    A turn is one model step: a distinct ``step`` >= 1 on a thought, tool
    call, tool result, final answer, ``max_steps`` stop, or error. Several
    tool calls that share a step are one turn.

    Tool counts come from ``tool_call`` events, sorted by tool name.

    Files are ``arguments.path`` values on successful ``edit_file`` and
    ``write_file`` calls. Success means a ``tool_result`` with the same
    ``call_id`` and ``ok`` true. Other tools are counted, but their paths
    are not listed, because a read or a search does not record an edit.
    The session ended cleanly when the last ``final``, ``max_steps``, or
    ``error`` event is ``final``.
    """
    turns: set[int] = set()
    tool_counts: dict[str, int] = {}
    result_ok = _tool_result_ok(events)
    edit_calls: list[tuple[str, str]] = []

    for event in events:
        kind = event.get("kind")
        if kind in _TURN_KINDS:
            step = _event_step(event)
            if step is not None and step >= 1:
                turns.add(step)
        if kind != "tool_call":
            continue
        tool = event.get("tool")
        if isinstance(tool, str) and tool:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        if tool not in _FILE_EDIT_TOOLS:
            continue
        arguments = event.get("arguments")
        path = arguments.get("path") if isinstance(arguments, dict) else None
        call_id = event.get("call_id")
        if isinstance(path, str) and path and isinstance(call_id, str) and call_id:
            edit_calls.append((call_id, path))

    last_terminal = ""
    for event in events:
        kind = event.get("kind")
        if isinstance(kind, str) and kind in _TERMINAL_KINDS:
            last_terminal = kind

    files: list[str] = []
    seen: set[str] = set()
    for call_id, path in edit_calls:
        if result_ok.get(call_id) is True and path not in seen:
            seen.add(path)
            files.append(path)
    files.sort()

    return {
        "session_id": _first_session_id(events, session_id),
        "turns": len(turns),
        "ended_cleanly": last_terminal == "final",
        "tools": {name: tool_counts[name] for name in sorted(tool_counts)},
        "files": files,
    }


def session_changes(events: list[dict[str, Any]], *, session_id: str = "") -> dict[str, Any]:
    """List successful ``edit_file`` and ``write_file`` mutations in log order.

    A change is a ``tool_call`` for one of those tools whose ``tool_result``
    shares ``call_id`` and has ``ok`` true, and whose arguments describe the
    mutation. ``edit_file`` needs a non-empty ``path`` and string ``old`` and
    ``new``. ``write_file`` needs a non-empty ``path`` and string ``content``
    (content may be empty). ``replace_all`` is kept when it is a boolean and
    is otherwise false.

    Failed results, unpaired calls, other tools, and calls missing those
    arguments are omitted. The same path is listed once per successful call.
    This reads the log only. It does not diff the workspace.
    """
    result_ok = _tool_result_ok(events)
    changes: list[dict[str, Any]] = []
    for event in events:
        if event.get("kind") != "tool_call":
            continue
        tool = event.get("tool")
        if tool not in _FILE_EDIT_TOOLS:
            continue
        call_id = event.get("call_id")
        if not isinstance(call_id, str) or call_id == "":
            continue
        if result_ok.get(call_id) is not True:
            continue
        arguments = event.get("arguments")
        if not isinstance(arguments, dict):
            continue
        path = arguments.get("path")
        if not isinstance(path, str) or path == "":
            continue
        step = _event_step(event)
        if tool == "edit_file":
            old = arguments.get("old")
            new = arguments.get("new")
            if not isinstance(old, str) or not isinstance(new, str):
                continue
            replace_all = arguments.get("replace_all", False)
            if not isinstance(replace_all, bool):
                replace_all = False
            changes.append(
                {
                    "tool": "edit_file",
                    "path": path,
                    "step": step,
                    "call_id": call_id,
                    "old": old,
                    "new": new,
                    "replace_all": replace_all,
                }
            )
        elif tool == "write_file":
            content = arguments.get("content")
            if not isinstance(content, str):
                continue
            changes.append(
                {
                    "tool": "write_file",
                    "path": path,
                    "step": step,
                    "call_id": call_id,
                    "content": content,
                }
            )
    return {
        "session_id": _first_session_id(events, session_id),
        "changes": changes,
    }


def summarize_session(path: Path) -> dict[str, Any]:
    """Read a session file into a short row for ``sessions list``."""
    summary: dict[str, Any] = {
        "session_id": path.stem,
        "path": str(path),
        "started": "",
        "task": "",
        "events": 0,
        "model": "",
        "status": "open",
    }
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                event = json.loads(stripped)
                if not isinstance(event, dict):
                    summary["status"] = "corrupt"
                    continue
                summary["events"] += 1
                if summary["started"] == "":
                    summary["started"] = str(event.get("ts", ""))
                    summary["task"] = str(event.get("task") or event.get("content") or "")
                    summary["model"] = str(event.get("model") or "")
                    summary["session_id"] = str(event.get("session_id") or path.stem)
                kind = event.get("kind")
                if kind == "final":
                    summary["status"] = "final"
                elif kind == "max_steps":
                    summary["status"] = "max_steps"
                elif kind == "error" and summary["status"] not in {"final", "max_steps"}:
                    summary["status"] = "error"
    except json.JSONDecodeError:
        summary["status"] = "corrupt"
    except OSError as exc:
        raise AgentError(f"cannot read {path}: {exc}") from exc
    return summary
