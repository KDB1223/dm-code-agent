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
