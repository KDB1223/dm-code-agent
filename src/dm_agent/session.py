"""Append-only JSONL session logs.

Each line is one event. Files are only opened for append after creation, so
a crash keeps every event that was already fsync'd. Replay reads these lines
back and does not re-run tools.
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timedelta, timezone
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


def _parse_ts(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp, or None when it is missing or invalid.

    ``utc_now`` writes a trailing ``Z``. That suffix is rewritten to
    ``+00:00`` so ``datetime.fromisoformat`` can parse it. A numeric
    offset such as ``+00:00`` is accepted as written. A non-string, an
    empty string, and any value ``fromisoformat`` rejects are missing.
    The original text is not rewritten for display.
    """
    if not isinstance(value, str) or value == "":
        return None
    text = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _duration_ms(start: datetime | None, end: datetime | None) -> int:
    """Non-negative whole milliseconds from ``start`` to ``end``.

    Missing ends are 0. An end earlier than the start is 0, not a
    negative count. The value is the timedelta floored toward zero at
    millisecond resolution, so a fraction of a millisecond does not
    round up. A pair that cannot be subtracted is 0.
    """
    if start is None or end is None:
        return 0
    try:
        delta = end - start
    except TypeError:
        return 0
    if delta < timedelta(0):
        return 0
    return delta // timedelta(milliseconds=1)


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


def session_timing(events: list[dict[str, Any]], *, session_id: str = "") -> dict[str, Any]:
    """Reconstruct wall-clock timing from timestamps already in the log.

    This reads ``events`` only. It does not call a model, execute tools,
    or inspect a workspace. Every event is considered for the session
    ``started``, ``ended``, ``duration_ms``, and ``unparsed_ts``. A ``ts``
    is parsed when ``_parse_ts`` accepts it. Other values increment
    ``unparsed_ts`` and do not move either end. A valid timestamp still
    counts when the event has no step and is not a tool call.

    ``started`` and ``ended`` are the original timestamp strings of the
    first and last events with a parseable ``ts``, in log order. They
    are not rewritten. ``duration_ms`` is ``_duration_ms`` of those
    instants: an integer, 0 when either end is missing, and 0 when
    ``ended`` is earlier than ``started``.

    A step row uses the same definition as ``session_stats``: ``kind``
    is in ``_TURN_KINDS`` and ``_event_step`` is an integer >= 1. Bools
    and other types are not a step. ``session_start``, ``user``, and
    step 0 do not create a row, but a parseable ``ts`` on those events
    still counts toward the session span and toward ``unparsed_ts``.
    Rows stay in first-seen order. ``started`` and ``ended`` on a row
    are the first and last parseable timestamps among events that
    belong to that step. A step whose events have no parseable ``ts``
    is still listed, with empty ends and duration 0.

    One call row is emitted per ``tool_call``, in log order, including
    a call with no path, no tool name, or no ``call_id``. A
    ``tool_result`` pairs when its ``call_id`` is a non-empty string.
    The last result in the log with that id supplies ``ended``, which
    is how a later result replaces an earlier one. A call whose
    ``call_id`` is missing, empty, or not a string stays unpaired.
    A result with no matching call does not invent a row. ``step`` is
    the call's integer step when it is >= 1, and null otherwise.
    ``tool`` and ``call_id`` are kept when they are non-empty strings,
    and are ``""`` otherwise. ``duration_ms`` is an int when both ends
    parse, 0 when the result is earlier than the call, and null when
    either end is missing or unparsed. 0 is a computed duration, not a
    stand-in for unknown. This is timing only. It does not report ok
    or failure.
    """
    started = ""
    ended = ""
    started_at: datetime | None = None
    ended_at: datetime | None = None
    unparsed_ts = 0
    steps: list[dict[str, Any]] = []
    step_rows: dict[int, dict[str, Any]] = {}
    step_times: dict[int, tuple[datetime | None, datetime | None]] = {}
    last_result: dict[str, dict[str, Any]] = {}
    tool_calls: list[dict[str, Any]] = []

    for event in events:
        raw_ts = event.get("ts")
        parsed = _parse_ts(raw_ts)
        if parsed is None:
            unparsed_ts += 1
        else:
            if started_at is None:
                started_at = parsed
                started = raw_ts
            ended_at = parsed
            ended = raw_ts

        kind = event.get("kind")
        if kind == "tool_result":
            call_id = event.get("call_id")
            if isinstance(call_id, str) and call_id:
                last_result[call_id] = event
        elif kind == "tool_call":
            tool_calls.append(event)

        step = _event_step(event)
        if kind in _TURN_KINDS and step is not None and step >= 1:
            row = step_rows.get(step)
            if row is None:
                row = {"step": step, "started": "", "ended": "", "duration_ms": 0}
                step_rows[step] = row
                step_times[step] = (None, None)
                steps.append(row)
            if parsed is not None:
                start_at, _previous_end = step_times[step]
                if start_at is None:
                    start_at = parsed
                    row["started"] = raw_ts
                step_times[step] = (start_at, parsed)
                row["ended"] = raw_ts

    for row in steps:
        start_at, end_at = step_times[row["step"]]
        row["duration_ms"] = _duration_ms(start_at, end_at)

    calls: list[dict[str, Any]] = []
    for event in tool_calls:
        step = _event_step(event)
        tool = event.get("tool")
        raw_call_id = event.get("call_id")
        call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id else ""
        start_parsed = _parse_ts(event.get("ts"))
        end_parsed: datetime | None = None
        ended_ts = ""
        if call_id and call_id in last_result:
            result = last_result[call_id]
            end_parsed = _parse_ts(result.get("ts"))
            if end_parsed is not None:
                ended_ts = result["ts"]
        if start_parsed is not None and end_parsed is not None:
            duration: int | None = _duration_ms(start_parsed, end_parsed)
        else:
            duration = None
        calls.append(
            {
                "step": step if step is not None and step >= 1 else None,
                "tool": tool if isinstance(tool, str) and tool else "",
                "call_id": call_id,
                "started": event["ts"] if start_parsed is not None else "",
                "ended": ended_ts,
                "duration_ms": duration,
            }
        )

    return {
        "session_id": _first_session_id(events, session_id),
        "started": started,
        "ended": ended,
        "duration_ms": _duration_ms(started_at, ended_at),
        "unparsed_ts": unparsed_ts,
        "steps": steps,
        "calls": calls,
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


def session_files(events: list[dict[str, Any]], *, session_id: str = "") -> dict[str, Any]:
    """Inventory paths recorded on tool calls, with per-tool ok/fail counts.

    Only ``tool_call`` events are walked. A call contributes a path when
    ``arguments`` is a dict and ``arguments.path`` is a non-empty string.
    The logged string is kept as-is. It is not normalized, resolved, or
    rewritten. Calls whose arguments are missing, not a dict, or whose
    path is missing, not a string, or empty are omitted. A tool that
    recorded no path, such as a typical ``run_shell``, is omitted the
    same way. ``search``, ``list_dir``, ``lint_python``, ``run_pytest``,
    ``read_file``, ``edit_file``, and ``write_file`` are included when
    they recorded a non-empty path, and so is any other tool that did.
    A call whose ``tool`` is not a string is omitted, because there is
    no tool name to count.

    A call is ok when ``_tool_result_ok`` maps its ``call_id`` to true:
    the last ``tool_result`` with that id has ``ok`` true. Otherwise it
    is a fail. That includes a failed result, a result whose ``ok`` is
    not true, and an unpaired call. A missing or empty ``call_id``
    cannot be paired, so that call is a fail when it otherwise has a
    valid path. Calls that share a path and tool name increment that
    tool's ``ok`` or ``fail`` count. A tool row is not emitted when both
    counts would be zero.

    Paths stay in first-seen log order. Tools under a path stay in
    first-seen order too, not alphabetical. ``session_id`` is the first
    id in the log, or the provided ``session_id`` when the log has none.
    This reads the events only. It does not list or diff the workspace,
    and it does not open any path from the log.
    """
    result_ok = _tool_result_ok(events)
    files: list[dict[str, Any]] = []
    by_path: dict[str, dict[str, Any]] = {}
    by_tool: dict[str, dict[str, dict[str, Any]]] = {}
    for event in events:
        if event.get("kind") != "tool_call":
            continue
        tool = event.get("tool")
        if not isinstance(tool, str):
            continue
        arguments = event.get("arguments")
        if not isinstance(arguments, dict):
            continue
        path = arguments.get("path")
        if not isinstance(path, str) or path == "":
            continue
        call_id = event.get("call_id")
        succeeded = (
            isinstance(call_id, str) and call_id != "" and result_ok.get(call_id) is True
        )
        file_entry = by_path.get(path)
        if file_entry is None:
            file_entry = {"path": path, "tools": []}
            by_path[path] = file_entry
            by_tool[path] = {}
            files.append(file_entry)
        row = by_tool[path].get(tool)
        if row is None:
            row = {"tool": tool, "ok": 0, "fail": 0}
            by_tool[path][tool] = row
            file_entry["tools"].append(row)
        if succeeded:
            row["ok"] += 1
        else:
            row["fail"] += 1
    return {
        "session_id": _first_session_id(events, session_id),
        "files": files,
    }


def session_tools(events: list[dict[str, Any]], *, session_id: str = "") -> dict[str, Any]:
    """Inventory tool calls grouped by tool name, with ok/fail counts and paths.

    Only ``tool_call`` events are walked. A call is counted when ``tool``
    is a non-empty string. A call whose ``tool`` is missing, not a string,
    or empty is omitted, because there is no name to group.

    A call is ok when ``_tool_result_ok`` maps its ``call_id`` to true:
    the last ``tool_result`` with that id has ``ok`` true. Otherwise it
    is a fail. That includes a failed result, a result whose ``ok`` is
    not true, and an unpaired call. A missing or empty ``call_id``
    cannot be paired, so that call is a fail.

    Tools stay in first-seen log order, not alphabetical. For each tool,
    unique ``arguments.path`` values that are non-empty strings are
    collected in first-seen order. The logged string is kept as-is. It
    is not normalized, resolved, or rewritten. A call whose arguments
    are missing or not a dict, or whose path is missing, not a string,
    or empty, adds no path. Paths are not taken from any other argument
    key. A tool that recorded no path, such as a typical ``run_shell``,
    is still included.

    ``session_id`` is the first id in the log, or the provided
    ``session_id`` when the log has none. This reads the events only.
    It does not list or diff the workspace, call a model, or execute
    tools, and it does not open any path from the log.
    """
    result_ok = _tool_result_ok(events)
    tools: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    seen_paths: dict[str, set[str]] = {}
    for event in events:
        if event.get("kind") != "tool_call":
            continue
        tool = event.get("tool")
        if not isinstance(tool, str) or tool == "":
            continue
        call_id = event.get("call_id")
        succeeded = (
            isinstance(call_id, str) and call_id != "" and result_ok.get(call_id) is True
        )
        row = by_name.get(tool)
        if row is None:
            row = {"tool": tool, "ok": 0, "fail": 0, "paths": []}
            by_name[tool] = row
            seen_paths[tool] = set()
            tools.append(row)
        if succeeded:
            row["ok"] += 1
        else:
            row["fail"] += 1
        arguments = event.get("arguments")
        if not isinstance(arguments, dict):
            continue
        path = arguments.get("path")
        if isinstance(path, str) and path != "" and path not in seen_paths[tool]:
            seen_paths[tool].add(path)
            row["paths"].append(path)
    return {
        "session_id": _first_session_id(events, session_id),
        "tools": tools,
    }


def session_failures(events: list[dict[str, Any]], *, session_id: str = "") -> dict[str, Any]:
    """List failed tool calls and session ``error`` events in log order.

    A ``tool_call`` is a failure unless ``_tool_result_ok`` maps its
    ``call_id`` to true: the last ``tool_result`` with that id has ``ok``
    true. That includes a failed result, a result whose ``ok`` is not true
    (including numeric 1), an unpaired call, and a missing or empty
    ``call_id`` (it cannot be paired). A call is included even when it has
    no non-empty string ``arguments.path``, which is how a typical
    ``run_shell`` is recorded. The logged path is kept as-is when it is a
    non-empty string. It is not normalized, resolved, or rewritten. A call
    whose ``tool`` is not a string is omitted. Ghost ``tool_result`` events
    with no matching ``tool_call`` are omitted. Successful calls are omitted.

    ``status`` is ``failed`` when a last ``tool_result`` was paired: the
    call id is a non-empty string, it is present in the ``_tool_result_ok``
    map, and that value is not true. ``status`` is ``unpaired`` when there
    was no pairable result. ``output`` is that last paired result's
    ``output`` when the field is a string, and ``""`` otherwise.

    An ``error`` event (``kind == "error"``) is included in the same list,
    at its place in the log. ``content`` is the event content when that
    field is a string, and ``""`` otherwise. ``max_steps``, ``final``, and
    thoughts are omitted. ``stats`` already reports whether the session
    ended with ``final``.

    ``session_id`` is the first non-empty string id in the log, or the
    provided ``session_id`` when the log has none. This reads the events
    only. It does not execute tools, call a model, or read or diff the
    workspace.
    """
    result_ok = _tool_result_ok(events)
    last_output: dict[str, str] = {}
    for event in events:
        if event.get("kind") != "tool_result":
            continue
        call_id = event.get("call_id")
        if not isinstance(call_id, str) or call_id == "":
            continue
        output = event.get("output")
        last_output[call_id] = output if isinstance(output, str) else ""

    failures: list[dict[str, Any]] = []
    for event in events:
        kind = event.get("kind")
        if kind == "error":
            content = event.get("content")
            failures.append(
                {
                    "kind": "error",
                    "step": _event_step(event),
                    "content": content if isinstance(content, str) else "",
                }
            )
            continue
        if kind != "tool_call":
            continue
        tool = event.get("tool")
        if not isinstance(tool, str):
            continue
        raw_call_id = event.get("call_id")
        call_id = raw_call_id if isinstance(raw_call_id, str) else ""
        paired = call_id != "" and call_id in result_ok
        if paired and result_ok[call_id] is True:
            continue
        arguments = event.get("arguments")
        path = None
        if isinstance(arguments, dict):
            raw_path = arguments.get("path")
            if isinstance(raw_path, str) and raw_path != "":
                path = raw_path
        if paired:
            status = "failed"
            output = last_output.get(call_id, "")
        else:
            status = "unpaired"
            output = ""
        failures.append(
            {
                "kind": "tool",
                "tool": tool,
                "step": _event_step(event),
                "call_id": call_id,
                "path": path,
                "status": status,
                "output": output,
            }
        )
    return {
        "session_id": _first_session_id(events, session_id),
        "failures": failures,
    }


def session_shell(events: list[dict[str, Any]], *, session_id: str = "") -> dict[str, Any]:
    """List every logged ``run_shell`` call in log order, with its command.

    Only a ``tool_call`` whose ``tool`` is exactly ``run_shell`` is
    included. ``run_pytest``, ``lint_python``, file tools, and any other
    name are ignored. A ``tool_result`` with no matching call does not
    invent a row. ``error``, ``final``, and thoughts are ignored.

    Pairing matches ``_tool_result_ok``: the last ``tool_result`` whose
    ``call_id`` is a non-empty string supplies ``ok`` and ``output``.
    A later result replaces an earlier one for that id. A result that
    appears before the call still pairs, because the scan covers the
    whole log. Two calls that share an id both use that last result.

    ``status`` is ``ok`` when the call id is a non-empty string and that
    last result has ``ok`` true. ``status`` is ``failed`` when the id is
    a non-empty string and a paired result exists whose ``ok`` is not
    true, including numeric 1 and a refusal. ``status`` is ``unpaired``
    when the id is missing, not a string, or empty, or when no result
    shares it. A refused command is ``failed`` when ``ok`` is not true.
    The output text is not parsed, and the word ``Refused`` is not a
    separate status.

    ``step`` is the integer step, including 0. Bools and other types are
    null. ``call_id`` is the recorded id when it is a non-empty string,
    and null otherwise. ``command`` is ``arguments.command`` when that
    value is a string, including empty or whitespace-only. It is not
    stripped or rewritten. A missing ``arguments`` object, a non-dict,
    or a non-string command is null. ``timeout`` is ``arguments.timeout``
    when it is an int or a float and not a bool, and null otherwise.
    The config default is not filled in. ``output`` is the last paired
    result's ``output`` when that value is a string, including empty,
    and null otherwise. An unpaired call has null output.

    ``n`` is the number of listed calls. ``n_ok``, ``n_failed``, and
    ``n_unpaired`` count those statuses and partition ``n``. There is
    no filter. ``session_id`` is the first non-empty string id in the
    log, or the provided ``session_id`` when the log has none.

    This reads the events only. It does not execute the command, call a
    model, or read or diff the workspace.
    """
    result_ok = _tool_result_ok(events)
    last_output: dict[str, Any] = {}
    for event in events:
        if event.get("kind") != "tool_result":
            continue
        raw_id = event.get("call_id")
        if isinstance(raw_id, str) and raw_id:
            last_output[raw_id] = event.get("output")

    calls: list[dict[str, Any]] = []
    for event in events:
        if event.get("kind") != "tool_call":
            continue
        if event.get("tool") != "run_shell":
            continue
        raw_call_id = event.get("call_id")
        call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id else None
        arguments = event.get("arguments")
        command = None
        timeout = None
        if isinstance(arguments, dict):
            raw_command = arguments.get("command")
            if isinstance(raw_command, str):
                command = raw_command
            raw_timeout = arguments.get("timeout")
            if isinstance(raw_timeout, (int, float)) and not isinstance(raw_timeout, bool):
                timeout = raw_timeout
        if call_id is not None and call_id in result_ok:
            status = "ok" if result_ok[call_id] else "failed"
            output = last_output.get(call_id)
            if not isinstance(output, str):
                output = None
        else:
            status = "unpaired"
            output = None
        calls.append(
            {
                "step": _event_step(event),
                "call_id": call_id,
                "command": command,
                "timeout": timeout,
                "status": status,
                "output": output,
            }
        )

    n_ok = sum(call["status"] == "ok" for call in calls)
    n_failed = sum(call["status"] == "failed" for call in calls)
    n_unpaired = sum(call["status"] == "unpaired" for call in calls)
    return {
        "session_id": _first_session_id(events, session_id),
        "n": len(calls),
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_unpaired": n_unpaired,
        "calls": calls,
    }


def _recorded_lines(text: str) -> tuple[list[str], bool]:
    """Split ``text`` on ``\\n`` and report whether it ends with a newline.

    An empty string has no lines. The flag is meaningful only when there is
    at least one line: a missing final newline is marked in the unified diff.
    Other separators, including ``\\r``, stay on the line.
    """
    if text == "":
        return [], True
    ends_nl = text.endswith("\n")
    body = text[:-1] if ends_nl else text
    return body.split("\n"), ends_nl


def _prefixed_lines(prefix: str, lines: list[str], *, ends_nl: bool) -> list[str]:
    """Hunk lines for one side. No lines means no ``\\ No newline`` marker."""
    if not lines:
        return []
    rendered = [prefix + line for line in lines]
    if not ends_nl:
        rendered.append("\\ No newline at end of file")
    return rendered


def _edit_diff(path: str, old: str, new: str, *, replace_all: bool) -> str:
    """Snippet hunk. Counts are always written, including 0 and 1.

    The log has no file line numbers and no surrounding context, so both
    sides start at line 1 of the recorded snippet. ``replace_all`` is not
    expanded; when it is true a single comment sits above the file header.
    """
    old_lines, old_nl = _recorded_lines(old)
    new_lines, new_nl = _recorded_lines(new)
    hunk = [
        f"--- a/{path}",
        f"+++ b/{path}",
        f"@@ -1,{len(old_lines)} +1,{len(new_lines)} @@",
        *_prefixed_lines("-", old_lines, ends_nl=old_nl),
        *_prefixed_lines("+", new_lines, ends_nl=new_nl),
    ]
    text = "\n".join(hunk) + "\n"
    if replace_all:
        text = "# replace_all=true\n" + text
    return text


def _write_diff(path: str, content: str) -> str:
    """Full-file add of recorded ``content``. Previous bytes are not in the log."""
    lines, ends_nl = _recorded_lines(content)
    if lines:
        header = f"@@ -0,0 +1,{len(lines)} @@"
    else:
        header = "@@ -0,0 +0,0 @@"
    hunk = [
        "--- /dev/null",
        f"+++ b/{path}",
        header,
        *_prefixed_lines("+", lines, ends_nl=ends_nl),
    ]
    return "\n".join(hunk) + "\n"


def session_patch(events: list[dict[str, Any]], *, session_id: str = "") -> dict[str, Any]:
    """Turn the mutations from ``session_changes`` into unified diffs.

    Which calls are included is exactly ``session_changes``: successful
    ``edit_file`` and ``write_file`` calls, in log order, one entry per
    call. The diff is rebuilt from the logged arguments. It is not a diff
    of the workspace, and this function does not read or write files.

    ``edit_file`` is a snippet hunk (``@@ -1,N +1,M @@``). A side with no
    lines uses start 1 and count 0. Every hunk line is ``-`` or ``+``.
    ``write_file`` is a full-file add of the recorded ``content``
    (``--- /dev/null``), not a deletion of any previous workspace bytes.
    Empty content is an empty new-file hunk (``@@ -0,0 +0,0 @@``).

    If a side has lines and does not end with a newline, ``\\ No newline
    at end of file`` follows that side's last line. ``replace_all`` is
    copied through and is not expanded into extra hunks. When it is true,
    the ``diff`` string starts with ``# replace_all=true`` so the text
    form can show it without changing the hunk body.
    """
    changes = session_changes(events, session_id=session_id)
    return {
        "session_id": changes["session_id"],
        "patches": [_patch_from_change(change) for change in changes["changes"]],
    }


def _patch_from_change(change: dict[str, Any]) -> dict[str, Any]:
    """One ``session_changes`` entry plus the ``diff`` string for that entry."""
    if change["tool"] == "write_file":
        return {
            "tool": "write_file",
            "path": change["path"],
            "step": change["step"],
            "call_id": change["call_id"],
            "content": change["content"],
            "diff": _write_diff(change["path"], change["content"]),
        }
    return {
        "tool": "edit_file",
        "path": change["path"],
        "step": change["step"],
        "call_id": change["call_id"],
        "old": change["old"],
        "new": change["new"],
        "replace_all": change["replace_all"],
        "diff": _edit_diff(
            change["path"],
            change["old"],
            change["new"],
            replace_all=change["replace_all"] is True,
        ),
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


def _summary_session_id(events: list[dict[str, Any]], fallback: str) -> str:
    """Session id ``summarize_session`` would keep for these events.

    The first JSON object wins. A truthy ``session_id`` is stringified.
    A missing or empty id, or a log with no object, uses ``fallback``
    (the file stem, when the caller has one). Later events are not
    consulted. This is not ``session_stats``, which walks the log for
    the first non-empty string id.
    """
    for event in events:
        if not isinstance(event, dict):
            continue
        raw = event.get("session_id")
        if raw:
            return str(raw)
        return fallback
    return fallback


def _summary_status(events: list[dict[str, Any]]) -> str:
    """Status word ``summarize_session`` would record for these events.

    ``open`` until a ``final`` or ``max_steps`` event. An ``error`` event
    sets ``error`` only when no ``final`` or ``max_steps`` has already
    been seen, so a later ``error`` does not replace those. A non-dict
    item sets ``corrupt``. A log ``load_events`` can read has no
    non-dict items; ``sessions compare`` still reports the status
    ``summarize_session`` computed for the file when the caller passes it.
    """
    status = "open"
    for event in events:
        if not isinstance(event, dict):
            status = "corrupt"
            continue
        kind = event.get("kind")
        if kind == "final":
            status = "final"
        elif kind == "max_steps":
            status = "max_steps"
        elif kind == "error" and status not in {"final", "max_steps"}:
            status = "error"
    return status


def _resolve_status(events: list[dict[str, Any]], status: str | None) -> str:
    """Use a caller-supplied status, or the ``summarize_session`` rules."""
    if status is None:
        return _summary_status(events)
    if isinstance(status, str):
        return status
    return str(status)


def session_compare(
    events_a: list[dict[str, Any]],
    events_b: list[dict[str, Any]],
    *,
    session_id_a: str = "",
    session_id_b: str = "",
    status_a: str | None = None,
    status_b: str | None = None,
) -> dict[str, Any]:
    """Compare two loaded session logs. Deltas are B minus A.

    ``SESSION_A`` is the baseline and ``SESSION_B`` is the candidate.
    The comparison is built from helpers that already read one log:

    * ``session_id`` follows ``summarize_session`` (see
      ``_summary_session_id``). ``session_id_a`` and ``session_id_b``
      are the fallbacks, normally the file stems.
    * ``status`` is ``status_a`` / ``status_b`` when the caller passes
      them (the CLI passes ``summarize_session``). Otherwise it is
      ``_summary_status``, the same rules. Status is not
      ``ended_cleanly``: a ``final`` followed by ``error`` stays
      ``final`` here, while ``session_stats`` reports the last terminal
      kind.
    * ``turns``, ``ended_cleanly``, ``tools``, and ``files`` come from
      ``session_stats``. Tool rows are the sorted union of those tool
      maps. A name present on only one side is 0 on the other. A row
      with delta 0 is kept. A name that is 0 on both sides is absent,
      because ``session_stats`` does not emit it.
    * ``files`` are the successful ``edit_file`` and ``write_file``
      paths from ``session_stats``, compared as recorded strings with
      no normalization. ``only_a``, ``only_b``, and ``both`` are sorted.
    * ``failures`` is the number of entries ``session_failures`` lists
      for that log. It is not a second definition of a failed call.

    Comparing a log to itself yields zero deltas, every edited path in
    ``both``, and empty ``only_a`` / ``only_b``. Empty event lists are
    valid: status ``open``, zero turns, zero failures, no tools, no
    files. This reads the events only. It does not call a model,
    execute tools, or diff a workspace.
    """
    resolved_status_a = _resolve_status(events_a, status_a)
    resolved_status_b = _resolve_status(events_b, status_b)

    id_a = _summary_session_id(events_a, session_id_a)
    id_b = _summary_session_id(events_b, session_id_b)
    stats_a = session_stats(events_a, session_id=session_id_a)
    stats_b = session_stats(events_b, session_id=session_id_b)
    failures_a = len(session_failures(events_a, session_id=session_id_a)["failures"])
    failures_b = len(session_failures(events_b, session_id=session_id_b)["failures"])

    names = sorted(set(stats_a["tools"]) | set(stats_b["tools"]))
    tools: list[dict[str, Any]] = []
    for name in names:
        count_a = stats_a["tools"].get(name, 0)
        count_b = stats_b["tools"].get(name, 0)
        tools.append({"tool": name, "a": count_a, "b": count_b, "delta": count_b - count_a})

    files_a = set(stats_a["files"])
    files_b = set(stats_b["files"])
    return {
        "session_a": {
            "session_id": id_a,
            "status": resolved_status_a,
            "turns": stats_a["turns"],
            "ended_cleanly": stats_a["ended_cleanly"],
            "failures": failures_a,
        },
        "session_b": {
            "session_id": id_b,
            "status": resolved_status_b,
            "turns": stats_b["turns"],
            "ended_cleanly": stats_b["ended_cleanly"],
            "failures": failures_b,
        },
        "turns": {
            "a": stats_a["turns"],
            "b": stats_b["turns"],
            "delta": stats_b["turns"] - stats_a["turns"],
        },
        "failures": {"a": failures_a, "b": failures_b, "delta": failures_b - failures_a},
        "ended_cleanly": {
            "a": stats_a["ended_cleanly"],
            "b": stats_b["ended_cleanly"],
        },
        "tools": tools,
        "files": {
            "only_a": sorted(files_a - files_b),
            "only_b": sorted(files_b - files_a),
            "both": sorted(files_a & files_b),
        },
    }
