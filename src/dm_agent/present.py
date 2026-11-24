"""Human-readable rendering of a session log.

Renderers are pure: they print what the log already recorded. They do not
import or call tools.
"""

from __future__ import annotations

import html
import json
from typing import Any


def format_inspection(events: list[dict[str, Any]]) -> str:
    """Structured view used by ``sessions show``."""
    if not events:
        return "session (empty)\n"
    session_id = str(events[0].get("session_id", "?"))
    blocks = [f"session {session_id}", f"events {len(events)}", ""]
    for event in events:
        header = (
            f"seq={event.get('seq')} step={event.get('step')} "
            f"kind={event.get('kind')} ts={event.get('ts')}"
        )
        payload = {
            key: value
            for key, value in event.items()
            if key not in {"seq", "step", "kind", "ts"}
        }
        body = json.dumps(payload, indent=2, ensure_ascii=False)
        blocks.append(header + "\n" + body)
    return "\n\n".join(blocks) + "\n"


def format_transcript(events: list[dict[str, Any]]) -> str:
    """Narrative dry replay. Tool results come from the log, not a re-run."""
    lines: list[str] = []
    session_id = str(events[0].get("session_id", "?")) if events else "?"
    lines.append(f"# session {session_id}")
    for event in events:
        kind = event.get("kind")
        step = event.get("step")
        ts = event.get("ts", "")
        if kind == "session_start":
            lines.append(
                f"[{ts}] session_start model={event.get('model')} "
                f"workspace={event.get('workspace')}"
            )
        elif kind == "user":
            lines.append(f"[{ts}] user\n{event.get('content', '')}")
        elif kind == "thought":
            lines.append(f"[{ts}] step {step} thought\n{event.get('content', '')}")
        elif kind == "tool_call":
            args = json.dumps(event.get("arguments", {}), ensure_ascii=False)
            lines.append(f"[{ts}] step {step} tool_call {event.get('tool')} {args}")
        elif kind == "tool_result":
            status = "ok" if event.get("ok") else "error"
            lines.append(
                f"[{ts}] step {step} tool_result {event.get('tool')} {status}\n"
                f"{event.get('output', '')}"
            )
        elif kind == "final":
            lines.append(f"[{ts}] step {step} final\n{event.get('content', '')}")
        elif kind == "max_steps":
            lines.append(f"[{ts}] step {step} max_steps\n{event.get('content', '')}")
        elif kind == "error":
            lines.append(f"[{ts}] step {step} error\n{event.get('content', '')}")
        else:
            lines.append(f"[{ts}] {kind}\n{json.dumps(event, ensure_ascii=False)}")
        lines.append("")
    lines.append(
        "# replay complete (tool results printed from the log; tools were not re-run)"
    )
    return "\n".join(lines).rstrip() + "\n"


def _change_header(index: int, change: dict[str, Any]) -> str:
    parts = [
        f"change {index}",
        str(change.get("tool") or ""),
        f"path={change.get('path', '')}",
    ]
    step = change.get("step")
    if isinstance(step, int) and not isinstance(step, bool):
        parts.append(f"step={step}")
    call_id = change.get("call_id")
    if isinstance(call_id, str) and call_id:
        parts.append(f"call_id={call_id}")
    if change.get("tool") == "edit_file" and change.get("replace_all") is True:
        parts.append("replace_all=true")
    return " ".join(parts)


def _labeled_text(label: str, value: Any) -> str:
    """Label plus the raw field. A leading or trailing newline stays a blank line."""
    text = value if isinstance(value, str) else ""
    if text == "":
        return f"{label}:\n"
    return f"{label}:\n{text}\n"


def format_changes(payload: dict[str, Any]) -> str:
    """Text for ``sessions changes``. Field order is fixed.

    Zero changes are two lines and a trailing newline. Each later change is
    separated by one blank line. ``replace_all=true`` is appended only when
    that flag is true. A missing step or call id is left out of the header.
    """
    session_id = str(payload.get("session_id") or "(unknown)")
    raw_changes = payload.get("changes") or []
    changes = raw_changes if isinstance(raw_changes, list) else []
    head = f"session {session_id}\nchanges {len(changes)}\n"
    if not changes:
        return head
    blocks: list[str] = []
    for index, change in enumerate(changes, start=1):
        item = change if isinstance(change, dict) else {}
        if item.get("tool") == "write_file":
            body = _labeled_text("content", item.get("content", ""))
        else:
            body = _labeled_text("old", item.get("old", "")) + _labeled_text(
                "new", item.get("new", "")
            )
        blocks.append(f"{_change_header(index, item)}\n{body}")
    return head + "\n" + "\n".join(blocks)


def format_patch(payload: dict[str, Any]) -> str:
    """Text for ``sessions patch``. Field order is fixed.

    Zero patches are two lines and a trailing newline. Otherwise a blank
    line, then each item's ``diff`` joined by one blank line. Those strings
    already end with a newline. A ``# replace_all=true`` line, when the
    flag is set, is already the first line of that item's ``diff``.
    """
    session_id = str(payload.get("session_id") or "(unknown)")
    raw_patches = payload.get("patches") or []
    patches = raw_patches if isinstance(raw_patches, list) else []
    head = f"session {session_id}\npatches {len(patches)}\n"
    if not patches:
        return head
    diffs: list[str] = []
    for patch in patches:
        item = patch if isinstance(patch, dict) else {}
        diff = item.get("diff", "")
        diffs.append(diff if isinstance(diff, str) else "")
    return head + "\n" + "\n".join(diffs)


def format_files(payload: dict[str, Any]) -> str:
    """Text for ``sessions files``. Field order is fixed.

    Zero files are two lines and a trailing newline, with no extra blank
    line. Otherwise a blank line, then one block per file in payload
    order. Blocks are separated by one blank line. Each path is its own
    line. Each tool row is indented two spaces as ``{tool} ok={ok}
    fail={fail}``. Both counts are always printed. A non-list ``files``
    value is empty. An entry that is not a dict is skipped.
    """
    session_id = str(payload.get("session_id") or "(unknown)")
    raw_files = payload.get("files")
    files = raw_files if isinstance(raw_files, list) else []
    blocks: list[str] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        path = item.get("path", "")
        if not isinstance(path, str):
            path = ""
        raw_tools = item.get("tools")
        tools = raw_tools if isinstance(raw_tools, list) else []
        lines = [path]
        for row in tools:
            if not isinstance(row, dict):
                continue
            tool = row.get("tool", "")
            if not isinstance(tool, str):
                tool = ""
            ok = row.get("ok", 0)
            fail = row.get("fail", 0)
            if isinstance(ok, bool) or not isinstance(ok, int):
                ok = 0
            if isinstance(fail, bool) or not isinstance(fail, int):
                fail = 0
            lines.append(f"  {tool} ok={ok} fail={fail}")
        blocks.append("\n".join(lines))
    head = f"session {session_id}\nfiles {len(blocks)}\n"
    if not blocks:
        return head
    return head + "\n" + "\n\n".join(blocks) + "\n"


def format_tools(payload: dict[str, Any]) -> str:
    """Text for ``sessions tools``. Field order is fixed.

    Zero tools are two lines and a trailing newline, with no extra blank
    line. Otherwise a blank line, then one block per tool in payload
    order. Blocks are separated by one blank line, and a trailing newline
    follows the last block. Each tool line is ``{tool} ok={ok} fail={fail}``.
    Both counts are always printed. Each non-empty string path is indented
    two spaces on the following lines, in payload order. A tool with no
    paths has only that one line. A non-list ``tools`` value is empty. An
    entry that is not a dict is skipped. The count is the number of blocks
    printed. A non-string tool name is printed as empty. A bool or non-int
    count is printed as 0. A non-list ``paths`` value adds no path lines.
    A path that is not a non-empty string is skipped.
    """
    session_id = str(payload.get("session_id") or "(unknown)")
    raw_tools = payload.get("tools")
    tools = raw_tools if isinstance(raw_tools, list) else []
    blocks: list[str] = []
    for item in tools:
        if not isinstance(item, dict):
            continue
        tool = item.get("tool", "")
        if not isinstance(tool, str):
            tool = ""
        ok = item.get("ok", 0)
        fail = item.get("fail", 0)
        if isinstance(ok, bool) or not isinstance(ok, int):
            ok = 0
        if isinstance(fail, bool) or not isinstance(fail, int):
            fail = 0
        lines = [f"{tool} ok={ok} fail={fail}"]
        raw_paths = item.get("paths")
        paths = raw_paths if isinstance(raw_paths, list) else []
        for path in paths:
            if isinstance(path, str) and path != "":
                lines.append(f"  {path}")
        blocks.append("\n".join(lines))
    head = f"session {session_id}\ntools {len(blocks)}\n"
    if not blocks:
        return head
    return head + "\n" + "\n\n".join(blocks) + "\n"


def _indented_body(text: str) -> str:
    """Indent each line by two spaces.

    A trailing newline ends the text. It does not become an extra blank
    body line. Internal newlines, including a blank line, stay.
    """
    if text.endswith("\n"):
        text = text[:-1]
    return "\n".join(f"  {line}" for line in text.split("\n"))


def _failure_step(step: Any) -> str:
    if isinstance(step, int) and not isinstance(step, bool):
        return str(step)
    return "?"


def format_failures(payload: dict[str, Any]) -> str:
    """Text for ``sessions failures``. Field order is fixed.

    Zero failures are two lines and a trailing newline, with no extra blank
    line. Otherwise a blank line, then one block per failure. Blocks are
    separated by one blank line, and a trailing newline follows the last
    block. A non-list ``failures`` value is empty. An entry that is not a
    dict is skipped. The count is the number of blocks printed.

    A tool block starts with ``tool {tool} step={step} call_id={call_id}
    status={status}``. A non-empty string ``path`` is appended on that
    line. A missing step is ``step=?``. A non-empty string ``output`` is
    indented two spaces per line. An error block starts with
    ``error step={step}``. A non-empty string ``content`` is indented the
    same way.
    """
    session_id = str(payload.get("session_id") or "(unknown)")
    raw_failures = payload.get("failures")
    failures = raw_failures if isinstance(raw_failures, list) else []
    blocks: list[str] = []
    for item in failures:
        if not isinstance(item, dict):
            continue
        step = _failure_step(item.get("step"))
        if item.get("kind") == "error":
            lines = [f"error step={step}"]
            content = item.get("content", "")
            if isinstance(content, str) and content != "":
                lines.append(_indented_body(content))
            blocks.append("\n".join(lines))
            continue
        tool = item.get("tool", "")
        if not isinstance(tool, str):
            tool = ""
        call_id = item.get("call_id", "")
        if not isinstance(call_id, str):
            call_id = ""
        status = item.get("status", "")
        if not isinstance(status, str):
            status = ""
        header = f"tool {tool} step={step} call_id={call_id} status={status}"
        path = item.get("path")
        if isinstance(path, str) and path != "":
            header += f" path={path}"
        lines = [header]
        output = item.get("output", "")
        if isinstance(output, str) and output != "":
            lines.append(_indented_body(output))
        blocks.append("\n".join(lines))
    head = f"session {session_id}\nfailures {len(blocks)}\n"
    if not blocks:
        return head
    return head + "\n" + "\n\n".join(blocks) + "\n"


def _compare_int(value: Any) -> int:
    """Integer count from a comparison payload. Bools and other types are 0."""
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _compare_delta(left: int, right: int) -> str:
    """Signed suffix for a non-zero delta. Zero is omitted, not ``(0)``."""
    delta = right - left
    if delta == 0:
        return ""
    return f"  ({delta:+d})"


def _compare_side(payload: dict[str, Any], key: str) -> dict[str, Any]:
    raw = payload.get(key)
    return raw if isinstance(raw, dict) else {}


def _compare_id(side: dict[str, Any]) -> str:
    session_id = side.get("session_id")
    if isinstance(session_id, str) and session_id:
        return session_id
    if session_id and not isinstance(session_id, str):
        return str(session_id)
    return "(unknown)"


def _compare_status(side: dict[str, Any]) -> str:
    status = side.get("status")
    if isinstance(status, str) and status:
        return status
    return "open"


def _format_compare_tools(payload: dict[str, Any]) -> str:
    raw = payload.get("tools")
    rows = raw if isinstance(raw, list) else []
    rendered: list[tuple[str, int, int]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("tool", "")
        if not isinstance(name, str):
            name = ""
        rendered.append((name, _compare_int(row.get("a")), _compare_int(row.get("b"))))
    if not rendered:
        return "tools (none)"
    width = max(len(name) for name, _count_a, _count_b in rendered)
    lines = ["tools"]
    for name, count_a, count_b in rendered:
        lines.append(
            f"  {name:<{width}}  {count_a} -> {count_b}{_compare_delta(count_a, count_b)}"
        )
    return "\n".join(lines)


def _format_compare_paths(paths: list[str]) -> list[str]:
    if not paths:
        return ["    (none)"]
    return [f"    {path}" for path in paths]


def _format_compare_files(payload: dict[str, Any]) -> str:
    raw = payload.get("files")
    data = raw if isinstance(raw, dict) else {}
    buckets: dict[str, list[str]] = {}
    for key in ("only_a", "only_b", "both"):
        items = data.get(key)
        paths: list[str] = []
        if isinstance(items, list):
            for path in items:
                if isinstance(path, str):
                    paths.append(path)
        buckets[key] = paths
    if not any(buckets.values()):
        return "files (none)"
    lines = ["files"]
    for key in ("only_a", "only_b", "both"):
        lines.append(f"  {key}")
        lines.extend(_format_compare_paths(buckets[key]))
    return "\n".join(lines)


def format_compare(payload: dict[str, Any]) -> str:
    """Text for ``sessions compare``. Field order is fixed.

    The CLI prints ``file_a`` and ``file_b`` before this text. This
    function does not include those paths.

    Labels in the header are padded to ``ended_cleanly``, then two
    spaces. Counts are always printed, including zero. An integer delta
    of 0 has no suffix. Any other delta has a signed suffix with a
    leading two spaces, such as ``  (+1)`` or ``  (-2)``.
    ``ended_cleanly`` is ``yes`` or ``no`` and has no delta.
    A missing side id is ``(unknown)``. A missing status is ``open``.

    Tool rows follow the payload order, names padded to the longest
    name in the list, then two spaces. An empty tool list is
    ``tools (none)``. File buckets are ``only_a``, ``only_b``, then
    ``both``. Paths are printed as given, four spaces in. An empty
    bucket under a non-empty files section is ``    (none)``. When
    every bucket is empty the section is ``files (none)``. A blank
    line separates the header, the tools section, and the files
    section. The result ends with one trailing newline.
    """
    if not isinstance(payload, dict):
        payload = {}
    side_a = _compare_side(payload, "session_a")
    side_b = _compare_side(payload, "session_b")
    turns = payload.get("turns")
    turn_row = turns if isinstance(turns, dict) else {}
    failures = payload.get("failures")
    failure_row = failures if isinstance(failures, dict) else {}
    ended = payload.get("ended_cleanly")
    ended_row = ended if isinstance(ended, dict) else {}
    turns_a = _compare_int(turn_row.get("a", side_a.get("turns")))
    turns_b = _compare_int(turn_row.get("b", side_b.get("turns")))
    failures_a = _compare_int(failure_row.get("a", side_a.get("failures")))
    failures_b = _compare_int(failure_row.get("b", side_b.get("failures")))
    ended_a = "yes" if ended_row.get("a", side_a.get("ended_cleanly")) is True else "no"
    ended_b = "yes" if ended_row.get("b", side_b.get("ended_cleanly")) is True else "no"
    width = len("ended_cleanly")
    header = [
        f"{'compare':<{width}}  {_compare_id(side_a)} -> {_compare_id(side_b)}",
        f"{'status':<{width}}  {_compare_status(side_a)} -> {_compare_status(side_b)}",
        f"{'turns':<{width}}  {turns_a} -> {turns_b}{_compare_delta(turns_a, turns_b)}",
        f"{'ended_cleanly':<{width}}  {ended_a} -> {ended_b}",
        f"{'failures':<{width}}  {failures_a} -> {failures_b}{_compare_delta(failures_a, failures_b)}",
    ]
    body = "\n".join(
        [
            *header,
            "",
            _format_compare_tools(payload),
            "",
            _format_compare_files(payload),
        ]
    )
    return body + "\n"


def format_stats(stats: dict[str, Any]) -> str:
    """Compact text for ``sessions stats``. Field order is fixed."""
    tools = stats.get("tools") or {}
    names = sorted(str(name) for name in tools)
    tool_text = " ".join(f"{name}={tools[name]}" for name in names) if names else "(none)"
    files = sorted(str(path) for path in (stats.get("files") or []))
    if files:
        files_block = "files\n" + "\n".join(f"  {path}" for path in files)
    else:
        files_block = "files (none)"
    ended = "yes" if stats.get("ended_cleanly") else "no"
    session_id = str(stats.get("session_id") or "(unknown)")
    turns = stats.get("turns", 0)
    lines = [
        f"session {session_id}",
        f"turns {turns}",
        f"ended_cleanly {ended}",
        f"tools {tool_text}",
        files_block,
    ]
    return "\n".join(lines) + "\n"


def render_html(events: list[dict[str, Any]]) -> str:
    """Self-contained HTML document. Every field is escaped."""
    session_id = html.escape(str(events[0].get("session_id", "session"))) if events else "session"
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>dm-agent session {session_id}</title>",
        "<style>",
        "body{font:15px/1.45 ui-sans-serif,system-ui,sans-serif;margin:2rem;color:#1c1917;background:#fafaf9}",
        "h1{font-size:1.25rem;margin-bottom:.25rem}",
        ".meta{color:#57534e;margin-bottom:1.5rem}",
        "article{background:#fff;border:1px solid #e7e5e4;border-radius:8px;padding:.75rem 1rem;margin:0 0 .75rem}",
        "header{font:12px/1.4 ui-monospace,monospace;color:#57534e;margin-bottom:.4rem}",
        "pre{white-space:pre-wrap;word-break:break-word;font:13px/1.45 ui-monospace,monospace;margin:0}",
        ".final{border-color:#65a30d}",
        ".error{border-color:#dc2626}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>dm-agent session {session_id}</h1>",
        f'<p class="meta">{len(events)} events. Rendered from the JSONL log. Tools were not re-run.</p>',
    ]
    for event in events:
        kind = str(event.get("kind", ""))
        css = "final" if kind == "final" else "error" if kind in {"error", "max_steps"} else ""
        header = (
            f"seq={event.get('seq')} step={event.get('step')} "
            f"kind={kind} ts={event.get('ts')}"
        )
        body = json.dumps(event, indent=2, ensure_ascii=False)
        class_attr = f' class="{css}"' if css else ""
        parts.append(
            f"<article{class_attr}><header>{html.escape(header)}</header>"
            f"<pre>{html.escape(body)}</pre></article>"
        )
    parts.append("</body></html>\n")
    return "\n".join(parts)
