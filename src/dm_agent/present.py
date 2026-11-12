"""Human-readable rendering of a session log.

Both renderers are pure: they print what the log already recorded. Replay
does not import or call tools.
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
