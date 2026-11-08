"""ReAct loop: plan, call a tool, observe, repeat.

The model is expected to use native tool calls. If a server only returns
text, a JSON object of the form ``{"tool": "...", "arguments": {...}}``
(optionally inside a ```json fence) is accepted as a tool call. Anything
else is the final answer.
"""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dm_agent.errors import LLMError
from dm_agent.llm import ModelTurn, ToolCall
from dm_agent.session import LOG_VERSION, SessionLog
from dm_agent.tools.registry import ToolRegistry

SYSTEM_PROMPT = """You are dm-agent, a local coding agent. You edit only the given workspace.

Work in a loop. When you need information or a change, call a tool. When the task is finished, reply with a plain-text final answer and do not call a tool.

Rules:
- Do not assume file contents. Read a file before editing it.
- Paths must stay inside the workspace. Prefer paths relative to the workspace root.
- Use edit_file for small changes. old must be the exact text and should be unique.
- Use write_file only when creating a file or replacing it wholesale.
- Use lint_python after edits. It reports syntax errors, bare except, and leftover breakpoint() or pdb.set_trace() calls.
- Use run_pytest to run pytest inside the workspace. Read the status line before the final answer.
- run_shell starts in the workspace. Absolute paths must point inside the workspace. Do not try to read or write the rest of the machine. Call programs by name (python3, pytest) rather than by a system absolute path.
- If a tool returns an error, change the request. Do not repeat an identical failing call.
- Keep edits small and consistent with the surrounding code.
- The final answer should say what you changed and which checks you ran.

If native tool calls are unavailable, reply with only a JSON object:
{"tool": "tool_name", "arguments": {}}
or, when you are done, a plain-text final answer with no JSON tool object.
"""


@dataclass
class RunResult:
    """Outcome of one agent run."""

    answer: str
    session_id: str
    session_path: Path
    steps: int
    finished: bool


class Agent:
    """Drive one task against a scripted or remote model."""

    def __init__(
        self,
        *,
        llm: Any,
        tools: ToolRegistry,
        session: SessionLog,
        max_steps: int,
        metadata: dict[str, Any] | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.llm = llm
        self.tools = tools
        self.session = session
        self.max_steps = max_steps
        self.metadata = dict(metadata or {})
        self.system_prompt = system_prompt
        self.on_event = on_event

    def run(self, task: str) -> RunResult:
        """Run the loop until a final answer or ``max_steps`` model turns."""
        self._log(
            kind="session_start",
            step=0,
            log_version=LOG_VERSION,
            task=task,
            **self.metadata,
        )
        self._log(kind="user", step=0, content=task)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        schemas = self.tools.schemas()

        for step in range(1, self.max_steps + 1):
            try:
                raw_turn = self.llm.complete(messages, schemas)
            except LLMError as exc:
                self._log(kind="error", step=step, content=str(exc))
                raise
            turn = coerce_model_turn(raw_turn)
            if turn.tool_calls:
                if turn.content and turn.content.strip():
                    self._log(kind="thought", step=step, content=turn.content)
                messages.append(_assistant_message(turn))
                for call in turn.tool_calls:
                    self._log(
                        kind="tool_call",
                        step=step,
                        call_id=call.id,
                        tool=call.name,
                        arguments=call.arguments,
                    )
                    result = self.tools.invoke(call.name, call.arguments)
                    self._log(
                        kind="tool_result",
                        step=step,
                        call_id=call.id,
                        tool=call.name,
                        ok=result.ok,
                        output=result.output,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result.output,
                        }
                    )
                continue

            answer = turn.content or ""
            self._log(kind="final", step=step, content=answer)
            return RunResult(
                answer=answer,
                session_id=self.session.session_id,
                session_path=self.session.path,
                steps=step,
                finished=True,
            )

        answer = (
            f"Stopped after {self.max_steps} steps without a final answer. "
            "Inspect the session log and raise max_steps if the task is still in progress."
        )
        self._log(kind="max_steps", step=self.max_steps, content=answer)
        return RunResult(
            answer=answer,
            session_id=self.session.session_id,
            session_path=self.session.path,
            steps=self.max_steps,
            finished=False,
        )

    def _log(self, **fields: Any) -> None:
        record = self.session.append(**fields)
        if self.on_event is not None:
            self.on_event(record)


def coerce_model_turn(turn: ModelTurn) -> ModelTurn:
    """Normalize tool-call payloads and accept the text JSON fallback."""
    calls = _normalize_calls(turn.tool_calls)
    if calls:
        return ModelTurn(content=turn.content, tool_calls=calls)
    data, prefix = extract_action(turn.content or "")
    if data is None:
        return ModelTurn(content=turn.content, tool_calls=[])
    name = _action_tool_name(data)
    if name is None:
        content = data.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return ModelTurn(content=content, tool_calls=[])
    args = data.get("arguments", data.get("args", {}))
    if not isinstance(args, dict):
        args = {}
    return ModelTurn(
        content=prefix,
        tool_calls=[ToolCall(id=new_call_id(), name=name, arguments=args)],
    )


def extract_action(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Find an action object in ``text``. The second item is leading prose."""
    if not text or "{" not in text:
        return None, None
    fence = re.search(r"```(?:json)?\s*", text, flags=re.IGNORECASE)
    if fence:
        brace = text.find("{", fence.end())
        fence_end = text.find("```", fence.end())
        if brace >= 0 and (fence_end < 0 or brace < fence_end):
            data = _parse_object_at(text, brace)
            if data is not None and _looks_like_action(data):
                prefix = text[: fence.start()].strip()
                return data, prefix or None
    stripped = text.strip()
    if stripped.startswith("{"):
        data = _parse_object_at(stripped, 0)
        if data is not None and _looks_like_action(data):
            return data, None
    brace = text.find("{")
    if brace >= 0:
        data = _parse_object_at(text, brace)
        if data is not None and _looks_like_action(data):
            prefix = text[:brace].strip()
            return data, prefix or None
    return None, None


def new_call_id() -> str:
    return "call_" + secrets.token_hex(4)


def _normalize_calls(calls: list[ToolCall]) -> list[ToolCall]:
    normalized: list[ToolCall] = []
    seen: set[str] = set()
    for call in calls:
        args = call.arguments
        if isinstance(args, str):
            try:
                loaded = json.loads(args or "{}")
            except json.JSONDecodeError:
                loaded = {"_raw": args}
            args = loaded if isinstance(loaded, dict) else {"_value": loaded}
        elif not isinstance(args, dict):
            args = {"_value": args}
        call_id = call.id or new_call_id()
        if call_id in seen:
            call_id = new_call_id()
        seen.add(call_id)
        normalized.append(ToolCall(id=call_id, name=call.name, arguments=args))
    return normalized


def _assistant_message(turn: ModelTurn) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": turn.content if turn.content else None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in turn.tool_calls
        ],
    }
    return message


def _parse_object_at(text: str, start: int) -> dict[str, Any] | None:
    try:
        data, _end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        return data
    return None


def _looks_like_action(data: dict[str, Any]) -> bool:
    action = data.get("action")
    if action in {"tool", "final"}:
        return True
    if "tool" in data:
        return True
    if "name" in data and ("arguments" in data or "args" in data):
        return True
    return False


def _action_tool_name(data: dict[str, Any]) -> str | None:
    tool = data.get("tool")
    if isinstance(tool, str) and tool.strip():
        return tool.strip()
    if data.get("action") == "final":
        return None
    name = data.get("name")
    if isinstance(name, str) and name.strip() and ("arguments" in data or "args" in data or data.get("action") == "tool"):
        return name.strip()
    return None
