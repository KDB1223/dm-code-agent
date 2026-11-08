"""Tool registry shared by the ReAct loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dm_agent.errors import PathJailError
from dm_agent.limits import clip_text


@dataclass
class ToolResult:
    """Observation returned to the model and written to the session log."""

    ok: bool
    output: str


@dataclass
class Tool:
    """One callable tool plus its JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], ToolResult]


class ToolRegistry:
    """Dispatch tool calls by name and clip their output."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI-style tool definitions, in registration order."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def invoke(self, name: str, arguments: dict[str, Any] | None) -> ToolResult:
        """Run a tool. Failures become observations; they do not raise."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, output=f"Unknown tool: {name}")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return ToolResult(ok=False, output="Tool arguments must be a JSON object")
        try:
            result = tool.handler(arguments)
        except PathJailError as exc:
            result = ToolResult(ok=False, output=f"Refused: {exc}")
        except Exception as exc:  # noqa: BLE001 — tools must not kill the loop
            result = ToolResult(ok=False, output=f"{type(exc).__name__}: {exc}")
        return ToolResult(ok=result.ok, output=clip_text(result.output))
