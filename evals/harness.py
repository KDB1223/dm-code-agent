"""Deterministic stand-in for the language model used by the offline evals.

Each task ships a list of turns. The harness plays them back in order, so a
score does not depend on a network, an API key, or sampling temperature.
"""

from __future__ import annotations

from typing import Any

from dm_agent.errors import LLMError
from dm_agent.llm import ModelTurn, ToolCall


class ScriptedLLM:
    """Return a fixed sequence of model turns."""

    def __init__(self, turns: list[dict[str, Any]]) -> None:
        self._turns = [_to_turn(turn, index) for index, turn in enumerate(turns, start=1)]

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        del messages, tools
        if not self._turns:
            raise LLMError("scripted harness ran out of turns")
        return self._turns.pop(0)


def _to_turn(payload: dict[str, Any], index: int) -> ModelTurn:
    content = payload.get("content")
    if content is not None and not isinstance(content, str):
        raise LLMError(f"turn {index} content must be a string")
    calls: list[ToolCall] = []
    raw_calls = payload.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise LLMError(f"turn {index} tool_calls must be a list")
    for number, raw in enumerate(raw_calls, start=1):
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
            raise LLMError(f"turn {index} tool call {number} needs a name")
        arguments = raw.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise LLMError(f"turn {index} tool call {number} arguments must be an object")
        calls.append(ToolCall(id=f"call_{index}_{number}", name=raw["name"], arguments=arguments))
    return ModelTurn(content=content, tool_calls=calls)
