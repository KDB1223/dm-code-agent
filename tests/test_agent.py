"""ReAct loop with a scripted model. No network."""

from pathlib import Path

import pytest

from dm_agent.agent import Agent, coerce_model_turn
from dm_agent.errors import LLMError
from dm_agent.llm import ModelTurn, ToolCall, public_base_url
from dm_agent.session import SessionLog, load_events
from dm_agent.tools import build_registry


class ScriptedLLM:
    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = list(turns)
        self.seen: list[list[dict]] = []

    def complete(self, messages, tools):
        self.seen.append(messages)
        assert any(item["function"]["name"] == "read_file" for item in tools)
        if not self.turns:
            raise AssertionError("scripted model ran out of turns")
        return self.turns.pop(0)


def test_loop_reads_then_answers_and_writes_a_full_log(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    session = SessionLog.create(tmp_path / ".dm-agent" / "sessions")
    model = ScriptedLLM(
        [
            ModelTurn(
                content="I should read a.txt",
                tool_calls=[ToolCall("c1", "read_file", {"path": "a.txt"})],
            ),
            ModelTurn(content="The file says hello.", tool_calls=[]),
        ]
    )
    agent = Agent(
        llm=model,
        tools=build_registry(tmp_path, shell_timeout=5),
        session=session,
        max_steps=5,
        metadata={
            "model": "scripted",
            "base_url": public_base_url("https://user:secret@example.com/v1"),
            "workspace": str(tmp_path),
        },
    )
    result = agent.run("文件里有什么？")
    assert result.finished
    assert "hello" in result.answer
    assert result.steps == 2
    events = load_events(session.path)
    assert [event["kind"] for event in events] == [
        "session_start",
        "user",
        "thought",
        "tool_call",
        "tool_result",
        "final",
    ]
    assert events[1]["content"] == "文件里有什么？"
    assert events[3]["step"] == 1
    assert events[4]["step"] == 1
    assert "hello" in events[4]["output"]
    assert any(message.get("role") == "tool" for message in model.seen[1])
    raw = session.path.read_text(encoding="utf-8")
    assert "secret" not in raw
    assert "example.com" in raw


def test_text_fallback_and_multiple_tools_share_a_step(tmp_path: Path) -> None:
    session = SessionLog.create(tmp_path / "sessions")
    model = ScriptedLLM(
        [
            ModelTurn(
                content=(
                    "I will create a file.\n"
                    '```json\n'
                    '{"tool": "write_file", "arguments": {"path": "out.txt", "content": "hi"}}\n'
                    "```"
                ),
                tool_calls=[],
            ),
            ModelTurn(
                content="Now list and read.",
                tool_calls=[
                    ToolCall("a", "list_dir", {"path": "."}),
                    ToolCall("a", "read_file", {"path": "out.txt"}),
                ],
            ),
            ModelTurn(content="Created out.txt.", tool_calls=[]),
        ]
    )
    result = Agent(
        llm=model,
        tools=build_registry(tmp_path, shell_timeout=5),
        session=session,
        max_steps=5,
    ).run("create out.txt")
    assert result.finished
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "hi"
    events = load_events(session.path)
    tool_calls = [event for event in events if event["kind"] == "tool_call"]
    assert [event["tool"] for event in tool_calls] == ["write_file", "list_dir", "read_file"]
    assert tool_calls[1]["step"] == tool_calls[2]["step"]
    assert tool_calls[1]["call_id"] != tool_calls[2]["call_id"]
    assert any(event["kind"] == "thought" and "create a file" in event["content"] for event in events)


def test_tool_error_is_an_observation_and_the_loop_continues(tmp_path: Path) -> None:
    session = SessionLog.create(tmp_path / "sessions")
    model = ScriptedLLM(
        [
            ModelTurn(
                content=None,
                tool_calls=[ToolCall("c1", "read_file", {"path": "missing.txt"})],
            ),
            ModelTurn(content="The file is not there.", tool_calls=[]),
        ]
    )
    result = Agent(
        llm=model,
        tools=build_registry(tmp_path, shell_timeout=5),
        session=session,
        max_steps=4,
    ).run("read missing.txt")
    assert result.finished
    tool_message = next(message for message in model.seen[1] if message["role"] == "tool")
    assert "Not a file" in tool_message["content"]


def test_max_steps_stops_without_a_final_answer(tmp_path: Path) -> None:
    session = SessionLog.create(tmp_path / "sessions")
    model = ScriptedLLM(
        [
            ModelTurn(content="again", tool_calls=[ToolCall("1", "list_dir", {})]),
            ModelTurn(content="again", tool_calls=[ToolCall("2", "list_dir", {})]),
            ModelTurn(content="should not run", tool_calls=[]),
        ]
    )
    result = Agent(
        llm=model,
        tools=build_registry(tmp_path, shell_timeout=5),
        session=session,
        max_steps=2,
    ).run("loop")
    assert result.finished is False
    assert "Stopped after 2 steps" in result.answer
    kinds = [event["kind"] for event in load_events(session.path)]
    assert kinds[-1] == "max_steps"
    assert "final" not in kinds
    assert model.turns  # the unused final turn is still queued


def test_llm_error_is_written_before_it_propagates(tmp_path: Path) -> None:
    session = SessionLog.create(tmp_path / "sessions")

    class Boom:
        def complete(self, messages, tools):
            del messages, tools
            raise LLMError("connection refused")

    agent = Agent(
        llm=Boom(),
        tools=build_registry(tmp_path, shell_timeout=5),
        session=session,
        max_steps=2,
    )
    with pytest.raises(LLMError, match="connection refused"):
        agent.run("hello")
    events = load_events(session.path)
    assert events[-1]["kind"] == "error"
    assert "connection refused" in events[-1]["content"]


def test_coerce_final_action_json() -> None:
    turn = coerce_model_turn(ModelTurn(content='{"action": "final", "content": "All set."}'))
    assert turn.tool_calls == []
    assert turn.content == "All set."
