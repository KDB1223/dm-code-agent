"""CLI session commands. Replay must not execute tools."""

import json
from pathlib import Path

import pytest

from dm_agent.cli import main
from dm_agent.present import format_changes
from dm_agent.session import SessionLog, load_events, session_changes


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "DM_AGENT_MAX_STEPS",
        "DM_AGENT_WORKSPACE",
        "DM_AGENT_LOG_DIR",
        "DM_AGENT_SHELL_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)


def _sample_log(workspace: Path) -> SessionLog:
    log = SessionLog.create(workspace / ".dm-agent" / "sessions")
    log.append(
        kind="session_start",
        step=0,
        task="fix the bug",
        model="scripted",
        workspace=str(workspace),
        log_version=1,
    )
    log.append(kind="user", step=0, content="<script>alert(1)</script>")
    log.append(
        kind="tool_call",
        step=1,
        tool="read_file",
        call_id="c1",
        arguments={"path": "a.py"},
    )
    log.append(kind="tool_result", step=1, tool="read_file", call_id="c1", ok=True, output="print(1)")
    log.append(kind="final", step=2, content="done")
    return log


def test_version_help_and_missing_key(monkeypatch, tmp_path: Path, capsys) -> None:
    _isolate(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert "dm-agent" in capsys.readouterr().out

    assert main(["run", "hello", "--workspace", str(tmp_path)]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err

    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("x", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert main(["run", "hello", "--workspace", str(not_a_dir)]) == 2


def test_sessions_list_show_replay_and_html(monkeypatch, tmp_path: Path, capsys) -> None:
    _isolate(monkeypatch, tmp_path)
    assert main(["sessions", "list", "--workspace", str(tmp_path)]) == 0
    assert "no sessions" in capsys.readouterr().out

    log = _sample_log(tmp_path)
    assert main(["sessions", "list", "--workspace", str(tmp_path)]) == 0
    listed = capsys.readouterr().out
    assert log.session_id in listed
    assert "fix the bug" in listed

    assert main(["sessions", "show", log.session_id[:18], "--workspace", str(tmp_path)]) == 0
    shown = capsys.readouterr().out
    assert "kind=tool_call" in shown
    assert "a.py" in shown

    assert main(["sessions", "replay", log.session_id, "--workspace", str(tmp_path)]) == 0
    replayed = capsys.readouterr().out
    assert "print(1)" in replayed
    assert "tools were not re-run" in replayed
    assert "step 1 tool_call read_file" in replayed

    assert main(["sessions", "stats", log.session_id, "--workspace", str(tmp_path)]) == 0
    summary = capsys.readouterr().out
    assert "turns 2" in summary
    assert "ended_cleanly yes" in summary
    assert "tools read_file=1" in summary
    assert "files (none)" in summary
    assert "OPENAI" not in summary

    out = tmp_path / "session.html"
    assert main(["sessions", "html", log.session_id + ".jsonl", "--workspace", str(tmp_path), "--output", str(out)]) == 0
    document = out.read_text(encoding="utf-8")
    assert "<script>" not in document
    assert "&lt;script&gt;" in document


def test_stats_json_is_offline_and_does_not_build_tools(monkeypatch, tmp_path: Path, capsys) -> None:
    _isolate(monkeypatch, tmp_path)
    log = _sample_log(tmp_path)
    log.append(
        kind="tool_call",
        step=2,
        tool="edit_file",
        call_id="c2",
        arguments={"path": "a.py", "old": "print(1)", "new": "print(2)"},
    )
    # The sample log already closed with final at step 2. This extra edit is
    # still in the file; the summary must follow the log, not re-run the tool.
    log.append(kind="tool_result", step=2, tool="edit_file", call_id="c2", ok=True, output="Updated a.py")

    def boom(*args, **kwargs):
        raise AssertionError("stats must not construct tools")

    monkeypatch.setattr("dm_agent.cli.build_registry", boom)

    def boom_llm(*args, **kwargs):
        raise AssertionError("stats must not call the model")

    monkeypatch.setattr("dm_agent.cli.OpenAICompatibleLLM", boom_llm)
    assert main(["sessions", "stats", log.session_id[:12], "--workspace", str(tmp_path), "--json"]) == 0
    first = capsys.readouterr().out
    assert main(["sessions", "stats", log.session_id, "--workspace", str(tmp_path), "--json"]) == 0
    second = capsys.readouterr().out
    assert first == second
    payload = json.loads(first)
    assert payload["turns"] == 2
    assert payload["ended_cleanly"] is True
    assert payload["tools"] == {"edit_file": 1, "read_file": 1}
    assert payload["files"] == ["a.py"]
    assert "api_key" not in payload


def test_changes_json_is_offline_and_matches_helper(monkeypatch, tmp_path: Path, capsys) -> None:
    _isolate(monkeypatch, tmp_path)
    log = _sample_log(tmp_path)
    log.append(
        kind="tool_call",
        step=2,
        tool="edit_file",
        call_id="c-edit",
        arguments={"path": "a.py", "old": "print(1)", "new": "print(2)"},
    )
    log.append(
        kind="tool_result",
        step=2,
        tool="edit_file",
        call_id="c-edit",
        ok=True,
        output="Updated a.py",
    )
    log.append(
        kind="tool_call",
        step=3,
        tool="edit_file",
        call_id="c-fail",
        arguments={"path": "a.py", "old": "missing", "new": "nope"},
    )
    log.append(
        kind="tool_result",
        step=3,
        tool="edit_file",
        call_id="c-fail",
        ok=False,
        output="old string was not found",
    )
    log.append(
        kind="tool_call",
        step=3,
        tool="write_file",
        call_id="c-write",
        arguments={"path": "notes/out.txt", "content": "你好"},
    )
    log.append(
        kind="tool_result",
        step=3,
        tool="write_file",
        call_id="c-write",
        ok=True,
        output="Wrote 6 bytes to notes/out.txt",
    )
    log.append(
        kind="tool_call",
        step=4,
        tool="write_file",
        call_id="c-open",
        arguments={"path": "notes/out.txt", "content": "later"},
    )

    def boom(*args, **kwargs):
        raise AssertionError("changes must not construct tools")

    monkeypatch.setattr("dm_agent.cli.build_registry", boom)

    def boom_llm(*args, **kwargs):
        raise AssertionError("changes must not call the model")

    monkeypatch.setattr("dm_agent.cli.OpenAICompatibleLLM", boom_llm)

    assert main(["sessions", "changes", log.session_id[:12], "--workspace", str(tmp_path), "--json"]) == 0
    first = capsys.readouterr().out
    assert main(["sessions", "changes", log.session_id, "--workspace", str(tmp_path), "--json"]) == 0
    second = capsys.readouterr().out
    assert first == second
    expected = session_changes(load_events(log.path), session_id=log.session_id)
    assert first == json.dumps(expected, indent=2, ensure_ascii=False) + "\n"
    payload = json.loads(first)
    assert payload["session_id"] == log.session_id
    assert [item["call_id"] for item in payload["changes"]] == ["c-edit", "c-write"]
    assert payload["changes"][1]["content"] == "你好"
    assert "\\u" not in first
    assert "c-fail" not in first
    assert "later" not in first
    assert not (tmp_path / "a.py").exists()
    assert not (tmp_path / "notes").exists()

    assert main(["sessions", "changes", log.session_id, "--workspace", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert text == f"file {log.path}\n" + format_changes(expected)
    assert "change 1 edit_file path=a.py step=2 call_id=c-edit\n" in text
    assert "你好" in text


def test_replay_does_not_build_tools(monkeypatch, tmp_path: Path, capsys) -> None:
    _isolate(monkeypatch, tmp_path)
    log = _sample_log(tmp_path)

    def boom(*args, **kwargs):
        raise AssertionError("replay must not construct tools")

    monkeypatch.setattr("dm_agent.cli.build_registry", boom)
    assert main(["sessions", "replay", log.session_id, "--workspace", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[-1]["kind"] == "final"


def test_ambiguous_session_id(monkeypatch, tmp_path: Path, capsys) -> None:
    _isolate(monkeypatch, tmp_path)
    log_dir = tmp_path / ".dm-agent" / "sessions"
    log_dir.mkdir(parents=True)
    for suffix, task in (("aaa111", "one"), ("aaa222", "two")):
        path = log_dir / f"20260101T000000Z-{suffix}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "ts": "2026-01-01T00:00:00.000Z",
                    "session_id": path.stem,
                    "seq": 1,
                    "step": 0,
                    "kind": "session_start",
                    "task": task,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    assert main(["sessions", "show", "20260101T000000Z-aaa", "--workspace", str(tmp_path)]) == 1
    assert "ambiguous" in capsys.readouterr().err
