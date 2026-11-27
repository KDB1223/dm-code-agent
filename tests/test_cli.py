"""CLI session commands. Replay must not execute tools."""

import json
from pathlib import Path

import pytest

from dm_agent.cli import main
from dm_agent.present import (
    format_changes,
    format_compare,
    format_failures,
    format_files,
    format_patch,
    format_shell,
    format_timing,
    format_tools,
)
from dm_agent.session import (
    SessionLog,
    load_events,
    session_changes,
    session_compare,
    session_failures,
    session_files,
    session_patch,
    session_shell,
    session_timing,
    session_tools,
    summarize_session,
)


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


def test_patch_json_is_offline_and_matches_helper(monkeypatch, tmp_path: Path, capsys) -> None:
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
        raise AssertionError("patch must not construct tools")

    monkeypatch.setattr("dm_agent.cli.build_registry", boom)

    def boom_llm(*args, **kwargs):
        raise AssertionError("patch must not call the model")

    monkeypatch.setattr("dm_agent.cli.OpenAICompatibleLLM", boom_llm)

    assert main(["sessions", "patch", log.session_id[:12], "--workspace", str(tmp_path), "--json"]) == 0
    first = capsys.readouterr().out
    assert main(["sessions", "patch", log.session_id, "--workspace", str(tmp_path), "--json"]) == 0
    second = capsys.readouterr().out
    assert first == second
    expected = session_patch(load_events(log.path), session_id=log.session_id)
    assert first == json.dumps(expected, indent=2, ensure_ascii=False) + "\n"
    payload = json.loads(first)
    assert payload["session_id"] == log.session_id
    assert [item["call_id"] for item in payload["patches"]] == ["c-edit", "c-write"]
    assert payload["patches"][0]["replace_all"] is False
    assert payload["patches"][1]["content"] == "你好"
    assert "\\u" not in first
    assert "c-fail" not in first
    assert "later" not in first
    assert "--- a/a.py\n" in payload["patches"][0]["diff"]
    assert "--- /dev/null\n" in payload["patches"][1]["diff"]
    assert not (tmp_path / "a.py").exists()
    assert not (tmp_path / "notes").exists()

    assert main(["sessions", "patch", log.session_id, "--workspace", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert text == f"file {log.path}\n" + format_patch(expected)
    assert "你好" in text
    joined = "\n".join(item["diff"] for item in expected["patches"])
    assert format_patch(expected) == f"session {log.session_id}\npatches 2\n\n{joined}"


def test_files_json_is_offline_and_matches_helper(monkeypatch, tmp_path: Path, capsys) -> None:
    _isolate(monkeypatch, tmp_path)
    log = _sample_log(tmp_path)
    log.append(
        kind="tool_call",
        step=2,
        tool="edit_file",
        call_id="c-fail",
        arguments={"path": "a.py", "old": "missing", "new": "nope"},
    )
    log.append(
        kind="tool_result",
        step=2,
        tool="edit_file",
        call_id="c-fail",
        ok=False,
        output="old string was not found",
    )
    log.append(
        kind="tool_call",
        step=3,
        tool="write_file",
        call_id="c-open",
        arguments={"path": "notes/你好.txt", "content": "你好"},
    )
    log.append(
        kind="tool_call",
        step=4,
        tool="run_shell",
        call_id="c-shell",
        arguments={"command": "pytest -q"},
    )
    log.append(
        kind="tool_result",
        step=4,
        tool="run_shell",
        call_id="c-shell",
        ok=True,
        output="passed",
    )
    log.append(
        kind="tool_call",
        step=5,
        tool="search",
        call_id="c-search",
        arguments={"path": "notes/你好.txt", "query": "你好"},
    )
    log.append(
        kind="tool_result",
        step=5,
        tool="search",
        call_id="c-search",
        ok=True,
        output="notes/你好.txt:1",
    )

    def boom(*args, **kwargs):
        raise AssertionError("files must not construct tools")

    monkeypatch.setattr("dm_agent.cli.build_registry", boom)

    def boom_llm(*args, **kwargs):
        raise AssertionError("files must not call the model")

    monkeypatch.setattr("dm_agent.cli.OpenAICompatibleLLM", boom_llm)

    assert main(["sessions", "files", log.session_id[:12], "--workspace", str(tmp_path), "--json"]) == 0
    first = capsys.readouterr().out
    assert main(["sessions", "files", log.session_id, "--workspace", str(tmp_path), "--json"]) == 0
    second = capsys.readouterr().out
    assert first == second
    expected = session_files(load_events(log.path), session_id=log.session_id)
    assert first == json.dumps(expected, indent=2, ensure_ascii=False) + "\n"
    payload = json.loads(first)
    assert payload["session_id"] == log.session_id
    assert [item["path"] for item in payload["files"]] == ["a.py", "notes/你好.txt"]
    assert payload["files"][0]["tools"] == [
        {"tool": "read_file", "ok": 1, "fail": 0},
        {"tool": "edit_file", "ok": 0, "fail": 1},
    ]
    assert payload["files"][1]["tools"] == [
        {"tool": "write_file", "ok": 0, "fail": 1},
        {"tool": "search", "ok": 1, "fail": 0},
    ]
    assert "你好" in first
    assert "\\u" not in first
    assert "run_shell" not in first
    assert not (tmp_path / "a.py").exists()
    assert not (tmp_path / "notes").exists()

    assert main(["sessions", "files", log.session_id, "--workspace", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert text == f"file {log.path}\n" + format_files(expected)
    assert "fail=1" in text
    assert "你好" in text


def test_tools_json_is_offline_and_matches_helper(monkeypatch, tmp_path: Path, capsys) -> None:
    _isolate(monkeypatch, tmp_path)
    log = _sample_log(tmp_path)
    log.append(
        kind="tool_call",
        step=2,
        tool="edit_file",
        call_id="c-fail",
        arguments={"path": "a.py", "old": "missing", "new": "nope"},
    )
    log.append(
        kind="tool_result",
        step=2,
        tool="edit_file",
        call_id="c-fail",
        ok=False,
        output="old string was not found",
    )
    log.append(
        kind="tool_call",
        step=3,
        tool="write_file",
        call_id="c-open",
        arguments={"path": "notes/你好.txt", "content": "你好"},
    )
    log.append(
        kind="tool_call",
        step=4,
        tool="run_shell",
        call_id="c-shell",
        arguments={"command": "pytest -q"},
    )
    log.append(
        kind="tool_result",
        step=4,
        tool="run_shell",
        call_id="c-shell",
        ok=True,
        output="passed",
    )
    log.append(
        kind="tool_call",
        step=5,
        tool="search",
        call_id="c-search",
        arguments={"path": "notes/你好.txt", "query": "你好"},
    )
    log.append(
        kind="tool_result",
        step=5,
        tool="search",
        call_id="c-search",
        ok=True,
        output="notes/你好.txt:1",
    )

    def boom(*args, **kwargs):
        raise AssertionError("tools must not construct tools")

    monkeypatch.setattr("dm_agent.cli.build_registry", boom)

    def boom_llm(*args, **kwargs):
        raise AssertionError("tools must not call the model")

    monkeypatch.setattr("dm_agent.cli.OpenAICompatibleLLM", boom_llm)

    assert main(["sessions", "tools", log.session_id[:12], "--workspace", str(tmp_path), "--json"]) == 0
    first = capsys.readouterr().out
    assert main(["sessions", "tools", log.session_id, "--workspace", str(tmp_path), "--json"]) == 0
    second = capsys.readouterr().out
    assert first == second
    expected = session_tools(load_events(log.path), session_id=log.session_id)
    assert first == json.dumps(expected, indent=2, ensure_ascii=False) + "\n"
    payload = json.loads(first)
    assert payload["session_id"] == log.session_id
    assert [item["tool"] for item in payload["tools"]] == [
        "read_file",
        "edit_file",
        "write_file",
        "run_shell",
        "search",
    ]
    assert payload["tools"][0] == {"tool": "read_file", "ok": 1, "fail": 0, "paths": ["a.py"]}
    assert payload["tools"][1] == {"tool": "edit_file", "ok": 0, "fail": 1, "paths": ["a.py"]}
    assert payload["tools"][2] == {
        "tool": "write_file",
        "ok": 0,
        "fail": 1,
        "paths": ["notes/你好.txt"],
    }
    assert payload["tools"][3] == {"tool": "run_shell", "ok": 1, "fail": 0, "paths": []}
    assert payload["tools"][4] == {
        "tool": "search",
        "ok": 1,
        "fail": 0,
        "paths": ["notes/你好.txt"],
    }
    assert "你好" in first
    assert "\\u" not in first
    assert "run_shell" in first
    assert not (tmp_path / "a.py").exists()
    assert not (tmp_path / "notes").exists()

    assert main(["sessions", "tools", log.session_id, "--workspace", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert text == f"file {log.path}\n" + format_tools(expected)
    assert "ok=1 fail=0" in text
    assert "run_shell ok=1 fail=0\n" in text
    assert "你好" in text


def test_failures_json_is_offline_and_matches_helper(monkeypatch, tmp_path: Path, capsys) -> None:
    _isolate(monkeypatch, tmp_path)
    log = _sample_log(tmp_path)
    log.append(
        kind="tool_call",
        step=2,
        tool="edit_file",
        call_id="c-fail",
        arguments={"path": "a.py", "old": "missing", "new": "nope"},
    )
    log.append(
        kind="tool_result",
        step=2,
        tool="edit_file",
        call_id="c-fail",
        ok=False,
        output="old string was not found",
    )
    log.append(
        kind="tool_call",
        step=3,
        tool="write_file",
        call_id="c-open",
        arguments={"path": "notes/你好.txt", "content": "你好"},
    )
    log.append(kind="error", step=4, content="connection reset")
    log.append(
        kind="tool_call",
        step=5,
        tool="run_shell",
        call_id="c-shell",
        arguments={"command": "pytest -q"},
    )
    log.append(
        kind="tool_result",
        step=5,
        tool="run_shell",
        call_id="c-shell",
        ok=False,
        output="failed",
    )
    log.append(kind="max_steps", step=6, content="stopped")

    def boom(*args, **kwargs):
        raise AssertionError("failures must not construct tools")

    monkeypatch.setattr("dm_agent.cli.build_registry", boom)

    def boom_llm(*args, **kwargs):
        raise AssertionError("failures must not call the model")

    monkeypatch.setattr("dm_agent.cli.OpenAICompatibleLLM", boom_llm)

    assert main(["sessions", "failures", log.session_id[:12], "--workspace", str(tmp_path), "--json"]) == 0
    first = capsys.readouterr().out
    assert main(["sessions", "failures", log.session_id, "--workspace", str(tmp_path), "--json"]) == 0
    second = capsys.readouterr().out
    assert first == second
    expected = session_failures(load_events(log.path), session_id=log.session_id)
    assert first == json.dumps(expected, indent=2, ensure_ascii=False) + "\n"
    payload = json.loads(first)
    assert payload["session_id"] == log.session_id
    assert [item["kind"] for item in payload["failures"]] == ["tool", "tool", "error", "tool"]
    assert payload["failures"][0]["tool"] == "edit_file"
    assert payload["failures"][0]["status"] == "failed"
    assert payload["failures"][0]["output"] == "old string was not found"
    assert payload["failures"][1]["path"] == "notes/你好.txt"
    assert payload["failures"][1]["status"] == "unpaired"
    assert payload["failures"][2]["content"] == "connection reset"
    assert payload["failures"][3]["tool"] == "run_shell"
    assert payload["failures"][3]["path"] is None
    assert "c1" not in [item.get("call_id") for item in payload["failures"]]
    assert "你好" in first
    assert "\\u" not in first
    assert "old string was not found" in first
    assert not (tmp_path / "a.py").exists()
    assert not (tmp_path / "notes").exists()

    assert main(["sessions", "failures", log.session_id, "--workspace", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert text == f"file {log.path}\n" + format_failures(expected)
    assert "status=failed" in text
    assert "old string was not found" in text
    assert "你好" in text
    assert "run_shell" in text


def test_sessions_help_lists_failures(capsys) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["sessions", "--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert (
        "{list,show,stats,changes,patch,files,tools,failures,timing,shell,compare,replay,html}"
        in help_text
    )
    assert "Tools are not executed and the model is not called." in " ".join(help_text.split())
    assert "unified diffs" in help_text
    assert "ok/fail counts" in help_text
    assert "failed tool calls and session error events" in help_text
    assert "Compare two recorded session logs" in help_text
    assert "timestamps already in the log" in " ".join(help_text.split())


def test_sessions_help_lists_timing(capsys) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["sessions", "--help"])
    assert caught.value.code == 0
    assert "timing" in capsys.readouterr().out

    with pytest.raises(SystemExit) as bare:
        main(["sessions"])
    assert bare.value.code == 0
    assert "timing" in capsys.readouterr().out

    with pytest.raises(SystemExit) as choice:
        main(["sessions", "help"])
    assert choice.value.code == 2
    assert "timing" in capsys.readouterr().err


def test_timing_json_is_offline_and_matches_helper(monkeypatch, tmp_path: Path, capsys) -> None:
    _isolate(monkeypatch, tmp_path)
    log_dir = tmp_path / ".dm-agent" / "sessions"
    log_dir.mkdir(parents=True)
    session_id = "20261108T100749Z-timing"
    events = [
        {
            "ts": "2026-11-08T10:07:49.000Z",
            "session_id": session_id,
            "seq": 1,
            "step": 0,
            "kind": "session_start",
            "task": "audit",
        },
        {
            "ts": "2026-11-08T10:07:49.100Z",
            "session_id": session_id,
            "seq": 2,
            "step": 1,
            "kind": "tool_call",
            "tool": "read_file",
            "call_id": "c-你好",
            "arguments": {"path": "a.py"},
        },
        {
            "ts": "2026-11-08T10:07:49.250Z",
            "session_id": session_id,
            "seq": 3,
            "step": 1,
            "kind": "tool_result",
            "tool": "read_file",
            "call_id": "c-你好",
            "ok": True,
            "output": "ok",
        },
        {
            "ts": None,
            "session_id": session_id,
            "seq": 4,
            "step": 2,
            "kind": "tool_call",
            "tool": "run_shell",
            "call_id": "c-open",
            "arguments": {"command": "pwd"},
        },
        {
            "ts": "2026-11-08T10:07:49.400Z",
            "session_id": session_id,
            "seq": 5,
            "step": 3,
            "kind": "final",
            "content": "done",
        },
    ]
    path = log_dir / f"{session_id}.jsonl"
    path.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
        encoding="utf-8",
    )

    def boom(*args, **kwargs):
        raise AssertionError("timing must not construct tools")

    monkeypatch.setattr("dm_agent.cli.build_registry", boom)
    monkeypatch.setattr("dm_agent.cli.Agent", boom)

    def boom_llm(*args, **kwargs):
        raise AssertionError("timing must not call the model")

    monkeypatch.setattr("dm_agent.cli.OpenAICompatibleLLM", boom_llm)

    assert main(["sessions", "timing", session_id[:12], "--workspace", str(tmp_path), "--json"]) == 0
    first = capsys.readouterr().out
    assert main(["sessions", "timing", session_id, "--workspace", str(tmp_path), "--json"]) == 0
    second = capsys.readouterr().out
    assert first == second
    expected = session_timing(load_events(path), session_id=session_id)
    assert first == json.dumps(expected, indent=2, ensure_ascii=False) + "\n"
    payload = json.loads(first)
    assert payload["session_id"] == session_id
    assert payload["duration_ms"] == 400
    assert payload["unparsed_ts"] == 1
    assert payload["calls"][0]["call_id"] == "c-你好"
    assert payload["calls"][0]["duration_ms"] == 150
    assert payload["calls"][1]["duration_ms"] is None
    assert "你好" in first
    assert "\\u" not in first
    assert "null" in first
    assert not (tmp_path / "a.py").exists()

    assert main(["sessions", "timing", session_id, "--workspace", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert text == f"file {path}\n" + format_timing(expected)
    assert "unknown" in text
    assert "150ms" in text

    empty = log_dir / "20261108T100800Z-empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert main(["sessions", "timing", empty.stem, "--workspace", str(tmp_path), "--json"]) == 0
    empty_payload = json.loads(capsys.readouterr().out)
    assert empty_payload["session_id"] == empty.stem
    assert empty_payload["duration_ms"] == 0
    assert empty_payload["steps"] == []
    assert empty_payload["calls"] == []


def test_timing_errors_match_sibling_commands(monkeypatch, tmp_path: Path, capsys) -> None:
    _isolate(monkeypatch, tmp_path)

    assert main(["sessions", "timing", "missing", "--workspace", str(tmp_path)]) == 1
    timing_err = capsys.readouterr().err
    assert main(["sessions", "stats", "missing", "--workspace", str(tmp_path)]) == 1
    assert timing_err == capsys.readouterr().err
    assert timing_err == "error: session not found: missing\n"

    log_dir = tmp_path / ".dm-agent" / "sessions"
    log_dir.mkdir(parents=True)
    for suffix in ("aaa111", "aaa222"):
        path = log_dir / f"20260101T000000Z-{suffix}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "ts": "2026-01-01T00:00:00.000Z",
                    "session_id": path.stem,
                    "seq": 1,
                    "step": 0,
                    "kind": "session_start",
                    "task": suffix,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    assert main(["sessions", "timing", "20260101T000000Z-aaa", "--workspace", str(tmp_path)]) == 1
    timing_err = capsys.readouterr().err
    assert main(["sessions", "show", "20260101T000000Z-aaa", "--workspace", str(tmp_path)]) == 1
    assert timing_err == capsys.readouterr().err
    assert "ambiguous" in timing_err

    assert main(["sessions", "timing", "bad/id", "--workspace", str(tmp_path)]) == 1
    timing_err = capsys.readouterr().err
    assert main(["sessions", "stats", "bad/id", "--workspace", str(tmp_path)]) == 1
    assert timing_err == capsys.readouterr().err
    assert "invalid session id" in timing_err

    assert main(["sessions", "timing", ".hidden", "--workspace", str(tmp_path)]) == 1
    timing_err = capsys.readouterr().err
    assert main(["sessions", "files", ".hidden", "--workspace", str(tmp_path)]) == 1
    assert timing_err == capsys.readouterr().err

    with pytest.raises(SystemExit) as timing_exit:
        main(["sessions", "timing", "--workspace", str(tmp_path)])
    timing_err = capsys.readouterr().err
    with pytest.raises(SystemExit) as stats_exit:
        main(["sessions", "stats", "--workspace", str(tmp_path)])
    stats_err = capsys.readouterr().err
    assert timing_exit.value.code == stats_exit.value.code == 2
    assert timing_err.strip().splitlines()[-1].replace("timing", "stats") == stats_err.strip().splitlines()[-1]
    assert "the following arguments are required: session_id" in timing_err
    assert "the following arguments are required: session_id" in stats_err


def test_sessions_help_lists_shell(capsys) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["sessions", "--help"])
    assert caught.value.code == 0
    parent = capsys.readouterr().out
    folded = " ".join(parent.split())
    assert "shell" in parent
    assert "Tools are not executed and the model is not called." in folded
    assert "run_shell" in folded

    with pytest.raises(SystemExit) as shell_help:
        main(["sessions", "shell", "--help"])
    assert shell_help.value.code == 0
    text = capsys.readouterr().out
    folded = " ".join(text.split())
    assert "Tools are not executed and the model is not called." in folded
    assert "run_shell" in folded
    assert "--json" in text
    assert "--workspace" in text
    assert "--config" in text
    assert "--log-dir" in text

    with pytest.raises(SystemExit) as bare:
        main(["sessions"])
    assert bare.value.code == 0
    assert "shell" in capsys.readouterr().out

    with pytest.raises(SystemExit) as choice:
        main(["sessions", "help"])
    assert choice.value.code == 2
    assert "shell" in capsys.readouterr().err


def test_shell_json_is_offline_and_matches_helper(monkeypatch, tmp_path: Path, capsys) -> None:
    _isolate(monkeypatch, tmp_path)
    log = _sample_log(tmp_path)
    log.append(
        kind="tool_call",
        step=3,
        tool="run_shell",
        call_id="c-ok",
        arguments={"command": "echo 你好", "timeout": 30},
    )
    log.append(
        kind="tool_result",
        step=3,
        tool="run_shell",
        call_id="c-ok",
        ok=True,
        output="你好\n",
    )
    log.append(
        kind="tool_call",
        step=4,
        tool="run_shell",
        call_id="c-refused",
        arguments={"command": "python3 -c 'open(\"/etc/passwd\")'"},
    )
    log.append(
        kind="tool_result",
        step=4,
        tool="run_shell",
        call_id="c-refused",
        ok=False,
        output="Refused: absolute path is outside the workspace",
    )
    log.append(
        kind="tool_call",
        step=5,
        tool="run_pytest",
        call_id="c-py",
        arguments={"path": "tests"},
    )
    log.append(
        kind="tool_result",
        step=5,
        tool="run_pytest",
        call_id="c-py",
        ok=False,
        output="failed",
    )
    log.append(
        kind="tool_call",
        step=6,
        tool="run_shell",
        call_id="c-open",
        arguments={"command": "pwd", "timeout": 1.5},
    )

    def boom(*args, **kwargs):
        raise AssertionError("shell must not construct tools")

    monkeypatch.setattr("dm_agent.cli.build_registry", boom)
    monkeypatch.setattr("dm_agent.cli.Agent", boom)

    def boom_llm(*args, **kwargs):
        raise AssertionError("shell must not call the model")

    monkeypatch.setattr("dm_agent.cli.OpenAICompatibleLLM", boom_llm)

    assert main(["sessions", "shell", log.session_id[:12], "--workspace", str(tmp_path), "--json"]) == 0
    first = capsys.readouterr().out
    assert main(["sessions", "shell", log.session_id, "--workspace", str(tmp_path), "--json"]) == 0
    second = capsys.readouterr().out
    assert first == second
    expected = session_shell(load_events(log.path), session_id=log.session_id)
    assert first == json.dumps(expected, indent=2, ensure_ascii=False) + "\n"
    payload = json.loads(first)
    assert payload["session_id"] == log.session_id
    assert payload["n"] == 3
    assert payload["n_ok"] == 1
    assert payload["n_failed"] == 1
    assert payload["n_unpaired"] == 1
    assert [item["call_id"] for item in payload["calls"]] == ["c-ok", "c-refused", "c-open"]
    assert payload["calls"][0]["command"] == "echo 你好"
    assert payload["calls"][0]["timeout"] == 30
    assert payload["calls"][0]["status"] == "ok"
    assert payload["calls"][0]["output"] == "你好\n"
    assert payload["calls"][1]["status"] == "failed"
    assert payload["calls"][1]["output"].startswith("Refused:")
    assert payload["calls"][2]["status"] == "unpaired"
    assert payload["calls"][2]["timeout"] == 1.5
    assert payload["calls"][2]["output"] is None
    assert "c1" not in [item["call_id"] for item in payload["calls"]]
    assert "run_pytest" not in first
    assert "你好" in first
    assert "\\u" not in first
    assert not (tmp_path / "a.py").exists()

    assert main(["sessions", "shell", log.session_id, "--workspace", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert text == f"file {log.path}\n" + format_shell(expected)
    assert "shell calls=3 ok=1 failed=1 unpaired=1" in text
    assert "echo 你好" in text
    assert "status=failed" in text
    assert "Refused:" in text

    empty = log.path.parent / "20261108T100800Z-empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert main(["sessions", "shell", empty.stem, "--workspace", str(tmp_path), "--json"]) == 0
    empty_out = capsys.readouterr().out
    empty_payload = json.loads(empty_out)
    assert empty_payload["session_id"] == empty.stem
    assert empty_payload["n"] == 0
    assert empty_payload["n_ok"] == 0
    assert empty_payload["n_failed"] == 0
    assert empty_payload["n_unpaired"] == 0
    assert empty_payload["calls"] == []
    assert main(["sessions", "shell", empty.stem, "--workspace", str(tmp_path)]) == 0
    assert "shell calls=0 ok=0 failed=0 unpaired=0" in capsys.readouterr().out

    custom = tmp_path / "custom-logs"
    custom_log = SessionLog.create(custom)
    custom_log.append(
        kind="tool_call",
        step=1,
        tool="run_shell",
        call_id="c-custom",
        arguments={"command": "true"},
    )
    custom_log.append(
        kind="tool_result",
        step=1,
        tool="run_shell",
        call_id="c-custom",
        ok=False,
        output="failed",
    )
    assert (
        main(
            [
                "sessions",
                "shell",
                custom_log.session_id,
                "--workspace",
                str(tmp_path),
                "--log-dir",
                str(custom),
                "--json",
            ]
        )
        == 0
    )
    custom_payload = json.loads(capsys.readouterr().out)
    assert custom_payload["n"] == 1
    assert custom_payload["n_failed"] == 1
    assert custom_payload["calls"][0]["command"] == "true"
    assert custom_payload["calls"][0]["status"] == "failed"


def test_shell_errors_match_sibling_commands(monkeypatch, tmp_path: Path, capsys) -> None:
    _isolate(monkeypatch, tmp_path)

    assert main(["sessions", "shell", "missing", "--workspace", str(tmp_path)]) == 1
    shell_err = capsys.readouterr().err
    assert main(["sessions", "stats", "missing", "--workspace", str(tmp_path)]) == 1
    assert shell_err == capsys.readouterr().err
    assert shell_err == "error: session not found: missing\n"

    log_dir = tmp_path / ".dm-agent" / "sessions"
    log_dir.mkdir(parents=True)
    for suffix in ("aaa111", "aaa222"):
        path = log_dir / f"20260101T000000Z-{suffix}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "ts": "2026-01-01T00:00:00.000Z",
                    "session_id": path.stem,
                    "seq": 1,
                    "step": 0,
                    "kind": "session_start",
                    "task": suffix,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    assert main(["sessions", "shell", "20260101T000000Z-aaa", "--workspace", str(tmp_path)]) == 1
    shell_err = capsys.readouterr().err
    assert main(["sessions", "show", "20260101T000000Z-aaa", "--workspace", str(tmp_path)]) == 1
    assert shell_err == capsys.readouterr().err
    assert "ambiguous" in shell_err

    assert main(["sessions", "shell", "bad/id", "--workspace", str(tmp_path)]) == 1
    shell_err = capsys.readouterr().err
    assert main(["sessions", "stats", "bad/id", "--workspace", str(tmp_path)]) == 1
    assert shell_err == capsys.readouterr().err
    assert "invalid session id" in shell_err

    assert main(["sessions", "shell", ".hidden", "--workspace", str(tmp_path)]) == 1
    shell_err = capsys.readouterr().err
    assert main(["sessions", "files", ".hidden", "--workspace", str(tmp_path)]) == 1
    assert shell_err == capsys.readouterr().err

    missing_cfg = tmp_path / "missing.toml"
    assert (
        main(
            [
                "sessions",
                "shell",
                "missing",
                "--config",
                str(missing_cfg),
                "--workspace",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert "not found" in capsys.readouterr().err

    with pytest.raises(SystemExit) as shell_exit:
        main(["sessions", "shell", "--workspace", str(tmp_path)])
    shell_err = capsys.readouterr().err
    with pytest.raises(SystemExit) as stats_exit:
        main(["sessions", "stats", "--workspace", str(tmp_path)])
    stats_err = capsys.readouterr().err
    assert shell_exit.value.code == stats_exit.value.code == 2
    assert shell_err.strip().splitlines()[-1].replace("shell", "stats") == stats_err.strip().splitlines()[-1]
    assert "the following arguments are required: session_id" in shell_err
    assert "the following arguments are required: session_id" in stats_err


def test_replay_does_not_build_tools(monkeypatch, tmp_path: Path, capsys) -> None:
    _isolate(monkeypatch, tmp_path)
    log = _sample_log(tmp_path)

    def boom(*args, **kwargs):
        raise AssertionError("replay must not construct tools")

    monkeypatch.setattr("dm_agent.cli.build_registry", boom)
    assert main(["sessions", "replay", log.session_id, "--workspace", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[-1]["kind"] == "final"


def test_compare_json_is_offline_and_matches_helper(monkeypatch, tmp_path: Path, capsys) -> None:
    _isolate(monkeypatch, tmp_path)
    log_dir = tmp_path / "custom-logs"
    baseline = SessionLog.create(log_dir)
    baseline.append(
        kind="session_start",
        step=0,
        task="baseline",
        model="scripted",
        workspace=str(tmp_path),
        log_version=1,
    )
    baseline.append(
        kind="tool_call",
        step=1,
        tool="read_file",
        call_id="c1",
        arguments={"path": "a.py"},
    )
    baseline.append(
        kind="tool_result",
        step=1,
        tool="read_file",
        call_id="c1",
        ok=True,
        output="x",
    )
    baseline.append(
        kind="tool_call",
        step=2,
        tool="edit_file",
        call_id="c2",
        arguments={"path": "a.py", "old": "x", "new": "y"},
    )
    baseline.append(
        kind="tool_result",
        step=2,
        tool="edit_file",
        call_id="c2",
        ok=True,
        output="Updated a.py",
    )
    baseline.append(kind="final", step=3, content="done")

    candidate = SessionLog.create(log_dir)
    candidate.append(
        kind="session_start",
        step=0,
        task="candidate",
        model="scripted",
        workspace=str(tmp_path),
        log_version=1,
    )
    candidate.append(
        kind="tool_call",
        step=1,
        tool="read_file",
        call_id="c1",
        arguments={"path": "a.py"},
    )
    candidate.append(
        kind="tool_result",
        step=1,
        tool="read_file",
        call_id="c1",
        ok=True,
        output="x",
    )
    candidate.append(
        kind="tool_call",
        step=2,
        tool="edit_file",
        call_id="c-fail",
        arguments={"path": "a.py", "old": "nope", "new": "y"},
    )
    candidate.append(
        kind="tool_result",
        step=2,
        tool="edit_file",
        call_id="c-fail",
        ok=False,
        output="old string was not found",
    )
    candidate.append(
        kind="tool_call",
        step=3,
        tool="write_file",
        call_id="c-write",
        arguments={"path": "notes/你好.txt", "content": "你好"},
    )
    candidate.append(
        kind="tool_result",
        step=3,
        tool="write_file",
        call_id="c-write",
        ok=True,
        output="wrote",
    )
    candidate.append(kind="error", step=4, content="connection reset")

    def boom(*args, **kwargs):
        raise AssertionError("compare must not construct tools")

    monkeypatch.setattr("dm_agent.cli.build_registry", boom)

    def boom_llm(*args, **kwargs):
        raise AssertionError("compare must not call the model")

    monkeypatch.setattr("dm_agent.cli.OpenAICompatibleLLM", boom_llm)

    shared = 0
    for left, right in zip(baseline.session_id, candidate.session_id):
        if left != right:
            break
        shared += 1
    prefix = baseline.session_id[: shared + 1]
    assert baseline.session_id.startswith(prefix)
    assert not candidate.session_id.startswith(prefix)
    argv = [
        "sessions",
        "compare",
        prefix,
        candidate.session_id,
        "--workspace",
        str(tmp_path),
        "--log-dir",
        str(log_dir),
        "--json",
    ]
    assert main(argv) == 0
    first = capsys.readouterr().out
    assert main(
        [
            "sessions",
            "compare",
            baseline.session_id + ".jsonl",
            candidate.session_id,
            "--workspace",
            str(tmp_path),
            "--log-dir",
            str(log_dir),
            "--json",
        ]
    ) == 0
    second = capsys.readouterr().out
    assert first == second
    summary_a = summarize_session(baseline.path)
    summary_b = summarize_session(candidate.path)
    expected = session_compare(
        load_events(baseline.path),
        load_events(candidate.path),
        session_id_a=baseline.path.stem,
        session_id_b=candidate.path.stem,
        status_a=summary_a["status"],
        status_b=summary_b["status"],
    )
    assert first == json.dumps(expected, indent=2, ensure_ascii=False) + "\n"
    assert not first.startswith("file_a")
    assert "file_b" not in first
    payload = json.loads(first)
    assert payload["session_a"]["session_id"] == summary_a["session_id"]
    assert payload["session_b"]["session_id"] == summary_b["session_id"]
    assert payload["session_a"]["status"] == "final"
    assert payload["session_b"]["status"] == "error"
    assert payload["session_a"]["ended_cleanly"] is True
    assert payload["session_b"]["ended_cleanly"] is False
    assert payload["turns"]["delta"] == 1
    assert payload["failures"]["delta"] == payload["failures"]["b"] - payload["failures"]["a"]
    assert payload["failures"]["a"] == 0
    assert payload["failures"]["b"] == 2
    assert payload["files"] == {
        "only_a": ["a.py"],
        "only_b": ["notes/你好.txt"],
        "both": [],
    }
    assert "你好" in first
    assert "\\u" not in first
    assert not (tmp_path / "a.py").exists()
    assert not (tmp_path / "notes").exists()

    assert main(
        [
            "sessions",
            "compare",
            baseline.session_id,
            candidate.session_id,
            "--workspace",
            str(tmp_path),
            "--log-dir",
            str(log_dir),
        ]
    ) == 0
    text = capsys.readouterr().out
    assert text == f"file_a {baseline.path}\nfile_b {candidate.path}\n" + format_compare(expected)
    assert "status         final -> error" in text
    assert "notes/你好.txt" in text

    assert main(
        [
            "sessions",
            "compare",
            baseline.session_id,
            baseline.session_id,
            "--workspace",
            str(tmp_path),
            "--log-dir",
            str(log_dir),
            "--json",
        ]
    ) == 0
    same = json.loads(capsys.readouterr().out)
    assert same["turns"]["delta"] == 0
    assert same["failures"]["delta"] == 0
    assert all(row["delta"] == 0 for row in same["tools"])
    assert same["files"]["only_a"] == []
    assert same["files"]["only_b"] == []
    assert same["files"]["both"] == ["a.py"]
    assert same["session_a"]["session_id"] == same["session_b"]["session_id"] == baseline.session_id


def test_compare_rejects_unknown_ambiguous_and_bad_usage(monkeypatch, tmp_path: Path, capsys) -> None:
    _isolate(monkeypatch, tmp_path)
    assert main(["sessions", "compare", "missing-a", "missing-b", "--workspace", str(tmp_path)]) == 1
    assert "not found" in capsys.readouterr().err

    log_dir = tmp_path / ".dm-agent" / "sessions"
    log_dir.mkdir(parents=True)
    for suffix in ("aaa111", "aaa222"):
        path = log_dir / f"20260101T000000Z-{suffix}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "ts": "2026-01-01T00:00:00.000Z",
                    "session_id": path.stem,
                    "seq": 1,
                    "step": 0,
                    "kind": "session_start",
                    "task": suffix,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    other = log_dir / "20260102T000000Z-bbb333.jsonl"
    other.write_text(
        json.dumps(
            {
                "ts": "2026-01-02T00:00:00.000Z",
                "session_id": other.stem,
                "seq": 1,
                "step": 1,
                "kind": "final",
                "content": "done",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "sessions",
                "compare",
                other.stem,
                "20260101T000000Z-aaa",
                "--workspace",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert "ambiguous" in capsys.readouterr().err
    assert main(["sessions", "compare", "bad/id", other.stem, "--workspace", str(tmp_path)]) == 1
    assert "invalid" in capsys.readouterr().err
    assert main(["sessions", "compare", other.stem, ".hidden", "--workspace", str(tmp_path)]) == 1
    assert "invalid" in capsys.readouterr().err
    missing_cfg = tmp_path / "missing.toml"
    assert (
        main(
            [
                "sessions",
                "compare",
                other.stem,
                other.stem,
                "--config",
                str(missing_cfg),
                "--workspace",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert "not found" in capsys.readouterr().err
    with pytest.raises(SystemExit) as caught:
        main(["sessions", "compare", "only-one"])
    assert caught.value.code == 2


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
