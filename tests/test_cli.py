"""CLI session commands. Replay must not execute tools."""

import json
from pathlib import Path

import pytest

from dm_agent.cli import main
from dm_agent.session import SessionLog


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

    out = tmp_path / "session.html"
    assert main(["sessions", "html", log.session_id + ".jsonl", "--workspace", str(tmp_path), "--output", str(out)]) == 0
    document = out.read_text(encoding="utf-8")
    assert "<script>" not in document
    assert "&lt;script&gt;" in document


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
