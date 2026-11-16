"""JSONL session log, inspection, and dry replay."""

import json
from pathlib import Path

from dm_agent.present import (
    format_changes,
    format_inspection,
    format_patch,
    format_stats,
    format_transcript,
    render_html,
)
from dm_agent.session import (
    SessionLog,
    load_events,
    resolve_session_file,
    session_changes,
    session_patch,
    session_stats,
    summarize_session,
)


def test_append_is_durable_and_one_json_object_per_line(tmp_path: Path) -> None:
    log = SessionLog.create(tmp_path)
    first = log.append(kind="user", step=0, content="line one\nline two")
    snapshot = log.path.read_bytes()
    second = log.append(kind="final", step=1, content="完成")
    assert log.path.read_bytes().startswith(snapshot)
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["content"] == "line one\nline two"
    assert json.loads(lines[1])["content"] == "完成"
    assert first["seq"] == 1
    assert second["seq"] == 2
    assert first["ts"].endswith("Z")
    events = load_events(log.path)
    assert [event["kind"] for event in events] == ["user", "final"]


def test_replay_prints_tool_results_without_rerunning(tmp_path: Path) -> None:
    log = SessionLog.create(tmp_path)
    log.append(kind="session_start", step=0, task="fix it", model="m", workspace=str(tmp_path))
    log.append(kind="user", step=0, content="fix it")
    log.append(
        kind="tool_call",
        step=1,
        tool="read_file",
        call_id="c1",
        arguments={"path": "a.py"},
    )
    log.append(
        kind="tool_result",
        step=1,
        tool="read_file",
        call_id="c1",
        ok=True,
        output="print('hello')",
    )
    log.append(kind="final", step=2, content="done")
    events = load_events(log.path)
    transcript = format_transcript(events)
    assert "print('hello')" in transcript
    assert "tool_call read_file" in transcript
    assert "step 1" in transcript
    assert "tools were not re-run" in transcript
    inspection = format_inspection(events)
    assert "kind=tool_result" in inspection
    assert log.session_id in inspection
    summary = summarize_session(log.path)
    assert summary["status"] == "final"
    assert summary["task"] == "fix it"
    assert summary["events"] == 5


def test_session_stats_counts_turns_tools_and_clean_edits() -> None:
    events = [
        {
            "session_id": "20260101T000000Z-abc",
            "step": 0,
            "kind": "session_start",
            "task": "fix it",
        },
        {"step": 0, "kind": "user", "content": "fix it"},
        {
            "step": 1,
            "kind": "thought",
            "content": "read first",
        },
        {
            "step": 1,
            "kind": "tool_call",
            "tool": "read_file",
            "call_id": "c-read",
            "arguments": {"path": "math_utils.py"},
        },
        {"step": 1, "kind": "tool_result", "tool": "read_file", "call_id": "c-read", "ok": True},
        {
            "step": 2,
            "kind": "tool_call",
            "tool": "edit_file",
            "call_id": "c-fail",
            "arguments": {"path": "missing.py"},
        },
        {
            "step": 2,
            "kind": "tool_result",
            "tool": "edit_file",
            "call_id": "c-fail",
            "ok": False,
            "output": "Not a file: missing.py",
        },
        {
            "step": 2,
            "kind": "tool_call",
            "tool": "write_file",
            "call_id": "c-write",
            "arguments": {"path": "notes/out.txt"},
        },
        {"step": 2, "kind": "tool_result", "tool": "write_file", "call_id": "c-write", "ok": True},
        {
            "step": 3,
            "kind": "tool_call",
            "tool": "edit_file",
            "call_id": "c-edit",
            "arguments": {"path": "math_utils.py"},
        },
        {"step": 3, "kind": "tool_result", "tool": "edit_file", "call_id": "c-edit", "ok": True},
        {
            "step": 3,
            "kind": "tool_call",
            "tool": "edit_file",
            "call_id": "c-edit-again",
            "arguments": {"path": "math_utils.py"},
        },
        {
            "step": 3,
            "kind": "tool_result",
            "tool": "edit_file",
            "call_id": "c-edit-again",
            "ok": True,
        },
        {"step": 4, "kind": "final", "content": "done"},
    ]
    stats = session_stats(events)
    assert stats == {
        "session_id": "20260101T000000Z-abc",
        "turns": 4,
        "ended_cleanly": True,
        "tools": {"edit_file": 3, "read_file": 1, "write_file": 1},
        "files": ["math_utils.py", "notes/out.txt"],
    }
    reordered = [events[0], events[1], *events[7:11], *events[2:7], *events[11:]]
    assert session_stats(reordered) == stats
    text = format_stats(stats)
    assert text == format_stats(session_stats(reordered))
    assert "turns 4" in text
    assert "ended_cleanly yes" in text
    assert "tools edit_file=3 read_file=1 write_file=1" in text
    assert "math_utils.py" in text
    assert "missing.py" not in text
    assert "notes/out.txt" in text


def test_session_stats_open_error_and_max_steps() -> None:
    open_stats = session_stats(
        [
            {"session_id": "s", "step": 1, "kind": "tool_call", "tool": "search", "arguments": {"path": "."}},
            {"step": 1, "kind": "tool_result", "tool": "search", "ok": True},
            {
                "step": 1,
                "kind": "tool_call",
                "tool": "write_file",
                "call_id": "unpaired",
                "arguments": {"path": "new.py"},
            },
        ]
    )
    assert open_stats["turns"] == 1
    assert open_stats["ended_cleanly"] is False
    assert open_stats["tools"] == {"search": 1, "write_file": 1}
    assert open_stats["files"] == []
    assert "files (none)" in format_stats(open_stats)

    stopped = session_stats([{"step": 2, "kind": "max_steps", "content": "stopped"}])
    assert stopped["turns"] == 1
    assert stopped["ended_cleanly"] is False
    assert "ended_cleanly no" in format_stats(stopped)

    failed = session_stats(
        [
            {"step": 1, "kind": "error", "content": "boom"},
            {"step": 2, "kind": "final", "content": "too late"},
            {"step": 3, "kind": "error", "content": "still broken"},
        ]
    )
    assert failed["turns"] == 3
    assert failed["ended_cleanly"] is False
    assert format_stats({}) == (
        "session (unknown)\nturns 0\nended_cleanly no\ntools (none)\nfiles (none)\n"
    )


def test_session_changes_reconstructs_logged_mutations() -> None:
    start = {
        "session_id": "20260101T000000Z-abc",
        "step": 0,
        "kind": "session_start",
        "task": "fix it",
    }
    user = {"step": 0, "kind": "user", "content": "fix it"}
    read_call = {
        "step": 1,
        "kind": "tool_call",
        "tool": "read_file",
        "call_id": "c-read",
        "arguments": {"path": "math_utils.py"},
    }
    read_result = {
        "step": 1,
        "kind": "tool_result",
        "tool": "read_file",
        "call_id": "c-read",
        "ok": True,
        "output": "return a - b",
    }
    edit_call = {
        "step": 2,
        "kind": "tool_call",
        "tool": "edit_file",
        "call_id": "c-edit",
        "arguments": {
            "path": "math_utils.py",
            "old": "return a - b",
            "new": "return a + b",
        },
    }
    edit_result = {
        "step": 2,
        "kind": "tool_result",
        "tool": "edit_file",
        "call_id": "c-edit",
        "ok": True,
        "output": "Updated math_utils.py",
    }
    write_call = {
        "step": 2,
        "kind": "tool_call",
        "tool": "write_file",
        "call_id": "c-write",
        "arguments": {"path": "notes/out.txt", "content": "hello"},
    }
    write_result = {
        "step": 2,
        "kind": "tool_result",
        "tool": "write_file",
        "call_id": "c-write",
        "ok": True,
    }
    final = {"step": 3, "kind": "final", "content": "done"}
    events = [
        start,
        user,
        read_call,
        read_result,
        edit_call,
        edit_result,
        write_call,
        write_result,
        final,
    ]
    payload = session_changes(events)
    assert payload == {
        "session_id": "20260101T000000Z-abc",
        "changes": [
            {
                "tool": "edit_file",
                "path": "math_utils.py",
                "step": 2,
                "call_id": "c-edit",
                "old": "return a - b",
                "new": "return a + b",
                "replace_all": False,
            },
            {
                "tool": "write_file",
                "path": "notes/out.txt",
                "step": 2,
                "call_id": "c-write",
                "content": "hello",
            },
        ],
    }
    assert list(payload) == ["session_id", "changes"]
    assert list(payload["changes"][0]) == [
        "tool",
        "path",
        "step",
        "call_id",
        "old",
        "new",
        "replace_all",
    ]
    assert list(payload["changes"][1]) == ["tool", "path", "step", "call_id", "content"]
    text = format_changes(payload)
    assert text == (
        "session 20260101T000000Z-abc\n"
        "changes 2\n"
        "\n"
        "change 1 edit_file path=math_utils.py step=2 call_id=c-edit\n"
        "old:\n"
        "return a - b\n"
        "new:\n"
        "return a + b\n"
        "\n"
        "change 2 write_file path=notes/out.txt step=2 call_id=c-write\n"
        "content:\n"
        "hello\n"
    )
    # Non-change events, including the paired results, can move. The text
    # follows the successful tool_call order.
    reordered = [
        write_result,
        start,
        read_result,
        user,
        edit_call,
        final,
        read_call,
        edit_result,
        write_call,
    ]
    assert session_changes(reordered)["changes"] == payload["changes"]
    assert format_changes(session_changes(reordered)) == text


def test_session_changes_skips_failed_unpaired_and_incomplete_calls() -> None:
    events = [
        {"session_id": "s", "step": 1, "kind": "thought", "content": "edit"},
        {
            "step": 1,
            "kind": "tool_call",
            "tool": "edit_file",
            "call_id": "c-fail",
            "arguments": {"path": "math_utils.py", "old": "a", "new": "b"},
        },
        {
            "step": 1,
            "kind": "tool_result",
            "tool": "edit_file",
            "call_id": "c-fail",
            "ok": False,
            "output": "old string was not found in the file",
        },
        {
            "step": 1,
            "kind": "tool_call",
            "tool": "write_file",
            "call_id": "c-unpaired",
            "arguments": {"path": "notes/out.txt", "content": "nope"},
        },
        {
            "step": 2,
            "kind": "tool_call",
            "tool": "edit_file",
            "call_id": "c-missing",
            "arguments": {"path": "math_utils.py"},
        },
        {"step": 2, "kind": "tool_result", "call_id": "c-missing", "ok": True},
        {
            "step": 2,
            "kind": "tool_call",
            "tool": "write_file",
            "call_id": "c-bad-content",
            "arguments": {"path": "notes/out.txt", "content": None},
        },
        {"step": 2, "kind": "tool_result", "call_id": "c-bad-content", "ok": True},
        {
            "step": 2,
            "kind": "tool_call",
            "tool": "write_file",
            "call_id": "c-empty-path",
            "arguments": {"path": "", "content": "x"},
        },
        {"step": 2, "kind": "tool_result", "call_id": "c-empty-path", "ok": True},
        {
            "step": 2,
            "kind": "tool_call",
            "tool": "edit_file",
            "call_id": "c-num-ok",
            "arguments": {"path": "num.py", "old": "a", "new": "b"},
        },
        {"step": 2, "kind": "tool_result", "call_id": "c-num-ok", "ok": 1},
        {
            "step": 2,
            "kind": "tool_call",
            "tool": "write_file",
            "call_id": "c-no-args",
        },
        {"step": 2, "kind": "tool_result", "call_id": "c-no-args", "ok": True},
        {
            "step": 3,
            "kind": "tool_call",
            "tool": "search",
            "call_id": "c-search",
            "arguments": {"path": "math_utils.py"},
        },
        {"step": 3, "kind": "tool_result", "call_id": "c-search", "ok": True},
        {
            "step": 4,
            "kind": "tool_call",
            "tool": "edit_file",
            "call_id": "c-first",
            "arguments": {
                "path": "math_utils.py",
                "old": "line one\nline two",
                "new": "\nline three",
            },
        },
        {"step": 4, "kind": "tool_result", "call_id": "c-first", "ok": True},
        {
            "step": 4,
            "kind": "tool_call",
            "tool": "edit_file",
            "call_id": "c-second",
            "arguments": {
                "path": "math_utils.py",
                "old": "keep\n",
                "new": "kept",
                "replace_all": False,
            },
        },
        {"step": 4, "kind": "tool_result", "call_id": "c-second", "ok": True},
        {
            "step": True,
            "kind": "tool_call",
            "tool": "edit_file",
            "call_id": "c-all",
            "arguments": {
                "path": "math_utils.py",
                "old": "a",
                "new": "b",
                "replace_all": True,
            },
        },
        {"kind": "tool_result", "call_id": "c-all", "ok": True},
        {
            "step": "5",
            "kind": "tool_call",
            "tool": "edit_file",
            "call_id": "c-flag",
            "arguments": {
                "path": "other.py",
                "old": "a",
                "new": "b",
                "replace_all": "yes",
            },
        },
        {"kind": "tool_result", "call_id": "c-flag", "ok": True},
        {
            "kind": "tool_call",
            "tool": "write_file",
            "call_id": "c-empty",
            "arguments": {"path": "notes/blank.txt", "content": ""},
        },
        {"kind": "tool_result", "call_id": "c-empty", "ok": True},
        {
            "step": 6,
            "kind": "tool_call",
            "tool": "lint_python",
            "call_id": "c-lint",
            "arguments": {"path": "math_utils.py"},
        },
        {"step": 6, "kind": "tool_result", "call_id": "c-lint", "ok": True},
    ]
    payload = session_changes(events)
    assert [item["call_id"] for item in payload["changes"]] == [
        "c-first",
        "c-second",
        "c-all",
        "c-flag",
        "c-empty",
    ]
    assert [item["path"] for item in payload["changes"]] == [
        "math_utils.py",
        "math_utils.py",
        "math_utils.py",
        "other.py",
        "notes/blank.txt",
    ]
    assert payload["changes"][0]["replace_all"] is False
    assert payload["changes"][1]["replace_all"] is False
    assert payload["changes"][2]["replace_all"] is True
    assert payload["changes"][2]["step"] is None
    assert payload["changes"][3]["replace_all"] is False
    assert payload["changes"][3]["step"] is None
    assert payload["changes"][4] == {
        "tool": "write_file",
        "path": "notes/blank.txt",
        "step": None,
        "call_id": "c-empty",
        "content": "",
    }
    # stats still lists each successful path once. changes keeps every call.
    assert session_stats(events)["files"] == [
        "math_utils.py",
        "notes/blank.txt",
        "notes/out.txt",
        "other.py",
    ]
    text = format_changes(payload)
    assert text.count("replace_all=true") == 1
    assert "replace_all=false" not in text
    assert "c-fail" not in text
    assert "c-unpaired" not in text
    assert "c-missing" not in text
    assert "nope" not in text
    assert "change 1 edit_file path=math_utils.py step=4 call_id=c-first\n" in text
    assert "old:\nline one\nline two\nnew:\n\nline three\n" in text
    assert "old:\nkeep\n\nnew:\nkept\n" in text
    assert "change 3 edit_file path=math_utils.py call_id=c-all replace_all=true\n" in text
    assert "step=True" not in text
    assert text.endswith("content:\n")
    assert not text.endswith("content:\n\n")


def test_session_changes_empty_events() -> None:
    assert session_changes([]) == {"session_id": "", "changes": []}
    assert format_changes(session_changes([])) == "session (unknown)\nchanges 0\n"
    assert format_changes({"session_id": "s", "changes": []}) == "session s\nchanges 0\n"
    fallback = session_changes(
        [{"kind": "final", "step": 1, "content": "done"}],
        session_id="from-stem",
    )
    assert fallback == {"session_id": "from-stem", "changes": []}
    assert format_changes(fallback) == "session from-stem\nchanges 0\n"


def _call(step: int, tool: str, call_id: str, arguments: dict) -> dict:
    return {
        "step": step,
        "kind": "tool_call",
        "tool": tool,
        "call_id": call_id,
        "arguments": arguments,
    }


def _ok(step: int, call_id: str, *, ok: bool = True) -> dict:
    return {"step": step, "kind": "tool_result", "call_id": call_id, "ok": ok}


def test_session_patch_edit_file_hunks() -> None:
    events = [
        {"session_id": "s", "step": 0, "kind": "session_start", "task": "edit"},
        _call(
            2,
            "edit_file",
            "c-one",
            {"path": "a.py", "old": "print(1)\n", "new": "print(2)\n"},
        ),
        _ok(2, "c-one"),
        _call(
            3,
            "edit_file",
            "c-multi",
            {
                "path": "a.py",
                "old": "alpha\nbeta\n",
                "new": "alpha\ngamma\n",
            },
        ),
        _ok(3, "c-multi"),
        _call(
            4,
            "edit_file",
            "c-insert",
            {"path": "new.py", "old": "", "new": "added\n"},
        ),
        _ok(4, "c-insert"),
        _call(
            4,
            "edit_file",
            "c-delete",
            {"path": "old.py", "old": "gone\n", "new": ""},
        ),
        _ok(4, "c-delete"),
    ]
    payload = session_patch(events)
    assert list(payload) == ["session_id", "patches"]
    assert [item["call_id"] for item in payload["patches"]] == [
        "c-one",
        "c-multi",
        "c-insert",
        "c-delete",
    ]
    assert list(payload["patches"][0]) == [
        "tool",
        "path",
        "step",
        "call_id",
        "old",
        "new",
        "replace_all",
        "diff",
    ]
    single = (
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-print(1)\n"
        "+print(2)\n"
    )
    multi = (
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-alpha\n"
        "-beta\n"
        "+alpha\n"
        "+gamma\n"
    )
    inserted = (
        "--- a/new.py\n"
        "+++ b/new.py\n"
        "@@ -1,0 +1,1 @@\n"
        "+added\n"
    )
    deleted = (
        "--- a/old.py\n"
        "+++ b/old.py\n"
        "@@ -1,1 +1,0 @@\n"
        "-gone\n"
    )
    assert payload["patches"][0]["diff"] == single
    assert payload["patches"][1]["diff"] == multi
    assert payload["patches"][2]["diff"] == inserted
    assert payload["patches"][3]["diff"] == deleted
    # Same path, two calls: two diffs, not one merged file.
    assert payload["patches"][0]["path"] == payload["patches"][1]["path"] == "a.py"
    assert single.count("--- a/a.py") == 1
    text = format_patch(payload)
    joined = "\n".join(item["diff"] for item in payload["patches"])
    assert text == f"session s\npatches 4\n\n{joined}"
    assert text.count("--- a/a.py\n") == 2


def test_session_patch_write_file_hunks() -> None:
    events = [
        {"session_id": "s", "step": 3, "kind": "user", "content": "write"},
        _call(3, "write_file", "c-write", {"path": "notes/out.txt", "content": "你好"}),
        _ok(3, "c-write"),
        _call(4, "write_file", "c-nl", {"path": "notes/out.txt", "content": "你好\n"}),
        _ok(4, "c-nl"),
        _call(5, "write_file", "c-empty", {"path": "notes/blank.txt", "content": ""}),
        _ok(5, "c-empty"),
    ]
    payload = session_patch(events)
    assert list(payload["patches"][0]) == [
        "tool",
        "path",
        "step",
        "call_id",
        "content",
        "diff",
    ]
    assert "old" not in payload["patches"][0]
    assert "replace_all" not in payload["patches"][0]
    unicode_diff = (
        "--- /dev/null\n"
        "+++ b/notes/out.txt\n"
        "@@ -0,0 +1,1 @@\n"
        "+你好\n"
        "\\ No newline at end of file\n"
    )
    newline_diff = (
        "--- /dev/null\n"
        "+++ b/notes/out.txt\n"
        "@@ -0,0 +1,1 @@\n"
        "+你好\n"
    )
    empty_diff = (
        "--- /dev/null\n"
        "+++ b/notes/blank.txt\n"
        "@@ -0,0 +0,0 @@\n"
    )
    assert payload["patches"][0]["content"] == "你好"
    assert payload["patches"][0]["diff"] == unicode_diff
    assert payload["patches"][1]["diff"] == newline_diff
    assert payload["patches"][2]["diff"] == empty_diff
    assert payload["patches"][2]["content"] == ""
    text = format_patch(payload)
    assert text == (
        "session s\npatches 3\n\n" + "\n".join(item["diff"] for item in payload["patches"])
    )


def test_session_patch_skips_failed_unpaired_and_incomplete_calls(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    events = [
        {"session_id": "s", "step": 1, "kind": "thought", "content": "edit"},
        _call(1, "edit_file", "c-fail", {"path": "math_utils.py", "old": "a", "new": "b"}),
        _ok(1, "c-fail", ok=False),
        _call(1, "write_file", "c-unpaired", {"path": "notes/out.txt", "content": "nope"}),
        _call(2, "edit_file", "c-missing", {"path": "math_utils.py"}),
        _ok(2, "c-missing"),
        _call(2, "write_file", "c-bad-content", {"path": "notes/out.txt", "content": None}),
        _ok(2, "c-bad-content"),
        _call(2, "write_file", "c-empty-path", {"path": "", "content": "x"}),
        _ok(2, "c-empty-path"),
        _call(2, "edit_file", "c-num-ok", {"path": "num.py", "old": "a", "new": "b"}),
        {"step": 2, "kind": "tool_result", "call_id": "c-num-ok", "ok": 1},
        _call(2, "write_file", "c-no-args", {}),
        _ok(2, "c-no-args"),
        _call(3, "search", "c-search", {"path": "math_utils.py"}),
        _ok(3, "c-search"),
        _call(
            4,
            "edit_file",
            "c-first",
            {"path": "math_utils.py", "old": "return a - b\n", "new": "return a + b\n"},
        ),
        _ok(4, "c-first"),
        _call(6, "lint_python", "c-lint", {"path": "math_utils.py"}),
        _ok(6, "c-lint"),
        _call(7, "write_file", "c-later", {"path": "notes/out.txt", "content": "later"}),
    ]
    changes = session_changes(events)
    payload = session_patch(events)
    assert payload["session_id"] == changes["session_id"] == "s"
    assert [item["call_id"] for item in payload["patches"]] == ["c-first"]
    assert [item["call_id"] for item in changes["changes"]] == ["c-first"]
    for key, value in changes["changes"][0].items():
        assert payload["patches"][0][key] == value
    text = format_patch(payload)
    for omitted in ("c-fail", "c-unpaired", "c-missing", "c-later", "nope", "later", "search"):
        assert omitted not in text
    assert list(tmp_path.iterdir()) == []
    assert not (tmp_path / "math_utils.py").exists()
    assert not (tmp_path / "notes").exists()


def test_session_patch_replace_all_is_annotated_once() -> None:
    events = [
        {"session_id": "s"},
        _call(
            2,
            "edit_file",
            "c-one",
            {"path": "a.py", "old": "token\n", "new": "other\n", "replace_all": False},
        ),
        _ok(2, "c-one"),
        _call(
            3,
            "edit_file",
            "c-all",
            {"path": "b.py", "old": "token\n", "new": "other\n", "replace_all": True},
        ),
        _ok(3, "c-all"),
    ]
    payload = session_patch(events)
    assert payload["patches"][0]["replace_all"] is False
    assert payload["patches"][1]["replace_all"] is True
    assert "# replace_all" not in payload["patches"][0]["diff"]
    flagged = payload["patches"][1]["diff"]
    assert flagged.startswith("# replace_all=true\n--- a/b.py\n+++ b/b.py\n")
    assert flagged.splitlines().count("-token") == 1
    assert flagged.splitlines().count("+other") == 1
    text = format_patch(payload)
    assert text.count("# replace_all=true") == 1
    assert "# replace_all=false" not in text
    assert "# replace_all=true\n--- a/b.py\n" in text
    assert text == (
        "session s\npatches 2\n\n" + "\n".join(item["diff"] for item in payload["patches"])
    )


def test_session_patch_marks_missing_newlines() -> None:
    events = [
        {"session_id": "s"},
        _call(1, "edit_file", "c-old", {"path": "old.py", "old": "print(1)", "new": "print(2)\n"}),
        _ok(1, "c-old"),
        _call(2, "edit_file", "c-new", {"path": "new.py", "old": "print(1)\n", "new": "print(2)"}),
        _ok(2, "c-new"),
        _call(3, "edit_file", "c-both", {"path": "a.py", "old": "print(1)", "new": "print(2)"}),
        _ok(3, "c-both"),
        _call(
            4,
            "edit_file",
            "c-multi",
            {"path": "snip.py", "old": "line one\nline two", "new": "\nline three"},
        ),
        _ok(4, "c-multi"),
    ]
    diffs = [item["diff"] for item in session_patch(events)["patches"]]
    assert diffs[0] == (
        "--- a/old.py\n"
        "+++ b/old.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-print(1)\n"
        "\\ No newline at end of file\n"
        "+print(2)\n"
    )
    assert diffs[1] == (
        "--- a/new.py\n"
        "+++ b/new.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-print(1)\n"
        "+print(2)\n"
        "\\ No newline at end of file\n"
    )
    assert diffs[2] == (
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-print(1)\n"
        "\\ No newline at end of file\n"
        "+print(2)\n"
        "\\ No newline at end of file\n"
    )
    assert diffs[3] == (
        "--- a/snip.py\n"
        "+++ b/snip.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-line one\n"
        "-line two\n"
        "\\ No newline at end of file\n"
        "+\n"
        "+line three\n"
        "\\ No newline at end of file\n"
    )


def test_session_patch_empty_events() -> None:
    assert session_patch([]) == {"session_id": "", "patches": []}
    assert format_patch(session_patch([])) == "session (unknown)\npatches 0\n"
    assert format_patch({"session_id": "s", "patches": []}) == "session s\npatches 0\n"
    fallback = session_patch(
        [{"kind": "final", "step": 1, "content": "done"}],
        session_id="from-stem",
    )
    assert fallback == {"session_id": "from-stem", "patches": []}
    assert format_patch(fallback) == "session from-stem\npatches 0\n"


def test_html_escapes_markup(tmp_path: Path) -> None:
    log = SessionLog.create(tmp_path)
    log.append(kind="user", step=0, content="<script>alert(1)</script>")
    document = render_html(load_events(log.path))
    assert "<script>" not in document
    assert "&lt;script&gt;" in document
    assert "Tools were not re-run" in document


def test_session_prefix_and_ambiguous_id(tmp_path: Path) -> None:
    first = tmp_path / "20260101T000000Z-aaa111.jsonl"
    second = tmp_path / "20260101T000000Z-aaa222.jsonl"
    payload = {
        "ts": "2026-01-01T00:00:00.000Z",
        "seq": 1,
        "step": 0,
        "kind": "session_start",
        "task": "one",
    }
    first.write_text(json.dumps({**payload, "session_id": first.stem}) + "\n", encoding="utf-8")
    second.write_text(json.dumps({**payload, "session_id": second.stem, "task": "two"}) + "\n", encoding="utf-8")
    assert resolve_session_file(tmp_path, "20260101T000000Z-aaa111") == first
    from dm_agent.errors import AgentError
    import pytest

    with pytest.raises(AgentError, match="ambiguous"):
        resolve_session_file(tmp_path, "20260101T000000Z-aaa")
    with pytest.raises(AgentError, match="not found"):
        resolve_session_file(tmp_path, "missing")
