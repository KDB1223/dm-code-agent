"""Workspace tools, including the jail around them."""

import time
from pathlib import Path

from dm_agent.limits import clip_text
from dm_agent.tools import build_registry
from dm_agent.tools.run_pytest import parse_pytest_counts
from dm_agent.tools.shell import execute_shell


def test_read_write_edit_round_trip(tmp_path: Path) -> None:
    registry = build_registry(tmp_path, shell_timeout=5)
    written = registry.invoke(
        "write_file",
        {"path": "pkg/app.py", "content": "def add(a, b):\n    return a - b\n"},
    )
    assert written.ok
    assert (tmp_path / "pkg" / "app.py").is_file()

    read = registry.invoke("read_file", {"path": "pkg/app.py", "offset": 2, "limit": 1})
    assert read.ok
    assert "return a - b" in read.output

    ambiguous = registry.invoke(
        "edit_file",
        {"path": "pkg/app.py", "old": "a", "new": "b"},
    )
    assert not ambiguous.ok
    assert "matched" in ambiguous.output

    edited = registry.invoke(
        "edit_file",
        {"path": "pkg/app.py", "old": "return a - b", "new": "return a + b"},
    )
    assert edited.ok
    assert "return a + b" in (tmp_path / "pkg" / "app.py").read_text(encoding="utf-8")

    replaced = registry.invoke(
        "write_file",
        {"path": "pkg/app.py", "content": "x x"},
    )
    assert replaced.ok
    all_done = registry.invoke(
        "edit_file",
        {"path": "pkg/app.py", "old": "x", "new": "y", "replace_all": True},
    )
    assert all_done.ok
    assert (tmp_path / "pkg" / "app.py").read_text(encoding="utf-8") == "y y"


def test_tools_refuse_paths_outside_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    registry = build_registry(workspace, shell_timeout=5)
    for name, arguments in (
        ("read_file", {"path": "../outside.txt"}),
        ("write_file", {"path": "../outside.txt", "content": "nope"}),
        ("edit_file", {"path": "/etc/passwd", "old": "a", "new": "b"}),
        ("list_dir", {"path": ".."}),
        ("search", {"pattern": "root", "path": "/etc"}),
        ("lint_python", {"path": "../outside.txt"}),
    ):
        result = registry.invoke(name, arguments)
        assert result.ok is False, name
        assert "Refused" in result.output
    assert not outside.exists()


def test_symlink_to_outside_is_not_read(tmp_path: Path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("secret-value", encoding="utf-8")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "link").symlink_to(outside)
    registry = build_registry(workspace, shell_timeout=5)
    result = registry.invoke("read_file", {"path": "link"})
    assert result.ok is False
    assert "secret-value" not in result.output


def test_list_dir_and_search(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("NEEDLE_OK = 1\n", encoding="utf-8")
    git = tmp_path / ".git"
    git.mkdir()
    (git / "hidden.py").write_text("NEEDLE_GIT = 1\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("NEEDLE_OUT", encoding="utf-8")
    (tmp_path / "escape").symlink_to(outside)

    registry = build_registry(tmp_path, shell_timeout=5)
    listing = registry.invoke("list_dir", {"path": "."})
    assert listing.ok
    assert "src/" in listing.output
    assert "escape" in listing.output
    assert "outside" in listing.output

    found = registry.invoke("search", {"pattern": "NEEDLE", "glob": "*.py"})
    assert found.ok
    assert "NEEDLE_OK" in found.output
    assert "NEEDLE_GIT" not in found.output

    literal = registry.invoke("search", {"pattern": "a.b", "literal": True, "path": "src"})
    assert literal.ok
    assert "No matches" in literal.output


def test_shell_runs_inside_workspace_and_refuses_escape(tmp_path: Path) -> None:
    registry = build_registry(tmp_path, shell_timeout=5)
    ran = registry.invoke("run_shell", {"command": "printf 'hello-agent\\n'"})
    assert ran.ok
    assert "hello-agent" in ran.output

    home = registry.invoke("run_shell", {"command": "printf %s \"$HOME\""})
    assert home.ok
    assert str(tmp_path.resolve()) in home.output

    (tmp_path / "sub").mkdir()
    nested = registry.invoke("run_shell", {"command": "cd sub && pwd"})
    assert nested.ok
    assert str((tmp_path / "sub").resolve()) in nested.output

    refused = registry.invoke("run_shell", {"command": "cat /etc/passwd"})
    assert not refused.ok
    assert "Refused" in refused.output
    assert "root:" not in refused.output

    target = tmp_path / "note.txt"
    target.write_text("inside", encoding="utf-8")
    allowed = registry.invoke("run_shell", {"command": f"cat {target}"})
    assert allowed.ok
    assert "inside" in allowed.output

    escape = tmp_path.parent / "dm-agent-should-not-write.txt"
    if escape.exists():
        escape.unlink()
    blocked = registry.invoke("run_shell", {"command": f"printf x > {escape}"})
    assert not blocked.ok
    assert not escape.exists()


def test_shell_timeout_and_truncation(tmp_path: Path) -> None:
    started = time.monotonic()
    timed = execute_shell(
        "sleep 5",
        cwd=str(tmp_path),
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        timeout=0.4,
        output_limit=1000,
    )
    elapsed = time.monotonic() - started
    assert timed.status == "timeout"
    assert elapsed < 3

    flooded = execute_shell(
        "python3 -c 'print(\"z\" * 100000)'",
        cwd=str(tmp_path),
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        timeout=5,
        output_limit=1000,
    )
    assert flooded.status == "truncated"
    assert "z" in flooded.output


def test_lint_python_reports_and_clears(tmp_path: Path) -> None:
    registry = build_registry(tmp_path, shell_timeout=5)
    (tmp_path / "app.py").write_text(
        "def f():\n"
        "    try:\n"
        "        return 1\n"
        "    except:\n"
        "        breakpoint()\n"
        "        return 0\n",
        encoding="utf-8",
    )
    (tmp_path / "bad.py").write_text("def f(:\n", encoding="utf-8")
    dirty = registry.invoke("lint_python", {"path": "app.py"})
    assert not dirty.ok
    assert "status: fail" in dirty.output
    assert "bare-except" in dirty.output
    assert "debug-call" in dirty.output

    syntax = registry.invoke("lint_python", {"path": "bad.py"})
    assert not syntax.ok
    assert "syntax-error" in syntax.output

    (tmp_path / "ok.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    clean = registry.invoke("lint_python", {"path": "ok.py"})
    assert clean.ok
    assert "status: ok" in clean.output
    assert "issues: 0" in clean.output


def test_lint_skips_virtualenv_and_flags_pdb(tmp_path: Path) -> None:
    vendored = tmp_path / ".venv" / "lib"
    vendored.mkdir(parents=True)
    (vendored / "skip.py").write_text("def f():\n    except:\n        pass\n", encoding="utf-8")
    (tmp_path / "app.py").write_text(
        "import pdb\n\n\ndef f():\n    pdb.set_trace()\n    return 1\n",
        encoding="utf-8",
    )
    registry = build_registry(tmp_path, shell_timeout=5)
    report = registry.invoke("lint_python", {"path": "."})
    assert not report.ok
    assert "debug-call" in report.output
    assert "skip.py" not in report.output


def test_run_pytest_pass_fail_and_keyword(tmp_path: Path) -> None:
    (tmp_path / "test_ok.py").write_text(
        "def test_one():\n    assert 1 + 1 == 2\n\ndef test_two():\n    assert False\n",
        encoding="utf-8",
    )
    registry = build_registry(tmp_path, shell_timeout=30)
    selected = registry.invoke("run_pytest", {"path": "test_ok.py", "keyword": "test_one"})
    assert selected.ok, selected.output
    assert "status: passed" in selected.output
    assert "passed: 1" in selected.output

    failed = registry.invoke("run_pytest", {"path": "test_ok.py"})
    assert not failed.ok
    assert "status: failed" in failed.output
    assert "failed: 1" in failed.output


def test_parse_pytest_counts_from_summary() -> None:
    text = "some traceback line with 1 error mentioned\n====\n1 failed, 2 passed, 1 error in 0.02s\n"
    assert parse_pytest_counts(text) == (2, 1, 1)


def test_unknown_tool_and_clip() -> None:
    registry = build_registry(Path("."), shell_timeout=5)
    missing = registry.invoke("not_a_tool", {})
    assert not missing.ok
    assert "Unknown tool" in missing.output
    clipped = clip_text("x" * 50, limit=10)
    assert clipped.startswith("x" * 10)
    assert "truncated" in clipped


def test_tool_schemas_include_the_course_tools(tmp_path: Path) -> None:
    names = [
        item["function"]["name"]
        for item in build_registry(tmp_path, shell_timeout=5).schemas()
    ]
    assert names == [
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "search",
        "run_shell",
        "lint_python",
        "run_pytest",
    ]
