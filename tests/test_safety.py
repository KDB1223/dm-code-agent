"""Path jail and static shell checks."""

from pathlib import Path

import pytest

from dm_agent.errors import PathJailError
from dm_agent.safety import check_shell_command, resolve_in_workspace


def test_relative_escape_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PathJailError, match="escapes"):
        resolve_in_workspace(tmp_path, "../secret.txt")


def test_absolute_escape_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PathJailError, match="escapes"):
        resolve_in_workspace(tmp_path, "/etc/passwd")


def test_inside_absolute_path_is_allowed(tmp_path: Path) -> None:
    target = tmp_path / "notes" / "a.txt"
    target.parent.mkdir()
    target.write_text("ok", encoding="utf-8")
    assert resolve_in_workspace(tmp_path, str(target)) == target.resolve()


def test_symlink_escape_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    link = workspace / "link"
    link.symlink_to(outside)
    with pytest.raises(PathJailError, match="escapes"):
        resolve_in_workspace(workspace, "link")


def test_nul_and_newline_paths_are_refused(tmp_path: Path) -> None:
    with pytest.raises(PathJailError):
        resolve_in_workspace(tmp_path, "bad\x00name")
    with pytest.raises(PathJailError):
        resolve_in_workspace(tmp_path, "bad\nname")


def test_shell_refuses_outside_absolute_and_dotdot(tmp_path: Path) -> None:
    with pytest.raises(PathJailError, match="/etc/passwd"):
        check_shell_command("cat /etc/passwd", tmp_path)
    with pytest.raises(PathJailError):
        check_shell_command("cat ../secret", tmp_path)
    with pytest.raises(PathJailError):
        check_shell_command("echo hi > /tmp/dm-agent-escape", tmp_path)
    with pytest.raises(PathJailError, match="tilde"):
        check_shell_command("cat ~root/.ssh/id_rsa", tmp_path)
    with pytest.raises(PathJailError, match="escapes"):
        check_shell_command("printf %s ~/../../etc/passwd", tmp_path)


def test_shell_allows_workspace_paths_and_urls(tmp_path: Path) -> None:
    note = tmp_path / "note.txt"
    note.write_text("hi", encoding="utf-8")
    check_shell_command(f"cat {note}", tmp_path)
    check_shell_command("cat sub/../note.txt", tmp_path)
    check_shell_command("printf %s ~/note.txt", tmp_path)
    check_shell_command("curl https://example.com/a", tmp_path)
    with pytest.raises(PathJailError):
        check_shell_command("curl https://example.com -o /tmp/x", tmp_path)
    with pytest.raises(PathJailError):
        check_shell_command("python3 -c 'print(open(\"/etc/passwd\").read())'", tmp_path)


def test_file_url_is_treated_as_a_path(tmp_path: Path) -> None:
    with pytest.raises(PathJailError):
        check_shell_command("python3 -c 'open(\"file:///etc/passwd\")'", tmp_path)
