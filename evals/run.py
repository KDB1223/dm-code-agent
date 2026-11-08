"""Score the offline tasks under ``evals/tasks``.

Each task copies a tiny workspace into a temp directory, runs the real ReAct
loop against :class:`evals.harness.ScriptedLLM`, checks the JSONL session for
the expected successful tool results, then runs ``check.py`` as a golden
check. Nothing here calls a remote model.

From the repository root::

    python -m evals.run
    python -m evals.run --transcript fix-add
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parent.parent
    for _entry in (_ROOT, _ROOT / "src"):
        _text = str(_entry)
        if _text not in sys.path:
            sys.path.insert(0, _text)

from dm_agent.agent import Agent
from dm_agent.errors import LLMError
from dm_agent.present import format_transcript
from dm_agent.session import SessionLog, load_events
from dm_agent.tools import build_registry
from evals.harness import ScriptedLLM

_REQUIRED_KINDS = ("user", "tool_call", "tool_result", "final")


@dataclass
class TaskResult:
    """One task's score, plus the dry transcript when a session was written."""

    task_id: str
    ok: bool
    detail: str
    transcript: str


def main(argv: list[str] | None = None) -> int:
    """Run every task, or one task when ``--transcript`` is set."""
    parser = argparse.ArgumentParser(
        prog="python -m evals.run",
        description="Offline golden-check evals for dm-agent. No API key required.",
    )
    parser.add_argument(
        "--transcript",
        metavar="TASK_ID",
        help="Run a single task and print its dry session transcript.",
    )
    args = parser.parse_args(argv)
    tasks = discover_tasks(Path(__file__).resolve().parent / "tasks")
    if not tasks:
        print("no tasks found under evals/tasks", file=sys.stderr)
        return 2
    if args.transcript:
        chosen = [path for path in tasks if _task_id(path) == args.transcript]
        if len(chosen) != 1:
            known = ", ".join(_task_id(path) for path in tasks)
            print(f"unknown task {args.transcript!r}. known: {known}", file=sys.stderr)
            return 2
        result = run_task(chosen[0])
        sys.stdout.write(result.transcript)
        sys.stdout.flush()
        mark = "PASS" if result.ok else "FAIL"
        print(f"{result.task_id} {mark} {result.detail}", file=sys.stderr)
        return 0 if result.ok else 1

    results = [run_task(path) for path in tasks]
    sys.stdout.write(format_summary(results))
    sys.stdout.flush()
    for item in results:
        if not item.ok:
            print(f"{item.task_id}: {item.detail}", file=sys.stderr)
    return 0 if all(item.ok for item in results) else 1


def discover_tasks(tasks_dir: Path) -> list[Path]:
    """Return task directories that contain ``task.json``, in name order."""
    if not tasks_dir.is_dir():
        return []
    found = [path for path in tasks_dir.iterdir() if (path / "task.json").is_file()]
    return sorted(found, key=lambda path: path.name)


def format_summary(results: list[TaskResult]) -> str:
    """Stable scoreboard. Stdout of a full run is exactly this string."""
    lines = [f"{item.task_id:<14} {'PASS' if item.ok else 'FAIL'}" for item in results]
    passed = sum(1 for item in results if item.ok)
    lines.append(f"score {passed}/{len(results)}")
    return "\n".join(lines) + "\n"


def run_task(task_dir: Path) -> TaskResult:
    """Copy the fixture, run the agent, audit the log, then run the golden check."""
    spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    task_id = str(spec["id"])
    workspace_src = task_dir / "workspace"
    check_script = task_dir / "check.py"
    if not workspace_src.is_dir() or not check_script.is_file():
        return TaskResult(task_id, False, "task is missing workspace/ or check.py", "")

    with tempfile.TemporaryDirectory(prefix=f"dm-eval-{task_id}-") as tmp:
        workspace = Path(tmp) / "workspace"
        shutil.copytree(workspace_src, workspace)
        session = SessionLog.create(workspace / ".dm-agent" / "sessions")
        agent = Agent(
            llm=ScriptedLLM(list(spec["turns"])),
            tools=build_registry(workspace, shell_timeout=60),
            session=session,
            max_steps=int(spec.get("max_steps", 8)),
            metadata={
                "model": "scripted-harness",
                "base_url": "harness://offline",
                "workspace": str(workspace),
                "max_steps": int(spec.get("max_steps", 8)),
            },
        )
        try:
            result = agent.run(str(spec["prompt"]))
        except LLMError as exc:
            transcript = _safe_transcript(session.path)
            return TaskResult(task_id, False, str(exc), transcript)
        transcript = format_transcript(load_events(session.path))
        problem = _audit_log(session.path, list(spec.get("expect_tools") or []))
        if problem:
            return TaskResult(task_id, False, problem, transcript)
        if not result.finished:
            return TaskResult(task_id, False, "agent stopped without a final answer", transcript)
        ok, detail = _run_check(workspace, check_script)
        if not ok:
            return TaskResult(task_id, False, detail, transcript)
        return TaskResult(task_id, True, "golden ok", transcript)


def _task_id(task_dir: Path) -> str:
    spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    return str(spec["id"])


def _audit_log(path: Path, expect_tools: list[str]) -> str:
    events = load_events(path)
    kinds = {event.get("kind") for event in events}
    for kind in _REQUIRED_KINDS:
        if kind not in kinds:
            return f"session log is missing a {kind} event"
    for name in expect_tools:
        success = any(
            event.get("kind") == "tool_result" and event.get("tool") == name and event.get("ok") is True
            for event in events
        )
        if not success:
            return f"session log has no successful {name} result"
    return ""


def _run_check(workspace: Path, check_script: Path) -> tuple[bool, str]:
    env = dict(os.environ)
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(workspace) if not previous else str(workspace) + os.pathsep + previous
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(check_script)],
        cwd=str(workspace),
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
        check=False,
    )
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode == 0:
        return True, output or "golden ok"
    tail = "\n".join(output.splitlines()[-40:])
    return False, tail or f"check exited {completed.returncode}"


def _safe_transcript(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return format_transcript(load_events(path))
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
