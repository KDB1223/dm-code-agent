"""Command-line interface for dm-agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dm_agent import __version__
from dm_agent.agent import Agent
from dm_agent.config import load_config
from dm_agent.errors import AgentError, ConfigError, LLMError
from dm_agent.llm import OpenAICompatibleLLM, public_base_url
from dm_agent.present import format_inspection, format_transcript, render_html
from dm_agent.session import (
    iter_session_files,
    load_events,
    resolve_session_file,
    summarize_session,
)
from dm_agent.tools import build_registry


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run a command. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    try:
        return handler(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (AgentError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dm-agent",
        description="Local-first coding agent with an append-only session log.",
    )
    parser.add_argument("--version", action="version", version=f"dm-agent {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--workspace",
        help="Workspace root the agent may edit. Default: current directory, or DM_AGENT_WORKSPACE.",
    )
    common.add_argument("--config", help="Path to a TOML config file.")
    common.add_argument(
        "--log-dir",
        help="Directory for JSONL session logs. Default: <workspace>/.dm-agent/sessions.",
    )

    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", parents=[common], help="Run a task against the workspace.")
    run.add_argument("task", help='Task text. Use "-" to read the task from stdin.')
    run.add_argument("--max-steps", type=int, help="Maximum model turns. Default 20.")
    run.add_argument("--model", help="Model name. Overrides OPENAI_MODEL and the config file.")
    run.set_defaults(handler=cmd_run)

    sessions = sub.add_parser("sessions", help="Inspect append-only session logs.")
    session_sub = sessions.add_subparsers(dest="sessions_command")

    listing = session_sub.add_parser(
        "list", parents=[common], help="List sessions in the workspace log directory."
    )
    listing.add_argument("--json", action="store_true", help="Print summaries as JSON.")
    listing.set_defaults(handler=cmd_list)

    show = session_sub.add_parser(
        "show", parents=[common], help="Show every event in a session."
    )
    show.add_argument("session_id", help="Session id or a unique prefix.")
    show.add_argument("--json", action="store_true", help="Print the raw JSONL file.")
    show.set_defaults(handler=cmd_show)

    replay = session_sub.add_parser(
        "replay",
        parents=[common],
        help="Dry-replay a session transcript. Does not run tools or call the model.",
    )
    replay.add_argument("session_id", help="Session id or a unique prefix.")
    replay.add_argument("--json", action="store_true", help="Print events as a JSON array.")
    replay.set_defaults(handler=cmd_replay)

    html = session_sub.add_parser(
        "html", parents=[common], help="Write a self-contained HTML view of a session."
    )
    html.add_argument("session_id", help="Session id or a unique prefix.")
    html.add_argument("--output", help="Output path. Default: next to the JSONL file.")
    html.set_defaults(handler=cmd_html)

    sessions.set_defaults(handler=cmd_sessions_help)
    return parser


def cmd_sessions_help(_args: argparse.Namespace) -> int:
    build_parser().parse_args(["sessions", "--help"])
    return 2


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(
        workspace=args.workspace,
        config_path=args.config,
        max_steps=args.max_steps,
        model=args.model,
        log_dir=args.log_dir,
    )
    if not cfg.workspace.is_dir():
        raise ConfigError(f"workspace is not a directory: {cfg.workspace}")
    if not cfg.api_key:
        raise ConfigError(
            "OPENAI_API_KEY is not set. Export it, or set llm.api_key in the config file."
        )
    task = _read_task(args.task)
    if task.strip() == "":
        raise ConfigError("task is empty")

    from dm_agent.session import SessionLog

    session = SessionLog.create(cfg.log_dir)
    llm = OpenAICompatibleLLM(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        model=cfg.model,
        temperature=cfg.temperature,
        timeout=cfg.request_timeout_seconds,
    )
    agent = Agent(
        llm=llm,
        tools=build_registry(cfg.workspace, shell_timeout=cfg.shell_timeout_seconds),
        session=session,
        max_steps=cfg.max_steps,
        metadata={
            "model": cfg.model,
            "base_url": public_base_url(cfg.base_url),
            "workspace": str(cfg.workspace),
            "max_steps": cfg.max_steps,
        },
        on_event=_progress,
    )
    result = agent.run(task)
    print(result.answer)
    print(f"session: {result.session_id}", file=sys.stderr)
    print(f"log: {result.session_path}", file=sys.stderr)
    return 0 if result.finished else 3


def cmd_list(args: argparse.Namespace) -> int:
    cfg = _session_config(args)
    files = iter_session_files(cfg.log_dir)
    rows = [summarize_session(path) for path in files]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    if not rows:
        print(f"no sessions in {cfg.log_dir}")
        return 0
    print(f"{'SESSION':<26} {'STATUS':<10} {'EVENTS':>6}  TASK")
    for row in rows:
        task = _one_line(str(row.get("task") or ""))
        print(
            f"{row['session_id']:<26} {row['status']:<10} {row['events']:>6}  {task}"
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    path = _session_path(args)
    if args.json:
        sys.stdout.write(path.read_text(encoding="utf-8"))
        return 0
    events = load_events(path)
    print(f"file {path}")
    sys.stdout.write(format_inspection(events))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Reprint the transcript from the log. Tools are not executed."""
    path = _session_path(args)
    events = load_events(path)
    if args.json:
        print(json.dumps(events, indent=2, ensure_ascii=False))
        return 0
    print(f"file {path}")
    sys.stdout.write(format_transcript(events))
    return 0


def cmd_html(args: argparse.Namespace) -> int:
    path = _session_path(args)
    document = render_html(load_events(path))
    if args.output:
        out = Path(args.output).expanduser()
    else:
        out = path.with_suffix(".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    print(out.resolve())
    return 0


def _session_config(args: argparse.Namespace):
    return load_config(
        workspace=getattr(args, "workspace", None),
        config_path=getattr(args, "config", None),
        log_dir=getattr(args, "log_dir", None),
    )


def _session_path(args: argparse.Namespace) -> Path:
    cfg = _session_config(args)
    token = args.session_id
    if token.endswith(".jsonl"):
        token = token[: -len(".jsonl")]
    return resolve_session_file(cfg.log_dir, token)


def _read_task(task: str) -> str:
    if task == "-":
        return sys.stdin.read()
    return task


def _progress(event: dict) -> None:
    kind = event.get("kind")
    step = event.get("step")
    if kind == "session_start":
        print(f"session {event.get('session_id')}", file=sys.stderr)
    elif kind == "thought":
        lines = str(event.get("content") or "").strip().splitlines()
        preview = lines[0][:160] if lines else ""
        print(f"[step {step}] plan: {preview}", file=sys.stderr)
    elif kind == "tool_call":
        payload = json.dumps(event.get("arguments", {}), ensure_ascii=False)
        if len(payload) > 180:
            payload = payload[:180] + "..."
        print(f"[step {step}] {event.get('tool')} {payload}", file=sys.stderr)
    elif kind == "tool_result":
        mark = "ok" if event.get("ok") else "error"
        print(f"[step {step}] {event.get('tool')} -> {mark}", file=sys.stderr)
    elif kind in {"final", "max_steps", "error"}:
        print(f"[step {step}] {kind}", file=sys.stderr)


def _one_line(text: str, limit: int = 60) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"
