"""TOML configuration with environment and CLI overrides.

Precedence, from lowest to highest:

1. Built-in defaults
2. A TOML file (``--config``, else ``<workspace>/dm-agent.toml``,
   else ``./dm-agent.toml``, else ``$XDG_CONFIG_HOME/dm-agent/config.toml``)
3. Environment variables
4. Command-line flags

The API key is read from ``OPENAI_API_KEY`` unless the file sets
``llm.api_key``. The environment always wins over the file so a shell export
cannot be silently ignored. The key is never written to the session log.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dm_agent.errors import ConfigError

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MAX_STEPS = 20
DEFAULT_SHELL_TIMEOUT = 30.0
DEFAULT_REQUEST_TIMEOUT = 120.0
DEFAULT_TEMPERATURE = 0.2
DEFAULT_LOG_DIR = ".dm-agent/sessions"


@dataclass(frozen=True)
class Config:
    """Resolved settings for one invocation."""

    workspace: Path
    log_dir: Path
    max_steps: int
    shell_timeout_seconds: float
    provider: str
    model: str
    base_url: str
    api_key: str
    temperature: float
    request_timeout_seconds: float
    config_path: Path | None


def load_config(
    *,
    workspace: str | None = None,
    config_path: str | None = None,
    max_steps: int | None = None,
    model: str | None = None,
    log_dir: str | None = None,
) -> Config:
    """Load settings for a CLI invocation.

    String arguments come from flags. ``None`` means "not passed".
    """
    file_path = _discover_config_file(workspace, config_path)
    data: dict = {}
    if file_path is not None:
        data = _read_toml(file_path)

    agent = _table(data, "agent")
    llm = _table(data, "llm")

    workspace_path = _resolve_workspace(workspace, agent)
    resolved_log_dir = _resolve_log_dir(log_dir, agent, workspace_path)
    resolved_steps = _resolve_max_steps(max_steps, agent)
    provider = _resolve_provider(llm)
    resolved_model = _resolve_model(model, llm)
    base_url = _resolve_base_url(llm)
    api_key = _resolve_api_key(llm)
    temperature = _resolve_temperature(llm)
    request_timeout = _resolve_request_timeout(llm)
    shell_timeout = _resolve_shell_timeout(agent)

    return Config(
        workspace=workspace_path,
        log_dir=resolved_log_dir,
        max_steps=resolved_steps,
        shell_timeout_seconds=shell_timeout,
        provider=provider,
        model=resolved_model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        request_timeout_seconds=request_timeout,
        config_path=file_path,
    )


def _discover_config_file(workspace: str | None, config_path: str | None) -> Path | None:
    if config_path:
        path = Path(config_path).expanduser()
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        return path.resolve()

    candidates: list[Path] = []
    if workspace:
        candidates.append(Path(workspace).expanduser() / "dm-agent.toml")
    elif os.environ.get("DM_AGENT_WORKSPACE"):
        candidates.append(Path(os.environ["DM_AGENT_WORKSPACE"]).expanduser() / "dm-agent.toml")
    candidates.append(Path.cwd() / "dm-agent.toml")
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        candidates.append(Path(xdg) / "dm-agent" / "config.toml")
    else:
        candidates.append(Path.home() / ".config" / "dm-agent" / "config.toml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _read_toml(path: Path) -> dict:
    try:
        with path.open("rb") as handle:
            loaded = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"invalid TOML in {path}: expected a table")
    return loaded


def _table(data: dict, name: str) -> dict:
    raw = data.get(name, {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"[{name}] must be a table")
    return raw


def _resolve_workspace(cli_value: str | None, agent: dict) -> Path:
    if cli_value:
        chosen = cli_value
    elif os.environ.get("DM_AGENT_WORKSPACE"):
        chosen = os.environ["DM_AGENT_WORKSPACE"]
    elif "workspace" in agent:
        chosen = _as_str(agent, "workspace", "agent")
    else:
        chosen = "."
    path = Path(chosen).expanduser().resolve()
    return path


def _resolve_log_dir(cli_value: str | None, agent: dict, workspace: Path) -> Path:
    if cli_value:
        raw = cli_value
    elif os.environ.get("DM_AGENT_LOG_DIR"):
        raw = os.environ["DM_AGENT_LOG_DIR"]
    elif "log_dir" in agent:
        raw = _as_str(agent, "log_dir", "agent")
    else:
        raw = DEFAULT_LOG_DIR
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = workspace / path
    return path.resolve()


def _resolve_max_steps(cli_value: int | None, agent: dict) -> int:
    if cli_value is not None:
        value = cli_value
    elif os.environ.get("DM_AGENT_MAX_STEPS"):
        value = _parse_int(os.environ["DM_AGENT_MAX_STEPS"], "DM_AGENT_MAX_STEPS")
    elif "max_steps" in agent:
        value = _as_int(agent, "max_steps", "agent")
    else:
        value = DEFAULT_MAX_STEPS
    if value < 1:
        raise ConfigError("max_steps must be >= 1")
    return value


def _resolve_provider(llm: dict) -> str:
    provider = str(llm.get("provider", "openai")).strip().lower()
    if provider != "openai":
        raise ConfigError(
            f"unsupported llm.provider {provider!r}; use 'openai' for any "
            "OpenAI-compatible HTTP endpoint"
        )
    return provider


def _resolve_model(cli_value: str | None, llm: dict) -> str:
    if cli_value:
        return cli_value
    env = os.environ.get("OPENAI_MODEL")
    if env:
        return env
    if "model" in llm:
        return _as_str(llm, "model", "llm")
    return DEFAULT_MODEL


def _resolve_base_url(llm: dict) -> str:
    env = os.environ.get("OPENAI_BASE_URL")
    if env:
        return env.rstrip("/")
    if "base_url" in llm:
        return _as_str(llm, "base_url", "llm").rstrip("/")
    return DEFAULT_BASE_URL


def _resolve_api_key(llm: dict) -> str:
    env = os.environ.get("OPENAI_API_KEY")
    if env:
        return env
    if "api_key" in llm:
        return _as_str(llm, "api_key", "llm")
    return ""


def _resolve_temperature(llm: dict) -> float:
    if "temperature" not in llm:
        return DEFAULT_TEMPERATURE
    value = llm["temperature"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("llm.temperature must be a number")
    temp = float(value)
    if not 0.0 <= temp <= 2.0:
        raise ConfigError("llm.temperature must be between 0 and 2")
    return temp


def _resolve_request_timeout(llm: dict) -> float:
    if "timeout_seconds" not in llm:
        return DEFAULT_REQUEST_TIMEOUT
    value = _as_number(llm, "timeout_seconds", "llm")
    if value <= 0:
        raise ConfigError("llm.timeout_seconds must be > 0")
    return value


def _resolve_shell_timeout(agent: dict) -> float:
    env = os.environ.get("DM_AGENT_SHELL_TIMEOUT")
    if env:
        value = _parse_float(env, "DM_AGENT_SHELL_TIMEOUT")
    elif "shell_timeout_seconds" in agent:
        value = _as_number(agent, "shell_timeout_seconds", "agent")
    else:
        value = DEFAULT_SHELL_TIMEOUT
    if value <= 0:
        raise ConfigError("shell_timeout_seconds must be > 0")
    return value


def _as_str(table: dict, key: str, where: str) -> str:
    value = table[key]
    if not isinstance(value, str):
        raise ConfigError(f"{where}.{key} must be a string")
    return value


def _as_int(table: dict, key: str, where: str) -> int:
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}.{key} must be an integer")
    return value


def _as_number(table: dict, key: str, where: str) -> float:
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}.{key} must be a number")
    return float(value)


def _parse_int(raw: str, label: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{label} must be an integer") from exc


def _parse_float(raw: str, label: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{label} must be a number") from exc
