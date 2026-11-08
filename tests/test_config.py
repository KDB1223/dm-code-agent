"""Config file, environment, and CLI precedence."""

from pathlib import Path

import pytest

from dm_agent.config import load_config
from dm_agent.errors import ConfigError


@pytest.fixture(autouse=True)
def clean_config_env(monkeypatch, tmp_path: Path):
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
    monkeypatch.chdir(tmp_path)


def test_defaults(tmp_path: Path) -> None:
    loaded = load_config(workspace=str(tmp_path))
    assert loaded.model == "gpt-4o-mini"
    assert loaded.base_url == "https://api.openai.com/v1"
    assert loaded.max_steps == 20
    assert loaded.api_key == ""
    assert loaded.config_path is None
    assert loaded.log_dir == (tmp_path / ".dm-agent" / "sessions").resolve()


def test_file_env_and_cli_precedence(tmp_path: Path, monkeypatch) -> None:
    config_file = tmp_path / "dm-agent.toml"
    config_file.write_text(
        "\n".join(
            [
                "[agent]",
                "max_steps = 4",
                'log_dir = "logs"',
                "[llm]",
                'model = "from-file"',
                'base_url = "https://file.example/v1"',
                'api_key = "file-key"',
                "temperature = 0.5",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_MODEL", "from-env")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("DM_AGENT_MAX_STEPS", "7")
    loaded = load_config(workspace=str(tmp_path), config_path=str(config_file), max_steps=9)
    assert loaded.max_steps == 9
    assert loaded.model == "from-env"
    assert loaded.api_key == "env-key"
    assert loaded.base_url == "https://file.example/v1"
    assert loaded.log_dir == (tmp_path / "logs").resolve()
    assert loaded.temperature == 0.5


def test_invalid_values_raise(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[llm]\ntemperature = 9\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="temperature"):
        load_config(workspace=str(tmp_path), config_path=str(bad))
    bad.write_text('[llm]\nprovider = "anthropic"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="provider"):
        load_config(workspace=str(tmp_path), config_path=str(bad))
    bad.write_text("[agent]\nmax_steps = false\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="max_steps"):
        load_config(workspace=str(tmp_path), config_path=str(bad))
    with pytest.raises(ConfigError, match="not found"):
        load_config(workspace=str(tmp_path), config_path=str(tmp_path / "missing.toml"))


def test_example_config_loads(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "config.example.toml"
    loaded = load_config(workspace=str(tmp_path), config_path=str(example))
    assert loaded.model == "gpt-4o-mini"
    assert loaded.max_steps == 20
    assert loaded.api_key == ""
    assert loaded.shell_timeout_seconds == 30
