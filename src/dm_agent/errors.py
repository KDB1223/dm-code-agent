"""Error types raised by dm-agent."""


class AgentError(Exception):
    """Base class for errors the CLI reports to the user."""


class ConfigError(AgentError):
    """The configuration file or environment is invalid."""


class PathJailError(AgentError):
    """A path or shell command would leave the workspace root."""


class LLMError(AgentError):
    """The language-model HTTP call failed or returned an unusable payload."""
