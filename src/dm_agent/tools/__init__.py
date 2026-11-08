"""Built-in workspace tools."""

from dm_agent.tools.files import file_tools
from dm_agent.tools.lint_python import lint_python_tool
from dm_agent.tools.registry import ToolRegistry
from dm_agent.tools.run_pytest import run_pytest_tool
from dm_agent.tools.search import search_tool
from dm_agent.tools.shell import shell_tool

__all__ = ["ToolRegistry", "build_registry"]


def build_registry(workspace, *, shell_timeout: float) -> ToolRegistry:
    """Assemble the default tool set, confined to ``workspace``."""
    registry = ToolRegistry()
    for tool in file_tools(workspace):
        registry.add(tool)
    registry.add(search_tool(workspace))
    registry.add(shell_tool(workspace, timeout=shell_timeout))
    registry.add(lint_python_tool(workspace))
    registry.add(run_pytest_tool(workspace, timeout=shell_timeout))
    return registry
