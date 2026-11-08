"""Shared output bounds for tool observations."""

MAX_TOOL_OUTPUT = 16_000


def clip_text(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    """Keep tool observations bounded so one result cannot fill the log or context."""
    if len(text) <= limit:
        return text
    extra = len(text) - limit
    return text[:limit] + f"\n...[truncated {extra} characters]"
