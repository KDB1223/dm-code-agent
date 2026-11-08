"""OpenAI-compatible chat completions client.

Any server that accepts ``POST {base_url}/chat/completions`` with bearer
auth and the standard tools array can be used. Set ``OPENAI_BASE_URL`` to
point at a local server or a hosted gateway. The native Anthropic API is
not implemented; most gateways already expose an OpenAI-compatible route.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from dm_agent.errors import LLMError


@dataclass
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelTurn:
    """A single non-streaming model response."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class OpenAICompatibleLLM:
    """Minimal chat-completions client built on the standard library."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        timeout: float = 120.0,
    ) -> None:
        if not api_key:
            raise LLMError(
                "OPENAI_API_KEY is not set. Export it, or set llm.api_key in dm-agent.toml."
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        """Send one chat completion and return the assistant turn."""
        url = self.base_url + "/chat/completions"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, method="POST")
        request.add_header("Content-Type", "application/json")
        request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise LLMError(f"LLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"LLM connection failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMError("LLM request timed out") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM returned non-JSON: {raw[:300]}") from exc
        return parse_chat_completion(payload)


def parse_chat_completion(payload: dict[str, Any]) -> ModelTurn:
    """Pull the assistant message out of a chat-completions JSON body."""
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        preview = json.dumps(payload)[:500]
        raise LLMError(f"unexpected LLM response shape: {preview}") from exc
    if not isinstance(message, dict):
        raise LLMError("unexpected LLM response shape: message is not an object")

    content = message.get("content")
    if content is None:
        text: str | None = None
    elif isinstance(content, str):
        text = content
    else:
        text = str(content)

    calls: list[ToolCall] = []
    for index, raw_call in enumerate(message.get("tool_calls") or []):
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function") or {}
        if not isinstance(function, dict):
            function = {}
        name = str(function.get("name") or "")
        arguments = _coerce_arguments(function.get("arguments"))
        call_id = str(raw_call.get("id") or f"call_{index}")
        calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return ModelTurn(content=text, tool_calls=calls)


def _coerce_arguments(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
        if isinstance(loaded, dict):
            return loaded
        return {"_value": loaded}
    return {"_value": raw}


def public_base_url(url: str) -> str:
    """Return scheme and host, dropping userinfo that might be a secret."""
    parts = urlsplit(url)
    if not parts.scheme or not parts.hostname:
        return url
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{parts.hostname}{port}"
