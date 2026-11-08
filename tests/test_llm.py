"""OpenAI-compatible client against a local fake server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from dm_agent.errors import LLMError
from dm_agent.llm import OpenAICompatibleLLM, parse_chat_completion, public_base_url


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - stdlib name
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.server.last_body = body  # type: ignore[attr-defined]
        self.server.last_auth = self.headers.get("Authorization")  # type: ignore[attr-defined]
        if self.server.mode == "error":  # type: ignore[attr-defined]
            payload = b'{"error":"nope"}'
            self.send_response(500)
        else:
            payload = json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": {"path": "a.py"},
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            ).encode("utf-8")
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):  # noqa: A003
        return


@pytest.fixture
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.mode = "ok"  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_client_posts_tools_and_parses_object_arguments(fake_server) -> None:
    port = fake_server.server_address[1]
    client = OpenAICompatibleLLM(
        api_key="secret-token",
        base_url=f"http://127.0.0.1:{port}/v1",
        model="test-model",
        timeout=5,
    )
    turn = client.complete([{"role": "user", "content": "hi"}], [{"type": "function"}])
    assert fake_server.last_auth == "Bearer secret-token"  # type: ignore[attr-defined]
    assert fake_server.last_body["model"] == "test-model"  # type: ignore[attr-defined]
    assert fake_server.last_body["tools"] == [{"type": "function"}]  # type: ignore[attr-defined]
    assert turn.tool_calls[0].name == "read_file"
    assert turn.tool_calls[0].arguments == {"path": "a.py"}


def test_http_error_raises(fake_server) -> None:
    fake_server.mode = "error"  # type: ignore[attr-defined]
    port = fake_server.server_address[1]
    client = OpenAICompatibleLLM(
        api_key="secret-token",
        base_url=f"http://127.0.0.1:{port}/v1",
        model="test-model",
        timeout=5,
    )
    with pytest.raises(LLMError, match="HTTP 500"):
        client.complete([{"role": "user", "content": "hi"}], [])


def test_parse_string_arguments_and_public_url() -> None:
    turn = parse_chat_completion(
        {
            "choices": [
                {
                    "message": {
                        "content": "thinking",
                        "tool_calls": [
                            {
                                "id": "call_9",
                                "function": {
                                    "name": "search",
                                    "arguments": "{\"pattern\": \"TODO\"}",
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )
    assert turn.content == "thinking"
    assert turn.tool_calls[0].arguments == {"pattern": "TODO"}
    assert public_base_url("https://user:secret@example.com:8443/v1") == "https://example.com:8443"
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        OpenAICompatibleLLM(api_key="", base_url="http://127.0.0.1", model="m")
