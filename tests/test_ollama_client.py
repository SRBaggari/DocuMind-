"""Tests for the Ollama client, run against a real local HTTP server that imitates it."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from config import OllamaConfig, validate_local_url
from core.ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)


class FakeOllama(BaseHTTPRequestHandler):
    chat_status = 200
    chat_body: dict = {"message": {"content": "Hello"}}
    delay = 0.0
    last_request: dict = {}
    reject_think = False

    def log_message(self, *args: object) -> None:
        pass

    def _send(self, status: int, body: object) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/api/version":
            self._send(200, {"version": "9.9"})
        else:
            self._send(200, {"models": [{"name": "alpha:1b"}, {"name": "beta:3b"}]})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).last_request = body
        time.sleep(self.delay)
        if self.reject_think and "think" in body:
            self._send(400, {"error": "model does not support think"})
        else:
            self._send(self.chat_status, self.chat_body)


@pytest.fixture
def server() -> Iterator[type[FakeOllama]]:
    class Handler(FakeOllama):
        pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Handler.url = f"http://127.0.0.1:{httpd.server_port}"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield Handler
    httpd.shutdown()
    httpd.server_close()


def client_for(server: type[FakeOllama], **options: float) -> OllamaClient:
    return OllamaClient(OllamaConfig(base_url=server.url, **options))


def test_status_lists_models(server) -> None:
    status = client_for(server).status()
    assert status.running and status.version == "9.9"
    assert status.models == ("alpha:1b", "beta:3b")
    assert "2 model(s)" in status.message


def test_generate_sends_settings_and_returns_text(server) -> None:
    client = client_for(server)
    assert client.generate("sys", "hi", "alpha:1b", temperature=0.5, max_tokens=7) == "Hello"
    body = server.last_request
    assert body["model"] == "alpha:1b" and body["stream"] is False
    assert body["options"]["temperature"] == 0.5 and body["options"]["num_predict"] == 7
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_default_temperature_and_output_limit_come_from_config(server) -> None:
    client_for(server, temperature=0.1, max_output_tokens=33).generate("s", "p", "m")
    assert server.last_request["options"]["temperature"] == 0.1
    assert server.last_request["options"]["num_predict"] == 33


def test_think_blocks_are_removed(server) -> None:
    server.chat_body = {"message": {"content": "<think>secret\nplan</think>\nAnswer"}}
    assert client_for(server).generate("s", "p", "m") == "Answer"


def test_model_without_think_support_is_retried(server) -> None:
    server.reject_think = True
    assert client_for(server).generate("s", "p", "m") == "Hello"
    assert "think" not in server.last_request


def test_missing_model(server) -> None:
    server.chat_status = 404
    with pytest.raises(OllamaModelNotFoundError, match="ollama pull alpha:1b"):
        client_for(server).generate("s", "p", "alpha:1b")


def test_server_error(server) -> None:
    server.chat_status = 500
    with pytest.raises(OllamaError, match="500"):
        client_for(server).generate("s", "p", "m")


def test_timeout(server) -> None:
    server.delay = 1.0
    with pytest.raises(OllamaTimeoutError, match="longer than"):
        client_for(server, generate_timeout_seconds=0.1).generate("s", "p", "m")


def test_empty_reply_is_an_error(server) -> None:
    server.chat_body = {"message": {"content": "  "}}
    with pytest.raises(OllamaError, match="empty"):
        client_for(server).generate("s", "p", "m")


def test_unreadable_reply_is_an_error(server) -> None:
    server.chat_body = {"unexpected": True}
    with pytest.raises(OllamaError, match="could not be read"):
        client_for(server).generate("s", "p", "m")


def test_service_down() -> None:
    client = OllamaClient(OllamaConfig(base_url="http://127.0.0.1:9", connect_timeout_seconds=0.5))
    status = client.status()
    assert status.running is False and status.models == ()
    with pytest.raises(OllamaUnavailableError):
        client.generate("s", "p", "m")


def test_remote_urls_are_refused() -> None:
    with pytest.raises(ValueError):
        validate_local_url("https://api.example.com")
