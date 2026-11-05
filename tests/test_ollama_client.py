"""Tests for the Ollama client, run against a real local HTTP server that imitates it."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from config import OllamaConfig, validate_local_url
from core.ollama_client import OllamaClient


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


def test_remote_urls_are_refused() -> None:
    with pytest.raises(ValueError):
        validate_local_url("https://api.example.com")
