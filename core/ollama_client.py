"""Talk to the Ollama service on this machine: status, model list and generation."""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import requests

from config import OllamaConfig

logger = logging.getLogger("documind.ollama")

INSTALL_HELP = (
    "Install Ollama from https://ollama.com/download, start it, then download a model "
    "with: ollama pull qwen2.5:3b"
)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class Reply(str):
    """A model's reply. ``truncated`` means it reached the output limit before finishing."""

    truncated: bool = False


class OllamaError(Exception):
    """Generation failed. The message is safe to show to the user."""


class OllamaUnavailableError(OllamaError):
    pass


class OllamaModelNotFoundError(OllamaError):
    pass


class OllamaTimeoutError(OllamaError):
    pass


@dataclass(frozen=True)
class OllamaStatus:
    installed: bool
    running: bool
    version: str | None = None
    models: tuple[str, ...] = ()

    @property
    def message(self) -> str:
        if self.running and self.models:
            return f"Ollama {self.version} is running with {len(self.models)} model(s)."
        if self.running:
            return "Ollama is running but has no models. Run: ollama pull qwen2.5:3b"
        if self.installed:
            return "Ollama is installed but not running. Start the Ollama app."
        return f"Ollama is not installed. {INSTALL_HELP}"


class LocalLLM(Protocol):
    def status(self) -> OllamaStatus: ...

    def generate(
        self,
        system: str,
        prompt: str,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...


def _is_installed() -> bool:
    if shutil.which("ollama"):
        return True
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return False
    return (Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe").exists()


class OllamaClient:
    def __init__(self, config: OllamaConfig) -> None:
        self._config = config
        self._session = requests.Session()
        # Environment proxies must never see document text bound for localhost.
        self._session.trust_env = False

    @property
    def default_model(self) -> str:
        return self._config.default_model

    def _get(self, path: str) -> dict:
        response = self._session.get(
            f"{self._config.base_url}{path}",
            timeout=self._config.connect_timeout_seconds,
            allow_redirects=False,
        )
        response.raise_for_status()
        return response.json()

    def status(self) -> OllamaStatus:
        """Report what is available. Never raises, so pages can always call it."""
        installed = _is_installed()
        try:
            version = self._get("/api/version").get("version")
            models = tuple(item["name"] for item in self._get("/api/tags").get("models", []))
        except (requests.RequestException, ValueError, KeyError):
            return OllamaStatus(installed=installed, running=False)
        return OllamaStatus(installed=True, running=True, version=version, models=models)

    def generate(
        self,
        system: str,
        prompt: str,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the model's reply, raising OllamaError if there is no usable one."""
        config = self._config
        output_tokens = max_tokens or config.max_output_tokens
        # Some models ignore think=False and reason inside the reply. Asking for thinking
        # makes Ollama return the reasoning in its own field, which is never used.
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": True,
            "options": {
                "temperature": config.temperature if temperature is None else temperature,
                "num_predict": output_tokens + config.thinking_tokens,
                "num_ctx": config.context_window_tokens,
            },
        }
        try:
            response = self._post_chat(body)
            if response.status_code == 400 and "think" in response.text.lower():
                del body["think"]
                body["options"]["num_predict"] = output_tokens
                response = self._post_chat(body)
        except requests.ConnectTimeout as exc:
            raise OllamaUnavailableError(
                "Could not reach Ollama on this computer. Check that it is running."
            ) from exc
        except requests.Timeout as exc:
            raise OllamaTimeoutError(
                f"The local model took longer than {config.generate_timeout_seconds:.0f} "
                "seconds to reply."
            ) from exc
        except requests.RequestException as exc:
            raise OllamaUnavailableError(
                "Could not reach Ollama on this computer. Check that it is running."
            ) from exc

        if response.status_code == 404:
            raise OllamaModelNotFoundError(
                f"The model {model} is not installed. Run: ollama pull {model}"
            )
        if response.status_code != 200:
            logger.warning("Ollama returned status %d", response.status_code)
            raise OllamaError(f"Ollama returned an error (status {response.status_code}).")
        try:
            data = response.json()
            content = data["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaError("Ollama returned a reply that could not be read.") from exc
        content = _THINK_BLOCK.sub("", content).strip()
        if not content:
            raise OllamaError("The local model returned an empty reply.")
        reply = Reply(content)
        reply.truncated = data.get("done_reason") == "length"
        return reply

    def _post_chat(self, body: dict) -> requests.Response:
        return self._session.post(
            f"{self._config.base_url}/api/chat",
            json=body,
            timeout=(self._config.connect_timeout_seconds, self._config.generate_timeout_seconds),
            allow_redirects=False,
        )
