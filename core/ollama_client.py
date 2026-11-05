"""Talk to the Ollama service on this machine: status, model list and generation."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import requests

from config import OllamaConfig

INSTALL_HELP = (
    "Install Ollama from https://ollama.com/download, start it, then download a model "
    "with: ollama pull qwen2.5:3b"
)


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

    @property
    def default_model(self) -> str:
        return self._config.default_model

    def _get(self, path: str) -> dict:
        response = self._session.get(
            f"{self._config.base_url}{path}",
            timeout=self._config.connect_timeout_seconds,
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
