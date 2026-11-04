"""Generate embeddings locally with Sentence Transformers.

The model loads on first use, not at startup, and stays cached in this object.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

from config import EmbeddingConfig

logger = logging.getLogger("documind.embedding")

_SPECIAL_TOKENS = 2
_BATCHES_PER_PROGRESS_STEP = 8

ProgressCallback = Callable[[float], None]


class EmbeddingModelError(Exception):
    """The embedding model is unavailable. The message is safe to show to the user."""


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    def embed_texts(
        self, texts: Sequence[str], progress: ProgressCallback | None = None
    ) -> np.ndarray: ...

    def embed_query(self, query: str) -> np.ndarray: ...


class EmbeddingEngine:
    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                self._model = self._create_model()
            return self._model

    def _create_model(self) -> Any:
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingModelError(
                "Sentence Transformers is not installed. Run: pip install -r requirements.txt"
            ) from exc

        name, device = self._config.model_name, self._config.device
        try:
            model = SentenceTransformer(name, device=device, local_files_only=True)
        except Exception:
            logger.info("Embedding model not cached locally; downloading it once")
            try:
                model = SentenceTransformer(name, device=device)
            except Exception as exc:
                raise EmbeddingModelError(
                    f"The embedding model {name} is not on this computer and could not be "
                    "downloaded. Connect to the internet once so it can be saved locally."
                ) from exc

        dimension = model.encode(["dimension check"], convert_to_numpy=True).shape[1]
        if dimension != self._config.dimension:
            raise EmbeddingModelError(
                f"The embedding model {name} produces {dimension}-dimensional vectors, "
                f"but DocuMind is configured for {self._config.dimension}."
            )
        return model

    def _windows(self, model: Any, text: str) -> list[tuple[str, int]]:
        """Split text into pieces the model reads in full, with their token counts."""
        limit = min(self._config.max_input_tokens, model.max_seq_length) - _SPECIAL_TOKENS
        offsets = model.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )["offset_mapping"]
        if len(offsets) <= limit:
            return [(text, max(len(offsets), 1))]
        windows: list[tuple[str, int]] = []
        for start in range(0, len(offsets), limit):
            piece = offsets[start : start + limit]
            windows.append((text[piece[0][0] : piece[-1][1]], len(piece)))
        return windows

    def embed_texts(
        self, texts: Sequence[str], progress: ProgressCallback | None = None
    ) -> np.ndarray:
        """Embed texts of any length as unit vectors, one row per text.

        Text longer than the model's input limit is embedded in windows, which are
        averaged by token count, so no part of it is silently ignored.
        """
        dimension = self._config.dimension
        if not texts:
            return np.empty((0, dimension), dtype=np.float32)
        model = self._load()

        window_texts: list[str] = []
        owners: list[int] = []
        weights: list[int] = []
        for owner, text in enumerate(texts):
            for window, tokens in self._windows(model, text):
                window_texts.append(window)
                owners.append(owner)
                weights.append(tokens)

        step = self._config.batch_size * _BATCHES_PER_PROGRESS_STEP
        parts: list[np.ndarray] = []
        for start in range(0, len(window_texts), step):
            parts.append(
                model.encode(
                    window_texts[start : start + step],
                    batch_size=self._config.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            )
            if progress:
                progress(min((start + step) / len(window_texts), 1.0))

        vectors = np.concatenate(parts).astype(np.float32)
        pooled = np.zeros((len(texts), dimension), dtype=np.float32)
        np.add.at(pooled, owners, vectors * np.asarray(weights, dtype=np.float32)[:, None])
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled / np.maximum(norms, 1e-12)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
