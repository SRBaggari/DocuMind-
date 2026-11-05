"""Exact cosine search over unit vectors, using FAISS or a NumPy fallback.

Vectors are keyed by the integer ``vector_id`` from the SQLite ``chunks`` table.
This index is derived data: ``DocumentIndexer.reconcile`` can rebuild it at any time.
"""

from __future__ import annotations

import io
import json
import logging
import threading
from collections.abc import Collection
from pathlib import Path
from typing import Protocol

import numpy as np

from utils.helpers import atomic_write_bytes

logger = logging.getLogger("documind.vector_store")

_META_FILENAME = "index_meta.json"


class _Backend(Protocol):
    file_name: str

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None: ...

    def remove(self, ids: np.ndarray) -> None: ...

    def search(
        self, query: np.ndarray, top_k: int, allowed: Collection[int] | None
    ) -> list[tuple[int, float]]: ...

    def ids(self) -> set[int]: ...

    def to_bytes(self) -> bytes: ...

    def load_bytes(self, data: bytes) -> None: ...


class FaissBackend:
    file_name = "vectors.faiss"

    def __init__(self, dimension: int) -> None:
        import faiss

        self._faiss = faiss
        self._dimension = dimension
        self._index = self._new_index()

    def _new_index(self):
        return self._faiss.IndexIDMap2(self._faiss.IndexFlatIP(self._dimension))

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        self._index.add_with_ids(vectors, ids)

    def remove(self, ids: np.ndarray) -> None:
        self._index.remove_ids(ids)

    def search(
        self, query: np.ndarray, top_k: int, allowed: Collection[int] | None
    ) -> list[tuple[int, float]]:
        if self._index.ntotal == 0:
            return []
        limit = min(top_k, self._index.ntotal)
        params = None
        if allowed is not None:
            selector = self._faiss.IDSelectorBatch(np.fromiter(allowed, dtype=np.int64))
            params = self._faiss.SearchParameters(sel=selector)
        scores, ids = self._index.search(query[None, :], limit, params=params)
        return [
            (int(vector_id), float(score))
            for vector_id, score in zip(ids[0], scores[0], strict=True)
            if vector_id != -1
        ]

    def ids(self) -> set[int]:
        return set(self._faiss.vector_to_array(self._index.id_map).tolist())

    def to_bytes(self) -> bytes:
        return self._faiss.serialize_index(self._index).tobytes()

    def load_bytes(self, data: bytes) -> None:
        self._index = self._faiss.deserialize_index(np.frombuffer(data, dtype=np.uint8))


class NumpyBackend:
    file_name = "vectors.npz"

    def __init__(self, dimension: int) -> None:
        self._ids = np.empty(0, dtype=np.int64)
        self._vectors = np.empty((0, dimension), dtype=np.float32)

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        self._ids = np.concatenate([self._ids, ids])
        self._vectors = np.concatenate([self._vectors, vectors])

    def remove(self, ids: np.ndarray) -> None:
        keep = ~np.isin(self._ids, ids)
        self._ids, self._vectors = self._ids[keep], self._vectors[keep]

    def search(
        self, query: np.ndarray, top_k: int, allowed: Collection[int] | None
    ) -> list[tuple[int, float]]:
        scores = self._vectors @ query
        positions = np.arange(len(self._ids))
        if allowed is not None:
            positions = positions[np.isin(self._ids, np.fromiter(allowed, dtype=np.int64))]
        ranked = positions[np.argsort(-scores[positions], kind="stable")][:top_k]
        return [(int(self._ids[i]), float(scores[i])) for i in ranked]

    def ids(self) -> set[int]:
        return set(self._ids.tolist())

    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        np.savez(buffer, ids=self._ids, vectors=self._vectors)
        return buffer.getvalue()

    def load_bytes(self, data: bytes) -> None:
        with np.load(io.BytesIO(data), allow_pickle=False) as stored:
            self._ids, self._vectors = stored["ids"], stored["vectors"]


def create_backend(dimension: int) -> _Backend:
    try:
        return FaissBackend(dimension)
    except ImportError:
        logger.warning("FAISS is not installed; using NumPy cosine search")
        return NumpyBackend(dimension)


class VectorStore:
    def __init__(self, directory: Path, dimension: int, backend: _Backend | None = None) -> None:
        self._directory = directory
        self._dimension = dimension
        self._backend = backend or create_backend(dimension)
        self._lock = threading.RLock()
        self.model_name: str | None = None
        self._load()

    @property
    def backend_name(self) -> str:
        return "FAISS" if isinstance(self._backend, FaissBackend) else "NumPy"

    @property
    def _path(self) -> Path:
        return self._directory / self._backend.file_name

    def _load(self) -> None:
        meta_path = self._directory / _META_FILENAME
        if not (self._path.exists() and meta_path.exists()):
            return
        try:
            self._backend.load_bytes(self._path.read_bytes())
            self.model_name = json.loads(meta_path.read_text(encoding="utf-8"))["model"]
        except Exception:
            logger.exception("Vector index is unreadable; it will be rebuilt")
            self._backend = type(self._backend)(self._dimension)
            self.model_name = None

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        """Add vectors. An ID already present is replaced, so duplicates cannot arise."""
        if not ids:
            return
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")
        id_array = np.asarray(ids, dtype=np.int64)
        with self._lock:
            self._backend.remove(id_array)
            self._backend.add(id_array, np.ascontiguousarray(vectors, dtype=np.float32))

    def remove(self, ids: Collection[int]) -> None:
        if ids:
            with self._lock:
                self._backend.remove(np.fromiter(ids, dtype=np.int64))

    def search(
        self, query: np.ndarray, top_k: int, allowed: Collection[int] | None = None
    ) -> list[tuple[int, float]]:
        """Return up to ``top_k`` ``(vector_id, cosine)`` pairs, best first."""
        if allowed is not None and not allowed:
            return []
        with self._lock:
            return self._backend.search(
                np.ascontiguousarray(query, dtype=np.float32), top_k, allowed
            )

    def ids(self) -> set[int]:
        with self._lock:
            return self._backend.ids()

    def clear(self) -> None:
        with self._lock:
            self._backend = type(self._backend)(self._dimension)

    def save(self, model_name: str) -> None:
        with self._lock:
            self.model_name = model_name
            atomic_write_bytes(self._path, self._backend.to_bytes())
            meta = json.dumps({"model": model_name}).encode("utf-8")
            atomic_write_bytes(self._directory / _META_FILENAME, meta)
