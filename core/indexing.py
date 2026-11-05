"""Keep chunks, embeddings, SQLite rows and the vector index in step.

SQLite decides which chunks exist. Each step writes files first, then commits rows,
then updates the index, so a crash at any point is repaired by ``reconcile``.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from config import Paths, Settings
from core.chunker import Chunk, chunk_pages
from core.embedding_engine import Embedder, ProgressCallback
from core.vector_store import VectorStore
from database.db_manager import DatabaseManager
from database.schema import IndexStatus
from utils.file_validation import safe_child_path
from utils.helpers import atomic_write_bytes

logger = logging.getLogger("documind.indexing")

CONTENT_FILENAME = "content.json"
_CHUNKS_FILENAME = "chunks.json"
_EMBEDDINGS_FILENAME = "embeddings.npy"
_EMBEDDINGS_META_FILENAME = "embeddings_meta.json"
_CHUNK_CACHE_DOCUMENTS = 8


class StoredContentError(Exception):
    """A document's stored files are missing or unreadable. Safe to show to the user."""


_MISSING_CONTENT = "This document's stored text is missing. Delete it and upload it again."


def document_dir(paths: Paths, document_id: str) -> Path:
    return safe_child_path(paths.processed_dir, document_id)


def read_content(paths: Paths, document_id: str) -> dict[str, Any]:
    path = document_dir(paths, document_id) / CONTENT_FILENAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StoredContentError(_MISSING_CONTENT) from exc


@dataclass(frozen=True)
class ReconcileReport:
    added: int = 0
    removed: int = 0
    documents_needing_index: int = 0


def _fingerprint(passages: list[str]) -> str:
    digest = hashlib.sha256()
    for passage in passages:
        digest.update(passage.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class DocumentIndexer:
    def __init__(
        self,
        db: DatabaseManager,
        store: VectorStore,
        embedder: Embedder,
        settings: Settings,
    ) -> None:
        self._db = db
        self._store = store
        self._embedder = embedder
        self._settings = settings
        self._read_chunks = lru_cache(maxsize=_CHUNK_CACHE_DOCUMENTS)(self._read_chunks_file)

    def _dir(self, document_id: str) -> Path:
        return document_dir(self._settings.paths, document_id)

    def _read_chunks_file(self, document_id: str) -> tuple[Chunk, ...]:
        path = self._dir(document_id) / _CHUNKS_FILENAME
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
            return tuple(Chunk(**{**item, "passages": tuple(item["passages"])}) for item in items)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise StoredContentError(_MISSING_CONTENT) from exc

    def load_chunks(self, document_id: str) -> tuple[Chunk, ...]:
        """Return a document's chunks in order, from a small in-memory cache."""
        return self._read_chunks(document_id)

    def load_passage_vectors(self, document_id: str) -> np.ndarray | None:
        """Return one stored vector per passage, in chunk order, or None if they are stale."""
        passages = [p for chunk in self.load_chunks(document_id) for p in chunk.passages]
        vectors = self._load_embeddings(document_id, _fingerprint(passages))
        return vectors if vectors is not None and len(vectors) == len(passages) else None

    def _load_embeddings(self, document_id: str, fingerprint: str) -> np.ndarray | None:
        """Return stored embeddings if they match this model and exact passage text."""
        directory = self._dir(document_id)
        try:
            meta = json.loads((directory / _EMBEDDINGS_META_FILENAME).read_text(encoding="utf-8"))
            if meta != {"model": self._embedder.model_name, "fingerprint": fingerprint}:
                return None
            return np.load(directory / _EMBEDDINGS_FILENAME, allow_pickle=False)
        except (OSError, ValueError, KeyError):
            return None

    def _save_embeddings(self, document_id: str, vectors: np.ndarray, fingerprint: str) -> None:
        directory = self._dir(document_id)
        buffer = io.BytesIO()
        np.save(buffer, vectors, allow_pickle=False)
        atomic_write_bytes(directory / _EMBEDDINGS_FILENAME, buffer.getvalue())
        meta = {"model": self._embedder.model_name, "fingerprint": fingerprint}
        atomic_write_bytes(
            directory / _EMBEDDINGS_META_FILENAME, json.dumps(meta).encode("utf-8")
        )

    def index_document(
        self,
        document_id: str,
        pages: list[dict[str, Any]],
        progress: ProgressCallback | None = None,
    ) -> int:
        """Chunk and embed a document and add it to the index. Returns the chunk count."""
        chunks = chunk_pages(document_id, pages, self._settings.chunking)
        passages = [passage for chunk in chunks for passage in chunk.passages]
        fingerprint = _fingerprint(passages)
        vectors = self._load_embeddings(document_id, fingerprint)
        if vectors is None or len(vectors) != len(passages):
            vectors = self._embedder.embed_texts(passages, progress)
            self._save_embeddings(document_id, vectors, fingerprint)

        payload = json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False)
        atomic_write_bytes(self._dir(document_id) / _CHUNKS_FILENAME, payload.encode("utf-8"))
        self._read_chunks.cache_clear()

        previous = self._db.get_vector_ids([document_id])
        rows = [
            (
                chunk.chunk_id,
                chunk.chunk_index,
                chunk.page_number,
                chunk.token_count,
                len(chunk.passages),
            )
            for chunk in chunks
        ]
        vector_ids = self._db.replace_chunks(document_id, rows, self._embedder.model_name)
        self._store.remove(previous)
        self._store.add(vector_ids, vectors)
        self._store.save(self._embedder.model_name)
        logger.info("Indexed document %s with %d chunks", document_id, len(chunks))
        return len(chunks)

    def remove_document(self, document_id: str) -> None:
        """Remove a document's vectors. Its rows go when the document row is deleted."""
        vector_ids = self._db.get_vector_ids([document_id])
        if vector_ids:
            self._store.remove(vector_ids)
            self._store.save(self._embedder.model_name)
        self._read_chunks.cache_clear()

    def index_unindexed(self, progress: ProgressCallback | None = None) -> int:
        """Index documents whose indexing is outstanding. Returns how many succeeded."""
        documents = self._db.list_unindexed()
        indexed = 0
        for position, document in enumerate(documents):
            try:
                pages = read_content(self._settings.paths, document.document_id)["pages"]
            except StoredContentError as exc:
                # One damaged document must not stop the others.
                self._db.clear_index_state(document.document_id, IndexStatus.FAILED, str(exc))
                continue
            self.index_document(document.document_id, pages)
            indexed += 1
            if progress:
                progress((position + 1) / len(documents))
        return indexed

    def reconcile(self) -> ReconcileReport:
        """Make the index match SQLite. Needs no model, so it is safe at startup."""
        model = self._embedder.model_name
        if self._store.model_name not in (None, model):
            logger.warning("Embedding model changed; all documents need indexing again")
            self._store.clear()
            self._db.clear_all_index_state()
            self._store.save(model)

        expected = self._db.get_vector_ids()
        actual = self._store.ids()
        extra = actual - expected
        missing = expected - actual
        self._store.remove(extra)
        added = self._restore(missing)
        if extra or added:
            self._store.save(model)
        report = ReconcileReport(added, len(extra), len(self._db.list_unindexed()))
        if extra or missing:
            logger.warning("Reconciled vector index: %s", report)
        return report

    def _restore(self, vector_ids: set[int]) -> int:
        """Add missing vectors from stored embeddings, by their recorded row."""
        by_document: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for row in self._db.get_vector_rows(vector_ids).values():
            by_document[row.document_id].append((row.vector_id, row.embedding_row))

        added = 0
        for document_id, entries in by_document.items():
            try:
                stored = np.load(self._dir(document_id) / _EMBEDDINGS_FILENAME, allow_pickle=False)
                ids = [vector_id for vector_id, _ in entries]
                self._store.add(ids, stored[[embedding_row for _, embedding_row in entries]])
                added += len(ids)
            except (OSError, ValueError, IndexError):
                logger.warning("Stored embeddings unusable for document %s", document_id)
                self._store.remove(self._db.get_vector_ids([document_id]))
                self._db.clear_index_state(
                    document_id, IndexStatus.PENDING, "This document needs indexing again."
                )
        return added
