"""Turn a query into ranked chunks, with document filtering and a relevance floor."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from config import RetrievalConfig
from core.embedding_engine import Embedder
from core.indexing import DocumentIndexer
from core.vector_store import VectorStore
from database.db_manager import DatabaseManager

logger = logging.getLogger("documind.retriever")

# Several passages of one chunk can match, so fetch extra before collapsing to chunks.
_PASSAGE_OVERSAMPLING = 5


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    passage: str
    page_number: int | None
    chunk_index: int
    score: float


class Retriever:
    def __init__(
        self,
        db: DatabaseManager,
        store: VectorStore,
        embedder: Embedder,
        indexer: DocumentIndexer,
        config: RetrievalConfig,
    ) -> None:
        self._db = db
        self._store = store
        self._embedder = embedder
        self._indexer = indexer
        self._config = config

    def retrieve(
        self,
        query: str,
        document_ids: Iterable[str] | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[RetrievedChunk]:
        """Return the chunks whose best passage is most similar to ``query``, best first.

        ``score`` is that passage's cosine similarity. An empty result means nothing
        passed the relevance floor, so callers must treat it as "no evidence" rather
        than falling back to weaker matches.
        """
        if not query.strip():
            return []
        allowed = None if document_ids is None else self._db.get_vector_ids(document_ids)
        if allowed is not None and not allowed:
            return []

        floor = self._config.min_similarity if min_similarity is None else min_similarity
        limit = top_k or self._config.top_k
        query_vector = self._embedder.embed_query(query)
        hits = self._store.search(query_vector, limit * _PASSAGE_OVERSAMPLING, allowed)
        hits = [(vector_id, score) for vector_id, score in hits if score >= floor]
        rows = self._db.get_vector_rows(vector_id for vector_id, _ in hits)

        results: list[RetrievedChunk] = []
        seen: set[str] = set()
        filenames: dict[str, str] = {}
        for vector_id, score in hits:
            row = rows.get(vector_id)
            if row is None:
                logger.warning("Vector %d has no database row; skipped", vector_id)
                continue
            if row.chunk_id in seen:
                continue
            seen.add(row.chunk_id)
            if row.document_id not in filenames:
                document = self._db.get_document(row.document_id)
                filenames[row.document_id] = document.original_filename if document else ""
            chunk = self._indexer.load_chunks(row.document_id)[row.chunk_index]
            results.append(
                RetrievedChunk(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    filename=filenames[row.document_id],
                    text=chunk.text,
                    passage=chunk.passages[row.passage_index],
                    page_number=row.page_number,
                    chunk_index=row.chunk_index,
                    score=score,
                )
            )
            if len(results) == limit:
                break
        return results
