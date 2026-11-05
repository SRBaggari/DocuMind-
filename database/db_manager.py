"""All SQLite access. A short-lived connection per operation keeps this safe across
Streamlit's threads and leaves no open handle to block file deletion on Windows.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from database.schema import (
    MIGRATIONS,
    ChatMessage,
    DocumentRecord,
    DocumentStatus,
    IndexStatus,
    VectorRow,
)
from utils.helpers import utc_now_iso

_BUSY_TIMEOUT_SECONDS = 10.0
_MAX_MESSAGES_PER_CONVERSATION = 200


class DuplicateDocumentError(Exception):
    """A document with the same content hash already exists."""


def _to_record(row: sqlite3.Row) -> DocumentRecord:
    values = dict(row)
    values["status"] = DocumentStatus(values["status"])
    values["warnings"] = tuple(json.loads(values["warnings"]))
    values["index_status"] = IndexStatus(values["index_status"])
    return DocumentRecord(**values)


class DatabaseManager:
    def __init__(self, database_path: Path) -> None:
        self._path = database_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path, timeout=_BUSY_TIMEOUT_SECONDS)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def _migrate(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version > len(MIGRATIONS):
                raise RuntimeError(
                    f"Database version {version} is newer than this version of DocuMind."
                )
            for number, script in enumerate(MIGRATIONS[version:], start=version + 1):
                connection.executescript(
                    f"BEGIN;\n{script}\nPRAGMA user_version = {number};\nCOMMIT;"
                )

    def insert_document(self, record: DocumentRecord) -> None:
        """Insert a new row, raising DuplicateDocumentError if its hash already exists."""
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO documents (
                        document_id, original_filename, file_type, file_size,
                        sha256, upload_timestamp, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.document_id,
                        record.original_filename,
                        record.file_type,
                        record.file_size,
                        record.sha256,
                        record.upload_timestamp,
                        record.status.value,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateDocumentError(record.sha256) from exc

    def complete_document(
        self,
        document_id: str,
        status: DocumentStatus,
        processed_timestamp: str,
        *,
        status_message: str | None = None,
        warnings: tuple[str, ...] = (),
        page_count: int | None = None,
        paragraph_count: int | None = None,
        word_count: int | None = None,
        char_count: int | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET status = ?, status_message = ?, warnings = ?, page_count = ?,
                    paragraph_count = ?, word_count = ?, char_count = ?,
                    processed_timestamp = ?
                WHERE document_id = ?
                """,
                (
                    status.value,
                    status_message,
                    json.dumps(list(warnings)),
                    page_count,
                    paragraph_count,
                    word_count,
                    char_count,
                    processed_timestamp,
                    document_id,
                ),
            )

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return _to_record(row) if row else None

    def find_by_hash(self, sha256: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return _to_record(row) if row else None

    def list_documents(self) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY upload_timestamp DESC, rowid DESC"
            ).fetchall()
        return [_to_record(row) for row in rows]

    def list_by_status(self, status: DocumentStatus) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents WHERE status = ?", (status.value,)
            ).fetchall()
        return [_to_record(row) for row in rows]

    def delete_document(self, document_id: str) -> None:
        """Delete a document's row and any conversation that quotes it."""
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM conversations WHERE conversation_id IN (
                    SELECT conversation_id FROM messages WHERE instr(payload, ?) > 0
                )
                """,
                (document_id,),
            )
            connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

    def create_conversation(self, document_id: str | None) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO conversations (document_id, created_at) VALUES (?, ?)",
                (document_id, utc_now_iso()),
            )
        return cursor.lastrowid

    def latest_conversation(self, document_id: str | None) -> int | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT conversation_id FROM conversations WHERE document_id IS ?
                ORDER BY conversation_id DESC LIMIT 1
                """,
                (document_id,),
            ).fetchone()
        return row["conversation_id"] if row else None

    def delete_conversation(self, conversation_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,)
            )

    def add_message(
        self, conversation_id: int, role: str, content: str, payload: dict | None = None
    ) -> None:
        """Add a message, keeping only each conversation's most recent messages."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO messages (conversation_id, role, content, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, json.dumps(payload or {}), utc_now_iso()),
            )
            connection.execute(
                """
                DELETE FROM messages WHERE conversation_id = ? AND message_id NOT IN (
                    SELECT message_id FROM messages WHERE conversation_id = ?
                    ORDER BY message_id DESC LIMIT ?
                )
                """,
                (conversation_id, conversation_id, _MAX_MESSAGES_PER_CONVERSATION),
            )

    def list_messages(self, conversation_id: int) -> list[ChatMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content, payload, created_at FROM messages
                WHERE conversation_id = ? ORDER BY message_id
                """,
                (conversation_id,),
            ).fetchall()
        return [
            ChatMessage(row["role"], row["content"], json.loads(row["payload"]), row["created_at"])
            for row in rows
        ]

    def replace_chunks(
        self,
        document_id: str,
        chunks: Sequence[tuple[str, int, int | None, int, int]],
        model: str,
    ) -> list[int]:
        """Store a document's chunks and vectors and mark it indexed, in one transaction.

        ``chunks`` holds ``(chunk_id, chunk_index, page_number, token_count, passages)``
        in chunk_index order. Returns one vector ID per passage, in that same order,
        which is also the row order of the document's embeddings file.
        """
        vector_ids: list[int] = []
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            for chunk_id, chunk_index, page_number, token_count, passages in chunks:
                connection.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, document_id, chunk_index, page_number, token_count
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (chunk_id, document_id, chunk_index, page_number, token_count),
                )
                for passage_index in range(passages):
                    cursor = connection.execute(
                        """
                        INSERT INTO vectors (chunk_id, passage_index, embedding_row)
                        VALUES (?, ?, ?)
                        """,
                        (chunk_id, passage_index, len(vector_ids)),
                    )
                    vector_ids.append(cursor.lastrowid)
            connection.execute(
                """
                UPDATE documents
                SET index_status = 'indexed', index_message = NULL,
                    chunk_count = ?, embedding_model = ?
                WHERE document_id = ?
                """,
                (len(chunks), model, document_id),
            )
        return vector_ids

    def clear_index_state(
        self,
        document_id: str,
        status: IndexStatus = IndexStatus.PENDING,
        message: str | None = None,
    ) -> None:
        """Remove a document's chunks and record why it is not indexed."""
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            connection.execute(
                """
                UPDATE documents
                SET index_status = ?, index_message = ?, chunk_count = NULL,
                    embedding_model = NULL
                WHERE document_id = ?
                """,
                (status.value, message, document_id),
            )

    def clear_all_index_state(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks")
            connection.execute(
                """
                UPDATE documents
                SET index_status = 'pending', index_message = NULL, chunk_count = NULL,
                    embedding_model = NULL
                """
            )

    def list_unindexed(self) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents WHERE status = 'ready' AND index_status != 'indexed'"
            ).fetchall()
        return [_to_record(row) for row in rows]

    def get_vector_rows(self, vector_ids: Iterable[int]) -> dict[int, VectorRow]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT v.vector_id, v.chunk_id, c.document_id, c.chunk_index,
                       c.page_number, v.passage_index, v.embedding_row
                FROM vectors v JOIN chunks c ON c.chunk_id = v.chunk_id
                WHERE v.vector_id IN (SELECT value FROM json_each(?))
                """,
                (json.dumps(list(vector_ids)),),
            ).fetchall()
        return {row["vector_id"]: VectorRow(**dict(row)) for row in rows}

    def get_vector_ids(self, document_ids: Iterable[str] | None = None) -> set[int]:
        """Return all vector IDs, or only those belonging to ``document_ids``."""
        with self._connect() as connection:
            if document_ids is None:
                rows = connection.execute("SELECT vector_id FROM vectors").fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT v.vector_id FROM vectors v
                    JOIN chunks c ON c.chunk_id = v.chunk_id
                    WHERE c.document_id IN (SELECT value FROM json_each(?))
                    """,
                    (json.dumps(list(document_ids)),),
                ).fetchall()
        return {row["vector_id"] for row in rows}
