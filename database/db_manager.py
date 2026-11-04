"""All SQLite access. A short-lived connection per operation keeps this safe across
Streamlit's threads and leaves no open handle to block file deletion on Windows.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from database.schema import MIGRATIONS, DocumentRecord, DocumentStatus

_BUSY_TIMEOUT_SECONDS = 10.0


class DuplicateDocumentError(Exception):
    """A document with the same content hash already exists."""


def _to_record(row: sqlite3.Row) -> DocumentRecord:
    values = dict(row)
    values["status"] = DocumentStatus(values["status"])
    values["warnings"] = tuple(json.loads(values["warnings"]))
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
        with self._connect() as connection:
            connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
