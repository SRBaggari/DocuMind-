"""SQLite schema, applied in order and tracked by PRAGMA user_version."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    NEEDS_OCR = "needs_ocr"
    FAILED = "failed"


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    original_filename: str
    file_type: str
    file_size: int
    sha256: str
    upload_timestamp: str
    status: DocumentStatus
    status_message: str | None = None
    warnings: tuple[str, ...] = ()
    page_count: int | None = None
    paragraph_count: int | None = None
    word_count: int | None = None
    char_count: int | None = None
    processed_timestamp: str | None = None


MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE documents (
        document_id         TEXT PRIMARY KEY,
        original_filename   TEXT NOT NULL,
        file_type           TEXT NOT NULL,
        file_size           INTEGER NOT NULL,
        sha256              TEXT NOT NULL UNIQUE,
        upload_timestamp    TEXT NOT NULL,
        status              TEXT NOT NULL
            CHECK (status IN ('processing', 'ready', 'needs_ocr', 'failed')),
        status_message      TEXT,
        warnings            TEXT NOT NULL DEFAULT '[]',
        page_count          INTEGER,
        paragraph_count     INTEGER,
        word_count          INTEGER,
        char_count          INTEGER,
        processed_timestamp TEXT
    );
    CREATE INDEX idx_documents_status ON documents (status);
    """,
)
