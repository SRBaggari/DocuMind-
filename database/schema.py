"""SQLite schema, applied in order and tracked by PRAGMA user_version."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    NEEDS_OCR = "needs_ocr"
    FAILED = "failed"


class IndexStatus(StrEnum):
    PENDING = "pending"
    INDEXED = "indexed"
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
    index_status: IndexStatus = IndexStatus.PENDING
    index_message: str | None = None
    chunk_count: int | None = None
    embedding_model: str | None = None


@dataclass(frozen=True)
class VectorRow:
    """Where a search vector came from. Text lives in the document's chunks.json file."""

    vector_id: int
    chunk_id: str
    document_id: str
    chunk_index: int
    page_number: int | None
    passage_index: int
    embedding_row: int


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    payload: dict
    created_at: str


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
    # AUTOINCREMENT stops a deleted vector's ID ever being given to a new vector.
    """
    ALTER TABLE documents ADD COLUMN index_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (index_status IN ('pending', 'indexed', 'failed'));
    ALTER TABLE documents ADD COLUMN index_message TEXT;
    ALTER TABLE documents ADD COLUMN chunk_count INTEGER;
    ALTER TABLE documents ADD COLUMN embedding_model TEXT;
    CREATE TABLE chunks (
        chunk_id    TEXT PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES documents (document_id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        page_number INTEGER,
        token_count INTEGER NOT NULL,
        UNIQUE (document_id, chunk_index)
    );
    CREATE TABLE vectors (
        vector_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk_id      TEXT NOT NULL REFERENCES chunks (chunk_id) ON DELETE CASCADE,
        passage_index INTEGER NOT NULL,
        embedding_row INTEGER NOT NULL,
        UNIQUE (chunk_id, passage_index)
    );
    """,
    # A NULL document_id is a conversation across all documents.
    """
    CREATE TABLE conversations (
        conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id     TEXT REFERENCES documents (document_id) ON DELETE CASCADE,
        created_at      TEXT NOT NULL
    );
    CREATE TABLE messages (
        message_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL
            REFERENCES conversations (conversation_id) ON DELETE CASCADE,
        role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content         TEXT NOT NULL,
        payload         TEXT NOT NULL DEFAULT '{}',
        created_at      TEXT NOT NULL
    );
    CREATE INDEX idx_messages_conversation ON messages (conversation_id);
    """,
)
