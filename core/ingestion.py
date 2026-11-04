"""Run an upload through validation, storage, extraction and metadata recording.

A row is written before any file, so a crash can leave a stale ``processing`` row
but never an untracked file. ``recover_interrupted`` clears those rows at startup.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from config import Settings
from config import settings as default_settings
from core.document_loader import (
    ExtractedDocument,
    ExtractionError,
    OCRRequiredError,
    load_document,
)
from database.db_manager import DatabaseManager, DuplicateDocumentError
from database.schema import DocumentRecord, DocumentStatus
from utils.file_validation import FileValidationError, safe_child_path, validate_upload
from utils.helpers import atomic_write_bytes, new_document_id, sha256_bytes, utc_now_iso
from utils.text_utils import count_paragraphs, count_words

logger = logging.getLogger("documind.ingestion")

_CONTENT_FILENAME = "content.json"
_CONTENT_FORMAT_VERSION = 1
_RETRYABLE_STATUSES = (DocumentStatus.FAILED, DocumentStatus.NEEDS_OCR)


class IngestOutcome(StrEnum):
    READY = "ready"
    NEEDS_OCR = "needs_ocr"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True)
class IngestResult:
    filename: str
    outcome: IngestOutcome
    message: str
    document: DocumentRecord | None = None


class DocumentIngestor:
    def __init__(self, db: DatabaseManager, settings: Settings = default_settings) -> None:
        self._db = db
        self._settings = settings

    def ingest(self, filename: str, data: bytes) -> IngestResult:
        """Process one upload. Expected failures are returned, not raised."""
        config = self._settings.ingestion
        try:
            display_name, extension = validate_upload(
                filename,
                data,
                config.allowed_extensions,
                config.max_file_size_mb,
                config.max_docx_uncompressed_mb,
            )
        except FileValidationError as exc:
            return IngestResult(filename, IngestOutcome.REJECTED, str(exc))

        digest = sha256_bytes(data)
        existing = self._db.find_by_hash(digest)
        if existing and existing.status in _RETRYABLE_STATUSES:
            self.delete(existing.document_id)
        elif existing:
            return self._duplicate(display_name, existing)

        record = DocumentRecord(
            document_id=new_document_id(),
            original_filename=display_name,
            file_type=extension.removeprefix("."),
            file_size=len(data),
            sha256=digest,
            upload_timestamp=utc_now_iso(),
            status=DocumentStatus.PROCESSING,
        )
        try:
            self._db.insert_document(record)
        except DuplicateDocumentError:
            # Another session stored the same file after the check above.
            winner = self._db.find_by_hash(digest)
            if winner:
                return self._duplicate(display_name, winner)
            raise
        return self._process(record, extension, data)

    def _duplicate(self, display_name: str, existing: DocumentRecord) -> IngestResult:
        return IngestResult(
            display_name,
            IngestOutcome.DUPLICATE,
            f"Already uploaded as {existing.original_filename}.",
            existing,
        )

    def _process(self, record: DocumentRecord, extension: str, data: bytes) -> IngestResult:
        document_id = record.document_id
        upload_path = self._upload_path(document_id, extension)
        try:
            atomic_write_bytes(upload_path, data)
            extracted = load_document(upload_path, self._settings.ingestion)
            self._save_content(document_id, extracted)
        except OCRRequiredError as exc:
            # The original is kept, because the document is valid and only needs OCR.
            self._db.complete_document(
                document_id,
                DocumentStatus.NEEDS_OCR,
                utc_now_iso(),
                status_message=str(exc),
            )
            return self._result(record, IngestOutcome.NEEDS_OCR, str(exc))
        except ExtractionError as exc:
            return self._fail(record, extension, str(exc))
        except Exception:
            logger.exception("Unexpected failure processing document %s", document_id)
            return self._fail(
                record, extension, "This document could not be processed. See the log for details."
            )

        text = extracted.text
        self._db.complete_document(
            document_id,
            DocumentStatus.READY,
            utc_now_iso(),
            warnings=tuple(extracted.warnings),
            page_count=extracted.page_count,
            paragraph_count=count_paragraphs(text),
            word_count=count_words(text),
            char_count=len(text),
        )
        logger.info("Processed document %s", document_id)
        return self._result(record, IngestOutcome.READY, "Processed successfully.")

    def _fail(self, record: DocumentRecord, extension: str, message: str) -> IngestResult:
        self._remove_files(record.document_id, extension)
        self._db.complete_document(
            record.document_id, DocumentStatus.FAILED, utc_now_iso(), status_message=message
        )
        logger.warning("Failed document %s", record.document_id)
        return self._result(record, IngestOutcome.FAILED, message)

    def _result(
        self, record: DocumentRecord, outcome: IngestOutcome, message: str
    ) -> IngestResult:
        document = self._db.get_document(record.document_id)
        return IngestResult(record.original_filename, outcome, message, document)

    def _upload_path(self, document_id: str, extension: str) -> Path:
        return safe_child_path(self._settings.paths.uploads_dir, document_id, extension)

    def _content_dir(self, document_id: str) -> Path:
        return safe_child_path(self._settings.paths.processed_dir, document_id)

    def _save_content(self, document_id: str, extracted: ExtractedDocument) -> None:
        content = {
            "format_version": _CONTENT_FORMAT_VERSION,
            "document_id": document_id,
            "page_count": extracted.page_count,
            "pages": [asdict(page) for page in extracted.pages],
            "blocks": [asdict(block) for block in extracted.blocks],
        }
        payload = json.dumps(content, ensure_ascii=False).encode("utf-8")
        atomic_write_bytes(self._content_dir(document_id) / _CONTENT_FILENAME, payload)

    def load_content(self, document_id: str) -> dict[str, Any]:
        """Return the stored extraction for a processed document."""
        path = self._content_dir(document_id) / _CONTENT_FILENAME
        return json.loads(path.read_text(encoding="utf-8"))

    def _remove_files(self, document_id: str, extension: str) -> None:
        self._upload_path(document_id, extension).unlink(missing_ok=True)
        content_dir = self._content_dir(document_id)
        if content_dir.exists():
            shutil.rmtree(content_dir)

    def delete(self, document_id: str) -> bool:
        """Delete a document's files, then its row, so a failure can be retried."""
        record = self._db.get_document(document_id)
        if record is None:
            return False
        self._remove_files(document_id, f".{record.file_type}")
        self._db.delete_document(document_id)
        logger.info("Deleted document %s", document_id)
        return True

    def recover_interrupted(self) -> int:
        """Mark documents left in ``processing`` by a crash as failed. Call once at startup."""
        interrupted = self._db.list_by_status(DocumentStatus.PROCESSING)
        for record in interrupted:
            self._fail(
                record,
                f".{record.file_type}",
                "Processing was interrupted. Upload this document again.",
            )
        return len(interrupted)
