"""Tests for the ingestion pipeline and metadata persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf

from config import Settings
from core import ingestion as ingestion_module
from core.ingestion import DocumentIngestor, IngestOutcome
from database.db_manager import DatabaseManager
from database.schema import DocumentRecord, DocumentStatus
from utils.helpers import new_document_id, utc_now_iso


def stored_files(settings: Settings) -> list[Path]:
    paths = settings.paths
    return [
        file
        for folder in (paths.uploads_dir, paths.processed_dir)
        for file in folder.rglob("*")
        if file.is_file()
    ]


def test_pdf_metadata_is_recorded(ingestor, make_pdf, test_settings) -> None:
    data = make_pdf(["Alpha beta gamma delta.", "Epsilon zeta."]).read_bytes()
    result = ingestor.ingest("report.pdf", data)
    doc = result.document
    assert result.outcome is IngestOutcome.READY
    assert (doc.file_type, doc.file_size, doc.page_count) == ("pdf", len(data), 2)
    assert doc.word_count == 6
    assert doc.char_count == len("Alpha beta gamma delta.\n\nEpsilon zeta.")
    assert doc.status is DocumentStatus.READY and doc.processed_timestamp
    assert (test_settings.paths.uploads_dir / f"{doc.document_id}.pdf").read_bytes() == data


def test_page_numbers_survive_storage(ingestor, make_pdf) -> None:
    data = make_pdf(["First page text.", "", "Third page text."]).read_bytes()
    doc = ingestor.ingest("a.pdf", data).document
    pages = ingestor.load_content(doc.document_id)["pages"]
    assert [(p["number"], p["kind"]) for p in pages] == [(1, "text"), (2, "empty"), (3, "text")]


def test_docx_structure_survives_storage(ingestor, sample_docx) -> None:
    doc = ingestor.ingest("a.docx", sample_docx.read_bytes()).document
    assert doc.page_count is None
    blocks = ingestor.load_content(doc.document_id)["blocks"]
    assert blocks[1] == {"kind": "heading", "text": "Revenue", "level": 1}


def test_full_text_is_not_stored_in_database(ingestor, test_settings) -> None:
    ingestor.ingest("a.txt", b"UNIQUEMARKERWORD appears only in the document body.")
    assert b"UNIQUEMARKERWORD" not in test_settings.paths.database_path.read_bytes()


def test_duplicate_content_is_not_stored_twice(ingestor, db) -> None:
    first = ingestor.ingest("a.txt", b"same content")
    second = ingestor.ingest("renamed.txt", b"same content")
    assert second.outcome is IngestOutcome.DUPLICATE
    assert second.document.document_id == first.document.document_id
    assert "a.txt" in second.message
    assert len(db.list_documents()) == 1


def test_document_ids_are_unique(ingestor) -> None:
    results = [ingestor.ingest(f"{n}.txt", f"body {n}".encode()) for n in range(20)]
    assert len({result.document.document_id for result in results}) == 20


def test_rejected_upload_leaves_nothing(ingestor, db, test_settings) -> None:
    result = ingestor.ingest("virus.exe", b"MZ")
    assert result.outcome is IngestOutcome.REJECTED
    assert db.list_documents() == [] and stored_files(test_settings) == []


def test_malicious_filename_cannot_escape_storage(ingestor, test_settings, tmp_path) -> None:
    doc = ingestor.ingest("../../../evil.txt", b"content").document
    assert doc.original_filename == "evil.txt"
    assert not (tmp_path / "evil.txt").exists()
    assert (test_settings.paths.uploads_dir / f"{doc.document_id}.txt").exists()


def test_failed_document_is_recorded_without_files(ingestor, test_settings) -> None:
    result = ingestor.ingest("blank.txt", b"   \n ")
    assert result.outcome is IngestOutcome.FAILED
    assert result.document.status is DocumentStatus.FAILED
    assert "no text" in result.document.status_message
    assert stored_files(test_settings) == []


def test_password_pdf_fails_gracefully(ingestor, make_pdf) -> None:
    path = make_pdf(["x"], encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    result = ingestor.ingest("locked.pdf", path.read_bytes())
    assert result.outcome is IngestOutcome.FAILED
    assert "password" in result.message


def test_scanned_pdf_is_kept_and_flagged(ingestor, make_pdf, test_settings) -> None:
    result = ingestor.ingest("scan.pdf", make_pdf([None]).read_bytes())
    assert result.outcome is IngestOutcome.NEEDS_OCR
    assert result.document.status is DocumentStatus.NEEDS_OCR
    assert "OCR" in result.document.status_message
    assert (test_settings.paths.uploads_dir / f"{result.document.document_id}.pdf").exists()


def test_failed_document_can_be_retried(ingestor, db) -> None:
    ingestor.ingest("blank.txt", b"  ")
    assert ingestor.ingest("blank.txt", b"  ").outcome is IngestOutcome.FAILED
    assert len(db.list_documents()) == 1


def test_partial_ocr_warning_is_persisted(ingestor, make_pdf) -> None:
    data = make_pdf(["Readable text on one page.", None]).read_bytes()
    doc = ingestor.ingest("mix.pdf", data).document
    assert doc.status is DocumentStatus.READY
    assert any("scanned" in warning for warning in doc.warnings)


def test_unexpected_error_marks_failed(ingestor, monkeypatch, test_settings) -> None:
    def boom(*_: object) -> None:
        raise RuntimeError("private detail")

    monkeypatch.setattr(ingestion_module, "load_document", boom)
    result = ingestor.ingest("a.txt", b"text")
    assert result.outcome is IngestOutcome.FAILED
    assert "private detail" not in result.message
    assert stored_files(test_settings) == []


def test_documents_survive_restart(ingestor, test_settings) -> None:
    doc = ingestor.ingest("a.txt", "Café notes".encode()).document
    reopened = DatabaseManager(test_settings.paths.database_path)
    assert reopened.get_document(doc.document_id) == doc
    fresh = DocumentIngestor(reopened, test_settings)
    assert fresh.recover_interrupted() == 0
    assert fresh.load_content(doc.document_id)["pages"][0]["text"] == "Café notes"
    assert fresh.ingest("b.txt", "Café notes".encode()).outcome is IngestOutcome.DUPLICATE


def test_interrupted_processing_is_recovered(ingestor, db, test_settings) -> None:
    document_id = new_document_id()
    db.insert_document(
        DocumentRecord(
            document_id, "a.txt", "txt", 4, "hash", utc_now_iso(), DocumentStatus.PROCESSING
        )
    )
    orphan = test_settings.paths.uploads_dir / f"{document_id}.txt"
    orphan.write_bytes(b"part")
    assert ingestor.recover_interrupted() == 1
    assert db.get_document(document_id).status is DocumentStatus.FAILED
    assert not orphan.exists()


def test_delete_removes_row_and_files(ingestor, db, test_settings) -> None:
    keep = ingestor.ingest("keep.txt", b"keep").document
    gone = ingestor.ingest("gone.txt", b"gone").document
    assert ingestor.delete(gone.document_id) is True
    assert db.get_document(gone.document_id) is None
    assert all(gone.document_id not in str(path) for path in stored_files(test_settings))
    assert db.get_document(keep.document_id) is not None
    assert ingestor.delete(gone.document_id) is False


def test_content_file_is_valid_utf8_json(ingestor, test_settings) -> None:
    doc = ingestor.ingest("a.txt", "日本語".encode()).document
    path = test_settings.paths.processed_dir / doc.document_id / "content.json"
    assert json.loads(path.read_text(encoding="utf-8"))["document_id"] == doc.document_id
