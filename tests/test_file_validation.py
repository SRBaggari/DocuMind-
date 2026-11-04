"""Tests for upload validation and path safety."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from utils.file_validation import (
    FileValidationError,
    safe_child_path,
    sanitize_filename,
    validate_upload,
)

ALLOWED = (".pdf", ".docx", ".txt")
VALID_ID = "a" * 32


def validate(name: str, data: bytes, max_mb: int = 1, max_docx_mb: int = 1) -> tuple[str, str]:
    return validate_upload(name, data, ALLOWED, max_mb, max_docx_mb)


def docx_bytes(document_xml: bytes = b"<w/>") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document_xml)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/report.pdf", "report.pdf"),
        ("..\\..\\windows\\report.pdf", "report.pdf"),
        ("C:\\Users\\me\\notes.txt", "notes.txt"),
        ("na\x00me\n.txt", "name.txt"),
    ],
)
def test_sanitize_filename_removes_paths_and_controls(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


@pytest.mark.parametrize("raw", ["", "...", "../", "\x00"])
def test_sanitize_filename_rejects_unusable_names(raw: str) -> None:
    with pytest.raises(FileValidationError):
        sanitize_filename(raw)


def test_long_filename_keeps_extension() -> None:
    name = sanitize_filename("x" * 400 + ".pdf")
    assert len(name) == 255 and name.endswith(".pdf")


def test_accepts_valid_files() -> None:
    assert validate("a.PDF", b"%PDF-1.7 body") == ("a.PDF", ".pdf")
    assert validate("a.docx", docx_bytes())[1] == ".docx"
    assert validate("a.txt", b"hello")[1] == ".txt"
    assert validate("u.txt", "hi".encode("utf-16"))[1] == ".txt"


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("a.exe", b"MZ", "Unsupported"),
        ("noextension", b"data", "Unsupported"),
        ("a.pdf.exe", b"%PDF-", "Unsupported"),
        ("a.txt", b"", "empty"),
        ("a.pdf", b"MZ not a pdf", "not a PDF"),
        ("a.docx", b"plain text", "corrupted"),
        ("a.txt", b"MZ\x00\x00\x00", "binary"),
        ("a.docx", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1rest", "password-protected"),
    ],
)
def test_rejects_bad_uploads(name: str, data: bytes, message: str) -> None:
    with pytest.raises(FileValidationError, match=message):
        validate(name, data)


def test_rejects_plain_zip_named_docx() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "hi")
    with pytest.raises(FileValidationError, match="not a Word document"):
        validate("a.docx", buffer.getvalue())


def test_rejects_oversized_file() -> None:
    with pytest.raises(FileValidationError, match="larger than"):
        validate("a.txt", b"x" * (1024 * 1024 + 1))


def test_rejects_zip_bomb() -> None:
    with pytest.raises(FileValidationError, match="expands"):
        validate("a.docx", docx_bytes(b"0" * (2 * 1024 * 1024)))


def test_safe_child_path(tmp_path: Path) -> None:
    assert safe_child_path(tmp_path, VALID_ID, ".pdf") == tmp_path.resolve() / f"{VALID_ID}.pdf"


@pytest.mark.parametrize("document_id", ["../evil", "", "A" * 32, "a" * 31, "a" * 32 + "/x"])
def test_safe_child_path_rejects_bad_ids(tmp_path: Path, document_id: str) -> None:
    with pytest.raises(ValueError):
        safe_child_path(tmp_path, document_id)


def test_safe_child_path_rejects_escaping_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        safe_child_path(tmp_path, VALID_ID, "/../../evil")
