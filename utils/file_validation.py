"""Validate uploads by extension, size and content, and build safe storage paths."""

from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from pathlib import Path

_MAX_FILENAME_LENGTH = 255
_PDF_SIGNATURE = b"%PDF-"
_PDF_SIGNATURE_WINDOW = 1024
# Password-protected Word files, and legacy .doc files, use this OLE container.
_OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")
_BINARY_SAMPLE_BYTES = 8192
_DOCUMENT_ID = re.compile(r"[0-9a-f]{32}")
_MEGABYTE = 1024 * 1024


class FileValidationError(Exception):
    """An upload was rejected. The message is safe to show to the user."""


def sanitize_filename(filename: str) -> str:
    """Reduce a client-supplied name to a display name with no path or control characters."""
    name = re.split(r"[\\/]", filename)[-1]
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C").strip(" .")
    if not name:
        raise FileValidationError("The file has no usable name.")
    if len(name) > _MAX_FILENAME_LENGTH:
        stem, suffix = Path(name).stem, Path(name).suffix
        name = stem[: _MAX_FILENAME_LENGTH - len(suffix)] + suffix
    return name


def _validate_pdf(data: bytes) -> None:
    if _PDF_SIGNATURE not in data[:_PDF_SIGNATURE_WINDOW]:
        raise FileValidationError("This file has a .pdf extension but is not a PDF.")


def _validate_docx(data: bytes, max_uncompressed_mb: int) -> None:
    if data.startswith(_OLE_SIGNATURE):
        raise FileValidationError(
            "This Word file is password-protected or in the older .doc format. "
            "Remove the password or save it as .docx, then upload it again."
        )
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
    except Exception as exc:
        # A damaged archive can raise more than BadZipFile, such as NotImplementedError.
        raise FileValidationError("This file is corrupted or is not a Word document.") from exc
    names = {entry.filename for entry in entries}
    if not {"[Content_Types].xml", "word/document.xml"} <= names:
        raise FileValidationError("This file has a .docx extension but is not a Word document.")
    if sum(entry.file_size for entry in entries) > max_uncompressed_mb * _MEGABYTE:
        raise FileValidationError(
            f"This Word document expands to more than {max_uncompressed_mb} MB, "
            "so it was not opened."
        )


def _validate_txt(data: bytes) -> None:
    if data.startswith(_UTF16_BOMS):
        return
    if b"\x00" in data[:_BINARY_SAMPLE_BYTES]:
        raise FileValidationError("This file has a .txt extension but contains binary data.")


def validate_upload(
    filename: str,
    data: bytes,
    allowed_extensions: tuple[str, ...],
    max_file_size_mb: int,
    max_docx_uncompressed_mb: int,
) -> tuple[str, str]:
    """Check an upload and return its ``(display_name, extension)``.

    Raises FileValidationError with a user-facing message if it is unacceptable.
    """
    display_name = sanitize_filename(filename)
    extension = Path(display_name).suffix.lower()
    if extension not in allowed_extensions:
        supported = ", ".join(allowed_extensions)
        raise FileValidationError(f"Unsupported file type. Supported types: {supported}.")
    if not data:
        raise FileValidationError("This file is empty.")
    if len(data) > max_file_size_mb * _MEGABYTE:
        raise FileValidationError(f"This file is larger than the {max_file_size_mb} MB limit.")
    if extension == ".pdf":
        _validate_pdf(data)
    elif extension == ".docx":
        _validate_docx(data, max_docx_uncompressed_mb)
    else:
        _validate_txt(data)
    return display_name, extension


def is_document_id(text: str) -> bool:
    return _DOCUMENT_ID.fullmatch(text) is not None


def safe_child_path(base_dir: Path, document_id: str, suffix: str = "") -> Path:
    """Build ``base_dir/<document_id><suffix>``, refusing anything that escapes ``base_dir``."""
    if not _DOCUMENT_ID.fullmatch(document_id):
        raise ValueError(f"Invalid document ID: {document_id!r}")
    base = base_dir.resolve()
    candidate = (base / f"{document_id}{suffix}").resolve()
    if candidate.parent != base:
        raise ValueError(f"Path escapes storage folder: {candidate}")
    return candidate
