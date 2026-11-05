"""Extract cleaned, page-aware text and structure from PDF, DOCX and TXT files."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from config import IngestionConfig
from core.text_cleaner import clean_text, unwrap_lines

_TABLE_CELL_SEPARATOR = " | "
_PAGE_LIST_LIMIT = 10


class ExtractionError(Exception):
    """Extraction failed. The message is safe to show to the user."""


class EmptyDocumentError(ExtractionError):
    pass


class CorruptedFileError(ExtractionError):
    pass


class PasswordProtectedError(ExtractionError):
    pass


class OCRRequiredError(ExtractionError):
    """The document contains only scanned images, so it has no extractable text."""


class MissingDependencyError(ExtractionError):
    pass


class PageKind(StrEnum):
    TEXT = "text"
    SCANNED = "scanned"
    EMPTY = "empty"
    FAILED = "failed"


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"


@dataclass(frozen=True)
class Page:
    """One PDF page, or the whole document for formats without pages (number is None)."""

    number: int | None
    text: str
    kind: PageKind = PageKind.TEXT


@dataclass(frozen=True)
class Block:
    kind: BlockKind
    text: str
    level: int | None = None


@dataclass
class ExtractedDocument:
    pages: list[Page]
    page_count: int | None = None
    blocks: list[Block] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)


def _describe_pages(numbers: list[int]) -> str:
    shown = ", ".join(str(number) for number in numbers[:_PAGE_LIST_LIMIT])
    extra = len(numbers) - _PAGE_LIST_LIMIT
    return f"{shown} and {extra} more" if extra > 0 else shown


def _import_pymupdf() -> Any:
    try:
        import pymupdf
    except ImportError as exc:
        raise MissingDependencyError(
            "PDF support is not installed. Run: pip install -r requirements.txt"
        ) from exc
    pymupdf.TOOLS.mupdf_display_errors(False)
    return pymupdf


def _import_docx() -> Any:
    try:
        import docx
    except ImportError as exc:
        raise MissingDependencyError(
            "Word support is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return docx


def _image_coverage(page: Any) -> float:
    page_area = abs(page.rect)
    if not page_area:
        return 0.0
    covered = sum(abs(page.rect & info["bbox"]) for info in page.get_image_info())
    return min(covered / page_area, 1.0)


def _extract_pdf_page(page: Any, number: int, config: IngestionConfig) -> Page:
    blocks = page.get_text("blocks")
    paragraphs = [unwrap_lines(block[4].strip()) for block in blocks if block[6] == 0]
    text = clean_text("\n\n".join(paragraphs))
    if len(text) < config.min_chars_per_page:
        if _image_coverage(page) >= config.min_image_coverage:
            return Page(number, text, PageKind.SCANNED)
        if not text:
            return Page(number, text, PageKind.EMPTY)
    return Page(number, text)


def load_pdf(path: Path, config: IngestionConfig) -> ExtractedDocument:
    pymupdf = _import_pymupdf()
    try:
        # Opened from memory: after a failed open MuPDF can keep a file locked on Windows.
        pdf = pymupdf.open(stream=path.read_bytes(), filetype="pdf")
    except Exception as exc:
        raise CorruptedFileError("This PDF is corrupted and could not be opened.") from exc

    with pdf:
        if pdf.needs_pass:
            raise PasswordProtectedError(
                "This PDF is password-protected. Remove the password, then upload it again."
            )
        if pdf.page_count == 0:
            raise CorruptedFileError("This PDF is corrupted: no pages could be read.")
        pages: list[Page] = []
        for index in range(pdf.page_count):
            try:
                pages.append(_extract_pdf_page(pdf[index], index + 1, config))
            except Exception:
                pages.append(Page(index + 1, "", PageKind.FAILED))
        repaired = pdf.is_repaired

    return _finish_pdf(pages, config, repaired)


def _finish_pdf(
    pages: list[Page], config: IngestionConfig, repaired: bool = False
) -> ExtractedDocument:
    by_kind = {
        kind: [page.number for page in pages if page.kind == kind and page.number]
        for kind in PageKind
    }
    scanned = by_kind[PageKind.SCANNED]
    failed = by_kind[PageKind.FAILED]
    empty = by_kind[PageKind.EMPTY]

    if not any(page.text for page in pages):
        if scanned:
            raise OCRRequiredError(
                "This PDF appears to be scanned images, so it contains no extractable text. "
                "Run OCR on it, then upload the result."
            )
        if len(failed) == len(pages):
            raise CorruptedFileError("This PDF is corrupted: no page could be read.")
        raise EmptyDocumentError("This PDF contains no text.")

    warnings: list[str] = []
    if repaired:
        warnings.append("This PDF is damaged and was repaired, so some content may be missing.")
    if scanned:
        mostly_scanned = len(scanned) / len(pages) >= config.scanned_page_ratio
        advice = "Most of this document needs OCR." if mostly_scanned else "OCR is needed."
        warnings.append(
            f"Pages {_describe_pages(scanned)} appear to be scanned images "
            f"and were not read. {advice}"
        )
    if failed:
        warnings.append(f"Pages {_describe_pages(failed)} could not be read.")
    if empty:
        warnings.append(f"Pages {_describe_pages(empty)} are empty.")
    return ExtractedDocument(pages=pages, page_count=len(pages), warnings=warnings)


def _heading_level(paragraph: Any) -> int | None:
    try:
        style_name = paragraph.style.name or ""
    except (AttributeError, KeyError, ValueError):
        return None
    if style_name == "Title":
        return 0
    if style_name.startswith("Heading "):
        suffix = style_name.removeprefix("Heading ")
        return int(suffix) if suffix.isdigit() else None
    return None


def _table_text(table: Any) -> str:
    rows: list[str] = []
    for row in table.rows:
        cells: list[str] = []
        seen: set[int] = set()
        for cell in row.cells:
            # A merged cell is returned once per grid column it spans.
            if id(cell._tc) in seen:
                continue
            seen.add(id(cell._tc))
            cells.append(" ".join(cell.text.split()))
        if any(cells):
            rows.append(_TABLE_CELL_SEPARATOR.join(cells))
    return "\n".join(rows)


def load_docx(path: Path) -> ExtractedDocument:
    docx = _import_docx()
    try:
        document = docx.Document(str(path))
        blocks: list[Block] = []
        for item in document.iter_inner_content():
            if isinstance(item, docx.table.Table):
                text = clean_text(_table_text(item))
                if text:
                    blocks.append(Block(BlockKind.TABLE, text))
                continue
            text = clean_text(item.text)
            if text:
                level = _heading_level(item)
                kind = BlockKind.PARAGRAPH if level is None else BlockKind.HEADING
                blocks.append(Block(kind, text, level))
    except Exception as exc:
        raise CorruptedFileError("This Word document is corrupted and could not be read.") from exc

    if not blocks:
        raise EmptyDocumentError("This Word document contains no text.")
    text = "\n\n".join(block.text for block in blocks)
    return ExtractedDocument(pages=[Page(None, text)], blocks=blocks)


def _decode_text(data: bytes) -> tuple[str, str | None]:
    """Decode bytes, returning the text and a warning if characters may be wrong."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace"), None
    try:
        return data.decode("utf-8-sig"), None
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252"), "This file is not UTF-8. It was read as Windows-1252."
    except UnicodeDecodeError:
        return (
            data.decode("utf-8", errors="replace"),
            "This file has an unknown encoding, so some characters may be wrong.",
        )


def load_txt(path: Path) -> ExtractedDocument:
    text, warning = _decode_text(path.read_bytes())
    text = clean_text(text)
    if not text:
        raise EmptyDocumentError("This text file contains no text.")
    return ExtractedDocument(pages=[Page(None, text)], warnings=[warning] if warning else [])


def load_document(path: Path, config: IngestionConfig) -> ExtractedDocument:
    """Extract ``path`` by its extension, raising ExtractionError if that fails."""
    extension = path.suffix.lower()
    if extension == ".pdf":
        return load_pdf(path, config)
    if extension == ".docx":
        return load_docx(path)
    if extension == ".txt":
        return load_txt(path)
    raise ExtractionError(f"Unsupported file type: {extension}")
