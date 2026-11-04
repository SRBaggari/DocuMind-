"""Tests for PDF, DOCX and TXT extraction."""

from __future__ import annotations

import sys
from pathlib import Path

import docx
import pymupdf
import pytest

from config import IngestionConfig
from core.document_loader import (
    BlockKind,
    CorruptedFileError,
    EmptyDocumentError,
    MissingDependencyError,
    OCRRequiredError,
    PageKind,
    PasswordProtectedError,
    load_document,
    load_docx,
    load_pdf,
    load_txt,
)

CONFIG = IngestionConfig()


def test_pdf_text_stays_on_its_own_page(make_pdf) -> None:
    path = make_pdf(["Alpha content on the first page.", "Beta content on page two."])
    result = load_pdf(path, CONFIG)
    assert result.page_count == 2
    assert [page.number for page in result.pages] == [1, 2]
    assert "Alpha" in result.pages[0].text and "Beta" not in result.pages[0].text
    assert "Beta" in result.pages[1].text


def test_pdf_reports_empty_pages_and_keeps_numbering(make_pdf) -> None:
    result = load_pdf(make_pdf(["First page text here.", "", "Third page text here."]), CONFIG)
    assert result.pages[1].kind is PageKind.EMPTY
    assert result.pages[2].number == 3
    assert any("Pages 2 are empty" in warning for warning in result.warnings)


def test_short_text_page_is_kept(make_pdf) -> None:
    result = load_pdf(make_pdf(["Title"]), CONFIG)
    assert result.pages[0].kind is PageKind.TEXT
    assert result.text == "Title"


def test_fully_scanned_pdf_requires_ocr(make_pdf) -> None:
    with pytest.raises(OCRRequiredError, match="OCR"):
        load_pdf(make_pdf([None, None]), CONFIG)


def test_partly_scanned_pdf_keeps_text_and_warns(make_pdf) -> None:
    result = load_pdf(make_pdf(["Readable first page content.", None, None]), CONFIG)
    assert result.pages[1].kind is PageKind.SCANNED
    assert "Readable" in result.text
    assert any("2, 3" in warning and "Most of" in warning for warning in result.warnings)


def test_small_logo_does_not_make_a_page_scanned(tmp_path: Path) -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
    page.insert_image(pymupdf.Rect(10, 10, 60, 60), stream=pixmap.tobytes("png"))
    page.insert_text((100, 100), "Thanks")
    path = tmp_path / "logo.pdf"
    pdf.save(path)
    pdf.close()
    assert load_pdf(path, CONFIG).pages[0].kind is PageKind.TEXT


def test_blank_pdf_is_empty(make_pdf) -> None:
    with pytest.raises(EmptyDocumentError):
        load_pdf(make_pdf(["", ""]), CONFIG)


def test_password_protected_pdf(make_pdf) -> None:
    path = make_pdf(
        ["Secret"], encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="user", owner_pw="owner"
    )
    with pytest.raises(PasswordProtectedError):
        load_pdf(path, CONFIG)


def test_garbage_pdf_is_corrupted(tmp_path: Path) -> None:
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF-1.7\n" + b"not a real pdf " * 50)
    with pytest.raises(CorruptedFileError):
        load_pdf(path, CONFIG)


def test_truncated_pdf_warns_that_content_may_be_missing(make_pdf, tmp_path: Path) -> None:
    data = make_pdf([f"Content of page {n}, long enough to count." for n in range(40)]).read_bytes()
    path = tmp_path / "truncated.pdf"
    path.write_bytes(data[: int(len(data) * 0.6)])
    result = load_pdf(path, CONFIG)
    assert any("damaged" in warning for warning in result.warnings)


def test_intact_pdf_has_no_damage_warning(make_pdf) -> None:
    assert load_pdf(make_pdf(["Healthy page content here."]), CONFIG).warnings == []


def test_docx_keeps_order_headings_and_tables(sample_docx: Path) -> None:
    result = load_docx(sample_docx)
    assert [block.kind for block in result.blocks] == [
        BlockKind.HEADING,
        BlockKind.HEADING,
        BlockKind.PARAGRAPH,
        BlockKind.TABLE,
        BlockKind.PARAGRAPH,
    ]
    assert [block.level for block in result.blocks[:2]] == [0, 1]
    assert result.blocks[3].text == "Region | Sales\nNorth | 450"
    assert result.page_count is None and result.pages[0].number is None


def test_docx_merged_cell_appears_once(tmp_path: Path) -> None:
    document = docx.Document()
    table = document.add_table(rows=1, cols=3)
    table.cell(0, 0).merge(table.cell(0, 1)).text = "Merged"
    table.cell(0, 2).text = "Other"
    path = tmp_path / "merged.docx"
    document.save(path)
    assert load_docx(path).blocks[0].text == "Merged | Other"


def test_empty_docx(tmp_path: Path) -> None:
    path = tmp_path / "empty.docx"
    docx.Document().save(path)
    with pytest.raises(EmptyDocumentError):
        load_docx(path)


def test_corrupted_docx(tmp_path: Path) -> None:
    path = tmp_path / "bad.docx"
    path.write_bytes(b"PK\x03\x04 broken")
    with pytest.raises(CorruptedFileError):
        load_docx(path)


@pytest.mark.parametrize(
    ("raw", "expected", "warns"),
    [
        ("Héllo wörld".encode("utf-8"), "Héllo wörld", False),
        (b"\xef\xbb\xbfWith BOM", "With BOM", False),
        ("UTF-16 text".encode("utf-16"), "UTF-16 text", False),
        ("Café".encode("cp1252"), "Café", True),
        (b"bad \x81 byte", "bad � byte", True),
    ],
)
def test_txt_encodings(tmp_path: Path, raw: bytes, expected: str, warns: bool) -> None:
    path = tmp_path / "file.txt"
    path.write_bytes(raw)
    result = load_txt(path)
    assert result.text == expected
    assert bool(result.warnings) is warns


def test_whitespace_only_txt_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "blank.txt"
    path.write_text(" \n\t\n ", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        load_txt(path)


def test_uppercase_extension_is_dispatched(tmp_path: Path) -> None:
    path = tmp_path / "NOTE.TXT"
    path.write_text("hello", encoding="utf-8")
    assert load_document(path, CONFIG).text == "hello"


def test_missing_pdf_library_gives_install_advice(make_pdf, monkeypatch) -> None:
    path = make_pdf(["text"])
    monkeypatch.setitem(sys.modules, "pymupdf", None)
    with pytest.raises(MissingDependencyError, match="pip install"):
        load_pdf(path, CONFIG)
