"""Shared fixtures. Sample documents are generated, so no binary files are stored in git."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import docx
import pymupdf
import pytest

from config import Paths, Settings, ensure_directories
from core.ingestion import DocumentIngestor
from database.db_manager import DatabaseManager


def _png_bytes() -> bytes:
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
    pixmap.set_rect(pixmap.irect, (120, 120, 120))
    return pixmap.tobytes("png")


@pytest.fixture
def make_pdf(tmp_path: Path) -> Callable[..., Path]:
    """Build a PDF. Each page is a string of text, "" for blank, or None for a full-page image."""

    def build(pages: list[str | None], name: str = "sample.pdf", **save_options: object) -> Path:
        pdf = pymupdf.open()
        for content in pages:
            page = pdf.new_page()
            if content is None:
                page.insert_image(page.rect, stream=_png_bytes())
            elif content:
                page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), content, fontsize=11)
        path = tmp_path / name
        pdf.save(path, **save_options)
        pdf.close()
        return path

    return build


@pytest.fixture
def sample_docx(tmp_path: Path) -> Path:
    document = docx.Document()
    document.add_heading("Quarterly Report", level=0)
    document.add_heading("Revenue", level=1)
    document.add_paragraph("Revenue grew by 12% in 2024.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Region"
    table.cell(0, 1).text = "Sales"
    table.cell(1, 0).text = "North"
    table.cell(1, 1).text = "450"
    document.add_paragraph("The closing paragraph follows the table.")
    path = tmp_path / "sample.docx"
    document.save(path)
    return path


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    settings = Settings(paths=Paths(data_dir=tmp_path / "data"))
    ensure_directories(settings.paths)
    return settings


@pytest.fixture
def db(test_settings: Settings) -> DatabaseManager:
    return DatabaseManager(test_settings.paths.database_path)


@pytest.fixture
def ingestor(db: DatabaseManager, test_settings: Settings) -> DocumentIngestor:
    return DocumentIngestor(db, test_settings)
