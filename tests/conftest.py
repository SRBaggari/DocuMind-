"""Shared fixtures. Sample documents are generated, so no binary files are stored in git."""

from __future__ import annotations

import re
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path

import docx
import numpy as np
import pymupdf
import pytest

from config import Paths, Settings, ensure_directories
from core.indexing import DocumentIndexer
from core.ingestion import DocumentIngestor
from core.ollama_client import OllamaStatus
from core.retriever import Retriever
from core.vector_store import VectorStore
from database.db_manager import DatabaseManager


class HashingEmbedder:
    """A fast deterministic stand-in for tests: texts sharing words have similar vectors.

    Tests of the real model are in test_embedding_engine.py.
    """

    def __init__(self, model_name: str = "test-hashing-model") -> None:
        self.model_name = model_name
        self.calls = 0

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(384, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            vector[zlib.crc32(word.encode()) % 384] += 1
        if not vector.any():
            vector[0] = 1
        return vector / np.linalg.norm(vector)

    def embed_texts(self, texts: Sequence[str], progress: object = None) -> np.ndarray:
        self.calls += 1
        return np.stack([self._vector(text) for text in texts]) if texts else np.empty((0, 384))

    def embed_query(self, query: str) -> np.ndarray:
        return self._vector(query)


class FakeLLM:
    """Returns scripted replies and records prompts. Real-model tests are in test_llm_real.py."""

    def __init__(self, reply: str | Exception | Callable[[str], str] = "A reply.") -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.models: list[str] = []

    def status(self) -> OllamaStatus:
        return OllamaStatus(installed=True, running=True, version="test", models=("fake:1b",))

    def generate(self, system, prompt, model, temperature=None, max_tokens=None) -> str:
        self.prompts.append(prompt)
        self.models.append(model)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply(prompt) if callable(self.reply) else self.reply


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


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def store(test_settings: Settings) -> VectorStore:
    return VectorStore(test_settings.paths.indexes_dir, test_settings.embedding.dimension)


@pytest.fixture
def indexer(db, store, embedder, test_settings) -> DocumentIndexer:
    return DocumentIndexer(db, store, embedder, test_settings)


@pytest.fixture
def indexed_ingestor(db, test_settings, indexer) -> DocumentIngestor:
    return DocumentIngestor(db, test_settings, indexer)


@pytest.fixture
def retriever(db, store, embedder, indexer, test_settings) -> Retriever:
    return Retriever(db, store, embedder, indexer, test_settings.retrieval)
