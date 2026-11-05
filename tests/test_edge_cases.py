"""Hostile and unusual input, run through the same paths a user would take."""

from __future__ import annotations

import io
import shutil
import threading
import zipfile
from pathlib import Path

import docx
import pymupdf
import pytest
from conftest import FakeLLM, HashingEmbedder

from core.document_comparator import ComparisonError, DocumentComparator
from core.embedding_engine import EmbeddingModelError
from core.extractive import GenerationMethod
from core.indexing import DocumentIndexer, document_dir
from core.ingestion import DocumentIngestor, IngestOutcome
from core.insights import InsightsEngine, InsightsError
from core.ollama_client import OllamaTimeoutError
from core.question_answering import NO_EVIDENCE_TEXT, QuestionAnswerer
from core.retriever import Retriever
from core.summarizer import Summarizer, SummaryError, SummaryMode
from core.vector_store import VectorStore
from database.schema import DocumentStatus, IndexStatus
from utils.file_validation import safe_child_path, sanitize_filename

TEXT = b"The Riverside authority reached revenue of 148 million dollars in 2024."


def pdf_bytes(*pages: str, password: str | None = None) -> bytes:
    pdf = pymupdf.open()
    for text in pages:
        page = pdf.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=10)
    options = {"encryption": pymupdf.PDF_ENCRYPT_AES_256, "user_pw": password} if password else {}
    data = pdf.tobytes(**options)
    pdf.close()
    return data


def scanned_pdf() -> bytes:
    pdf = pymupdf.open()
    page = pdf.new_page()
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 300), False)
    pixmap.set_rect(pixmap.irect, (200, 200, 200))
    page.insert_image(page.rect, stream=pixmap.tobytes("png"))
    data = pdf.tobytes()
    pdf.close()
    return data


def docx_bytes(build) -> bytes:
    document = docx.Document()
    build(document)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def qa(db, retriever, embedder, test_settings) -> QuestionAnswerer:
    return QuestionAnswerer(db, retriever, embedder, FakeLLM("Revenue [S1]."), test_settings)


@pytest.fixture
def summarizer(db, test_settings) -> Summarizer:
    return Summarizer(db, FakeLLM(), test_settings)


@pytest.mark.parametrize(
    ("name", "data", "fragment"),
    [
        ("empty.txt", b"", "empty"),
        ("blank.txt", b"   \n\n\t  \n", "no text"),
        ("notes.exe", b"MZ", "Unsupported"),
        ("noext", b"text", "Unsupported"),
        ("fake.pdf", b"<html>not a pdf</html>", "not a PDF"),
        ("broken.pdf", b"%PDF-1.7\n garbage garbage", "corrupted"),
        ("fake.docx", b"plain text", "corrupted"),
        ("old.docx", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1rest", "password-protected"),
        ("binary.txt", b"abc\x00\x01\x02def", "binary"),
    ],
)
def test_bad_files_are_refused_with_a_reason(ingestor, db, name, data, fragment) -> None:
    result = ingestor.ingest(name, data)
    assert result.outcome is not IngestOutcome.READY
    assert fragment.lower() in result.message.lower()
    assert all(d.status is not DocumentStatus.READY for d in db.list_documents())


def test_file_over_the_size_limit_is_refused(ingestor, test_settings) -> None:
    limit = test_settings.ingestion.max_file_size_mb * 1024 * 1024
    assert ingestor.ingest("big.txt", b"a" * (limit + 1)).outcome is IngestOutcome.REJECTED


def test_password_protected_pdf_explains_itself(ingestor) -> None:
    result = ingestor.ingest("locked.pdf", pdf_bytes("Secret text.", password="pw"))
    assert result.outcome is not IngestOutcome.READY and "password" in result.message.lower()


@pytest.mark.parametrize("data", [scanned_pdf(), pdf_bytes("", "")], ids=["scanned", "blank"])
def test_pdf_with_no_extractable_text_is_not_silently_empty(ingestor, data) -> None:
    result = ingestor.ingest("scan.pdf", data)
    assert result.outcome is not IngestOutcome.READY and result.message


def test_duplicate_upload_keeps_one_document(ingestor, db) -> None:
    ingestor.ingest("a.txt", TEXT)
    assert ingestor.ingest("b.txt", TEXT).outcome is IngestOutcome.DUPLICATE
    assert len(db.list_documents()) == 1


def test_same_file_uploaded_at_once_keeps_one_document(ingestor, db) -> None:
    threads = [
        threading.Thread(target=ingestor.ingest, args=(f"{i}.txt", TEXT)) for i in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(db.list_documents()) == 1


def test_zip_bomb_docx_is_not_opened(ingestor, test_settings) -> None:
    buffer = io.BytesIO()
    size = (test_settings.ingestion.max_docx_uncompressed_mb + 1) * 1024 * 1024
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", b"0" * size)
    result = ingestor.ingest("bomb.docx", buffer.getvalue())
    assert result.outcome is IngestOutcome.REJECTED and "expands" in result.message


def test_docx_external_entities_are_not_resolved(ingestor, tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRETVALUE", encoding="utf-8")
    source = docx_bytes(lambda d: d.add_paragraph("PLACEHOLDER"))
    buffer = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(source)) as original,
        zipfile.ZipFile(buffer, "w") as patched,
    ):
        for item in original.infolist():
            content = original.read(item.filename)
            if item.filename == "word/document.xml":
                text = content.decode("utf-8")
                doctype = f'<!DOCTYPE x [<!ENTITY e SYSTEM "{secret.as_uri()}">]>'
                text = text.replace("?>", f"?>{doctype}", 1).replace("PLACEHOLDER", "&e;")
                content = text.encode("utf-8")
            patched.writestr(item, content)
    result = ingestor.ingest("xxe.docx", buffer.getvalue())
    if result.outcome is IngestOutcome.READY:
        text = ingestor.load_content(result.document.document_id)
        assert "TOPSECRETVALUE" not in str(text)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd.txt", "passwd.txt"),
        ("..\\..\\windows\\system32\\evil.txt", "evil.txt"),
        ("C:\\temp\\a.txt", "a.txt"),
        ("a\x00b\x07.txt", "ab.txt"),
    ],
)
def test_path_traversal_names_are_reduced(raw, expected) -> None:
    assert sanitize_filename(raw) == expected


def test_stored_paths_never_use_the_uploaded_name(ingestor, test_settings) -> None:
    document = ingestor.ingest("../../../escape.txt", TEXT).document
    assert document.original_filename == "escape.txt"
    stored = list(test_settings.paths.uploads_dir.iterdir())
    assert [path.stem for path in stored] == [document.document_id]


@pytest.mark.parametrize("bad", ["../secret", "a/b", "", "x" * 32, "A" * 32])
def test_only_real_document_ids_build_paths(tmp_path, bad) -> None:
    with pytest.raises(ValueError):
        safe_child_path(tmp_path, bad)


def test_deleting_a_document_cannot_touch_other_files(ingestor, db, test_settings) -> None:
    keep = ingestor.ingest("keep.txt", TEXT).document
    gone = ingestor.ingest("gone.txt", TEXT + b" More.").document
    ingestor.delete(gone.document_id)
    assert db.get_document(keep.document_id)
    assert document_dir(test_settings.paths, keep.document_id).exists()
    assert not document_dir(test_settings.paths, gone.document_id).exists()


@pytest.mark.parametrize(
    "text",
    [
        "Résumé naïve café: 東京 مرحبا שלום Привет 🙂 costs €5 and £7.",
        "Zero\u200bwidth and bidi \u202ereversed\u202c text with enough words here.",
        "Lone surrogates are replaced \ufffd and tabs\tstay readable in this sentence.",
        "a" * 5000,
        ("word " * 40 + "\n") * 300,
    ],
    ids=["scripts", "invisible", "replacement", "unbroken", "repetitive"],
)
def test_unusual_text_flows_through_every_feature(
    indexed_ingestor, db, indexer, qa, summarizer, test_settings, text
) -> None:
    result = indexed_ingestor.ingest("odd.txt", text.encode("utf-8"))
    assert result.outcome is IngestOutcome.READY, result.message
    other = indexed_ingestor.ingest("other.txt", TEXT).document
    document = result.document
    for mode in SummaryMode:
        try:
            summarizer.summarize(document.document_id, mode, use_llm=False)
        except SummaryError:
            pass
    qa.ask("What does it say?", [document.document_id], use_llm=False)
    InsightsEngine(db, test_settings).analyze(document.document_id)
    comparator = DocumentComparator(db, indexer, FakeLLM(), test_settings)
    comparator.compare(document.document_id, other.document_id)


def test_bad_encoding_is_read_without_failing(ingestor) -> None:
    result = ingestor.ingest("latin.txt", "Café crème at 5 €".encode("cp1252"))
    assert result.outcome is IngestOutcome.READY
    assert "Café" in ingestor.load_content(result.document.document_id)["pages"][0]["text"]


def test_utf16_text_is_read(ingestor) -> None:
    result = ingestor.ingest("wide.txt", "Hello wide text file.".encode("utf-16"))
    assert result.outcome is IngestOutcome.READY
    assert "wide text" in ingestor.load_content(result.document.document_id)["pages"][0]["text"]


def test_tables_reach_search_and_answers(indexed_ingestor, retriever) -> None:
    def build(document) -> None:
        document.add_paragraph("Regional sales are shown below for the northern territory.")
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Region"
        table.cell(0, 1).text = "Sales"
        table.cell(1, 0).text = "Northland"
        table.cell(1, 1).text = "4500"

    document = indexed_ingestor.ingest("table.docx", docx_bytes(build)).document
    hits = retriever.retrieve("Northland Sales 4500 Region", [document.document_id])
    assert hits and "Northland" in hits[0].text


def test_very_large_document_is_chunked_and_summarized_in_parts(
    indexed_ingestor, db, test_settings
) -> None:
    text = "\n\n".join(
        f"Section {n} reports that station {n} recorded {n * 3} faults during the review period "
        f"and engineers replaced {n} pumps across the district before the winter maintenance."
        for n in range(1, 2500)
    )
    document = indexed_ingestor.ingest("large.txt", text.encode()).document
    assert document.status is DocumentStatus.READY and document.chunk_count > 100
    llm = FakeLLM("## Overview\nShort.\n\n## Main Purpose\nReview.\n\n## Important Findings\n- Faults")
    summary = Summarizer(db, llm, test_settings).summarize(
        document.document_id, SummaryMode.QUICK, model="m"
    )
    budget = test_settings.ollama.context_window_tokens
    assert len(llm.prompts) > 10 and summary.method is GenerationMethod.LLM
    assert max(len(prompt.split()) for prompt in llm.prompts) < budget


class NoEmbedder(HashingEmbedder):
    def embed_texts(self, texts, progress=None):
        raise EmbeddingModelError("The model is missing.")

    def embed_query(self, query):
        raise EmbeddingModelError("The model is missing.")


def test_missing_embedding_model_keeps_the_document_and_recovers(
    db, store, embedder, test_settings
) -> None:
    offline = DocumentIndexer(db, store, NoEmbedder(), test_settings)
    document = DocumentIngestor(db, test_settings, offline).ingest("a.txt", TEXT).document
    assert document.status is DocumentStatus.READY
    assert db.get_document(document.document_id).index_status is not IndexStatus.INDEXED
    summary = Summarizer(db, FakeLLM(), test_settings).summarize(
        document.document_id, SummaryMode.QUICK, use_llm=False
    )
    assert summary.sections

    online = DocumentIndexer(db, store, embedder, test_settings)
    assert online.index_unindexed() == 1
    assert db.get_document(document.document_id).index_status is IndexStatus.INDEXED


def test_ollama_timeout_during_a_question_still_answers(
    db, retriever, embedder, test_settings, indexed_ingestor
) -> None:
    indexed_ingestor.ingest("a.txt", TEXT)
    llm = FakeLLM(OllamaTimeoutError("Too slow."))
    answer = QuestionAnswerer(db, retriever, embedder, llm, test_settings).ask(
        "The Riverside authority reached revenue of how many dollars?", model="m"
    )
    assert answer.method is GenerationMethod.EXTRACTIVE
    assert any("Too slow" in notice for notice in answer.notices)


def test_no_relevant_results_are_reported(qa, indexed_ingestor) -> None:
    indexed_ingestor.ingest("a.txt", TEXT)
    answer = qa.ask("zebra quantum saxophone", model="m")
    assert answer.text == NO_EVIDENCE_TEXT and not answer.sources


def test_vectors_left_behind_by_a_crash_are_ignored_and_repaired(
    db, store, indexer, indexed_ingestor, retriever
) -> None:
    document = indexed_ingestor.ingest("a.txt", TEXT).document
    ids = db.get_vector_ids([document.document_id])
    db.delete_document(document.document_id)
    assert retriever.retrieve("Riverside revenue dollars") == []
    assert ids <= store.ids()
    assert indexer.reconcile().removed == len(ids)
    assert store.ids() == set()


def test_corrupted_index_file_is_rebuilt(db, indexed_ingestor, embedder, test_settings) -> None:
    indexed_ingestor.ingest("a.txt", TEXT)
    directory = test_settings.paths.indexes_dir
    for path in directory.iterdir():
        path.write_bytes(b"not an index")
    store = VectorStore(directory, test_settings.embedding.dimension)
    indexer = DocumentIndexer(db, store, embedder, test_settings)
    indexer.reconcile()
    retriever = Retriever(db, store, embedder, indexer, test_settings.retrieval)
    assert retriever.retrieve("The Riverside authority reached revenue of dollars")


def test_missing_stored_text_gives_a_clear_error_everywhere(
    db, indexer, indexed_ingestor, retriever, embedder, test_settings
) -> None:
    first = indexed_ingestor.ingest("a.txt", TEXT).document
    second = indexed_ingestor.ingest("b.txt", TEXT + b" Extra.").document
    shutil.rmtree(document_dir(test_settings.paths, first.document_id))

    with pytest.raises(SummaryError):
        Summarizer(db, FakeLLM(), test_settings).summarize(first.document_id, SummaryMode.QUICK)
    with pytest.raises(InsightsError):
        InsightsEngine(db, test_settings).analyze(first.document_id)
    with pytest.raises(ComparisonError):
        DocumentComparator(db, indexer, FakeLLM(), test_settings).compare(
            first.document_id, second.document_id
        )
    answer = QuestionAnswerer(db, retriever, embedder, FakeLLM(), test_settings).ask(
        "The Riverside authority reached revenue of dollars", [first.document_id], use_llm=False
    )
    assert answer.has_evidence is False


def test_one_damaged_document_does_not_stop_indexing_the_others(
    db, ingestor, indexer, test_settings
) -> None:
    damaged = ingestor.ingest("damaged.txt", TEXT).document
    healthy = ingestor.ingest("healthy.txt", TEXT + b" Extra.").document
    shutil.rmtree(document_dir(test_settings.paths, damaged.document_id))
    assert indexer.index_unindexed() == 1
    assert db.get_document(healthy.document_id).index_status is IndexStatus.INDEXED
    assert db.get_document(damaged.document_id).index_status is IndexStatus.FAILED


def test_a_document_with_missing_files_can_still_be_deleted(
    db, indexed_ingestor, test_settings
) -> None:
    document = indexed_ingestor.ingest("a.txt", TEXT).document
    shutil.rmtree(document_dir(test_settings.paths, document.document_id))
    for path in test_settings.paths.uploads_dir.iterdir():
        path.unlink()
    assert indexed_ingestor.delete(document.document_id)
    assert db.get_document(document.document_id) is None
