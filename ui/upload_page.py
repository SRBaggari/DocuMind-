"""Upload page: add documents and see their metadata and processing status.

A functional page for Phase 2. Phase 6 replaces its layout and styling.
"""

from __future__ import annotations

import streamlit as st

from core.embedding_engine import EmbeddingModelError
from core.indexing import DocumentIndexer
from core.ingestion import DocumentIngestor, IngestOutcome, IngestResult
from database.db_manager import DatabaseManager
from database.schema import DocumentRecord, DocumentStatus, IndexStatus
from utils.helpers import format_size

_RESULTS_KEY = "upload_results"
_UPLOADER_GENERATION_KEY = "uploader_generation"
_STATUS_LABELS = {
    DocumentStatus.PROCESSING: "Processing",
    DocumentStatus.READY: "Ready",
    DocumentStatus.NEEDS_OCR: "Needs OCR",
    DocumentStatus.FAILED: "Failed",
}
_INDEX_LABELS = {
    IndexStatus.PENDING: "Not indexed",
    IndexStatus.INDEXED: "Indexed",
    IndexStatus.FAILED: "Failed",
}


def _show_result(result: IngestResult) -> None:
    text = f"**{result.filename}**: {result.message}"
    if result.outcome is IngestOutcome.READY:
        st.success(text)
    elif result.outcome in (IngestOutcome.DUPLICATE, IngestOutcome.NEEDS_OCR):
        st.warning(text)
    else:
        st.error(text)
    if result.document:
        for warning in result.document.warnings:
            st.warning(f"**{result.filename}**: {warning}")


def _process_uploads(files: list, ingestor: DocumentIngestor) -> None:
    progress = st.progress(0.0, text="Processing documents")
    results: list[IngestResult] = []
    for index, file in enumerate(files, start=1):
        progress.progress((index - 1) / len(files), text=f"Processing {file.name}")
        results.append(
            ingestor.ingest(
                file.name,
                file.getvalue(),
                lambda share, name=file.name, done=index - 1: progress.progress(
                    (done + share) / len(files), text=f"Indexing {name}"
                ),
            )
        )
    progress.empty()
    st.session_state[_RESULTS_KEY] = results
    st.session_state[_UPLOADER_GENERATION_KEY] = (
        st.session_state.get(_UPLOADER_GENERATION_KEY, 0) + 1
    )
    st.rerun()


def _document_row(document: DocumentRecord) -> dict[str, object]:
    return {
        "File name": document.original_filename,
        "Format": document.file_type.upper(),
        "Size": format_size(document.file_size),
        "Pages": document.page_count,
        "Words": document.word_count,
        "Uploaded": document.upload_timestamp.replace("T", " ")[:16] + " UTC",
        "Status": _STATUS_LABELS[document.status],
        "Chunks": document.chunk_count,
        "Search": (
            _INDEX_LABELS[document.index_status]
            if document.status is DocumentStatus.READY
            else "-"
        ),
    }


def _show_index_retry(db: DatabaseManager, indexer: DocumentIndexer) -> None:
    unindexed = db.list_unindexed()
    if not unindexed:
        return
    for document in unindexed:
        if document.index_message:
            st.caption(f"{document.original_filename}: {document.index_message}")
    if st.button(f"Index {len(unindexed)} waiting document(s)"):
        with st.spinner("Indexing documents"):
            try:
                indexer.index_unindexed()
            except EmbeddingModelError as exc:
                st.error(str(exc))
                return
        st.rerun()


def _show_library(
    db: DatabaseManager, ingestor: DocumentIngestor, indexer: DocumentIndexer
) -> None:
    documents = db.list_documents()
    st.subheader("Documents")
    if not documents:
        st.caption("No documents uploaded yet.")
        return
    st.dataframe([_document_row(doc) for doc in documents], hide_index=True)

    for document in documents:
        if document.status_message:
            st.caption(f"{document.original_filename}: {document.status_message}")

    _show_index_retry(db, indexer)

    names = {doc.document_id: doc.original_filename for doc in documents}
    selected = st.selectbox("Remove a document", list(names), format_func=names.get)
    if st.button("Remove document"):
        ingestor.delete(selected)
        st.session_state.pop(_RESULTS_KEY, None)
        st.rerun()


def render(
    db: DatabaseManager,
    ingestor: DocumentIngestor,
    indexer: DocumentIndexer,
    max_file_size_mb: int,
) -> None:
    st.subheader("Upload documents")
    generation = st.session_state.get(_UPLOADER_GENERATION_KEY, 0)
    files = st.file_uploader(
        f"PDF, DOCX or TXT, up to {max_file_size_mb} MB each",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key=f"uploader_{generation}",
    )
    if st.button("Process documents", disabled=not files, type="primary"):
        _process_uploads(files, ingestor)

    for result in st.session_state.get(_RESULTS_KEY, []):
        _show_result(result)

    _show_library(db, ingestor, indexer)
