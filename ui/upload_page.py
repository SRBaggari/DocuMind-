"""Upload page: add documents and follow their processing."""

from __future__ import annotations

import streamlit as st

from core.ingestion import IngestOutcome, IngestResult
from ui import components as ui
from utils.helpers import format_size

_RESULTS_KEY = "upload_results"
_UPLOADER_GENERATION_KEY = "uploader_generation"
_TONES = {
    IngestOutcome.READY: "good",
    IngestOutcome.DUPLICATE: "warn",
    IngestOutcome.NEEDS_OCR: "warn",
}


def _show_result(result: IngestResult) -> None:
    tone = _TONES.get(result.outcome, "bad")
    label = {"good": "Ready", "warn": "Check"}.get(tone, "Failed")
    with ui.card(f"result-{id(result)}"):
        text, status = st.columns([6, 1], vertical_alignment="center")
        document = result.document
        details = result.message
        if document and document.word_count:
            pages = f"{document.page_count} pages · " if document.page_count else ""
            details = (
                f"{document.file_type.upper()} · {format_size(document.file_size)} · {pages}"
                f"{document.word_count:,} words. {result.message}"
            )
        with text:
            ui.document_row(result.filename, details)
        with status:
            ui.pill(label, tone)
        if document:
            for warning in document.warnings:
                st.caption(ui.safe_markdown(warning))
        if document and result.outcome is IngestOutcome.READY:
            st.button(
                "Open document",
                key=f"open_{document.document_id}",
                on_click=ui.open_document,
                args=(document.document_id,),
            )


def _process_uploads(ctx: ui.AppContext, files: list) -> None:
    progress = st.progress(0.0, text="Processing documents")
    results: list[IngestResult] = []
    for index, file in enumerate(files, start=1):
        progress.progress((index - 1) / len(files), text=ui.safe_markdown(f"Processing {file.name}"))
        result = ctx.services.ingestor.ingest(
            file.name,
            file.getvalue(),
            lambda share, name=file.name, done=index - 1: progress.progress(
                (done + share) / len(files), text=ui.safe_markdown(f"Indexing {name}")
            ),
        )
        results.append(result)
        if result.outcome is IngestOutcome.READY and result.document:
            ctx.db.log_activity(
                "upload", f"Uploaded {result.filename}", result.document.document_id
            )
    progress.empty()
    st.session_state[_RESULTS_KEY] = results
    st.session_state[_UPLOADER_GENERATION_KEY] = (
        st.session_state.get(_UPLOADER_GENERATION_KEY, 0) + 1
    )
    st.rerun()


def render(ctx: ui.AppContext) -> None:
    ui.page_header("Upload documents", "Files are stored on this computer and never uploaded.")
    generation = st.session_state.get(_UPLOADER_GENERATION_KEY, 0)
    files = st.file_uploader(
        f"Drag files here or choose Upload. PDF, DOCX or TXT, up to {ctx.max_file_size_mb} MB each.",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key=f"uploader_{generation}",
    )
    left, right, _ = st.columns([1, 1, 2])
    if left.button(
        "Process documents",
        disabled=not files,
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
    ):
        _process_uploads(ctx, files)
    right.button(
        "Back to My Documents",
        icon=":material/folder_open:",
        width="stretch",
        on_click=ui.navigate,
        args=("documents",),
    )

    for result in st.session_state.get(_RESULTS_KEY, []):
        _show_result(result)
