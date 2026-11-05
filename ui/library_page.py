"""My Documents: find, open and remove documents."""

from __future__ import annotations

import math

import streamlit as st

from core.embedding_engine import EmbeddingModelError
from core.indexing import StoredContentError, read_content
from database.schema import DocumentRecord, DocumentStatus, IndexStatus
from ui import components as ui
from ui.upload_page import RESULTS_KEY as UPLOAD_RESULTS_KEY
from utils.helpers import format_size, format_timestamp

_CONFIRM_DELETE_KEY = "confirm_delete"
_VISIBLE_KEY = "library_visible"
_PAGE_SIZE = 25
_PREVIEW_CHARS = 1500
_STATUS = {
    DocumentStatus.PROCESSING: ("Processing", "neutral"),
    DocumentStatus.READY: ("Ready", "good"),
    DocumentStatus.NEEDS_OCR: ("Needs OCR", "warn"),
    DocumentStatus.FAILED: ("Failed", "bad"),
}
_SORTS = {
    "Newest first": (lambda d: d.upload_timestamp, True),
    "Oldest first": (lambda d: d.upload_timestamp, False),
    "Name": (lambda d: d.original_filename.lower(), False),
    "Largest first": (lambda d: d.file_size, True),
    "Most words": (lambda d: d.word_count or 0, True),
}


def filter_documents(
    documents: list[DocumentRecord], search: str, file_types: list[str], sort: str
) -> list[DocumentRecord]:
    """Apply the library's search, type filter and sort order."""
    term = search.strip().lower()
    found = [
        document
        for document in documents
        if term in document.original_filename.lower()
        and (not file_types or document.file_type.upper() in file_types)
    ]
    key, reverse = _SORTS[sort]
    return sorted(found, key=key, reverse=reverse)


def _meta(document: DocumentRecord) -> str:
    parts = [document.file_type.upper(), format_size(document.file_size)]
    if document.page_count:
        parts.append(f"{document.page_count} pages")
    if document.word_count:
        parts.append(f"{document.word_count:,} words")
    parts.append(format_timestamp(document.upload_timestamp))
    return " · ".join(parts)


def _delete(ctx: ui.AppContext, document_id: str) -> None:
    ctx.services.ingestor.delete(document_id)
    uploads = st.session_state.get(UPLOAD_RESULTS_KEY, [])
    st.session_state[UPLOAD_RESULTS_KEY] = [
        item for item in uploads if not item.document or item.document.document_id != document_id
    ]
    st.session_state.pop(_CONFIRM_DELETE_KEY, None)
    st.session_state.pop(ui.OPEN_DOCUMENT_KEY, None)
    if st.session_state.get(ui.ACTIVE_DOCUMENT_KEY) == document_id:
        st.session_state.pop(ui.ACTIVE_DOCUMENT_KEY)


def _confirm_delete(ctx: ui.AppContext, document: DocumentRecord) -> None:
    st.warning(
        ui.safe_markdown(
            f"Delete {document.original_filename}? Its file, search index and any chats that "
            "quote it are removed. This cannot be undone."
        )
    )
    left, right, _ = st.columns([1, 1, 4])
    left.button(
        "Delete",
        type="primary",
        key=f"yes_{document.document_id}",
        on_click=_delete,
        args=(ctx, document.document_id),
    )
    right.button(
        "Cancel",
        key=f"no_{document.document_id}",
        on_click=lambda: st.session_state.pop(_CONFIRM_DELETE_KEY, None),
    )


def _show_row(ctx: ui.AppContext, document: DocumentRecord) -> None:
    with ui.card(f"doc-{document.document_id}"):
        text, status, opening, removal = st.columns([6, 1.1, 1, 1], vertical_alignment="center")
        with text:
            ui.document_row(document.original_filename, _meta(document))
        with status:
            ui.pill(*_STATUS[document.status])
        opening.button(
            "Open",
            key=f"open_{document.document_id}",
            on_click=ui.open_document,
            args=(document.document_id,),
        )
        removal.button(
            "Delete",
            key=f"del_{document.document_id}",
            on_click=lambda: st.session_state.update(
                {_CONFIRM_DELETE_KEY: document.document_id}
            ),
        )
        if document.status_message:
            st.caption(ui.safe_markdown(document.status_message))
        if st.session_state.get(_CONFIRM_DELETE_KEY) == document.document_id:
            _confirm_delete(ctx, document)


def _index_waiting(ctx: ui.AppContext) -> None:
    unindexed = ctx.db.list_unindexed()
    if not unindexed:
        return
    left, right = st.columns([4, 1], vertical_alignment="center")
    left.caption(f"{len(unindexed)} document(s) are waiting to be added to search.")
    if right.button("Index now", icon=":material/refresh:"):
        with st.spinner("Indexing documents"):
            try:
                ctx.services.indexer.index_unindexed()
            except EmbeddingModelError as exc:
                st.error(str(exc))
                return
        st.rerun()


def _show_detail(ctx: ui.AppContext, document: DocumentRecord) -> None:
    st.button(
        "Back to My Documents",
        icon=":material/arrow_back:",
        on_click=lambda: st.session_state.pop(ui.OPEN_DOCUMENT_KEY, None),
    )
    ui.page_header(document.original_filename, _meta(document))
    ui.stat_cards(
        [
            ("Words", f"{document.word_count or 0:,}"),
            ("Pages", f"{document.page_count:,}" if document.page_count else "N/A"),
            ("Paragraphs", f"{document.paragraph_count or 0:,}"),
            ("Chunks", f"{document.chunk_count or 0:,}"),
            (
                "Search",
                "Indexed" if document.index_status is IndexStatus.INDEXED else "Not indexed",
            ),
        ]
    )
    st.write("")
    for warning in document.warnings:
        st.warning(ui.safe_markdown(warning))
    if document.status_message:
        st.warning(ui.safe_markdown(document.status_message))
    if document.status is not DocumentStatus.READY:
        return

    actions = st.columns(4)
    for column, (label, icon, page) in zip(
        actions,
        (
            ("Summarize", ":material/summarize:", "summarize"),
            ("Ask questions", ":material/forum:", "ask"),
            ("Compare", ":material/compare_arrows:", "compare"),
            ("View insights", ":material/insights:", "insights"),
        ),
        strict=True,
    ):
        column.button(
            label,
            icon=icon,
            width="stretch",
            key=f"detail_{page}",
            on_click=ui.navigate,
            args=(page, document.document_id),
        )

    try:
        pages = read_content(ctx.services.ingestor.paths, document.document_id)["pages"]
    except StoredContentError as exc:
        st.error(str(exc))
        return
    text = "\n\n".join(page["text"] for page in pages if page["text"])
    with ui.card("preview"):
        ui.section_title("Text preview")
        clipped = len(text) > _PREVIEW_CHARS
        ui.quote(text[:_PREVIEW_CHARS] + ("..." if clipped else ""))


def render(ctx: ui.AppContext) -> None:
    open_id = st.session_state.get(ui.OPEN_DOCUMENT_KEY)
    document = ctx.db.get_document(open_id) if open_id else None
    if document:
        _show_detail(ctx, document)
        return

    header, action = st.columns([5, 1], vertical_alignment="center")
    with header:
        ui.page_header("My Documents", "Search, open and manage your documents.")
    action.button(
        "Upload",
        icon=":material/upload:",
        type="primary",
        width="stretch",
        on_click=ui.navigate,
        args=("upload",),
    )
    documents = ctx.db.list_documents()
    if not documents:
        ui.no_documents()
        return

    search, kinds, sort = st.columns([3, 2, 2])
    term = search.text_input("Search", placeholder="Search by file name")
    file_types = kinds.multiselect(
        "File type", sorted({doc.file_type.upper() for doc in documents})
    )
    order = sort.selectbox("Sort by", list(_SORTS))
    _index_waiting(ctx)

    found = filter_documents(documents, term, file_types, order)
    st.caption(f"{len(found)} of {len(documents)} documents")
    if not found:
        ui.empty_state("No matching documents", "Try a different search or file type.")
    visible = st.session_state.get(_VISIBLE_KEY, _PAGE_SIZE)
    for document in found[:visible]:
        _show_row(ctx, document)
    if len(found) > visible:
        remaining = math.ceil((len(found) - visible) / _PAGE_SIZE)
        st.button(
            f"Show more ({remaining} more page(s))",
            on_click=lambda: st.session_state.update({_VISIBLE_KEY: visible + _PAGE_SIZE}),
        )
