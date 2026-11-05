"""Dashboard: totals, recent documents, quick actions and recent activity."""

from __future__ import annotations

import streamlit as st

from database.schema import DocumentStatus
from ui import components as ui
from utils.helpers import format_size, format_timestamp

_RECENT_DOCUMENTS = 5
_RECENT_ACTIVITY = 6
_QUICK_ACTIONS = (
    ("Upload documents", ":material/upload:", "upload"),
    ("Summarize a document", ":material/summarize:", "summarize"),
    ("Ask your documents", ":material/forum:", "ask"),
    ("Compare two documents", ":material/compare_arrows:", "compare"),
)


def render(ctx: ui.AppContext) -> None:
    ui.hero("Welcome to DocuMind", "Understand your documents. Everything runs on this computer.")
    documents = ctx.db.list_documents()
    if not documents:
        ui.no_documents("Upload a PDF, Word or text file to summarize, search and compare it.")
        return

    ready = [doc for doc in documents if doc.status is DocumentStatus.READY]
    ui.stat_cards(
        [
            ("Documents", f"{len(ready):,}", f"{len(documents) - len(ready)} need attention"),
            ("Pages processed", f"{sum(doc.page_count or 0 for doc in ready):,}", "PDF pages"),
            ("Words analyzed", f"{sum(doc.word_count or 0 for doc in ready):,}", "Across all documents"),
            ("Local AI", ctx.model or "Extractive", "Selected in Settings"),
        ]
    )
    st.write("")

    left, right = st.columns([3, 2], gap="medium")
    with left, ui.card("recent"):
        ui.section_title("Recent documents")
        for document in documents[:_RECENT_DOCUMENTS]:
            text, action = st.columns([5, 1], vertical_alignment="center")
            with text:
                ui.document_row(
                    document.original_filename,
                    f"{document.file_type.upper()} · {format_size(document.file_size)} · "
                    f"{format_timestamp(document.upload_timestamp)}",
                )
            action.button(
                "Open",
                key=f"dash_open_{document.document_id}",
                on_click=ui.open_document,
                args=(document.document_id,),
            )

    with right:
        with ui.card("actions"):
            ui.section_title("Quick actions")
            for label, icon, page in _QUICK_ACTIONS:
                st.button(
                    label,
                    icon=icon,
                    width="stretch",
                    on_click=ui.navigate,
                    args=(page,),
                    key=f"quick_{page}",
                )
        with ui.card("activity"):
            ui.section_title("Recent activity")
            activity = ctx.db.recent_activity(_RECENT_ACTIVITY)
            if not activity:
                st.caption("Nothing yet.")
            for item in activity:
                ui.activity_row(item.summary, format_timestamp(item.created_at))
