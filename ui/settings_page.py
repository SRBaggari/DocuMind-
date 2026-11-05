"""Settings page: local AI, embedding model, storage and data management."""

from __future__ import annotations

import streamlit as st

from config import settings
from core.ollama_client import INSTALL_HELP
from ui import components as ui
from utils.helpers import directory_size, format_size

MODEL_SETTING = "model"
EXTRACTIVE_CHOICE = "extractive"
_EXTRACTIVE_LABEL = "None (extractive only)"
_CONFIRM_WORD = "DELETE"
_CONFIRM_KEY = "clear_confirmation"
_SESSION_RESULTS = (
    "summary_result",
    "comparison_result",
    "insights_result",
    "upload_results",
    ui.ACTIVE_DOCUMENT_KEY,
    ui.OPEN_DOCUMENT_KEY,
)


def _show_local_ai(ctx: ui.AppContext) -> None:
    with ui.card("ollama"):
        ui.section_title("Local AI")
        status = ctx.status
        if status.running:
            ui.pill(f"Ollama {status.version} running", "good")
        elif status.installed:
            ui.pill("Ollama not running", "warn")
        else:
            ui.pill("Ollama not installed", "neutral")
        st.caption(status.message)
        if not status.running:
            st.caption(INSTALL_HELP)
        if status.models:
            options = [*status.models, _EXTRACTIVE_LABEL]
            current = ctx.model if ctx.model in options else _EXTRACTIVE_LABEL
            choice = st.selectbox(
                "Model", options, index=options.index(current), key="settings_model"
            )
            if choice != current:
                saved = EXTRACTIVE_CHOICE if choice == _EXTRACTIVE_LABEL else choice
                ctx.db.set_setting(MODEL_SETTING, saved)
                st.rerun()
            st.caption(
                "Without a model, DocuMind copies relevant sentences instead of writing answers."
            )
        if st.button("Check again", icon=":material/refresh:"):
            st.cache_data.clear()
            st.rerun()


def _show_embedding(ctx: ui.AppContext) -> None:
    engine, store = ctx.services.engine, ctx.services.store
    with ui.card("embedding"):
        ui.section_title("Embedding model")
        if engine.is_loaded:
            ui.pill("Loaded", "good")
        elif engine.is_downloaded:
            ui.pill("Downloaded, loads when needed", "good")
        else:
            ui.pill("Not downloaded", "warn")
        st.caption(engine.model_name)
        st.caption(
            f"Search index: {store.backend_name}, {len(store.ids()):,} passages. "
            "The model is downloaded once and then runs offline."
        )


def _show_storage(ctx: ui.AppContext) -> None:
    paths = settings.paths
    with ui.card("storage"):
        ui.section_title("Local storage")
        ui.stat_cards(
            [
                ("Documents", f"{len(ctx.db.list_documents()):,}"),
                ("Uploaded files", format_size(directory_size(paths.uploads_dir))),
                ("Extracted text", format_size(directory_size(paths.processed_dir))),
                ("Search index", format_size(directory_size(paths.indexes_dir))),
            ]
        )
        st.caption(f"Location: {paths.data_dir}")


def _clear_all(ctx: ui.AppContext) -> None:
    removed = ctx.services.ingestor.delete_all()
    for key in _SESSION_RESULTS:
        st.session_state.pop(key, None)
    st.session_state[_CONFIRM_KEY] = ""
    st.toast(f"Removed {removed} document(s) and all chat history.")


def _show_data_management(ctx: ui.AppContext) -> None:
    with ui.card("data"):
        ui.section_title("Data management")
        st.caption(
            "This deletes every document, search index and chat from this computer. "
            "It cannot be undone."
        )
        typed = st.text_input(f"Type {_CONFIRM_WORD} to confirm", key=_CONFIRM_KEY)
        st.button(
            "Clear all application data",
            type="primary",
            icon=":material/delete_forever:",
            disabled=typed != _CONFIRM_WORD,
            on_click=_clear_all,
            args=(ctx,),
        )


def render(ctx: ui.AppContext) -> None:
    ui.page_header("Settings", "Local AI, storage and your data.")
    left, right = st.columns(2, gap="medium")
    with left:
        _show_local_ai(ctx)
    with right:
        _show_embedding(ctx)
    _show_storage(ctx)
    _show_data_management(ctx)
