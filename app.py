"""DocuMind entry point. Run with: streamlit run app.py"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from config import APP_NAME, PROJECT_ROOT, ensure_directories, settings
from core.document_comparator import DocumentComparator
from core.embedding_engine import EmbeddingEngine
from core.indexing import DocumentIndexer
from core.ingestion import DocumentIngestor
from core.insights import InsightsEngine
from core.ollama_client import OllamaClient, OllamaStatus
from core.question_answering import QuestionAnswerer
from core.retriever import Retriever
from core.summarizer import Summarizer
from core.vector_store import VectorStore
from database.db_manager import DatabaseManager
from ui import (
    chat_page,
    comparison_page,
    dashboard,
    insights_page,
    library_page,
    settings_page,
    summary_page,
    upload_page,
)
from ui import components as ui
from utils.logging_config import configure_logging

_STATUS_TTL_SECONDS = 15
_DEFAULT_PAGE = "dashboard"
_SIDEBAR_TAGLINE = "Local document intelligence"
# A local file, because Streamlit fetches Material and emoji page icons from a web server.
_PAGE_ICON = PROJECT_ROOT / "assets" / "favicon.png"
_NAVIGATION: tuple[tuple[str, str, str], ...] = (
    ("dashboard", "Dashboard", ":material/dashboard:"),
    ("documents", "My Documents", ":material/folder_open:"),
    ("summarize", "Summarization", ":material/summarize:"),
    ("ask", "Ask Documents", ":material/forum:"),
    ("compare", "Compare Documents", ":material/compare_arrows:"),
    ("insights", "Insights", ":material/insights:"),
    ("settings", "Settings", ":material/settings:"),
)
_PAGES: dict[str, Callable[[ui.AppContext], None]] = {
    "dashboard": dashboard.render,
    "documents": library_page.render,
    "upload": lambda ctx: upload_page.render(
        ctx.db, ctx.services.ingestor, ctx.services.indexer, ctx.max_file_size_mb
    ),
    "summarize": lambda ctx: summary_page.render(
        ctx.db, ctx.services.summarizer, ctx.model
    ),
    "ask": lambda ctx: chat_page.render(ctx.db, ctx.services.answerer, ctx.model),
    "compare": lambda ctx: comparison_page.render(
        ctx.db, ctx.services.comparator, ctx.model
    ),
    "insights": lambda ctx: insights_page.render(ctx.db, ctx.services.insights),
    "settings": settings_page.render,
}
# Upload has no sidebar entry of its own, because it belongs to My Documents.
_SIDEBAR_PARENT = {"upload": "documents"}


@st.cache_resource
def get_services() -> ui.Services:
    """Create the shared services once per server process."""
    ensure_directories(settings.paths)
    configure_logging(settings.paths.logs_dir)
    db = DatabaseManager(settings.paths.database_path)
    store = VectorStore(settings.paths.indexes_dir, settings.embedding.dimension)
    engine = EmbeddingEngine(settings.embedding)
    indexer = DocumentIndexer(db, store, engine, settings)
    ingestor = DocumentIngestor(db, settings, indexer)
    ingestor.recover_interrupted()
    indexer.reconcile()
    ollama = OllamaClient(settings.ollama)
    retriever = Retriever(db, store, engine, indexer, settings.retrieval)
    return ui.Services(
        db,
        ingestor,
        indexer,
        Summarizer(db, ollama, settings),
        QuestionAnswerer(db, retriever, engine, ollama, settings),
        DocumentComparator(db, indexer, ollama, settings),
        InsightsEngine(db, settings),
        ollama,
        engine,
        store,
    )


@st.cache_data(ttl=_STATUS_TTL_SECONDS, show_spinner=False)
def get_ollama_status() -> OllamaStatus:
    return get_services().ollama.status()


def choose_model(db: DatabaseManager, status: OllamaStatus) -> str | None:
    """Return the saved model if it is usable, then the default. None means extractive."""
    saved = db.get_setting(settings_page.MODEL_SETTING)
    if saved == settings_page.EXTRACTIVE_CHOICE:
        return None
    for candidate in (saved, settings.ollama.default_model):
        if candidate in status.models:
            return candidate
    return None


def current_page() -> str:
    if ui.PAGE_KEY not in st.session_state:
        requested = st.query_params.get("page", _DEFAULT_PAGE)
        st.session_state[ui.PAGE_KEY] = requested if requested in _PAGES else _DEFAULT_PAGE
    return st.session_state[ui.PAGE_KEY]


def render_sidebar(page: str, model: str | None) -> None:
    selected = _SIDEBAR_PARENT.get(page, page)
    with st.sidebar:
        ui.brand(_SIDEBAR_TAGLINE)
        for name, label, icon in _NAVIGATION:
            st.button(
                label,
                icon=icon,
                key=f"nav_{name}",
                type="primary" if name == selected else "secondary",
                width="stretch",
                on_click=ui.navigate,
                args=(name,),
            )
        ui.sidebar_status(model)


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon=str(_PAGE_ICON), layout="wide")
    ui.inject_styles()
    services = get_services()
    status = get_ollama_status()
    model = choose_model(services.db, status)
    page = current_page()
    render_sidebar(page, model)
    _PAGES[page](ui.AppContext(services, status, model, settings.ingestion.max_file_size_mb))


if __name__ == "__main__":
    main()
