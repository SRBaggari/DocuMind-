"""DocuMind entry point. Run with: streamlit run app.py

The full navigation and design are added in Phase 6; this file stays a thin shell.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from config import APP_NAME, APP_TAGLINE, ensure_directories, settings
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
from ui import chat_page, comparison_page, insights_page, summary_page, upload_page
from utils.logging_config import configure_logging

_STATUS_TTL_SECONDS = 15
_NO_LLM = "None (extractive only)"


@dataclass(frozen=True)
class Services:
    db: DatabaseManager
    ingestor: DocumentIngestor
    indexer: DocumentIndexer
    summarizer: Summarizer
    answerer: QuestionAnswerer
    comparator: DocumentComparator
    insights: InsightsEngine
    ollama: OllamaClient


@st.cache_resource
def get_services() -> Services:
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
    return Services(
        db,
        ingestor,
        indexer,
        Summarizer(db, ollama, settings),
        QuestionAnswerer(db, retriever, engine, ollama, settings),
        DocumentComparator(db, indexer, ollama, settings),
        InsightsEngine(db, settings),
        ollama,
    )


@st.cache_data(ttl=_STATUS_TTL_SECONDS, show_spinner=False)
def get_ollama_status() -> OllamaStatus:
    return get_services().ollama.status()


def select_model() -> str | None:
    """Let the user pick an installed model. None means extractive mode."""
    status = get_ollama_status()
    st.sidebar.caption(status.message)
    if not status.models:
        return None
    options = [*status.models, _NO_LLM]
    default = settings.ollama.default_model
    index = options.index(default) if default in options else len(options) - 1
    if default not in options:
        st.sidebar.warning(
            f"The default model {default} is not installed. Choose a model, or run: "
            f"ollama pull {default}"
        )
    choice = st.sidebar.selectbox("Local model", options, index=index)
    return None if choice == _NO_LLM else choice


def main() -> None:
    st.set_page_config(page_title=APP_NAME, layout="wide")
    st.title(APP_NAME)
    st.caption(APP_TAGLINE)
    services = get_services()
    page = st.sidebar.radio("Page", ["Documents", "Summarize", "Ask", "Compare", "Insights"])
    model = select_model()
    if page == "Documents":
        upload_page.render(
            services.db,
            services.ingestor,
            services.indexer,
            settings.ingestion.max_file_size_mb,
        )
    elif page == "Summarize":
        summary_page.render(services.db, services.summarizer, model)
    elif page == "Ask":
        chat_page.render(services.db, services.answerer, model)
    elif page == "Compare":
        comparison_page.render(services.db, services.comparator, model)
    else:
        insights_page.render(services.db, services.insights)


if __name__ == "__main__":
    main()
