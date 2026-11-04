"""DocuMind entry point. Run with: streamlit run app.py

Navigation and the remaining pages are added in Phase 6; this file stays a thin shell.
"""

from __future__ import annotations

import streamlit as st

from config import APP_NAME, APP_TAGLINE, ensure_directories, settings
from core.ingestion import DocumentIngestor
from database.db_manager import DatabaseManager
from ui import upload_page
from utils.logging_config import configure_logging


@st.cache_resource
def get_services() -> tuple[DatabaseManager, DocumentIngestor]:
    """Create the shared services once per server process."""
    ensure_directories(settings.paths)
    configure_logging(settings.paths.logs_dir)
    db = DatabaseManager(settings.paths.database_path)
    ingestor = DocumentIngestor(db, settings)
    ingestor.recover_interrupted()
    return db, ingestor


def main() -> None:
    st.set_page_config(page_title=APP_NAME, layout="wide")
    st.title(APP_NAME)
    st.caption(APP_TAGLINE)
    db, ingestor = get_services()
    upload_page.render(db, ingestor, settings.ingestion.max_file_size_mb)


if __name__ == "__main__":
    main()
