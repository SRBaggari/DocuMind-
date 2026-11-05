"""Reusable interface pieces and shared page state.

Document and model text is untrusted. Anything built as HTML here is escaped, and
``safe_markdown`` removes images and links, because a browser would fetch an image from
its URL and that request could carry document text off this computer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path

import streamlit as st

from config import PROJECT_ROOT
from core.document_comparator import DocumentComparator
from core.embedding_engine import EmbeddingEngine
from core.indexing import DocumentIndexer
from core.ingestion import DocumentIngestor
from core.insights import InsightsEngine
from core.ollama_client import OllamaClient, OllamaStatus
from core.question_answering import QuestionAnswerer
from core.summarizer import Summarizer
from core.vector_store import VectorStore
from database.db_manager import DatabaseManager
from database.schema import DocumentRecord

AVATARS = {"user": ":material/person:", "assistant": ":material/description:"}
PAGE_KEY = "page"
ACTIVE_DOCUMENT_KEY = "active_document"
OPEN_DOCUMENT_KEY = "open_document"

_STYLES_PATH = PROJECT_ROOT / "assets" / "styles.css"
_ACTIVE_MARKDOWN = re.compile(r"([\\\[\]!<>`])")
_AUTOLINK = re.compile(r"\b(https?|ftp|www)(?=[:.])|(?=@)", re.IGNORECASE)
_NO_WIDTH = "\u200b"
_STAT_COLORS = 4

LOGO = (
    '<svg width="30" height="30" viewBox="0 0 30 30" fill="none" aria-hidden="true">'
    '<rect width="30" height="30" rx="8" fill="#6366f1"/>'
    '<path d="M9 7.5h8l4.5 4.5v10.5H9z" fill="#fff"/>'
    '<path d="M17 7.5V12h4.5" fill="#c7cbfb"/>'
    '<path d="M12 16h6M12 19h4" stroke="#6366f1" stroke-width="1.5" stroke-linecap="round"/>'
    "</svg>"
)


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
    engine: EmbeddingEngine
    store: VectorStore


@dataclass(frozen=True)
class AppContext:
    services: Services
    status: OllamaStatus
    # The local model to use. None means extractive mode.
    model: str | None
    max_file_size_mb: int

    @property
    def db(self) -> DatabaseManager:
        return self.services.db


def navigate(page: str, document_id: str | None = None) -> None:
    """Go to a page, optionally with a document selected there. Use as an on_click callback."""
    st.session_state[PAGE_KEY] = page
    st.session_state.pop(OPEN_DOCUMENT_KEY, None)
    st.query_params["page"] = page
    if document_id:
        st.session_state[ACTIVE_DOCUMENT_KEY] = document_id


def open_document(document_id: str) -> None:
    navigate("documents", document_id)
    st.session_state[OPEN_DOCUMENT_KEY] = document_id


def _escape(text: str) -> str:
    """Escape text for one line of HTML.

    A blank line would end the HTML block, and Markdown would then parse the rest of the
    text, including any images in it, so line breaks become ``<br>``.
    """
    return re.sub(r"\r\n?|\n", "<br>", escape(text))


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def inject_styles(path: Path = _STYLES_PATH) -> None:
    _html(f"<style>{path.read_text(encoding='utf-8')}</style>")


def safe_markdown(text: str) -> str:
    """Make untrusted text safe to show as Markdown.

    Escaping beats filtering here: every character that can start an image, link or HTML
    tag is made literal, so no nesting or reference trick can rebuild one. Bold and lists
    still work, and the text reads the same. A zero-width space stops bare addresses
    from turning into links.
    """
    escaped = _ACTIVE_MARKDOWN.sub(r"\\\1", text)
    return _AUTOLINK.sub(lambda match: (match[0] or "") + _NO_WIDTH, escaped)


def brand(tagline: str) -> None:
    _html(
        f'<div class="dm-brand">{LOGO}<div><div class="dm-brand-name">DocuMind</div>'
        f'<div class="dm-brand-tag">{_escape(tagline)}</div></div></div>'
    )


def sidebar_status(model: str | None) -> None:
    label = pill_markup(f"Local LLM: {model}", "accent") if model else pill_markup("Extractive mode")
    _html(
        f'<div class="dm-sidebar-status">{label}'
        "<small>All processing stays on this computer.</small></div>"
    )


def page_header(title: str, subtitle: str) -> None:
    _html(
        f'<div class="dm-page-title" role="heading" aria-level="1">{_escape(title)}</div>'
        f'<div class="dm-page-subtitle">{_escape(subtitle)}</div>'
    )


def hero(title: str, text: str) -> None:
    _html(
        f'<div class="dm-hero"><div class="dm-hero-title">{_escape(title)}</div>'
        f'<div class="dm-hero-text">{_escape(text)}</div></div>'
    )


def section_title(title: str) -> None:
    _html(f'<div class="dm-section-title">{_escape(title)}</div>')


def pill_markup(text: str, tone: str = "neutral") -> str:
    return f'<span class="dm-pill dm-pill-{tone}">{_escape(text)}</span>'


def pill(text: str, tone: str = "neutral") -> None:
    _html(pill_markup(text, tone))


def stat_cards(items: Sequence[tuple[str, str] | tuple[str, str, str]]) -> None:
    """Show a row of labelled figures. Each item is (label, value) or (label, value, hint)."""
    columns = st.columns(len(items))
    for index, (column, item) in enumerate(zip(columns, items, strict=True)):
        hint = f'<div class="dm-stat-hint">{_escape(item[2])}</div>' if len(item) > 2 else ""
        column.markdown(
            f'<div class="dm-stat dm-stat-{index % _STAT_COLORS + 1}">'
            f'<div class="dm-stat-label">{_escape(item[0])}</div>'
            f'<div class="dm-stat-value">{_escape(item[1])}</div>{hint}</div>',
            unsafe_allow_html=True,
        )


def quote(text: str, source: str | None = None) -> None:
    """Show document text exactly as written, with no formatting applied to it."""
    footer = f'<div class="dm-quote-source">{_escape(source)}</div>' if source else ""
    _html(f'<div class="dm-quote">{_escape(text)}</div>{footer}')


def document_row(title: str, meta: str) -> None:
    _html(
        f'<div class="dm-row-title">{_escape(title)}</div>'
        f'<div class="dm-row-meta">{_escape(meta)}</div>'
    )


def empty_state(title: str, text: str) -> None:
    _html(
        f'<div class="dm-empty"><div class="dm-empty-title">{_escape(title)}</div>'
        f"<div>{_escape(text)}</div></div>"
    )


def activity_row(summary: str, when: str) -> None:
    _html(f'<div class="dm-activity"><span>{_escape(summary)}</span><time>{_escape(when)}</time></div>')


def card(key: str):
    """Return a styled container. Each ``key`` must be unique on the page."""
    return st.container(key=f"card-{key}")


def document_select(
    label: str,
    documents: Sequence[DocumentRecord],
    key: str,
    preferred: str | None = None,
) -> str:
    """Show a document picker that starts on ``preferred``, or on the active document."""
    names = {document.document_id: document.original_filename for document in documents}
    ids = list(names)
    start = preferred if preferred in names else st.session_state.get(ACTIVE_DOCUMENT_KEY)
    index = ids.index(start) if start in names else 0
    return st.selectbox(label, ids, index=index, format_func=names.get, key=key)


def no_documents(message: str = "Upload a document to get started.") -> None:
    empty_state("No documents yet", message)
    _, middle, _ = st.columns([2, 1, 2])
    middle.button(
        "Upload documents",
        icon=":material/upload:",
        type="primary",
        width="stretch",
        on_click=navigate,
        args=("upload",),
        key="empty_upload",
    )


def extractive_notice(ctx: AppContext) -> None:
    """Say when no local model is in use, and where to change that."""
    if ctx.model is None:
        left, right = st.columns([5, 1], vertical_alignment="center")
        left.caption(
            "Extractive mode: results are sentences copied from your documents. "
            "Choose a local model in Settings for AI-written results."
        )
        right.button(
            "Settings",
            icon=":material/settings:",
            on_click=navigate,
            args=("settings",),
            key="notice_settings",
        )
