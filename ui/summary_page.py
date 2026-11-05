"""Summarization page. A functional page for Phase 4; Phase 6 replaces its layout."""

from __future__ import annotations

import streamlit as st

from core.extractive import GenerationMethod
from core.summarizer import Summarizer, SummaryError, SummaryMode
from database.db_manager import DatabaseManager
from database.schema import DocumentStatus

_RESULT_KEY = "summary_result"
_MODE_LABELS = {
    SummaryMode.QUICK: "Quick",
    SummaryMode.DETAILED: "Detailed",
    SummaryMode.EXECUTIVE: "Executive",
}


def render(db: DatabaseManager, summarizer: Summarizer, model: str | None) -> None:
    st.subheader("Summarize a document")
    documents = db.list_by_status(DocumentStatus.READY)
    if not documents:
        st.caption("Upload a document first.")
        return
    names = {doc.document_id: doc.original_filename for doc in documents}
    document_id = st.selectbox("Document", list(names), format_func=names.get)
    mode = st.radio(
        "Summary type",
        list(SummaryMode),
        format_func=_MODE_LABELS.get,
        horizontal=True,
    )

    if st.button("Generate summary", type="primary"):
        progress = st.progress(0.0, text="Summarizing")
        try:
            st.session_state[_RESULT_KEY] = summarizer.summarize(
                document_id,
                mode,
                model=model,
                use_llm=model is not None,
                progress=lambda share: progress.progress(share, text="Summarizing"),
            )
        except SummaryError as exc:
            st.session_state.pop(_RESULT_KEY, None)
            st.error(str(exc))
        progress.empty()

    summary = st.session_state.get(_RESULT_KEY)
    if summary is None:
        return
    show = st.info if summary.method is GenerationMethod.LLM else st.warning
    show(summary.method_label)
    for notice in summary.notices:
        st.warning(notice)
    for section in summary.sections:
        st.markdown(f"#### {section.title}")
        st.markdown(section.content)
    st.download_button(
        "Download as Markdown",
        summary.to_markdown(),
        file_name=f"{summary.filename}-{summary.mode.value}-summary.md",
    )
