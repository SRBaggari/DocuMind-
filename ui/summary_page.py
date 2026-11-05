"""Summarization page."""

from __future__ import annotations

import streamlit as st

from core.exporter import safe_filename
from core.extractive import GenerationMethod
from core.summarizer import Summary, SummaryError, SummaryMode
from database.schema import DocumentStatus
from ui import components as ui

_RESULT_KEY = "summary_result"
_MODE_LABELS = {
    SummaryMode.QUICK: "Quick",
    SummaryMode.DETAILED: "Detailed",
    SummaryMode.EXECUTIVE: "Executive",
}


def _show(summary: Summary) -> None:
    llm = summary.method is GenerationMethod.LLM
    ui.pill(f"Local LLM: {summary.model}" if llm else "Extractive NLP", "accent" if llm else "warn")
    st.caption(summary.method_label)
    for notice in summary.notices:
        st.warning(ui.safe_markdown(notice))
    for index, section in enumerate(summary.sections):
        with ui.card(f"section-{index}"):
            ui.section_title(section.title)
            st.markdown(ui.safe_markdown(section.content))
    name = f"{safe_filename(summary.filename)}-{summary.mode.value}-summary"
    left, right, _ = st.columns([1, 1, 2])
    left.download_button(
        "Download Markdown",
        summary.to_markdown(),
        file_name=f"{name}.md",
        icon=":material/download:",
        width="stretch",
    )
    right.download_button(
        "Download text",
        summary.to_text(),
        file_name=f"{name}.txt",
        icon=":material/download:",
        width="stretch",
    )


def render(ctx: ui.AppContext) -> None:
    ui.page_header("Summarization", "Create a quick, detailed or executive summary.")
    documents = ctx.db.list_by_status(DocumentStatus.READY)
    if not documents:
        ui.no_documents()
        return
    ui.extractive_notice(ctx)

    # Summaries can take minutes, so coming back to this page must show the finished one.
    previous = st.session_state.get(_RESULT_KEY)
    with ui.card("controls"):
        left, right = st.columns([3, 2], vertical_alignment="bottom")
        with left:
            document_id = ui.document_select(
                "Document",
                documents,
                "summary_document",
                previous.document_id if previous else None,
            )
        mode = right.segmented_control(
            "Summary type",
            list(SummaryMode),
            default=previous.mode if previous else SummaryMode.QUICK,
            format_func=_MODE_LABELS.get,
            key="summary_mode",
        ) or SummaryMode.QUICK
        generate = st.button("Generate summary", type="primary", icon=":material/auto_awesome:")

    if generate:
        progress = st.progress(0.0, text="Summarizing. Long documents can take several minutes.")
        try:
            summary = ctx.services.summarizer.summarize(
                document_id,
                mode,
                model=ctx.model,
                use_llm=ctx.model is not None,
                progress=lambda share: progress.progress(share, text="Summarizing"),
            )
            st.session_state[_RESULT_KEY] = summary
            ctx.db.log_activity(
                "summary", f"Summarized {summary.filename} ({mode.value})", document_id
            )
        except SummaryError as exc:
            st.session_state.pop(_RESULT_KEY, None)
            st.error(str(exc))
        progress.empty()

    result = st.session_state.get(_RESULT_KEY)
    if result and (result.document_id, result.mode) == (document_id, mode):
        _show(result)
