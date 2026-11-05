"""Insights page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.exporter import insights_json, keywords_csv, safe_filename
from core.insights import DocumentInsights, Evidence, InsightsError, Mention
from database.schema import DocumentStatus
from ui import charts
from ui import components as ui

_RESULT_KEY = "insights_result"
_CHART_TERMS = 15


def _show_statistics(insights: DocumentInsights) -> None:
    stats = insights.statistics
    ui.stat_cards(
        [
            ("Words", f"{stats.words:,}"),
            ("Characters", f"{stats.characters:,}"),
            ("Pages", "N/A" if stats.pages is None else f"{stats.pages:,}"),
            ("Paragraphs", f"{stats.paragraphs:,}"),
            ("Chunks", f"{stats.chunks:,}"),
            ("Keywords", f"{stats.keywords:,}"),
            ("Reading time", f"{stats.reading_minutes} min"),
        ]
    )


def _mention_table(title: str, mentions: tuple[Mention, ...]) -> None:
    ui.section_title(title)
    if not mentions:
        st.caption("None found.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    title: mention.text,
                    "Mentions": mention.count,
                    "Pages": ", ".join(map(str, mention.pages)),
                }
                for mention in mentions
            ]
        ),
        hide_index=True,
    )


def _evidence_card(key: str, title: str, items: tuple[Evidence, ...], empty: str) -> None:
    with ui.card(key):
        ui.section_title(title)
        if not items:
            st.caption(empty)
        for item in items:
            ui.quote(item.text, f"p. {item.page}" if item.page else None)


def _show_themes(insights: DocumentInsights) -> None:
    if insights.themes:
        ui.section_title("Main themes")
        st.plotly_chart(charts.topic_distribution(insights.themes))
        for theme in insights.themes:
            with st.expander(ui.safe_markdown(theme.label)):
                st.caption(ui.safe_markdown(", ".join(term for term, _ in theme.terms)))
                for section in theme.sections:
                    ui.quote(
                        section.snippet,
                        f"p. {section.page_number}" if section.page_number else None,
                    )
        st.caption("Theme names are generated from important terms and are not reviewed labels.")
    else:
        st.caption("There is too little text to identify themes.")

    terms = insights.frequent_terms[:_CHART_TERMS]
    if terms:
        ui.section_title("Frequent terms")
        st.plotly_chart(charts.keyword_frequency(terms))
        with st.expander("Keyword table"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Keyword": k.term, "TF-IDF": round(k.score, 3), "Uses": k.count}
                        for k in insights.keywords
                    ]
                ),
                hide_index=True,
            )


def _show_extracted(insights: DocumentInsights) -> None:
    left, right = st.columns(2)
    with left:
        _mention_table("Names", insights.names)
        st.caption("Found from capitalization, so they may include places or terms.")
    with right:
        _mention_table("Dates", insights.dates)
    ui.section_title("Numbers and statistics")
    if insights.statistics_found:
        st.dataframe(
            pd.DataFrame(
                [
                    {"Figure": item.value, "Sentence": item.sentence, "Page": item.page}
                    for item in insights.statistics_found
                ]
            ),
            hide_index=True,
        )
    else:
        st.caption("None found.")


def _show_findings(insights: DocumentInsights) -> None:
    with ui.card("sections"):
        ui.section_title("Important sections")
        st.caption("Ranked by how much of the document's key vocabulary each section contains.")
        for section in insights.important_sections:
            ui.quote(section.snippet, section.title)
    _evidence_card("conclusions", "Key conclusions", insights.conclusions, "None found.")
    _evidence_card("actions", "Potential action items", insights.action_items, "None found.")


def _show(insights: DocumentInsights) -> None:
    _show_statistics(insights)
    st.caption("Everything here is copied or calculated from the document. No AI model wrote it.")
    themes, extracted, findings = st.tabs(["Themes and keywords", "Names and figures", "Findings"])
    with themes:
        _show_themes(insights)
    with extracted:
        _show_extracted(insights)
    with findings:
        _show_findings(insights)

    name = safe_filename(insights.filename)
    left, right, _ = st.columns([1, 1, 2])
    left.download_button(
        "Keywords CSV",
        keywords_csv(insights.keywords, insights.keyphrases),
        file_name=f"{name}-keywords.csv",
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
    )
    right.download_button(
        "Insights JSON",
        insights_json(insights),
        file_name=f"{name}-insights.json",
        mime="application/json",
        icon=":material/download:",
        width="stretch",
    )


def render(ctx: ui.AppContext) -> None:
    ui.page_header("Insights", "Themes, names, figures and conclusions from one document.")
    documents = ctx.db.list_by_status(DocumentStatus.READY)
    if not documents:
        ui.no_documents()
        return

    with ui.card("controls"):
        previous = st.session_state.get(_RESULT_KEY)
        document_id = ui.document_select(
            "Document", documents, "insights_document", previous.document_id if previous else None
        )
        run = st.button("Analyze document", type="primary", icon=":material/insights:")

    if run:
        try:
            st.session_state[_RESULT_KEY] = ctx.services.insights.analyze(document_id)
            name = st.session_state[_RESULT_KEY].filename
            ctx.db.log_activity("insights", f"Analyzed {name}", document_id)
        except InsightsError as exc:
            st.session_state.pop(_RESULT_KEY, None)
            st.error(str(exc))

    insights = st.session_state.get(_RESULT_KEY)
    if insights and insights.document_id == document_id:
        _show(insights)
