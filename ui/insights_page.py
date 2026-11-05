"""Insights page. A functional page for Phase 5; Phase 6 replaces its layout."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.exporter import insights_json, keywords_csv, safe_filename
from core.insights import DocumentInsights, Evidence, InsightsEngine, InsightsError, Mention
from database.db_manager import DatabaseManager
from database.schema import DocumentStatus
from ui import charts

_RESULT_KEY = "insights_result"
_CHART_TERMS = 15


def _show_statistics(insights: DocumentInsights) -> None:
    stats = insights.statistics
    tiles = (
        ("Words", f"{stats.words:,}"),
        ("Characters", f"{stats.characters:,}"),
        ("Pages", "Not paged" if stats.pages is None else f"{stats.pages:,}"),
        ("Paragraphs", f"{stats.paragraphs:,}"),
        ("Chunks", f"{stats.chunks:,}"),
        ("Keywords", f"{stats.keywords:,}"),
        ("Reading time", f"{stats.reading_minutes} min"),
    )
    for column, (label, value) in zip(st.columns(len(tiles)), tiles):
        column.metric(label, value)


def _mention_table(title: str, mentions: tuple[Mention, ...]) -> None:
    st.markdown(f"**{title}**")
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


def _show_evidence(title: str, items: tuple[Evidence, ...], empty: str) -> None:
    st.markdown(f"#### {title}")
    if not items:
        st.caption(empty)
    for item in items:
        page = f" (p. {item.page})" if item.page else ""
        st.markdown(f"- {item.text}{page}")


def _show(insights: DocumentInsights) -> None:
    _show_statistics(insights)
    st.caption("Everything below is copied or calculated from the document. No AI model wrote it.")

    st.markdown("#### Main themes")
    if insights.themes:
        st.plotly_chart(charts.topic_distribution(insights.themes))
        for theme in insights.themes:
            with st.expander(theme.label):
                st.write(", ".join(term for term, _ in theme.terms))
                for section in theme.sections:
                    page = f"p. {section.page_number}: " if section.page_number else ""
                    st.caption(f"{page}{section.snippet}")
        st.caption("Theme names are generated from important terms and are not reviewed labels.")
    else:
        st.caption("There is too little text to identify themes.")

    st.markdown("#### Frequent terms")
    terms = insights.frequent_terms[:_CHART_TERMS]
    if terms:
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
    else:
        st.caption("No keywords found.")

    left, right = st.columns(2)
    with left:
        _mention_table("Names", insights.names)
        st.caption("Found from capitalization, so review them. They may include places or terms.")
    with right:
        _mention_table("Dates", insights.dates)

    st.markdown("#### Numbers and statistics")
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

    st.markdown("#### Important sections")
    st.caption("Ranked by how much of the document's key vocabulary each section contains.")
    for section in insights.important_sections:
        st.markdown(f"**{section.title}**: {section.snippet}")

    _show_evidence("Key conclusions", insights.conclusions, "No concluding statements found.")
    _show_evidence(
        "Potential action items", insights.action_items, "No recommended actions found."
    )

    name = safe_filename(insights.filename)
    left, right = st.columns(2)
    left.download_button(
        "Download keywords as CSV",
        keywords_csv(insights.keywords, insights.keyphrases),
        file_name=f"{name}-keywords.csv",
        mime="text/csv",
    )
    right.download_button(
        "Download insights as JSON",
        insights_json(insights),
        file_name=f"{name}-insights.json",
        mime="application/json",
    )


def render(db: DatabaseManager, engine: InsightsEngine) -> None:
    st.subheader("Document insights")
    documents = db.list_by_status(DocumentStatus.READY)
    if not documents:
        st.caption("Upload a document first.")
        return
    names = {doc.document_id: doc.original_filename for doc in documents}
    document_id = st.selectbox("Document", list(names), format_func=names.get)

    if st.button("Analyze document", type="primary"):
        try:
            st.session_state[_RESULT_KEY] = engine.analyze(document_id)
        except InsightsError as exc:
            st.session_state.pop(_RESULT_KEY, None)
            st.error(str(exc))

    insights = st.session_state.get(_RESULT_KEY)
    if insights and insights.document_id == document_id:
        _show(insights)
