"""Compare Documents page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.document_comparator import (
    NOT_CORRECTNESS_NOTE,
    Comparison,
    ComparisonError,
    StatementDifference,
    TopicOverlap,
    UniquePassage,
    format_percent,
)
from core.exporter import comparison_pdf, safe_filename
from database.schema import DocumentStatus
from ui import charts
from ui import components as ui

_RESULT_KEY = "comparison_result"
_PDF_KEY = "comparison_pdf"
_CHART_KEYWORDS = 12


def _place(name: str, page: int | None) -> str:
    return f"{name}, p. {page}" if page else name


def _show_overview(comparison: Comparison) -> None:
    scores = comparison.scores
    ui.stat_cards(
        [
            ("Overall similarity", format_percent(scores.overall), "70% semantic, 30% lexical"),
            ("Semantic", format_percent(scores.semantic), "Similar meaning"),
            ("Lexical", format_percent(scores.lexical), "Shared vocabulary"),
            ("Exact wording", format_percent(scores.exact), "Same five-word runs"),
        ]
    )
    st.caption(NOT_CORRECTNESS_NOTE)
    for notice in comparison.notices:
        st.warning(ui.safe_markdown(notice))
    with ui.card("summary"):
        ui.section_title("Summary")
        st.caption("Calculated from the results. No AI model wrote this.")
        ui.quote(comparison.summary)
    if comparison.explanation:
        with ui.card("explanation"):
            ui.section_title("Explanation")
            ui.pill(f"Local LLM: {comparison.explanation_model}", "accent")
            st.caption("Check this against the calculated results.")
            st.markdown(ui.safe_markdown(comparison.explanation))


def _topic_column(
    key: str, title: str, topics: tuple[TopicOverlap, ...], keywords: list[str]
) -> None:
    with ui.card(key):
        ui.section_title(title)
        for topic in topics:
            st.markdown(f"- {ui.safe_markdown(topic.label)}")
        if keywords:
            st.caption("Keywords: " + ui.safe_markdown(", ".join(keywords[:_CHART_KEYWORDS])))
        elif not topics:
            st.caption("None found.")


def _show_topics(comparison: Comparison) -> None:
    columns = st.columns(3)
    with columns[0]:
        _topic_column(
            "topics-common", "Common topics", comparison.common_topics, comparison.common_keywords
        )
    with columns[1]:
        _topic_column(
            "topics-a",
            f"Mainly in {comparison.filename_a}",
            comparison.unique_topics_a,
            comparison.unique_keywords_a,
        )
    with columns[2]:
        _topic_column(
            "topics-b",
            f"Mainly in {comparison.filename_b}",
            comparison.unique_topics_b,
            comparison.unique_keywords_b,
        )
    st.caption("Topic names are generated from important terms and are not reviewed labels.")


def _show_difference(index: int, item: StatementDifference, comparison: Comparison) -> None:
    with ui.card(f"difference-{index}"):
        in_a = ", ".join(item.only_a) or "nothing"
        in_b = ", ".join(item.only_b) or "nothing"
        ui.pill(item.kind.value, "warn")
        st.caption(ui.safe_markdown(f"Document A has {in_a}. Document B has {in_b}."))
        left, right = st.columns(2)
        with left:
            ui.quote(item.text_a, _place(comparison.filename_a, item.page_a))
        with right:
            ui.quote(item.text_b, _place(comparison.filename_b, item.page_b))


def _unique_column(name: str, passages: tuple[UniquePassage, ...]) -> None:
    ui.section_title(f"Only in {name}")
    if not passages:
        st.caption("No passages stand out as missing from the other document.")
    for passage in passages:
        ui.quote(
            passage.text,
            f"{_place(name, passage.page)} · closest match {passage.best_score:.2f}",
        )


def _show_differences(comparison: Comparison) -> None:
    if not comparison.differences:
        st.caption("No matching statements with different content were found.")
    for index, item in enumerate(comparison.differences):
        _show_difference(index, item, comparison)
    left, right = st.columns(2)
    with left:
        _unique_column(comparison.filename_a, comparison.unique_passages_a)
    with right:
        _unique_column(comparison.filename_b, comparison.unique_passages_b)


def _show_side_by_side(comparison: Comparison) -> None:
    if not comparison.similar_passages:
        st.caption("No passages are similar enough to list.")
    for index, passage in enumerate(comparison.similar_passages):
        with ui.card(f"similar-{index}"):
            ui.pill(f"{passage.kind.value} · {passage.score:.2f}", "accent")
            left, right = st.columns(2)
            with left:
                ui.quote(passage.text_a, _place(comparison.filename_a, passage.page_a))
            with right:
                ui.quote(passage.text_b, _place(comparison.filename_b, passage.page_b))


def _show_charts(comparison: Comparison) -> None:
    name_a, name_b = comparison.filename_a, comparison.filename_b
    keywords = comparison.keywords[:_CHART_KEYWORDS]
    if keywords:
        ui.section_title("Keyword importance")
        st.plotly_chart(charts.keyword_comparison(keywords, name_a, name_b))
        with st.expander("Keyword table"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Keyword": k.term, name_a: round(k.score_a, 3), name_b: round(k.score_b, 3)}
                        for k in keywords
                    ]
                ),
                hide_index=True,
            )
    if comparison.grid:
        ui.section_title("Where the documents match")
        st.caption("Each cell shows the strongest passage match between two parts.")
        st.plotly_chart(charts.similarity_heatmap(comparison.grid, name_a, name_b))


def _show(comparison: Comparison) -> None:
    overview, topics, differences, side_by_side, chart_tab = st.tabs(
        ["Overview", "Topics", "Differences", "Side by side", "Charts"]
    )
    with overview:
        _show_overview(comparison)
    with topics:
        _show_topics(comparison)
    with differences:
        _show_differences(comparison)
    with side_by_side:
        _show_side_by_side(comparison)
    with chart_tab:
        _show_charts(comparison)
    if _PDF_KEY not in st.session_state:
        st.session_state[_PDF_KEY] = comparison_pdf(comparison)
    st.download_button(
        "Download PDF report",
        st.session_state[_PDF_KEY],
        file_name=(
            f"{safe_filename(comparison.filename_a)}-vs-"
            f"{safe_filename(comparison.filename_b)}.pdf"
        ),
        mime="application/pdf",
        icon=":material/download:",
    )


def render(ctx: ui.AppContext) -> None:
    ui.page_header("Compare Documents", "See what two documents share and where they differ.")
    documents = ctx.db.list_by_status(DocumentStatus.READY)
    if len(documents) < 2:
        ui.no_documents("Upload at least two documents to compare them.")
        return

    with ui.card("controls"):
        left, right = st.columns(2)
        with left:
            document_a = ui.document_select("Document A", documents, "compare_a")
        with right:
            others = [doc for doc in documents if doc.document_id != document_a]
            document_b = ui.document_select("Document B", others, f"compare_b_{document_a}", 0)
        run = st.button("Compare", type="primary", icon=":material/compare_arrows:")

    if run:
        try:
            with st.spinner("Comparing the documents"):
                comparison = ctx.services.comparator.compare(document_a, document_b, ctx.model)
            st.session_state[_RESULT_KEY] = comparison
            st.session_state.pop(_PDF_KEY, None)
            ctx.db.log_activity(
                "comparison",
                f"Compared {comparison.filename_a} and {comparison.filename_b}",
                document_a,
                document_b,
            )
        except ComparisonError as exc:
            st.session_state.pop(_RESULT_KEY, None)
            st.error(str(exc))

    result = st.session_state.get(_RESULT_KEY)
    if result and (result.document_a, result.document_b) == (document_a, document_b):
        _show(result)
