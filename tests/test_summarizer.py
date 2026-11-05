"""Tests for LLM and extractive summarization."""

from __future__ import annotations

import pytest
from conftest import FakeLLM

from core.extractive import GenerationMethod, document_sentences, rank_sentences, select_diverse
from core.ollama_client import OllamaModelNotFoundError, OllamaUnavailableError
from core.summarizer import NOT_STATED, Summarizer, SummaryError, SummaryMode
from utils.text_utils import unverified_numbers

REPORT = (
    "The purpose of this report is to review water supply in 2024.\n\n"
    "Revenue reached 148 million dollars during the fiscal year.\n\n"
    "Leakage fell from 17.4 percent to 15.9 percent after new sensors were installed.\n\n"
    "Engineers recommend extending sensor coverage across the whole network.\n\n"
    "In conclusion the authority remains financially stable for the coming decade."
).encode()

QUICK_REPLY = "## Overview\nIt covers water.\n\n## Main Purpose\nReview.\n\n## Important Findings\n- Revenue 148"


@pytest.fixture
def document(indexed_ingestor):
    return indexed_ingestor.ingest("report.txt", REPORT).document


def summarizer_with(db, settings, llm) -> Summarizer:
    return Summarizer(db, llm, settings)


def test_extractive_quick_copies_real_sentences(db, test_settings, document) -> None:
    llm = FakeLLM()
    summary = summarizer_with(db, test_settings, llm).summarize(
        document.document_id, SummaryMode.QUICK, use_llm=False
    )
    assert llm.prompts == []
    text = REPORT.decode()
    for section in summary.sections:
        for line in section.content.splitlines():
            assert line.removeprefix("- ") in text
    assert "purpose of this report" in summary.sections[1].content


def test_extractive_executive_finds_statistics_and_actions(db, test_settings, document) -> None:
    summary = summarizer_with(db, test_settings, FakeLLM()).summarize(
        document.document_id, SummaryMode.EXECUTIVE, use_llm=False
    )
    sections = {section.title: section.content for section in summary.sections}
    assert "17.4 percent" in sections["Important Statistics"]
    assert "recommend" in sections["Recommended Next Actions"]
    assert "In conclusion" in sections["Conclusions"]


def test_extractive_does_not_invent_missing_sections(db, test_settings, indexed_ingestor) -> None:
    doc = indexed_ingestor.ingest("a.txt", b"Cats sleep on the warm sofa every single day.").document
    summary = summarizer_with(db, test_settings, FakeLLM()).summarize(
        doc.document_id, SummaryMode.EXECUTIVE, use_llm=False
    )
    sections = {section.title: section.content for section in summary.sections}
    assert sections["Recommended Next Actions"] == NOT_STATED


def test_extractive_detailed_uses_docx_headings(
    db, test_settings, indexed_ingestor, sample_docx
) -> None:
    doc = indexed_ingestor.ingest("a.docx", sample_docx.read_bytes()).document
    summary = summarizer_with(db, test_settings, FakeLLM()).summarize(
        doc.document_id, SummaryMode.DETAILED, use_llm=False
    )
    assert summary.sections[0].title == "Revenue"


def test_extractive_pdf_cites_real_pages(db, test_settings, indexed_ingestor, make_pdf) -> None:
    pages = ["Revenue reached 148 million dollars this year.", "", "Costs fell by 12 percent overall."]
    doc = indexed_ingestor.ingest("a.pdf", make_pdf(pages).read_bytes()).document
    summary = summarizer_with(db, test_settings, FakeLLM()).summarize(
        doc.document_id, SummaryMode.EXECUTIVE, use_llm=False
    )
    statistics = {s.title: s.content for s in summary.sections}["Important Statistics"]
    assert "148 million dollars this year. (p. 1)" in statistics
    assert "12 percent overall. (p. 3)" in statistics


def test_document_without_text_is_refused(db, test_settings, ingestor) -> None:
    doc = ingestor.ingest("blank.txt", b"   ").document
    with pytest.raises(SummaryError):
        summarizer_with(db, test_settings, FakeLLM()).summarize(
            doc.document_id, SummaryMode.QUICK, use_llm=False
        )


def test_unknown_document_is_refused(db, test_settings) -> None:
    with pytest.raises(SummaryError):
        summarizer_with(db, test_settings, FakeLLM()).summarize("a" * 32, SummaryMode.QUICK)


def test_markdown_export_states_the_method(db, test_settings, document) -> None:
    summary = summarizer_with(db, test_settings, FakeLLM()).summarize(
        document.document_id, SummaryMode.QUICK, use_llm=False
    )
    markdown = summary.to_markdown()
    assert "Extractive summary" in markdown and "## Overview" in markdown


def test_select_diverse_skips_near_duplicates() -> None:
    pages = [
        {
            "number": 1,
            "text": "Revenue grew strongly in the northern region.\n"
            "Revenue grew strongly in the southern region.\n"
            "Staff morale improved after the office moved.",
        }
    ]
    chosen = select_diverse(rank_sentences(document_sentences(pages)), 3, min_score=-1.0)
    assert len(chosen) == 2
