"""Tests for file exports and charts."""

from __future__ import annotations

import csv
import io
import json

import pymupdf
import pytest
from conftest import FakeLLM

from core.document_comparator import DocumentComparator
from core.exporter import comparison_pdf, insights_json, keywords_csv, safe_filename
from core.insights import InsightsEngine
from core.keyword_extractor import Keyword

REPORT = (
    "The Riverside Water Authority serves 410,000 residents across three counties.\n\n"
    "Operating revenue reached 148 million dollars on 1 July 2024, up 3.1 percent.\n\n"
    "Engineers recommend extending sensors across the network by 2028."
)


@pytest.fixture
def insights(db, test_settings, ingestor):
    document = ingestor.ingest("report.txt", REPORT.encode()).document
    return InsightsEngine(db, test_settings).analyze(document.document_id)


def test_csv_neutralizes_spreadsheet_formulas() -> None:
    text = keywords_csv([Keyword("=cmd()", 0.5, 2), Keyword("water", 0.25, 1)], [Keyword("a b", 0.1, 1)])
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["term"] == "'=cmd()" and rows[0]["type"] == "keyword"
    assert rows[1] == {"term": "water", "type": "keyword", "tfidf_score": "0.2500", "count": "1"}
    assert rows[2]["type"] == "keyphrase"


def test_insights_json_round_trips(insights) -> None:
    data = json.loads(insights_json(insights))
    assert data["filename"] == "report.txt"
    assert data["statistics"]["words"] == insights.statistics.words
    assert data["names"][0]["text"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [("report.pdf", "report.pdf"), ("../../etc/pass wd", "etc_pass_wd"), ("***", "document")],
)
def test_safe_filename(name, expected) -> None:
    assert safe_filename(name) == expected


def test_pdf_report_opens_and_survives_markup_in_documents(
    db, indexer, indexed_ingestor, test_settings
) -> None:
    first = "Revenue <b>reached & grew</b> to 148 million dollars for the whole year in Århus."
    second = first.replace("148", "152")
    a = indexed_ingestor.ingest("a<1>.txt", first.encode()).document.document_id
    b = indexed_ingestor.ingest("b.txt", second.encode()).document.document_id
    comparison = DocumentComparator(db, indexer, FakeLLM(), test_settings).compare(a, b)
    data = comparison_pdf(comparison)
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        text = " ".join(page.get_text() for page in pdf)
    joined = " ".join(text.split())
    assert "Document comparison report" in joined
    assert "<b>reached" in joined and "Århus" in joined
    assert "does not show that either one is correct" in joined
