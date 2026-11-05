"""Tests for document statistics and extracted insights."""

from __future__ import annotations

import pytest

from core.extractive import Sentence
from core.insights import (
    InsightsEngine,
    InsightsError,
    find_dates,
    find_names,
    find_statistics,
)

REPORT = (
    "The Riverside Water Authority serves 410,000 residents across three counties.\n\n"
    "Operating revenue reached 148 million dollars on 1 July 2024, up 3.1 percent.\n\n"
    "The board led by Maria Lopez approved a plan in March 2024.\n\n"
    "Engineers recommend extending sensors across the network by 2028.\n\n"
    "In conclusion the authority remains financially stable."
)


def sentences(*texts: str, page: int | None = 2) -> list[Sentence]:
    return [Sentence(text, page, index) for index, text in enumerate(texts)]


@pytest.fixture
def insights(db, test_settings, ingestor):
    document = ingestor.ingest("report.txt", REPORT.encode()).document
    return InsightsEngine(db, test_settings).analyze(document.document_id)


def test_statistics_match_the_document(insights, db) -> None:
    document = db.get_document(insights.document_id)
    stats = insights.statistics
    assert stats.words == document.word_count and stats.characters == document.char_count
    assert stats.paragraphs == 5 and stats.pages is None
    assert stats.keywords == len(insights.keywords) and stats.chunks >= 1
    assert stats.reading_minutes == 1


def test_reading_time_rounds_up(db, test_settings, ingestor) -> None:
    text = " ".join(["word"] * 201) + " end of the text."
    document = ingestor.ingest("long.txt", text.encode()).document
    assert InsightsEngine(db, test_settings).analyze(document.document_id).statistics.reading_minutes == 2


def test_insights_are_copied_from_the_document(insights) -> None:
    assert "Riverside Water Authority" in [name.text for name in insights.names]
    assert "Maria Lopez" in [name.text for name in insights.names]
    assert {"1 July 2024", "March 2024", "2028"} <= {date.text for date in insights.dates}
    values = [item.value for item in insights.statistics_found]
    assert "148 million dollars" in values and "3.1 percent" in values
    assert any("stable" in item.text for item in insights.conclusions)
    assert any("recommend" in item.text for item in insights.action_items)
    for item in insights.statistics_found:
        assert item.sentence in REPORT


def test_sentence_start_words_are_not_names() -> None:
    found = find_names(sentences("Engineers met Dana Reyes in Oslo.", "Revenue grew in Oslo."))
    assert "Engineers" not in [m.text for m in found]
    assert "Revenue" not in [m.text for m in found]
    assert {"Dana Reyes", "Oslo"} <= {m.text for m in found}


def test_months_alone_are_not_names() -> None:
    assert not find_names(sentences("The plan started in March and ended in May."))


def test_mentions_count_and_record_pages() -> None:
    found = find_names(sentences("We met Dana Reyes.", "Then Dana Reyes left."))
    assert (found[0].text, found[0].count, found[0].pages) == ("Dana Reyes", 2, (2,))


def test_dates_in_several_formats() -> None:
    found = {m.text for m in find_dates(sentences("Due 2024-03-01, 3/4/2025, Q2 2023, June 5, 2022."))}
    assert {"2024-03-01", "3/4/2025", "Q2 2023", "June 5, 2022"} <= found


def test_plain_small_numbers_are_not_statistics() -> None:
    assert not find_statistics(sentences("There are 3 options in section 12 of the plan."))


def test_currency_and_percent_statistics() -> None:
    values = {s.value for s in find_statistics(sentences("It cost $4.5 million, or 12% of 7,500 units."))}
    assert {"$4.5 million", "12%", "7,500 units"} <= values


def test_document_without_text_is_refused(db, test_settings) -> None:
    with pytest.raises(InsightsError):
        InsightsEngine(db, test_settings).analyze("missing")


def test_themes_do_not_depend_on_page_layout(db, test_settings, ingestor) -> None:
    text = "\n\n".join(
        f"{subject} " * 3 + f"The {subject} team reviewed the {detail} during {season} and wrote "
        f"a detailed note about {detail} for the regional {subject} office to read later on."
        for subject, detail, season in (
            ("water", "reservoir levels", "summer"),
            ("budget", "capital spending", "autumn"),
            ("safety", "inspection records", "winter"),
            ("transport", "rail timetables", "spring"),
            ("housing", "rental contracts", "summer"),
            ("energy", "solar panels", "autumn"),
            ("health", "clinic waiting lists", "winter"),
            ("schools", "exam timetables", "spring"),
            ("parks", "tree planting", "summer"),
            ("libraries", "opening hours", "autumn"),
            ("waste", "recycling routes", "winter"),
            ("policing", "patrol staffing", "spring"),
        )
    )
    document = ingestor.ingest("flat.txt", text.encode()).document
    assert InsightsEngine(db, test_settings).analyze(document.document_id).themes


def test_short_document_still_gives_insights(db, test_settings, ingestor) -> None:
    document = ingestor.ingest("short.txt", b"Only one short sentence here.").document
    result = InsightsEngine(db, test_settings).analyze(document.document_id)
    assert result.themes == () and result.statistics.words == 5
