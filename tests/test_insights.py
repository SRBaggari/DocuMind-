"""Tests for document statistics and extracted insights."""

from __future__ import annotations

from core.extractive import Sentence
from core.insights import (
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
