"""Tests for keyword, keyphrase and topic extraction."""

from __future__ import annotations

import pytest

from core.chunker import Chunk
from core.keyword_extractor import (
    compare_keywords,
    extract_keyphrases,
    extract_keywords,
    identify_topics,
)

FINANCE = [
    "Quarterly revenue growth improved operating margin and investor returns.",
    "The revenue forecast raised the operating margin target for investors.",
    "Investors welcomed revenue growth and stronger margin guidance.",
]
BIOLOGY = [
    "Cell membranes regulate protein transport inside the mitochondria.",
    "Mitochondria produce energy while protein synthesis occurs in the cell.",
    "Protein folding inside the cell depends on membrane chaperones.",
]


def chunks(texts: list[str]) -> list[Chunk]:
    return [Chunk(f"c{n}", "doc", text, n + 1, n, 10) for n, text in enumerate(texts)]


def test_keywords_rank_important_terms_and_drop_stop_words() -> None:
    keywords = extract_keywords(FINANCE, 5)
    terms = [keyword.term for keyword in keywords]
    assert len(keywords) == 5 and "revenue" in terms[:2]
    assert not {"the", "and", "for"} & set(terms)
    scores = [keyword.score for keyword in keywords]
    assert scores == sorted(scores, reverse=True)


def test_keyword_counts_are_real_frequencies() -> None:
    by_term = {keyword.term: keyword for keyword in extract_keywords(FINANCE, 50)}
    assert by_term["revenue"].count == 3


def test_numbers_are_not_keywords() -> None:
    assert [k.term for k in extract_keywords(["2024 1234 revenue"], 5)] == ["revenue"]


def test_max_keywords_is_respected() -> None:
    assert len(extract_keywords(FINANCE, 3)) == 3


def test_keyphrases_skip_overlapping_fragments() -> None:
    phrases = [keyword.term for keyword in extract_keyphrases(FINANCE, 5)]
    assert phrases and all(len(phrase.split()) >= 2 for phrase in phrases)
    for index, phrase in enumerate(phrases):
        assert not any(phrase in other for other in phrases[index + 1 :])


def test_keyphrases_only_join_adjacent_words() -> None:
    text = ["Revenue fell by 3 percent. Kilometres of pipe. Smart meter rollout, smart meter data."]
    phrases = [keyword.term for keyword in extract_keyphrases(text, 10)]
    assert not {"fell percent", "kilometres pipe", "pipe smart"} & set(phrases)
    assert "smart meter" in " ".join(phrases)


def test_generic_units_are_not_keywords() -> None:
    terms = [k.term for k in extract_keywords(["Million dollars percent turbine turbine"], 5)]
    assert terms == ["turbine"]


def test_empty_or_stop_word_text_gives_nothing() -> None:
    assert extract_keywords([], 5) == []
    assert extract_keywords(["", "  "], 5) == []
    assert extract_keywords(["the and of to"], 5) == []
    assert extract_keyphrases(["the and"], 5) == []


def test_compare_keywords_separates_documents() -> None:
    by_term = {item.term: item for item in compare_keywords(FINANCE, BIOLOGY, 40)}
    assert by_term["revenue"].score_a > 0 and by_term["revenue"].score_b == 0
    assert by_term["protein"].score_b > 0 and by_term["protein"].score_a == 0
    assert by_term["revenue"].difference > 0 > by_term["protein"].difference


def test_compare_keywords_with_empty_side() -> None:
    assert compare_keywords(FINANCE, [], 5) == []


def test_topics_separate_subjects_and_cite_sections() -> None:
    topics = identify_topics(chunks(FINANCE + BIOLOGY), 2, 6, 3)
    assert len(topics) == 2
    groups = [{term for term, _ in topic.terms} for topic in topics]
    assert any("revenue" in group and "protein" not in group for group in groups)
    assert any("protein" in group and "revenue" not in group for group in groups)
    assert sum(topic.share for topic in topics) == pytest.approx(1.0)
    for topic in topics:
        assert topic.label.count(",") == 2
        assert topic.sections and topic.sections[0].page_number is not None


def test_topics_are_repeatable() -> None:
    data = chunks(FINANCE + BIOLOGY)
    assert identify_topics(data, 2, 6, 3) == identify_topics(data, 2, 6, 3)


def test_too_little_text_gives_no_topics() -> None:
    assert identify_topics(chunks(FINANCE[:2]), 3, 5, 2) == []
    assert identify_topics(chunks(["the and", "of to", "a an"]), 3, 5, 2) == []
