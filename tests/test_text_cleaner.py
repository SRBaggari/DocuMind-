"""Tests for text cleaning and word counting."""

from __future__ import annotations

import pytest

from core.text_cleaner import clean_text, unwrap_lines
from utils.text_utils import count_paragraphs, count_words


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a   b\t c", "a b c"),
        ("a b", "a b"),
        ("one\r\ntwo\rthree", "one\ntwo\nthree"),
        ("a\n\n\n\n\nb", "a\n\nb"),
        ("a  \n   \n  b", "a\n\nb"),
        ("x\x00y\x07z\x1b", "xyz"),
        ("soft­hyphen​zero", "softhyphenzero"),
        ("oﬃce ﬁle", "office file"),
        ("café", "café"),
        ('Costs $3.50 (approx.) - "final"!', 'Costs $3.50 (approx.) - "final"!'),
    ],
)
def test_clean_text(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    "text",
    ["می‌خواهم", "क्‍ष", "नमस्ते दुनिया", "日本語のテキスト", "مرحبا بالعالم"],
)
def test_non_latin_text_is_unchanged(text: str) -> None:
    assert clean_text(text) == text


def test_unwrap_lines_joins_hyphenated_words_only() -> None:
    assert unwrap_lines("exam-\nple text\nwraps") == "example text wraps"
    assert unwrap_lines("Covid-\n19 and North-\nSouth") == "Covid- 19 and North- South"


def test_count_words() -> None:
    assert count_words("Hello, big world - 42 | ok") == 5
    assert count_words("日本語 text") == 4
    assert count_words("") == 0


def test_count_paragraphs() -> None:
    assert count_paragraphs("one\nstill one\n\ntwo\n\n \n\nthree") == 3
