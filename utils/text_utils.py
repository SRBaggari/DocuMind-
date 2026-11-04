"""Pure text helpers: word counts, token estimates and sentence splitting."""

from __future__ import annotations

import math
import re

# Chinese and Japanese do not separate words with spaces, so each character counts as one.
_UNSPACED_SCRIPT = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_BOUNDARY = re.compile(
    r"[.!?\u2026][\"'\u201d\u2019)\]]*\s+|[\u3002\uff01\uff1f\u0964]\s*"
)
_ABBREVIATIONS = frozenset(
    "al approx co corp dept dr e.g etc fig i.e inc jr ltd mr mrs ms no prof sr st vs".split()
)
# Typical subword tokens per English word, and characters per token for unspaced text.
_TOKENS_PER_WORD = 1.3
_CHARS_PER_TOKEN = 4


def count_words(text: str) -> int:
    total = 0
    for token in text.split():
        unspaced = len(_UNSPACED_SCRIPT.findall(token))
        remainder = _UNSPACED_SCRIPT.sub("", token) if unspaced else token
        total += unspaced + (1 if any(ch.isalnum() for ch in remainder) else 0)
    return total


def count_paragraphs(text: str) -> int:
    return sum(1 for part in _PARAGRAPH_BREAK.split(text) if part.strip())


def estimate_tokens(text: str) -> int:
    """Estimate subword tokens without loading a tokenizer."""
    return math.ceil(max(count_words(text) * _TOKENS_PER_WORD, len(text) / _CHARS_PER_TOKEN))


def _ends_with_abbreviation(text: str) -> bool:
    words = text.split()
    if not words:
        return False
    last = words[-1].lstrip("(\"'").rstrip(".")
    return last.lower() in _ABBREVIATIONS or (len(last) == 1 and last.isupper())


def split_sentences(text: str) -> list[str]:
    """Split one line of text into sentences, keeping abbreviations and decimals intact."""
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        if match.group()[0] == ".":
            following = text[match.end() : match.end() + 1]
            if following.islower() or _ends_with_abbreviation(text[start : match.start()]):
                continue
        sentences.append(text[start : match.end()].strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences
