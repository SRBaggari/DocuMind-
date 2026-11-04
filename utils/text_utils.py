"""Pure text measurements. Sentence splitting and token estimation arrive in Phase 3."""

from __future__ import annotations

import re

# Chinese and Japanese do not separate words with spaces, so each character counts as one.
_UNSPACED_SCRIPT = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def count_words(text: str) -> int:
    total = 0
    for token in text.split():
        unspaced = len(_UNSPACED_SCRIPT.findall(token))
        remainder = _UNSPACED_SCRIPT.sub("", token) if unspaced else token
        total += unspaced + (1 if any(ch.isalnum() for ch in remainder) else 0)
    return total


def count_paragraphs(text: str) -> int:
    return sum(1 for part in _PARAGRAPH_BREAK.split(text) if part.strip())
