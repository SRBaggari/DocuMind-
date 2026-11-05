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
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
_INJECTION = re.compile(
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|rules?|prompts?|above|"
    r"previous)\b|\byou are now\b|\bsystem prompt\b|\bnew instructions?\b",
    re.IGNORECASE,
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


def extract_numbers(text: str) -> set[str]:
    """Return the figures in ``text`` in one form, so "1,206" and "1206" are equal."""
    return {match.replace(",", "").rstrip(".") for match in _NUMBER.findall(text)}


def mask_numbers(text: str) -> str:
    """Hide figures, so sentences that differ only in their numbers look identical."""
    return _NUMBER.sub("#", text)


def unverified_numbers(generated: str, source: str) -> list[str]:
    """List figures in generated text that never appear in the source.

    Single digits are skipped, because models use them for list items and for
    numbers the source spells out.
    """
    known = extract_numbers(source)
    return sorted(n for n in extract_numbers(generated) - known if len(n) > 1)


def strip_instructions(text: str) -> tuple[str, bool]:
    """Remove sentences that look like commands to the model, however they are phrased."""
    removed = False
    lines = []
    for line in text.split("\n"):
        kept = []
        for sentence in split_sentences(line):
            if _INJECTION.search(sentence):
                removed = True
            else:
                kept.append(sentence)
        lines.append(" ".join(kept))
    return "\n".join(lines), removed
