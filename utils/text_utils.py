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
    r"previous)\b"
    r"|\bsystem prompt\b"
    r"|\bnew instructions?\s*:"
    r"|\bnew task\s*(?:from|:)"
    r"|\byou are now (?:an? |the )?(?:ai|assistant|chatbot|model|unrestricted|jailbroken)\b"
    # A short role change, such as "You are now a pirate."
    r"|^\W*you are now (?:an? |the )\w+\W*$"
    r"|\b(?:assistant|chatbot|language model|ai model)\s*[,:]\s*"
    r"(?:you|please|ignore|reply|respond|answer|it is)\b"
    r"|\byour only (?:reply|response|answer)\b"
    # A command at the start of a sentence, or a request put to "you".
    r"|^\W*(?:reply|respond|answer|say|print|output)\s+(?:only\b|with\b|the word\b|exactly\b)"
    r"|\b(?:please|you must|you should|you are to|that you)\s+(?:reply|respond|answer|say|print|"
    r"output)\b.{0,50}\b(?:only|nothing else|every question|any question|single word|exactly)\b",
    re.IGNORECASE,
)
_BULLET_ONLY = re.compile(r"\s*[•▪◦·∙‣●○■□\-–*]*\s*")
_LETTER_SPACED = re.compile(r"(?:\b[A-Z]{1,2} ){4,}")
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")
# Case matters here: "System: Windows 11" is a fact, and "SYSTEM:" imitates a chat role.
_ROLE_PREFIX = re.compile(r"^\W*(?:SYSTEM|ASSISTANT|DEVELOPER)\s*:")
_UNIT_DIGIT = re.compile(
    r"[$€£]\s?(\d)(?![\d,]|\.\d)|(?<![\w.,])(\d)\s?(?:%|percent\b|thousand\b|million\b|billion\b)"
)
_SIGNED_NUMBER = re.compile(
    r"(?<![\w.,])(?P<sign>[<>≤≥]\s?|[-−+](?=[$€£\d]))?(?P<currency>[$€£]\s?)?"
    r"(?P<value>\d[\d,]*(?:\.\d+)?)(?P<unit>\s?(?:%|percent\b))?"
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


def number_tokens(text: str) -> list[str]:
    """List figures in order with their sign, currency and percent, in one standard form."""
    tokens = []
    for match in _SIGNED_NUMBER.finditer(text):
        value = match["value"].replace(",", "").rstrip(".")
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        sign = (match["sign"] or "").replace("−", "-").strip()
        currency = (match["currency"] or "").strip()
        tokens.append(f"{sign}{currency}{value}{'%' if match['unit'] else ''}")
    return tokens


def tidy_evidence(text: str) -> str:
    """Drop layout debris from PDF text: lone bullets and letter-spaced headings."""
    lines = [
        line
        for line in text.split("\n")
        if not line.strip() or not (_BULLET_ONLY.fullmatch(line) or _LETTER_SPACED.search(line))
    ]
    return _EXTRA_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def mask_numbers(text: str) -> str:
    """Hide figures, so sentences that differ only in their numbers look identical."""
    return _NUMBER.sub("#", text)


def unverified_numbers(generated: str, source: str) -> list[str]:
    """List figures in generated text that never appear in the source.

    A bare single digit is skipped, because models use those for list items and for
    numbers the source spells out. With a unit, such as "7%", it is checked.
    """
    known = extract_numbers(source)
    invented = {n for n in extract_numbers(generated) - known if len(n) > 1}
    with_unit = {match[1] or match[2] for match in _UNIT_DIGIT.finditer(generated)}
    return sorted(invented | (with_unit - known))


def strip_instructions(text: str) -> tuple[str, bool]:
    """Remove sentences that look like commands to the model, however they are phrased."""
    removed = False
    lines = []
    for line in text.split("\n"):
        kept = []
        for sentence in split_sentences(line):
            if _INJECTION.search(sentence) or _ROLE_PREFIX.search(sentence):
                removed = True
            else:
                kept.append(sentence)
        lines.append(" ".join(kept))
    return "\n".join(lines), removed
