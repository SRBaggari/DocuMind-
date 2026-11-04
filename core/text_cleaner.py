"""Normalise extracted text without changing its meaning or stripping non-English scripts."""

from __future__ import annotations

import re
import unicodedata

_LIGATURES = str.maketrans(
    {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
)
# Zero-width joiners change how Persian and Indic text is written, and the marks
# below set right-to-left text direction, so these format characters are kept.
_KEPT_FORMAT_CHARS = frozenset("‌‍‎‏")
_KEPT_CONTROL_CHARS = frozenset("\n\t")

_LINE_END_HYPHEN = re.compile(r"(?<=[^\W\d_])-\n(?=[^\W\d_A-Z])")
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def _is_removable(char: str) -> bool:
    category = unicodedata.category(char)
    if category == "Cc":
        return char not in _KEPT_CONTROL_CHARS
    if category == "Cf":
        return char not in _KEPT_FORMAT_CHARS
    return False


def unwrap_lines(block: str) -> str:
    """Join the wrapped lines of one PDF text block into a single paragraph."""
    return _LINE_END_HYPHEN.sub("", block).replace("\n", " ")


def clean_text(text: str) -> str:
    """Return ``text`` with normalised Unicode, spacing and paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text).translate(_LIGATURES)
    if not text.isprintable():
        text = "".join(ch for ch in text if not _is_removable(ch))
    text = _HORIZONTAL_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()
