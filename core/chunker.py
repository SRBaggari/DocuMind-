"""Split page text into overlapping, sentence-aligned chunks.

A chunk never crosses a page, so its page number is always exact. Formats without
pages (DOCX, TXT) are one long page with ``page_number`` set to None.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from config import ChunkingConfig
from utils.text_utils import estimate_tokens, split_sentences


_PASSAGE_PARAGRAPH_RATIO = 0.25


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    page_number: int | None
    chunk_index: int
    token_count: int
    # Smaller pieces of ``text``. Search runs over these, and results return the chunk.
    passages: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Unit:
    """A sentence, with the paragraph and line it came from."""

    text: str
    tokens: int
    paragraph: int
    line: int


def _split_oversized(text: str, limit: int) -> list[str]:
    """Halve text at a space until each piece fits. Only very long sentences reach here."""
    if estimate_tokens(text) <= limit or len(text) < 2:
        return [text]
    middle = len(text) // 2
    cut = text.rfind(" ", 0, middle)
    cut = cut if cut > 0 else middle
    return _split_oversized(text[:cut].strip(), limit) + _split_oversized(
        text[cut:].strip(), limit
    )


def _page_units(text: str, limit: int) -> list[_Unit]:
    units: list[_Unit] = []
    line_number = 0
    for paragraph_number, paragraph in enumerate(text.split("\n\n")):
        for line in paragraph.split("\n"):
            line_number += 1
            for sentence in split_sentences(line):
                for piece in _split_oversized(sentence, limit):
                    if piece:
                        units.append(
                            _Unit(piece, estimate_tokens(piece), paragraph_number, line_number)
                        )
    return units


def _render(units: list[_Unit]) -> str:
    parts: list[str] = []
    for previous, unit in zip([None, *units], units, strict=False):
        if previous is not None:
            if unit.paragraph != previous.paragraph:
                parts.append("\n\n")
            elif unit.line != previous.line:
                parts.append("\n")
            else:
                parts.append(" ")
        parts.append(unit.text)
    return "".join(parts)


def _passages(units: list[_Unit], size: int) -> tuple[str, ...]:
    """Group a chunk's sentences into passages, starting a new one at most paragraphs."""
    groups: list[list[_Unit]] = []
    tokens = 0
    for unit in units:
        new_paragraph = bool(groups) and unit.paragraph != groups[-1][-1].paragraph
        worth_closing = tokens >= size * _PASSAGE_PARAGRAPH_RATIO
        if not groups or tokens + unit.tokens > size or (new_paragraph and worth_closing):
            groups.append([])
            tokens = 0
        groups[-1].append(unit)
        tokens += unit.tokens
    return tuple(_render(group) for group in groups)


def _overlap_tail(units: list[_Unit], overlap_tokens: int) -> list[_Unit]:
    tail: list[_Unit] = []
    total = 0
    for unit in reversed(units):
        if total + unit.tokens > overlap_tokens:
            break
        tail.insert(0, unit)
        total += unit.tokens
    return tail


def _paragraph_remainder(units: list[_Unit], start: int) -> int:
    paragraph = units[start].paragraph
    total = 0
    for unit in units[start:]:
        if unit.paragraph != paragraph:
            break
        total += unit.tokens
    return total


def _pack_page(text: str, config: ChunkingConfig) -> Iterable[list[_Unit]]:
    size = config.chunk_size_tokens
    units = _page_units(text, size)
    current: list[_Unit] = []
    tokens = 0
    has_new_content = False

    for index, unit in enumerate(units):
        starts_paragraph = bool(current) and unit.paragraph != current[-1].paragraph
        overflows = tokens + unit.tokens > size
        paragraph_would_split = (
            starts_paragraph
            and tokens >= size * config.paragraph_break_ratio
            and tokens + _paragraph_remainder(units, index) > size
        )
        if has_new_content and (overflows or paragraph_would_split):
            yield current
            current = _overlap_tail(current, config.chunk_overlap_tokens)
            tokens = sum(item.tokens for item in current)
            has_new_content = False
            while current and tokens + unit.tokens > size:
                tokens -= current.pop(0).tokens
        current.append(unit)
        tokens += unit.tokens
        has_new_content = True

    if has_new_content:
        yield current


def chunk_pages(
    document_id: str, pages: Iterable[Mapping[str, Any]], config: ChunkingConfig
) -> list[Chunk]:
    """Chunk stored pages, each a mapping with ``number`` and ``text`` keys."""
    chunks: list[Chunk] = []
    for page in pages:
        if not page["text"]:
            continue
        for units in _pack_page(page["text"], config):
            index = len(chunks)
            text = _render(units)
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}-{index:06d}",
                    document_id=document_id,
                    text=text,
                    page_number=page["number"],
                    chunk_index=index,
                    token_count=estimate_tokens(text),
                    passages=_passages(units, config.passage_size_tokens),
                )
            )
    return chunks
