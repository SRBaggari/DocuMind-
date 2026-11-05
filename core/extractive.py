"""Select sentences from a document with TF-IDF. Used when no local LLM is available.

Everything here copies text from the document, so it cannot invent facts.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.text_utils import count_words, split_sentences

_MIN_SENTENCE_WORDS = 4
_TABLE_CELL_MARKER = " | "
_MAX_WORD_OVERLAP = 0.6
_MAX_DOCUMENT_PARTS = 8

CONCLUSION_PATTERN = re.compile(
    r"\b(conclu\w+|in summary|overall|therefore|expects?|future)\b", re.IGNORECASE
)
ACTION_PATTERN = re.compile(
    r"\b(recommends?|should|must|needs? to|next steps?)\b", re.IGNORECASE
)


class GenerationMethod(StrEnum):
    LLM = "llm"
    EXTRACTIVE = "extractive"
    NONE = "none"


@dataclass(frozen=True)
class Sentence:
    text: str
    page_number: int | None
    position: int


def document_sentences(pages: Iterable[Mapping[str, Any]]) -> list[Sentence]:
    """List a document's unique prose sentences, keeping their page and order."""
    sentences: list[Sentence] = []
    seen: set[str] = set()
    for page in pages:
        for line in page["text"].split("\n"):
            if _TABLE_CELL_MARKER in line:
                continue
            for text in split_sentences(line):
                if count_words(text) >= _MIN_SENTENCE_WORDS and text not in seen:
                    seen.add(text)
                    sentences.append(Sentence(text, page["number"], len(sentences)))
    return sentences


def rank_sentences(
    sentences: Sequence[Sentence],
    query: str | None = None,
    extra_scores: np.ndarray | None = None,
) -> list[tuple[Sentence, float]]:
    """Rank sentences by similarity to ``query``, or to the document as a whole."""
    if not sentences:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform([sentence.text for sentence in sentences])
    except ValueError:
        return []
    if query is None:
        target = np.asarray(matrix.mean(axis=0))
    else:
        target = vectorizer.transform([query]).toarray()
    scores = np.asarray(matrix @ target.T).ravel()
    if extra_scores is not None:
        scores = scores + extra_scores
    order = np.argsort(-scores, kind="stable")
    return [(sentences[i], float(scores[i])) for i in order]


def _word_overlap(left: str, right: str) -> float:
    left_words, right_words = set(left.lower().split()), set(right.lower().split())
    return len(left_words & right_words) / max(min(len(left_words), len(right_words)), 1)


def select_diverse(
    ranked: Iterable[tuple[Sentence, float]], limit: int, min_score: float = 0.0
) -> list[Sentence]:
    """Take the best sentences that do not repeat each other, in document order."""
    chosen: list[Sentence] = []
    for sentence, score in ranked:
        if len(chosen) == limit or score <= min_score:
            break
        if all(_word_overlap(sentence.text, kept.text) < _MAX_WORD_OVERLAP for kept in chosen):
            chosen.append(sentence)
    return sorted(chosen, key=lambda sentence: sentence.position)


def _heading_parts(blocks: Sequence[Mapping[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group DOCX blocks under their headings."""
    parts: list[tuple[str, list[str]]] = []
    for block in blocks:
        if block["kind"] == "heading" and block["level"]:
            parts.append((block["text"], []))
        elif parts and block["kind"] != "heading":
            parts[-1][1].append(block["text"])
    return [
        (title, [{"number": None, "text": "\n".join(body)}]) for title, body in parts if body
    ]


def _page_parts(pages: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Split a document without headings into a few even parts."""
    if len(pages) == 1 and pages[0]["number"] is None:
        paragraphs = [text for text in pages[0]["text"].split("\n\n") if text.strip()]
        pages = [{"number": None, "text": text} for text in paragraphs]
    size = max(-(-len(pages) // _MAX_DOCUMENT_PARTS), 1)
    parts = []
    for start in range(0, len(pages), size):
        group = pages[start : start + size]
        numbers = [page["number"] for page in group if page["number"]]
        if numbers:
            span = f"Page {numbers[0]}" if len(set(numbers)) == 1 else (
                f"Pages {numbers[0]}-{numbers[-1]}"
            )
        else:
            span = f"Part {len(parts) + 1}"
        parts.append((span, group))
    return parts


def document_parts(
    pages: list[dict[str, Any]], blocks: Sequence[Mapping[str, Any]]
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Split a document into titled parts: its headings if it has any, or even page groups."""
    return _heading_parts(blocks) or _page_parts(pages)
