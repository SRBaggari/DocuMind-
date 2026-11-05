"""Select sentences from a document with TF-IDF. Used when no local LLM is available.

Everything here copies text from the document, so it cannot invent facts.
"""

from __future__ import annotations

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
