"""Extract TF-IDF keywords and keyphrases, and NMF topics.

Stop words are English, so other languages get weaker results. Topic labels are lists
of a topic's top terms; they are generated and have not been reviewed by a person.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer,
)

_TOKEN_PATTERN = r"(?u)\b[^\W\d_][\w'-]+\b"
# Words that appear in most reports without saying what any of them is about.
_GENERIC_TERMS = frozenset(
    "percent million billion thousand dollars year years month months day days".split()
)
_STOP_WORDS = sorted(ENGLISH_STOP_WORDS | _GENERIC_TERMS)
_STOP_WORD_SET = frozenset(_STOP_WORDS)
_PHRASE_WORD = re.compile(r"[^\W\d_][\w'-]+")
_PHRASE_BREAK = re.compile(r"[^\w\s'-]|\b\d")
_MAX_FEATURES = 20_000
_PHRASE_RANGE = (2, 3)
_LABEL_TERMS = 3
_SNIPPET_CHARS = 240
_NMF_MAX_ITERATIONS = 500
_MIN_SEGMENTS_FOR_TOPICS = 3
_LARGE_CORPUS_SEGMENTS = 10


class ChunkLike(Protocol):
    chunk_id: str
    document_id: str
    text: str
    page_number: int | None


@dataclass(frozen=True)
class Keyword:
    term: str
    score: float
    count: int


@dataclass(frozen=True)
class KeywordComparison:
    term: str
    score_a: float
    score_b: float

    @property
    def difference(self) -> float:
        return self.score_a - self.score_b


@dataclass(frozen=True)
class TopicSection:
    chunk_id: str
    document_id: str
    page_number: int | None
    snippet: str
    weight: float


@dataclass(frozen=True)
class Topic:
    label: str
    terms: tuple[tuple[str, float], ...]
    share: float
    sections: tuple[TopicSection, ...]


def _vectorizer(ngram_range: tuple[int, int], **options: object) -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words=_STOP_WORDS,
        token_pattern=_TOKEN_PATTERN,
        ngram_range=ngram_range,
        sublinear_tf=True,
        max_features=_MAX_FEATURES,
        **options,
    )


def _adjacent_phrases(text: str) -> list[str]:
    """List two- and three-word phrases whose words really are next to each other.

    A stop word, number or punctuation mark ends a run, so "fell by 3 percent" and
    "kilometres of pipe" do not turn into phrases that the text never contains.
    """
    phrases: list[str] = []
    for fragment in _PHRASE_BREAK.split(text.lower()):
        run: list[str] = []
        for word in [*_PHRASE_WORD.findall(fragment), ""]:
            if word and word not in _STOP_WORD_SET:
                run.append(word)
                continue
            for size in (2, 3):
                phrases += [" ".join(run[i : i + size]) for i in range(len(run) - size + 1)]
            run = []
    return phrases


def _rank_terms(
    segments: Sequence[str], ngram_range: tuple[int, int]
) -> list[tuple[str, float]]:
    texts = [segment for segment in segments if segment.strip()]
    if not texts:
        return []
    if ngram_range == _PHRASE_RANGE:
        vectorizer = TfidfVectorizer(
            analyzer=_adjacent_phrases, sublinear_tf=True, max_features=_MAX_FEATURES
        )
    else:
        vectorizer = _vectorizer(ngram_range)
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # Every word was a stop word or too short.
        return []
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    order = np.argsort(-scores, kind="stable")
    return [(str(terms[i]), float(scores[i])) for i in order]


def _with_counts(
    ranked: list[tuple[str, float]], segments: Sequence[str], ngram_range: tuple[int, int]
) -> list[Keyword]:
    if not ranked:
        return []
    vocabulary = [term for term, _ in ranked]
    if ngram_range == _PHRASE_RANGE:
        counter = CountVectorizer(vocabulary=vocabulary, analyzer=_adjacent_phrases)
    else:
        counter = CountVectorizer(
            vocabulary=vocabulary,
            token_pattern=_TOKEN_PATTERN,
            ngram_range=ngram_range,
            stop_words=_STOP_WORDS,
        )
    counts = np.asarray(counter.transform(segments).sum(axis=0)).ravel()
    return [
        Keyword(term, score, int(count)) for (term, score), count in zip(ranked, counts)
    ]


def extract_keywords(segments: Sequence[str], max_keywords: int) -> list[Keyword]:
    """Return the highest-scoring single words across a document's segments."""
    ngram_range = (1, 1)
    return _with_counts(_rank_terms(segments, ngram_range)[:max_keywords], segments, ngram_range)


def extract_keyphrases(segments: Sequence[str], max_phrases: int) -> list[Keyword]:
    """Return top two- and three-word phrases, skipping fragments of a better phrase."""
    selected: list[tuple[str, float]] = []
    for term, score in _rank_terms(segments, _PHRASE_RANGE):
        if any(term in kept or kept in term for kept, _ in selected):
            continue
        selected.append((term, score))
        if len(selected) == max_phrases:
            break
    return _with_counts(selected, segments, _PHRASE_RANGE)


def compare_keywords(
    segments_a: Sequence[str], segments_b: Sequence[str], max_terms: int
) -> list[KeywordComparison]:
    """Score terms in two documents on one shared TF-IDF scale, most important first."""
    texts_a = [text for text in segments_a if text.strip()]
    texts_b = [text for text in segments_b if text.strip()]
    if not texts_a or not texts_b:
        return []
    vectorizer = _vectorizer((1, 1))
    try:
        matrix = vectorizer.fit_transform([*texts_a, *texts_b])
    except ValueError:
        return []
    scores_a = np.asarray(matrix[: len(texts_a)].mean(axis=0)).ravel()
    scores_b = np.asarray(matrix[len(texts_a) :].mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    order = np.argsort(-np.maximum(scores_a, scores_b), kind="stable")[:max_terms]
    return [
        KeywordComparison(str(terms[i]), float(scores_a[i]), float(scores_b[i])) for i in order
    ]


def document_similarity(text_a: str, text_b: str) -> float:
    """Return the TF-IDF cosine similarity of two texts, from 0 to 1."""
    try:
        matrix = _vectorizer((1, 1)).fit_transform([text_a, text_b])
    except ValueError:
        return 0.0
    return float((matrix[0] @ matrix[1].T).toarray()[0, 0])


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= _SNIPPET_CHARS:
        return flat
    return flat[:_SNIPPET_CHARS].rsplit(" ", 1)[0] + "..."


def identify_topics(
    chunks: Sequence[ChunkLike],
    max_topics: int,
    terms_per_topic: int,
    sections_per_topic: int,
) -> list[Topic]:
    """Find topics with NMF. Returns an empty list if there is too little text."""
    if len(chunks) < _MIN_SEGMENTS_FOR_TOPICS:
        return []
    large = len(chunks) >= _LARGE_CORPUS_SEGMENTS
    vectorizer = _vectorizer((1, 2), min_df=2 if large else 1, max_df=0.9 if large else 1.0)
    try:
        matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
    except ValueError:
        return []
    component_count = min(max_topics, len(chunks) - 1, matrix.shape[1])
    if component_count < 1:
        return []

    model = NMF(
        n_components=component_count,
        init="nndsvda",
        random_state=0,
        max_iter=_NMF_MAX_ITERATIONS,
    )
    weights = model.fit_transform(matrix)
    terms = vectorizer.get_feature_names_out()
    dominant = weights.argmax(axis=1)
    total_weight = float(weights.sum()) or 1.0

    topics: list[Topic] = []
    for index, component in enumerate(model.components_):
        if not component.any():
            continue
        top = np.argsort(-component, kind="stable")[:terms_per_topic]
        top_terms = tuple(
            (str(terms[i]), float(component[i])) for i in top if component[i] > 0
        )
        members = np.flatnonzero((dominant == index) & (weights[:, index] > 0))
        members = members[np.argsort(-weights[members, index], kind="stable")]
        sections = tuple(
            TopicSection(
                chunks[i].chunk_id,
                chunks[i].document_id,
                chunks[i].page_number,
                _snippet(chunks[i].text),
                float(weights[i, index]),
            )
            for i in members[:sections_per_topic]
        )
        topics.append(
            Topic(
                label=", ".join(term for term, _ in top_terms[:_LABEL_TERMS]),
                terms=top_terms,
                share=float(weights[:, index].sum()) / total_weight,
                sections=sections,
            )
        )
    return sorted(topics, key=lambda topic: -topic.share)
