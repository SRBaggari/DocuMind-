"""Extract themes, names, dates, figures, conclusions and statistics from one document.

Names, dates and figures come from pattern matching, not a trained model, so they are
candidates to review. Everything listed here is copied from the document.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import Settings
from core.chunker import chunk_pages
from core.extractive import (
    ACTION_PATTERN,
    CONCLUSION_PATTERN,
    Sentence,
    document_parts,
    document_sentences,
    rank_sentences,
    select_diverse,
)
from core.indexing import read_content
from core.keyword_extractor import (
    Keyword,
    Topic,
    extract_keyphrases,
    extract_keywords,
    identify_topics,
)
from database.db_manager import DatabaseManager
from database.schema import DocumentStatus

_MAX_ITEMS = 30
_MAX_SECTIONS = 5
_MAX_CONCLUSIONS = 5
_MAX_ACTIONS = 5
_SECTION_KEYWORDS = 40

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December|"
    "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_DATE = re.compile(
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\.?,?\s+\d{{4}}\b"
    rf"|\b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"
    rf"|\b(?:{_MONTHS})\.?\s+\d{{4}}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|\bQ[1-4]\s+\d{4}\b"
    r"|\b(?:19|20)\d{2}\b"
)
_STATISTIC = re.compile(
    r"(?:[$€£]\s?\d[\d,]*(?:\.\d+)?(?:\s(?:thousand|million|billion|trillion))?)"
    r"|(?:(?<![\w.])\d[\d,]*(?:\.\d+)?\s?(?:%|percent\b|per cent\b))"
    r"|(?:(?<![\w.])\d[\d,]*(?:\.\d+)?\s(?:thousand|million|billion|trillion)\b(?:\s[a-z]+)?)"
    r"|(?:(?<![\w.])\d{1,3}(?:,\d{3})+\b(?:\s[a-z]+)?)"
)
_CAPITALIZED = r"(?:[A-Z][\w'’-]+|[A-Z]{2,})"
_NAME = re.compile(rf"\b{_CAPITALIZED}(?:\s+(?:of\s+|the\s+|for\s+|and\s+)?{_CAPITALIZED})*\b")
_NOT_NAMES = frozenset(_MONTHS.lower().split("|")) | frozenset(
    "monday tuesday wednesday thursday friday saturday sunday".split()
)


class InsightsError(Exception):
    """Insights cannot be produced. The message is safe to show to the user."""


@dataclass(frozen=True)
class Mention:
    text: str
    count: int
    pages: tuple[int, ...]


@dataclass(frozen=True)
class Statistic:
    value: str
    sentence: str
    page: int | None


@dataclass(frozen=True)
class Evidence:
    text: str
    page: int | None


@dataclass(frozen=True)
class ImportantSection:
    title: str
    score: float
    snippet: str


@dataclass(frozen=True)
class DocumentStatistics:
    words: int
    characters: int
    pages: int | None
    paragraphs: int
    chunks: int
    keywords: int
    reading_minutes: int


@dataclass(frozen=True)
class DocumentInsights:
    document_id: str
    filename: str
    statistics: DocumentStatistics
    themes: tuple[Topic, ...]
    keywords: tuple[Keyword, ...]
    keyphrases: tuple[Keyword, ...]
    frequent_terms: tuple[tuple[str, int], ...]
    names: tuple[Mention, ...]
    dates: tuple[Mention, ...]
    statistics_found: tuple[Statistic, ...]
    important_sections: tuple[ImportantSection, ...]
    conclusions: tuple[Evidence, ...]
    action_items: tuple[Evidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mentions(found: list[tuple[str, int | None]], minimum: int = 1) -> tuple[Mention, ...]:
    counts = Counter(text for text, _ in found)
    pages: dict[str, set[int]] = {}
    for text, page in found:
        if page:
            pages.setdefault(text, set()).add(page)
    ranked = [item for item in counts.most_common() if item[1] >= minimum]
    return tuple(
        Mention(text, count, tuple(sorted(pages.get(text, ())))) for text, count in ranked
    )[:_MAX_ITEMS]


def _clean_name(raw: str) -> str | None:
    """Drop leading words that are capitalized only because they start a sentence."""
    words = raw.split()
    while words and words[0].lower() in ENGLISH_STOP_WORDS:
        words.pop(0)
    while words and words[-1].lower() in ENGLISH_STOP_WORDS:
        words.pop()
    if not words or all(word.lower() in _NOT_NAMES for word in words):
        return None
    return " ".join(words)


def find_names(sentences: list[Sentence]) -> tuple[Mention, ...]:
    """List capitalized names. A single word must not just be starting its sentence."""
    multi_word: list[tuple[str, int | None]] = []
    single_word: list[tuple[str, int | None]] = []
    for sentence in sentences:
        for match in _NAME.finditer(sentence.text):
            name = _clean_name(match[0])
            if name is None:
                continue
            if " " in name:
                multi_word.append((name, sentence.page_number))
            elif match.start() > 0:
                single_word.append((name, sentence.page_number))
    return _mentions(multi_word + single_word)


def find_dates(sentences: list[Sentence]) -> tuple[Mention, ...]:
    found = [
        (match[0], sentence.page_number)
        for sentence in sentences
        for match in _DATE.finditer(sentence.text)
    ]
    return _mentions(found)


def find_statistics(sentences: list[Sentence]) -> tuple[Statistic, ...]:
    """List figures that carry a unit or scale, each with the sentence stating it."""
    found: list[Statistic] = []
    seen: set[tuple[str, str]] = set()
    for sentence in sentences:
        for match in _STATISTIC.finditer(sentence.text):
            key = (match[0], sentence.text)
            if key not in seen:
                seen.add(key)
                found.append(Statistic(match[0], sentence.text, sentence.page_number))
    return tuple(found[:_MAX_ITEMS])


def _evidence(sentences: list[Sentence], pattern: re.Pattern[str], limit: int):
    ranked = [item for item in rank_sentences(sentences) if pattern.search(item[0].text)]
    return tuple(
        Evidence(s.text, s.page_number) for s in select_diverse(ranked, limit, min_score=-1.0)
    )


class InsightsEngine:
    def __init__(self, db: DatabaseManager, settings: Settings) -> None:
        self._db = db
        self._settings = settings

    def analyze(self, document_id: str) -> DocumentInsights:
        document = self._db.get_document(document_id)
        if document is None or document.status is not DocumentStatus.READY:
            raise InsightsError("This document has no extracted text to analyze.")
        content = read_content(self._settings.paths, document_id)
        pages = [page for page in content["pages"] if page["text"]]
        texts = [page["text"] for page in pages]
        sentences = document_sentences(pages)
        config = self._settings.analysis
        chunks = chunk_pages(document_id, pages, self._settings.chunking)
        keywords = extract_keywords(texts, config.max_keywords)
        words = document.word_count or 0
        statistics = DocumentStatistics(
            words=words,
            characters=document.char_count or 0,
            pages=document.page_count,
            paragraphs=document.paragraph_count or 0,
            chunks=document.chunk_count or len(chunks),
            keywords=len(keywords),
            reading_minutes=math.ceil(words / config.reading_words_per_minute),
        )
        return DocumentInsights(
            document_id,
            document.original_filename,
            statistics,
            tuple(
                identify_topics(
                    chunks, config.max_topics, config.terms_per_topic, config.sections_per_topic
                )
            ),
            tuple(keywords),
            tuple(extract_keyphrases(texts, config.max_keywords)),
            tuple(
                (k.term, k.count) for k in sorted(keywords, key=lambda k: (-k.count, k.term))
            ),
            find_names(sentences),
            find_dates(sentences),
            find_statistics(sentences),
            self._important_sections(pages, content["blocks"]),
            _evidence(sentences, CONCLUSION_PATTERN, _MAX_CONCLUSIONS),
            _evidence(sentences, ACTION_PATTERN, _MAX_ACTIONS),
        )

    def _important_sections(
        self, pages: list[dict[str, Any]], blocks: list[dict[str, Any]]
    ) -> tuple[ImportantSection, ...]:
        """Rank parts by how much of the document's key vocabulary each one contains."""
        keywords = extract_keywords([page["text"] for page in pages], _SECTION_KEYWORDS)
        total = sum(keyword.score for keyword in keywords) or 1.0
        sections = []
        for title, part in document_parts(pages, blocks):
            words = set(re.findall(r"[\w'-]+", " ".join(p["text"] for p in part).lower()))
            score = sum(k.score for k in keywords if k.term in words) / total
            best = select_diverse(rank_sentences(document_sentences(part)), 1, min_score=-1.0)
            if best:
                sections.append(ImportantSection(title, score, best[0].text))
        return tuple(sorted(sections, key=lambda s: -s.score)[:_MAX_SECTIONS])
