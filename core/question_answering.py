"""Answer questions from retrieved document evidence, or say that there is none.

Answers are sentences copied from the best-matching chunks. Every Answer records how
it was produced.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from config import ChunkingConfig, Settings
from core.chunker import chunk_pages
from core.embedding_engine import Embedder, EmbeddingModelError
from core.extractive import (
    GenerationMethod,
    Sentence,
    document_sentences,
    rank_sentences,
    select_diverse,
)
from core.indexing import read_content
from core.ollama_client import LocalLLM
from core.retriever import RetrievedChunk, Retriever
from database.db_manager import DatabaseManager
from database.schema import DocumentStatus

NO_EVIDENCE_TEXT = "I could not find enough evidence in the selected documents to answer this."
_EXTRACTIVE_INTRO = "These are the most relevant sentences from the documents:"
_EXTRACTIVE_NOTICE = (
    "Extractive NLP mode: these sentences are copied from the documents. "
    "No AI model wrote or interpreted them."
)

_KEYWORD_MIN_SCORE = 0.05
_RERANK_TERM_WEIGHT = 0.15
_EXTRACTIVE_SENTENCES = 3
_WORD = re.compile(r"[^\W\d_]{3,}")

@dataclass(frozen=True)
class Source:
    label: str
    document_id: str
    filename: str
    page_number: int | None
    chunk_id: str
    text: str
    score: float
    cited: bool = False

    @property
    def reference(self) -> str:
        return f"{self.filename}, p. {self.page_number}" if self.page_number else self.filename


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    method: GenerationMethod
    model: str | None = None
    sources: tuple[Source, ...] = ()
    notices: tuple[str, ...] = ()

    @property
    def has_evidence(self) -> bool:
        return self.method is not GenerationMethod.NONE


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in ENGLISH_STOP_WORDS}


def _rerank(hits: Sequence[RetrievedChunk], question: str) -> list[RetrievedChunk]:
    """Nudge chunks that contain the question's own words above near-ties."""
    terms = _content_words(question)

    def rank(hit: RetrievedChunk) -> float:
        coverage = len(terms & _content_words(hit.passage)) / len(terms) if terms else 0.0
        return hit.score + _RERANK_TERM_WEIGHT * coverage

    return sorted(hits, key=rank, reverse=True)


class QuestionAnswerer:
    def __init__(
        self,
        db: DatabaseManager,
        retriever: Retriever,
        embedder: Embedder,
        llm: LocalLLM,
        settings: Settings,
    ) -> None:
        self._db = db
        self._retriever = retriever
        self._embedder = embedder
        self._llm = llm
        self._settings = settings

    def ask(
        self,
        question: str,
        document_ids: Iterable[str] | None = None,
        model: str | None = None,
        use_llm: bool = True,
    ) -> Answer:
        """Answer ``question`` from evidence retrieved for it."""
        question = question.strip()
        if not question:
            return Answer(question, "Please enter a question.", GenerationMethod.NONE)

        ids = None if document_ids is None else list(document_ids)
        notices: list[str] = []
        semantic = True
        try:
            hits = self._retriever.retrieve(question, ids)
        except EmbeddingModelError as exc:
            semantic = False
            notices.append(f"Semantic search is unavailable, so keyword search was used. {exc}")
            hits = self._keyword_search(question, ids)
        if not hits:
            return Answer(question, NO_EVIDENCE_TEXT, GenerationMethod.NONE, notices=tuple(notices))

        hits = _rerank(hits, question)
        sources = tuple(
            Source(
                f"S{index}",
                hit.document_id,
                hit.filename,
                hit.page_number,
                hit.chunk_id,
                hit.passage,
                hit.score,
            )
            for index, hit in enumerate(hits, start=1)
        )

        return self._answer_extractively(question, hits, sources, notices, semantic)

    def _answer_extractively(
        self,
        question: str,
        hits: Sequence[RetrievedChunk],
        sources: tuple[Source, ...],
        notices: list[str],
        semantic: bool,
    ) -> Answer:
        references: dict[int, str] = {}
        sentences: list[Sentence] = []
        for hit, source in zip(hits, sources):
            page = {"number": hit.page_number, "text": hit.text}
            for sentence in document_sentences([page]):
                if all(sentence.text != other.text for other in sentences):
                    position = len(sentences)
                    sentences.append(replace(sentence, position=position))
                    references[position] = source.reference

        extra = None
        if semantic and sentences:
            vectors = self._embedder.embed_texts([sentence.text for sentence in sentences])
            extra = vectors @ self._embedder.embed_query(question)
        chosen = select_diverse(
            rank_sentences(sentences, question, extra), _EXTRACTIVE_SENTENCES
        )
        if not chosen:
            return Answer(question, NO_EVIDENCE_TEXT, GenerationMethod.NONE, notices=tuple(notices))
        lines = [f"- {s.text} ({references[s.position]})" for s in chosen]
        return Answer(
            question,
            "\n".join([_EXTRACTIVE_INTRO, *lines]),
            GenerationMethod.EXTRACTIVE,
            None,
            sources,
            (*notices, _EXTRACTIVE_NOTICE),
        )

    def _keyword_search(
        self, query: str, document_ids: list[str] | None
    ) -> list[RetrievedChunk]:
        """Rank chunks by TF-IDF, for when the embedding model is unavailable."""
        allowed = None if document_ids is None else set(document_ids)
        config = self._settings.chunking
        chunks: list[RetrievedChunk] = []
        for document in self._db.list_documents():
            if document.status is not DocumentStatus.READY:
                continue
            if allowed is not None and document.document_id not in allowed:
                continue
            pages = read_content(self._settings.paths, document.document_id)["pages"]
            for chunk in chunk_pages(document.document_id, pages, _keyword_config(config)):
                chunks.append(
                    RetrievedChunk(
                        chunk.chunk_id,
                        chunk.document_id,
                        document.original_filename,
                        chunk.text,
                        chunk.text,
                        chunk.page_number,
                        chunk.chunk_index,
                        0.0,
                    )
                )
        if not chunks:
            return []
        vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        try:
            matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
        except ValueError:
            return []
        scores = np.asarray(matrix @ vectorizer.transform([query]).T.toarray()).ravel()
        order = np.argsort(-scores, kind="stable")[: self._settings.retrieval.top_k]
        return [
            replace(chunks[i], score=float(scores[i]))
            for i in order
            if scores[i] >= _KEYWORD_MIN_SCORE
        ]


def _keyword_config(config: ChunkingConfig) -> ChunkingConfig:
    return replace(config, chunk_overlap_tokens=0)
