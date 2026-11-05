"""Tests for grounded question answering, its fallbacks and conversation history."""

from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import FakeLLM, HashingEmbedder

from core.embedding_engine import EmbeddingModelError
from core.extractive import GenerationMethod
from core.indexing import DocumentIndexer
from core.ingestion import DocumentIngestor
from core.ollama_client import OllamaModelNotFoundError, OllamaTimeoutError, OllamaUnavailableError
from core.question_answering import NO_EVIDENCE_TEXT, QuestionAnswerer
from core.retriever import Retriever

CATS = b"Cats sleep on warm sofas. Cats chase small mice at night in the barn."
ROCKETS = b"Rockets burn liquid fuel to reach orbit. The launch cost 62 million dollars in 2023."
QUESTION = "Where do cats sleep on sofas?"


def answerer_with(db, retriever, embedder, settings, llm) -> QuestionAnswerer:
    return QuestionAnswerer(db, retriever, embedder, llm, settings)


@pytest.fixture
def make(db, retriever, embedder, test_settings):
    def build(reply: object = "They sleep on warm sofas [S1].") -> tuple[QuestionAnswerer, FakeLLM]:
        llm = FakeLLM(reply)
        return answerer_with(db, retriever, embedder, test_settings, llm), llm

    return build


@pytest.fixture
def cats(indexed_ingestor):
    return indexed_ingestor.ingest("cats.txt", CATS).document


def test_extractive_answer_only_contains_document_sentences(make, cats) -> None:
    answerer, llm = make()
    answer = answerer.ask(QUESTION, use_llm=False)
    assert llm.prompts == []
    for line in answer.text.splitlines()[1:]:
        assert line.removeprefix("- ").removesuffix(" (cats.txt)") in CATS.decode()


class NoModelEmbedder(HashingEmbedder):
    def embed_query(self, query):
        raise EmbeddingModelError("Model missing.")

    def embed_texts(self, texts, progress=None):
        raise EmbeddingModelError("Model missing.")


def test_missing_embedding_model_uses_keyword_search(db, store, test_settings) -> None:
    embedder = NoModelEmbedder()
    indexer = DocumentIndexer(db, store, embedder, test_settings)
    ingestor = DocumentIngestor(db, test_settings, indexer)
    ingestor.ingest("cats.txt", CATS)
    retriever = Retriever(db, store, embedder, indexer, test_settings.retrieval)
    answerer = answerer_with(db, retriever, embedder, test_settings, FakeLLM())
    answer = answerer.ask("warm sofas", use_llm=False)
    assert answer.method is GenerationMethod.EXTRACTIVE
    assert "warm sofas" in answer.text
    assert any("keyword search" in notice for notice in answer.notices)
    assert answerer.ask("rocket fuel", use_llm=False).has_evidence is False


def test_blank_question(make, cats) -> None:
    answerer, llm = make()
    assert answerer.ask("   ", model="m").has_evidence is False and llm.prompts == []
