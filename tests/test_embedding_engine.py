"""Tests against the real embedding model. Skipped if it has not been downloaded yet."""

from __future__ import annotations

import numpy as np
import pytest

from config import EmbeddingConfig
from core.embedding_engine import EmbeddingEngine, EmbeddingModelError


@pytest.fixture(scope="module")
def engine() -> EmbeddingEngine:
    engine = EmbeddingEngine(EmbeddingConfig())
    try:
        engine.embed_query("warm up")
    except EmbeddingModelError as exc:
        pytest.skip(str(exc))
    return engine


def test_model_is_not_loaded_until_needed() -> None:
    assert EmbeddingEngine(EmbeddingConfig()).is_loaded is False


def test_vectors_are_unit_length_float32(engine) -> None:
    vectors = engine.embed_texts(["First text.", "Second text."])
    assert vectors.shape == (2, 384) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_empty_input(engine) -> None:
    assert engine.embed_texts([]).shape == (0, 384)


def test_meaning_decides_similarity(engine) -> None:
    query = engine.embed_query("How much money did the company earn?")
    related, unrelated = engine.embed_texts(
        [
            "The firm reported annual revenue of two million dollars.",
            "Penguins nest on Antarctic ice.",
        ]
    )
    assert query @ related > query @ unrelated + 0.2


def test_text_past_the_model_limit_still_counts(engine) -> None:
    filler = "The committee reviewed the routine schedule for the coming months. " * 40
    ending = "Penguins nest on Antarctic ice in the winter darkness. " * 12
    full, truncated = engine.embed_texts([filler + ending, filler])
    query = engine.embed_query("Where do penguins nest?")
    assert query @ full > query @ truncated + 0.05


def test_batch_and_single_results_match(engine) -> None:
    together = engine.embed_texts(["Alpha text.", "Beta text."])
    assert np.allclose(together[1], engine.embed_query("Beta text."), atol=1e-4)


def test_progress_reaches_completion(engine) -> None:
    seen: list[float] = []
    engine.embed_texts(["One.", "Two."], seen.append)
    assert seen[-1] == 1.0


def test_unavailable_model_gives_clear_error() -> None:
    engine = EmbeddingEngine(EmbeddingConfig(model_name="documind-test/does-not-exist"))
    with pytest.raises(EmbeddingModelError, match="could not be downloaded"):
        engine.embed_query("text")
