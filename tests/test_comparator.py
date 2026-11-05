"""Tests for comparison scores, detected differences and edge cases."""

from __future__ import annotations

import numpy as np
import pymupdf
import pytest
from conftest import FakeLLM

from core.document_comparator import _match_passages, shingle_overlap
from core.extractive import Sentence
from core.keyword_extractor import document_similarity
from core.ollama_client import OllamaUnavailableError

REPORT = (
    "The Riverside authority reached operating revenue of 148 million dollars in 2024.\n\n"
    "All samples met national drinking water standards during the year.\n\n"
    "Engineers repaired 1,206 leaks across the old cast iron pipe network this season.\n\n"
    "Customer satisfaction measured by the annual survey rose to 87 percent overall."
)
REVISED = (
    REPORT.replace("148 million", "152 million").replace("All samples", "Most samples")
    + "\n\nA desalination pilot in Portmere will supply coastal suburbs with seawater drinking "
    "water from 2027 through a jointly operated regional energy utility partnership program."
)
UNRELATED = (
    "Rockets burn liquid fuel to reach orbit around the planet every single month.\n\n"
    "Astronauts train underwater for many long months before their first spacewalk.\n\n"
    "The launch tower stands beside the ocean and holds the booster upright for testing."
)


GARDENING = (
    "Tomato seedlings need strong sunlight and regular watering through the warm summer.\n\n"
    "Compost improves heavy clay soil and feeds beans, carrots and onions in raised beds.\n\n"
    "Pruning apple trees in winter encourages fruit and keeps the branches open to the air.\n\n"
    "Slugs damage young lettuce, so gardeners protect the leaves with copper tape and netting."
)


def test_shingle_overlap_values() -> None:
    assert shingle_overlap("a b c d e f", "a b c d e f") == 1.0
    assert shingle_overlap("a b c d e f", "a b c d e g") == pytest.approx(1 / 3)
    assert shingle_overlap("one two", "three four") == 0.0
    assert shingle_overlap("", "") == 0.0


def test_document_similarity_of_empty_text() -> None:
    assert document_similarity("", "") == 0.0


def test_matching_in_blocks_equals_the_full_matrix(monkeypatch) -> None:
    monkeypatch.setattr("core.document_comparator._BLOCK_ROWS", 7)
    generator = np.random.default_rng(0)
    a = generator.normal(size=(40, 16)).astype(np.float32)
    b = generator.normal(size=(23, 16)).astype(np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    full = a @ b.T
    matches = _match_passages(a, b, 5)
    assert np.allclose(matches.best_a, full.max(axis=1), atol=1e-5)
    assert np.allclose(matches.best_b, full.max(axis=0), atol=1e-5)
    assert (matches.match_a == full.argmax(axis=1)).all()
    assert matches.grid.max() == pytest.approx(full.max(), abs=1e-5)
