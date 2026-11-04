"""Tests for chunk boundaries and metadata."""

from __future__ import annotations

from config import ChunkingConfig
from core.chunker import chunk_pages
from utils.text_utils import estimate_tokens, split_sentences

DOC = "d" * 32
SMALL = ChunkingConfig(chunk_size_tokens=40, chunk_overlap_tokens=10)


def sentences(count: int, start: int = 0) -> str:
    numbers = range(start, start + count)
    return " ".join(f"Sentence number {n} talks about topic {n}." for n in numbers)


def page(text: str, number: int | None = 1) -> dict:
    return {"number": number, "text": text}


def test_required_metadata_and_stable_ids() -> None:
    chunks = chunk_pages(DOC, [page("Short text.", 7)], SMALL)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == f"{DOC}-000000"
    assert (chunk.document_id, chunk.text, chunk.page_number, chunk.chunk_index) == (
        DOC,
        "Short text.",
        7,
        0,
    )
    assert chunks == chunk_pages(DOC, [page("Short text.", 7)], SMALL)


def test_chunks_respect_size_and_keep_whole_sentences() -> None:
    text = sentences(30)
    chunks = chunk_pages(DOC, [page(text)], SMALL)
    whole = set(split_sentences(text))
    assert len(chunks) > 3
    for chunk in chunks:
        assert estimate_tokens(chunk.text) <= SMALL.chunk_size_tokens + 2
        assert set(split_sentences(chunk.text)) <= whole


def test_consecutive_chunks_overlap_and_lose_nothing() -> None:
    text = sentences(30)
    chunks = chunk_pages(DOC, [page(text)], SMALL)
    for left, right in zip(chunks, chunks[1:]):
        assert set(split_sentences(left.text)) & set(split_sentences(right.text))
    covered = {s for chunk in chunks for s in split_sentences(chunk.text)}
    assert covered == set(split_sentences(text))


def test_no_overlap_when_disabled() -> None:
    config = ChunkingConfig(chunk_size_tokens=40, chunk_overlap_tokens=0)
    chunks = chunk_pages(DOC, [page(sentences(30))], config)
    pieces = [s for chunk in chunks for s in split_sentences(chunk.text)]
    assert len(pieces) == len(set(pieces))


def test_chunks_never_cross_pages() -> None:
    chunks = chunk_pages(
        DOC, [page("Only on page one.", 1), page("", 2), page("Only on page three.", 3)], SMALL
    )
    assert [(c.page_number, c.chunk_index) for c in chunks] == [(1, 0), (3, 1)]
    assert "page three" not in chunks[0].text


def test_missing_page_numbers_stay_none() -> None:
    assert chunk_pages(DOC, [page(sentences(30), None)], SMALL)[0].page_number is None


def test_paragraph_boundary_is_preferred() -> None:
    first, second = sentences(4), sentences(4, start=10)
    config = ChunkingConfig(chunk_size_tokens=60, chunk_overlap_tokens=0)
    chunks = chunk_pages(DOC, [page(f"{first}\n\n{second}")], config)
    assert [chunk.text for chunk in chunks] == [first, second]


def test_paragraphs_and_table_rows_are_preserved_inside_a_chunk() -> None:
    text = "Intro line.\n\nName | Value\nNorth | 450"
    assert chunk_pages(DOC, [page(text)], SMALL)[0].text == text


def test_giant_sentence_is_split_without_losing_words() -> None:
    words = [f"word{n}" for n in range(200)]
    config = ChunkingConfig(chunk_size_tokens=40, chunk_overlap_tokens=0)
    chunks = chunk_pages(DOC, [page(" ".join(words))], config)
    assert len(chunks) > 1
    assert " ".join(chunk.text for chunk in chunks).split() == words


def test_text_without_spaces_is_split() -> None:
    config = ChunkingConfig(chunk_size_tokens=40, chunk_overlap_tokens=0)
    chunks = chunk_pages(DOC, [page("x" * 5000)], config)
    assert all(chunk.token_count <= config.chunk_size_tokens for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == "x" * 5000


def test_final_chunk_is_never_only_overlap() -> None:
    chunks = chunk_pages(DOC, [page(sentences(30))], SMALL)
    previous = set(split_sentences(chunks[-2].text))
    assert set(split_sentences(chunks[-1].text)) - previous


def test_passages_cover_the_chunk_and_stop_at_paragraphs() -> None:
    paragraphs = [sentences(4, start=n * 10) for n in range(3)]
    config = ChunkingConfig(chunk_size_tokens=500, chunk_overlap_tokens=0, passage_size_tokens=128)
    chunk = chunk_pages(DOC, [page("\n\n".join(paragraphs))], config)[0]
    assert list(chunk.passages) == paragraphs


def test_passages_respect_their_size() -> None:
    config = ChunkingConfig(chunk_size_tokens=500, chunk_overlap_tokens=0, passage_size_tokens=30)
    chunk = chunk_pages(DOC, [page(sentences(20))], config)[0]
    assert len(chunk.passages) > 3
    assert all(estimate_tokens(passage) <= 32 for passage in chunk.passages)
    assert " ".join(chunk.passages) == chunk.text


def test_short_heading_stays_with_its_paragraph() -> None:
    chunk = chunk_pages(DOC, [page("Methods\n\nWe ran a survey.")], SMALL)[0]
    assert chunk.passages == ("Methods\n\nWe ran a survey.",)


def test_empty_pages_give_no_chunks() -> None:
    assert chunk_pages(DOC, [page("")], SMALL) == []


def test_split_sentences_handles_abbreviations_and_decimals() -> None:
    text = (
        'Dr. Smith paid $3.50 for it, e.g. a pen. "Really?" He left! '
        "J. Doe stayed. 日本語です。次の文。"
    )
    assert split_sentences(text) == [
        "Dr. Smith paid $3.50 for it, e.g. a pen.",
        '"Really?"',
        "He left!",
        "J. Doe stayed.",
        "日本語です。",
        "次の文。",
    ]
