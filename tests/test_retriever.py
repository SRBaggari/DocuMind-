"""Tests for indexing, similarity search, filtering and consistency with SQLite."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest
from conftest import HashingEmbedder

from config import Settings
from core.embedding_engine import EmbeddingModelError
from core.indexing import DocumentIndexer
from core.ingestion import DocumentIngestor, IngestOutcome
from core.retriever import Retriever
from core.vector_store import FaissBackend, NumpyBackend, VectorStore
from database.db_manager import DatabaseManager
from database.schema import MIGRATIONS, IndexStatus

CATS = b"Cats sleep on warm sofas and chase small mice at night."
ROCKETS = b"Rockets burn liquid fuel to reach orbit around the planet."


def indexed_ingestor_embeddings(indexer: DocumentIndexer, document_id: str):
    return indexer._dir(document_id) / "embeddings.npy"


def unit(*values: float) -> np.ndarray:
    vector = np.zeros(8, dtype=np.float32)
    vector[: len(values)] = values
    return vector / np.linalg.norm(vector)


@pytest.fixture(params=[FaissBackend, NumpyBackend])
def small_store(request, tmp_path) -> VectorStore:
    return VectorStore(tmp_path, 8, backend=request.param(8))


def test_search_ranks_by_cosine(small_store) -> None:
    small_store.add([10, 20, 30], np.stack([unit(1), unit(1, 1), unit(0, 1)]))
    results = small_store.search(unit(1), 3)
    assert [vector_id for vector_id, _ in results] == [10, 20, 30]
    assert [round(score, 4) for _, score in results] == [1.0, 0.7071, 0.0]


def test_top_k_and_more_than_available(small_store) -> None:
    small_store.add([1, 2], np.stack([unit(1), unit(0, 1)]))
    assert len(small_store.search(unit(1), 1)) == 1
    assert len(small_store.search(unit(1), 50)) == 2


def test_search_on_empty_store(small_store) -> None:
    assert small_store.search(unit(1), 5) == []


def test_filter_limits_results(small_store) -> None:
    small_store.add([1, 2, 3], np.stack([unit(1), unit(1, 0.1), unit(0, 1)]))
    assert {i for i, _ in small_store.search(unit(1), 5, allowed={2, 3})} == {2, 3}
    assert small_store.search(unit(1), 5, allowed=set()) == []


def test_adding_an_existing_id_replaces_it(small_store) -> None:
    small_store.add([1], unit(1)[None, :])
    small_store.add([1], unit(0, 1)[None, :])
    assert small_store.ids() == {1}
    assert small_store.search(unit(0, 1), 1)[0][1] == pytest.approx(1.0)


def test_remove(small_store) -> None:
    small_store.add([1, 2], np.stack([unit(1), unit(0, 1)]))
    small_store.remove({1})
    assert small_store.ids() == {2}


@pytest.mark.parametrize("backend", [FaissBackend, NumpyBackend])
def test_store_survives_reload(tmp_path, backend) -> None:
    first = VectorStore(tmp_path, 8, backend=backend(8))
    first.add([5], unit(1)[None, :])
    first.save("model-x")
    second = VectorStore(tmp_path, 8, backend=backend(8))
    assert second.ids() == {5} and second.model_name == "model-x"
    assert second.search(unit(1), 1)[0][0] == 5


def test_corrupt_index_file_starts_empty(tmp_path) -> None:
    first = VectorStore(tmp_path, 8, backend=FaissBackend(8))
    first.add([5], unit(1)[None, :])
    first.save("m")
    (tmp_path / "vectors.faiss").write_bytes(b"garbage")
    assert VectorStore(tmp_path, 8, backend=FaissBackend(8)).ids() == set()


def test_index_path_with_non_ascii_characters(tmp_path) -> None:
    folder = tmp_path / "données 日本"
    store = VectorStore(folder, 8, backend=FaissBackend(8))
    store.add([1], unit(1)[None, :])
    store.save("m")
    assert VectorStore(folder, 8, backend=FaissBackend(8)).ids() == {1}


def test_retrieval_returns_source_details(indexed_ingestor, retriever) -> None:
    indexed_ingestor.ingest("rockets.txt", ROCKETS)
    doc = indexed_ingestor.ingest("cats.txt", CATS).document
    best = retriever.retrieve("warm sofas and mice")[0]
    assert (best.filename, best.document_id) == ("cats.txt", doc.document_id)
    assert best.text == CATS.decode() and best.chunk_id.startswith(doc.document_id)


def test_fact_inside_a_mixed_chunk_is_found_by_its_passage(
    indexed_ingestor, retriever
) -> None:
    filler = " ".join(f"Topic{n} words number{n} filler{n} content{n}." for n in range(25))
    text = f"{filler}\n\nMeals are reimbursed up to sixty dollars daily.\n\n{filler}"
    indexed_ingestor.ingest("policy.txt", text.encode())
    best = retriever.retrieve("meals reimbursed sixty dollars")[0]
    assert "Meals are reimbursed" in best.passage
    assert len(best.passage) < len(best.text)


def test_one_chunk_appears_once_however_many_passages_match(indexed_ingestor, retriever) -> None:
    text = "\n\n".join(f"Shared keyword paragraph {n}. " * 8 for n in range(4))
    doc = indexed_ingestor.ingest("a.txt", text.encode()).document
    hits = retriever.retrieve("shared keyword paragraph")
    assert len({hit.chunk_id for hit in hits}) == len(hits) == doc.chunk_count


def test_pdf_hit_cites_the_right_page(indexed_ingestor, retriever, make_pdf) -> None:
    pages = ["Nothing to see here at all.", "", "Penguins swim in icy water."]
    indexed_ingestor.ingest("a.pdf", make_pdf(pages).read_bytes())
    assert retriever.retrieve("penguins icy water")[0].page_number == 3


def test_document_filter(indexed_ingestor, retriever) -> None:
    cats = indexed_ingestor.ingest("cats.txt", CATS).document
    rockets = indexed_ingestor.ingest("rockets.txt", ROCKETS).document
    query = "cats chase mice"
    only = retriever.retrieve(query, [rockets.document_id], min_similarity=-1)
    assert {hit.document_id for hit in only} == {rockets.document_id}
    assert retriever.retrieve(query, [cats.document_id])[0].filename == "cats.txt"
    assert retriever.retrieve(query, []) == []


def test_unrelated_query_returns_nothing(indexed_ingestor, retriever) -> None:
    indexed_ingestor.ingest("cats.txt", CATS)
    assert retriever.retrieve("quantum tax legislation") == []
    assert retriever.retrieve("   ") == []


def test_document_metadata_records_index(indexed_ingestor, embedder) -> None:
    doc = indexed_ingestor.ingest("cats.txt", CATS).document
    assert doc.index_status is IndexStatus.INDEXED
    assert doc.chunk_count == 1 and doc.embedding_model == embedder.model_name


def test_vector_ids_map_to_their_own_chunks(indexed_ingestor, indexer, db, store) -> None:
    text = "\n\n".join(f"Unique subject{n} paragraph. " * 60 for n in range(6))
    doc = indexed_ingestor.ingest("long.txt", text.encode()).document
    assert doc.chunk_count > 3
    assert store.ids() == db.get_vector_ids()
    rows = list(db.get_vector_rows(db.get_vector_ids()).values())
    embeddings = np.load(indexed_ingestor_embeddings(indexer, doc.document_id))
    assert sorted(row.embedding_row for row in rows) == list(range(len(embeddings)))
    for row in rows:
        chunk = indexer.load_chunks(doc.document_id)[row.chunk_index]
        assert chunk.chunk_id == row.chunk_id
        expected = indexer._embedder.embed_query(chunk.passages[row.passage_index])
        assert np.allclose(embeddings[row.embedding_row], expected)


def test_delete_removes_vectors_and_never_reuses_ids(
    indexed_ingestor, db, store, retriever
) -> None:
    keep = indexed_ingestor.ingest("rockets.txt", ROCKETS).document
    gone = indexed_ingestor.ingest("cats.txt", CATS).document
    gone_ids = db.get_vector_ids([gone.document_id])
    indexed_ingestor.delete(gone.document_id)
    assert store.ids() == db.get_vector_ids([keep.document_id])
    assert retriever.retrieve("warm sofas mice") == []
    new = indexed_ingestor.ingest("new.txt", b"Brand new text about gardens.").document
    assert not gone_ids & db.get_vector_ids([new.document_id])


def test_reindexing_reuses_embeddings(indexed_ingestor, indexer, embedder, db, store) -> None:
    doc = indexed_ingestor.ingest("cats.txt", CATS).document
    calls = embedder.calls
    pages = indexed_ingestor.load_content(doc.document_id)["pages"]
    indexer.index_document(doc.document_id, pages)
    assert embedder.calls == calls
    assert store.ids() == db.get_vector_ids() and len(store.ids()) == 1


def restart(
    settings: Settings, embedder: HashingEmbedder
) -> tuple[DatabaseManager, VectorStore, DocumentIndexer]:
    db = DatabaseManager(settings.paths.database_path)
    store = VectorStore(settings.paths.indexes_dir, settings.embedding.dimension)
    return db, store, DocumentIndexer(db, store, embedder, settings)


def test_search_works_after_restart(indexed_ingestor, test_settings, embedder) -> None:
    indexed_ingestor.ingest("cats.txt", CATS)
    db, store, indexer = restart(test_settings, embedder)
    calls = embedder.calls
    assert indexer.reconcile().added == 0
    fresh = Retriever(db, store, embedder, indexer, test_settings.retrieval)
    assert fresh.retrieve("warm sofas")[0].filename == "cats.txt"
    assert embedder.calls == calls


def test_lost_index_is_rebuilt_without_embedding(
    indexed_ingestor, test_settings, embedder
) -> None:
    indexed_ingestor.ingest("cats.txt", CATS)
    for path in test_settings.paths.indexes_dir.iterdir():
        path.unlink()
    db, store, indexer = restart(test_settings, embedder)
    calls = embedder.calls
    assert indexer.reconcile().added == 1
    assert store.ids() == db.get_vector_ids() and embedder.calls == calls


def test_reconcile_removes_stale_vectors(indexed_ingestor, db, store, indexer) -> None:
    indexed_ingestor.ingest("cats.txt", CATS)
    store.add([9999], np.ones((1, 384), dtype=np.float32) / np.sqrt(384))
    assert indexer.reconcile().removed == 1
    assert store.ids() == db.get_vector_ids()


def test_crash_between_commit_and_index_is_repaired(
    indexed_ingestor, store, indexer, db
) -> None:
    indexed_ingestor.ingest("cats.txt", CATS)
    store.remove(store.ids())
    assert indexer.reconcile().added == 1
    assert store.search(indexer._embedder.embed_query(CATS.decode()), 1)[0][1] > 0.99
    assert store.ids() == db.get_vector_ids()


def test_missing_embeddings_file_marks_document_for_indexing(
    indexed_ingestor, store, indexer, db, test_settings
) -> None:
    doc = indexed_ingestor.ingest("cats.txt", CATS).document
    (test_settings.paths.processed_dir / doc.document_id / "embeddings.npy").unlink()
    store.remove(store.ids())
    assert indexer.reconcile().documents_needing_index == 1
    assert db.get_document(doc.document_id).index_status is IndexStatus.PENDING
    assert db.get_vector_ids() == set()


def test_changing_model_requires_reindexing(indexed_ingestor, test_settings, db) -> None:
    doc = indexed_ingestor.ingest("cats.txt", CATS).document
    other = HashingEmbedder("another-model")
    _, store, indexer = restart(test_settings, other)
    assert indexer.reconcile().documents_needing_index == 1
    assert store.ids() == set()
    assert indexer.index_unindexed() == 1
    assert other.calls == 1
    assert db.get_document(doc.document_id).embedding_model == "another-model"


class UnavailableEmbedder(HashingEmbedder):
    def embed_texts(self, texts, progress=None):
        raise EmbeddingModelError("Model not available.")


def test_missing_model_keeps_document_and_can_retry(db, store, test_settings) -> None:
    broken = DocumentIndexer(db, store, UnavailableEmbedder(), test_settings)
    result = DocumentIngestor(db, test_settings, broken).ingest("cats.txt", CATS)
    assert result.outcome is IngestOutcome.READY
    assert "Model not available" in result.message
    assert result.document.index_status is IndexStatus.PENDING
    working = DocumentIndexer(db, store, HashingEmbedder(), test_settings)
    assert working.index_unindexed() == 1
    assert db.list_unindexed() == [] and len(store.ids()) == 1


def test_database_upgrades_from_phase_two(tmp_path) -> None:
    path = tmp_path / "old.db"
    connection = sqlite3.connect(path)
    connection.executescript(MIGRATIONS[0] + "PRAGMA user_version = 1;")
    connection.execute(
        "INSERT INTO documents VALUES ('a', 'a.txt', 'txt', 1, 'h', 't', 'ready',"
        " NULL, '[]', NULL, 1, 1, 1, 't')"
    )
    connection.commit()
    connection.close()
    db = DatabaseManager(path)
    assert [doc.index_status for doc in db.list_unindexed()] == [IndexStatus.PENDING]
