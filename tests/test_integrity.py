"""Consistency between SQLite, the vector index and files, under failure and concurrency."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from conftest import HashingEmbedder

from core.embedding_engine import EmbeddingModelError
from core.indexing import DocumentIndexer, document_dir
from core.ingestion import DocumentIngestor
from core.vector_store import VectorStore
from database.schema import IndexStatus
from utils.instance_lock import acquire_instance_lock

TEXT = b"The Riverside authority reached revenue of 148 million dollars in 2024."
OTHER = b"Rockets burn liquid fuel to reach orbit around the planet every month."
QUERY = "The Riverside authority reached revenue of dollars"


class RenamedEmbedder(HashingEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "another-model"


class BrokenEmbedder(RenamedEmbedder):
    def embed_query(self, query):
        raise EmbeddingModelError("No such model.")


class GatedEmbedder(HashingEmbedder):
    """Holds embedding open, so another thread can act while a document is indexing."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def embed_texts(self, texts, progress=None):
        self.started.set()
        self.release.wait(10)
        return super().embed_texts(texts, progress)


def indexed_files(settings) -> list[Path]:
    return [path for path in settings.paths.processed_dir.rglob("*") if path.is_file()]


def test_delete_during_indexing_leaves_nothing_behind(db, store, test_settings) -> None:
    embedder = GatedEmbedder()
    indexer = DocumentIndexer(db, store, embedder, test_settings)
    ingestor = DocumentIngestor(db, test_settings, indexer)
    upload = threading.Thread(target=ingestor.ingest, args=("a.txt", TEXT))
    upload.start()
    assert embedder.started.wait(10)
    document = db.list_documents()[0]
    remover = threading.Thread(target=ingestor.delete, args=(document.document_id,))
    remover.start()
    embedder.release.set()
    upload.join(10)
    remover.join(10)

    assert db.list_documents() == []
    assert indexed_files(test_settings) == []
    assert store.ids() == set() and db.get_vector_ids() == set()


def test_indexing_a_deleted_document_writes_nothing(db, indexer, ingestor, test_settings) -> None:
    document = ingestor.ingest("a.txt", TEXT).document
    pages = ingestor.load_content(document.document_id)["pages"]
    ingestor.delete(document.document_id)
    assert indexer.index_document(document.document_id, pages) == 0
    assert indexed_files(test_settings) == []


def test_delete_succeeds_when_a_file_is_locked(
    db, indexed_ingestor, store, test_settings, monkeypatch
) -> None:
    document = indexed_ingestor.ingest("a.txt", TEXT).document
    keep = indexed_ingestor.ingest("b.txt", OTHER).document

    def refuse(*_args, **_kwargs):
        raise PermissionError("in use")

    monkeypatch.setattr("core.ingestion.shutil.rmtree", refuse)
    assert indexed_ingestor.delete(document.document_id)
    monkeypatch.undo()

    assert db.get_document(document.document_id) is None
    assert store.ids() == db.get_vector_ids([keep.document_id])
    assert document_dir(test_settings.paths, document.document_id).exists()
    indexed_ingestor.sweep_orphans()
    assert not document_dir(test_settings.paths, document.document_id).exists()
    assert document_dir(test_settings.paths, keep.document_id).exists()


def test_clear_all_continues_past_a_failure_and_clears_history(
    db, indexed_ingestor, test_settings, monkeypatch
) -> None:
    first = indexed_ingestor.ingest("a.txt", TEXT).document
    indexed_ingestor.ingest("b.txt", OTHER)
    db.log_activity("upload", "Uploaded a.txt")
    real_delete = indexed_ingestor.delete

    def flaky(document_id):
        if document_id == first.document_id:
            raise RuntimeError("Unexpected.")
        return real_delete(document_id)

    monkeypatch.setattr(indexed_ingestor, "delete", flaky)
    removed, _left = indexed_ingestor.delete_all()
    assert removed == 1 and db.recent_activity(5) == []


def test_sweep_removes_orphans_and_temp_files_only(ingestor, test_settings) -> None:
    keep = ingestor.ingest("a.txt", TEXT).document
    paths = test_settings.paths
    orphan_id = "f" * 32
    (paths.uploads_dir / f"{orphan_id}.txt").write_bytes(b"secret")
    (paths.processed_dir / orphan_id).mkdir()
    (paths.processed_dir / orphan_id / "content.json").write_text("{}", encoding="utf-8")
    (paths.uploads_dir / ".tmp-abc").write_bytes(b"secret")
    (document_dir(paths, keep.document_id) / ".tmp-def").write_bytes(b"secret")
    (paths.uploads_dir / ".gitkeep").write_bytes(b"")
    (paths.uploads_dir / "notes.txt").write_bytes(b"not ours")

    ingestor.sweep_orphans(include_temp=True)

    names = {path.name for path in paths.uploads_dir.iterdir()}
    assert names == {f"{keep.document_id}.txt", ".gitkeep", "notes.txt"}
    assert [p.name for p in paths.processed_dir.iterdir()] == [keep.document_id]
    assert not list(document_dir(paths, keep.document_id).glob(".tmp-*"))


def test_temp_files_survive_a_sweep_while_running(ingestor, test_settings) -> None:
    temp = test_settings.paths.uploads_dir / ".tmp-live"
    temp.write_bytes(b"an upload in progress")
    ingestor.sweep_orphans()
    assert temp.exists()


def test_failed_save_does_not_leave_orphan_vectors(
    db, store, indexer, ingestor, monkeypatch
) -> None:
    document = ingestor.ingest("a.txt", TEXT).document
    pages = ingestor.load_content(document.document_id)["pages"]

    def fail(_model):
        raise OSError("Disk full.")

    monkeypatch.setattr(store, "save", fail)
    with pytest.raises(OSError):
        indexer.index_document(document.document_id, pages)
    monkeypatch.undo()
    assert store.ids() == set()


@pytest.mark.parametrize("damage", ["meta", "index", "both"])
def test_changed_model_with_a_lost_index_never_restores_old_vectors(
    db, indexed_ingestor, test_settings, damage
) -> None:
    document = indexed_ingestor.ingest("a.txt", TEXT).document
    directory = test_settings.paths.indexes_dir
    for path in directory.iterdir():
        is_meta = path.suffix == ".json"
        if damage == "both" or (damage == "meta") == is_meta:
            path.unlink()

    store = VectorStore(directory, test_settings.embedding.dimension)
    indexer = DocumentIndexer(db, store, RenamedEmbedder(), test_settings)
    indexer.reconcile()
    assert db.get_document(document.document_id).index_status is IndexStatus.PENDING
    assert store.ids() == set()


def test_unusable_model_name_keeps_the_index(db, store, indexed_ingestor, test_settings) -> None:
    document = indexed_ingestor.ingest("a.txt", TEXT).document
    before = store.ids()
    assert before
    reopened = VectorStore(test_settings.paths.indexes_dir, test_settings.embedding.dimension)
    DocumentIndexer(db, reopened, BrokenEmbedder(), test_settings).reconcile()
    assert db.get_document(document.document_id).index_status is IndexStatus.INDEXED
    assert reopened.ids() == before


def test_one_failing_document_does_not_block_the_others(db, ingestor, store, test_settings) -> None:
    bad = ingestor.ingest("bad.txt", TEXT).document
    good = ingestor.ingest("good.txt", OTHER).document

    class Poisoned(HashingEmbedder):
        def embed_texts(self, texts, progress=None):
            if any("Riverside" in text for text in texts):
                raise RuntimeError("Out of memory.")
            return super().embed_texts(texts, progress)

    indexer = DocumentIndexer(db, store, Poisoned(), test_settings)
    assert indexer.index_unindexed() == 1
    assert db.get_document(good.document_id).index_status is IndexStatus.INDEXED
    assert db.get_document(bad.document_id).index_status is IndexStatus.FAILED


def test_a_missing_model_still_stops_the_whole_batch(db, ingestor, store, test_settings) -> None:
    ingestor.ingest("a.txt", TEXT)

    class Missing(HashingEmbedder):
        def embed_texts(self, texts, progress=None):
            raise EmbeddingModelError("Missing.")

    with pytest.raises(EmbeddingModelError):
        DocumentIndexer(db, store, Missing(), test_settings).index_unindexed()


@pytest.mark.parametrize("content", [b"", None], ids=["empty", "wrong-width"])
def test_bad_embeddings_file_cannot_stop_startup(
    db, indexed_ingestor, embedder, test_settings, content
) -> None:
    document = indexed_ingestor.ingest("a.txt", TEXT).document
    path = document_dir(test_settings.paths, document.document_id) / "embeddings.npy"
    if content is None:
        np.save(path, np.zeros((5, 7), dtype=np.float32))
    else:
        path.write_bytes(content)
    for item in test_settings.paths.indexes_dir.iterdir():
        item.unlink()
    store = VectorStore(test_settings.paths.indexes_dir, test_settings.embedding.dimension)
    DocumentIndexer(db, store, embedder, test_settings).reconcile()
    assert db.get_document(document.document_id).index_status is IndexStatus.PENDING


def test_second_copy_is_refused_and_a_finished_copy_releases_the_lock(tmp_path) -> None:
    lock = tmp_path / "documind.lock"
    acquire_instance_lock(lock)
    acquire_instance_lock(lock)
    root = Path(__file__).parent.parent
    script = (
        f"import sys; sys.path.insert(0, {str(root)!r})\n"
        "from pathlib import Path\n"
        "from utils.instance_lock import AlreadyRunningError, acquire_instance_lock\n"
        "try:\n"
        f"    acquire_instance_lock(Path({str(lock)!r}))\n"
        "except AlreadyRunningError:\n"
        "    sys.exit(3)\n"
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=30)
    assert run.returncode == 3, run.stderr


def test_lock_is_free_after_its_owner_exits(tmp_path) -> None:
    lock = tmp_path / "other.lock"
    root = Path(__file__).parent.parent
    script = (
        f"import sys; sys.path.insert(0, {str(root)!r})\n"
        "from pathlib import Path\n"
        "from utils.instance_lock import acquire_instance_lock\n"
        f"acquire_instance_lock(Path({str(lock)!r}))\n"
        "import os; os._exit(0)\n"
    )
    assert subprocess.run([sys.executable, "-c", script], timeout=30).returncode == 0
    acquire_instance_lock(lock)


def test_search_during_delete_never_fails(db, indexed_ingestor, retriever) -> None:
    documents = [
        indexed_ingestor.ingest(f"{n}.txt", TEXT + f" Item {n}.".encode()).document
        for n in range(6)
    ]
    errors: list[Exception] = []

    def search() -> None:
        try:
            for _ in range(40):
                retriever.retrieve(QUERY)
        except Exception as exc:
            errors.append(exc)

    reader = threading.Thread(target=search)
    reader.start()
    for document in documents:
        indexed_ingestor.delete(document.document_id)
    reader.join(20)
    assert errors == [] and db.list_documents() == []
