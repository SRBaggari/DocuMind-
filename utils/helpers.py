"""Small shared helpers: hashing, IDs, timestamps, formatting and atomic writes."""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def new_document_id() -> str:
    return uuid.uuid4().hex


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def format_timestamp(iso_timestamp: str) -> str:
    """Show a stored UTC timestamp in the computer's local time."""
    moment = datetime.fromisoformat(iso_timestamp).astimezone()
    return f"{moment.day} {moment:%b %Y, %H:%M}"


def directory_size(path: Path) -> int:
    """Return the total size of the files under ``path``, or 0 if it does not exist."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` so that ``path`` never holds a partial file after a crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
