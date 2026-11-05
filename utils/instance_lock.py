"""Allow one running copy of the app per data folder.

Two copies would overwrite each other's search index, and each would treat the other's
uploads as interrupted. The operating system releases the lock when a process ends, even
after a crash, so a stale lock cannot keep the app from starting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO

_held: dict[Path, IO[bytes]] = {}


class AlreadyRunningError(Exception):
    """Another copy of the app is using this data folder."""


def _lock(handle: IO[bytes]) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def acquire_instance_lock(path: Path) -> None:
    """Hold ``path`` for the life of this process. Repeat calls from it change nothing."""
    if path in _held:
        return
    handle = path.open("a+b")
    try:
        _lock(handle)
    except OSError as exc:
        handle.close()
        raise AlreadyRunningError(str(path)) from exc
    _held[path] = handle
