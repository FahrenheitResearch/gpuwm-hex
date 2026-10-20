"""Advisory file locks that work on POSIX and on Windows.

THE BREAKAGE THIS PREVENTS (hex 0.4.1 doors, Windows desktop, 2026-10-20):
the asset fetch, physics-resources fetch, init, s2s and cycle doors imported
``fcntl``, which Windows does not have, so ``gpuwm-hex fetch-assets`` died with
``ModuleNotFoundError: No module named 'fcntl'`` on every Windows install.
Windows has no shared lock through ``msvcrt``, so a shared request is taken
exclusively there: the same exclusion, less read concurrency.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import time


def lock(stream, *, shared: bool = False, blocking: bool = True) -> None:
    """Lock an open file; non-blocking and held elsewhere raises BlockingIOError."""
    if os.name == "nt":
        import msvcrt
        while True:
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if not blocking:
                    raise BlockingIOError(f"{getattr(stream, 'name', stream)} is locked by another process") from None
                time.sleep(0.05)
    import fcntl
    fcntl.flock(stream, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | (0 if blocking else fcntl.LOCK_NB))


def unlock(stream) -> None:
    if os.name == "nt":
        import msvcrt
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        return
    import fcntl
    fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def held(path: Path, *, mode: str = "a"):
    """Hold an exclusive lock on ``path`` (created with its parents) for the block."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode) as stream:
        lock(stream)
        try:
            yield stream
        finally:
            unlock(stream)
