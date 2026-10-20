"""Preserve opaque CLI output inodes before the verifier retires its directory.

No model fields are read or transformed. The stored-word proof must separately
authenticate every retained file against the actual verifier's recorded hash.
"""
from pathlib import Path
import os
import stat
import threading


class OutputArchive:
    def __init__(self, source, target, *, interval=0.05):
        self.source, self.target = Path(source), Path(target)
        self.interval = interval
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _snapshot(self):
        try:
            paths = list(self.source.iterdir())
        except FileNotFoundError:
            return
        for source in paths:
            try:
                info = source.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    continue
                target = self.target / source.name
                try:
                    previous = target.stat(follow_symlinks=False)
                except FileNotFoundError:
                    previous = None
                if previous and (previous.st_dev, previous.st_ino) == (info.st_dev, info.st_ino):
                    continue
                temporary = self.target / ('.retaining-' + source.name)
                try:
                    temporary.unlink(missing_ok=True)
                    os.link(source, temporary, follow_symlinks=False)
                    temporary.replace(target)
                except FileNotFoundError:
                    temporary.unlink(missing_ok=True)
            except FileNotFoundError:
                continue

    def _run(self):
        try:
            self._snapshot()
            self._ready.set()
            while not self._stop.wait(self.interval):
                self._snapshot()
            self._snapshot()
        except BaseException as error:
            self._error = error
            self._ready.set()

    def __enter__(self):
        self.target.mkdir(parents=True, exist_ok=False)
        self._thread.start()
        self._ready.wait()
        if self._error:
            raise self._error
        return self

    def __exit__(self, *unused):
        self._stop.set()
        self._thread.join()
        if self._error:
            raise self._error
