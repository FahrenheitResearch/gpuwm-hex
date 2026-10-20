"""Keep owned hard links to three forecast histories until sheets are rendered.

The verify-mpas door retires its fresh output after recording the full grade.
This observer preserves file inodes only; Rust still owns weather decoding.
Every retained history must match the door's final provenance hash before use.
An observer miss or a replaced inode fails sheet admission, never the word bar.
"""
from pathlib import Path
import datetime
import hashlib
import os
import threading


class SheetInputs:
    def __init__(self, output, retained, start):
        self.output = Path(output)
        self.retained = Path(retained)
        time = datetime.datetime.strptime(start, '%Y-%m-%d_%H:%M:%S')
        self.names = ['output.' + (time + datetime.timedelta(hours=lead)).strftime(
            '%Y-%m-%d_%H.%M.%S') + '.nc' for lead in (6, 12, 24)]
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.observe, daemon=True)

    def observe(self):
        try:
            self.retained.mkdir(parents=True, exist_ok=False)
            self.ready.set()
            while True:
                for name in self.names:
                    source = self.output / name
                    target = self.retained / name
                    try:
                        stat = source.stat()
                    except FileNotFoundError:
                        continue
                    if target.exists():
                        previous = target.stat()
                        if (previous.st_dev, previous.st_ino) == (stat.st_dev, stat.st_ino):
                            continue
                        target.unlink()
                    try:
                        os.link(source, target)
                    except FileNotFoundError:
                        # A source can disappear during the door's retirement.
                        # Final provenance authentication detects any lost file.
                        continue
                if self.stop.wait(0.25):
                    break
        except Exception as error:
            self.error = error
            self.ready.set()

    def start(self):
        self.thread.start()
        self.ready.wait()
        if self.error:
            raise self.error

    def finish(self, grade):
        self.stop.set()
        self.thread.join()
        if self.error:
            raise self.error
        hashes = grade.get('provenance', {}).get('hex_output_sha256', {})
        admitted = {}
        for name in self.names:
            path = self.retained / name
            if name not in hashes or not path.is_file():
                raise ValueError('Required forecast sheet history was not retained: ' + name)
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != hashes[name]:
                raise ValueError('Retained sheet history differs from the actual word proof: ' + name)
            admitted[name] = actual
        return admitted
