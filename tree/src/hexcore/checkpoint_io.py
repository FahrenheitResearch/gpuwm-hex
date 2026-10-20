"""Publish complete forecast checkpoints at their discoverable filename."""
import os
from pathlib import Path
import pickle


def write_checkpoint(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("xb") as stream:
        pickle.dump(payload, stream, protocol=5)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
