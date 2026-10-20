"""Observe actual forecast completion without holding up other finished decks."""
import json
from pathlib import Path
import time


def completed_specs(paths):
    pending = list(paths)
    while pending:
        ready = next((path for path in pending if Path(
            json.loads(path.read_text())["model_exit"]).is_file()), None)
        if ready is None:
            time.sleep(15)
            continue
        pending.remove(ready)
        yield ready
