"""The start-time maps, drawn the moment the forecast writes its first history frame.

THE BREAKAGE THIS PREVENTS: the global and variable-resolution doors rendered their maps only after the whole forecast
(`forecast --no-products`, then `products`), so the first map of a 24 h run waited for all 24 hours to integrate.
On a g7e (RTX PRO 6000) box on 2026-10-23 the 60 km globe wrote its f000 frame 20 s into the forecast stage and
the f000 maps take 13 s to draw (71 pictures, all views); the run's first map still came only after the forecast
and the full products pass.

The forecast door lists each finished frame in ``history-ready.jsonl`` (its path, step and time, written after the
frame's durable sidecar), or, on an executor that keeps no such list, writes the sidecar itself (the variable-
resolution door's: 2026-10-23 its first look waited for a list that never came). FirstLook waits for the first, then draws that one frame with the products door in
a CPU-only process of its own, into the same tree and case the full products stage writes (same file names, so the
full pass replaces them with identical pictures). The cards are never touched; a forecast that ends without a frame
ends the wait. Nothing here can fail the run: a failed first look is reported, and the full pass draws every frame.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

READY = "history-ready.jsonl"
DURABLE = "*history.*.nc.durable.json"


class FirstLook(threading.Thread):
    def __init__(self, forecast: Path, *, grid: Path | str, init: Path | str, start_time: str, out: Path | str,
                 case: str, extra: Sequence[str] = (), poll_seconds: float = 0.5, log: Path | None = None):
        super().__init__(name="first-look", daemon=True)
        self.forecast, self.out, self.case = Path(forecast), Path(out), case
        self.argv = ["products", "--mesh", str(grid), "--init", str(init), "--simulation-start", start_time,
                     "--out", str(out), "--case", case, *extra]
        self.poll, self.log = poll_seconds, log or Path(forecast).parent / "first-look.log"
        self.stopped = threading.Event()
        self.result: dict = {"status": "waiting"}

    def _first_frame(self) -> Path | None:
        ready = self.forecast / READY
        while True:
            stopping = self.stopped.is_set()  # read once more after the stop, so a frame listed just before it counts
            try:
                for line in ready.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    if row.get("path"):
                        return Path(row["path"])
            except (OSError, ValueError):
                pass
            # Executors that write no history-ready.jsonl (the variable-resolution door's) still write each frame's
            # durable sidecar after the frame itself: the earliest frame whose sidecar names its full size is done.
            for sidecar in sorted(self.forecast.glob(DURABLE)):
                frame = sidecar.with_name(sidecar.name[: -len(".durable.json")])
                try:
                    if frame.stat().st_size == int(json.loads(sidecar.read_text(encoding="utf-8"))["bytes"]):
                        return frame
                except (OSError, ValueError, KeyError, TypeError):
                    continue
            if stopping:
                return None
            self.stopped.wait(self.poll)

    def run(self) -> None:
        began = time.monotonic()
        frame = self._first_frame()
        if frame is None:
            self.result = {"status": "no-frame"}
            return
        waited = time.monotonic() - began
        argv = [sys.executable, "-m", "hexcore.cli", self.argv[0], "--history", str(frame), *self.argv[1:]]
        drawn = time.monotonic()
        with self.log.open("w", encoding="utf-8") as stream:
            rc = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT).returncode
        self.result = {"status": "drawn" if rc == 0 else "failed", "returncode": rc, "frame": str(frame),
                       "frame_wait_seconds": round(waited, 1), "draw_seconds": round(time.monotonic() - drawn, 1),
                       "log": str(self.log)}

    def finish(self, timeout: float | None = None) -> dict:
        """Stop waiting for a frame (the forecast has ended) and wait for a draw in progress."""
        self.stopped.set()
        self.join(timeout)
        return dict(self.result)
