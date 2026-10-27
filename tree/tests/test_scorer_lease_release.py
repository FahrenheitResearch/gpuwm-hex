"""The automatic observation scorer must start CPU-only, so a card lease is not held for it.

Breakage (U-H7): the scorer watcher (hexcore.observation_scorecard watch) starts after the forecast's RECEIPT
inside the leased command's process tree. The pool runners keep a lease until every descendant exits unless
every process left was started with CUDA_VISIBLE_DEVICES set and empty (release_for_cpu_only_tail); a watcher
started with the card visible held a 32,000 MiB RTX 5090 lease ~5 min after RECEIPT (mz24-a32-arwen-lts,
2026-10-27). This pins the watcher's half of that contract: an empty CUDA_VISIBLE_DEVICES in its own environment,
in its own session, with no stdout/stderr pipe back to the door.
"""
import os
import subprocess

from hexcore import observation_scorecard


def test_watcher_starts_with_an_empty_cuda_visible_devices(monkeypatch, tmp_path):
    (tmp_path / "observation-scorecard").mkdir()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    seen = {}

    class Started:
        pid = 4242

    def popen(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        return Started()

    monkeypatch.setattr(observation_scorecard.subprocess, "Popen", popen)
    observation_scorecard.queue_retry(tmp_path, {"start": "2026-10-25_12:00:00"})

    assert seen["argv"][1:4] == ["-m", "hexcore.observation_scorecard", "watch"]
    env = seen["env"]
    assert "CUDA_VISIBLE_DEVICES" in env and env["CUDA_VISIBLE_DEVICES"] == "", env.get("CUDA_VISIBLE_DEVICES")
    assert seen["stdin"] is subprocess.DEVNULL
    assert seen["stdout"] not in (None, subprocess.PIPE) and seen["stderr"] not in (None, subprocess.PIPE)
    if os.name != "nt":
        assert seen["start_new_session"] is True
