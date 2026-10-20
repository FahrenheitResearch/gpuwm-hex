"""The preset instrument digests each large input once per file version.

Breakage this guards: the instrument read the whole 66 GB conus3km input set
before the run and again after it (232.6 s on G3, 2026-10-12), so the reuse
must not hide a changed input.
"""
import hashlib
import os
import sys
import time
from types import SimpleNamespace

import pytest

from hexcore import file_digest
from hexcore.drivers import forecast_receipt


def _inputs(tmp_path):
    lbc = tmp_path / "lbc"
    lbc.mkdir()
    files = {"grid": tmp_path / "grid.nc", "static": tmp_path / "static.nc", "init": tmp_path / "init.nc"}
    for index, path in enumerate([*files.values(), lbc / "lbc.2026-10-03_12.00.00.nc", lbc / "lbc.2026-10-03_15.00.00.nc"]):
        path.write_bytes(bytes([index]) * 4096)
        os.utime(path, (1_700_000_000, 1_700_000_000))
    return SimpleNamespace(**files, lbc_dir=lbc)


def _counting(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUWM_HEX_DIGEST_CACHE", str(tmp_path / "digests"))
    monkeypatch.setattr(file_digest, "MIN_CACHED_BYTES", 0)
    reads = []
    real = file_digest._hash
    monkeypatch.setattr(file_digest, "_hash", lambda path: reads.append(path) or real(path))
    return reads


def test_the_after_snapshot_of_unchanged_inputs_reads_nothing_again(tmp_path, monkeypatch):
    arguments = _inputs(tmp_path)
    reads = _counting(monkeypatch, tmp_path)
    before = forecast_receipt.input_snapshot(arguments)
    assert len(before) == 5 and len(reads) == 5
    for name, record in before.items():
        data = open(name, "rb").read()
        assert record == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    after = forecast_receipt.input_snapshot(arguments)
    assert after == before and len(reads) == 5


@pytest.mark.skipif(sys.platform != "linux", reason="st_ctime is the status-change time only on POSIX; hex runs on Linux")
def test_a_rewritten_input_is_read_again_and_its_digest_differs(tmp_path, monkeypatch):
    arguments = _inputs(tmp_path)
    reads = _counting(monkeypatch, tmp_path)
    before = forecast_receipt.input_snapshot(arguments)
    # Same size, same modification time set back: only the status-change
    # time (which no user call can set back) tells the versions apart.  The
    # pause outlasts the kernel's coarse timestamp tick.
    time.sleep(0.05)
    arguments.init.write_bytes(b"\xff" * 4096)
    os.utime(arguments.init, (1_700_000_000, 1_700_000_000))
    after = forecast_receipt.input_snapshot(arguments)
    assert len(reads) == 6 and reads[-1].name == "init.nc"
    assert after[str(arguments.init)]["sha256"] == hashlib.sha256(b"\xff" * 4096).hexdigest()
    assert after != before
