"""Cold receipts must observe the page cache, not only request eviction."""
import os
import sys

import pytest

from bench_projected_intermediate import evict


@pytest.mark.skipif(sys.platform != "linux", reason="C3 component measurements run on Linux boxes")
def test_real_input_file_is_nonresident_after_eviction(tmp_path):
    path = tmp_path / "input.grib2"
    with path.open("wb") as handle:
        handle.write(b"input-cache-residency" * 4096)
        handle.flush()
        os.fsync(handle.fileno())
    assert path.read_bytes().startswith(b"input-cache-residency")
    rows = evict([path])
    assert rows[0]["resident_pages"] == 0
    assert rows[0]["pages"] > 0
    assert rows[0]["bytes"] == path.stat().st_size
