"""Mapped history handoff preserves words and isolates child-side mutations."""
import pickle

import numpy as np
import pytest

from hexcore import history_spool


@pytest.mark.parametrize("order", ["C", "F", "strided"])
def test_words_types_aliases_and_child_mutations(tmp_path, monkeypatch, order):
    monkeypatch.setenv("GPUWM_HEX_HOST_COPY_THREADS", "4")
    words = np.array([0, 0x80000000, 0x7fc01234, 0x3f800000] * 300, dtype=np.uint32)
    array = words.view(np.float32).reshape(40, 30)
    if order == "F":
        array = array.copy(order="F")
    elif order == "strided":
        array = array[:, ::3]
    original = array.tobytes()
    payload = {"a": array, "same": array, "big_endian": np.arange(13, dtype=">i8"),
               "empty": np.empty((2, 0), dtype=np.float64), "step": 90}
    path = tmp_path / "frame.pkl"
    stats = history_spool.dump_payload(payload, path)
    before = path.with_suffix(".buffers").read_bytes()
    actual = history_spool.load_payload(path)
    assert actual["same"] is actual["a"]
    for name in ("a", "big_endian", "empty"):
        assert actual[name].dtype == payload[name].dtype
        assert actual[name].shape == payload[name].shape
        assert actual[name].tobytes() == payload[name].tobytes()
    actual["a"][0, 0] = 17
    assert array.tobytes() == original
    assert path.with_suffix(".buffers").read_bytes() == before
    assert actual["step"] == 90 and stats["mode"] == "mapped-protocol5"


@pytest.mark.parametrize("payload", [{"step": 10}, {"empty": np.empty(0)}])
def test_payload_without_nonempty_buffers(tmp_path, payload):
    path = tmp_path / "frame.pkl"
    history_spool.dump_payload(payload, path)
    actual = history_spool.load_payload(path)
    assert pickle.dumps(actual, protocol=5) == pickle.dumps(payload, protocol=5)


def test_partial_writes_are_completed(tmp_path, monkeypatch):
    write = history_spool.os.pwrite
    monkeypatch.setattr(history_spool.os, "pwrite", lambda fd, data, off: write(fd, data[:7], off))
    path = tmp_path / "frame.pkl"
    array = np.arange(101, dtype=np.float64)
    history_spool.dump_payload(array, path)
    assert history_spool.load_payload(path).tobytes() == array.tobytes()


def test_failed_write_is_not_published_as_a_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(history_spool.os, "pwrite", lambda *args: 0)
    path = tmp_path / "frame.pkl"
    with pytest.raises(OSError, match="made no progress"):
        history_spool.dump_payload(np.ones(100), path)
    assert not path.exists()
