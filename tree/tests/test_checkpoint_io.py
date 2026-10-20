"""Resume must never discover a checkpoint while its bytes are incomplete."""
import pickle

import pytest

from hexcore import checkpoint_io
from hexcore.s2s_door import latest_checkpoint


def test_published_checkpoint_is_flushed_roundtrip(tmp_path):
    path = tmp_path / "checkpoint.pkl"
    payload = {"clock": 432000, "state": [1.0, 2.0]}
    checkpoint_io.write_checkpoint(path, payload)
    assert pickle.loads(path.read_bytes()) == payload
    assert not path.with_name(path.name + ".part").exists()
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        checkpoint_io.write_checkpoint(path, {"clock": 0})
    assert path.read_bytes() == before


def test_interrupted_dump_is_invisible_to_resume(tmp_path, monkeypatch):
    member = tmp_path / "attempt" / "mem001"
    member.mkdir(parents=True)
    path = member / "checkpoint.2026-10-22_00.00.00.pkl"

    def interrupted(payload, stream, protocol):
        stream.write(b"incomplete")
        stream.flush()
        assert not path.exists()
        raise RuntimeError("interrupted checkpoint writer")

    monkeypatch.setattr(checkpoint_io.pickle, "dump", interrupted)
    with pytest.raises(RuntimeError, match="interrupted checkpoint writer"):
        checkpoint_io.write_checkpoint(path, {"clock": 432000})
    assert path.with_name(path.name + ".part").read_bytes() == b"incomplete"
    assert latest_checkpoint({"run_id": "mem001"}, [{"out": str(member.parent)}]) is None
