"""Exact reference transfer, bound to real bytes and the producer."""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.drivers import contract_host_snapshot as bank


def bundle(tmp_path):
    paths = [tmp_path / name for name in ("grid.nc", "init.nc", "lbc.00.nc")]
    for index, path in enumerate(paths):
        path.write_bytes(bytes([index]) * 21)
    words = np.array([0, 0x80000000, 0x7fc00013, 1, 0xff800000], dtype="u4")
    return SimpleNamespace(grid_path=paths[0], init_path=paths[1], lbc_paths=[paths[2]],
        start_time=datetime(2026, 1, 1), _references={"values": words.view("f4"),
            ("driving", datetime(2026, 1, 1), b"\x00\x80"): np.array([3], dtype="i8")})


def test_preserves_every_float_word_and_typed_reference_key(tmp_path):
    source = bundle(tmp_path)
    selected = [SimpleNamespace(name="deck")]
    saved = bank.save(source, selected, tmp_path / "snapshot")
    target = bundle(tmp_path)
    target._references = {}
    bank.load(target, selected, tmp_path / "snapshot")
    assert target._references.keys() == source._references.keys()
    for key in source._references:
        assert source._references[key].tobytes() == target._references[key].tobytes()
    assert saved["references"] == 2


@pytest.mark.parametrize("which", ["grid_path", "init_path", "lbc_paths", "producer", "decks"])
def test_refuses_foreign_inputs_or_producer(tmp_path, monkeypatch, which):
    source = bundle(tmp_path)
    selected = [SimpleNamespace(name="deck")]
    bank.save(source, selected, tmp_path / "snapshot")
    if which == "producer":
        monkeypatch.setattr(bank, "source_digest", lambda: "changed")
    elif which == "decks":
        selected = [SimpleNamespace(name="different")]
    else:
        path = getattr(source, which)
        Path(path[0] if isinstance(path, list) else path).write_bytes(b"different")
    with pytest.raises(ValueError, match="another analysis"):
        bank.load(source, selected, tmp_path / "snapshot")


def test_corrupt_array_cannot_be_restored(tmp_path):
    source = bundle(tmp_path)
    selected = [SimpleNamespace(name="deck")]
    root = tmp_path / "snapshot"
    bank.save(source, selected, root)
    array = next(root.rglob("0.npy"))
    data = bytearray(array.read_bytes())
    data[-1] ^= 1
    array.write_bytes(data)
    with pytest.raises(ValueError, match="missing or corrupt"):
        bank.load(source, selected, root)


def test_separate_cpu_and_gpu_hosts_preserve_reference_words(tmp_path, monkeypatch):
    source = bundle(tmp_path)
    selected = [SimpleNamespace(name="deck")]
    monkeypatch.setattr(bank.prep_cache, "producer_digest", lambda: "cpu-host-environment")
    saved = bank.save(source, selected, tmp_path / "snapshot")
    assert saved["host_producer"] == "cpu-host-environment"
    monkeypatch.setattr(bank.prep_cache, "producer_digest", lambda: "gpu-host-environment")
    target = bundle(tmp_path)
    target._references = {}
    bank.load(target, selected, tmp_path / "snapshot")
    for key in source._references:
        assert source._references[key].tobytes() == target._references[key].tobytes()
