"""Cold cache persistence overlaps finalization without changing its bytes."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident

import numpy as np
import pytest

from hexcore import vertical_spec as v


def fields():
    return {name: (np.arange(96, dtype=np.float32).reshape(8, 12) + index)
            for index, name in enumerate(("edgeNormalVectors", "localVerticalUnitVectors",
                                          "cellTangentPlane", "coeffs_reconstruct"))}


def test_background_cache_overlaps_and_preserves_complete_bytes(tmp_path, monkeypatch):
    values = fields()
    expected, actual = tmp_path / "expected.npz", tmp_path / "actual.npz"
    np.savez_compressed(expected, cache_key=np.str_("grid-key"), **values)
    entered, release = Event(), Event()
    caller = get_ident()
    original = v._write_derived_geometry_cache

    def write(*args):
        assert get_ident() != caller
        entered.set()
        assert release.wait(5)
        original(*args)

    monkeypatch.setattr(v, "_write_derived_geometry_cache", write)
    with v._background_geometry_cache() as submit:
        submit(actual, "grid-key", values)
        assert entered.wait(5)
        # Caller finalization can proceed while compression is still blocked.
        assert not actual.exists()
        release.set()
    assert actual.read_bytes() == expected.read_bytes()
    assert not list(tmp_path.glob("*.tmp-*"))


def test_cache_hit_schedules_no_write(tmp_path, monkeypatch):
    from hexcore import rust_geometry

    values = fields()
    monkeypatch.setattr(rust_geometry, "derived_geometry", lambda mesh: values)
    cache = tmp_path / "cache.npz"
    with v._background_geometry_cache() as submit:
        computed, origin = v._derived_geometry_arrays(None, cache, "key", submit)
        assert computed is values and origin == "computed"
    with v._background_geometry_cache() as submit:
        loaded, origin = v._derived_geometry_arrays(None, cache, "key", submit)
        assert origin == "cache"
        # No pending write exists on a hit, so the one allowed slot is free.
        submit(tmp_path / "other.npz", "key", values)
    for name in values:
        assert loaded[name].tobytes() == values[name].tobytes()


def test_only_one_cache_write_in_flight(tmp_path):
    with v._background_geometry_cache() as submit:
        submit(tmp_path / "cache.npz", "key", fields())
        with pytest.raises(RuntimeError, match="only one geometry cache write"):
            submit(tmp_path / "other.npz", "key", fields())


def test_failed_cache_write_is_joined_and_partial_removed(tmp_path, monkeypatch):
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"previous-good-cache")

    def fail(path, **unused):
        path.with_name(path.name + ".npz").write_bytes(b"partial")
        raise OSError("compression failed")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="compression failed"):
        with v._background_geometry_cache() as submit:
            submit(cache, "key", fields())
    assert cache.read_bytes() == b"previous-good-cache"
    assert not list(tmp_path.glob("*.tmp-*"))


def test_caller_failure_still_joins_writer_and_preserves_error(tmp_path, monkeypatch):
    entered, release, caller_failed = Event(), Event(), Event()

    def write(*unused):
        entered.set()
        assert release.wait(5)
        raise OSError("secondary cache failure")

    def caller():
        with v._background_geometry_cache() as submit:
            submit(tmp_path / "cache.npz", "key", fields())
            assert entered.wait(5)
            caller_failed.set()
            raise ValueError("artifact failure")

    monkeypatch.setattr(v, "_write_derived_geometry_cache", write)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(caller)
        try:
            assert caller_failed.wait(5)
            assert not future.done()
        finally:
            release.set()
        with pytest.raises(ValueError, match="artifact failure"):
            future.result(timeout=5)
