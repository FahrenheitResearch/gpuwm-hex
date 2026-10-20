# SPDX-License-Identifier: Apache-2.0
"""Rollback copies defer device-to-host transfer without changing payload bits.

The CPU tests exercise the copy destination and exporter contract. The CUDA
tests exercise real device copies and stream ordering; full forecast identity
is checked through the forecast door separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from hexcore.cuda_arwen_physics_v841 import PersistentTwoPhaseCudaPhysicsBackendV841
from hexcore.cuda_seam_snapshot_v841 import DeviceBoundarySnapshotV841


@pytest.fixture(autouse=True)
def _original_copy_path(monkeypatch):
    # These tests pin the per-step CuPy copy path with host fakes.  The buffer
    # reuse path (GPUWM_HEX_SNAPSHOT_REUSE=1, the default) moves bytes with a
    # device-to-device memcpyAsync on raw pointers, which a host fake cannot
    # stand in for; its evidence is the byte-identical forecast pair recorded
    # with the round that added it.
    monkeypatch.setenv("GPUWM_HEX_SNAPSHOT_REUSE", "0")


class _DeviceArray:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values


class _FakeCupy:
    ndarray = _DeviceArray

    def __init__(self) -> None:
        self.host_copies = []
        self.device_copies = []
        self.fail_device_copy = False

    def array(self, value, *, copy, order):
        if self.fail_device_copy:
            raise RuntimeError("device copy failed")
        result = _DeviceArray(np.array(value.values, copy=copy, order=order))
        self.device_copies.append(result)
        return result

    def asnumpy(self, value):
        self.host_copies.append(value)
        source = value.values if isinstance(value, _DeviceArray) else value
        return np.array(source, copy=True, order="C")


class _BoundarySeam:
    """The public exporter owns the manifest, boundary guard and metadata."""

    def __init__(self, cp, arrays) -> None:
        self._cp = cp
        self.arrays = arrays
        self.at_boundary = True
        self.identity = {"schema": 2, "precision": "float32"}
        self.call_counts = {"radiation": 3, "surface": 5}
        self.elapsed_seconds = np.float64(-0.0)

    def export_state(self):
        if not self.at_boundary:
            raise RuntimeError("export_state requires a step boundary")
        return {
            "identity": dict(self.identity),
            "arrays": {
                name: self._cp.asnumpy(value)
                for name, value in self.arrays.items()
            },
            "scalars": {
                "step_index": 5,
                "elapsed_seconds": self.elapsed_seconds,
                "call_counts": dict(self.call_counts),
                "history_time": None,
                "enabled": True,
            },
        }


def _float_bits(dtype):
    if dtype == np.float32:
        bits = np.array(
            [0x00000000, 0x80000000, 0x7FC00031, 0xFFC12345,
             0x00000001, 0x7F800000, 0xFF800000, 0x3F800000],
            dtype=np.uint32,
        )
    else:
        bits = np.array(
            [0x0000000000000000, 0x8000000000000000,
             0x7FF8000000000031, 0xFFF8123456789ABC,
             0x0000000000000001, 0x7FF0000000000000,
             0xFFF0000000000000, 0x3FF0000000000000],
            dtype=np.uint64,
        )
    return bits.view(dtype).reshape(2, 4)


def _assert_bits_equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes(order="C") == expected.tobytes(order="C")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_device_snapshot_preserves_strided_float_bits_without_host_copy(dtype):
    cp = _FakeCupy()
    source = _DeviceArray(_float_bits(dtype)[:, ::-1])
    expected = np.array(source.values, copy=True)
    seam = _BoundarySeam(cp, {"state/field": source})
    helper = DeviceBoundarySnapshotV841(seam)

    payload = helper.export_state()
    saved = payload["arrays"]["state/field"]

    assert isinstance(saved, _DeviceArray)
    assert saved is not source
    assert saved.values.flags.c_contiguous
    assert not np.shares_memory(saved.values, source.values)
    assert cp.host_copies == []
    _assert_bits_equal(saved.values, expected)

    source.values[...] = 7
    _assert_bits_equal(saved.values, expected)
    _assert_bits_equal(helper.to_host(payload)["arrays"]["state/field"], expected)
    assert len(cp.host_copies) == 1
    assert cp.host_copies[0] is saved


def test_snapshot_detaches_host_arrays_and_preserves_scalar_shapes():
    cp = _FakeCupy()
    host = np.arange(24, dtype=np.int16).reshape(4, 6)[::2, ::-2]
    scalar = np.array(-0.0, dtype=np.float32)
    expected_host = host.copy()
    expected_scalar = scalar.copy()
    seam = _BoundarySeam(cp, {"radiation/host": host, "state/scalar": scalar})

    payload = DeviceBoundarySnapshotV841(seam).export_state()

    assert cp.host_copies == []
    assert cp.device_copies == []
    for name, source in seam.arrays.items():
        saved = payload["arrays"][name]
        assert isinstance(saved, np.ndarray)
        assert saved.flags.c_contiguous
        assert not np.shares_memory(saved, source)
    host[...] = -1
    scalar[...] = 1
    _assert_bits_equal(payload["arrays"]["radiation/host"], expected_host)
    _assert_bits_equal(payload["arrays"]["state/scalar"], expected_scalar)


def test_snapshot_reuses_exporter_schema_and_preserves_scalar_values():
    cp = _FakeCupy()
    seam = _BoundarySeam(cp, {"state/field": _DeviceArray(np.zeros(3))})
    original = seam.export_state()
    payload = DeviceBoundarySnapshotV841(seam).export_state()

    assert tuple(payload) == tuple(original)
    assert tuple(payload["arrays"]) == tuple(original["arrays"])
    assert payload["identity"] == original["identity"]
    assert payload["scalars"] == original["scalars"]
    elapsed = payload["scalars"]["elapsed_seconds"]
    assert type(elapsed) is type(original["scalars"]["elapsed_seconds"])
    assert elapsed.tobytes() == original["scalars"]["elapsed_seconds"].tobytes()
    assert payload["scalars"]["history_time"] is None
    assert payload["scalars"]["enabled"] is True
    assert seam._cp is cp
    seam.call_counts["surface"] += 1
    seam.identity["schema"] = 3
    assert payload["scalars"]["call_counts"]["surface"] == 5
    assert payload["identity"]["schema"] == 2


def test_each_boundary_snapshot_owns_its_device_copies():
    cp = _FakeCupy()
    source = _DeviceArray(np.array([1, 2, 3], dtype=np.int64))
    seam = _BoundarySeam(cp, {"state/field": source})
    helper = DeviceBoundarySnapshotV841(seam)

    first = helper.export_state()
    source.values[...] = [4, 5, 6]
    second = helper.export_state()
    source.values[...] = 0

    assert first["arrays"]["state/field"] is not second["arrays"]["state/field"]
    np.testing.assert_array_equal(first["arrays"]["state/field"].values, [1, 2, 3])
    np.testing.assert_array_equal(second["arrays"]["state/field"].values, [4, 5, 6])
    assert cp.host_copies == []


def test_boundary_guard_failure_leaves_live_seam_unchanged():
    cp = _FakeCupy()
    seam = _BoundarySeam(cp, {"state/field": _DeviceArray(np.arange(3))})
    seam.at_boundary = False
    original = dict(seam.__dict__)

    with pytest.raises(RuntimeError, match="step boundary"):
        DeviceBoundarySnapshotV841(seam).export_state()

    assert seam.__dict__ == original
    assert seam._cp is cp
    assert cp.device_copies == []
    assert cp.host_copies == []


def test_copy_failure_leaves_live_seam_export_usable():
    cp = _FakeCupy()
    cp.fail_device_copy = True
    source = _DeviceArray(np.arange(3, dtype=np.int32))
    seam = _BoundarySeam(cp, {"state/field": source})
    original = dict(seam.__dict__)

    with pytest.raises(RuntimeError, match="device copy failed"):
        DeviceBoundarySnapshotV841(seam).export_state()

    assert seam.__dict__ == original
    assert seam._cp is cp
    _assert_bits_equal(seam.export_state()["arrays"]["state/field"], source.values)


def test_memory_fallback_releases_partial_copies_and_uses_original_host_exporter():
    import weakref

    class MemoryLimitedCupy(_FakeCupy):
        def __init__(self):
            super().__init__()
            self.copy_attempts = 0
            self.partial_copies = []

        def array(self, value, *, copy, order):
            self.copy_attempts += 1
            if self.copy_attempts == 2:
                raise MemoryError("snapshot allocation exhausted")
            result = _DeviceArray(np.array(value.values, copy=copy, order=order))
            self.partial_copies.append(weakref.ref(result))
            return result

        def asnumpy(self, value):
            assert all(saved() is None for saved in self.partial_copies)
            return super().asnumpy(value)

    cp = MemoryLimitedCupy()
    first = _DeviceArray(_float_bits(np.float32)[:, ::-1])
    second = _DeviceArray(_float_bits(np.float64))
    host = np.arange(8, dtype=np.int16)[::2]
    seam = _BoundarySeam(
        cp, {"state/first": first, "state/second": second, "radiation/host": host}
    )
    original = dict(seam.__dict__)

    payload = DeviceBoundarySnapshotV841(seam).export_state()

    assert cp.copy_attempts == 2
    assert len(cp.partial_copies) == 1
    assert len(cp.host_copies) == len(seam.arrays)
    assert seam.__dict__ == original
    assert seam._cp is cp
    assert payload["identity"] == seam.identity
    assert payload["scalars"]["call_counts"] == seam.call_counts
    for name, source in seam.arrays.items():
        saved = payload["arrays"][name]
        values = source.values if isinstance(source, _DeviceArray) else source
        assert isinstance(saved, np.ndarray)
        assert not np.shares_memory(saved, values)
        _assert_bits_equal(saved, values)


def _interrupted_adapter(seam, snapshot):
    backend = object.__new__(PersistentTwoPhaseCudaPhysicsBackendV841)
    backend._seam = seam
    backend._phase = "phase2"
    backend._boundary_snapshot = snapshot
    backend._step_start = 17.0
    backend._candidate_scalar_target = object()
    backend._candidate_scalar_backup = object()
    backend._pending_gwdo_result = object()
    backend._pending_refl10cm = object()
    return backend


def test_adapter_restore_materializes_and_releases_snapshot_before_new_seam():
    cp = _FakeCupy()
    source = _DeviceArray(_float_bits(np.float32)[:, ::-1])
    host = np.arange(6, dtype=np.int32)[::2]
    seam = _BoundarySeam(cp, {"state/field": source, "radiation/host": host})
    expected = seam.export_state()
    snapshot = DeviceBoundarySnapshotV841(seam).export_state()
    backend = _interrupted_adapter(seam, snapshot)
    events = []
    restored = []
    fresh = _BoundarySeam(cp, {})

    def restore(payload):
        events.append("restore")
        restored.append(payload)

    fresh.restore_state = restore

    def factory():
        assert snapshot["arrays"] == {}
        assert backend._seam is seam
        events.append("factory")
        return fresh

    def guard():
        assert backend._seam is fresh
        events.append("guard")

    backend._new_seam = factory
    backend._private_binding_guard = guard
    source.values[...] = 7
    host[...] = -1

    backend._restore_boundary(snapshot)

    assert events == ["factory", "restore", "guard"]
    assert len(restored) == 1
    payload = restored[0]
    assert payload["identity"] == expected["identity"]
    assert payload["scalars"] == expected["scalars"]
    for name, saved in payload["arrays"].items():
        assert isinstance(saved, np.ndarray)
        _assert_bits_equal(saved, expected["arrays"][name])
    assert backend._seam is fresh
    assert backend._phase == "boundary"
    assert backend._boundary_snapshot is None
    assert backend._step_start is None
    assert backend._candidate_scalar_target is None
    assert backend._candidate_scalar_backup is None
    assert backend._pending_gwdo_result is None
    assert backend._pending_refl10cm is None
    assert seam._cp is cp


def test_adapter_restore_constructor_failure_marks_transaction_broken():
    cp = _FakeCupy()
    seam = _BoundarySeam(cp, {"state/field": _DeviceArray(np.arange(3))})
    snapshot = DeviceBoundarySnapshotV841(seam).export_state()
    backend = _interrupted_adapter(seam, snapshot)
    failure = MemoryError("replacement seam allocation exhausted")

    def factory():
        assert snapshot["arrays"] == {}
        raise failure

    def guard():
        pytest.fail("the binding guard must follow successful construction")

    backend._new_seam = factory
    backend._private_binding_guard = guard

    with pytest.raises(RuntimeError, match="failed to reconstruct") as error:
        backend._restore_boundary(snapshot)

    assert error.value.__cause__ is failure
    assert backend._phase == "broken"
    assert backend._seam is seam
    assert seam._cp is cp
    assert snapshot["arrays"] == {}


def test_to_host_reads_payload_without_replacing_its_arrays_or_metadata():
    cp = _FakeCupy()
    seam = _BoundarySeam(
        cp,
        {"state/field": _DeviceArray(_float_bits(np.float32)),
         "radiation/host": np.arange(3, dtype=np.float64)},
    )
    helper = DeviceBoundarySnapshotV841(seam)
    payload = helper.export_state()
    original_arrays = dict(payload["arrays"])
    original_bits = {
        name: (value.values if isinstance(value, _DeviceArray) else value).tobytes()
        for name, value in original_arrays.items()
    }

    first = helper.to_host(payload)
    second = helper.to_host(payload)

    assert first is not payload
    assert first["arrays"] is not payload["arrays"]
    assert first["identity"] is payload["identity"]
    assert first["scalars"] is payload["scalars"]
    assert isinstance(first["arrays"]["state/field"], np.ndarray)
    for name, saved in original_arrays.items():
        assert payload["arrays"][name] is saved
        value = saved.values if isinstance(saved, _DeviceArray) else saved
        assert value.tobytes() == original_bits[name]
        _assert_bits_equal(first["arrays"][name], value)
        _assert_bits_equal(second["arrays"][name], value)
    assert len(cp.host_copies) == 2
    assert all(value is original_arrays["state/field"] for value in cp.host_copies)
    assert seam._cp is cp


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cuda_snapshot_preserves_bits_and_precedes_same_stream_mutation(dtype):
    import cupy as cp

    expected = _float_bits(dtype)[:, ::-1].copy()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        base = cp.asarray(_float_bits(dtype))
        source = base[:, ::-1]
        host = np.arange(6, dtype=np.int32)[::2]
        seam = _BoundarySeam(cp, {"state/field": source, "radiation/host": host})
        helper = DeviceBoundarySnapshotV841(seam)
        payload = helper.export_state()
        saved = payload["arrays"]["state/field"]
        assert isinstance(saved, cp.ndarray)
        assert saved.data.ptr != source.data.ptr
        assert saved.flags.c_contiguous
        source[...] = 7
        converted = helper.to_host(payload)
    stream.synchronize()

    _assert_bits_equal(converted["arrays"]["state/field"], expected)
    _assert_bits_equal(cp.asnumpy(saved), expected)
    assert np.all(cp.asnumpy(source) == 7)
    np.testing.assert_array_equal(converted["arrays"]["radiation/host"], [0, 2, 4])
    assert isinstance(payload["arrays"]["state/field"], cp.ndarray)
    assert seam._cp is cp
