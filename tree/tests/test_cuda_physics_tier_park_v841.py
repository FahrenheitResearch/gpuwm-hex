# SPDX-License-Identifier: Apache-2.0
"""The park's rules, exercised on a fake array type.

These do not need a GPU: the module's contract is about WHICH references it may
move and what must come back, and that is decided by the object graph, not by
the device.  The behaviour a fake cannot show -- what the device pool actually
releases, and whether the run's payload digest survives -- is measured by
``tools/measure_physics_tier_park.py`` against the real forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
import types

import numpy as np
import pytest

from hexcore.cuda_physics_tier_park_v841 import (
    CudaPhysicsTierParkV841,
    PhysicsTierParkRefusal,
    walk_device_allocations,
)


class FakeMemory:
    """One device allocation: a byte buffer with a distinct fake address."""

    _next_address = [0x100000]

    def __init__(self, nbytes: int) -> None:
        self.buffer = bytearray(nbytes)
        self.address = FakeMemory._next_address[0]
        FakeMemory._next_address[0] += max(nbytes, 512) + 512


class FakePointer:
    def __init__(self, memory: FakeMemory, offset: int) -> None:
        self.memory = memory
        self.offset = offset

    @property
    def ptr(self) -> int:
        return self.memory.address + self.offset

    def __add__(self, offset: int) -> "FakePointer":
        return FakePointer(self.memory, self.offset + int(offset))


class FakeArray:
    """Enough of ``cupy.ndarray`` for the walker and the park to work on.

    The constructor mirrors the two shapes the park uses: ``FakeArray(shape,
    dtype=...)`` allocates, and ``FakeArray(shape, dtype=..., memptr=...,
    strides=...)`` borrows an existing allocation.
    """

    def __init__(self, shape, dtype=np.float32, memptr=None, strides=None, base=None):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        if memptr is None:
            self.data = FakePointer(FakeMemory(self.nbytes), 0)
        else:
            self.data = memptr
        self.base = base
        self.strides = (
            tuple(int(x) for x in strides)
            if strides is not None
            else tuple(int(x) for x in np.empty(self.shape, self.dtype).strides)
        )

    # -- the two transfer calls the park makes --------------------------
    def _window(self) -> memoryview:
        start = self.data.offset
        return memoryview(self.data.memory.buffer)[start : start + self.nbytes]

    def get(self, out=None):
        source = np.frombuffer(bytes(self._window()), dtype=np.uint8)
        if out is None:
            return source.copy()
        out[...] = source
        return out

    def set(self, values) -> None:
        self._window()[:] = np.ascontiguousarray(values).tobytes()


class FakePool:
    def used_bytes(self) -> int:
        return 0

    def total_bytes(self) -> int:
        return 0


def _build_cp() -> types.SimpleNamespace:
    runtime = types.SimpleNamespace(deviceSynchronize=lambda: None)
    cuda = types.SimpleNamespace(
        runtime=runtime,
        alloc_pinned_memory=lambda nbytes: bytearray(nbytes),
    )
    return types.SimpleNamespace(
        ndarray=FakeArray,
        uint8=np.uint8,
        empty=lambda shape, dtype=np.float32: FakeArray(shape, dtype=dtype),
        cuda=cuda,
        get_default_memory_pool=FakePool,
    )


@pytest.fixture(name="cp")
def fixture_cp():
    return _build_cp()


class Holder:
    """A plain mutable object: the common case in the seam graph."""


@dataclass(frozen=True)
class FrozenHolder:
    field: object


def test_walker_records_every_reference_to_one_array(cp):
    array = FakeArray((4, 3))
    root = Holder()
    root.first = array
    root.second = array
    root.bag = {"third": array}

    allocations, holders = walk_device_allocations(cp, root)

    assert len(allocations) == 1
    assert len(holders) == 3
    assert len({holder["token"] for holder in holders}) == 1


def test_park_refuses_a_second_park(cp):
    root = Holder()
    root.a = FakeArray((2, 2))
    park = CudaPhysicsTierParkV841(cp, root)
    park.park()

    with pytest.raises(PhysicsTierParkRefusal) as error:
        park.park()
    assert "already parked" in str(error.value)


def test_unpark_before_park_refuses(cp):
    park = CudaPhysicsTierParkV841(cp, Holder())
    with pytest.raises(PhysicsTierParkRefusal) as error:
        park.unpark()
    assert "not parked" in str(error.value)


def test_shared_allocation_is_never_moved(cp):
    shared = FakeArray((8,))
    private = FakeArray((8,))
    seam = Holder()
    seam.shared = shared
    seam.private = private
    dycore = Holder()
    dycore.same = shared

    park = CudaPhysicsTierParkV841(cp, seam, shared_roots=(dycore,))
    park.park()

    receipt = park.receipt()
    assert receipt["shared_skipped_count"] == 1
    assert receipt["shared_skipped_bytes"] == shared.nbytes
    assert receipt["parked_bytes"] == private.nbytes
    assert seam.shared is shared
    assert seam.private is None


def test_shared_root_given_at_the_call_site_is_honoured(cp):
    shared = FakeArray((8,))
    seam = Holder()
    seam.shared = shared
    in_flight = Holder()
    in_flight.same = shared

    park = CudaPhysicsTierParkV841(cp, seam)
    park.park(shared_roots=(in_flight,))

    assert park.receipt()["parked_bytes"] == 0
    assert seam.shared is shared


def test_frozen_dataclass_reference_pins_its_allocation(cp):
    array = FakeArray((6,))
    seam = Holder()
    seam.mutable = array
    seam.frozen = FrozenHolder(field=array)

    park = CudaPhysicsTierParkV841(cp, seam)
    park.park()

    receipt = park.receipt()
    assert receipt["parked_bytes"] == 0
    assert receipt["unrebindable_bytes"] == array.nbytes
    assert seam.mutable is array


def test_tuple_held_allocation_is_pinned_down(cp):
    array = FakeArray((6,))
    seam = Holder()
    seam.mutable = array
    seam.frozen_pair = (array, 1)

    park = CudaPhysicsTierParkV841(cp, seam)
    park.park()

    assert park.receipt()["parked_bytes"] == 0
    assert seam.mutable is array


def test_round_trip_restores_bytes_and_object_identity(cp):
    array = FakeArray((5, 7))
    payload = (np.arange(array.nbytes, dtype=np.uint16) % 251).astype(np.uint8)
    array.set(payload)
    seam = Holder()
    seam.one = array
    seam.two = array
    seam.bag = {"three": array}

    moved = CudaPhysicsTierParkV841(cp, seam)
    assert moved.park() == array.nbytes
    assert seam.one is None and seam.two is None and seam.bag["three"] is None

    moved.unpark()

    assert seam.one is not None
    assert seam.one is seam.two
    assert seam.bag["three"] is seam.one
    assert seam.one.shape == (5, 7)
    assert seam.one.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(seam.one.get(), payload)


def test_view_comes_back_as_a_view_of_the_same_allocation(cp):
    base = FakeArray((10,))
    base.set((np.arange(base.nbytes, dtype=np.uint16) % 255).astype(np.uint8))
    view = FakeArray((4,), memptr=base.data + 8, base=base)
    seam = Holder()
    seam.base = base
    seam.view = view

    park = CudaPhysicsTierParkV841(cp, seam)
    assert park.park() == base.nbytes
    park.unpark()

    assert seam.view.data.ptr - seam.base.data.ptr == 8
    assert seam.view.data.memory is seam.base.data.memory
    np.testing.assert_array_equal(
        seam.view.get(), base.get()[8 : 8 + seam.view.nbytes]
    )


def test_receipt_counts_parks_and_unparks(cp):
    seam = Holder()
    seam.a = FakeArray((3,))
    park = CudaPhysicsTierParkV841(cp, seam)

    for _ in range(3):
        park.park()
        park.unpark()

    receipt = park.receipt()
    assert receipt["parks"] == 3
    assert receipt["unparks"] == 3
    assert park.is_parked is False


def test_park_leaves_no_live_reference_to_the_parked_arrays(cp):
    """The defect that made the first version release exactly zero bytes.

    A recursive walk built with closures leaves a reference cycle, and that
    cycle held every array the walk touched until the cyclic collector
    happened to run.  Dropping the holders has to free the array NOW, with
    the cyclic collector disabled, or the park is a no-op wearing a receipt.
    """

    import gc

    array = FakeArray((4, 4))
    seam = Holder()
    seam.a = array
    park = CudaPhysicsTierParkV841(cp, seam)
    park.park()

    gc.disable()
    try:
        referrers = [
            r
            for r in gc.get_referrers(array)
            if r is not seam.__dict__ and not isinstance(r, list)
        ]
        # The only thing that may still point at it is this test's own local.
        assert referrers == [], referrers
    finally:
        gc.enable()


def test_walk_is_iterative_so_it_builds_no_cycle(cp):
    import gc

    array = FakeArray((4, 4))
    root = Holder()
    root.a = array
    gc.collect()
    before = len(gc.garbage)
    allocations, holders = walk_device_allocations(cp, root)
    for holder in holders:
        holder["array"] = None
    del allocations, holders
    gc.collect()
    assert len(gc.garbage) == before
