"""Rank snapshot affinity checks and an explicit staged multi-card GPU guard.

Default tests use a current-device-sensitive CUDA double. The --gpu guard
explicitly disables cross-card peer access before exchanging snapshots.
--allow-shared-device exercises both staged DMA legs on one physical card;
its receipt distinguishes this from separate-card affinity qualification.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hexcore.drivers.samehost_forecast_v841 import _initial_rank_surface_updates, device_list

_ACCUMULATORS = ("rainnc", "snownc", "graupelnc")


class _Array:
    def __init__(self, values, device):
        self.values = np.asarray(values).copy()
        self.device = SimpleNamespace(id=device)
        self.ready = False


class _Stream:
    def __init__(self):
        self.pending = []
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1
        for array in self.pending:
            array.ready = True
        self.pending.clear()


class _PeerError(RuntimeError):
    def __init__(self, status):
        super().__init__(f"peer access status {status}")
        self.status = status


class _FakeCUDA:
    """A current-device-sensitive CUDA double.

    The current device is per thread, as CUDA's is: ranks on distinct cards
    take their initial snapshots on their own threads (2026-10-13), and a
    worker's Device scope never moves the coordinator's device.
    """

    ndarray = _Array

    def __init__(self, current=9):
        import threading

        self._local = threading.local()
        self._default = current
        self.streams = {}
        self.upload_devices = []
        self.disabled_pairs = set()
        self.disable_calls = []
        owner = self
        class Device:
            def __init__(self, device):
                self.id = int(device)
            def __enter__(self):
                self.previous = owner.current
                owner.current = self.id
                return self
            def __exit__(self, *_):
                owner.current = self.previous
                return False
        def disable_peer_access(destination):
            pair = owner.current, int(destination)
            owner.disable_calls.append(pair)
            if pair in owner.disabled_pairs:
                raise _PeerError(705)
            owner.disabled_pairs.add(pair)
        self.cuda = SimpleNamespace(
            Device=Device, get_current_stream=self.get_current_stream,
            runtime=SimpleNamespace(deviceDisablePeerAccess=disable_peer_access,
                                    CUDARuntimeError=_PeerError),
        )

    @property
    def current(self):
        return getattr(self._local, "current", self._default)

    @current.setter
    def current(self, device):
        self._local.current = device

    def get_current_stream(self):
        return self.streams.setdefault(self.current, _Stream())

    def asarray(self, values):
        array = _Array(values, self.current)
        self.upload_devices.append(self.current)
        self.get_current_stream().pending.append(array)
        return array


class _Backend:
    def __init__(self, cp, values, foreign_device=None):
        self.cp, self.values, self.foreign_device = cp, values, foreign_device
        self.snapshots = 0

    def diagnostic_snapshot(self):
        self.snapshots += 1
        scope = (nullcontext() if self.foreign_device is None
                 else self.cp.cuda.Device(self.foreign_device))
        with scope:
            return {"precipitation": {name: self.cp.asarray(array) for name, array in self.values.items()}}


class _Proof:
    @staticmethod
    def _previous_surface_updates(stack, accumulators):
        snapshot = stack["backend"].diagnostic_snapshot()["precipitation"]
        return {name: snapshot[name] for name in accumulators}


def _values(rank, count=4):
    words = np.resize(np.array([0, 0x80000000, 0x7fc01234, 0x3f800000], dtype=np.uint32), count)
    return {name: (words ^ np.uint32((rank << 8) | slot)).view(np.float32)
            for slot, name in enumerate(_ACCUMULATORS)}


def _disable_selected_peer_access(cp, devices):
    """Disable every selected cross-card direction in this guard process."""
    disabled = []
    for source in sorted(set(devices)):
        with cp.cuda.Device(source):
            for destination in sorted(set(devices)):
                if source == destination:
                    continue
                try:
                    cp.cuda.runtime.deviceDisablePeerAccess(destination)
                except cp.cuda.runtime.CUDARuntimeError as error:
                    if error.status != 705:  # cudaErrorPeerAccessNotEnabled
                        raise
                disabled.append([source, destination])
    return disabled


class SurfaceAffinityTests(unittest.TestCase):
    def test_every_rank_uploads_previous_arrays_on_its_own_device(self):
        cp, devices = _FakeCUDA(), (2, 4, 6)
        stacks = [{"backend": _Backend(cp, _values(rank))} for rank in range(3)]
        previous = _initial_rank_surface_updates(_Proof, stacks, devices, _ACCUMULATORS, cp)
        self.assertEqual(cp.current, 9)
        # Distinct cards snapshot concurrently, so only each card's count is fixed.
        self.assertEqual(sorted(cp.upload_devices), [2, 2, 2, 4, 4, 4, 6, 6, 6])
        for rank, values in enumerate(previous):
            self.assertEqual(stacks[rank]["backend"].snapshots, 1)
            self.assertEqual(cp.streams[devices[rank]].synchronize_calls, 1)
            for name, array in values.items():
                self.assertEqual(array.device.id, devices[rank])
                self.assertTrue(array.ready)
                self.assertEqual(array.values.tobytes(), stacks[rank]["backend"].values[name].tobytes())

    def test_same_card_ranks_keep_separate_snapshots_and_restore_coordinator(self):
        cp, devices = _FakeCUDA(), (0, 0, 3)
        stacks = [{"backend": _Backend(cp, _values(rank))} for rank in range(3)]
        previous = _initial_rank_surface_updates(_Proof, stacks, devices, _ACCUMULATORS, cp)
        self.assertEqual(cp.current, 9)
        self.assertIsNot(previous[0]["rainnc"], previous[1]["rainnc"])
        self.assertEqual([values["rainnc"].device.id for values in previous], list(devices))
        self.assertEqual(cp.streams[0].synchronize_calls, 2)
        self.assertEqual(cp.streams[3].synchronize_calls, 1)

    def test_missing_device_scope_is_rejected_before_surface_kernels(self):
        cp = _FakeCUDA()
        stacks = [{"backend": _Backend(cp, _values(0))}]
        with patch.object(cp.cuda, "Device", side_effect=lambda _: nullcontext()):
            with self.assertRaisesRegex(RuntimeError, "rank 0 previous surface accumulator 'rainnc' is on device 9; assigned device is 2"):
                _initial_rank_surface_updates(_Proof, stacks, (2,), _ACCUMULATORS, cp)
        self.assertEqual(cp.current, 9)

    def test_foreign_resident_leaf_is_rejected(self):
        cp = _FakeCUDA()
        stack = {"backend": _Backend(cp, _values(0), foreign_device=7)}
        with self.assertRaisesRegex(RuntimeError, "on device 7; assigned device is 2"):
            _initial_rank_surface_updates(_Proof, [stack], (2,), _ACCUMULATORS, cp)
        self.assertEqual(cp.current, 9)

    def test_none_and_host_values_keep_existing_semantics(self):
        cp = _FakeCUDA()
        host = {"rainnc": np.array([1], dtype=np.float32)}
        proof = SimpleNamespace(_previous_surface_updates=lambda stack, _: stack["value"])
        previous = _initial_rank_surface_updates(proof, [{"value": None}, {"value": host}],
                                                (2, 4), _ACCUMULATORS, cp)
        self.assertIsNone(previous[0])
        self.assertIs(previous[1], host)
        self.assertEqual(cp.current, 9)

    def test_card_assignment_count_is_checked_before_snapshot(self):
        cp = _FakeCUDA()
        with self.assertRaisesRegex(ValueError, "one device per rank"):
            _initial_rank_surface_updates(_Proof, [{"backend": _Backend(cp, _values(0))}],
                                          (), _ACCUMULATORS, cp)
        self.assertFalse(cp.upload_devices)

    def test_explicit_peer_disable_covers_every_direction_and_restores_context(self):
        cp = _FakeCUDA()
        expected = [[source, destination] for source in (2, 4, 6) for destination in (2, 4, 6)
                    if source != destination]
        for _ in range(2):
            disabled = _disable_selected_peer_access(cp, (4, 2, 6, 2))
            self.assertEqual(disabled, expected)
            self.assertEqual(cp.current, 9)
        self.assertEqual(cp.disabled_pairs, {tuple(pair) for pair in expected})
        self.assertEqual(cp.disable_calls, [tuple(pair) for pair in expected] * 2)

    def test_peer_disable_failure_is_not_reported_as_a_successful_guard(self):
        cp = _FakeCUDA()
        error = _PeerError(700)
        with patch.object(cp.cuda.runtime, "deviceDisablePeerAccess", side_effect=error):
            with self.assertRaises(_PeerError) as caught:
                _disable_selected_peer_access(cp, (2, 4))
        self.assertIs(caught.exception, error)
        self.assertEqual(cp.current, 9)
        self.assertFalse(cp.disabled_pairs)


def _tables(ranks):
    result = []
    for rank in range(ranks):
        peers = []
        for slot, peer in enumerate(other for other in range(ranks) if other != rank):
            peers.append(SimpleNamespace(peer=peer, cell_send=np.array([0, 1], dtype=np.int64),
                cell_recv=np.array([2 + 2 * slot, 3 + 2 * slot], dtype=np.int64),
                edge_send=np.empty(0, dtype=np.int64), edge_recv=np.empty(0, dtype=np.int64)))
        result.append(SimpleNamespace(rank=rank, peers=peers, n_owned_cells=2, n_local_cells=2 * ranks,
                                      n_owned_edges=0, n_local_edges=0))
    return result


def gpu_guard(args):
    import cupy as cp
    from hexcore.partition_samehost_transport_v841 import SameHostTransport
    from hexcore.partition_samehost_v841 import SameHostRanks
    from hexcore.drivers.samehost_physics_scope_v841 import scoped_physics_resources
    from gpuwm.core import refl
    devices = device_list(args.devices)
    shared = len(set(devices)) < 2
    if shared and not args.allow_shared_device:
        raise ValueError("the staged affinity guard requires at least two distinct physical devices")
    if max(devices) >= cp.cuda.runtime.getDeviceCount():
        raise ValueError("the staged affinity guard requires every selected device to be visible")
    disabled = _disable_selected_peer_access(cp, devices)
    tables = _tables(len(devices))
    stacks = [{"backend": _Backend(cp, _values(rank, 2 * len(devices)))} for rank in range(len(devices))]
    previous = _initial_rank_surface_updates(_Proof, stacks, devices, _ACCUMULATORS, cp)
    host = []
    for device, values in zip(devices, previous):
        with cp.cuda.Device(device):
            host.append({name: cp.asnumpy(array) for name, array in values.items()})
    expected = [{name: array.copy() for name, array in values.items()} for values in host]
    for rank, table in enumerate(tables):
        for peer in table.peers:
            for name in _ACCUMULATORS:
                expected[rank][name][peer.cell_recv] = host[peer.peer][name][:2]
    transport = SameHostTransport(tables, devices, xp=cp, copy_policy="staged", timeout=30,
                                  stage_same_device=shared)
    with scoped_physics_resources() as scope:
        with SameHostRanks(devices, transport, cp=cp, physics_scope=scope) as ranks:
            # This is the real no-argument LRU factory that failed at history.
            # Keep both outputs alive to check separate storage on one card too.
            reflection_tables = ranks.call(lambda rank: refl._device_tables())
            if len({int(array.data.ptr) for array in reflection_tables}) != len(devices):
                raise AssertionError("reflectivity tables share storage across rank scopes")
            reflection_bytes = []
            for rank, (device, array) in enumerate(zip(devices, reflection_tables)):
                with cp.cuda.Device(device):
                    if int(array.device.id) != device:
                        raise AssertionError(f"rank {rank} reflectivity table is on another device")
                    reflection_bytes.append(cp.asnumpy(array).tobytes())
            if len(set(reflection_bytes)) != 1:
                raise AssertionError("reflectivity table bytes differ between rank scopes")
            ranks.call(lambda rank: transport.exchange(rank, "surface-affinity-guard",
                       tuple((name, previous[rank][name], "cell") for name in _ACCUMULATORS)))
            for rank, device in enumerate(devices):
                with cp.cuda.Device(device):
                    for name, array in previous[rank].items():
                        if int(array.device.id) != device or cp.asnumpy(array).tobytes() != expected[rank][name].tobytes():
                            raise AssertionError(f"rank {rank} accumulator {name!r} failed staged affinity or byte identity")
    exchange = transport.receipt()
    if exchange["stats"]["peer_bytes"] or exchange["stats"]["staged_bytes"] <= 0:
        raise AssertionError("staged affinity guard did not exercise staged cross-card traffic")
    if any(route["actual"] != "staged" for route in exchange["routes"]
           if route["source_device"] != route["destination_device"]):
        raise AssertionError("staged affinity guard used a cross-card peer route")
    receipt = {"schema": "samehost-surface-affinity-guard/v1", "devices": list(devices),
        "physical_device_count": len(set(devices)), "separate_card_affinity_qualified": not shared,
        "peer_access_explicitly_disabled": disabled, "copy_policy": "staged", "byte_identical": True,
        "previous_surface_device_ids": [{name: int(array.device.id) for name, array in values.items()}
                                        for values in previous], "exchange": exchange,
        "reflectivity_table_device_ids": [int(array.device.id) for array in reflection_tables],
        "reflectivity_tables_distinct_storage": True,
        "reflectivity_table_bytes_identical": True,
        "reflectivity_table_sha256": hashlib.sha256(reflection_bytes[0]).hexdigest(),
        "physics_scope": scope.receipt(),
        "source_sha256": hashlib.sha256(Path(sys.modules[_initial_rank_surface_updates.__module__].__file__).read_bytes()).hexdigest(),
        "runtime": {"python": sys.version, "cupy": cp.__version__, "cuda": cp.cuda.runtime.runtimeGetVersion()}}
    if args.output:
        Path(args.output).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--allow-shared-device", action="store_true")
    parser.add_argument("--output")
    args, remaining = parser.parse_known_args()
    if args.gpu:
        if remaining:
            parser.error("GPU guard does not accept extra arguments")
        gpu_guard(args)
    else:
        unittest.main(argv=[sys.argv[0], *remaining])
