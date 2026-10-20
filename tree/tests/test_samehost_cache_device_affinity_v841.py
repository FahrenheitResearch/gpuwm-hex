"""Device-array caches and imported factory aliases stay within each rank."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
import threading

import pytest

from hexcore.drivers.samehost_physics_scope_v841 import scoped_physics_resources
from test_samehost_physics_scope_v841 import _modules


_BITS = bytes.fromhex("00000000000000803412c07f01000000")


def _array_modules():
    modules, identity = _modules()

    class DeviceArray:
        def __init__(self):
            marker = identity.get()
            self.owner = marker
            self.device = SimpleNamespace(id=marker[1] if isinstance(marker, tuple) else 99)
            self.payload = _BITS
            self.mutable = {}

    def create_array():
        return DeviceArray()

    for group in ("reflectivity", "myjsfc", "noah", "milbrandt", "rrtm_taumol", "rrtm_lw",
                  "rrtmgp", "mynn_flags", "kernels"):
        modules[group].create_handle = create_array
    exec("""
@lru_cache(maxsize=None)
def _kernel(defines, shared_bytes):
    kernel = get_kernel_int_defines('jacobi', 'eigh', defines)
    kernel[0].mutable['max_dynamic_shared_size_bytes'] = shared_bytes
    return kernel
""", vars(modules["jacobi"]))
    return modules, identity


def _factories(modules, table, xp):
    return {
        "reflectivity": lambda: modules["reflectivity"]._device_tables(),
        "myjsfc": lambda: modules["myjsfc"]._device_tables(),
        "noah": lambda: modules["noah"]._device_tables(table, (1, 2)),
        "milbrandt": lambda: modules["milbrandt"].ck_vector_device(),
        "rrtm_taumol": lambda: modules["rrtm_taumol"]._device_tables(table, xp),
        "rrtm_lw": lambda: modules["rrtm_lw"]._device_constant(xp, "f4", (id(table),), lambda: _BITS),
        "rrtmgp_jump": lambda: modules["rrtmgp"]._mcica_jump_tables(55, 112),
        "rrtmgp_scratch": lambda: modules["rrtmgp"]._chunk_scratch("upper", (4, 55), xp=xp)[0],
        "mynn_flags": lambda: modules["mynn_flags"]._validity_flags(),
        "jacobi": lambda: modules["jacobi"]._kernel((("K", 64),), 49152)[0],
    }


@pytest.mark.parametrize("devices", [(0, 0), (0, 1)])
def test_all_new_cached_arrays_have_current_device_and_private_rank_owners(devices):
    modules, identity = _array_modules()
    table, xp = object(), ModuleType("cupy")
    factories = _factories(modules, table, xp)
    # An already populated process cache must not become either rank's cache.
    ambient = {name: factory() for name, factory in factories.items()}
    barrier = threading.Barrier(2)
    with scoped_physics_resources(modules=modules) as scope:
        def worker(rank):
            marker = (rank, devices[rank])
            token = identity.set(marker)
            try:
                with scope.rank(*marker):
                    selected = {name: factory() for name, factory in factories.items()}
                    for name, array in selected.items():
                        assert array.device.id == devices[rank]
                        assert array.owner == marker
                        assert array.payload == _BITS
                        assert array is factories[name]()
                        assert array is not ambient[name]
                    selected["rrtmgp_scratch"].mutable["rank"] = rank
                    selected["mynn_flags"].mutable["rank"] = rank
                    assert selected["jacobi"].mutable["max_dynamic_shared_size_bytes"] == 49152
                    barrier.wait(timeout=5)
                    assert selected["rrtmgp_scratch"].mutable["rank"] == rank
                    assert selected["mynn_flags"].mutable["rank"] == rank
                    return selected
            finally:
                identity.reset(token)

        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(worker, (0, 1)))
        assert all(outputs[0][name] is not outputs[1][name] for name in factories)
        assert scope.receipt()["active_rank_contexts"] == 0
    assert all(factory() is ambient[name] for name, factory in factories.items())
    assert scope.receipt()["module_entrypoints_restored"]


def test_snapshot_imported_alias_cannot_return_an_ambient_kernel_handle():
    modules, identity = _array_modules()
    original_alias = modules["jacobi"].get_kernel_int_defines
    ambient = modules["jacobi"]._kernel((("K", 64),), 49152)
    with scoped_physics_resources(modules=modules) as scope:
        token = identity.set((3, 1))
        try:
            with scope.rank(3, 1):
                selected = modules["jacobi"]._kernel((("K", 64),), 49152)
                assert selected is not ambient
                assert selected[0].device.id == 1 and selected[0].owner == (3, 1)
                state = scope._state("jacobi", (3, 1))
                assert state["globals"]["get_kernel_int_defines"] is modules["kernels"].get_kernel_int_defines
        finally:
            identity.reset(token)
    assert modules["jacobi"].get_kernel_int_defines is original_alias


def test_populated_scope_dispatch_avoids_global_state_lock_on_cache_hits():
    modules, identity = _array_modules()

    class CountLock:
        def __init__(self, lock):
            self.lock, self.entries = lock, 0

        def __enter__(self):
            self.entries += 1
            return self.lock.__enter__()

        def __exit__(self, *args):
            return self.lock.__exit__(*args)

    with scoped_physics_resources(modules=modules) as scope:
        token = identity.set((0, 0))
        try:
            with scope.rank(0, 0):
                original_lock = scope.lock
                counted = CountLock(original_lock)
                scope.lock = counted
                try:
                    first = modules["reflectivity"]._device_tables()
                    assert counted.entries == 1
                    for _ in range(100):
                        assert modules["reflectivity"]._device_tables() is first
                    assert counted.entries == 1
                finally:
                    scope.lock = original_lock
        finally:
            identity.reset(token)


def test_factory_body_defaults_and_closure_survive_rank_scoping():
    modules, identity = _array_modules()
    original = modules["reflectivity"]._device_tables.__wrapped__
    with scoped_physics_resources(modules=modules) as scope:
        state = scope._state("reflectivity", (0, 0))
        cloned = state["functions"]["_device_tables"].__wrapped__
        assert cloned.__code__ is original.__code__
        assert cloned.__defaults__ is original.__defaults__
        assert cloned.__kwdefaults__ is original.__kwdefaults__
        assert cloned.__closure__ is original.__closure__
