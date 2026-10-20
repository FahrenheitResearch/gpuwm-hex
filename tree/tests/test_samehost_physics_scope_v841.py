"""Concurrent rank resources cannot share scratch or bound CUDA handles."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import functools
from types import ModuleType
import sys
import threading

import pytest

from hexcore.drivers.samehost_physics_scope_v841 import _GROUPS, scoped_physics_resources


def _module(name, source, **values):
    module = ModuleType(name)
    vars(module).update(values)
    exec(source, vars(module))
    return module


def _modules():
    identity = ContextVar("test_physics_scope_identity", default="ambient")

    class Engine:
        def __init__(self, table):
            self.table = table
            self.owner = identity.get()
            self.scratch = {}

        def release_scratch(self):
            self.scratch.clear()

    def create_handle():
        return {"owner": identity.get(), "unique": object()}

    modules = {}
    modules["cumulus_clock"] = _module("gpuwm.test_cumulus_clock", """
_ADVANCE = None
_CLEAR = None
def _kernels():
    global _ADVANCE, _CLEAR
    if _ADVANCE is None:
        _ADVANCE = create_handle()
        if rendezvous is not None:
            rendezvous.wait(timeout=5)
        _CLEAR = create_handle()
    return _ADVANCE, _CLEAR
""", create_handle=create_handle, rendezvous=None)
    modules["legacy"] = _module("gpuwm.test_legacy", """
_CUDA_SW_CACHE = None
def _cuda_sw(tab):
    global _CUDA_SW_CACHE
    if _CUDA_SW_CACHE is None or _CUDA_SW_CACHE[0] is not tab:
        _CUDA_SW_CACHE = (tab, Engine(tab))
    return _CUDA_SW_CACHE[1]
""", Engine=Engine)
    modules["longwave"] = _module("gpuwm.test_longwave", """
_GPU_MODULE = None
_GPU_KERNELS = {}
_GPU_PREFLIGHTED = False
def _gpu_module():
    global _GPU_MODULE
    if _GPU_MODULE is None:
        _GPU_MODULE = create_handle()
    return _GPU_MODULE
def _gpu_kernel(name):
    if name not in _GPU_KERNELS:
        _GPU_KERNELS[name] = (_gpu_module(), name)
    return _GPU_KERNELS[name]
def gpu_preflight(force=False):
    global _GPU_PREFLIGHTED
    if _GPU_PREFLIGHTED and not force:
        return
    _gpu_kernel('probe')
    _GPU_PREFLIGHTED = True
""", create_handle=create_handle)
    modules["mcica"] = _module("gpuwm.test_mcica", """
_MCICA_GPU_MODULE = None
_MCICA_GPU_KERNELS = {}
_MCICA_GPU_PREFLIGHTED = False
def _mcica_gpu_module():
    global _MCICA_GPU_MODULE
    if _MCICA_GPU_MODULE is None:
        _MCICA_GPU_MODULE = create_handle()
    return _MCICA_GPU_MODULE
def _mcica_gpu_kernel(name):
    if name not in _MCICA_GPU_KERNELS:
        _MCICA_GPU_KERNELS[name] = (_mcica_gpu_module(), name)
    return _MCICA_GPU_KERNELS[name]
def mcica_gpu_preflight(force=False):
    global _MCICA_GPU_PREFLIGHTED
    if _MCICA_GPU_PREFLIGHTED and not force:
        return
    _mcica_gpu_kernel('probe')
    _MCICA_GPU_PREFLIGHTED = True
""", create_handle=create_handle)
    modules["prep"] = _module("gpuwm.test_prep", """
_XP_TABLES = {}
def _tables(xp):
    if xp.__name__ not in _XP_TABLES:
        _XP_TABLES[xp.__name__] = create_handle()
    return _XP_TABLES[xp.__name__]
""", create_handle=create_handle)
    modules["kernels"] = _module("gpuwm.test_kernels", """
@lru_cache(maxsize=None)
def load_module(name):
    return create_handle()
@lru_cache(maxsize=None)
def load_module_int_defines(name, defines):
    return create_handle()
@lru_cache(maxsize=None)
def get_kernel(name, func):
    return (load_module(name), func)
@lru_cache(maxsize=None)
def get_kernel_int_defines(name, func, defines):
    return (load_module_int_defines(name, defines), func)
""", create_handle=create_handle, lru_cache=functools.lru_cache)
    for group in ("reflectivity", "myjsfc"):
        modules[group] = _module(f"gpuwm.test_{group}", """
@lru_cache(maxsize=1)
def _device_tables():
    return create_handle()
""", create_handle=create_handle, lru_cache=functools.lru_cache)
    modules["noah"] = _module("gpuwm.test_noah", """
_DEVICE_TABLES = {}
def _device_tables(params, dzs):
    key = (id(params), tuple(dzs))
    if key not in _DEVICE_TABLES:
        _DEVICE_TABLES[key] = create_handle()
    return _DEVICE_TABLES[key]
""", create_handle=create_handle)
    modules["milbrandt"] = _module("gpuwm.test_milbrandt", """
_CK_DEVICE = None
def ck_vector_device():
    global _CK_DEVICE
    if _CK_DEVICE is None:
        _CK_DEVICE = create_handle()
    return _CK_DEVICE
""", create_handle=create_handle)
    modules["rrtm_taumol"] = _module("gpuwm.test_rrtm_taumol", """
_DEVICE_CACHE = {}
def _device_tables(tables, xp):
    key = (id(tables), xp.__name__)
    if key not in _DEVICE_CACHE:
        _DEVICE_CACHE[key] = create_handle()
    return _DEVICE_CACHE[key]
""", create_handle=create_handle)
    modules["rrtm_lw"] = _module("gpuwm.test_rrtm_lw", """
_DEVICE_CONSTANT_CACHE = {}
def _device_constant(xp, dtype, key, build):
    cache_key = (*key, xp.__name__, dtype)
    if cache_key not in _DEVICE_CONSTANT_CACHE:
        _DEVICE_CONSTANT_CACHE[cache_key] = create_handle()
    return _DEVICE_CONSTANT_CACHE[cache_key]
""", create_handle=create_handle)
    modules["rrtmgp"] = _module("gpuwm.test_rrtmgp", """
_CHUNK_SCRATCH = {}
@lru_cache(maxsize=8)
def _mcica_jump_tables(nlay, ngpt):
    return create_handle()
def _chunk_scratch(name, shape, *, xp, dtype='f4'):
    key = (name, tuple(shape), xp.__name__, dtype)
    if key not in _CHUNK_SCRATCH:
        _CHUNK_SCRATCH[key] = create_handle()
        return _CHUNK_SCRATCH[key], True
    return _CHUNK_SCRATCH[key], False
@lru_cache(maxsize=1)
def _rte_sm_count():
    return create_handle()
""", create_handle=create_handle, lru_cache=functools.lru_cache)
    modules["jacobi"] = _module("gpuwm.test_jacobi", """
@lru_cache(maxsize=None)
def _kernel(defines, shared_bytes):
    return get_kernel_int_defines('jacobi', 'eigh', defines)
""", get_kernel_int_defines=modules["kernels"].get_kernel_int_defines,
        lru_cache=functools.lru_cache)
    modules["mynn_flags"] = _module("gpuwm.test_mynn_flags", """
_VALIDITY_FLAGS = {}
def _validity_flags():
    key = 0
    if key not in _VALIDITY_FLAGS:
        _VALIDITY_FLAGS[key] = create_handle()
    return _VALIDITY_FLAGS[key]
""", create_handle=create_handle)
    return modules, identity


def test_concurrent_cumulus_kernel_pairs_publish_completely_per_rank():
    modules, identity = _modules()
    module = modules["cumulus_clock"]
    module.rendezvous = threading.Barrier(2)
    with scoped_physics_resources(modules=modules) as scope:
        def initialize(rank):
            token = identity.set(rank)
            try:
                with scope.rank(rank, 0):
                    pair = module._kernels()
                    assert all(handle["owner"] == rank for handle in pair)
                    assert module._kernels() == pair
                    return pair
            finally:
                identity.reset(token)
        with ThreadPoolExecutor(max_workers=2) as workers:
            pairs = list(workers.map(initialize, (0, 1)))
        assert pairs[0][0] is not pairs[1][0]
        assert pairs[0][1] is not pairs[1][1]
    assert module._ADVANCE is module._CLEAR is None


@pytest.mark.parametrize("devices", [(0, 0), (0, 1)])
def test_concurrent_scopes_own_scratch_tables_and_every_kernel_cache(devices):
    modules, identity = _modules()
    table = object()
    barrier = threading.Barrier(2)
    xp = ModuleType("cupy")
    with scoped_physics_resources(modules=modules) as scope:
        def worker(rank):
            marker = (rank, devices[rank])
            token = identity.set(marker)
            try:
                with scope.rank(*marker):
                    engine = modules["legacy"]._cuda_sw(table)
                    engine.scratch["field"] = bytes([rank])
                    modules["longwave"].gpu_preflight()
                    modules["mcica"].mcica_gpu_preflight()
                    lw = modules["longwave"]._gpu_kernel("band")
                    mcica = modules["mcica"]._mcica_gpu_kernel("cloud")
                    tables = modules["prep"]._tables(xp)
                    kernel = modules["kernels"].get_kernel("ysu", "column")
                    tier = modules["kernels"].get_kernel_int_defines("wsm6", "column", (("KMAX", 64),))
                    barrier.wait(timeout=5)
                    assert engine.owner == marker and engine.table is table
                    assert engine.scratch["field"] == bytes([rank])
                    assert lw[0]["owner"] == mcica[0]["owner"] == tables["owner"] == kernel[0]["owner"] == tier[0]["owner"] == marker
                    assert modules["legacy"]._cuda_sw(table) is engine
                    assert modules["kernels"].get_kernel("ysu", "column") is kernel
                    return engine, lw[0], mcica[0], tables, kernel[0]
            finally:
                identity.reset(token)
        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(worker, (0, 1)))
        assert all(left is not right for left, right in zip(outputs[0], outputs[1]))
        receipt = scope.receipt()
        assert len(receipt["scopes"]) == 10
        assert receipt["active_rank_contexts"] == 0
    assert scope.receipt()["module_entrypoints_restored"]
    assert modules["legacy"]._CUDA_SW_CACHE is None
    assert modules["longwave"]._GPU_MODULE is None
    assert modules["longwave"]._GPU_KERNELS == {}
    assert modules["mcica"]._MCICA_GPU_MODULE is None
    assert modules["prep"]._XP_TABLES == {}


def test_release_one_rank_workspace_cannot_clear_another_rank():
    modules, _identity = _modules()
    table = object()
    with scoped_physics_resources(modules=modules) as scope:
        with scope.rank(0, 0):
            first = modules["legacy"]._cuda_sw(table)
            first.scratch["kept"] = b"first"
        with scope.rank(1, 0):
            second = modules["legacy"]._cuda_sw(table)
            second.scratch["kept"] = b"second"
            second.release_scratch()
        assert first.scratch == {"kept": b"first"} and second.scratch == {}
        with scope.rank(0, 0):
            assert modules["legacy"]._cuda_sw(table) is first


def test_unscoped_calls_retain_original_cache_behavior_and_coefficients():
    modules, _identity = _modules()
    table = object()
    original = modules["legacy"]._cuda_sw(table)
    with scoped_physics_resources(modules=modules) as scope:
        assert modules["legacy"]._cuda_sw(table) is original
        with scope.rank(0, 0):
            rank = modules["legacy"]._cuda_sw(table)
            assert rank is not original and rank.table is original.table
    assert modules["legacy"]._cuda_sw(table) is original


def test_functions_and_old_and_new_aliases_restore_after_exception(monkeypatch):
    modules, _identity = _modules()
    original = modules["kernels"].get_kernel
    old_alias = ModuleType("gpuwm.test_old_kernel_reader")
    old_alias.get_kernel = original
    monkeypatch.setitem(sys.modules, old_alias.__name__, old_alias)
    with pytest.raises(ValueError, match="after cache call"):
        with scoped_physics_resources(modules=modules) as scope:
            new_alias = ModuleType("gpuwm.test_new_kernel_reader")
            new_alias.get_kernel = modules["kernels"].get_kernel
            monkeypatch.setitem(sys.modules, new_alias.__name__, new_alias)
            with scope.rank(0, 0):
                first = old_alias.get_kernel("ysu", "column")
                assert new_alias.get_kernel("ysu", "column") is first
                raise ValueError("after cache call")
    assert old_alias.get_kernel is new_alias.get_kernel is modules["kernels"].get_kernel is original
    assert scope.receipt()["module_entrypoints_restored"]


def test_original_function_code_and_compiler_arguments_are_retained():
    modules, _identity = _modules()
    originals = {group: {name: getattr(module, name) for name in _GROUPS[group][1]}
        for group, module in modules.items()}
    with scoped_physics_resources(modules=modules) as scope:
        for group, functions in originals.items():
            state = scope._state(group, (0, 0))
            for name, original in functions.items():
                original_body = getattr(original, "__wrapped__", original)
                cloned = state["functions"][name]
                clone_body = cloned.__wrapped__ if hasattr(original, "cache_parameters") else cloned
                assert clone_body.__code__ is original_body.__code__
                assert clone_body.__defaults__ == original_body.__defaults__
                assert clone_body.__kwdefaults__ == original_body.__kwdefaults__


@pytest.mark.parametrize("group,module_slot,kernels_slot,flag_slot,preflight", [
    ("longwave", "_GPU_MODULE", "_GPU_KERNELS", "_GPU_PREFLIGHTED", "gpu_preflight"),
    ("mcica", "_MCICA_GPU_MODULE", "_MCICA_GPU_KERNELS", "_MCICA_GPU_PREFLIGHTED", "mcica_gpu_preflight"),
])
def test_device_keyed_module_and_preflight_caches_start_empty_per_rank(group, module_slot, kernels_slot, flag_slot, preflight):
    modules, identity = _modules()
    module = modules[group]
    source = f'''
{module_slot} = {{99: object()}}
{kernels_slot} = {{}}
{flag_slot} = {{99}}
def {preflight}(device=0, force=False):
    if device in {flag_slot} and not force:
        return
    {module_slot}[device] = create_handle()
    {flag_slot}.add(device)
'''
    exec(source, vars(module))
    original = getattr(module, preflight)
    ambient_modules = getattr(module, module_slot)
    ambient_flags = getattr(module, flag_slot)
    with scoped_physics_resources(modules=modules) as scope:
        for rank in range(2):
            with scope.rank(rank, rank):
                getattr(module, preflight)(device=rank)
                state = scope._state(group, (rank, rank))
                assert set(state["globals"][module_slot]) == {rank}
                assert state["globals"][flag_slot] == {rank}
                assert state["functions"][preflight].__code__ is original.__code__
        assert getattr(module, module_slot) is ambient_modules
        assert getattr(module, flag_slot) is ambient_flags
    assert set(ambient_modules) == ambient_flags == {99}


def test_unknown_preflight_cache_type_names_the_concrete_breakage():
    modules, _identity = _modules()
    modules["longwave"]._GPU_PREFLIGHTED = 17
    with scoped_physics_resources(modules=modules) as scope:
        with pytest.raises(RuntimeError, match="invalid singleton"):
            scope._state("longwave", (0, 0))
