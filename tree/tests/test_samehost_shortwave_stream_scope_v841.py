"""The SW fence proxy belongs to one engine without changing shared CUDA."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from types import ModuleType, SimpleNamespace
import threading

import pytest

from hexcore.drivers.samehost_physics_scope_v841 import PhysicsResourceScopeError, scoped_physics_resources
from test_samehost_physics_scope_v841 import _module, _modules


_BITS = bytes.fromhex("00000000000000803412c07f01000000")


def _fixture():
    modules, identity = _modules()
    device = ContextVar("test_sw_device", default=0)
    stream = ContextVar("test_sw_stream")
    log = ContextVar("test_sw_log")

    class Stream:
        def __init__(self, error=None):
            self.calls = 0
            self.error = error

        def synchronize(self):
            self.calls += 1
            log.get().append("stream-sync")
            if self.error is not None:
                raise self.error

    def device_sync():
        log.get().append("device-sync")

    def asarray(value):
        log.get().append("upload")
        return value

    runtime = SimpleNamespace(deviceSynchronize=device_sync, getDevice=device.get, other=object())
    cuda = SimpleNamespace(runtime=runtime, get_current_stream=stream.get, Event=object())
    cp = ModuleType("cupy")
    cp.cuda, cp.asarray = cuda, asarray
    cp.ndarray, cp.float32, cp.other = type("Array", (), {}), object(), object()

    class CudaSW:
        def __init__(self, table):
            if isinstance(table, BaseException):
                raise table
            self.cp = cp
            self.table = table
            self.owner = identity.get()
            self.scratch = {}
            self.scratch_xp = cp

        def rrtmg_sw_batched_device(self, payload=_BITS):
            cp = self.cp
            out = cp.asarray(payload)
            log.get().append("kernel")
            self.cp.cuda.runtime.deviceSynchronize()
            log.get().append("returned")
            return out

    sw = ModuleType("gpuwm.test_sw")
    sw.CudaSW = CudaSW
    sw.other = object()
    modules["legacy"] = _module("gpuwm.test_legacy", """
_CUDA_SW_CACHE = None
def _cuda_sw(tab):
    global _CUDA_SW_CACHE
    if _CUDA_SW_CACHE is None or _CUDA_SW_CACHE[0] is not tab:
        _CUDA_SW_CACHE = (tab, _sw.CudaSW(tab))
    return _CUDA_SW_CACHE[1]
""", _sw=sw)

    @contextmanager
    def current(rank, selected_device, selected_stream, entries):
        tokens = ((identity, identity.set((rank, selected_device))),
                  (device, device.set(selected_device)), (stream, stream.set(selected_stream)),
                  (log, log.set(entries)))
        try:
            yield
        finally:
            for variable, token in reversed(tokens):
                variable.reset(token)

    return SimpleNamespace(modules=modules, sw=sw, cp=cp, Stream=Stream, current=current)


@pytest.mark.parametrize("devices", [(0, 0), (0, 1)])
def test_rank_engines_keep_original_code_and_only_their_own_stream_wait(devices):
    f = _fixture()
    table = object()
    original_factory = f.modules["legacy"]._cuda_sw
    original_init = f.sw.CudaSW.__init__
    original_body = f.sw.CudaSW.rrtmg_sw_batched_device
    ambient = original_factory(table)
    barrier = threading.Barrier(2)
    with scoped_physics_resources(modules=f.modules) as scope:
        def worker(rank):
            entries, selected_stream = [], f.Stream()
            with f.current(rank, devices[rank], selected_stream, entries), scope.rank(rank, devices[rank]):
                engine = f.modules["legacy"]._cuda_sw(table)
                assert engine is f.modules["legacy"]._cuda_sw(table)
                assert type(engine) is f.sw.CudaSW
                assert type(engine).__init__ is original_init
                assert engine.rrtmg_sw_batched_device.__func__ is original_body
                assert engine.scratch_xp is f.cp
                assert scope._state("legacy", (rank, devices[rank]))["functions"]["_cuda_sw"].__code__ is original_factory.__code__
                engine.scratch["rank"] = rank
                barrier.wait(timeout=5)
                assert engine.scratch["rank"] == rank
                assert engine.rrtmg_sw_batched_device() == _BITS
                assert entries == ["upload", "kernel", "stream-sync", "returned"]
                assert selected_stream.calls == 1
                assert engine.owner == (rank, devices[rank])
                return engine

        with ThreadPoolExecutor(max_workers=2) as pool:
            engines = list(pool.map(worker, (0, 1)))
        assert engines[0] is not engines[1] and all(engine is not ambient for engine in engines)
        rows = [row for row in scope.receipt()["scopes"] if row["group"] == "legacy"]
        assert len(rows) == 2 and all(row["shortwave_completion_on_current_stream"] for row in rows)
    assert f.modules["legacy"]._cuda_sw is original_factory
    assert original_factory(table) is ambient
    assert f.sw.CudaSW.__init__ is original_init and f.sw.CudaSW.rrtmg_sw_batched_device is original_body


def test_ambient_class_cupy_runtime_and_unrelated_attributes_remain_original():
    f = _fixture()
    table = object()
    runtime, cuda, sw_class = f.cp.cuda.runtime, f.cp.cuda, f.sw.CudaSW
    device_sync = runtime.deviceSynchronize
    with scoped_physics_resources(modules=f.modules) as scope:
        entries, selected_stream = [], f.Stream()
        with f.current(0, 0, selected_stream, entries):
            ambient = f.modules["legacy"]._cuda_sw(table)
            assert ambient.cp is f.cp
            assert ambient.rrtmg_sw_batched_device() == _BITS
            assert entries == ["upload", "kernel", "device-sync", "returned"]
            with scope.rank(0, 0):
                engine = f.modules["legacy"]._cuda_sw(table)
                assert engine.cp.asarray is f.cp.asarray
                assert engine.cp.ndarray is f.cp.ndarray
                assert engine.cp.float32 is f.cp.float32
                assert engine.cp.__name__ == f.cp.__name__
                assert engine.cp.other is f.cp.other
                assert engine.cp.cuda.Event is cuda.Event
                assert engine.cp.cuda.get_current_stream is cuda.get_current_stream
                assert engine.cp.cuda.runtime.getDevice is runtime.getDevice
                assert engine.cp.cuda.runtime.other is runtime.other
                assert scope._state("legacy", (0, 0))["globals"]["_sw"].other is f.sw.other
            assert f.sw.CudaSW is sw_class
            assert f.cp.cuda is cuda and cuda.runtime is runtime
            assert runtime.deviceSynchronize is device_sync
    assert ambient.cp is f.cp and f.cp.cuda.runtime.deviceSynchronize is device_sync


def test_fence_resolves_current_stream_at_call_and_propagates_cuda_error():
    f = _fixture()
    first, second = f.Stream(), f.Stream()
    error = RuntimeError("injected stream failure")
    failed = f.Stream(error)
    with scoped_physics_resources(modules=f.modules) as scope:
        with f.current(2, 0, first, []), scope.rank(2, 0):
            engine = f.modules["legacy"]._cuda_sw(object())
        for selected in (first, second):
            with f.current(2, 0, selected, []), scope.rank(2, 0):
                engine.cp.cuda.runtime.deviceSynchronize()
        assert first.calls == second.calls == 1
        with f.current(2, 0, failed, []), scope.rank(2, 0):
            with pytest.raises(RuntimeError) as caught:
                engine.cp.cuda.runtime.deviceSynchronize()
        assert caught.value is error and failed.calls == 1


def test_fence_refuses_wrong_rank_device_inactive_and_restored_scope():
    f = _fixture()
    selected = f.Stream()
    with scoped_physics_resources(modules=f.modules) as scope:
        with f.current(0, 0, selected, []), scope.rank(0, 0):
            engine = f.modules["legacy"]._cuda_sw(object())
        with f.current(0, 0, selected, []):
            with pytest.raises(PhysicsResourceScopeError, match="owning active rank"):
                engine.cp.cuda.runtime.deviceSynchronize()
        with f.current(1, 0, selected, []), scope.rank(1, 0):
            with pytest.raises(PhysicsResourceScopeError, match="owning active rank"):
                engine.cp.cuda.runtime.deviceSynchronize()
        with f.current(0, 1, selected, []), scope.rank(0, 0):
            with pytest.raises(PhysicsResourceScopeError, match="CUDA device differs"):
                engine.cp.cuda.runtime.deviceSynchronize()
    with f.current(0, 0, selected, []):
        with pytest.raises(PhysicsResourceScopeError, match="owning active rank"):
            engine.cp.cuda.runtime.deviceSynchronize()
    assert selected.calls == 0


def test_constructor_failure_keeps_original_error_cache_and_shared_class():
    f = _fixture()
    error = RuntimeError("injected constructor failure")
    original_class = f.sw.CudaSW
    with scoped_physics_resources(modules=f.modules) as scope:
        with f.current(0, 0, f.Stream(), []), scope.rank(0, 0):
            with pytest.raises(RuntimeError) as caught:
                f.modules["legacy"]._cuda_sw(error)
            assert caught.value is error
            assert scope._state("legacy", (0, 0))["globals"]["_CUDA_SW_CACHE"] is None
    assert f.sw.CudaSW is original_class and f.modules["legacy"]._CUDA_SW_CACHE is None
