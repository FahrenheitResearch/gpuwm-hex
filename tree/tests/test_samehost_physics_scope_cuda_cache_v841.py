"""Rank-scoped clones of device-keyed kernel loaders stay cached.

The engine wraps its kernel loaders in gpuwm.core.device_cache.cuda_cache,
which exposes cache_info/cache_clear but not cache_parameters.  Before the
fix the rank clone of such a loader was uncached, so every step reloaded and
unloaded the sfclay module on every rank (nsys, RTX PRO 6000 host, 2026-10-11).
"""

from functools import lru_cache, wraps
import threading

from hexcore.drivers.samehost_physics_scope_v841 import scoped_physics_resources

from test_samehost_physics_scope_v841 import _module, _modules


def _cuda_cache_like(factory):
    """Mimic cuda_cache(maxsize=None): device-keyed lru, no cache_parameters."""
    @lru_cache(maxsize=None)
    def cached(owner, args, kwargs):
        return factory(*args, **dict(kwargs))

    @wraps(factory)
    def call(*args, **kwargs):
        return cached(("device",), args, tuple(kwargs.items()))

    call.cache_clear = cached.cache_clear
    call.cache_info = cached.cache_info
    return call


def test_cuda_cache_wrapped_loaders_load_each_module_once_per_rank():
    modules, identity = _modules()
    loads = []
    lock = threading.Lock()

    def record(name):
        with lock:
            loads.append((identity.get(), name))
        return {"owner": identity.get(), "unique": object()}

    modules["kernels"] = _module("gpuwm.test_kernels_cuda_cache", """
@cuda_cache
def load_module(name):
    return record(name)
@cuda_cache
def load_module_int_defines(name, defines):
    return record((name, defines))
@cuda_cache
def get_kernel(name, func):
    return (load_module(name), func)
@cuda_cache
def get_kernel_int_defines(name, func, defines):
    return (load_module_int_defines(name, defines), func)
""", cuda_cache=_cuda_cache_like, record=record)
    assert not hasattr(modules["kernels"].get_kernel, "cache_parameters")
    with scoped_physics_resources(modules=modules) as scope:
        for rank in (0, 1):
            token = identity.set((rank, rank))
            try:
                with scope.rank(rank, rank):
                    first = modules["kernels"].get_kernel("sfclay", "sfclay_column")
                    for _ in range(5):
                        assert modules["kernels"].get_kernel("sfclay", "sfclay_column") is first
                    assert first[0]["owner"] == (rank, rank)
            finally:
                identity.reset(token)
    # One load per rank, never one per call.
    assert sorted(loads) == [((0, 0), "sfclay"), ((1, 1), "sfclay")]
