"""U-MPASPHYS-1: repeat kernel resolutions return the first resolution without re-hashing the source.

CPU tier: the cache is built without a device and its resolver is replaced by a counting stub, so
the test checks the memo's contract (same kernel, resolver once per distinct argument set,
cleared with the memory cache), not CUDA compilation.
"""
from types import SimpleNamespace

import pytest

from hexcore.cuda_backend import runtime
from hexcore.mpasphys import compile_v842, math_v842


def _cache(cls):
    cache = cls.__new__(cls)
    cache._resolved = {}
    cache._kernels = {}
    cache._modules = {}
    cache._compile_seconds = {}
    cache._observed = {}
    cache.post_launch = None
    cache.base_options = ("--std=c++17",)
    cache.capability = SimpleNamespace(sm="sm_120")
    cache._compile_manifest = SimpleNamespace(fingerprint_sha256="0" * 64)
    return cache


def test_base_cache_resolves_once_per_argument_set(monkeypatch):
    cache = _cache(runtime.KernelCache)
    calls = []

    def resolve(self, name, source, *, module_key, options=()):
        calls.append((name, source, tuple(options)))
        kernel = object()
        self._kernels[f"{self._module_key(source, tuple(str(o) for o in options))}:{name}"] = kernel
        return kernel

    monkeypatch.setattr(runtime.KernelCache, "_raw_kernel_resolve", resolve)
    first = cache.raw_kernel("k", "src", module_key="hexcore.test")
    for _ in range(5):
        assert cache.raw_kernel("k", "src", module_key="hexcore.test") is first
    other = cache.raw_kernel("k", "src2", module_key="hexcore.test")
    assert other is not first
    assert cache.raw_kernel("k", "src", module_key="hexcore.test", options=["-x"]) is not first
    assert len(calls) == 3
    cache.clear_memory_cache()
    cache.raw_kernel("k", "src", module_key="hexcore.test")
    assert len(calls) == 4


def test_mpas_cache_resolves_once_and_keys_numerics(monkeypatch):
    cache = _cache(compile_v842.MpasPhysicsKernelCache)
    calls = []

    def resolve(self, name, source, *, module_key, options=()):
        calls.append(name)
        kernel = object()
        self._last_resolution = (f"key:{name}", kernel, module_key)
        return kernel

    monkeypatch.setattr(compile_v842.MpasPhysicsKernelCache, "_raw_kernel_v842", resolve)
    strict = {"value": False}
    monkeypatch.setattr(compile_v842._hex_numerics, "is_strict", lambda: strict["value"])
    first = cache.raw_kernel("k", "src", module_key="hexcore.test")
    assert cache.raw_kernel("k", "src", module_key="hexcore.test") is first
    assert calls == ["k"]
    strict["value"] = True
    assert cache.raw_kernel("k", "src", module_key="hexcore.test") is not first
    assert calls == ["k", "k"]


def test_carrier_math_binding_is_cached_and_pure():
    pytest.importorskip("gpuwm.core.kernels")
    source = "float f(float x){return expf(x)+sinf(x);}"
    try:
        first = math_v842.bind_common_carrier_math(source)
    except FileNotFoundError:
        pytest.skip("installed gpuwm does not carry the carrier maths headers")
    assert math_v842.bind_common_carrier_math(source) is first
    assert "gfk_exp_fma(" in first and "glibc_sinf(" in first
