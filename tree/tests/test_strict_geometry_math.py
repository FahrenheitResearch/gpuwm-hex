"""G12 glibc 2.39 geometry must reproduce the native glibc 2.43 words."""
from pathlib import Path
import ctypes
import shutil
import subprocess
import sys
import os
import json

import numpy as np
import pytest
from hexcore import native_geometry_math as geometry, numerics, rust_hostprep
from hexcore import geometry_math_build as builder


@pytest.mark.parametrize("name,bits,expected", [
    ("asin", [1058437387], 1059117419),
    ("atan2", [3364558335, 3350512756], 3220260415),
])
def test_reference_words_for_scalar_array_and_batch(monkeypatch, name, bits, expected):
    monkeypatch.setattr(numerics, "is_strict", lambda: True)
    args = [np.array(b, dtype=np.uint32).view(np.float32) for b in bits]
    function = getattr(geometry, name)
    assert int(np.asarray(function(*args)).view(np.uint32)) == expected
    for count in (3, 4097):
        arrays = [np.full(count, a, dtype=np.float32) for a in args]
        assert np.all(function(*arrays).view(np.uint32) == expected)
    assert not geometry.unmodified_gnu_float32()
    assert rust_hostprep.deformation_engine_digest().startswith("gnu243-geometry:")
    # The opt-in must not change the existing fast GNU host binding.
    monkeypatch.setattr(numerics, "is_strict", lambda: False)
    native = getattr(geometry._LIBM, {"asin": "asinf", "atan2": "atan2f"}[name])
    assert function(*args).tobytes() == np.float32(native(*[float(a) for a in args])).tobytes()
    assert geometry.strict_binding() is None


def test_missing_strict_library_is_a_named_refusal(monkeypatch, tmp_path):
    monkeypatch.setattr(numerics, "is_strict", lambda: True)
    monkeypatch.setattr(geometry, "__file__", str(tmp_path / "native_geometry_math.py"))
    geometry._strict_library.cache_clear()
    geometry._strict_intrinsic.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="libhex_geometry_math.so.*host libm"):
            geometry.asin(np.float32(0.5))
    finally:
        geometry._strict_library.cache_clear()
        geometry._strict_intrinsic.cache_clear()


def test_source_build_reuses_cache_and_keys_included_headers(tmp_path, monkeypatch):
    package = tmp_path / "hexcore"
    source = package / "data/mpasphys-maths"
    shutil.copytree(Path(geometry.__file__).parent / "data/mpasphys-maths", source)
    monkeypatch.setenv("HEXCORE_NATIVE_CPU_CACHE", str(tmp_path / "cache"))
    first = builder.cached_geometry_math(package)
    stamp = first.stat().st_mtime_ns
    def no_build(*args):
        pytest.fail("a complete source-addressed cache entry must not rebuild")
    with monkeypatch.context() as patch:
        patch.setattr(builder, "build_geometry_math", no_build)
        assert builder.cached_geometry_math(package) == first
    assert first.stat().st_mtime_ns == stamp
    with (source / "glibc_trig_flt32.cuh").open("a") as header:
        header.write("\n// Cache invalidation probe.\n")
    second = builder.cached_geometry_math(package)
    assert second != first
    for path in (first, second):
        intrinsic = ctypes.CDLL(str(path)).hex_asinf
        intrinsic.argtypes = [ctypes.c_float]
        intrinsic.restype = ctypes.c_float
        assert np.float32(intrinsic(float(np.uint32(1058437387).view(np.float32)))).view(np.uint32) == 1059117419


def test_no_compiler_names_install_fix_without_host_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXCORE_NATIVE_CPU_CACHE", str(tmp_path))
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match=r"install C and C\+\+ compilers.*host libm"):
        builder.cached_geometry_math(Path(geometry.__file__).parent)


def test_concurrent_source_users_publish_one_library(tmp_path):
    env = dict(os.environ, HEXCORE_NATIVE_CPU_CACHE=str(tmp_path))
    # Match the parent package in both source and installed layouts; pytest's
    # in-process sys.path changes do not reach a newly started interpreter.
    env["PYTHONPATH"] = str(Path(builder.__file__).resolve().parents[1]) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    probe = (
        "from pathlib import Path; import json; "
        "from hexcore import geometry_math_build as b; "
        "p=b.cached_geometry_math(Path(b.__file__).parent); "
        "print(json.dumps([str(p),p.stat().st_mtime_ns]))"
    )
    processes = [subprocess.Popen([sys.executable, "-B", "-c", probe], env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for _ in range(2)]
    results = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=60)
        assert process.returncode == 0, stderr
        results.append(json.loads(stdout))
    assert results[0] == results[1]
    assert len(list(tmp_path.rglob("libhex_geometry_math.so"))) == 1
    assert not list(tmp_path.rglob("build-*"))
