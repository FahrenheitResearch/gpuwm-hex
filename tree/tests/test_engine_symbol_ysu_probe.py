"""The reservation probe uses the engine's current YSU compilation route."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def test_ysu_reservation_probe_resolves_current_engine_loader(monkeypatch, capsys):
    from gpuwm import wrf_exact, ysu_generations
    from gpuwm.core import kernels
    # gpuwm/core/ysu.py imports CuPy at module scope; public CPU CI (ci.yml battery) installs no CuPy,
    # so there the real provider cannot load. G0 installs the [gpu-cu13] extra and still runs this.
    pytest.importorskip("cupy", reason="gpuwm.core.ysu imports CuPy at module scope; CPU CI has no CuPy")
    # Load the real provider before replacing device calls below.
    import gpuwm.core.ysu

    folder = Path(__file__).resolve().parents[1] / "tools" / "device_memory_ledger"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("ysu_reservation_probe", folder / "reservation_probe.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    expected, defines = ysu_generations.ysu_kernel_selection(
        ysu_generations.YSU_DEFAULT_GENERATION, strict=wrf_exact.ENABLED)
    requests = []
    launches = []

    def get_function(name):
        assert name == "ysu_column"
        return lambda grid, block, args: launches.append((grid, block, args))

    def load(name):
        requests.append((name, ()))
        return SimpleNamespace(get_function=get_function)

    def load_defines(name, values):
        requests.append((name, values))
        return SimpleNamespace(get_function=get_function)

    monkeypatch.setattr(kernels, "load_module", load)
    monkeypatch.setattr(kernels, "load_module_int_defines", load_defines)
    runtime = SimpleNamespace(malloc=lambda size: 1, free=lambda pointer: None,
                              deviceSynchronize=lambda: None, getDevice=lambda: 0,
                              getDeviceProperties=lambda device: {
                                  "multiProcessorCount": 1, "maxThreadsPerMultiProcessor": 1024})
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)))
    monkeypatch.setattr(probe, "func_attributes", lambda cp, fn: {"max_threads_per_block": 128})
    monkeypatch.setattr(probe, "nvsmi_process_mib", lambda pid: 1.0)
    parameters = []

    def build_args(cp, np, params):
        parameters.extend(params)
        return ("owned-arguments",), []

    monkeypatch.setattr(probe, "build_args", build_args)
    assert probe.one(["--arwen-checkout", str(folder), "--module", "gpuwm.core.ysu",
                      "--kernel", "ysu_column", "--blocks", "1"]) == 0
    assert requests == [(expected, defines)]
    assert parameters == probe.signature(kernels.module_source(expected), "ysu_column")
    assert parameters
    assert launches == [((1,), (128,), ("owned-arguments",))]
    assert json.loads(capsys.readouterr().out)["params"] == len(parameters)
