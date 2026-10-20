"""WB3 words: every hex-cuda microphysics precipitation writeback against the native correct-mode build.

fixtures/wb3_native_words.bin holds precipw from the referee's own precip_to_MPAS loop (MPAS-A v8.4.2 plus the
gated water-budget patch, the source of native build a75371a6), compiled with the pinned strict GNU flags by
make_referee_words.py: dry_density = .false. (exact, as released) then .true. (correct). Each kernel must give
the exact words from its exact text and the correct words from its correct-mode text. The WSM6 writeback
must change nothing else.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.gpu

from hexcore.mpasphys import kessler_v842, thompson_v842, wsm6_v842

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
TESTS = HERE.parent
MODES = {"exact": (), "correct": ("WB3",)}


def referee():
    manifest = json.loads((FIXTURES / "wb3_native_words.json").read_text())
    nz, nc = manifest["nz"], manifest["nc"]
    raw = np.fromfile(FIXTURES / "wb3_inputs.bin", dtype=">f4").astype(np.float32).reshape(3, nz, nc)
    words = np.fromfile(FIXTURES / "wb3_native_words.bin", dtype=">f4").astype(np.float32).reshape(2, nc)
    return raw[0], raw[1], raw[2], {"exact": words[0], "correct": words[1]}


def bits(a):
    return np.asarray(a, dtype=np.float32).view(np.uint32).tolist()


def engine_root():
    import gpuwm
    return Path(gpuwm.__file__).resolve().parents[1]


def helper(relative, name):
    spec = importlib.util.spec_from_file_location(name, TESTS / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_referee_words_are_the_exact_and_correct_formulas():
    qv, rho, dz, want = referee()
    # Host binary32 replay of the referee loop, both branches, in its own operation order.
    for mode, density in (("exact", lambda k: rho[k] / (np.float32(1.0) + qv[k])), ("correct", lambda k: rho[k])):
        total = np.zeros(qv.shape[1], np.float32)
        for k in range(qv.shape[0]):
            total = (total + (qv[k] * density(k)) * dz[k]).astype(np.float32)
        assert bits(total) == bits(want[mode]), mode
    assert np.all(want["correct"][2:] > want["exact"][2:])     # exact is low wherever there is vapour


@pytest.mark.parametrize("mode", MODES)
def test_kessler_precipitation_writeback(mode):
    qv, rho, dz, want = referee()
    nz, nc = qv.shape
    module = kessler_v842._module(kessler_v842.shared_math_header(), *((MODES[mode],) if MODES[mode] else ()))
    out = {n: cp.zeros(nc, cp.float32) for n in ("rainnc", "rainncv", "precipw")}
    counter = cp.zeros(nc, cp.int32)
    module.get_function("kessler_precip_to_mpas")(((nc + 63) // 64,), (64,), (
        cp.asarray(rho), cp.asarray(qv), cp.asarray(dz), cp.zeros(nc, cp.float32),
        out["rainnc"], out["rainncv"], out["precipw"], counter,
        np.float32(0.0), np.int32(0), np.int32(nz), np.int32(nc)))
    assert bits(out["precipw"].get()) == bits(want[mode])


@pytest.mark.parametrize("mode", MODES)
def test_thompson_precipitation_writeback(mode):
    qv, rho, dz, want = referee()
    nz, nc = qv.shape
    maths = helper("thompson_v842_oracle/replay.py", "wb3_thompson_replay").engine_math_source(engine_root())
    fields = {"qv": cp.asarray(qv), "rho": cp.asarray(rho), "dz": cp.asarray(dz)}
    surface = {n: cp.zeros(nc, cp.float32) for n in thompson_v842.SURFACE_ARGUMENTS}
    diagnostics = {n: cp.zeros(nc, cp.float32) for n in thompson_v842.PRECIP_ARGUMENTS}
    diagnostics["i_rainnc"] = cp.zeros(nc, cp.int32)
    thompson_v842.launch_precip("to", fields, surface, diagnostics, device_math_source=maths, l_acrain=False,
                                bucket=0.0, **({"corrections": MODES[mode]} if MODES[mode] else {}))
    assert bits(diagnostics["precipw"].get()) == bits(want[mode])


@pytest.fixture(scope="module")
def wsm6_header(tmp_path_factory):
    output = tmp_path_factory.mktemp("wsm6-maths") / "wsm6_device_math.cuh"
    helper("wsm6_oracle/diagnostic_shared_math.py", "wb3_wsm6_maths").assemble(engine_root(), output)
    return output.read_text()


def wsm6_writeback(header, corrections):
    """Run wsm6_to_mpas on the referee columns; return every field and counter it writes."""
    qv, rho, dz, _ = referee()
    nz, nc = qv.shape
    rng = np.random.default_rng(3)
    v = rng.uniform(0.5, 1.5, (10, nz, nc)).astype(np.float32)
    v[0] = rng.uniform(250.0, 300.0, (nz, nc))                 # theta from the scheme
    v[1], v[7], v[9] = qv, rho, dz
    fields = {}
    names = wsm6_v842.FIELD_NAMES
    surface = set(names[30:40]) | set(names[41:])            # as wsm6_v842.step lays them out
    for name in names:
        shape = (nz + 1, nc) if name == "zgrid" else (nc,) if name in surface else (nz, nc)
        fields[name] = cp.asarray(rng.uniform(0.5, 1.5, shape).astype(np.float32))
    fields["zgrid"] = cp.asarray(np.cumsum(np.vstack([np.zeros((1, nc), np.float32), dz]), axis=0,
                                           dtype=np.float32))
    pointers = cp.asarray([fields[n].data.ptr for n in wsm6_v842.FIELD_NAMES], dtype=cp.uint64)
    counter = cp.zeros(nc, cp.int32)
    s = cp.asarray(rng.uniform(0.0, 1.0, (7, nc)).astype(np.float32))
    r = cp.asarray(rng.uniform(1e-6, 1e-4, (3, nz, nc)).astype(np.float32))
    prod = cp.asarray(rng.uniform(0.0, 1e-3, (2, nz, nc)).astype(np.float32))
    constants = cp.asarray(rng.uniform(1.0, 2.0, 26).astype(np.float32))
    kernel = wsm6_v842._kernels(header, max(2, nz), *((corrections,) if corrections else ()))[3]
    kernel(((nc + 31) // 32,), (32,), (
        np.int32(nz), np.int32(nc), np.int32(nc), np.int32(nz), np.float32(10.0), np.float32(287.0),
        np.float32(1.6), np.float32(1.61), np.float32(0.4), np.float32(1.0e5), np.float32(9.81),
        np.float32(0.0), np.int32(0), constants, pointers, counter, cp.asarray(v), s, r, prod))
    return {name: array.get() for name, array in fields.items()} | {"i_rainnc": counter.get()}


def test_wsm6_precipitation_writeback_changes_only_precipw(wsm6_header):
    _, _, _, want = referee()
    exact, correct = (wsm6_writeback(wsm6_header, MODES[m]) for m in ("exact", "correct"))
    assert bits(exact["precipw"]) == bits(want["exact"])
    assert bits(correct["precipw"]) == bits(want["correct"])
    for name in exact:
        if name != "precipw":
            assert np.array_equal(exact[name].view(np.uint32) if exact[name].dtype == np.float32 else exact[name],
                                  correct[name].view(np.uint32) if correct[name].dtype == np.float32
                                  else correct[name]), name
