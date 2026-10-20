"""WB5 words: the hex-cuda WSM6 scheme against the native correct-mode build, whole columns, both modes.

fixtures/wb5_native_columns.npz holds every output word of whole WSM6 columns run by the referee's own
mp_wsm6.F90 (MMM-physics 585bcc6, the source of native build a75371a6) with wsm6_plm_surface off (exact,
as released) and on (correct), built by make_referee_words.py with the WSM6 oracle's strict flags. The
port's wsm6_columns kernel must give the exact words from its exact text and the correct words from its
correct-mode text: the surface precipitation changes, the atmosphere does not.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.gpu

from hexcore.mpasphys import wsm6_v842

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
NAMES = ("volume", "surface", "radii", "dbz", "production")
MODES = {"exact": (), "correct": ("WB5",)}


@pytest.fixture(scope="module")
def referee():
    manifest = json.loads((FIXTURES / "wb5_native_columns.json").read_text())
    with np.load(FIXTURES / "wb5_native_columns.npz") as data:
        words = {k: data[k] for k in data.files}
    return manifest, words


@pytest.fixture(scope="module")
def header(tmp_path_factory):
    import gpuwm
    spec = importlib.util.spec_from_file_location("wb5_wsm6_maths",
                                                  HERE.parent / "wsm6_oracle/diagnostic_shared_math.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path_factory.mktemp("wsm6-maths") / "wsm6_device_math.cuh"
    module.assemble(Path(gpuwm.__file__).resolve().parents[1], output)
    return output.read_text()


def port_column(header, manifest, words, n, mode):
    case = manifest["cases"][n]
    nz = case["nz"]
    initial = [words[f"case{n}_v"], np.full(7, 1.25, np.float32), np.full((3, nz), 1.7e-5, np.float32),
               np.full(nz, -35, np.float32), np.full((2, nz), .0027, np.float32)]
    arrays = [cp.asarray(np.ascontiguousarray(a[..., None])) for a in initial]
    wsm6_v842.columns(*arrays, cp.asarray(np.array(manifest["constants"], np.float32)), dt=case["dt"],
                      hail_opt=case["hail"], device_math_header=header,
                      **({"corrections": MODES[mode]} if MODES[mode] else {}))
    return [cp.asnumpy(a)[..., 0] for a in arrays]


def test_the_fixture_exercises_the_fix(referee):
    manifest, _ = referee
    assert len(manifest["surface_changed_cases"]) >= 24          # the rain-out acts in most raining columns
    assert manifest["non_surface_changed_cases"] == []           # native correct changes surface words only


def test_the_slot_compiles_both_water_budget_corrections_together(header):
    # A correct-mode WSM6 slot passes every applied finding it has text for: ('WB3', 'WB5') together.
    assert len(wsm6_v842._kernels(header, 8, ("WB3", "WB5"))) == 4


@pytest.mark.parametrize("mode", MODES)
def test_wsm6_columns_equal_the_native_build_word_for_word(referee, header, mode):
    manifest, words = referee
    mismatched = {}
    for n in range(len(manifest["cases"])):
        got = port_column(header, manifest, words, n, mode)
        for name, a in zip(NAMES, got):
            want = words[f"case{n}_{mode}_{name}"]
            count = int(np.count_nonzero(a.view(np.uint32) != want.view(np.uint32)))
            if count:
                mismatched[(n, name)] = count
    assert mismatched == {}
