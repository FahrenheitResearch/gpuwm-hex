"""RP7 (correct mode): MYNN's lowest-layer pressure thickness can no longer be zero.

THE BREAKAGE: mynn_tendencies sets delp(kts) = psfc - (hydrostatic pressure at the top of layer 1), mixing
MPAS's full surface pressure with its hydrostatic level pressures. A near-surface non-hydrostatic deficit of
half a layer makes it 0.0, moisture_check divides 0 by it, rqvblten (and through the moisture conversion
rqc/rqi/rqs) is NaN at level 1 of one column, and the dycore refuses the step with ~4,100 whole columns
non-finite (x1.40962 120 km plain hex, 2021-07-01 start, t = 1658400 s; MYNN-NAN-1).

The fixture is that column as the box's guard dumped it after the call (cell 37138, 24.8N 253.1E; its psfc
equals the hydrostatic interface pressure, so delp(kts) = 0.0 with these words). Exact mode keeps native
MPAS v8.4.2's NaN; RP7 makes level 1 finite and leaves every other level's words unchanged.
"""
from pathlib import Path

import numpy as np
import pytest

from hexcore.mpasphys import bl_mynn_v842 as m
from hexcore.mpasphys.driver_v842 import CORRECTIONS_V842

FIXTURE = Path(__file__).with_name("fixtures") / "mynn_rp7_column_x1.40962_20210701.npz"
EXACT = m.MYNN_TEXT_CORRECTIONS["RP7"][0][0]
CORRECTED = m.MYNN_TEXT_CORRECTIONS["RP7"][0][1]


def test_rp7_text_is_substituted_once_and_only_in_correct_mode():
    assert m.MYNN_MODULE_CUDA.count(EXACT) == 1
    assert "RP7" in CORRECTIONS_V842 and CORRECTIONS_V842["RP7"]["slots"] == {"pbl": ("bl_mynn",)}
    from hexcore.mpasphys.driver_v842 import correct_source, table_declarations, Backend
    assert "RP7" in table_declarations("pbl", "bl_mynn", Backend.HEX_CUDA)
    corrected = correct_source(m.MYNN_MODULE_CUDA, m.MYNN_TEXT_CORRECTIONS, {"RP1", "RP7"}, "test")
    assert EXACT not in corrected and corrected.count(CORRECTED) == 1
    assert correct_source(m.MYNN_MODULE_CUDA, m.MYNN_TEXT_CORRECTIONS, (), "test") == m.MYNN_MODULE_CUDA


def _maths():
    roots = sorted(Path.home().glob(".cache/gpuwm-hex/physics-resources/releases/*/v8.4.2/maths/*/"
                                    "bl-mynn-shared-maths.cuh"))
    if not roots:
        pytest.skip("installed MYNN shared maths absent (gpuwm-hex physics-resources build)")
    return roots[0].read_text()


def _column(cp):
    dump = np.load(FIXTURE)
    f32 = np.float32
    rd, rv = f32(287.0), f32(461.6)
    # mpas_atmphys_constants.F words in _MYNN_CONSTANTS order.
    constants = cp.asarray(np.array([f32(7.0)*rd/f32(2.0), f32(4.0)*rv, 2106.0, 4190.0, rv/rd-f32(1.0), rd/rv,
                                     9.80616, 0.4, 1.0e5, rd, rv, 0.6112, 17.67, 29.65, 273.15, 3.50e5,
                                     f32(2.50e6)+f32(3.50e5), 2.50e6], np.float32))
    fields = dict(delt=600.0, kds=1, kde=56, kms=1, kme=56, kts=1, kte=55, f_qc=True, f_qi=True, f_qs=True,
                  f_qoz=False, f_nc=False, f_ni=True, f_nifa=False, f_nwfa=False, f_nbca=False, do_restart=True,
                  do_dacycling=False, initflag=0, icloud_bl=0, spp_pbl=0, errflg=0,
                  its=1, ite=1, ims=1, ime=1, ids=1, ide=2, jts=1, jte=1, jms=1, jme=1, jds=1, jde=2,
                  errmsg=cp.empty(512, dtype=cp.uint8))
    for name in dump.files:
        if name != "cols" and not name.startswith("scalar_"):
            fields[name] = cp.asfortranarray(cp.asarray(dump[name]))
    return fields, constants


@pytest.mark.gpu
def test_rp7_makes_the_mynn_nan1_column_finite():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # noqa: BLE001
        pytest.skip("no CUDA device")
    driver = m.MynnBlDriver(shared_math_source=_maths())
    out = {}
    for label, corrections in (("exact", frozenset({"RP1"})), ("rp7", frozenset({"RP1", "RP7"}))):
        fields, constants = _column(cp)
        driver.run(fields, constants, options=m.MynnOptions(), corrections=corrections)
        out[label] = {k: cp.asnumpy(fields[k]) for k in ("rqvblten", "rqcblten", "rqiblten", "rqsblten",
                                                           "rthblten", "rublten", "qke")}
    names = ("rqvblten", "rqcblten", "rqiblten", "rqsblten")
    # Native MPAS v8.4.2 (exact): level 1 of every moisture tendency is NaN.
    assert all(np.isnan(out["exact"][n][0, 0]) for n in names)
    # RP7: finite everywhere; only the level-1 moisture words change.
    assert all(np.isfinite(out["rp7"][n]).all() for n in out["rp7"])
    for name in out["rp7"]:
        a, b = out["exact"][name], out["rp7"][name]
        if name in names:
            a, b = a[:, 1:], b[:, 1:]
        assert a.view(np.uint32).tolist() == b.view(np.uint32).tolist(), name
