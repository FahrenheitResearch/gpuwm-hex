"""The suite backend joins the seam's extra phase-one rates to the carrier.

``PersistentSuiteCudaPhysicsBackendV841._with_scalar_rates`` is the one
place MYNN's mixed nwfa/nifa rates (gpuwm mpas_column_suite
``last_scalar_rates``) enter hex's raw physics carrier; the coupling kernel
then forms ``mass * rate`` for every species of the row
(tests/test_physics_carrier_kernels.py), which is how NOAA MPAS adds
rnwfablten/rnifablten (mpas_atmphys_todynamics.F:394-405, 1d6e370e).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from hexcore import species_row as sr  # noqa: E402
from hexcore.cuda_physics_v841 import (  # noqa: E402
    CudaPhaseOneExecutionProvenanceV841,
    CudaRawColumnPhysicsV841,
)
from hexcore.cuda_suite_physics_v841 import (  # noqa: E402
    PersistentSuiteCudaPhysicsBackendV841,
)

NLEV, NCOL = 5, 7
MASSES = ("qv", "qc", "qr", "qi", "qs", "qg")


def _array(value):
    return cp.full((NLEV, NCOL), value, dtype=cp.float32)


def _backend(rates):
    backend = object.__new__(PersistentSuiteCudaPhysicsBackendV841)
    backend._seam = SimpleNamespace(last_scalar_rates=rates)
    backend._constructor = SimpleNamespace(n_levels=NLEV, n_columns=NCOL)
    return backend


def _frozen_carrier():
    return CudaRawColumnPhysicsV841(
        du=_array(0.0), dv=_array(0.0), dtheta=_array(0.0),
        dscalars={name: _array(float(i)) for i, name in enumerate(MASSES)},
        time_seconds=0.0,
        execution_provenance=CudaPhaseOneExecutionProvenanceV841.arwen_gwd_off(
            aggregate_executed=True))


def _tempo_names():
    row = sr.registered_species_rows().get("mp_tempo")
    if row is None:
        pytest.skip("this tree registers no TEMPO species row")
    return row.names()


def test_no_rates_returns_the_frozen_carrier_itself():
    raw = _frozen_carrier()
    out, extra = _backend({})._with_scalar_rates(raw, _tempo_names())
    assert out is raw and extra == ()


def test_rates_join_against_the_runs_own_row():
    names = _tempo_names()
    nwfa, nifa = _array(3.0), _array(-2.0)
    raw = _frozen_carrier()
    out, extra = _backend({"nwfa": nwfa, "nifa": nifa})._with_scalar_rates(raw, names)
    assert extra == ("nifa", "nwfa")
    assert tuple(out.scalar_names) == tuple(names)
    assert out.dscalars["nwfa"] is nwfa and out.dscalars["nifa"] is nifa
    for name in MASSES:
        assert out.dscalars[name] is raw.dscalars[name]
    assert out.du is raw.du and out.dtheta is raw.dtheta
    assert out.time_seconds == raw.time_seconds


def test_a_rate_for_a_species_outside_the_row_refuses():
    with pytest.raises(ValueError, match="outside"):
        _backend({"nbca": _array(1.0)})._with_scalar_rates(
            _frozen_carrier(), _tempo_names())


def test_a_mass_named_twice_refuses():
    with pytest.raises(ValueError, match="both"):
        _backend({"qv": _array(1.0)})._with_scalar_rates(
            _frozen_carrier(), _tempo_names())


def test_a_wrong_shape_refuses():
    bad = cp.zeros((NLEV + 1, NCOL), dtype=cp.float32)
    with pytest.raises(Exception):
        _backend({"nwfa": bad})._with_scalar_rates(
            _frozen_carrier(), _tempo_names())
