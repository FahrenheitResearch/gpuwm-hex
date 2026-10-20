"""Noah-MP's vegetation fraction comes from the init's ``shdmax``, not the engine's fill.

The defect (obs scorecard, 2026-10-12): the frozen engine constructor hands
``initialize_physics`` no ``shdmax``, so every vegetated land column ran
Noah-MP at FVEG = 0.90 (the driver's launch-field fill) whatever the mesh said.
These CPU tests hold the three things the fix depends on: the sealed mapping
carries the init's field in percent, the engine constructor never sees a
keyword it does not have, and the bind writes the driver's own carrier or
refuses by name.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hexcore import cuda_arwen_physics_v841 as adapter
from hexcore.cuda_arwen_physics_v841 import SealedArwenConstructorV841

from test_ntiedtke_column_row import _mapping


def _with_shdmax(values, shdmax):
    out = dict(values)
    out["shdmax"] = np.asarray(shdmax, dtype=np.float32)
    return out


def test_the_sealed_mapping_carries_shdmax_and_keeps_it_from_the_engine_constructor():
    values = _with_shdmax(
        _mapping(scheme="ntiedtke", cumulus_seconds=60.0), [5.0, 48.5, 92.0]
    )
    sealed = SealedArwenConstructorV841.from_mapping(values)
    assert "shdmax" not in sealed.arwen_kwargs()
    assert set(sealed.arwen_kwargs()) == adapter._CONSTRUCTOR_KEYS
    np.testing.assert_array_equal(
        sealed.noahmp_statics()["shdmax"], np.float32([5.0, 48.5, 92.0])
    )
    assert "init shdmax" in sealed.receipt()["noahmp_vegetation_fraction"]


def test_a_mapping_without_shdmax_seals_as_before_and_names_the_engine_fill():
    plain = SealedArwenConstructorV841.from_mapping(
        _mapping(scheme="gf", cumulus_seconds=60.0)
    )
    assert dict(plain.noahmp_statics()) == {}
    assert "90 %" in plain.receipt()["noahmp_vegetation_fraction"]
    carried = SealedArwenConstructorV841.from_mapping(
        _with_shdmax(_mapping(scheme="gf", cumulus_seconds=60.0), [5.0, 48.5, 92.0])
    )
    # The field is part of the sealed identity: two forecasts that differ only
    # in vegetation fraction are different forecasts.
    assert carried.identity_sha256 != plain.identity_sha256


def test_shdmax_outside_percent_is_refused():
    base = _mapping(scheme="gf", cumulus_seconds=60.0)
    with pytest.raises(ValueError, match=r"percent \[0, 100\]"):
        SealedArwenConstructorV841.from_mapping(_with_shdmax(base, [5.0, 101.0, 92.0]))
    with pytest.raises((ValueError, FloatingPointError), match="shdmax"):
        SealedArwenConstructorV841.from_mapping(_with_shdmax(base, [5.0, np.nan, 92.0]))


def test_a_zero_to_one_fraction_on_a_real_mesh_is_refused_by_name():
    ncol = 128
    base = _mapping(scheme="gf", cumulus_seconds=60.0, ncol=ncol)
    with pytest.raises(ValueError, match="looks like a 0..1 fraction"):
        SealedArwenConstructorV841.from_mapping(
            _with_shdmax(base, np.linspace(0.0, 0.95, ncol))
        )


def _backend_for(sealed):
    backend = object.__new__(adapter.PersistentTwoPhaseCudaPhysicsBackendV841)
    backend._constructor = sealed
    return backend


def test_the_bind_overwrites_the_engine_fill_in_the_driver_carrier(capsys):
    sealed = SealedArwenConstructorV841.from_mapping(
        _with_shdmax(_mapping(scheme="ntiedtke", cumulus_seconds=60.0), [5.0, 48.5, 92.0])
    )
    fields = {"shdmax": np.full((1, 3), 90.0, np.float32)}
    seam = SimpleNamespace(_driver=SimpleNamespace(fields=fields))
    backend = _backend_for(sealed)
    backend._bind_noahmp_statics(seam)
    np.testing.assert_array_equal(fields["shdmax"], np.float32([[5.0, 48.5, 92.0]]))
    receipt = backend._noahmp_statics_receipt
    assert receipt["bound"] is True
    assert receipt["fields"]["shdmax"]["engine_fill_min_max"] == [90.0, 90.0]
    assert "engine fill 90..90 %" in capsys.readouterr().out


def test_a_seam_without_the_carrier_is_refused_and_names_the_breakage():
    sealed = SealedArwenConstructorV841.from_mapping(
        _with_shdmax(_mapping(scheme="ntiedtke", cumulus_seconds=60.0), [5.0, 48.5, 92.0])
    )
    backend = _backend_for(sealed)
    with pytest.raises(TypeError, match="90 % vegetation on every column"):
        backend._bind_noahmp_statics(SimpleNamespace(_driver=SimpleNamespace(fields={})))
    with pytest.raises(ValueError, match="carrier holds 2 columns"):
        backend._bind_noahmp_statics(
            SimpleNamespace(
                _driver=SimpleNamespace(fields={"shdmax": np.zeros((1, 2), np.float32)})
            )
        )


def test_a_mapping_without_shdmax_binds_nothing_and_says_so():
    sealed = SealedArwenConstructorV841.from_mapping(
        _mapping(scheme="gf", cumulus_seconds=60.0)
    )
    fields = {"shdmax": np.full((1, 3), 90.0, np.float32)}
    backend = _backend_for(sealed)
    backend._bind_noahmp_statics(SimpleNamespace(_driver=SimpleNamespace(fields=fields)))
    assert backend._noahmp_statics_receipt["bound"] is False
    np.testing.assert_array_equal(fields["shdmax"], np.full((1, 3), 90.0, np.float32))


def test_the_forecast_door_reads_shdmax_from_the_init():
    import inspect

    from hexcore.drivers import run_cuda_v841_forecast as forecast

    source = inspect.getsource(forecast.build_forecast_constructor_values)
    assert 'surface["shdmax"]' in source
    assert 'read(dataset, "shdmax", dtype=np.float32, dimensions=("nCells",))' in source
