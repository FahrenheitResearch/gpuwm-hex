"""A forecast's start frame carries a real 2 m humidity and surface pressure.

THE BREAKAGE THIS PREVENTS: through 0.3.2 every hex forecast published its
F000 frame with ``q2`` = +0 in every cell and ``psfc`` = 100000 Pa in every
cell, the seam's values from before its first physics call.  A
limited-area forecast's hour-0 2 m dewpoint map was drawn from that zero
humidity while hours 1 to 36 were right.  The forecast
driver now completes the start frame from the start file (``q2``) or the
model's own start state (``q2`` fallback, ``psfc``), and these tests go red
if the completion stops happening, stops happening first, or starts
touching a frame the physics has already run on.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from hexcore.drivers import run_cuda_v841_forecast as forecast

N_CELLS = 64
N_LEVELS = 5


def _start_frame(*, with_psfc: bool = True, with_q2: bool = True) -> dict:
    rng = np.random.default_rng(20260929)
    # below saturation everywhere at 280 K and up to 101500 Pa (about 6.2 g/kg)
    qv = rng.uniform(0.001, 0.004, size=(N_LEVELS, N_CELLS)).astype(np.float32)
    arrays = {
        "qv": qv,
        "t2": rng.uniform(280.0, 300.0, size=N_CELLS).astype(np.float32),
        "surface_pressure": rng.uniform(80000.0, 101500.0, size=N_CELLS).astype(
            np.float32
        ),
        "u10": rng.uniform(-5.0, 5.0, size=N_CELLS).astype(np.float32),
    }
    if with_q2:
        arrays["q2"] = np.zeros(N_CELLS, dtype=np.float32)
    if with_psfc:
        arrays["psfc"] = np.full(N_CELLS, 100000.0, dtype=np.float32)
    receipt = {
        "step": 0,
        "arwen_v2_surface_execution": {
            "surface_classification": {},
            "last_noahmp_census": None,
        },
        "arrays": {
            name: forecast._snapshot_array_record(value)
            for name, value in arrays.items()
        },
    }
    return {"arrays": arrays, "receipt": receipt}


def _not_carried() -> dict:
    return {"carried": False, "q2": None, "reason": "test: the start file's q2 is zero"}


def test_carried_start_humidity_is_published_bitwise():
    snapshot = _start_frame()
    start = np.linspace(0.003, 0.011, N_CELLS, dtype=np.float32)
    result = forecast.complete_f000_surface_diagnostics(
        snapshot, {"carried": True, "q2": start, "sha256": "x"}
    )
    published = snapshot["arrays"]["q2"]
    assert published.dtype == np.float32
    assert published.view(np.uint32).tolist() == start.view(np.uint32).tolist()
    assert result["q2"]["source"] == "start file q2"
    # the carrier is not aliased: a later write to the start file's array
    # cannot reach the published frame
    start[0] = 1.0
    assert published[0] != np.float32(1.0)


def test_without_start_humidity_q2_is_the_lowest_level_bounded_by_saturation():
    snapshot = _start_frame()
    arrays = snapshot["arrays"]
    arrays["qv"][0, 3] = np.float32(0.5)  # far past saturation
    arrays["qv"][0, 4] = np.float32(-1.0e-6)  # the init's small negatives
    lowest = arrays["qv"][0].copy()
    ceiling = forecast.saturation_mixing_ratio(arrays["t2"], arrays["surface_pressure"])
    result = forecast.complete_f000_surface_diagnostics(snapshot, _not_carried())
    q2 = arrays["q2"]
    inside = (lowest >= 0.0) & (lowest <= ceiling)
    assert np.array_equal(q2[inside], lowest[inside])
    assert q2[3] == ceiling[3]
    assert q2[4] == np.float32(0.0)
    assert np.all(q2 >= 0.0)
    assert np.all(q2 <= ceiling)
    assert result["q2"]["cells_bounded_at_saturation"] == 1
    assert result["q2"]["cells_floored_at_zero"] == 1
    assert "lowest model level" in result["q2"]["source"]
    assert result["q2"]["why"] == "test: the start file's q2 is zero"


def test_psfc_becomes_the_model_surface_pressure():
    snapshot = _start_frame()
    expected = snapshot["arrays"]["surface_pressure"].copy()
    result = forecast.complete_f000_surface_diagnostics(snapshot, _not_carried())
    psfc = snapshot["arrays"]["psfc"]
    assert psfc.view(np.uint32).tolist() == expected.view(np.uint32).tolist()
    assert psfc is not snapshot["arrays"]["surface_pressure"]
    assert result["psfc"]["placeholder_minimum"] == 100000.0
    assert result["psfc"]["placeholder_maximum"] == 100000.0


def test_the_receipt_rows_describe_the_published_bytes():
    snapshot = _start_frame()
    stale_q2 = snapshot["receipt"]["arrays"]["q2"]["sha256"]
    forecast.complete_f000_surface_diagnostics(snapshot, _not_carried())
    rows = snapshot["receipt"]["arrays"]
    for name in ("q2", "psfc"):
        assert rows[name] == forecast._snapshot_array_record(snapshot["arrays"][name])
    assert rows["q2"]["sha256"] != stale_q2
    assert rows["q2"]["nonzero"] == N_CELLS


def test_nothing_else_in_the_frame_moves():
    snapshot = _start_frame()
    before = {
        name: value.copy()
        for name, value in snapshot["arrays"].items()
        if name not in forecast.F000_COMPLETED_SURFACE_DIAGNOSTICS
    }
    forecast.complete_f000_surface_diagnostics(snapshot, _not_carried())
    for name, value in before.items():
        assert np.array_equal(snapshot["arrays"][name], value), name


def test_a_row_without_the_fields_gains_none_at_f000():
    snapshot = _start_frame(with_psfc=False, with_q2=False)
    result = forecast.complete_f000_surface_diagnostics(snapshot, _not_carried())
    assert "q2" not in snapshot["arrays"] and "psfc" not in snapshot["arrays"]
    assert result["absent_from_this_row"] == ["q2", "psfc"]


def test_refused_on_any_frame_but_step_zero():
    snapshot = _start_frame()
    snapshot["receipt"]["step"] = 12
    with pytest.raises(ValueError, match="step-0 frame only"):
        forecast.complete_f000_surface_diagnostics(snapshot, _not_carried())


def test_refused_once_the_physics_has_run():
    snapshot = _start_frame()
    snapshot["receipt"]["arwen_v2_surface_execution"]["last_noahmp_census"] = {
        "land": 1
    }
    before = snapshot["arrays"]["q2"].copy()
    with pytest.raises(ValueError, match="already run"):
        forecast.complete_f000_surface_diagnostics(snapshot, _not_carried())
    assert np.array_equal(snapshot["arrays"]["q2"], before)


def test_refused_when_q2_is_not_the_placeholder_and_the_frame_is_left_alone():
    snapshot = _start_frame()
    snapshot["arrays"]["q2"][5] = np.float32(0.004)
    psfc = snapshot["arrays"]["psfc"].copy()
    with pytest.raises(ValueError, match="placeholder"):
        forecast.complete_f000_surface_diagnostics(snapshot, _not_carried())
    assert np.array_equal(snapshot["arrays"]["psfc"], psfc)


def test_a_start_file_for_another_mesh_is_refused():
    snapshot = _start_frame()
    with pytest.raises(ValueError, match="different mesh"):
        forecast.complete_f000_surface_diagnostics(
            snapshot,
            {"carried": True, "q2": np.full(N_CELLS + 1, 0.005, dtype=np.float32)},
        )


def test_saturation_curve_is_the_init_writers():
    # rw_mpas_init's diagnose_q2 at rh2 = 100 %: es = 6.112 exp(17.27 (T -
    # 273.16) / (T - 35.86)) hPa, rs = 0.622 es 100 / (psfc - es 100).
    temperature = np.array([253.15, 273.15, 293.15, 308.15], dtype=np.float32)
    pressure = np.array([70000.0, 85000.0, 101325.0, 100000.0], dtype=np.float32)
    es = 6.112 * np.exp(17.27 * (temperature.astype(np.float64) - 273.16)
                        / (temperature.astype(np.float64) - 35.86))
    expected = 0.622 * es * 100.0 / (pressure - es * 100.0)
    got = forecast.saturation_mixing_ratio(temperature, pressure)
    assert got.dtype == np.float32
    assert np.allclose(got, expected, rtol=1e-6)
    # 20 C at sea level holds about 14.7 g/kg
    assert abs(float(got[2]) - 0.01470) < 5e-5


def test_start_humidity_loader(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")

    def write(name: str, q2: np.ndarray | None) -> object:
        path = tmp_path / name
        with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
            dataset.createDimension("Time", None)
            dataset.createDimension("nCells", N_CELLS)
            if q2 is not None:
                variable = dataset.createVariable("q2", "f4", ("Time", "nCells"))
                variable[0, :] = q2
        return path

    zero = forecast.load_f000_start_humidity(
        write("zero.nc", np.zeros(N_CELLS, dtype=np.float32))
    )
    assert zero["carried"] is False and zero["q2"] is None
    assert "relative humidity" in zero["reason"]

    field = np.linspace(0.001, 0.01, N_CELLS, dtype=np.float32)
    real = forecast.load_f000_start_humidity(write("real.nc", field))
    assert real["carried"] is True
    assert real["q2"].view(np.uint32).tolist() == field.view(np.uint32).tolist()

    absent = forecast.load_f000_start_humidity(write("absent.nc", None))
    assert absent["carried"] is False and "no q2" in absent["reason"]


def test_the_driver_completes_the_start_frame_before_anything_reads_it():
    source = inspect.getsource(forecast.execute_forecast)
    # The consumer is defined before the producer, but runs only after capture
    # (and after its transfer event on the asynchronous path).
    finish = source[source.index("def finish_capture("):source.index("def capture(step")]
    capture = source[source.index("def capture(step"):]
    complete = finish.index("complete_f000_surface_diagnostics(snapshot")
    assert capture.index("proof.capture_snapshot(") < capture.index("finish_capture(step, snapshot)")
    assert capture.index("snapshot.result()") < capture.index("finish_capture(step, completed)")
    for reader in (
        "proof.physical_snapshot_gate(",
        "proof._snapshot_hash_projection(snapshot)",
        "_snapshot_q2_hash(snapshot)",
        "proof.write_snapshot_netcdf(",
    ):
        assert complete < finish.index(reader), reader
    assert "if step == 0:" in finish[:complete]


def test_the_host_carries_the_start_humidity():
    source = inspect.getsource(forecast.prepare_forecast_host)
    # the call may be wrapped in the preparation timer (_timed_preparation)
    assert ("load_f000_start_humidity(paths[\"init\"])" in source
            or "load_f000_start_humidity, paths[\"init\"])" in source)
    assert '"f000_start_humidity": f000_start_humidity' in source
