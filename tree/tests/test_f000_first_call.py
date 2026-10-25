"""The f000 frame's surface-layer diagnostics come from the first physics call (CPU tier).

Breakage (hexcore.f000_first_call): a start file with no 10 m wind published f000 with
u10/v10 = -1e30 in every cell, so every f000 sounding showed "0-6 km shear NaN kt".
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from hexcore import f000_first_call as first

MISSING = np.float32(-1.0e30)


def _frame(n=6):
    return {
        "u10": np.full(n, MISSING, np.float32),
        "v10": np.full(n, MISSING, np.float32),
        "t2": np.linspace(280, 285, n).astype(np.float32),
        "q2": np.full(n, 0.004, np.float32),
        "hfx": np.zeros(n, np.float32),
        "qfx": np.zeros(n, np.float32),
        "lh": np.zeros(n, np.float32),
        "pblh": np.zeros(n, np.float32),
    }


def _call(n=6):
    return {
        "u10": np.arange(n, dtype=np.float32) - 2.0,
        "v10": np.arange(n, dtype=np.float32) + 1.0,
        "t2": np.full(n, 290.0, np.float32),
        "q2": np.full(n, 0.007, np.float32),
        "hfx": np.full(n, 12.0, np.float32),
        "qfx": np.full(n, 1e-5, np.float32),
        "lh": np.full(n, 30.0, np.float32),
        "pblh": np.full(n, 150.0, np.float32),
    }


def test_missing_wind_and_placeholders_come_from_the_first_call():
    frame, call = _frame(), _call()
    record = first.complete_arwen_start_frame(frame, call, {"q2": {"source": "start file q2"}})
    assert record["applied"]
    np.testing.assert_array_equal(frame["u10"], call["u10"])
    np.testing.assert_array_equal(frame["v10"], call["v10"])
    for name in ("hfx", "qfx", "lh", "pblh"):
        np.testing.assert_array_equal(frame[name], call[name])
    # the start file's own 2 m temperature and humidity are kept
    np.testing.assert_array_equal(frame["t2"], _frame()["t2"])
    np.testing.assert_array_equal(frame["q2"], _frame()["q2"])
    assert record["fields"]["u10"]["cells_without_value_after"] == 0
    assert set(record["unchanged"]) == {"t2", "q2"}


def test_a_carried_analysis_wind_is_kept_cell_by_cell():
    frame, call = _frame(), _call()
    frame["u10"][:3] = 5.0
    frame["v10"][:] = 1.5
    first.complete_arwen_start_frame(frame, call, {"q2": {"source": "start file q2"}})
    np.testing.assert_array_equal(frame["u10"][:3], 5.0)
    np.testing.assert_array_equal(frame["u10"][3:], call["u10"][3:])
    np.testing.assert_array_equal(frame["v10"], 1.5)


def test_uncarried_humidity_takes_the_first_call_not_the_lowest_level_fallback():
    frame, call = _frame(), _call()
    record = first.complete_arwen_start_frame(
        frame, call, {"q2": {"source": "lowest model level qv, bounded to [0, saturation]"}})
    np.testing.assert_array_equal(frame["q2"], call["q2"])
    assert "q2" in record["fields"]


def test_no_first_call_leaves_the_frame_as_captured():
    frame = _frame()
    record = first.complete_arwen_start_frame(frame, None, None)
    assert record["applied"] is False
    assert np.all(frame["u10"] == MISSING)


def test_a_computed_flux_is_never_overwritten():
    frame, call = _frame(), _call()
    frame["hfx"][:] = 3.0
    first.complete_arwen_start_frame(frame, call, None)
    np.testing.assert_array_equal(frame["hfx"], 3.0)


def test_mpas_names_and_all_zero_humidity():
    frame = {"u10": np.full(4, MISSING, np.float32), "t2m": np.full(4, 281.0, np.float32),
             "q2": np.zeros(4, np.float32), "hpbl": np.zeros(4, np.float32),
             "ust": np.zeros(4, np.float32)}
    call = {"u10": np.ones(4, np.float32), "t2m": np.full(4, 283.0, np.float32),
            "q2": np.full(4, 0.006, np.float32), "hpbl": np.full(4, 400.0, np.float32),
            "ust": np.full(4, 0.3, np.float32)}
    first.complete_start_frame(frame, call, analysis=first.MPAS_ANALYSIS,
                               placeholders=first.MPAS_PLACEHOLDERS, humidity=first.MPAS_HUMIDITY)
    np.testing.assert_array_equal(frame["u10"], 1.0)
    np.testing.assert_array_equal(frame["t2m"], 281.0)
    np.testing.assert_array_equal(frame["q2"], np.float32(0.006))
    np.testing.assert_array_equal(frame["hpbl"], 400.0)
    np.testing.assert_array_equal(frame["ust"], np.float32(0.3))


def test_shape_mismatch_refuses():
    frame, call = _frame(), _call(5)
    with pytest.raises(ValueError, match="u10"):
        first.complete_arwen_start_frame(frame, call, None)


def test_handoff_releases_a_waiting_writer():
    handoff = first.FirstCallHandoff()
    seen = []
    worker = threading.Thread(target=lambda: seen.append(handoff.take()))
    worker.start()
    handoff.put({"u10": 1})
    handoff.put(None)  # a second put is ignored
    worker.join(5)
    assert seen == [{"u10": 1}]


def test_driver_completion_updates_the_receipt_rows():
    pytest.importorskip("netCDF4")
    from hexcore.drivers import run_cuda_v841_forecast as forecast

    arrays = _frame()
    snapshot = {"arrays": arrays, "receipt": {
        "step": 0, "arrays": {name: {} for name in arrays},
        "f000_completed_surface_diagnostics": {"q2": {"source": "start file q2"}}}}
    record = forecast.complete_f000_from_first_call(snapshot, _call())
    assert record["applied"]
    assert snapshot["receipt"]["arrays"]["u10"]["minimum"] == pytest.approx(-2.0)
    with pytest.raises(ValueError):
        forecast.complete_f000_from_first_call({"arrays": {}, "receipt": {"step": 3}}, _call())
