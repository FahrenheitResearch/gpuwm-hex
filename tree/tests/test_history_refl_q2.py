"""The history stream carries ``refl10cm`` and ``q2`` -- divergence 3's referee.

THE BREAKAGE THIS PREVENTS: the first obs-referee run (2026-08-25, the proving node)
found four registered metrics unscorable because the model side was silent --
all three MRMS reflectivity metrics returned ``model bundle ... lacks field
'reflectivity_dbz'`` and ``asos-dewpoint-rmse`` failed the same way on ``q2``.
A history stream that drops either field makes the declared referee for the
condensate-surplus divergence unrunnable, silently.  These tests go red the
moment either field leaves the default history stream again.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "tools" / "run_cuda_v841_full_physics_x4.py"
PRODUCER_PATH = ROOT / "verification" / "producers" / "model_bundle.py"


def _load(name: str, path: Path) -> object:
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_runner() -> object:
    return _load("_test_refl_q2_runner", RUNNER_PATH)


def _load_producer() -> object:
    return _load("_test_refl_q2_producer", PRODUCER_PATH)


def _minimal_snapshot(runner) -> tuple[dict, dict]:
    n_cells = runner.N_CELLS
    n_edges = runner.N_EDGES
    n_levels = runner.N_LEVELS
    rng = np.random.default_rng(20260825)
    q2 = rng.uniform(0.0, 0.02, size=n_cells).astype(np.float32)
    # Native q2 carries a handful of small negative values; publication must
    # preserve them bitwise rather than clamp or hide them.
    q2[7] = np.float32(-1.25e-6)
    refl = rng.uniform(-35.0, 20.0, size=(n_levels, n_cells)).astype(np.float32)
    snapshot = {
        "arrays": {
            "q2": q2,
            "refl10cm": refl,
            "t2": rng.uniform(250.0, 310.0, size=n_cells).astype(np.float32),
        },
        "receipt": {"schema": runner.SNAPSHOT_SCHEMA, "label": "TEST", "step": 1},
    }
    static = {
        "indexToCellID": np.arange(1, n_cells + 1, dtype=np.int32),
        "latCell": np.zeros(n_cells, dtype=np.float32),
        "lonCell": np.zeros(n_cells, dtype=np.float32),
        "ter": np.zeros(n_cells, dtype=np.float32),
        "indexToEdgeID": np.arange(1, n_edges + 1, dtype=np.int32),
        "latEdge": np.zeros(n_edges, dtype=np.float32),
        "lonEdge": np.zeros(n_edges, dtype=np.float32),
    }
    return snapshot, static


def test_history_writer_publishes_q2_bitwise_with_dims_and_units(tmp_path) -> None:
    from netCDF4 import Dataset

    runner = _load_runner()
    snapshot, static = _minimal_snapshot(runner)
    path = tmp_path / "cuda-history.2026-08-12_07.00.00.nc"
    runner.write_snapshot_netcdf(path, snapshot, static)
    with Dataset(path, "r") as dataset:
        assert "q2" in dataset.variables, (
            "the history stream must publish q2; without it asos-dewpoint-rmse "
            "is unscorable"
        )
        variable = dataset.variables["q2"]
        assert variable.dimensions == ("Time", "nCells")
        assert variable.getncattr("units") == "kg kg^{-1}"
        stored = np.asarray(variable[0])
        np.testing.assert_array_equal(stored, snapshot["arrays"]["q2"])
        assert stored[7] < 0.0, "negative q2 values are preserved, not clamped"
        assert dataset.getncattr("q2_products_allowed") == "true"


def test_history_writer_publishes_refl10cm_with_dims_units_finite(tmp_path) -> None:
    from netCDF4 import Dataset

    runner = _load_runner()
    snapshot, static = _minimal_snapshot(runner)
    path = tmp_path / "cuda-history.2026-08-12_08.00.00.nc"
    runner.write_snapshot_netcdf(path, snapshot, static)
    with Dataset(path, "r") as dataset:
        assert "refl10cm" in dataset.variables, (
            "the history stream must publish refl10cm; without it all three "
            "MRMS reflectivity metrics are unscorable"
        )
        variable = dataset.variables["refl10cm"]
        assert variable.dimensions == ("Time", "nCells", "nVertLevels")
        assert variable.getncattr("units") == "dBZ"
        stored = np.asarray(variable[0])
        assert np.all(np.isfinite(stored))
        np.testing.assert_array_equal(stored, snapshot["arrays"]["refl10cm"].T)


def test_refl_due_reaches_the_phase_two_seam_by_signature() -> None:
    runner = _load_runner()
    parameters = inspect.signature(runner.execute_composite_step).parameters
    assert "refl_10cm_due" in parameters, (
        "execute_composite_step must carry the history-step diagflag to the "
        "microphysics seam or refl10cm is computed at the wrong time"
    )
    assert parameters["refl_10cm_due"].default is False

    sys.path.insert(0, str(ROOT / "src"))
    from hexcore.cuda_arwen_physics_v841 import (
        PersistentTwoPhaseCudaPhysicsBackendV841,
    )

    finish = inspect.signature(
        PersistentTwoPhaseCudaPhysicsBackendV841.finish_step
    ).parameters
    assert "refl_10cm_due" in finish
    assert finish["refl_10cm_due"].default is False
    assert hasattr(
        PersistentTwoPhaseCudaPhysicsBackendV841, "take_history_refl10cm"
    ), "capture has no committed-boundary accessor for the stashed field"

    capture = inspect.signature(runner.capture_snapshot).parameters
    assert "expect_refl10cm" in capture
    assert capture["expect_refl10cm"].default is False


def test_snapshot_schema_bumped_and_projection_still_excludes_q2() -> None:
    runner = _load_runner()
    assert runner.SNAPSHOT_SCHEMA.endswith("/v3"), (
        "publishing q2 and refl10cm changes the capsule content; the schema "
        "string must say so"
    )
    direct = {
        "arrays": {
            "qv": np.array([1.0], dtype=np.float32),
            "q2": np.array([-1.0], dtype=np.float32),
            "refl10cm": np.array([[3.0]], dtype=np.float32),
        }
    }
    projection = runner._snapshot_hash_projection(direct)
    assert "q2" not in projection
    assert "refl10cm" in projection, (
        "refl10cm is a published forecast field; the restart projection must "
        "cover it"
    )


def test_producer_requires_the_two_new_fields_by_name() -> None:
    producer = _load_producer()
    assert "Q2" in producer.REQUIRED
    assert "REFL_10CM" in producer.REQUIRED
    assert "reflectivity_dbz" not in getattr(producer, "FIELDS_ABSENT", {})
    assert "dewpoint_k" not in getattr(producer, "FIELDS_ABSENT", {})


def test_producer_composite_reflectivity_is_the_column_maximum() -> None:
    producer = _load_producer()
    column = np.array(
        [
            [[-35.0, 10.0], [5.0, -35.0]],
            [[12.5, -2.0], [4.0, 41.0]],
            [[0.0, 3.0], [-1.0, 40.0]],
        ],
        dtype=np.float64,
    )
    composite = producer.composite_reflectivity_dbz(column)
    np.testing.assert_allclose(
        composite, np.array([[12.5, 10.0], [5.0, 41.0]])
    )


def test_producer_dewpoint_matches_the_engine_formula() -> None:
    producer = _load_producer()
    # The engine's own surface check: 14 g/kg at 1000 hPa dews between 18
    # and 21 C (rustwx-calc/src/derived.rs, dewpoint_from_mixing_ratio test).
    dewpoint = producer.dewpoint_k(
        np.array([0.014]), np.array([100000.0])
    )
    assert 18.0 + 273.15 < float(dewpoint[0]) < 21.0 + 273.15

    # Exact transcription of dewpoint_from_mixing_ratio (derived.rs:653-658):
    q = 0.014
    e_hpa = max(q * 1000.0 / (0.622 + q), 1.0e-10)
    ln_e = np.log(e_hpa / 6.112)
    expected_c = (243.5 * ln_e) / (17.67 - ln_e)
    np.testing.assert_allclose(float(dewpoint[0]), expected_c + 273.15, rtol=1e-12)

    # A negative q2 (native carries a few) clamps to the vapour-pressure floor
    # instead of producing NaN.
    cold = producer.dewpoint_k(np.array([-1.0e-6]), np.array([100000.0]))
    assert np.isfinite(cold[0])
    assert cold[0] < 200.0
