"""The New Tiedtke row (RRFS v2 at 12 km): admission, pins and the seam.

CPU tests check the row's admission surface without a card: the sealed
constructor admits ``cumulus_scheme='ntiedtke'`` only at the every-step
cadence, the engine source pin refuses a missing or moved file by name, and
every default path is untouched.  The GPU test builds the row's seam on a
small column set and checks that New Tiedtke runs every step and that its raw
momentum reaches ``du``/``dv``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from hexcore import convection_admission
from hexcore import cuda_ntiedtke_column_v841 as row
from hexcore.cuda_arwen_physics_v841 import SealedArwenConstructorV841


def _mapping(*, scheme, cumulus_seconds, ncol=3, nlev=4, dt=60.0):
    f32 = np.float32
    return {
        "n_levels": nlev,
        "n_columns": ncol,
        "dt": dt,
        "radiation_seconds": 1800.0,
        "surface_pbl_seconds": dt,
        "cumulus_seconds": cumulus_seconds,
        "cumulus_scheme": scheme,
        "microphysics_scheme": "wsm6",
        "start_time": datetime(2026, 10, 3, 12, tzinfo=timezone.utc),
        "p_top_pa": 2000.0,
        "dx_m": 12000.0,
        "gf_ishallow": 1 if scheme == "gf" else 0,
        "wsm6_hail_opt": 0,
        "xice_threshold": 0.5,
        "latitude_deg": np.full(ncol, 40.0, f32),
        "longitude_deg": np.full(ncol, -100.0, f32),
        "terrain_height_m": np.zeros(ncol, f32),
        "z_interface_nominal_m": np.linspace(0.0, 20000.0, nlev + 1),
        "landmask": np.ones(ncol, f32),
        "xland": np.ones(ncol, f32),
        "ivgtyp": np.full(ncol, 10, np.int32),
        "isltyp": np.full(ncol, 6, np.int32),
        "vegfra": np.full(ncol, 50.0, f32),
        "tsk": np.full(ncol, 300.0, f32),
        "tmn": np.full(ncol, 285.0, f32),
        "xice": np.zeros(ncol, f32),
        "snow": np.zeros(ncol, f32),
        "snow_depth": np.zeros(ncol, f32),
        "soil_temperature": np.full((4, ncol), 285.0, f32),
        "soil_moisture": np.full((4, ncol), 0.3, f32),
        "dx_column_m": np.full(ncol, 12000.0, f32),
    }


def test_the_sealed_constructor_admits_new_tiedtke_every_step():
    sealed = SealedArwenConstructorV841.from_mapping(
        _mapping(scheme="ntiedtke", cumulus_seconds=60.0)
    )
    assert sealed.cumulus_scheme == "ntiedtke"
    assert sealed.arwen_kwargs()["cumulus_scheme"] == "ntiedtke"


def test_new_tiedtke_on_a_held_cadence_is_refused_by_name():
    with pytest.raises(ValueError, match="cumulus_scheme='ntiedtke' requires"):
        SealedArwenConstructorV841.from_mapping(
            _mapping(scheme="ntiedtke", cumulus_seconds=120.0)
        )


def test_gf_shallow_with_new_tiedtke_is_refused():
    values = _mapping(scheme="ntiedtke", cumulus_seconds=60.0)
    values["gf_ishallow"] = 1
    with pytest.raises(ValueError, match="gf_ishallow=1 requires"):
        SealedArwenConstructorV841.from_mapping(values)


def test_the_grell_freitas_constructor_is_unchanged():
    sealed = SealedArwenConstructorV841.from_mapping(
        _mapping(scheme="gf", cumulus_seconds=60.0)
    )
    assert sealed.cumulus_scheme == "gf"
    with pytest.raises(ValueError, match="must be 'gf', 'kf', 'ntiedtke', or None"):
        SealedArwenConstructorV841.from_mapping(
            _mapping(scheme="tiedtke", cumulus_seconds=60.0)
        )


def _fake_engine(tmp_path: Path, *, moved: str | None = None, absent: str | None = None):
    for relative, digest in row.NTIEDTKE_ENGINE_SOURCES.items():
        if relative == absent:
            continue
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not the pinned bytes")
    return tmp_path


def test_moved_engine_sources_are_recorded_not_refused(tmp_path):
    """Retired gate (2026-10-10): other engine bytes run and are recorded."""

    moved = tmp_path / "moved"
    found = row.verify_ntiedtke_sources(_fake_engine(moved))
    wanted = sha256(b"not the pinned bytes").hexdigest()
    assert dict(found) == {relative: wanted for relative in row.NTIEDTKE_ENGINE_SOURCES}
    record = row.measured_row_match(found)
    assert record["matched"] is False
    assert all(not item["matched"] for item in record["files"].values())
    assert row.measured_row_match(dict(row.NTIEDTKE_ENGINE_SOURCES))["matched"] is True


def test_a_missing_engine_source_is_still_refused(tmp_path):
    with pytest.raises(row.NtiedtkeRowError, match="does not carry it"):
        row.verify_ntiedtke_sources(
            _fake_engine(tmp_path / "absent", absent="gpuwm/core/ntiedtke.py")
        )


def test_the_importable_engine_runs_and_records_its_bytes():
    """Any importable engine's New Tiedtke bytes are hashed, not refused."""

    gpuwm = pytest.importorskip("gpuwm")
    root = Path(gpuwm.__file__).resolve().parents[1]
    files = [root / relative for relative in row.NTIEDTKE_ENGINE_SOURCES]
    if not all(path.is_file() for path in files):
        pytest.skip("the importable engine carries no New Tiedtke sources")
    found = {
        relative: sha256((root / relative).read_bytes()).hexdigest()
        for relative in row.NTIEDTKE_ENGINE_SOURCES
    }
    assert dict(row.verify_ntiedtke_sources(root)) == found
    assert row.measured_row_match(found)["matched"] is (
        found == dict(row.NTIEDTKE_ENGINE_SOURCES)
    )


def test_the_row_is_a_subclass_of_the_frozen_column_batch():
    pytest.importorskip("gpuwm.core.mpas_column_batch")
    from gpuwm.core.mpas_column_batch import MpasColumnBatchPhysics

    cls = row.seam_class()
    assert cls.__bases__ == (MpasColumnBatchPhysics,)
    assert cls.__module__ == row.__name__
    # Phase one and phase two are still the frozen class's orchestration.
    assert cls._PHASE1_ORCHESTRATION is MpasColumnBatchPhysics._PHASE1_ORCHESTRATION
    assert cls._PHASE2_MICROPHYSICS is MpasColumnBatchPhysics._PHASE2_MICROPHYSICS


def test_the_seam_walks_new_tiedtke_at_the_narrow_tile():
    # Hex memory census item 3: the workspace is held at the chunk width, so
    # the seam's instance caps it at 128 columns per multiprocessor, and an
    # engine with the per-instance cap reads it (2 chunks of 20,481 columns
    # at x1.40962 on a 170-SM card instead of one of 40,962).
    ntiedtke = pytest.importorskip("gpuwm.core.ntiedtke")
    instance = row._capturing_new_tiedtke()
    assert instance.tile_columns_per_sm == row.HEX_NT_TILE_COLUMNS_PER_SM == 128
    try:
        width = ntiedtke.nt_tile_columns(40_962, 170, instance.tile_columns_per_sm)
    except TypeError:
        pytest.skip("this engine predates the per-instance New Tiedtke tile cap")
    assert width == 20_481
    assert ntiedtke.nt_tile_columns(40_962, 170) == 40_962


def test_the_row_is_the_scheme_the_rrfs_profile_names():
    assert convection_admission.constructor_scheme(
        convection_admission.SCHEME_NTIEDTKE
    ) == "ntiedtke"
    assert row.CU_PHYSICS_NTIEDTKE == 16


# ---------------------------------------------------------------------------
# on a card
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_new_tiedtke_runs_every_step_and_its_momentum_reaches_du_dv():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device")
    nlev, ncol = 20, 64
    values = _mapping(scheme="ntiedtke", cumulus_seconds=60.0, ncol=ncol, nlev=nlev)
    seam = row.run_ntiedtke_column_batch(**values)
    assert seam._cfg.cu_physics == 16
    assert seam._identity["cumulus_scheme"] == "ntiedtke"
    rng = np.random.default_rng(7)
    z = np.linspace(0.0, 20000.0, nlev + 1).astype(np.float32)
    zc = 0.5 * (z[1:] + z[:-1])
    p = (100000.0 * np.exp(-zc / 8000.0)).astype(np.float32)
    pi = (100000.0 * np.exp(-z / 8000.0)).astype(np.float32)
    theta = (300.0 + 4.0e-3 * zc).astype(np.float32)
    tk = theta * (p / 1.0e5) ** 0.2857
    qv = (0.016 * np.exp(-zc / 2500.0)).astype(np.float32)
    rho = (p / (287.0 * tk)).astype(np.float32)

    def col(profile, jitter=0.0):
        base = np.repeat(profile[:, None], ncol, axis=1)
        if jitter:
            base = base * (1.0 + jitter * rng.standard_normal(base.shape))
        return cp.asarray(np.ascontiguousarray(base.astype(np.float32)))

    u = col(np.linspace(2.0, 25.0, nlev).astype(np.float32), 0.05)
    v = col(np.linspace(-1.0, 8.0, nlev).astype(np.float32), 0.05)
    zero = cp.zeros((nlev, ncol), dtype=cp.float32)
    inputs = dict(
        dt=60.0, u=u, v=v, theta=col(theta), pressure=col(p),
        pressure_interface=col(pi), z_interface=col(z),
        w=cp.zeros((nlev + 1, ncol), dtype=cp.float32), rho_dry=col(rho),
        qv=col(qv, 0.02), qc=zero.copy(), qr=zero.copy(), qi=zero.copy(),
        qs=zero.copy(), qg=zero.copy(),
        rthdynten=cp.full((nlev, ncol), 2.0e-5, dtype=cp.float32),
        rqvdynten=cp.full((nlev, ncol), 2.0e-8, dtype=cp.float32),
    )
    for _ in range(3):
        out = seam.run_phase1(**inputs)
        assert out.cumulus_ran
        assert seam._ntiedtke.last_momentum is not None
        np.testing.assert_array_equal(
            cp.asnumpy(out.du), cp.asnumpy(seam._out["du"] + seam._cu_du)
        )
        np.testing.assert_array_equal(
            cp.asnumpy(out.dv), cp.asnumpy(seam._out["dv"] + seam._cu_dv)
        )
        for name in ("du", "dv", "dtheta", "dqv"):
            assert np.all(np.isfinite(cp.asnumpy(getattr(out, name))))
        seam.run_phase2(
            theta=inputs["theta"], qv=inputs["qv"], qc=inputs["qc"],
            qr=inputs["qr"], qi=inputs["qi"], qs=inputs["qs"], qg=inputs["qg"],
            pressure=inputs["pressure"], rho_dry=inputs["rho_dry"],
            z_interface=inputs["z_interface"],
        )
    assert seam._ntiedtke.calls == 3


def test_concurrent_ranks_share_one_seam_class(monkeypatch):
    """8 ranks constructed on a thread pool must see ONE seam class.

    The race this pins (2026-10-10, 8-card RTX PRO 6000 host, 30 km New Tiedtke on 8 same-host
    ranks): two builders, the later one replacing the global, and the
    private-radius identity check refusing the rank whose class lost.
    """

    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    built = []

    def slow_builder():
        time.sleep(0.05)
        cls = type("Seam%d" % len(built), (), {})
        built.append(cls)
        return cls

    monkeypatch.setattr(row, "_SEAM_CLASS", None)
    monkeypatch.setattr(row, "_make_seam_class", slow_builder)
    barrier = threading.Barrier(8)

    def rank(_):
        barrier.wait()
        return row.seam_class()

    with ThreadPoolExecutor(8) as pool:
        classes = list(pool.map(rank, range(8)))
    assert len(built) == 1
    assert all(cls is built[0] for cls in classes)
    assert row.seam_class() is built[0]


# ---------------------------------------------------------------------------
# scale awareness: one spacing per cell, as MPAS-A v8.4.1 hands cu_ntiedtke
# (mpas_atmphys_driver_convection.F:840, cu_ntiedtke.F90:228-239)
# ---------------------------------------------------------------------------
def test_scale_aware_is_on_unless_the_switch_says_off():
    assert row.scale_aware_requested({}) is True
    assert row.scale_aware_requested({row.SCALE_AWARE_ENV: "on"}) is True
    assert row.scale_aware_requested({row.SCALE_AWARE_ENV: "1"}) is True
    for word in ("off", "OFF", "0", "false", " no "):
        assert row.scale_aware_requested({row.SCALE_AWARE_ENV: word}) is False


def test_the_scalar_dx_difference_is_no_longer_declared():
    assert not any("scalar dx_m" in item for item in row.DECLARED_DIFFERENCES)
    assert "len_disp / meshDensity**0.25" in row.__doc__


def test_the_receipt_states_each_cells_factor_from_the_source_formula():
    dx = np.array([60000.0, 30000.0, 15000.0, 9000.0, 3000.0], np.float32)
    record = row.scale_awareness_receipt(dx, scale_aware=True, dxref_m=15000.0)
    assert record["scale_aware"] is True
    assert record["cells"] == 5 and record["cells_finer_than_dxref"] == 2
    assert record["mesh_dx_m"] == {"min": 3000.0, "max": 60000.0}
    assert record["scheme_dx_m"] == record["mesh_dx_m"]
    # cu_ntiedtke.F90:233 at 3 km and :236 at 15 km (the smallest coarse value).
    assert record["deep_adjustment_time_factor"]["max"] == pytest.approx(
        (1.06133 + np.log(5.0)) ** 3, rel=1e-5
    )
    assert record["deep_adjustment_time_factor"]["min"] == pytest.approx(
        1.0 + 1.33e-5 * 15000.0, rel=1e-6
    )
    assert record["shallow_mass_flux_divisor"]["max"] == pytest.approx(
        ((1.06133 + np.log(5.0)) ** 3) ** 0.5, rel=1e-5
    )
    assert record["shallow_mass_flux_divisor"]["min"] == 1.0

    off = row.scale_awareness_receipt(dx, scale_aware=False, dxref_m=15000.0)
    assert off["scale_aware"] is False
    assert off["mesh_dx_m"] == {"min": 3000.0, "max": 60000.0}
    assert off["scheme_dx_m"] == {"min": 15000.0, "max": 60000.0}
    assert off["shallow_mass_flux_divisor"] == {"min": 1.0, "max": 1.0}
    assert off["deep_adjustment_time_factor"]["max"] == pytest.approx(
        1.0 + 1.33e-5 * 60000.0, rel=1e-6
    )


def test_a_uniform_mesh_at_or_above_dxref_is_the_same_on_and_off():
    """15 km and coarser take cu_ntiedtke.F90:236 either way (`<`, not `<=`)."""

    for spacing in (15000.0, 30000.0, 60000.0):
        dx = np.full(7, spacing, np.float32)
        on = row.scale_awareness_receipt(dx, scale_aware=True, dxref_m=15000.0)
        off = row.scale_awareness_receipt(dx, scale_aware=False, dxref_m=15000.0)
        assert on["cells_finer_than_dxref"] == 0
        for key in ("scheme_dx_m", "deep_adjustment_time_factor",
                    "shallow_mass_flux_divisor"):
            assert on[key] == off[key]


def _variable_mesh_values(ncol=64, nlev=20):
    values = _mapping(scheme="ntiedtke", cumulus_seconds=60.0, ncol=ncol, nlev=nlev)
    dx = np.repeat(np.array([60000.0, 3000.0], np.float32), ncol // 2)
    values["dx_m"] = 3000.0
    values["dx_column_m"] = dx
    return values, dx


def _phase1_inputs(cp, ncol, nlev):
    z = np.linspace(0.0, 20000.0, nlev + 1).astype(np.float32)
    zc = 0.5 * (z[1:] + z[:-1])
    p = (100000.0 * np.exp(-zc / 8000.0)).astype(np.float32)
    pi = (100000.0 * np.exp(-z / 8000.0)).astype(np.float32)
    theta = (300.0 + 4.0e-3 * zc).astype(np.float32)
    tk = theta * (p / 1.0e5) ** 0.2857
    qv = (0.016 * np.exp(-zc / 2500.0)).astype(np.float32)
    rho = (p / (287.0 * tk)).astype(np.float32)

    def col(profile):
        base = np.repeat(profile[:, None], ncol, axis=1)
        return cp.asarray(np.ascontiguousarray(base.astype(np.float32)))

    zero = cp.zeros((nlev, ncol), dtype=cp.float32)
    return dict(
        dt=60.0, u=col(np.linspace(2.0, 25.0, nlev).astype(np.float32)),
        v=col(np.linspace(-1.0, 8.0, nlev).astype(np.float32)),
        theta=col(theta), pressure=col(p), pressure_interface=col(pi),
        z_interface=col(z), w=cp.zeros((nlev + 1, ncol), dtype=cp.float32),
        rho_dry=col(rho), qv=col(qv), qc=zero.copy(), qr=zero.copy(),
        qi=zero.copy(), qs=zero.copy(), qg=zero.copy(),
        rthdynten=cp.full((nlev, ncol), 2.0e-5, dtype=cp.float32),
        rqvdynten=cp.full((nlev, ncol), 2.0e-8, dtype=cp.float32),
    )


def _kernel_words(cp, seam, name, ncol):
    workspace = seam._ntiedtke._pipeline[1].w
    return cp.asnumpy(workspace.bind(name, 1))[:ncol].copy()


@pytest.mark.gpu
def test_each_cells_own_spacing_reaches_the_kernels_by_default(monkeypatch):
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device")
    if not row.engine_takes_column_dx():
        pytest.skip("the importable engine's New Tiedtke has no per-column dx")
    monkeypatch.delenv(row.SCALE_AWARE_ENV, raising=False)
    ncol, nlev = 64, 20
    values, dx = _variable_mesh_values(ncol, nlev)
    seam = row.run_ntiedtke_column_batch(**values)
    record = seam.ntiedtke_receipt["scale_awareness"]
    assert record["scale_aware"] is True
    assert record["engine_reads_per_cell_dx"] is True
    assert record["cells_finer_than_dxref"] == ncol // 2
    assert record["mesh_dx_m"] == {"min": 3000.0, "max": 60000.0}
    assert seam._ntiedtke.scale_aware is True

    seam.run_phase1(**_phase1_inputs(cp, ncol, nlev))
    np.testing.assert_array_equal(_kernel_words(cp, seam, "dx", ncol), dx)
    factor = _kernel_words(cp, seam, "scale_fac", ncol)
    divisor = _kernel_words(cp, seam, "scale_fac2", ncol)
    half = ncol // 2
    np.testing.assert_allclose(factor[:half], 1.0 + 1.33e-5 * 60000.0, rtol=2e-6)
    np.testing.assert_allclose(
        factor[half:], (1.06133 + np.log(5.0)) ** 3, rtol=2e-5
    )
    np.testing.assert_array_equal(divisor[:half], 1.0)
    np.testing.assert_allclose(divisor[half:], np.sqrt(factor[half:]), rtol=2e-6)


@pytest.mark.gpu
def test_the_off_switch_holds_fine_cells_at_dxref(monkeypatch):
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device")
    if not row.engine_takes_column_dx():
        pytest.skip("the importable engine's New Tiedtke has no per-column dx")
    monkeypatch.setenv(row.SCALE_AWARE_ENV, "off")
    ncol, nlev = 64, 20
    values, _ = _variable_mesh_values(ncol, nlev)
    seam = row.run_ntiedtke_column_batch(**values)
    record = seam.ntiedtke_receipt["scale_awareness"]
    assert record["scale_aware"] is False
    assert record["scheme_dx_m"] == {"min": 15000.0, "max": 60000.0}
    seam.run_phase1(**_phase1_inputs(cp, ncol, nlev))
    seen = _kernel_words(cp, seam, "dx", ncol)
    half = ncol // 2
    np.testing.assert_array_equal(seen[:half], 60000.0)
    np.testing.assert_array_equal(seen[half:], 15000.0)
    np.testing.assert_array_equal(_kernel_words(cp, seam, "scale_fac2", ncol), 1.0)


@pytest.mark.gpu
def test_an_engine_without_per_cell_dx_is_refused_on_a_variable_mesh(monkeypatch):
    """The breakage: fine cells given the coarse cells' adjustment time."""

    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device")
    monkeypatch.delenv(row.SCALE_AWARE_ENV, raising=False)
    monkeypatch.setattr(row, "engine_takes_column_dx", lambda: False)
    values, _ = _variable_mesh_values()
    with pytest.raises(row.NtiedtkeRowError, match="one nominal spacing"):
        row.run_ntiedtke_column_batch(**values)
    # A uniform mesh loses nothing to such an engine and still runs.
    uniform = _mapping(scheme="ntiedtke", cumulus_seconds=60.0, ncol=64, nlev=20)
    seam = row.run_ntiedtke_column_batch(**uniform)
    assert seam.ntiedtke_receipt["scale_awareness"]["engine_reads_per_cell_dx"] is False
    monkeypatch.setenv(row.SCALE_AWARE_ENV, "off")
    with pytest.raises(row.NtiedtkeRowError, match="carries the switch"):
        row.run_ntiedtke_column_batch(**uniform)
