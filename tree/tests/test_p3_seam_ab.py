"""P3 seam A/B: the hex plumbing versus the engine's real writer.

THE GATE (2026-08-31, the mp=50 route's own evidence): a P3 column batch
driven through hexcore's seam plumbing -- ``ColumnBatch`` validation, the
mp_p3 scalar-requirement row, ``execute_column_backend`` -- must hand the
engine byte-for-byte the same physics the engine computes when called
directly on the same columns.  Anything less means the hex seam
transformed state on the way through, which is exactly the substitution
class the retired mp=50 refusal existed to stop.

The counterparty is the REAL engine column batch
(``gpuwm.core.mpas_column_batch``, ``microphysics_scheme="p3"``), never a
fixture: the cross-lane law is that a reader is tested against the other
side's real writer.  The engine tree is named by ``HEXCORE_AB_ARWEN_ROOT``
(a gpuwm checkout carrying the P3 transport, i.e. the 2.6.1 line); the
suite skips by name without it, because a silently substituted engine
would grade the wrong counterparty.

Three stages, each byte-compared between the arms:

* phase 1 on the fresh columns (radiation due at step 1),
* phase 2 in place on the eight P3 species plus theta,
* phase 1 again (the held-radiation step).
"""

from __future__ import annotations

import datetime
import os
import sys

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

_ARWEN_ROOT = os.environ.get("HEXCORE_AB_ARWEN_ROOT")
if not _ARWEN_ROOT:
    pytest.skip(
        "HEXCORE_AB_ARWEN_ROOT names no engine checkout; the A/B grades "
        "the real gpuwm column batch (2.6.1 line with the P3 transport) "
        "and refuses to substitute whatever gpuwm sys.path happens to "
        "hold",
        allow_module_level=True,
    )
if _ARWEN_ROOT not in sys.path:
    sys.path.insert(0, _ARWEN_ROOT)

from hexcore import physics_seam as ps  # noqa: E402

NZ, NCOL = 20, 8
DT = 120.0
START = datetime.datetime(2021, 6, 1, 18, 0)
P3_SPECIES = ("qv", "qc", "qr", "qi", "ni", "nr", "qir", "qib")


def _mpas_columns(seed=0):
    """Small P3 columns in hex's own [level, cell] float32 vocabulary."""
    rng = np.random.default_rng(seed)
    z_iface = np.linspace(0.0, 16000.0, NZ + 1)
    z_mid = 0.5 * (z_iface[:-1] + z_iface[1:])
    p_iface = 101325.0 * np.exp(-z_iface / 8000.0)
    p_mid = 101325.0 * np.exp(-z_mid / 8000.0)
    theta = 300.0 + 25.0 * z_mid / 16000.0
    exner = (p_mid / 1.0e5) ** (287.0 / 1004.5)
    temp = theta * exner
    rho_dry = p_mid / (287.0 * temp)
    qv = 0.014 * np.exp(-z_mid / 3000.0)
    qc = np.where(z_mid < 4000.0, 8.0e-4, 0.0)

    def cols(profile, jitter=1.0e-3):
        base = np.repeat(np.asarray(profile)[:, None], NCOL, axis=1)
        base *= 1.0 + jitter * rng.standard_normal(base.shape)
        return np.ascontiguousarray(base, dtype=np.float32)

    fields = {
        "u": cols(np.full(NZ, 5.0)),
        "v": cols(np.full(NZ, -3.0)),
        "theta": cols(theta),
        "pressure": cols(p_mid, jitter=0.0),
        "pressure_interface": cols(p_iface, jitter=0.0),
        "z_interface": np.ascontiguousarray(
            np.repeat(z_iface[:, None], NCOL, axis=1), dtype=np.float32),
        "w": cols(0.05 * np.sin(np.linspace(0, np.pi, NZ + 1)),
                  jitter=0.0),
        "rho_dry": cols(rho_dry, jitter=0.0),
        # ColumnBatch validation enforces the non-negative-qv law on the
        # MPAS side of the seam; the engine's own clamp is gated by its
        # dedicated read-only test.
        "qv": np.abs(cols(qv)),
        "qc": cols(qc),
        "qr": cols(np.full(NZ, 1.0e-5)),
        "qi": np.zeros((NZ, NCOL), dtype=np.float32),
        "ni": np.zeros((NZ, NCOL), dtype=np.float32),
        "qir": np.zeros((NZ, NCOL), dtype=np.float32),
        "qib": np.zeros((NZ, NCOL), dtype=np.float32),
    }
    fields["nr"] = np.ascontiguousarray(
        fields["qr"] * np.float32(1.0e10), dtype=np.float32)
    fields["z_nominal"] = z_iface
    return fields


def _column_batch(fields):
    theta = fields["theta"]
    qv = fields["qv"]
    exner = np.ascontiguousarray(
        (fields["pressure"] / np.float32(ps.REFERENCE_PRESSURE))
        ** np.float32(287.0 / 1004.5), dtype=np.float32)
    return ps.ColumnBatch(
        metric_density=fields["rho_dry"].copy(),
        dry_density=fields["rho_dry"].copy(),
        moist_density=np.ascontiguousarray(
            fields["rho_dry"] * (np.float32(1.0) + qv),
            dtype=np.float32),
        modified_theta=np.ascontiguousarray(
            theta * (np.float32(1.0) + np.float32(ps.RV_OVER_RD) * qv),
            dtype=np.float32),
        theta=theta.copy(),
        temperature=np.ascontiguousarray(theta * exner, dtype=np.float32),
        eos_pressure=fields["pressure"].copy(),
        eos_pressure_interface=fields["pressure_interface"].copy(),
        hydrostatic_moist_pressure=None,
        hydrostatic_moist_pressure_interface=None,
        hydrostatic_dry_pressure=None,
        hydrostatic_dry_pressure_interface=None,
        height=np.ascontiguousarray(
            0.5 * (fields["z_interface"][:-1] + fields["z_interface"][1:]),
            dtype=np.float32),
        height_interface=fields["z_interface"].copy(),
        layer_thickness=np.ascontiguousarray(
            fields["z_interface"][1:] - fields["z_interface"][:-1],
            dtype=np.float32),
        u=fields["u"].copy(),
        v=fields["v"].copy(),
        w=fields["w"].copy(),
        scalars={name: fields[name].copy() for name in P3_SPECIES},
        surface={},
        static={},
    )


def _engine_seam(z_nominal):
    from gpuwm.core import mpas_column_batch as mcb

    return mcb.run_mpas_column_batch(
        n_levels=NZ, n_columns=NCOL, dt=DT,
        microphysics_scheme="p3",
        radiation_seconds=600.0, surface_pbl_seconds=120.0,
        cumulus_seconds=600.0, cumulus_scheme="kf",
        start_time=START,
        latitude_deg=np.full(NCOL, 35.0),
        longitude_deg=np.full(NCOL, -97.0),
        terrain_height_m=np.zeros(NCOL),
        z_interface_nominal_m=z_nominal,
        p_top_pa=float(101325.0 * np.exp(-2.0)), dx_m=15000.0)


_TENDENCY_NAMES = ("du", "dv", "dtheta", "dqv", "dqc", "dqr", "dqi",
                   "dqs", "dqg", "h_diabatic")


def _drive_engine(seam, arrays):
    """The three-stage sequence both arms run, captured as host bytes.

    ``arrays`` maps the phase-1 input names plus the P3 species to host
    float32 arrays; species and theta evolve in the device copies across
    the phase-2 call exactly as the MPAS integrator would hold them.
    """
    dev = {name: cp.asarray(value) for name, value in arrays.items()
           if name != "z_nominal"}
    captured = {}
    result = seam.run_phase1(
        dt=DT, u=dev["u"], v=dev["v"], theta=dev["theta"],
        pressure=dev["pressure"],
        pressure_interface=dev["pressure_interface"],
        z_interface=dev["z_interface"], w=dev["w"],
        rho_dry=dev["rho_dry"],
        **{name: dev[name] for name in P3_SPECIES})
    captured["phase1"] = {
        name: cp.asnumpy(getattr(result, name))
        for name in _TENDENCY_NAMES}
    receipt = seam.run_phase2(
        theta=dev["theta"], pressure=dev["pressure"],
        z_interface=dev["z_interface"],
        **{name: dev[name] for name in P3_SPECIES})
    captured["phase2_state"] = {
        name: cp.asnumpy(dev[name]) for name in ("theta",) + P3_SPECIES}
    captured["phase2_receipt"] = {
        name: cp.asnumpy(value) for name, value in receipt.items()}
    held = seam.run_phase1(
        dt=DT, u=dev["u"], v=dev["v"], theta=dev["theta"],
        pressure=dev["pressure"],
        pressure_interface=dev["pressure_interface"],
        z_interface=dev["z_interface"], w=dev["w"],
        rho_dry=dev["rho_dry"],
        **{name: dev[name] for name in P3_SPECIES})
    captured["phase1_held"] = {
        name: cp.asnumpy(getattr(held, name))
        for name in _TENDENCY_NAMES}
    seam.run_phase2(
        theta=dev["theta"], pressure=dev["pressure"],
        z_interface=dev["z_interface"],
        **{name: dev[name] for name in P3_SPECIES})
    return captured, result


class _EngineP3Backend:
    """The engine column batch behind hexcore's backend protocol."""

    name = "gpuwm-p3-column-batch"

    def __init__(self, z_nominal):
        self._z_nominal = z_nominal
        self.captured = None

    def requirements(self, selection):
        return ps._gpuwm_requirements(selection)

    def run(self, batch, selection, dt):
        assert selection.mp_physics == 50
        assert float(dt) == DT
        arrays = {
            "u": batch.u, "v": batch.v, "theta": batch.theta,
            "pressure": batch.eos_pressure,
            "pressure_interface": batch.eos_pressure_interface,
            "z_interface": batch.height_interface, "w": batch.w,
            "rho_dry": batch.dry_density,
        }
        arrays.update({name: batch.scalars[name] for name in P3_SPECIES})
        seam = _engine_seam(self._z_nominal)
        self.captured, result = _drive_engine(seam, arrays)
        phase1 = self.captured["phase1"]
        return ps.ColumnTendencies(
            du=phase1["du"], dv=phase1["dv"], dtheta=phase1["dtheta"],
            dscalars={name: phase1[f"d{name}"]
                      for name in ("qv", "qc", "qr", "qi")})


@pytest.fixture(scope="module")
def ab_arms():
    fields = _mpas_columns(seed=5)
    selection = ps.resolve_mpas_physics(config_microp_scheme="mp_p3")

    # ARM A: through the hex plumbing -- batch validation, the mp=50
    # requirement row, execute_column_backend's isolate copy, tendency
    # canonicalization.
    backend = _EngineP3Backend(fields["z_nominal"])
    batch = _column_batch(fields)
    tendencies = ps.execute_column_backend(
        batch, selection, backend, dt=DT)

    # ARM B: the same columns handed to the engine directly.
    arrays = {name: fields[name] for name in
              ("u", "v", "theta", "pressure", "pressure_interface",
               "z_interface", "w", "rho_dry") + P3_SPECIES}
    direct, _ = _drive_engine(_engine_seam(fields["z_nominal"]), arrays)
    return backend.captured, direct, tendencies, selection


def test_the_requirement_row_admits_the_batch(ab_arms):
    _, _, tendencies, selection = ab_arms
    assert ps._required_scalar_names(selection) == frozenset(P3_SPECIES)
    assert isinstance(tendencies, ps.ColumnTendencies)


def test_phase1_rates_are_byte_identical(ab_arms):
    hex_arm, direct, _, _ = ab_arms
    for name in _TENDENCY_NAMES:
        assert hex_arm["phase1"][name].tobytes() \
            == direct["phase1"][name].tobytes(), (
                f"phase-1 {name} differs between the hex plumbing and "
                "the direct engine call")
    # No phase-1 scheme forces snow, graupel on a P3 state.
    assert not hex_arm["phase1"]["dqs"].any()
    assert not hex_arm["phase1"]["dqg"].any()


def test_phase2_species_are_byte_identical(ab_arms):
    hex_arm, direct, _, _ = ab_arms
    for name in ("theta",) + P3_SPECIES:
        assert hex_arm["phase2_state"][name].tobytes() \
            == direct["phase2_state"][name].tobytes(), (
                f"post-phase-2 {name} differs between the arms")
    assert set(hex_arm["phase2_receipt"]) == {"rainncv", "snowncv", "sr"}
    for name, value in hex_arm["phase2_receipt"].items():
        assert value.tobytes() \
            == direct["phase2_receipt"][name].tobytes(), name


def test_the_held_radiation_step_is_byte_identical(ab_arms):
    hex_arm, direct, _, _ = ab_arms
    for name in _TENDENCY_NAMES:
        assert hex_arm["phase1_held"][name].tobytes() \
            == direct["phase1_held"][name].tobytes(), (
                f"held-step {name} differs between the arms")


def test_the_hex_boundary_hands_back_the_same_bytes(ab_arms):
    """What crosses the hex ColumnTendencies boundary is the engine's
    own answer, unrescaled and uncopied-with-cast."""
    hex_arm, _, tendencies, _ = ab_arms
    assert tendencies.du.tobytes() == hex_arm["phase1"]["du"].tobytes()
    assert tendencies.dv.tobytes() == hex_arm["phase1"]["dv"].tobytes()
    assert tendencies.dtheta.tobytes() \
        == hex_arm["phase1"]["dtheta"].tobytes()
    for name in ("qv", "qc", "qr", "qi"):
        assert tendencies.dscalars[name].tobytes() \
            == hex_arm["phase1"][f"d{name}"].tobytes()
