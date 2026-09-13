#!/usr/bin/env python3
"""Drive the engine's column-batch seam on fixed columns and compare two engines.

WHY THIS EXISTS.  This port owns no physics: every physics column runs
through ``gpuwm.core.mpas_column_batch.run_mpas_column_batch``, and the port
pins that seam by the bytes of sixteen engine files.  When the pin moves to
a new engine, the question a reader has is not "did the sixteen digests
change" (the pin instrument answers that) but "did a NUMBER move through the
seam, which one, and by how much".  Until this tool the only answer was the
x4 full-physics proof: one mesh, one case, one hour, on a 32 GiB card that
also holds 6.9 GiB of pinned assets.  A re-pin taken on a box without those
cannot re-run that proof and must not claim byte-neutrality it did not
measure.

So this is the seam-level instrument.  ``--run`` builds a fixed set of
convectively unstable columns from integer arithmetic (the same bytes on
every numpy), constructs the engine seam exactly as the port's adapter does
-- WSM6 microphysics, YSU surface/PBL, legacy RRTMG radiation, Grell-Freitas
cumulus on one arm and no cumulus on the other -- integrates a fixed number
of phase-1/phase-2 steps, and writes every seam output to an ``.npz``.  Run
it once under each engine (two virtualenvs, one card) and ``--compare``
reports, field by field, how many values moved, the largest move and the RMS
move, for the cumulus arm and the no-cumulus arm separately.  The no-cumulus
arm is the control: a difference that appears only on the cumulus arm is the
cumulus scheme's, and a difference on both is somewhere else in the seam.

The measurement is small on purpose -- eight columns, forty levels, twenty
steps -- because its job is attribution, not climatology.  A number that
moves here names its cause; how large the same cause is on a real mesh over
a real day is the obs referee's question, not this tool's.

Two column sets are built in: ``--profile convective`` (the default; a
moist mixed layer under a conditionally unstable troposphere, on which
Grell-Freitas fires) and ``--profile capped`` (a 6.5 K/km profile with no
mixed layer, on which it never does and the two arms are byte-identical).
The first attributes a cumulus move; the second is the control that shows a
surface or boundary-layer move with no cumulus in the picture.

Usage::

    python tools/measure_engine_seam_ab.py --run --out seam-<version>.npz
    python tools/measure_engine_seam_ab.py --compare seam-a.npz seam-b.npz \
        --out seam-ab.json

Needs a CUDA device and cupy; the engine is whatever ``import gpuwm`` resolves
in the interpreter that runs it, and the ``.npz`` records that engine's
version and the SHA-256 of its Grell-Freitas, physics-driver and
microphysics sources so a comparison can never mislabel its arms.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

NZ = 40
NCOL = 8
DT = 120.0
STEPS = 20
DX_M = 25_000.0
P_TOP_PA = 5_000.0
START = _dt.datetime(2021, 6, 1, 18, 0)
WSM6_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")
TENDENCY_NAMES = ("du", "dv", "dtheta", "dqv", "dqc", "dqr", "dqi",
                  "dqs", "dqg", "h_diabatic")
#: The engine sources each ``.npz`` records the digest of: the six the port
#: pins that sit on this seam's fixed-step path, plus the surface-layer,
#: boundary-layer and land-surface files the seam executes and the port does
#: NOT pin -- so a comparison can say which side of that line a move came
#: from.
ENGINE_SOURCES = ("gpuwm/core/gf.py", "gpuwm/core/kernels/gf.cu",
                  "gpuwm/core/physics.py", "gpuwm/core/microphysics.py",
                  "gpuwm/core/rrtmg_legacy.py",
                  "gpuwm/core/mpas_column_batch.py",
                  "gpuwm/core/kernels/ysu.cu", "gpuwm/core/ysu.py",
                  "gpuwm/core/kernels/sfclay.cu", "gpuwm/core/sfclay.py",
                  "gpuwm/core/noahmp_driver_gpu.py",
                  "gpuwm/core/kernels/noahmp_driver.cu",
                  "gpuwm/core/kernels/noahmp_energy.cu",
                  "gpuwm/core/rrtmg_legacy_prep.py",
                  "gpuwm/core/kernels/rrtmg_sw.cu")


# ---------------------------------------------------------------------------
# the columns: integer arithmetic in, float32 out, the same bytes everywhere
# ---------------------------------------------------------------------------
def _jitter(shape, seed: int) -> np.ndarray:
    """A deterministic pseudo-random field in [-1, 1) built without numpy's
    RNG, so a numpy upgrade cannot move the input."""

    n = int(np.prod(shape))
    i = np.arange(n, dtype=np.uint64) + np.uint64(seed) * np.uint64(1_000_003)
    golden = np.uint64(11400714819323198485)
    bits = (i * golden) ^ ((i * golden) >> np.uint64(29))
    frac = (bits & np.uint64(0x00FFFFFF)).astype(np.float64) / float(1 << 24)
    return (2.0 * frac - 1.0).reshape(shape)


def build_capped_columns() -> dict[str, np.ndarray]:
    """Eight capped columns: a 6.5 K/km troposphere from a 302 K, 17 g/kg
    surface, with no mixed layer.

    The surface parcel's level of free convection sits about 250 hPa above
    the ground, beyond the Grell-Freitas cap, so the cumulus arm and the
    no-cumulus arm are byte-identical on this profile and every difference
    between two engines belongs to the surface, boundary-layer and
    radiation path.  Kept as the ``--profile capped`` control because the
    2.6.5 -> 2.7.3 comparison found a boundary-layer move on exactly this
    profile that the convective one does not reach.
    """

    z_iface = np.linspace(0.0, 20_000.0, NZ + 1)
    z_mid = 0.5 * (z_iface[:-1] + z_iface[1:])
    temp = np.where(z_mid < 12_000.0, 302.0 - 6.5e-3 * z_mid,
                    302.0 - 6.5e-3 * 12_000.0)
    g, rd = 9.80665, 287.04
    p_iface = np.empty(NZ + 1)
    p_iface[0] = 100_000.0
    for k in range(NZ):
        p_iface[k + 1] = p_iface[k] * np.exp(
            -g * (z_iface[k + 1] - z_iface[k]) / (rd * temp[k]))
    p_mid = np.sqrt(p_iface[:-1] * p_iface[1:])
    exner = (p_mid / 1.0e5) ** (rd / 1004.5)
    theta = temp / exner
    rho_dry = p_mid / (rd * temp)
    qv = np.where(z_mid < 12_000.0, 0.017 * np.exp(-z_mid / 2_500.0), 2.0e-6)
    qc = np.where((z_mid > 800.0) & (z_mid < 2_500.0), 3.0e-4, 0.0)
    w_iface = 0.4 * np.sin(np.pi * np.clip(z_iface / 8_000.0, 0.0, 1.0))

    def cols(profile, seed, amplitude=3.0e-3):
        base = np.repeat(np.asarray(profile, dtype=np.float64)[:, None],
                         NCOL, axis=1)
        base = base * (1.0 + amplitude * _jitter(base.shape, seed))
        return np.ascontiguousarray(base, dtype=np.float32)

    fields = {
        "u": cols(np.full(NZ, 6.0), 1),
        "v": cols(np.full(NZ, -2.0), 2),
        "theta": cols(theta, 3, amplitude=1.0e-3),
        "pressure": np.ascontiguousarray(
            np.repeat(p_mid[:, None], NCOL, axis=1), dtype=np.float32),
        "pressure_interface": np.ascontiguousarray(
            np.repeat(p_iface[:, None], NCOL, axis=1), dtype=np.float32),
        "z_interface": np.ascontiguousarray(
            np.repeat(z_iface[:, None], NCOL, axis=1), dtype=np.float32),
        "w": cols(w_iface, 4, amplitude=0.2),
        "rho_dry": np.ascontiguousarray(
            np.repeat(rho_dry[:, None], NCOL, axis=1), dtype=np.float32),
        "qv": np.abs(cols(qv, 5)),
        "qc": np.abs(cols(qc, 6)),
        "qr": np.zeros((NZ, NCOL), dtype=np.float32),
        "qi": np.zeros((NZ, NCOL), dtype=np.float32),
        "qs": np.zeros((NZ, NCOL), dtype=np.float32),
        "qg": np.zeros((NZ, NCOL), dtype=np.float32),
    }
    fields["z_nominal"] = z_iface
    return fields


def build_columns() -> dict[str, np.ndarray]:
    """Eight convectively unstable columns in the seam's [level, column] layout.

    A well-mixed 303 K boundary layer to 1.5 km carrying 16 g/kg, capped by
    a 6 K/km free troposphere to 12 km and an isothermal layer above: a
    surface parcel condenses near 1.2 km and is buoyant a few hundred metres
    higher, close enough to the source level for the Grell-Freitas cap test
    to admit it.  A few tenths of a metre per second of low-level ascent,
    and per-column jitter of a few tenths of a percent so no two columns
    are the same and the cumulus scheme's per-column decisions (kbcon,
    ktop, ierr) are exercised rather than repeated.
    """

    z_iface = np.linspace(0.0, 20_000.0, NZ + 1)
    z_mid = 0.5 * (z_iface[:-1] + z_iface[1:])
    g, rd, cp_ = 9.80665, 287.04, 1004.5
    kappa = rd / cp_
    top_ml = 1_500.0
    theta_ml = 303.0
    # Temperature: dry-adiabatic mixed layer (theta constant), then a
    # 6 K/km lapse to 12 km, then isothermal.  Build T and p together,
    # hydrostatically, so theta is exactly constant in the mixed layer.
    temp = np.empty(NZ)
    p_iface = np.empty(NZ + 1)
    p_iface[0] = 100_000.0
    p_mid = np.empty(NZ)
    t_top_ml = None
    for k in range(NZ):
        # provisional mid-level pressure from the layer below
        if k == 0:
            p_guess = p_iface[0] * np.exp(-g * z_mid[0] / (rd * 302.0))
        else:
            p_guess = p_mid[k - 1] * np.exp(
                -g * (z_mid[k] - z_mid[k - 1]) / (rd * temp[k - 1]))
        if z_mid[k] <= top_ml:
            temp[k] = theta_ml * (p_guess / 1.0e5) ** kappa
            t_top_ml = temp[k]
        elif z_mid[k] <= 12_000.0:
            temp[k] = t_top_ml - 6.0e-3 * (z_mid[k] - top_ml)
        else:
            temp[k] = t_top_ml - 6.0e-3 * (12_000.0 - top_ml)
        p_mid[k] = p_guess
        p_iface[k + 1] = p_iface[k] * np.exp(
            -g * (z_iface[k + 1] - z_iface[k]) / (rd * temp[k]))
    exner = (p_mid / 1.0e5) ** kappa
    theta = temp / exner
    rho_dry = p_mid / (rd * temp)
    # Saturation mixing ratio (Bolton), to keep the mixed layer moist but
    # unsaturated: 16 g/kg or 92 % of saturation, whichever is lower.
    es = 611.2 * np.exp(17.67 * (temp - 273.15) / (temp - 29.65))
    qs = 0.622 * es / np.maximum(p_mid - es, 1.0)
    qv = np.where(z_mid <= top_ml, np.minimum(0.016, 0.92 * qs),
                  np.minimum(0.016 * np.exp(-(z_mid - top_ml) / 2_200.0),
                             0.85 * qs))
    qv = np.where(z_mid > 12_000.0, 2.0e-6, qv)
    qc = np.where((z_mid > 1_200.0) & (z_mid < 2_200.0), 2.0e-4, 0.0)
    w_iface = 0.5 * np.sin(np.pi * np.clip(z_iface / 9_000.0, 0.0, 1.0))

    def cols(profile, seed, amplitude=3.0e-3):
        base = np.repeat(np.asarray(profile, dtype=np.float64)[:, None],
                         NCOL, axis=1)
        base = base * (1.0 + amplitude * _jitter(base.shape, seed))
        return np.ascontiguousarray(base, dtype=np.float32)

    fields = {
        "u": cols(np.full(NZ, 6.0), 1),
        "v": cols(np.full(NZ, -2.0), 2),
        "theta": cols(theta, 3, amplitude=1.0e-3),
        "pressure": np.ascontiguousarray(
            np.repeat(p_mid[:, None], NCOL, axis=1), dtype=np.float32),
        "pressure_interface": np.ascontiguousarray(
            np.repeat(p_iface[:, None], NCOL, axis=1), dtype=np.float32),
        "z_interface": np.ascontiguousarray(
            np.repeat(z_iface[:, None], NCOL, axis=1), dtype=np.float32),
        "w": cols(w_iface, 4, amplitude=0.2),
        "rho_dry": np.ascontiguousarray(
            np.repeat(rho_dry[:, None], NCOL, axis=1), dtype=np.float32),
        "qv": np.abs(cols(qv, 5, amplitude=1.0e-2)),
        "qc": np.abs(cols(qc, 6)),
        "qr": np.zeros((NZ, NCOL), dtype=np.float32),
        "qi": np.zeros((NZ, NCOL), dtype=np.float32),
        "qs": np.zeros((NZ, NCOL), dtype=np.float32),
        "qg": np.zeros((NZ, NCOL), dtype=np.float32),
    }
    fields["z_nominal"] = z_iface
    return fields


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def _engine_sources() -> dict[str, str]:
    import gpuwm

    root = Path(gpuwm.__file__).resolve().parent.parent
    out = {}
    for relative in ENGINE_SOURCES:
        path = root / relative
        out[relative] = (hashlib.sha256(path.read_bytes()).hexdigest()
                         if path.is_file() else None)
    return out


def _engine_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("gpuwm")
    except PackageNotFoundError:
        import gpuwm

        return f"checkout:{Path(gpuwm.__file__).resolve().parent.parent}"


def _seam(z_nominal, cumulus: bool):
    from gpuwm.core import mpas_column_batch as mcb

    return mcb.run_mpas_column_batch(
        n_levels=NZ, n_columns=NCOL, dt=DT,
        microphysics_scheme="wsm6",
        radiation_seconds=600.0, surface_pbl_seconds=DT,
        cumulus_seconds=DT if cumulus else None,
        cumulus_scheme="gf" if cumulus else None,
        start_time=START,
        latitude_deg=np.full(NCOL, 35.0),
        longitude_deg=np.full(NCOL, -97.0),
        terrain_height_m=np.zeros(NCOL),
        z_interface_nominal_m=z_nominal,
        p_top_pa=P_TOP_PA, dx_m=DX_M)


def run_arm(fields, cumulus: bool) -> dict[str, np.ndarray]:
    import cupy as cp

    seam = _seam(fields["z_nominal"], cumulus)
    dev = {name: cp.asarray(value) for name, value in fields.items()
           if name != "z_nominal"}
    captured: dict[str, np.ndarray] = {}
    for step in range(1, STEPS + 1):
        result = seam.run_phase1(
            dt=DT, u=dev["u"], v=dev["v"], theta=dev["theta"],
            pressure=dev["pressure"],
            pressure_interface=dev["pressure_interface"],
            z_interface=dev["z_interface"], w=dev["w"],
            rho_dry=dev["rho_dry"],
            **{name: dev[name] for name in WSM6_SPECIES})
        for name in TENDENCY_NAMES:
            captured[f"step{step:02d}/phase1/{name}"] = cp.asnumpy(
                getattr(result, name))
        # Apply the phase-1 rates to the prognostic columns the way a
        # forward step would, so the next step's physics sees a column
        # the physics itself moved.
        dev["theta"] += cp.asarray(getattr(result, "dtheta")) * cp.float32(DT)
        for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
            dev[name] += cp.asarray(getattr(result, f"d{name}")) * cp.float32(DT)
            dev[name] = cp.maximum(dev[name], cp.float32(0.0))
        # WSM6 alone takes rho_dry on phase 2 (its adapter derives
        # rho = 1/alt); the other rows refuse it.
        receipt = seam.run_phase2(
            theta=dev["theta"], pressure=dev["pressure"],
            rho_dry=dev["rho_dry"], z_interface=dev["z_interface"],
            **{name: dev[name] for name in WSM6_SPECIES})
        for name, value in receipt.items():
            captured[f"step{step:02d}/phase2/{name}"] = cp.asnumpy(value)
        for name in ("theta",) + WSM6_SPECIES:
            captured[f"step{step:02d}/state/{name}"] = cp.asnumpy(dev[name])
        buckets = seam.accumulated_precipitation()
        for name, value in buckets.items():
            captured[f"step{step:02d}/bucket/{name}"] = cp.asnumpy(value)
    captured["call_counts"] = np.array(
        json.dumps(seam.call_counts, sort_keys=True))
    return captured


PROFILES = {"convective": build_columns, "capped": build_capped_columns}


def run(out: Path, profile: str = "convective") -> None:
    fields = PROFILES[profile]()
    arms = {}
    for label, cumulus in (("gf", True), ("nocu", False)):
        arm = run_arm(fields, cumulus)
        for name, value in arm.items():
            arms[f"{label}/{name}"] = value
    metadata = {
        "engine_version": _engine_version(),
        "engine_sources": _engine_sources(),
        "profile": profile,
        "nz": NZ, "ncol": NCOL, "dt": DT, "steps": STEPS,
        "dx_m": DX_M, "p_top_pa": P_TOP_PA,
        "start_time": START.isoformat(),
        "arms": {"gf": "cumulus_scheme='gf', cumulus_seconds=dt",
                 "nocu": "cumulus_scheme=None"},
        "input_sha256": hashlib.sha256(
            b"".join(fields[name].tobytes() for name in sorted(fields))
        ).hexdigest(),
        "instrument_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
    }
    arms["metadata"] = np.array(json.dumps(metadata, sort_keys=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arms)
    print(f"wrote {out}: engine {metadata['engine_version']}, "
          f"{len(arms) - 1} arrays")


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------
def _load(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    metadata = json.loads(str(arrays.pop("metadata")))
    return metadata, arrays


def _delta(a: np.ndarray, b: np.ndarray) -> dict:
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    diff = b64 - a64
    # Elements whose BITS differ, so a NaN that stayed a NaN does not count
    # as a move and a -0.0 that became +0.0 does.
    if a.dtype.kind == "f" and a.dtype == b.dtype:
        words = f"u{a.dtype.itemsize}"
        moved = int(np.count_nonzero(a.view(words) != b.view(words)))
    else:
        moved = int(np.count_nonzero(a != b))
    scale = float(np.max(np.abs(a64))) if a.size else 0.0
    largest = float(np.max(np.abs(diff))) if diff.size else 0.0
    return {
        "bytes_identical": a.tobytes() == b.tobytes(),
        "elements": int(a.size),
        "moved": moved,
        "max_abs_delta": largest,
        "rms_delta": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
        "max_abs_reference": scale,
        "max_rel_delta": (largest / scale) if scale > 0.0 else None,
    }


def compare(a_path: Path, b_path: Path, out: Path | None) -> dict:
    meta_a, arrays_a = _load(a_path)
    meta_b, arrays_b = _load(b_path)
    if meta_a["input_sha256"] != meta_b["input_sha256"]:
        raise SystemExit("the two runs were not driven with the same columns; "
                         "nothing they disagree on can be attributed")
    if set(arrays_a) != set(arrays_b):
        raise SystemExit("the two runs captured different arrays; the "
                         "engines expose different seam surfaces and a "
                         "field-by-field comparison would be partial")
    report = {
        "a": {"path": a_path.name, "engine_version": meta_a["engine_version"],
              "engine_sources": meta_a["engine_sources"]},
        "b": {"path": b_path.name, "engine_version": meta_b["engine_version"],
              "engine_sources": meta_b["engine_sources"]},
        "columns": {k: meta_a[k] for k in ("profile", "nz", "ncol", "dt",
                                            "steps", "dx_m", "p_top_pa",
                                            "start_time")},
        "input_sha256": meta_a["input_sha256"],
        "engine_sources_moved": sorted(
            name for name in meta_a["engine_sources"]
            if meta_a["engine_sources"][name] != meta_b["engine_sources"][name]),
        "arms": {},
    }
    for arm in ("gf", "nocu"):
        fields: dict[str, dict] = {}
        for name in sorted(arrays_a):
            if not name.startswith(arm + "/") or name.endswith("call_counts"):
                continue
            fields[name[len(arm) + 1:]] = _delta(arrays_a[name], arrays_b[name])
        moved = {name: row for name, row in fields.items() if not row["bytes_identical"]}
        # Roll the per-step rows up by field so a reader sees the shape at
        # a glance: which quantity, how many of the ten steps, worst move.
        by_field: dict[str, dict] = {}
        for name, row in moved.items():
            _, kind, field = name.split("/", 2)
            key = f"{kind}/{field}"
            entry = by_field.setdefault(key, {
                "steps_moved": 0, "max_abs_delta": 0.0,
                "max_rel_delta": 0.0, "max_abs_reference": 0.0})
            entry["steps_moved"] += 1
            entry["max_abs_delta"] = max(entry["max_abs_delta"], row["max_abs_delta"])
            entry["max_abs_reference"] = max(
                entry["max_abs_reference"], row["max_abs_reference"])
            if row["max_rel_delta"] is not None:
                entry["max_rel_delta"] = max(entry["max_rel_delta"], row["max_rel_delta"])
        report["arms"][arm] = {
            "arrays_compared": len(fields),
            "arrays_byte_identical": len(fields) - len(moved),
            "arrays_moved": len(moved),
            "call_counts_a": json.loads(str(arrays_a[f"{arm}/call_counts"])),
            "call_counts_b": json.loads(str(arrays_b[f"{arm}/call_counts"])),
            "by_field": dict(sorted(by_field.items())),
            "per_array": moved,
        }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    return report


def _print_summary(report: dict) -> None:
    print(f"A: gpuwm {report['a']['engine_version']}")
    print(f"B: gpuwm {report['b']['engine_version']}")
    print(f"engine sources that differ: {report['engine_sources_moved']}")
    for arm, block in report["arms"].items():
        print(f"\n[{arm}] {block['arrays_byte_identical']} of "
              f"{block['arrays_compared']} arrays byte-identical, "
              f"{block['arrays_moved']} moved")
        for field, row in block["by_field"].items():
            rel = row["max_rel_delta"]
            print(f"  {field:28s} steps {row['steps_moved']:2d}  "
                  f"max|d| {row['max_abs_delta']:.3e}  "
                  f"(ref max {row['max_abs_reference']:.3e}, "
                  f"rel {rel:.3e})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", action="store_true",
                        help="drive the seam under the current engine")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="compare two --run outputs")
    parser.add_argument("--out", type=Path, help="the .npz (run) or .json "
                        "(compare) to write")
    parser.add_argument("--profile", choices=sorted(PROFILES),
                        default="convective",
                        help="which fixed column set to drive (run only)")
    args = parser.parse_args(argv)
    if args.run == bool(args.compare):
        parser.error("exactly one of --run / --compare")
    if args.run:
        if args.out is None:
            parser.error("--run needs --out <file.npz>")
        run(args.out, args.profile)
        return 0
    report = compare(Path(args.compare[0]), Path(args.compare[1]), args.out)
    _print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
