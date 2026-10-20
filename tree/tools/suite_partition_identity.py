"""Card-count identity of the selectable column suite on a real mesh's columns.

What this proves, and what it does not.  Hex runs column physics per rank on
the rank's own cells, one seam per card.  Whether 1, 2 and 4 cards give the
same physics is therefore the question whether the suite seam, handed the
cells of a METIS partition on its own card, computes every column exactly as
a single seam on one card computes it.  This harness answers that on the
mesh's real columns: each worker process takes one card, builds the engine's
suite seam (``gpuwm.core.mpas_column_suite``) on its partition's cells from
the native initial state, the native surface and soil, and the mesh's
orographic statistics, steps it through radiation, surface, PBL with the
drag, cumulus and microphysics, and writes every tendency and every updated
state field by global cell index.  ``compare`` reassembles each card count
and requires byte equality with the one-card run.

It does not run the dycore: the atmosphere each step is the init's, advanced
only by the physics itself (phase two's in-place update feeds the next phase
one).  The dycore's own card-count identity is a separate, existing proof.

Every NetCDF byte is decoded by the engine's Rust bridge (rw_netcdf).

  worker  --rank R --nranks N --part FILE|none --init IC --oro ORO --out DIR
  compare --out DIR --counts 1,2,4
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa", "nifa")
OUTPUTS = ("du", "dv", "dtheta", "dqv", "dqc", "dqr", "dqi", "dqs", "dqg", "h_diabatic")
DT = 60.0


def _read(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    from gpuwm.netcdf_bridge import open_dataset

    dataset = open_dataset(path)
    out = {}
    for name in names:
        if name not in dataset.variables:
            continue
        out[name] = np.asarray(dataset.variables[name][:])
    return out


def _load(init: Path, oro: Path) -> dict[str, np.ndarray]:
    names = [
        "latCell", "lonCell", "zgrid", "theta", "rho", "pressure_base", "pressure_p",
        "uReconstructZonal", "uReconstructMeridional", "w", *SPECIES,
        "landmask", "xland", "ivgtyp", "isltyp", "vegfra", "skintemp", "tmn",
        "xice", "snow", "snowh", "tslb", "smois",
    ]
    fields = _read(init, names)
    missing = [name for name in ("theta", "zgrid", "qv", "tslb") if name not in fields]
    if missing:
        raise SystemExit(f"init lacks {missing}")
    from hexcore.cuda_suite_physics_v841 import physics_suite

    spec = physics_suite("mynn-ruc-gsl")["orographic_statistics"]["variables"]
    stats = _read(oro, list(spec.values()))
    fields["oro"] = {engine: stats[file].astype(np.float32) for engine, file in spec.items()}
    return fields


def _squeeze(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    while value.ndim > 1 and value.shape[0] == 1:
        value = value[0]
    return value


def worker(args: argparse.Namespace) -> int:
    import cupy as cp
    from gpuwm.core.mpas_column_suite import run_mpas_column_suite

    fields = _load(args.init, args.oro)
    ncell = int(_squeeze(fields["latCell"]).shape[0])
    if args.part == "none":
        owner = np.zeros(ncell, np.int64)
    else:
        owner = np.loadtxt(args.part, dtype=np.int64)
        if owner.shape != (ncell,):
            raise SystemExit(f"partition has {owner.shape} cells, mesh {ncell}")
    cells = np.flatnonzero(owner == args.rank)
    ncol = int(cells.size)

    def col(name, dtype=np.float32):
        value = _squeeze(fields[name])
        return np.ascontiguousarray(value[cells].astype(dtype))

    def lev(name):  # [nCells, nLev] -> [nLev, ncol]
        value = _squeeze(fields[name]).astype(np.float32)
        return np.ascontiguousarray(value[cells].T)

    zgrid = _squeeze(fields["zgrid"]).astype(np.float64)
    nlev = zgrid.shape[1] - 1
    nominal = np.mean(zgrid - zgrid[:, :1], axis=0)
    theta = lev("theta")
    pressure = np.ascontiguousarray(
        (_squeeze(fields["pressure_base"]) + _squeeze(fields["pressure_p"]))[cells].T.astype(np.float32))
    z_iface = np.ascontiguousarray(zgrid[cells].T.astype(np.float32))
    z_mid = 0.5 * (z_iface[:-1] + z_iface[1:])
    # Interface pressure from the mid-level pressures: log-linear between
    # levels, extrapolated at the surface and top (a fixed input every
    # card count receives identically).
    logp = np.log(pressure.astype(np.float64))
    p_if = np.empty((nlev + 1, ncol), np.float64)
    weight = (z_iface[1:-1] - z_mid[:-1]) / (z_mid[1:] - z_mid[:-1])
    p_if[1:-1] = np.exp(logp[:-1] + weight * (logp[1:] - logp[:-1]))
    p_if[0] = np.exp(logp[0] + (z_iface[0] - z_mid[0]) * (logp[1] - logp[0]) / (z_mid[1] - z_mid[0]))
    p_if[-1] = np.exp(logp[-1] + (z_iface[-1] - z_mid[-1]) * (logp[-1] - logp[-2]) / (z_mid[-1] - z_mid[-2]))
    # p_top is ONE scalar for the mesh, taken from every cell (never from
    # the partition's), so every card count builds the same seam constants:
    # the top-layer pressure extrapolated half a layer up, area-unweighted.
    p_all = (_squeeze(fields["pressure_base"]) + _squeeze(fields["pressure_p"])).astype(np.float64)
    z_all = 0.5 * (zgrid[:, :-1] + zgrid[:, 1:])
    slope = (np.log(p_all[:, -1]) - np.log(p_all[:, -2])) / (z_all[:, -1] - z_all[:, -2])
    p_top = float(np.mean(np.exp(np.log(p_all[:, -1]) + (zgrid[:, -1] - z_all[:, -1]) * slope)))
    soil_t = np.ascontiguousarray(_squeeze(fields["tslb"])[cells].T.astype(np.float32))
    soil_m = np.ascontiguousarray(_squeeze(fields["smois"])[cells].T.astype(np.float32))
    xland = col("xland")
    seam = run_mpas_column_suite(
        n_levels=nlev, n_columns=ncol, dt=DT,
        radiation_seconds=3 * DT, surface_pbl_seconds=DT,
        cumulus_seconds=DT, cumulus_scheme="gf", gf_ishallow=1,
        microphysics_scheme="thompson_aero",
        radiation_scheme="rrtmg_legacy", surface_layer_scheme="mynn",
        land_surface_scheme="ruc", pbl_scheme="mynn", drag_scheme="gsl",
        start_time=datetime(2026, 10, 3, 12),
        latitude_deg=np.degrees(col("latCell", np.float64)).astype(np.float32),
        longitude_deg=np.degrees(col("lonCell", np.float64)).astype(np.float32),
        terrain_height_m=np.ascontiguousarray(z_iface[0]),
        z_interface_nominal_m=nominal, p_top_pa=p_top, dx_m=12000.0,
        dx_column_m=np.full(ncol, 12000.0, np.float32),
        landmask=col("landmask"), xland=xland, ivgtyp=col("ivgtyp", np.int32),
        isltyp=col("isltyp", np.int32), vegfra=col("vegfra"), tsk=col("skintemp"),
        tmn=col("tmn"), xice=col("xice"), snow=col("snow"), snow_depth=col("snowh"),
        soil_temperature=soil_t, soil_moisture=soil_m, fractional_seaice=1,
        orographic_statistics={k: v[cells] for k, v in fields["oro"].items()},
    )
    up = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
    base = {
        "u": up(lev("uReconstructZonal")), "v": up(lev("uReconstructMeridional")),
        "pressure": up(pressure), "pressure_interface": up(p_if.astype(np.float32)),
        "z_interface": up(z_iface), "w": up(_squeeze(fields["w"])[cells].T),
        "rho_dry": up(lev("rho")) if "rho" in fields else None,
    }
    state2 = {"theta": up(theta), "pressure": base["pressure"], "z_interface": base["z_interface"]}
    for name in SPECIES:
        state2[name] = up(lev(name)) if name in fields else cp.zeros((nlev, ncol), cp.float32)
    frames = {}
    started = time.time()
    for step in range(args.steps):
        p1 = {k: v for k, v in base.items() if v is not None}
        if "rho_dry" not in p1:
            p1["rho_dry"] = up(pressure / (287.0 * theta))
        p1["theta"] = state2["theta"]
        for name in SPECIES:
            p1[name] = state2[name]
        result = seam.run_phase1(dt=DT, **p1)
        for name in OUTPUTS:
            frames[f"s{step}_{name}"] = cp.asnumpy(getattr(result, name))
        receipt = seam.run_phase2(**state2)
        for key in ("rainncv", "snowncv", "sr"):
            frames[f"s{step}_{key}"] = cp.asnumpy(receipt[key])
        frames[f"s{step}_flags"] = np.array(
            [result.radiation_ran, result.surface_pbl_ran, result.cumulus_ran])
    for name, value in state2.items():
        if name not in ("pressure", "z_interface"):
            frames[f"final_{name}"] = cp.asnumpy(value)
    for name, value in seam.soil_state().items():
        frames[f"final_{name}"] = cp.asnumpy(value)
    for name, value in seam.accumulated_precipitation().items():
        frames[f"final_{name}"] = cp.asnumpy(value)
    out = Path(args.out) / f"n{args.nranks}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"rank{args.rank}.npz", cells=cells, **frames)
    receipt = {
        "rank": args.rank, "nranks": args.nranks, "columns": ncol,
        "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "steps": args.steps, "seconds": time.time() - started,
        "selection": seam.selection, "n_soil_levels": seam.n_soil_levels,
        "call_counts": seam.call_counts, "ruc_census": seam.last_ruc_census,
        "finite": bool(all(np.all(np.isfinite(v)) for k, v in frames.items()
                           if v.dtype.kind == "f")),
    }
    (out / f"rank{args.rank}.json").write_text(json.dumps(receipt, indent=2, default=str))
    print(json.dumps(receipt, default=str))
    return 0


def compare(args: argparse.Namespace) -> int:
    root = Path(args.out)
    counts = [int(c) for c in args.counts.split(",")]

    def assemble(n: int) -> dict[str, np.ndarray]:
        parts = [np.load(root / f"n{n}" / f"rank{r}.npz") for r in range(n)]
        cells = np.concatenate([p["cells"] for p in parts])
        order = np.argsort(cells)
        out = {}
        for key in parts[0].files:
            if key == "cells" or key.endswith("_flags"):
                continue
            joined = np.concatenate([p[key] for p in parts], axis=-1)
            out[key] = joined[..., order]
        out["_flags"] = np.stack([p[k] for p in parts for k in p.files if k.endswith("_flags")])
        return out

    reference = assemble(counts[0])
    report = {"reference_cards": counts[0], "fields": len(reference) - 1, "results": {}}
    ok = True
    for n in counts[1:]:
        other = assemble(n)
        differing = []
        for key, value in reference.items():
            if key == "_flags":
                continue
            if value.shape != other[key].shape or value.tobytes() != other[key].tobytes():
                differing.append(key)
        digest = hashlib.sha256(b"".join(other[k].tobytes() for k in sorted(other) if k != "_flags")).hexdigest()
        report["results"][str(n)] = {"identical": not differing, "differing": differing[:20],
                                     "content_sha256": digest}
        ok &= not differing
    report["reference_content_sha256"] = hashlib.sha256(
        b"".join(reference[k].tobytes() for k in sorted(reference) if k != "_flags")).hexdigest()
    report["all_identical"] = bool(ok)
    (root / "identity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--rank", type=int, required=True)
    w.add_argument("--nranks", type=int, required=True)
    w.add_argument("--part", required=True)
    w.add_argument("--init", type=Path, required=True)
    w.add_argument("--oro", type=Path, required=True)
    w.add_argument("--steps", type=int, default=6)
    w.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--counts", default="1,2,4")
    args = parser.parse_args(argv)
    return worker(args) if args.cmd == "worker" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
