import os, json, argparse, time, hashlib
from pathlib import Path
import numpy as np
from dycore_fixture import *

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--steps", type=int, default=80)
parser.add_argument("--out", default="integration")
parser.add_argument("--family", action="store_true")
parser.add_argument("--arms", default="fixed,baseline,balanced")
parser.add_argument("--lts", action="store_true")
parser.add_argument("--rate-shift", type=float, default=0.0)
parser.add_argument("--rates-only", action="store_true")
parser.add_argument("--refined", action="store_true")
args = parser.parse_args()
if args.refined:
    args.family = True
mesh, cfg, coeff, winds = setup(
    "experiments/refined-0.nc" if args.refined else "experiments/grid.nc"
)
nc = len(mesh.arrays["areaCell"])
dt = cfg.config_dt
pattern = np.sin(mesh.arrays["latCell"]) * np.cos(mesh.arrays["lonCell"])
pattern -= np.dot(pattern, mesh.arrays["areaCell"]) / sum(mesh.arrays["areaCell"])
pattern /= abs(pattern).max()


def driver_of(s, saved, p):
    if args.lts:
        assert args.gpu
        from lts_driver import build

        return build(
            s,
            saved,
            p,
            cfg,
            getattr(p, "coeff", coeff),
            winds,
            getattr(p, "rate_shift", 0.0),
        )
    kw = dict(
        terrain_metrics=p.terrain_metrics,
        reference_wind_profiles=winds,
        advection_coefficients=getattr(p, "coeff", coeff),
    )
    if args.gpu:
        from hexcore.cuda_driver import CudaDryDycoreDriver

        s = PrognosticState(
            *(
                np.ascontiguousarray(getattr(s, k))
                for k in ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
            ),
            s.time_seconds,
        )
        saved = DrySavedDiagnostics(
            **{k: np.ascontiguousarray(getattr(saved, k)) for k in saved.__slots__}
        )
        return CudaDryDycoreDriver.from_host(
            p.mesh, s, p.vertical, p.reference, cfg, saved_diagnostics=saved, **kw
        )
    return DryDycoreDriver(p.mesh, p.vertical, p.reference, cfg, **kw)


def advance(d, s, saved):
    if args.lts:
        from lts_driver import advance as advance_lts

        return advance_lts(d)
    if not args.gpu:
        result = d.step(s, saved_diagnostics=saved)
        return result.state, result.saved_diagnostics
    import cupy as cp

    result = d.step_device()
    d.atmosphere = result.atmosphere
    st = result.atmosphere.state
    sa = result.atmosphere.saved
    s = PrognosticState(
        *(
            cp.asnumpy(getattr(st, k))
            for k in ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
        ),
        float(st.time_seconds),
    )
    saved = DrySavedDiagnostics(
        **{k: cp.asnumpy(getattr(sa, k)) for k in DrySavedDiagnostics.__slots__}
    )
    return s, saved


def psfc(s, p):
    # Native physics preparation diagnostic, dry qv=0 in this fixture.
    from hexcore.driver import _pressure_exner

    pressure, _, _ = _pressure_exner(s, p.vertical, **KW)
    dz = np.diff(p.vertical.zgrid, axis=0)
    r = s.rho * p.vertical.zz
    correction = np.float32(0.5) * (r[1] - r[0]) * dz[0] / (dz[0] + dz[1])
    return pressure[0] + np.float32(0.5) * np.float32(9.80616) * dz[0] * (
        r[0] - correction
    )


def run(arm):
    os.environ["GPUWM_HEX_MOVING_BALANCE"] = "1" if arm == "balanced" else "0"
    s, saved, p = fixture(
        mesh, terrain_for(mesh) if args.family else np.zeros(nc), amp=1e-5
    )
    d = driver_of(s, saved, p)
    previous = psfc(s, p)
    rows = []
    budgets = []
    rate_changes = []
    wall = time.monotonic()
    for step in range(args.steps):
        if arm != "fixed" and step in (20, 40, 60):
            old_rates = getattr(d, "fixture_rates", None)
            if args.rates_only:
                p.rate_shift = args.rate_shift
                d = driver_of(s, saved, p)
            else:
                old = p
                # Signed terrain changes test the remap, without changing the horizontal
                # topology or LTS classes; a distinct regenerated family is a separate test.
                if args.family:
                    index = step // 20
                    tm, _, tc, _ = setup(
                        f"experiments/refined-{index}.nc"
                        if args.refined
                        else f"experiments/move{index}.nc"
                    )
                    _, _, p = fixture(tm, terrain_for(tm))
                    p.coeff = tc
                    from scipy.sparse import load_npz

                    area = np.load(
                        f"experiments/refined-areas{index}.npz"
                        if args.refined
                        else f"experiments/areas{index}.npz"
                    )
                    scale = 6371229.0**2
                    horizontal = ConservativeMap(
                        load_npz(
                            f"experiments/refined-overlap{index}.npz"
                            if args.refined
                            else f"experiments/overlap{index}.npz"
                        )
                        * scale,
                        area["source"] * scale,
                        area["target"] * scale,
                        {},
                    )
                    horizontal = horizontal.with_physical_areas(
                        old.mesh.arrays["areaCell"], tm.arrays["areaCell"]
                    )
                    w = HeightConservativeMap(
                        horizontal,
                        old.vertical.zgrid,
                        p.vertical.zgrid,
                        old.vertical.dzw,
                        p.vertical.dzw,
                        endpoint_alignment=False,
                    )
                else:
                    _, _, p = fixture(
                        mesh,
                        (20.0 if step == 20 else -15.0 if step == 40 else 10.0)
                        * pattern,
                    )
                    w = mapper(old, p)
                remap_started = time.monotonic()
                s, saved, budget = remap_atmosphere(
                    payload_of(s, saved, old),
                    w,
                    source_mesh=old.mesh,
                    target_prepared=p,
                    config=cfg,
                )
                remap_seconds = time.monotonic() - remap_started
                jumped = psfc(s, p)
                old_psfc_on_new = w.horizontal.intensive(previous)
                jump = float(abs(jumped - old_psfc_on_new).max())
                source_ref_psfc = psfc(
                    NS(
                        rho=old.reference.rho_base,
                        rho_theta=old.reference.rho_theta_base,
                    ),
                    old,
                )
                target_ref_psfc = psfc(
                    NS(rho=p.reference.rho_base, rho_theta=p.reference.rho_theta_base),
                    p,
                )
                perturbation_jump = float(
                    abs(
                        jumped
                        - target_ref_psfc
                        - w.horizontal.intensive(previous - source_ref_psfc)
                    ).max()
                )
                budgets.append(
                    dict(
                        time=step * dt,
                        budget=budget,
                        remap_seconds=remap_seconds,
                        surface_pressure_remap_jump_pa=jump,
                        reference_relative_pressure_remap_jump_pa=perturbation_jump,
                    )
                )
                p.rate_shift = args.rate_shift
                d = driver_of(s, saved, p)
                # Post-move dynamics metric; separately record the instantaneous remap jump.
                previous = psfc(s, p)
            if old_rates is not None:
                rate_changes.append(
                    dict(
                        time=step * dt,
                        cells_changed=int(np.sum(old_rates != d.fixture_rates)),
                        counts={
                            str(r): int(np.sum(d.fixture_rates == r))
                            for r in np.unique(d.fixture_rates)
                        },
                    )
                )
        s, saved = advance(d, s, saved)
        current = psfc(s, p)
        rows.append(
            dict(
                time=s.time_seconds,
                w=float(abs(saved.vertical_velocity).max()),
                dpdt=float(abs(current - previous).max() / dt),
            )
        )
        previous = current
    hashes = {
        k: hashlib.sha256(getattr(s, k).tobytes()).hexdigest()
        for k in ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
    }
    out = dict(
        rows=rows,
        budgets=budgets,
        rate_changes=rate_changes,
        wall_seconds=time.monotonic() - wall,
        final_hashes=hashes,
        saved_hashes={
            k: hashlib.sha256(getattr(saved, k).tobytes()).hexdigest()
            for k in saved.__slots__
        },
    )
    if args.gpu:
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        out["device"] = dict(
            name=props["name"].decode(),
            total_memory_bytes=props["totalGlobalMem"],
            pool_bytes=cp.get_default_memory_pool().total_bytes(),
        )
    print(arm, out["wall_seconds"], rows[-1], flush=True)
    return out


def terrain_for(m):
    return 100.0 * (np.sin(m.arrays["latCell"]) * np.cos(m.arrays["lonCell"])) ** 2


results = {arm: run(arm) for arm in args.arms.split(",")}
metrics = []
for step in (20, 40, 60):
    if args.steps <= step:
        continue
    row = dict(move_seconds=step * dt)
    for arm in results:
        selected = results[arm]["rows"][step : step + 20]
        row[arm] = {key: max(x[key] for x in selected) for key in ("w", "dpdt")}
    for arm in results:
        if arm == "fixed":
            continue
        row[arm]["ratios"] = {
            key: row[arm][key] / row["fixed"][key] for key in ("w", "dpdt")
        }
    metrics.append(row)
print(json.dumps(metrics, indent=2), flush=True)
Path("evidence/" + args.out + ".json").write_text(
    json.dumps(
        dict(
            gpu=args.gpu,
            refined=args.refined,
            lts=args.lts,
            rates_only=args.rates_only,
            rate_shift=args.rate_shift,
            family=args.family,
            steps=args.steps,
            dt=dt,
            metrics=metrics,
            results=results,
        ),
        indent=2,
    )
    + "\n"
)
