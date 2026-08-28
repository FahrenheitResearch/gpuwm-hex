#!/usr/bin/env python3
"""The regional (limited-area) v8.4.1 forecast on the card.

Runs ``config_apply_lbcs=true`` on a native-culled regional mesh through the
port's CUDA whole-step driver, writes MPAS history frames, and records the
house masked digests (whole-file SHA-256 with the ``file_id`` attribute value
hashed as NUL bytes) so two runs of the same configuration can be compared as
content rather than as container bytes.

Three modes, and the difference between them is the point:

``--mint-anchor``
    Assembles the driver WITHOUT consulting
    ``cuda_backend.regional_admission``.  This is the instrument that
    produces the evidence a registry row names, so it necessarily predates
    the row.  It announces itself; its output is a mint, not admitted
    execution.

default
    Goes through :func:`hexcore.cuda_regional_forecast_v841.open_regional_forecast_v841`,
    which refuses by name unless the configuration holds a registered anchor.

``--compare-cpu``
    Runs the v8.4.1 CPU authority (``DryDycoreDriver`` with a
    ``RegionalRuntime``) over the same steps and compares every published
    field bitwise, per boundary ring.  This is what says whether the device
    is reproducing the authority or merely reproducing itself.

Usage:

    python tools/run_cuda_regional_forecast.py --arm x1 --steps 15 \\
        --mint-anchor --out evidence/.../run1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import string
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from netCDF4 import Dataset  # noqa: E402

from hexcore.config_v841 import V841DryDycoreConfig  # noqa: E402
from hexcore.driver import (  # noqa: E402
    load_mpas_initial_state,
    load_mpas_vertical_grid,
)
from hexcore.dynamics_v841 import (  # noqa: E402
    load_v841_reference_wind_profiles,
)
from hexcore.mesh import Mesh  # noqa: E402
from hexcore.output import (  # noqa: E402
    HistoryField,
    HistoryStreamOptions,
    write_history,
)
from hexcore.regional_v841 import RVORD_F32  # noqa: E402

_XTIME = "%Y-%m-%d_%H:%M:%S"
_PUBLISHED = ("u", "w", "theta", "rho", "qv", "pressure")


def dry_reference_config(
    *,
    config_dt: float = 120.0,
    config_len_disp: float = 25000.0,
    config_apply_lbcs: bool = True,
) -> V841DryDycoreConfig:
    """The CANDIDATE-REGIONAL-DRY namelist, as the port admits it.

    With no arguments this is byte-for-byte the namelist the regional
    reference arms were run with
    (sha256 e623fa93f893115a1072a5ac1e1300641fbefa011803c9ffe12ee6efdf37eea4),
    identical to the one ``tools/run_regional_dry_ladder.py`` pins the CPU
    authority against, so the two lanes are the same configuration.

    THREE VALUES MOVE, AND ONLY BECAUSE THE MESH DOES.  ``config_dt`` and
    ``config_len_disp`` are properties of a mesh's spacing, not of the
    regional lane: a 120 s step and a 25 km mixing length are the reference
    culls' 120 km cells, and running them on a 4.6 km cull would be a Courant
    number of 27 and a mixing length five times the domain.  They are
    arguments so that a cull of a DIFFERENT mesh is a command line rather
    than a second copy of this namelist, and every run records what it used
    (``config_sha256`` in the receipt covers the whole configuration).
    ``config_apply_lbcs`` is false only for the global control arm, which
    exists to measure what the limited-area cull SAVED against the same
    dycore on the uncut parent.
    """

    return V841DryDycoreConfig(
        config_dt=float(config_dt),
        config_split_dynamics_transport=True,
        config_dynamics_split_steps=3,
        config_number_of_sub_steps=6,
        config_scalar_advection=True,
        config_monotonic=False,
        config_positive_definite=False,
        config_scalar_adv_order=3,
        config_scalar_vadv_order=3,
        config_coef_3rd_order=0.25,
        config_apvm_upwinding=0.5,
        config_horiz_mixing="2d_smagorinsky",
        config_len_disp=float(config_len_disp),
        config_visc4_2dsmag=0.05,
        config_smagorinsky_coef=0.125,
        config_del4u_div_factor=10.0,
        config_h_ScaleWithMesh=True,
        config_smdiv=0.1,
        config_divergence_damping=True,
        config_xnutr=0.2,
        config_zd=22_000.0,
        config_apply_lbcs=bool(config_apply_lbcs),
        config_moist_physics=True,
    )


class _GlobalControlRuntime:
    """The no-op stand-in for the regional runtime on the control arm.

    The control arm exists so a limited-area cost claim has something to be a
    ratio OF: the same dycore, the same timestep, the same initial state, on
    the parent the cull came out of.  Everything the publishing code asks the
    regional runtime for is either an identity (there is no garbage element to
    slice off a global mesh) or a count.
    """

    def __init__(self, mesh: Any) -> None:
        self.n_cells_solve = int(np.asarray(mesh.areaCell).size)
        self.n_edges_solve = int(np.asarray(mesh.dcEdge).size)
        self.masks_host = None

    @staticmethod
    def history_slice(array: Any) -> Any:
        return array

    def receipt(self) -> dict[str, Any]:
        return {
            "arm": "global-control",
            "config_apply_lbcs": False,
            "n_cells_solve": self.n_cells_solve,
            "n_edges_solve": self.n_edges_solve,
        }


def assemble_global_control_driver(
    mesh: Any,
    state: Any,
    vertical: Any,
    reference: Any,
    saved: Any,
    config: Any,
    wind: Any,
) -> Any:
    """The same v8.4.1 CUDA dycore on an uncut parent, no regional stage.

    This mirrors :func:`hexcore.cuda_regional_forecast_v841.assemble_regional_driver_v841`
    stage for stage with the padding, the masks and the boundary series
    removed.  The ``dss`` rebuild from ``config_xnutr``/``config_zd`` that core
    always performs is not repeated here because ``from_host`` already does it
    (cuda_driver.py:2452-2474) -- the regional assembler builds the driver
    through its bare constructor and therefore has to; both arms end up with
    the same vertical damping profile, which is what the ratio between them
    requires.
    """

    from hexcore.cuda_backend import require_cuda
    from hexcore.cuda_backend import KernelCache
    from hexcore.cuda_driver import CudaDryDycoreDriver
    from hexcore.transport import build_advection_coefficients

    require_cuda(min_compute=(12, 0))
    n_vert_levels = int(np.asarray(getattr(state, "rho")).shape[0])
    coefficients = build_advection_coefficients(
        mesh,
        config_scalar_adv_order=config.config_scalar_adv_order,
        n_vert_levels=n_vert_levels,
        source_order_v841=True,
    )
    return CudaDryDycoreDriver.from_host(
        mesh,
        state,
        vertical.vertical_grid,
        reference,
        config,
        saved_diagnostics=saved,
        terrain_metrics=vertical.terrain_metrics,
        advection_coefficients=coefficients,
        kernel_cache=KernelCache(),
        reference_wind_profiles=wind,
    )


def frame_time(path: Path) -> datetime:
    with Dataset(path) as dataset:
        dataset.set_auto_maskandscale(False)
        raw = np.asarray(dataset.variables["xtime"][0])
        text = (
            raw.tobytes().decode("utf-8", "replace")
            if raw.dtype.kind in ("S", "U")
            else bytes(raw).decode()
        )
    return datetime.strptime(text.strip()[:19], _XTIME)


def read_frame(path: Path) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    with Dataset(path) as dataset:
        dataset.set_auto_maskandscale(False)
        for name in _PUBLISHED:
            if name not in dataset.variables:
                continue
            slab = np.array(dataset.variables[name][0][:], dtype=np.float32)
            fields[name] = np.ascontiguousarray(slab.T)
    return fields


def compare_bits(name: str, port: np.ndarray, other: np.ndarray) -> dict:
    port = np.ascontiguousarray(np.asarray(port, dtype=np.float32))
    other = np.ascontiguousarray(np.asarray(other, dtype=np.float32))
    if port.shape != other.shape:
        return {
            "field": name,
            "bitwise_equal": False,
            "reason": f"shape {port.shape} != {other.shape}",
        }
    same = port.view(np.uint32) == other.view(np.uint32)
    if bool(same.all()):
        return {"field": name, "bitwise_equal": True, "values": int(port.size)}
    diff = np.abs(port.astype(np.float64) - other.astype(np.float64))
    diff[~np.isfinite(diff)] = np.inf
    bad = ~same
    return {
        "field": name,
        "bitwise_equal": False,
        "values": int(port.size),
        "mismatch_count": int(bad.sum()),
        "max_abs_diff": float(np.max(diff[bad])),
    }


def ring_report(
    name: str,
    port: np.ndarray,
    other: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Where a difference lives, by boundary ring.

    A regional claim that does not separate the driven rings from the
    interior says nothing: rings 6-7 are overwritten from the lateral
    boundary every step and would agree even if the dycore were wrong.
    """

    port = np.ascontiguousarray(np.asarray(port, dtype=np.float32))
    other = np.ascontiguousarray(np.asarray(other, dtype=np.float32))
    if port.shape != other.shape:
        return {"field": name, "reason": "shape"}
    same = port.view(np.uint32) == other.view(np.uint32)
    rows: dict[str, Any] = {}
    for ring in range(0, 8):
        selected = mask == ring
        if not bool(selected.any()):
            continue
        block = same[:, selected]
        rows[str(ring)] = {
            "elements": int(selected.sum()),
            "values": int(block.size),
            "bitwise_equal": bool(block.all()),
            "mismatches": int(block.size - int(block.sum())),
        }
    return {"field": name, "by_ring": rows}


def fresh_file_id() -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(10))


def published_fields(
    driver: Any, runtime: Any, reference_state: Any, zz32: np.ndarray
) -> dict[str, np.ndarray]:
    """The six history fields, sliced free of the garbage element."""

    cp = driver.cp
    state = driver.atmosphere.state
    saved = driver.atmosphere.saved
    one = np.float32(1.0)

    def host(array: Any) -> np.ndarray:
        return np.ascontiguousarray(
            cp.asnumpy(runtime.history_slice(array)), dtype=np.float32
        )

    qv = host(state.scalars[0])
    theta_m = host(saved.theta_m)
    return {
        "u": host(saved.normal_velocity),
        "w": host(saved.vertical_velocity),
        "qv": qv,
        # atm_compute_output_diagnostics F:958-961
        "theta": theta_m / (one + RVORD_F32 * qv),
        "rho": host(state.rho) * zz32,
        "pressure": (
            np.asarray(reference_state.pressure_base, np.float32)
            + host(saved.pressure_perturbation)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--arm", choices=("x1", "x4"), default="x1")
    # The three paths a cull of ANY parent needs.  Supplying --grid takes the
    # tool off the reference-dir layout entirely: the two arms above are the
    # 2026-08-25 record set's own directory shape, and a mesh this program
    # placed for itself does not live in it.  Named paths rather than a third
    # arm because a new region is data, not a code path.
    parser.add_argument("--grid", default=None)
    parser.add_argument("--init", default=None)
    parser.add_argument("--lbc-dir", default=None)
    parser.add_argument("--start-time", default=None, metavar="YYYY-MM-DD_HH:MM:SS")
    parser.add_argument("--dt", type=float, default=None, metavar="SECONDS")
    parser.add_argument("--len-disp", type=float, default=None, metavar="METRES")
    parser.add_argument(
        "--global-control",
        action="store_true",
        help="the same dycore on an UNCUT global parent at "
             "config_apply_lbcs=false and no regional runtime -- the control "
             "arm a limited-area cost claim is measured against",
    )
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--history-every", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mint-anchor", action="store_true")
    parser.add_argument("--measure-garbage", action="store_true")
    parser.add_argument("--compare-cpu", type=int, default=0)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--debug-nonfinite",
        action="store_true",
        help="name the first kernel whose SOLVE-region output is non-finite",
    )
    args = parser.parse_args()

    arm_label = args.arm
    if args.grid is not None:
        # Named-path route: the caller's own cull, of whatever parent.
        if args.init is None:
            raise SystemExit("--grid needs --init: the mesh's initial state")
        if args.start_time is None:
            raise SystemExit(
                "--grid needs --start-time: with no reference frames dir "
                "there is nothing to read the first valid time off, and a "
                "boundary series admitted against the wrong start time would "
                "nudge every ring towards the wrong hour"
            )
        grid = Path(args.grid)
        init = Path(args.init)
        start_time = datetime.strptime(args.start_time, _XTIME)
        arm_label = "named-paths"
        if args.global_control:
            if args.lbc_dir is not None:
                raise SystemExit(
                    "--global-control runs config_apply_lbcs=false on an "
                    "uncut parent; a boundary series has nothing to drive"
                )
            lbc_paths = []
        else:
            if args.lbc_dir is None:
                raise SystemExit(
                    "--grid needs --lbc-dir: a limited-area mesh with no "
                    "boundary series integrates an unforced boundary, which "
                    "is the breakage config_apply_lbcs exists to prevent"
                )
            lbc_paths = sorted(str(p) for p in Path(args.lbc_dir).glob("lbc.*.nc"))
            if not lbc_paths:
                raise SystemExit(f"no lbc files under {args.lbc_dir}")
    else:
        if args.global_control:
            raise SystemExit(
                "--global-control has no reference arm: name the uncut "
                "parent with --grid/--init/--start-time"
            )
        reference = Path(
            args.reference_dir
            or os.environ.get("GPUWM_HEX_REGIONAL_REFERENCE_DIR", "")
        )
        if not reference.is_dir():
            raise SystemExit(
                "reference dir not found; pass --reference-dir or set "
                "GPUWM_HEX_REGIONAL_REFERENCE_DIR"
            )
        if args.arm == "x1":
            grid = reference / "cull-x1" / "conus.grid.nc"
            init = reference / "init-x1" / "conus.init.nc"
            lbc_dir = reference / "lbc-x1"
            frames_dir = reference / "ladder-x1"
        else:
            grid = reference / "cull-x4" / "grid" / "conus.region.nc"
            init = reference / "init" / "conus.init.nc"
            lbc_dir = reference / "lbc"
            frames_dir = reference / "ladder-x4"
        lbc_paths = sorted(str(p) for p in lbc_dir.glob("lbc.*.nc"))
        if not lbc_paths:
            raise SystemExit(f"no lbc files under {lbc_dir}")
        frames = sorted(frames_dir.glob("history.*.nc"))
        if not frames:
            raise SystemExit(f"no history frames under {frames_dir}")
        start_time = frame_time(frames[0])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    mesh = Mesh.from_netcdf(grid, init, validate=False)
    vertical = load_mpas_vertical_grid(
        init, mesh, allow_regional_sentinels=True
    )
    state, reference_state, saved = load_mpas_initial_state(
        init,
        mesh,
        vertical.vertical_grid,
        scalar_names=("qv",),
        terrain_metrics=vertical.terrain_metrics,
        allow_regional_sentinels=True,
        return_saved_diagnostics=True,
    )
    config = dry_reference_config(
        config_dt=120.0 if args.dt is None else args.dt,
        config_len_disp=25000.0 if args.len_disp is None else args.len_disp,
        config_apply_lbcs=not args.global_control,
    )
    wind = load_v841_reference_wind_profiles(init)

    from hexcore import cuda_regional_forecast_v841 as regional_module

    if args.global_control:
        driver = assemble_global_control_driver(
            mesh, state, vertical, reference_state, saved, config, wind
        )
        runtime = _GlobalControlRuntime(mesh)
        setup_wall = time.perf_counter() - t0
        print(
            "[control] the SAME dycore on the uncut parent at "
            "config_apply_lbcs=false: no regional runtime, no boundary "
            "series, no anchor gate -- this arm measures what the cull "
            "saved, not whether a limited-area forecast is right.",
            flush=True,
        )
    elif args.mint_anchor:
        print(
            "[mint] assembling the regional driver WITHOUT the anchor gate: "
            "this run is the instrument that produces the evidence a "
            "regional_admission row names, so it necessarily predates the "
            "row.  Its output is a mint, not admitted execution.",
            flush=True,
        )
        opener = regional_module.assemble_regional_driver_v841
        driver = None
    else:
        opener = regional_module.open_regional_forecast_v841
        driver = None

    if not args.global_control:
        driver = opener(
            mesh,
            state,
            vertical.vertical_grid,
            reference_state,
            saved,
            vertical.terrain_metrics,
            config,
            reference_wind_profiles=wind,
            lbc_paths=lbc_paths,
            start_time=start_time,
            measure_garbage=bool(args.measure_garbage),
        )
        runtime = driver.regional_v841
        setup_wall = time.perf_counter() - t0

    if args.debug_nonfinite and not args.global_control:
        import cupy as _cp

        seen: dict[str, int] = {}

        def audit(name: str, launch_args: Any) -> None:
            if seen:
                return
            for index, value in enumerate(launch_args):
                if (
                    isinstance(value, _cp.ndarray)
                    and value.dtype == _cp.int32
                    and value.size == 1
                    and int(value.get()[0]) != 0
                ):
                    seen[name] = index
                    print(
                        f"[debug] validation flag raised by {name} "
                        f"arg#{index}",
                        flush=True,
                    )
                    return
                if not isinstance(value, _cp.ndarray):
                    continue
                if value.dtype != _cp.float32 or value.ndim == 0:
                    continue
                solve = runtime.discipline._extents.get(int(value.shape[-1]))
                if solve is None:
                    continue
                block = value[..., :solve]
                if not bool(_cp.isfinite(block).all()):
                    seen[name] = index
                    print(
                        f"[debug] first non-finite SOLVE region after {name} "
                        f"arg#{index} shape={tuple(value.shape)}",
                        flush=True,
                    )
                    return

        runtime.discipline.audit = audit

    zz32 = np.asarray(vertical.vertical_grid.zz, dtype=np.float32)
    if runtime.masks_host is None:
        # The control arm has no rings: every element is interior.
        bdy_mask_cell = np.zeros(int(runtime.n_cells_solve), np.int64)
        bdy_mask_edge = np.zeros(int(runtime.n_edges_solve), np.int64)
    else:
        bdy_mask_cell = np.asarray(runtime.masks_host.bdy_mask_cell, np.int64)
        bdy_mask_edge = np.asarray(runtime.masks_host.bdy_mask_edge, np.int64)

    cpu_driver = None
    cpu_state = cpu_saved = None
    if args.compare_cpu > 0 and args.global_control:
        raise SystemExit(
            "--compare-cpu grades the regional stages against the CPU "
            "authority; the control arm runs none of them"
        )
    if args.compare_cpu > 0:
        from hexcore.driver import DryDycoreDriver
        from hexcore.regional_v841 import RegionalRuntime

        cpu_runtime = RegionalRuntime(
            mesh,
            dtype=np.dtype(np.float32),
            lbc_paths=lbc_paths,
            start_time=start_time,
            config_h_scale_with_mesh=bool(config.config_h_ScaleWithMesh),
            zz=vertical.vertical_grid.zz,
            scalar_names=("lbc_qv",),
            moist_indices=(0,),
        )
        cpu_driver = DryDycoreDriver(
            mesh,
            vertical.vertical_grid,
            reference_state,
            config,
            terrain_metrics=vertical.terrain_metrics,
            reference_wind_profiles=wind,
            regional=cpu_runtime,
            index_qv=0,
        )
        cpu_state, cpu_saved = state, saved

    shared_dycore_probe: list[dict[str, Any]] = []
    if cpu_driver is not None:
        # The attribution control.  solve_diagnostics runs ONLY shared
        # v8.4.1 kernels on the step-start state -- no regional stage, no
        # lateral boundary, no zone mask -- so whatever it measures is the
        # shared CUDA dycore against the numpy authority on this mesh.  A
        # forecast residue of the same size and shape as this probe is that
        # residue, not a regional one.
        device_diag = driver.horizontal.solve_diagnostics(
            driver.atmosphere.state.rho,
            driver.atmosphere.state.rho_u,
            dt=float(config.config_dt),
            apvm_upwinding=config.config_apvm_upwinding,
            normal_velocity=driver.atmosphere.saved.normal_velocity,
            rk_step=3,
        )
        host_diag = cpu_driver._diagnostics(
            state,
            outer_dt=float(config.config_dt),
            cached_v=None,
            rk_step=3,
            normal_velocity=np.asarray(saved.normal_velocity, np.float32),
        )
        for name, mask in (
            ("divergence", bdy_mask_cell),
            ("kinetic_energy", bdy_mask_cell),
            ("tangential_velocity", bdy_mask_edge),
            ("h_edge", bdy_mask_edge),
        ):
            device_value = np.ascontiguousarray(
                driver.cp.asnumpy(
                    runtime.history_slice(getattr(device_diag, name))
                ),
                dtype=np.float32,
            )
            host_value = np.asarray(getattr(host_diag, name), np.float32)
            shared_dycore_probe.append(
                {
                    **compare_bits(name, device_value, host_value),
                    **ring_report(name, device_value, host_value, mask),
                }
            )
        print(
            "    shared-dycore probe (no regional stage runs): "
            + ", ".join(
                f"{r['field']}={'BITWISE' if r.get('bitwise_equal') else str(r.get('mismatch_count')) + '/' + str(r.get('values'))}"
                for r in shared_dycore_probe
            ),
            flush=True,
        )

    # The anchor has to bind to BYTES, not to a lane name: a receipt that
    # cannot say which source produced it cannot be re-run against.
    #
    # THE BREAKAGE THIS PREVENTS, measured 2026-08-26: the first version of
    # this list named cuda_driver.py but NOT cuda_transport.py.  The NVRTC
    # reciprocal-rewrite fix (#355) changed BOTH -- the third-order stencil
    # denominator moved from a source literal to a translation-unit constant
    # in each -- and the receipt would have caught only half of that.  Every
    # translation unit the regional step actually launches through is listed
    # here now, so a receipt that still verifies is a receipt whose whole
    # device arithmetic is the arithmetic in this tree.
    #
    # regional_admission.py is deliberately NOT in the list.  It decides
    # ADMISSION, not arithmetic, and the anchor row's own prose lives in it --
    # hashing it would make every edit to a row's basis text invalidate the
    # receipts that row names, which is circular and says nothing about the
    # numbers.  The registry is held to its evidence by
    # tests/test_regional_forecast_anchor.py instead.
    source_digests = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "src/hexcore/cuda_driver.py",
            "src/hexcore/cuda_horizontal.py",
            "src/hexcore/cuda_horizontal_v841.py",
            "src/hexcore/cuda_acoustic_v841.py",
            "src/hexcore/cuda_transport.py",
            "src/hexcore/cuda_transport_v841.py",
            "src/hexcore/cuda_dynamics_v841.py",
            "src/hexcore/cuda_v841.py",
            "src/hexcore/cuda_regional_v841.py",
            "src/hexcore/cuda_regional_forecast_v841.py",
            "src/hexcore/cuda_backend/containers.py",
            "src/hexcore/cuda_backend/runtime.py",
            "src/hexcore/regional_v841.py",
            "tools/run_cuda_regional_forecast.py",
        )
    }

    report: dict[str, Any] = {
        "instrument": "run_cuda_regional_forecast",
        "source_sha256": source_digests,
        "shared_dycore_probe": shared_dycore_probe,
        "schema": "mpas-port.regional-cuda-forecast/v1",
        "label": args.label,
        "arm": arm_label,
        "global_control": bool(args.global_control),
        "config_dt": float(config.config_dt),
        "config_len_disp": float(config.config_len_disp),
        "config_apply_lbcs": bool(config.config_apply_lbcs),
        "grid": str(grid),
        "init": str(init),
        "lbc_paths": lbc_paths,
        "start_time": start_time.strftime(_XTIME),
        "minted_without_anchor_gate": bool(args.mint_anchor),
        "config_sha256": driver.configuration_sha256,
        "n_cells_solve": int(runtime.n_cells_solve),
        "n_edges_solve": int(runtime.n_edges_solve),
        "setup_wall_seconds": round(setup_wall, 2),
        "steps": int(args.steps),
        "frames": [],
        "cpu_authority": [],
    }

    from run_cuda_v841_full_physics_x4 import netcdf_masked_digests

    def publish(step: int, valid: datetime) -> dict[str, Any]:
        fields = published_fields(driver, runtime, reference_state, zz32)
        path = out / f"history.{valid.strftime('%Y-%m-%d_%H.%M.%S')}.nc"
        history_fields = {
            "u": HistoryField(fields["u"].T[None, ...], ("Time", "nEdges", "nVertLevels")),
            "w": HistoryField(fields["w"].T[None, ...], ("Time", "nCells", "nVertLevelsP1")),
            "theta": HistoryField(fields["theta"].T[None, ...], ("Time", "nCells", "nVertLevels")),
            "rho": HistoryField(fields["rho"].T[None, ...], ("Time", "nCells", "nVertLevels")),
            "qv": HistoryField(fields["qv"].T[None, ...], ("Time", "nCells", "nVertLevels")),
            "pressure": HistoryField(
                fields["pressure"].T[None, ...], ("Time", "nCells", "nVertLevels")
            ),
        }
        write_history(
            path,
            mesh,
            history_fields,
            (valid,),
            initial_time=start_time,
            time_seconds=(float(step) * float(config.config_dt),),
            n_vert_levels=int(zz32.shape[0]),
            global_attrs={
                "title": "gpuwm-hex regional (limited-area) v8.4.1 CUDA forecast",
                "physics_suite": "none",
                "config_apply_lbcs": (
                    "true" if config.config_apply_lbcs else "false"
                ),
                "file_id": fresh_file_id(),
            },
            # io_type="netcdf" writes NETCDF3_64BIT_OFFSET, the classic
            # header the house file_id-masked digest is defined against
            # (tools/run_cuda_v841_full_physics_x4.netcdf_masked_digests).
            stream_options=HistoryStreamOptions(
                io_type="netcdf", clobber_mode="truncate"
            ),
        )
        digests = netcdf_masked_digests(path)
        row = {
            "step": step,
            "xtime": valid.strftime(_XTIME),
            "file": path.name,
            "sha256": digests["sha256"],
            "masked_sha256": digests["masked_sha256"],
            "file_id": digests["file_id"],
            "field_sha256": {
                name: hashlib.sha256(
                    np.ascontiguousarray(value).tobytes("C")
                ).hexdigest()
                for name, value in sorted(fields.items())
            },
            "finite": {
                name: bool(np.all(np.isfinite(value)))
                for name, value in sorted(fields.items())
            },
        }
        report["frames"].append(row)
        print(
            f"[{time.strftime('%H:%M:%S')}] step {step} {path.name} "
            f"masked={row['masked_sha256'][:16]}",
            flush=True,
        )
        return row

    publish(0, start_time)
    step_walls = []
    for step in range(1, int(args.steps) + 1):
        started = time.perf_counter()
        result = driver.step_device()
        driver.atmosphere = result.atmosphere
        step_walls.append(round(time.perf_counter() - started, 3))
        valid = start_time + timedelta(seconds=step * float(config.config_dt))
        if cpu_driver is not None and step <= args.compare_cpu:
            cpu_result = cpu_driver.step(cpu_state, saved_diagnostics=cpu_saved)
            cpu_state = cpu_result.state
            cpu_saved = cpu_result.saved_diagnostics
            one = np.float32(1.0)
            cpu_qv = np.asarray(cpu_state.scalars[0], np.float32)
            host_fields = {
                "u": np.asarray(cpu_saved.normal_velocity, np.float32),
                "w": np.asarray(cpu_saved.vertical_velocity, np.float32),
                "qv": cpu_qv,
                "theta": np.asarray(cpu_saved.theta_m, np.float32)
                / (one + RVORD_F32 * cpu_qv),
                "rho": np.asarray(cpu_state.rho, np.float32) * zz32,
                "pressure": np.asarray(
                    reference_state.pressure_base, np.float32
                )
                + np.asarray(cpu_saved.pressure_perturbation, np.float32),
            }
            device_fields = published_fields(
                driver, runtime, reference_state, zz32
            )
            rows = []
            for name in _PUBLISHED:
                mask = bdy_mask_edge if name == "u" else bdy_mask_cell
                rows.append(
                    {
                        **compare_bits(
                            name, device_fields[name], host_fields[name]
                        ),
                        **ring_report(
                            name, device_fields[name], host_fields[name], mask
                        ),
                    }
                )
            report["cpu_authority"].append({"step": step, "fields": rows})
            print(
                f"    cpu-authority step {step}: "
                + ", ".join(
                    f"{r['field']}={'BITWISE' if r.get('bitwise_equal') else 'diff'}"
                    for r in rows
                ),
                flush=True,
            )
        if step % max(1, int(args.history_every)) == 0 or step == args.steps:
            publish(step, valid)

    report["step_wall_seconds"] = step_walls
    report["regional_runtime"] = runtime.receipt()
    # The card's own accounting, not a sampler's: free/total straight off the
    # driver at the end of the run, so the two arms' footprints are read the
    # same way at the same point in the step sequence.
    try:
        cp_free, cp_total = driver.cp.cuda.runtime.memGetInfo()
        pool = driver.cp.get_default_memory_pool()
        report["device_memory"] = {
            "free_bytes": int(cp_free),
            "total_bytes": int(cp_total),
            "process_resident_bytes": int(cp_total) - int(cp_free),
            "cupy_pool_used_bytes": int(pool.used_bytes()),
            "cupy_pool_total_bytes": int(pool.total_bytes()),
        }
    except Exception as error:  # pragma: no cover - diagnostic only
        report["device_memory"] = {"unavailable": str(error)}
    receipt = out / "forecast.json"
    receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"receipt: {receipt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
