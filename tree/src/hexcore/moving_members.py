"""Stage a moving-region case and prepare its mesh-family members (CPU only).

The stages, each skipped when its receipt closed (so two controllers asking
for the same member wait for one preparation and then reuse it):

* :func:`stage_case` -- the case's initial-condition source through the
  table-driven init-source path (:mod:`hexcore.init_sources`: acquire, then
  decode into one WPS intermediate), and optionally the storm's IBTrACS best
  track for verification;
* :func:`prepare_case` -- the start member; for a corridor case first the
  FRAME (fixed rings alone) and the TEMPLATE (the generator's mesh of the
  full request about the initial centre);
* :func:`prepare_member` -- one member: generator spec, grid (``rw_mpas_mesh``,
  or a stitched / frozen-far-field member through ``--from-centres``),
  static (``rw_mpas_static``), init (the init door, with the source row's
  first-guess declarations) and its mesh row;
* :func:`build_operator` / :func:`build_zone` -- the conservative overlap (or
  zone carry) from one member to the next.

Every Rust tool resolves through :mod:`hexcore.engines` (explicit variable,
the installed gpuwm's bundled bridge, then PATH), so a pip install runs it.
Moved here from the source tree's ``tools/moving48`` shell scripts, which
were bound to one rented host's queue and work folders.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .moving_case import (
    MovingCaseError,
    best_track_from_ibtracs,
    cycle_of,
    frame_spec,
    load_case,
    locked,
    register_member,
    resolve_vertical_spec,
    spec as member_spec,
    stitch,
)

STAGE_FILE = "stage.json"
IBTRACS_URL = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
               "v04r01/access/csv/ibtracs.{subset}.list.v04r01.csv")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _stamp(folder: Path, name: str, text: str) -> None:
    (folder / "logs").mkdir(parents=True, exist_ok=True)
    with (folder / "logs" / "prep-stages.txt").open("a") as stream:
        stream.write(f"{_now()} {name} {text}\n")


def _run_logged(argv: list[str], log: Path, *, env: dict[str, str] | None = None) -> float:
    """Run one stage with its output in ``log``; refuse by name on a non-zero exit."""

    log.parent.mkdir(parents=True, exist_ok=True)
    from .stage_timing import stage

    stages = []
    execution_env = os.environ if env is None else env
    try:
        with stage(stages, log.stem, parallelism={
            "device": "CPU",
            "requested_rayon_threads": execution_env.get("RAYON_NUM_THREADS", "runtime default"),
            "requested_omp_threads": execution_env.get("OMP_NUM_THREADS", "runtime default"),
            "requested_tile_workers": execution_env.get("RW_MPAS_TILE_WORKERS", "runtime default"),
            "visible_devices": execution_env.get("CUDA_VISIBLE_DEVICES", "inherited"),
        }) as timing:
            with log.open("w") as stream:
                rc = subprocess.call([str(a) for a in argv], stdout=stream, stderr=subprocess.STDOUT, env=env)
            timing["returncode"] = rc
    finally:
        if stages and stages[0].get("returncode", 0) != 0:
            stages[0]["status"] = "failed"
        log.with_suffix(".timing.json").write_text(json.dumps(dict(
            schema="gpuwm-hex-stage-timing-v1", argv=[str(a) for a in argv], stages=stages), indent=2) + "\n")
    if rc != 0:
        tail = log.read_text(errors="replace").strip().splitlines()[-5:]
        raise MovingCaseError(f"{Path(str(argv[0])).name} exited {rc} (log {log}): " + " | ".join(tail))
    return stages[0]["wall_seconds"]


def _tool(spec_name: str) -> Path:
    from . import engines

    try:
        return engines.resolve(getattr(engines, spec_name))
    except engines.EngineRefusal as error:
        raise MovingCaseError(str(error)) from None


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------
def read_stage(casedir: Path) -> dict[str, Any]:
    path = Path(casedir) / STAGE_FILE
    if not path.is_file():
        raise MovingCaseError(
            f"{path} does not exist: stage the case first (gpuwm-hex moving stage --case ... --casedir {casedir})")
    return json.loads(path.read_text())


def stage_case(case_path: Path, casedir: Path, *, source: str | None = None, staged: list[Path] = (),
               best_track: bool = False, ibtracs_subset: str = "last3years", ibtracs_file: Path | None = None,
               jobs: int = 3, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Initial-condition source -> one WPS intermediate; optionally the best track."""

    from . import init_sources

    case = load_case(case_path)
    casedir = Path(casedir).expanduser().absolute()
    casedir.mkdir(parents=True, exist_ok=True)
    table = init_sources.load_table()
    source_id = source or case.get("init_source") or "gfs"
    row = init_sources.resolve_source(table, source_id)
    cycle = init_sources.Cycle.parse(cycle_of(case))
    out_dir = casedir / "source"
    previous = casedir / STAGE_FILE
    if previous.is_file():
        stage = json.loads(previous.read_text())
        if stage.get("source") == row["id"] and stage.get("cycle") == cycle.start_time \
                and Path(stage.get("met", "")).is_file():
            log(f"stage: {row['id']} {cycle.start_time} already staged ({stage['met']})")
            if best_track and not (casedir / "obs" / "best-track.json").is_file():
                stage["best_track"] = _best_track(case, casedir, ibtracs_subset, ibtracs_file, log)
                previous.write_text(json.dumps(stage, indent=2) + "\n")
            return stage
    writer = init_sources.resolve_writer(None)
    init_sources.resolve_vtable(row)
    began = time.monotonic()
    log(f"stage: acquiring {row['id']} for {cycle.start_time}")
    products = init_sources.acquire(init_sources.plan_products(row, cycle, None), out_dir / "products", cycle,
                                    staged=[Path(p).expanduser().absolute() for p in staged], jobs=jobs)
    inputs = [product["file"] for product in products]
    remap = None
    if row.get("remap"):
        remap = init_sources.remap(row, products, cycle, out_dir, init_sources.resolve_remapper(None))
        inputs = remap.pop("outputs")
    log("stage: decoding to a WPS intermediate")
    decoded = init_sources.decode(row, inputs, cycle, out_dir, writer)
    stage = dict(schema="gpuwm-hex.moving-stage/v1", created_utc=_now(), case=case["_path"],
                 source=row["id"], cycle=cycle.start_time, met=decoded["met"], products=products,
                 decode=decoded, remap=remap, seconds=round(time.monotonic() - began, 1))
    if best_track:
        stage["best_track"] = _best_track(case, casedir, ibtracs_subset, ibtracs_file, log)
    tmp = previous.with_suffix(".json.partial")
    tmp.write_text(json.dumps(stage, indent=2) + "\n")
    tmp.replace(previous)
    return stage


def _best_track(case, casedir: Path, subset: str, local: Path | None, log) -> dict[str, Any]:
    sid = case["storm"].get("ibtracs_sid")
    if not sid:
        raise MovingCaseError("the case names no storm.ibtracs_sid, so there is no best track to fetch")
    obs = casedir / "obs"
    obs.mkdir(parents=True, exist_ok=True)
    if local is None:
        url = IBTRACS_URL.format(subset=subset)
        local = obs / Path(url).name
        if not local.is_file() or local.stat().st_size == 0:
            log(f"stage: fetching {url}")
            partial = local.with_name(local.name + ".part")
            request = urllib.request.Request(url, headers={"User-Agent": "gpuwm-hex"})
            with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as out:
                while True:
                    block = response.read(1 << 22)
                    if not block:
                        break
                    out.write(block)
            partial.replace(local)
    track = best_track_from_ibtracs(Path(local), sid)
    out = obs / "best-track.json"
    out.write_text(json.dumps(track, indent=1) + "\n")
    log(f"stage: best track {sid} {track['name']}: {len(track['rows'])} rows")
    return dict(path=str(out), ibtracs=str(local), ibtracs_sha256=_sha256(Path(local)), sid=sid)


# ---------------------------------------------------------------------------
# members
# ---------------------------------------------------------------------------
def _grid_centres(path: Path):
    import numpy as np
    from netCDF4 import Dataset

    with Dataset(path) as grid:
        return np.stack([np.asarray(grid.variables[k][:], dtype=np.float64) for k in ("xCell", "yCell", "zCell")], -1)


def _min_dc_edge(path: Path) -> float:
    import numpy as np
    from netCDF4 import Dataset

    with Dataset(path) as grid:
        return float(np.min(np.asarray(grid.variables["dcEdge"][:])))


def _from_centres(request: Path, grid: Path, receipt: Path) -> None:
    tool = _tool("MESH")
    subprocess.run([str(tool), "--from-centres", str(request), "--out", str(grid), "--receipt", str(receipt),
                    "--clobber"], check=True)


def build_stitched_mesh(case: dict[str, Any], name: str, lat: float, lon: float, folder: Path) -> dict[str, Any]:
    """A member stitched from FOLDER/frame.grid.nc and FOLDER/template.grid.nc (corridor case)."""

    from .moving_zone import build_insert_member, spec_anchor, unit_from_deg, write_centres

    template_grid = folder / "template.grid.nc"
    template_spec = json.loads((folder / "template.spec.json").read_text())
    new_spec = member_spec(case, name, float(lat), float(lon))
    (folder / f"{name}.spec.json").write_text(json.dumps(new_spec, indent=2) + "\n")
    radii = stitch(case)
    started = time.time()
    frame, template = _grid_centres(folder / "frame.grid.nc"), _grid_centres(template_grid)
    rows = int(case["mesh"]["corridor"].get("seam_rows", 6))
    attempts = []
    for extra in (0, 1, 2, 3):
        # A stubborn seam defect belongs to where the seam lies: a wider annulus moves it.
        try:
            result = build_insert_member(
                frame, template, spec_anchor(template_spec), unit_from_deg(float(lat), float(lon)), new_spec,
                inner_radius_km=radii["inner_radius_km"],
                outer_radius_km=radii["outer_radius_km"] + extra * radii["fixed_spacing_km"],
                fixed_spacing_km=radii["fixed_spacing_km"],
                progress=lambda row: print(json.dumps(row), flush=True))
            break
        except ValueError as error:
            if "outside the fixed corridor" in str(error):
                raise
            attempts.append(dict(seam_rows=rows + extra, refusal=str(error)))
            print(json.dumps(dict(retry=attempts[-1])), flush=True)
    else:
        raise MovingCaseError(f"no seam width produced an admissible member: {attempts}")
    result.receipt["refused_attempts"] = attempts
    request = folder / f"{name}.centres.nc"
    write_centres(request, result, template_grid, spec=new_spec)
    grid = folder / f"{name}.grid.nc"
    generator_receipt = folder / f"{name}-from-centres.json"
    began = time.time()
    _from_centres(request, grid, generator_receipt)
    receipt = dict(emitted=dict(path=str(grid), sha256=_sha256(grid), n_cells=len(result.centres)),
                   receipt=json.loads(generator_receipt.read_text()).get("receipt", {}),
                   moving_zone=dict(result.receipt, frame_grid=str(folder / "frame.grid.nc"),
                                    template_grid=str(template_grid), min_dc_edge=_min_dc_edge(grid),
                                    template_min_dc_edge=_min_dc_edge(template_grid),
                                    from_centres_seconds=time.time() - began,
                                    total_seconds=time.time() - started))
    (folder / f"{name}-mesh.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def build_zone_mesh(case: dict[str, Any], prev: str, name: str, lat: float, lon: float,
                    folder: Path) -> dict[str, Any]:
    """The next frozen-far-field member's grid, from the previous member's generators."""

    import numpy as np
    from netCDF4 import Dataset

    from .moving_zone import build_member_centres, write_centres

    prev_grid = folder / f"{prev}.grid.nc"
    prev_spec = json.loads((folder / f"{prev}.spec.json").read_text())
    new_spec = member_spec(case, name, float(lat), float(lon))
    (folder / f"{name}.spec.json").write_text(json.dumps(new_spec, indent=2) + "\n")
    zone = case.get("moving_zone", {})
    with Dataset(prev_grid) as grid:
        centres = np.stack([np.asarray(grid.variables[k][:], dtype=np.float64) for k in ("xCell", "yCell", "zCell")], -1)
        density = np.asarray(grid.variables["meshDensity"][:], dtype=np.float64)
        cells_on_cell = np.asarray(grid.variables["cellsOnCell"][:], dtype=np.int64)
        counts = np.asarray(grid.variables["nEdgesOnCell"][:], dtype=np.int64)
    started = time.time()
    halo = int(zone.get("halo_rings", 2))
    attempts = []
    for rings in (halo, halo + 1, halo + 2):
        # A stubborn seam defect is a property of where the band ends: a wider band moves the seam.
        try:
            result = build_member_centres(
                centres, density, cells_on_cell, counts, prev_spec, new_spec,
                spacing_tolerance=float(zone.get("spacing_tolerance", 5e-3)),
                halo_rings=rings, fixed_rings=int(zone.get("fixed_rings", 3)),
                progress=lambda row: print(json.dumps(row), flush=True))
            break
        except ValueError as error:
            attempts.append(dict(halo_rings=rings, refusal=str(error)))
            print(json.dumps(dict(retry=attempts[-1])), flush=True)
    else:
        raise MovingCaseError(f"no band width produced an admissible member: {attempts}")
    result.receipt["refused_attempts"] = attempts
    request = folder / f"{name}.centres.nc"
    write_centres(request, result, prev_grid, spec=new_spec)
    grid = folder / f"{name}.grid.nc"
    generator_receipt = folder / f"{name}-from-centres.json"
    began = time.time()
    _from_centres(request, grid, generator_receipt)
    receipt = dict(emitted=dict(path=str(grid), sha256=_sha256(grid), n_cells=len(result.centres)),
                   receipt=json.loads(generator_receipt.read_text()).get("receipt", {}),
                   moving_zone=dict(result.receipt, previous_member=prev, previous_grid=str(prev_grid),
                                    from_centres_seconds=time.time() - began,
                                    total_seconds=time.time() - started))
    (folder / f"{name}-mesh.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def _tool_env(threads: int | None, tile_workers: int | None) -> dict[str, str]:
    env = dict(os.environ)
    if threads:
        env["RAYON_NUM_THREADS"] = str(threads)
        env["OMP_NUM_THREADS"] = str(threads)
    if tile_workers:
        env["RW_MPAS_TILE_WORKERS"] = str(tile_workers)
    return env


def generate_mesh(spec_path: Path, grid: Path, receipt: Path, log: Path, env: dict[str, str]) -> float:
    return _run_logged([_tool("MESH"), "--spec", spec_path, "--out", grid, "--receipt", receipt], log, env=env)


def prepare_member(case_path: Path, folder: Path, name: str, lat: float, lon: float, *,
                   casedir: Path | None = None, prev: str | None = None, stitched: bool = False,
                   rotate_from: str | None = None,
                   geog: Path | None = None, threads: int | None = None, tile_workers: int | None = None,
                   log: Callable[[str], None] = print) -> dict[str, Any]:
    """spec -> grid -> static -> init -> mesh row, reusing closed stages with their artifacts.

    ``folder`` holds the members; ``casedir`` (default: ``folder``) holds the
    staged source (:func:`stage_case`).
    """

    from . import init_sources

    case = load_case(case_path)
    if rotate_from is not None and (prev or stitched or case["mesh"].get("corridor")):
        raise MovingCaseError("template rotation requires a whole global cap family, without corridor or frozen far field")
    if rotate_from == name:
        raise MovingCaseError("template rotation needs a different source member")
    folder = Path(folder).expanduser().absolute()
    folder.mkdir(parents=True, exist_ok=True)
    stage = read_stage(Path(casedir or folder).expanduser().absolute())
    vertical = resolve_vertical_spec(case)
    env = _tool_env(threads, tile_workers)
    logs = folder / "logs"
    with locked(folder / f"{name}.prep.lock"):
        spec_path = folder / f"{name}.spec.json"
        if not spec_path.is_file():
            spec_path.write_text(json.dumps(member_spec(case, name, float(lat), float(lon)), indent=2) + "\n")
        mesh_receipt = folder / f"{name}-mesh.json"
        grid = folder / f"{name}.grid.nc"
        def mesh_ready():
            return mesh_receipt.is_file() and grid.is_file()
        mesh_rebuilt = not mesh_ready()
        if not mesh_ready() and rotate_from is not None:
            from .moving_mesh import rotate_grid
            from .stage_timing import stage as timed_stage

            source_spec = json.loads((folder / f"{rotate_from}.spec.json").read_text())
            source_center = source_spec["regions"][-1]["shape"]["center_deg"]
            if source_spec != member_spec(case, rotate_from, *source_center):
                raise MovingCaseError("rotation template does not match this case's resolution and concentric cap family")
            source_grid = folder / f"{rotate_from}.grid.nc"
            source_receipt = json.loads((folder / f"{rotate_from}-mesh.json").read_text())
            source_sha256 = _sha256(source_grid)
            if source_sha256 != source_receipt["emitted"]["sha256"]:
                raise MovingCaseError("rotation template bytes differ from its mesh receipt")
            grid = folder / f"{name}.grid.nc"
            timings = []
            with timed_stage(timings, "rotate-mesh", parallelism={"device": "CPU", "threads": 1}):
                rotation = rotate_grid(source_grid, grid,
                                       start=source_center, end=(float(lat), float(lon)))
            rotation["source_sha256"] = source_sha256
            receipt = dict(emitted=dict(path=str(grid), sha256=rotation["sha256"],
                                       n_cells=source_receipt["emitted"]["n_cells"]),
                           rotation=rotation, stages=timings)
            mesh_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        if not mesh_ready() and stitched:
            # Stitched member: the corridor frame's far field + the template's fine region moved
            # rigidly.  A refused stitch never falls through to a whole-globe generator mesh.
            began = time.time()
            build_stitched_mesh(case, name, lat, lon, folder)
            _stamp(folder, name, f"stitch-mesh {time.time() - began:.0f}s")
        if not mesh_ready() and prev:
            began = time.time()
            build_zone_mesh(case, prev, name, lat, lon, folder)
            _stamp(folder, name, f"zone-mesh {time.time() - began:.0f}s")
        if not mesh_ready():
            seconds = generate_mesh(spec_path, folder / f"{name}.grid.nc", mesh_receipt, logs / f"{name}-mesh.log", env)
            _stamp(folder, name, f"mesh {seconds:.0f}s")
        static_receipt = folder / f"{name}-static.json"
        static_rebuilt = mesh_rebuilt or not static_receipt.is_file() or not (folder / f"{name}.static.nc").is_file()
        if static_rebuilt:
            argv = [_tool("STATIC"), "--grid", folder / f"{name}.grid.nc", "--out", folder / f"{name}.static.nc",
                    "--receipt", static_receipt]
            if geog is not None:
                argv += ["--geog", Path(geog).expanduser().absolute()]
            # Sub-kilometre members: a cell narrower than one 30-arc-second pixel owns no sample and its terrain is
            # 0/0 (init then stops at "terrain contains non-finite values"; derecho 1 km member, 859 cells). The same
            # factor the point door passes (mesh_point.thirty_second_supersample, hex 51d84998); N = 1 above 1.31 km.
            from .mesh_point import thirty_second_supersample
            supersample = thirty_second_supersample(folder / f"{name}.grid.nc")
            if supersample > 1:
                argv += ["--supersample-30s", str(supersample)]
            seconds = _run_logged(argv, logs / f"{name}-static.log", env=env)
            _stamp(folder, name, f"static {seconds:.0f}s")
        init_receipt = folder / f"{name}-init.json"
        init = folder / f"{name}.init.nc"
        if static_rebuilt or not init_receipt.is_file() or not init.is_file():
            init.unlink(missing_ok=True)
            table = init_sources.load_table()
            row = init_sources.resolve_source(table, stage["source"])
            cycle = init_sources.Cycle.parse(stage["cycle"])
            argv = init_sources.init_argv(table, row, met=Path(stage["met"]), grid=folder / f"{name}.grid.nc",
                                          static=folder / f"{name}.static.nc", vertical_spec=vertical,
                                          out=init, cycle=cycle)
            seconds = _run_logged([sys.executable, "-m", "hexcore.cli", *argv], logs / f"{name}-init.log", env=env)
            init_receipt.write_text(json.dumps(dict(argv=argv, seconds=round(seconds, 1), source=stage["source"],
                                                    met=stage["met"], init_sha256=_sha256(init)), indent=2) + "\n")
            _stamp(folder, name, f"init {seconds:.0f}s")
        member = register_member(case, name, float(lat), float(lon), folder)
        _stamp(folder, name, "rows-done")
    log(json.dumps(member))
    return member


def prepare_case(case_path: Path, casedir: Path, *, geog: Path | None = None, threads: int | None = None,
                 tile_workers: int | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """The start member (and, for a corridor case, its frame and template) in CASEDIR."""

    case = load_case(case_path)
    casedir = Path(casedir).expanduser().absolute()
    read_stage(casedir)
    lat, lon = (float(v) for v in case["storm"]["initial_center_deg"])
    corridor = bool(case["mesh"].get("corridor"))
    if corridor:
        env = _tool_env(threads, tile_workers)
        for name, body in (("frame", frame_spec(case)), ("template", member_spec(case, "template", lat, lon))):
            spec_path = casedir / f"{name}.spec.json"
            if not spec_path.is_file():
                spec_path.write_text(json.dumps(body, indent=2) + "\n")
            if not (casedir / f"{name}-mesh.json").is_file() or not (casedir / f"{name}.grid.nc").is_file():
                log(f"prepare: {name} mesh")
                seconds = generate_mesh(spec_path, casedir / f"{name}.grid.nc", casedir / f"{name}-mesh.json",
                                        casedir / "logs" / f"{name}-mesh.log", env)
                _stamp(casedir, name, f"mesh {seconds:.0f}s")
    log("prepare: start member")
    return prepare_member(case_path, casedir, "start", lat, lon, casedir=casedir, stitched=corridor, geog=geog,
                          threads=threads, tile_workers=tile_workers, log=log)


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
def build_operator(source: Path, target: Path, out: Path) -> dict[str, Any]:
    from .moving_remap import ConservativeMap, SphericalCells

    from .stage_timing import stage

    began = time.monotonic()
    stages = []
    with stage(stages, "read-operator-geometry", parallelism={"device": "CPU", "threads": 1}):
        source_cells, target_cells = SphericalCells.read(source), SphericalCells.read(target)
    with stage(stages, "build-conservative-operator", parallelism={"device": "CPU"}) as timing:
        weights = ConservativeMap.build(source_cells, target_cells)
        execution = weights.receipt["overlap_execution"]
        timing["parallelism"] = {"device": "CPU", "executor": execution["executor"],
                                 "workers": execution["workers"],
                                 "scope": "overlap parallel; validation/reconciliation coordinator"}
    tmp = Path(f"{out}.{os.getpid()}.partial.npz")
    with stage(stages, "save-conservative-operator", parallelism={"device": "CPU", "threads": 1}):
        try:
            weights.save(tmp)
            tmp.replace(out)
        finally:
            tmp.unlink(missing_ok=True)
    # Keep execution evidence in the controller's log after disposable NPZs
    # are retired. This is the same receipt saved inside the operator archive.
    return dict(source=str(source), target=str(target), out=str(out), seconds=time.monotonic() - began,
                stages=stages, overlap_execution=execution)


def build_zone(source_grid, target_grid, source_static, target_static, source_init, target_init,
               out: Path) -> dict[str, Any]:
    from .moving_zone import ZoneMap

    began = time.monotonic()
    zone = ZoneMap.build(source_grid, target_grid, source_static=source_static, target_static=target_static,
                         source_init=source_init, target_init=target_init)
    tmp = Path(f"{out}.{os.getpid()}.partial.npz")
    zone.save(tmp)
    tmp.replace(out)
    return dict(out=str(out), seconds=time.monotonic() - began)
