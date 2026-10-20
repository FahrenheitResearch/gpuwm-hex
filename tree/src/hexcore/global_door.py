"""``gpuwm-hex global``: one command from a global row and a cycle to a scored,
drawn and exported forecast.

    gpuwm-hex global --row global-60km --cycle 2026-10-11T12 --work-dir RUN

THE STAGES, in order, each one an existing door this module composes and does
not reimplement:

  mesh      the NCAR-published quasi-uniform mesh the row names, fetched once
            into the mesh cache and pinned by the SHA-256 of the archive and
            of the grid file inside it;
  static    the native static for that grid, built once by rw_mpas_static with
            the native init_atmosphere static stream and the native GWD
            statistics, and registered as a runtime mesh row (bytes pinned) so
            the forecast door binds it like any registered mesh;
  init      ``init-source`` for the row's source (GFS by default) at the cycle;
  forecast  the forecast door, which picks how many cards the mesh needs from
            its own capacity rows and spreads it over them (``--devices``
            overrides);
  products  the standard product set through the Rust renderer;
  export    the history on a regular lat-lon grid, without drawing;
  score     the observation scorecard over the whole globe.

A stage with nothing to do says so: a cached mesh or static is skipped OUT
LOUD, with the digest that made the skip safe, because a quiet skip is how a
reader comes to believe a static was built when a stale one was used.

THE MACHINE SURFACE.  Every run writes, under ``--work-dir``:
``global-plan.json`` (the run plan: row, paths, every stage's argument vector,
the card estimate), ``status.json`` (atomic, one document) and
``events.jsonl`` (append-only), the contract in :mod:`hexcore.run_status`, and
``global-receipt.json`` at the end.  ``gpuwm-hex run-plan`` prints the plan
without running anything.

A new resolution, mesh or physics arm is a row of ``data/global-runs.v1.json``,
never a code path here.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from typing import Any, Mapping, Sequence
import urllib.request

from .errors import MpasPortError

DATA = Path(__file__).with_name("data")
TABLE = DATA / "global-runs.v1.json"
SCHEMA = "gpuwm-hex.global-runs/v1"
PLAN_SCHEMA = "gpuwm-hex.global-plan/v1"
RECEIPT_SCHEMA = "gpuwm-hex.global-run/v1"
FETCH_SCHEMA = "gpuwm-hex.global-mesh-fetch/v1"
STAGES = ("mesh", "static", "init", "forecast", "products", "export", "score")
CACHE_ENV = "GPUWM_HEX_GLOBAL_CACHE"
PENDING = "PENDING"


class GlobalDoorError(MpasPortError):
    """The global door refuses, and says why."""


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------
def load_table(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path) if path else TABLE
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise GlobalDoorError(
            f"{path} is not a {SCHEMA} document (schema {data.get('schema')!r}); its rows "
            f"cannot be read as stage bindings")
    ids = [row["id"] for row in data["rows"]]
    duplicate = sorted({name for name in ids if ids.count(name) > 1})
    if duplicate:
        raise GlobalDoorError(f"global rows {duplicate} occur more than once in {path}")
    for row in data["rows"]:
        if row["mesh"] not in data["meshes"]:
            raise GlobalDoorError(f"global row {row['id']!r} names mesh {row['mesh']!r}, which the table lacks")
        if row["physics"] not in data["physics"]:
            raise GlobalDoorError(f"global row {row['id']!r} names physics {row['physics']!r}, which the table lacks")
    data["_path"] = str(path)
    data["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return data


def resolve_row(table: Mapping[str, Any], name: str, physics: str | None = None
                ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = [row for row in table["rows"] if row["id"] == name]
    if len(rows) != 1:
        known = ", ".join(row["id"] for row in table["rows"])
        raise GlobalDoorError(f"no global row {name!r} (rows: {known})")
    row = dict(rows[0])
    if physics is not None:
        if physics not in table["physics"]:
            raise GlobalDoorError(
                f"--physics-row {physics!r} is not a physics row (rows: {', '.join(table['physics'])})")
        row["physics"] = physics
    return row, dict(table["meshes"][row["mesh"]]), dict(table["physics"][row["physics"]])


def engine_identities() -> frozenset[str]:
    """The algorithm identities of the importable engine that a step cap can depend on: its default
    YSU generation in the arithmetic mode this process selects (``GPUWM_WRF_EXACT``), and its
    cumulus schemes.  Empty when the engine publishes neither table.  The forecast door runs on the
    same engine, so these are the schemes the run takes; the door asks only to decide whether an
    arm's ``max_dt_seconds`` still binds."""

    found: set[str] = set()
    try:
        from gpuwm.wrf_exact import ENABLED
        from gpuwm.ysu_generations import YSU_DEFAULT_GENERATION, ysu_algorithm_identity
        found.add(str(ysu_algorithm_identity(YSU_DEFAULT_GENERATION, strict=bool(ENABLED))))
    except (ImportError, AttributeError, KeyError, ValueError):
        pass
    try:
        from gpuwm.checkpoint_identity import CUMULUS_ALGORITHM_IDENTITIES
        found.update(str(value) for value in CUMULUS_ALGORITHM_IDENTITIES.values())
    except (ImportError, AttributeError):
        pass
    return frozenset(found)


def row_step(mesh: Mapping[str, Any], physics: Mapping[str, Any], *,
             engine: frozenset[str] | set[str] = frozenset()) -> dict[str, Any]:
    """The forecast step a row runs: the mesh's own (MPAS's 6 s per km, rounded down to divide the
    radiation interval), unless the physics arm declares ``max_dt_seconds``. Then the largest step at
    or under that cap that divides the radiation interval, and the arm's measured reason with it.

    The cap is the physics arm's, not the mesh's: the mesh row is shared by every arm, and a step
    a scheme cannot take is a property of the scheme (``max_dt_why`` in the table).

    A cap dies with its defect: ``max_dt_retired_by_engine`` lists the engine algorithm identities
    that carry the fix, and the cap is retired when ``engine`` (:func:`engine_identities`) holds all
    of them.  Then the mesh's own step runs and the plan records which cap was retired and by what;
    otherwise the plan names the identities the engine lacks."""
    from .mpas_defaults import SECONDS_PER_KM, default_dt_seconds

    mesh_dt = float(default_dt_seconds(float(mesh["nominal_dx_m"]))["dt_seconds"])
    cap = physics.get("max_dt_seconds")
    if cap is None or mesh_dt <= float(cap):
        return {"dt_seconds": mesh_dt, "capped": False, "mesh_dt_seconds": mesh_dt, "source": "mesh"}
    required = sorted(str(value) for value in physics.get("max_dt_retired_by_engine") or ())
    if required and set(required) <= set(engine):
        return {"dt_seconds": mesh_dt, "capped": False, "mesh_dt_seconds": mesh_dt, "source": "mesh",
                "cap_retired": {"cap_seconds": float(cap), "engine_identities": required}}
    capped = float(default_dt_seconds(float(cap) * 1000.0 / SECONDS_PER_KM)["dt_seconds"])
    step = {"dt_seconds": capped, "capped": True, "mesh_dt_seconds": mesh_dt, "cap_seconds": float(cap),
            "source": "physics arm", "why": physics.get("max_dt_why")}
    if required:
        step["engine_lacks"] = sorted(set(required) - set(engine))
    return step


def default_step_cap(dt_selection: Mapping[str, Any], selectors: Mapping[str, Any], *,
                     radiation_seconds: float, engine: frozenset[str] | set[str] | None = None,
                     table: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The forecast door's DEFAULT step under the physics arm's long-step cap (:func:`row_step`).

    THE BREAKAGE THIS PREVENTS (hex 0.4.0 recut, 2026-10-16): only this door capped its rows, so a
    bare ``gpuwm-hex forecast`` on the 120 km or 60 km globe took the mesh's own step (600 s, 360 s)
    with YSU and showed the single boundary-layer-top layers 5 to 24 K hotter than their neighbours
    that the cap exists for (the ``arwen`` arm's ``max_dt_why``).  A YSU selection now takes the same
    cap on any door; Grell-Freitas with it takes the ``morrison`` arm's, which no engine bound retires.
    The cap dies with its defect exactly as here: an engine carrying every identity in the arm's
    ``max_dt_retired_by_engine`` runs the mesh's own step.  Only the default (``source == "mesh"``)
    is capped; an explicit ``--dt`` or a namelist ``config_dt`` is the caller's and runs as given."""
    from .mpas_defaults import SECONDS_PER_KM, default_dt_seconds

    selection = dict(dt_selection)
    if selection.get("source") != "mesh" or str(selectors.get("config_pbl_scheme", "")) != "bl_ysu":
        return selection
    table = load_table() if table is None else table
    arm_name = "morrison" if str(selectors.get("config_convection_scheme", "")) == "cu_grell_freitas" else "arwen"
    arm = table["physics"][arm_name]
    cap = arm.get("max_dt_seconds")
    mesh_dt = float(selection["dt_seconds"])
    if cap is None or mesh_dt <= float(cap):
        return selection
    required = sorted(str(value) for value in arm.get("max_dt_retired_by_engine") or ())
    engine = engine_identities() if engine is None else engine
    if required and set(required) <= set(engine):
        selection["cap_retired"] = {"cap_seconds": float(cap), "engine_identities": required, "arm": arm_name}
        return selection
    capped = default_dt_seconds(float(cap) * 1000.0 / SECONDS_PER_KM, radiation_seconds=float(radiation_seconds))
    capped.update(source="physics cap", mesh_dt_seconds=mesh_dt, cap_seconds=float(cap), arm=arm_name,
                  why=arm.get("max_dt_why"),
                  rule=f"the {arm_name} arm's {float(cap):g} s cap (data/global-runs.v1.json) under the mesh's "
                       f"own {mesh_dt:g} s, rounded down to divide the {float(radiation_seconds):g} s radiation interval")
    if required:
        capped["engine_lacks"] = sorted(set(required) - set(engine))
    return capped


def cache_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().absolute()
    env = os.environ.get(CACHE_ENV)
    if env:
        return Path(env).expanduser().absolute()
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "gpuwm-hex" / "global"


def runtime_mesh_name(mesh_id: str) -> str:
    """The runtime row name the door registers the pair under.  A runtime row
    may never shadow a shipped registry row (``x1.40962`` is one, pinned to
    the NCAR-published static), so the native-static pair gets its own name."""

    return f"hex-{mesh_id}-native"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
def _cycle(text: str) -> _dt.datetime:
    try:
        return _dt.datetime.strptime(text, "%Y-%m-%dT%H")
    except ValueError as error:
        raise GlobalDoorError(
            f"--cycle {text!r} must be YYYY-MM-DDTHH; it is the init's valid time") from error


def _visible_cards() -> list[int]:
    """Total memory of each visible card in bytes, read without CUDA (nvidia-smi),
    honouring CUDA_VISIBLE_DEVICES; [] when it cannot be read."""

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    cards, uuids = {}, {}
    for line in out.stdout.splitlines():
        try:
            index, uuid, mib = (part.strip() for part in line.split(","))
            cards[int(index)] = int(float(mib)) * 1024 * 1024
            uuids[uuid] = int(index)
        except ValueError:
            continue
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        chosen = []
        for token in visible.split(","):
            token = token.strip()
            if token.isdigit() and int(token) in cards:
                chosen.append(cards[int(token)])
            elif token.startswith("GPU-"):
                matching = [index for uuid, index in uuids.items() if uuid.startswith(token)]
                if len(matching) == 1:
                    chosen.append(cards[matching[0]])
        return chosen
    return [cards[key] for key in sorted(cards)]


def card_estimate(cells: int, devices: str | None) -> dict[str, Any]:
    """The forecast door's own card-count row for this mesh on this host."""

    if devices:
        return {"selected_cards": len([d for d in devices.split(",") if d.strip()]),
                "basis": "--devices names the cards"}
    memory = _visible_cards()
    if not memory:
        return {"selected_cards": None,
                "basis": "no card visible to nvidia-smi on this host; the forecast door decides at bind"}
    try:
        from .device_admission import select_card_count
        policy = select_card_count(int(cells), min(memory), len(memory))
    except Exception as error:  # the door re-decides at bind; the plan says why it could not
        return {"selected_cards": None, "basis": f"card-count rows did not answer here: {error}"}
    policy["card_memory_mib"] = [round(m / 1048576) for m in memory]
    return policy


def _visible_card_names() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()] if out.returncode == 0 else []


def wall_estimate(table: Mapping[str, Any], mesh_id: str, cards: int | None, hours: float,
                  card_names: Sequence[str] | None = None) -> dict[str, Any] | None:
    """A measured rate for this mesh, card count AND card, or None.  A rate
    measured on another part is not an estimate for this one (a 96 GB
    workstation card and a 16 GB consumer card differ by several times)."""

    names = [name.lower() for name in (card_names if card_names is not None else _visible_card_names())]
    rows = [r for r in table.get("measured_seconds_per_forecast_hour", [])
            if r["mesh"] == mesh_id and (cards is None or r["cards"] == cards)
            and any(str(r.get("card_match", r["card"])).lower() in name for name in names)]
    if not rows:
        return None
    row = rows[0]
    return {"seconds": round(row["seconds"] * hours, 1), "per_forecast_hour": row["seconds"],
            "measured_on": f"{row['cards']} x {row['card']}", "basis": row["basis"]}


def history_cadence(hours: float, explicit: int | None, row_default: int, steps_seconds: set[float],
                    row_id: str) -> dict[str, Any]:
    """The history cadence a run writes at, in minutes, and where it came from.

    THE BREAKAGE THIS PREVENTS (unified-image check, 2026-10-16): ``global --row global-60km --hours 1``
    handed the row's 180-minute default to the forecast door, which refused it (rc 2, "does not divide
    the 1.0 h run") only after the init had run, so no short run of a global row could go through.
    A cadence the caller did not set is fitted to the run: the row default when it divides the run,
    else the largest divisor of it that divides the run in whole steps, else the run length itself
    (one frame at the end).  A cadence the caller DID set is theirs: one that does not divide the run
    is refused here, before any stage runs, for the forecast door's reason."""

    run_seconds = float(hours) * 3600.0

    def fits(minutes: int) -> bool:
        seconds = minutes * 60.0
        return (minutes > 0 and abs(run_seconds / seconds - round(run_seconds / seconds)) < 1e-9
                and all(abs(seconds / dt - round(seconds / dt)) < 1e-9 for dt in steps_seconds))

    if explicit is not None:
        if not fits(int(explicit)):
            raise GlobalDoorError(
                f"--history-every-minutes {explicit} does not divide the {hours:g} h run of row {row_id} "
                f"in whole {' / '.join(f'{dt:g}' for dt in sorted(steps_seconds))} s steps; the last frame "
                "would fall inside the run rather than at its end, so the output would silently not cover "
                "the length asked for.  Leave it unset and the door picks a cadence that divides the run.")
        return {"minutes": int(explicit), "source": "--history-every-minutes", "row_default": row_default}
    if fits(row_default):
        return {"minutes": row_default, "source": "row default", "row_default": row_default}
    for minutes in range(row_default - 1, 0, -1):
        if row_default % minutes == 0 and fits(minutes):
            return {"minutes": minutes, "row_default": row_default,
                    "source": f"largest divisor of the row's {row_default} min that divides the {hours:g} h run"}
    run_minutes = run_seconds / 60.0
    if abs(run_minutes - round(run_minutes)) < 1e-9 and fits(int(round(run_minutes))):
        return {"minutes": int(round(run_minutes)), "row_default": row_default,
                "source": f"the run length (one frame at the end): no divisor of {row_default} min divides it"}
    # Not a whole number of steps at all: the forecast door refuses that by name.
    return {"minutes": row_default, "source": "row default", "row_default": row_default}


def plan(args: argparse.Namespace, table: Mapping[str, Any] | None = None) -> dict[str, Any]:
    table = table or load_table(args.table)
    cycle = _cycle(args.cycle)
    row, mesh, physics = resolve_row(table, args.row, args.physics_row)
    hours = float(args.hours if args.hours is not None else row.get("hours", 24))
    step = row_step(mesh, physics, engine=engine_identities())
    cadence = history_cadence(hours, args.history_every_minutes, int(row.get("history_every_minutes", 180)),
                              {float(step["dt_seconds"]), float(step["mesh_dt_seconds"])}, row["id"])
    every = cadence["minutes"]
    work = Path(args.work_dir).expanduser().absolute()
    cache = cache_root(args.cache_dir)
    mesh_dir = cache / "meshes" / row["mesh"]
    grid = mesh_dir / mesh["grid_member"]
    static = mesh_dir / f"{row['mesh']}.native.static.nc"
    rows_file = mesh_dir / "mesh-rows.json"
    vertical = (DATA / table["vertical_spec"]).resolve()
    source = args.source
    init_dir = work / "init"
    init = init_dir / "init.nc"
    forecast = work / "forecast"
    start = cycle.strftime("%Y-%m-%d_%H:00:00")
    common = ["--grid", str(grid), "--static", str(static)]
    init_argv = ["init-source", "--source", source, "--cycle", args.cycle, "--out-dir", str(init_dir),
                 *common, "--vertical-spec", str(vertical)]
    for staged in args.staged or []:
        init_argv += ["--staged", str(staged)]
    if getattr(args, "member", None) is not None:
        if getattr(args, "members", None) or getattr(args, "lagged", None) is not None:
            raise GlobalDoorError("--member runs one member and --members an ensemble; pass one of them")
        init_argv += ["--member", str(args.member)]
    forecast_argv = ["forecast", "--mesh", runtime_mesh_name(row["mesh"]), *common, "--init", str(init),
                     "--start-time", start,
                     "--init-source", f"{source} {args.cycle}"
                     + (f" member {args.member}" if getattr(args, "member", None) is not None else "")
                     + f"; global row {row['id']}",
                     "--hours", f"{hours:g}", "--history-every-minutes", str(every),
                     "--out", str(forecast), "--scratch", str(work / "scratch"),
                     "--case-label", f"{row['id']}-{cycle.strftime('%Y%m%d%H')}", "--no-products"]
    for key, value in physics["overrides"].items():
        forecast_argv += ["--physics", f"{key}={value}"]
    if step["capped"]:
        # Only a binding cap is passed: an uncapped row keeps the forecast door's own default.
        # A --forecast-arg=--dt=... given by the caller comes later and wins.
        forecast_argv += ["--dt", f"{step['dt_seconds']:g}"]
    if args.devices:
        forecast_argv += ["--samehost-devices", args.devices]
    if getattr(args, "stochastic", None) is not None:
        forecast_argv += ["--stochastic", args.stochastic]
    for extra in args.forecast_arg or []:
        forecast_argv.append(extra)
    products_argv = ["products", str(forecast), "--out", str(work / "products")]
    export_argv = ["export", str(forecast), "--out", str(work / "export"), "--window", "global",
                   "--field-set", args.export_field_set, "--overwrite"]
    score_config = work / "observation-scorecard.config.json"
    score_argv = ["score", str(forecast), "--config", str(score_config)]
    cards = card_estimate(mesh["cells"], args.devices)
    result = {
        "schema": PLAN_SCHEMA,
        "row": row, "mesh": {"id": row["mesh"], **mesh}, "physics": {"row": row["physics"], **physics},
        "table": {"path": table["_path"], "sha256": table["_sha256"]},
        "cycle": args.cycle, "start_time": start, "hours": hours, "history_every_minutes": every,
        "history_cadence": cadence,
        "source": source,
        "paths": {"work": str(work), "cache": str(cache), "grid": str(grid), "static": str(static),
                  "mesh_rows": str(rows_file), "vertical_spec": str(vertical), "init": str(init),
                  "forecast": str(forecast), "products": str(work / "products"),
                  "export": str(work / "export"), "score_config": str(score_config)},
        "static_argv": ["--grid", str(grid), "--out", str(static), *table["static"]["argv"],
                        *(["--geog", str(Path(args.geog).expanduser().absolute())] if args.geog else []),
                        "--receipt", str(mesh_dir / "static-receipt.json")],
        "commands": {"init": init_argv, "forecast": forecast_argv, "products": products_argv,
                     "export": export_argv, "score": score_argv},
        # The scorer reads init and start from the forecast receipt when it can; a run stopped before
        # writing one (a killed or timed-out forecast still leaves its frames) would otherwise score nothing.
        "score_config": {**dict(table.get("score", {})), "init": str(init), "grid": str(grid),
                         "mesh": runtime_mesh_name(row["mesh"]), "start": start},
        "step": step,
        "cards": cards,
        "wall_estimate": wall_estimate(table, row["mesh"], cards.get("selected_cards"), hours),
        "stages": list(STAGES),
    }
    if getattr(args, "members", None) or getattr(args, "lagged", None):
        from .global_ensemble import member_plan, parse_members
        from .init_sources import load_table as load_sources, resolve_source

        from .lagged_members import parse_lags
        ensemble_source = bool(resolve_source(load_sources(None), source).get("members"))
        if not ensemble_source and not getattr(args, "lagged", None):
            raise GlobalDoorError(f"--members needs an ensemble source; {source!r} publishes one state")
        if ensemble_source and not args.members:
            raise GlobalDoorError("an ensemble source needs --members with --lagged")
        if not ensemble_source and args.members not in (None, "0"):
            raise GlobalDoorError("a deterministic source has one state per lag; omit --members or use 0")
        result["members"] = member_plan(result, parse_members(args.members or "0"),
                                         parse_lags(args.lagged) if args.lagged is not None else None)
        from .stochastic import member_number, member_options
        options = {} if getattr(args, 'stochastic', None) is None else json.loads(args.stochastic)
        if 'member' in options:
            raise GlobalDoorError('ensemble stochastic member identity comes from the source roster')
        seed = options.pop('seed', getattr(args, 'stochastic_seed', 53))
        for member in result['members']:
            identity = member['member'] if member['member'] is not None and not member['provenance']['lag_hours'] else member_number(member['run_id'])
            member['stochastic'] = member_options(identity, seed=seed,
                source_members=len({item['member'] for item in result['members']}), ensemble_size=len(result['members']), **options)
    return result


# ---------------------------------------------------------------------------
# the stages
# ---------------------------------------------------------------------------
def _download(url: str, target: Path, status, expected_bytes: int | None) -> None:
    partial = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "gpuwm-hex"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
        total = int(response.headers.get("Content-Length") or expected_bytes or 0)
        done, last = 0, 0.0
        while True:
            block = response.read(1 << 22)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total and time.monotonic() - last > 5.0:
                status.step(int(100 * done / total), note=f"{done} of {total} bytes")
                last = time.monotonic()
    os.replace(partial, target)


def stage_mesh(value: Mapping[str, Any], status) -> dict[str, Any]:
    mesh = value["mesh"]
    grid = Path(value["paths"]["grid"])
    for key in ("archive_sha256", "grid_sha256"):
        if mesh.get(key) in (None, "", PENDING):
            raise GlobalDoorError(
                f"mesh {mesh['id']} has no pinned {key} in the global table; an unpinned archive could "
                f"be any bytes, so the door does not fetch it. Pin it in data/global-runs.v1.json")
    if grid.is_file() and sha256_file(grid) == mesh["grid_sha256"]:
        status.skip("mesh", f"the cache holds {grid.name} with the pinned sha256 {mesh['grid_sha256'][:16]}")
        return {"grid": str(grid), "cached": True}
    archive = grid.parent / Path(mesh["url"]).name
    if not (archive.is_file() and sha256_file(archive) == mesh["archive_sha256"]):
        status.stage("mesh", step_count=100)
        status.note(f"fetching {mesh['url']}")
        _download(mesh["url"], archive, status, mesh.get("archive_bytes"))
        got = sha256_file(archive)
        if got != mesh["archive_sha256"]:
            archive.unlink()
            raise GlobalDoorError(
                f"{mesh['url']} has sha256 {got}, not the pinned {mesh['archive_sha256']}; the archive "
                f"is not the published mesh and was removed")
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember(mesh["grid_member"])
        if not member.isfile():
            raise GlobalDoorError(f"{archive.name} holds {mesh['grid_member']} but not as a file")
        with tar.extractfile(member) as source, grid.with_name(grid.name + ".part").open("wb") as out:
            shutil.copyfileobj(source, out, 1 << 22)
    os.replace(grid.with_name(grid.name + ".part"), grid)
    got = sha256_file(grid)
    if got != mesh["grid_sha256"]:
        grid.unlink()
        raise GlobalDoorError(
            f"{mesh['grid_member']} inside the pinned archive has sha256 {got}, not the pinned "
            f"{mesh['grid_sha256']}; the grid was removed")
    receipt = {"schema": FETCH_SCHEMA, "url": mesh["url"], "archive": str(archive),
               "archive_sha256": mesh["archive_sha256"], "grid": str(grid), "grid_sha256": got,
               "fetched_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()}
    (grid.parent / "fetch-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    archive.unlink()
    return {"grid": str(grid), "cached": False}


def _static_receipt_matches(static: Path, receipt_path: Path, grid_sha: str, argv: Sequence[str]) -> bool:
    try:
        receipt = json.loads(receipt_path.with_name("global-static.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (static.is_file() and receipt.get("grid_sha256") == grid_sha
            and receipt.get("static_sha256") == sha256_file(static)
            and receipt.get("flags") == [a for a in argv if a.startswith("--native")])


def stage_static(value: Mapping[str, Any], status) -> dict[str, Any]:
    from . import engines

    paths = value["paths"]
    grid, static, rows_file = Path(paths["grid"]), Path(paths["static"]), Path(paths["mesh_rows"])
    argv = list(value["static_argv"])
    receipt_path = Path(argv[argv.index("--receipt") + 1])
    grid_sha = value["mesh"]["grid_sha256"]
    if not _static_receipt_matches(static, receipt_path, grid_sha, argv):
        status.stage("static")
        try:
            exe = engines.resolve(engines.STATIC)
        except engines.EngineRefusal as error:
            raise GlobalDoorError(str(error)) from None
        probe = subprocess.run([str(exe), "--help"], capture_output=True, text=True)
        missing = [flag for flag in argv if flag.startswith("--native") and flag not in probe.stdout + probe.stderr]
        if missing:
            raise GlobalDoorError(
                f"{exe} does not know {missing}: it predates the native static builder, and a static "
                f"built without the native stream is not the static this row declares. Use a gpuwm "
                f"engine whose rw_mpas_static carries the native path (or set GPUWM_RW_MPAS_STATIC)")
        status.note(f"building the native static with {exe}")
        if static.exists():
            static.unlink()
        started = time.monotonic()
        result = subprocess.run([str(exe), *argv], capture_output=True, text=True)
        (static.parent / "static.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise GlobalDoorError(
                f"rw_mpas_static exited {result.returncode} (log {static.parent / 'static.log'}):\n"
                f"{engines.refusal_text(result.stderr)}")
        (receipt_path.with_name("global-static.json")).write_text(json.dumps({
            "builder": str(exe), "builder_sha256": sha256_file(exe), "argv": argv,
            "flags": [a for a in argv if a.startswith("--native")], "grid_sha256": grid_sha,
            "static_sha256": sha256_file(static), "seconds": round(time.monotonic() - started, 1),
        }, indent=2) + "\n", encoding="utf-8")
    else:
        status.skip("static", f"the cache holds {static.name}, built natively from grid "
                              f"{grid_sha[:16]} (global-static.json)")
    _register(value, grid, static, rows_file, receipt_path)
    return {"static": str(static), "mesh_rows": str(rows_file)}


def _register(value: Mapping[str, Any], grid: Path, static: Path, rows_file: Path, receipt_path: Path) -> None:
    from . import mesh_rows
    from .mpas_defaults import default_dt_seconds

    mesh = value["mesh"]
    dx = float(mesh["nominal_dx_m"])
    dims = mesh_rows._dimensions(grid)
    row = mesh_rows.MeshRow(
        kind=mesh_rows.ARCHIVED_GLOBAL_KIND, name=runtime_mesh_name(mesh["id"]),
        n_cells=int(dims["nCells"]), n_edges=int(dims["nEdges"]), n_levels=55, n_interfaces=56,
        n_soil_levels=4, nominal_dx_m=dx, dt_seconds=float(default_dt_seconds(dx)["dt_seconds"]),
        grid=str(grid), grid_bytes=grid.stat().st_size, grid_sha256=mesh["grid_sha256"],
        static=str(static), static_bytes=static.stat().st_size, static_sha256=sha256_file(static),
        spec_sha256=mesh["archive_sha256"], generator_receipt=str(grid.parent / "fetch-receipt.json"),
        static_receipt=str(receipt_path), point_deg=(0.0, 0.0), fine_dx_m=dx, core_radius_km=0.0,
        background_km=dx / 1000.0,
        admission={"scope": "global row: NCAR-published quasi-uniform grid, native static; every "
                            "geometry and Courant admission is measured again at forecast bind",
                   "url": mesh["url"]})
    mesh_rows.write_rows(rows_file, [row])


def _init_matches(value: Mapping[str, Any]) -> bool:
    init = Path(value["paths"]["init"])
    try:
        receipt = json.loads((init.parent / "init-source-receipt.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    digits = "".join(ch for ch in str(receipt.get("cycle", "")) if ch.isdigit())
    return (init.is_file()
            and (receipt.get("init") or {}).get("path") == str(init)
            and str(receipt.get("source")) == str(value["source"])
            and digits.startswith(value["cycle"].replace("-", "").replace("T", "")))


def _run_door(argv: Sequence[str], env: Mapping[str, str]) -> int:
    from .cli import main

    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        return int(main(list(argv)))
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _stage_list(args: argparse.Namespace) -> list[str]:
    if args.stage == "all":
        chosen = list(STAGES)
    else:
        chosen = [args.stage]
    skip = set(args.skip or [])
    unknown = skip - set(STAGES)
    if unknown:
        raise GlobalDoorError(f"--skip {sorted(unknown)} names no stage (stages: {', '.join(STAGES)})")
    return [stage for stage in chosen if stage not in skip]


def run_global(args: argparse.Namespace) -> int:
    from .run_status import StatusWriter

    if args.list:
        table = load_table(args.table)
        if args.json:
            print(json.dumps({"rows": table["rows"], "physics": table["physics"],
                              "meshes": table["meshes"]}, indent=2))
        else:
            for row in table["rows"]:
                mesh = table["meshes"][row["mesh"]]
                print(f"{row['id']:24s} {row['mesh']:12s} {mesh['cells']:>9,d} cells  physics={row['physics']:9s} {row.get('about', '')}")
        return 0
    for flag in ("row", "cycle", "work_dir"):
        if getattr(args, flag) is None:
            raise GlobalDoorError(f"--{flag.replace('_', '-')} is required")
    from .global_cycle import resolve_cycle_argument

    # --cycle latest: the newest due cycle of the source, waited for until it posts (a plan never waits).
    resolution = resolve_cycle_argument(args, say=lambda text: print(f"[global] {text}", flush=True),
                                        wait=args.stage != "plan")
    value = plan(args)
    if resolution is not None:
        value["cycle_resolution"] = resolution
    if args.stage == "plan":
        print(json.dumps(value, indent=2))
        return 0
    work = Path(value["paths"]["work"])
    work.mkdir(parents=True, exist_ok=True)
    (work / "global-plan.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    stages = _stage_list(args)
    status = StatusWriter(work, "global", stages=tuple(stages), row=value["row"]["id"])
    if status.log_path is not None:
        print(f"log: {status.log_path}", flush=True)
    receipt: dict[str, Any] = {"schema": RECEIPT_SCHEMA, "plan": str(work / "global-plan.json"),
                               "row": value["row"]["id"], "cycle": value["cycle"], "stages": {}}
    rows_env = os.pathsep.join(filter(None, [value["paths"]["mesh_rows"], os.environ.get("GPUWM_HEX_MESH_ROWS")]))
    env = {"GPUWM_HEX_MESH_ROWS": rows_env}
    # ONE scorer per run. The forecast door queues the automatic scorer when the forecast ends; it
    # reads GPUWM_HEX_OBSERVATION_CONFIG, so it scores with this row's config (the whole globe), and
    # the score stage below joins it instead of scoring the run a second time. Before this the hook
    # never saw the row's config (it scored the CONUS default) and the stage then scored the globe again,
    # over 30 min of single-process scoring per 24 h run (hex global on 1-2 cards, 2026-10-16).
    score_config = Path(value["paths"]["score_config"])
    if "forecast" in stages or "score" in stages:
        config = dict(value["score_config"])
        if args.observation_config:
            config.update(json.loads(Path(args.observation_config).read_text(encoding="utf-8")))
            config["region"] = value["score_config"].get("region", config.get("region"))
        score_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if score_config.is_file():
        env["GPUWM_HEX_OBSERVATION_CONFIG"] = str(score_config)
    try:
        if value.get("members"):
            from .global_ensemble import run_ensemble

            for name in [stage for stage in stages if stage in ("mesh", "static")]:
                started = time.monotonic()
                result = stage_mesh(value, status) if name == "mesh" else stage_static(value, status)
                receipt["stages"][name] = {**result, "seconds": round(time.monotonic() - started, 1)}
            ensemble = run_ensemble(value, args, [s for s in stages if s not in ("mesh", "static")], status, env)
            receipt["stages"].update(ensemble["stages"])
            receipt["ensemble"] = str(work / "ensemble.json")
            status.done(f"{value['row']['id']} {value['cycle']}: {len(value['members'])} members, {', '.join(stages)}")
            _write_receipt(work, receipt)
            return 0
        for name in stages:
            started = time.monotonic()
            if name == "mesh":
                result = stage_mesh(value, status)
            elif name == "static":
                result = stage_static(value, status)
            else:
                if name == "init" and _init_matches(value) and not args.rebuild_init:
                    status.skip("init", f"{value['paths']['init']} already holds {value['source']} {value['cycle']}")
                    receipt["stages"][name] = {"skipped": True}
                    continue
                if name == "score":
                    from .observation_scorecard import completed_scorecard, join_automatic_scorer
                    joined = (completed_scorecard(value["paths"]["forecast"], score_config)
                              or join_automatic_scorer(value["paths"]["forecast"], score_config))
                    if joined is not None:
                        status.skip("score", joined["note"])
                        receipt["stages"][name] = {**joined, "seconds": round(time.monotonic() - started, 1)}
                        continue
                status.stage(name, eta_s=(value["wall_estimate"] or {}).get("seconds") if name == "forecast" else None)
                status.note("gpuwm-hex " + " ".join(value["commands"][name]))
                rc = _run_door(value["commands"][name], env)
                result = {"returncode": rc}
                if name == "score" and rc == 3:
                    completed = completed_scorecard(value["paths"]["forecast"], score_config)
                    if completed is not None:
                        # Exit 3 reports incomplete coverage, not a failed forecast or scorer.
                        # Preserve that coverage status and the standalone command's exit code.
                        result = {**completed, "scorer_returncode": rc}
                        status.note(completed["note"])
                        rc = 0
                if rc != 0:
                    receipt["stages"][name] = {**result, "seconds": round(time.monotonic() - started, 1)}
                    status.failed(f"the {name} stage exited {rc}")
                    _write_receipt(work, receipt)
                    return rc
            receipt["stages"][name] = {**result, "seconds": round(time.monotonic() - started, 1)}
        status.done(f"{value['row']['id']} {value['cycle']}: {', '.join(stages)}")
        _write_receipt(work, receipt)
        return 0
    except MpasPortError as error:
        status.refused(str(error))
        receipt["refused"] = str(error)
        _write_receipt(work, receipt)
        raise
    except Exception as error:
        status.failed(f"{type(error).__name__}: {error}")
        receipt["failed"] = f"{type(error).__name__}: {error}"
        _write_receipt(work, receipt)
        raise


def _write_receipt(work: Path, receipt: Mapping[str, Any]) -> None:
    (work / "global-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")


def run_plan_command(args: argparse.Namespace) -> int:
    args.stage = "plan"
    args.list = False
    return run_global(args)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--row", default=None, help="a row of data/global-runs.v1.json (see --list)")
    parser.add_argument("--cycle", default=None,
                        help="YYYY-MM-DDTHH, the init cycle; or latest: the source's newest due cycle, "
                             "waited for until it posts (see --max-wait-minutes)")
    parser.add_argument("--work-dir", type=Path, default=None, help="the run folder")
    parser.add_argument("--source", default="gfs", help="init-source row (default gfs; see init-source --list)")
    parser.add_argument("--member", type=int, default=None, help="one member of an ensemble source (e.g. gefs)")
    parser.add_argument("--physics-row", default=None, help="override the row's physics arm (arwen, morrison)")
    parser.add_argument("--hours", type=float, default=None, help="forecast length (default: the row's)")
    parser.add_argument("--history-every-minutes", type=int, default=None)
    parser.add_argument("--devices", default=None, metavar="D0,D1,...",
                        help="cards to run on (default: the forecast door's own count for this mesh)")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help=f"mesh and static cache (default ${CACHE_ENV}, else ~/.cache/gpuwm-hex/global)")
    parser.add_argument("--geog", type=Path, default=None,
                        help="WPS_GEOG root for the static (default: rw_mpas_static's own, which "
                             "`gpuwm fetch-geog` stages)")
    parser.add_argument("--staged", type=Path, action="append", default=[],
                        help="directory already holding the source's files (init-source --staged)")
    parser.add_argument("--observation-config", type=Path, default=None,
                        help="observation scorecard overrides (the region stays the row's)")
    parser.add_argument("--export-field-set", default="surface", choices=("surface", "full"))
    parser.add_argument("--forecast-arg", action="append", default=[], metavar="ARG",
                        help="one extra forecast-door argument, repeatable (e.g. --forecast-arg=--checkpoint-hours=6)")
    parser.add_argument("--table", type=Path, default=None, help="alternative table (tests)")
    from .global_cycle import add_cycle_arguments
    from .global_ensemble import add_ensemble_arguments

    add_cycle_arguments(parser)
    add_ensemble_arguments(parser)


def add_global_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "global", help="one command: global mesh, native static, init, forecast, products, export, score",
        description=__doc__.split("\n\n")[0])
    _add_common(parser)
    parser.add_argument("--stage", default="all", choices=("plan", "all", *STAGES),
                        help="plan prints the run plan; all (default) runs every stage in order")
    parser.add_argument("--skip", action="append", default=[], metavar="STAGE",
                        help="leave one stage out of 'all', repeatable")
    parser.add_argument("--rebuild-init", action="store_true", help="rebuild the init even when present")
    parser.add_argument("--list", action="store_true", help="print the rows and exit")
    parser.add_argument("--json", action="store_true", help="with --list, print JSON")
    parser.set_defaults(handler=run_global)


def run_sources(args: argparse.Namespace) -> int:
    """What a workspace can offer: every init source and every global row."""

    from .init_sources import load_table as load_sources

    sources = load_sources(None)
    table = load_table(args.table)
    value = {
        "schema": "gpuwm-hex.sources/v1",
        "init_sources": [{"id": row["id"], "display_name": row.get("display_name"),
                          "aliases": row.get("aliases", []), "role": row.get("role"),
                          "status": row.get("status", "runnable"),
                          "members": bool(row.get("members"))} for row in sources["sources"]],
        "global_rows": [{"id": row["id"], "mesh": row["mesh"], "cells": table["meshes"][row["mesh"]]["cells"],
                         "nominal_dx_m": table["meshes"][row["mesh"]]["nominal_dx_m"],
                         "physics": row["physics"], "hours": row.get("hours"), "about": row.get("about")}
                        for row in table["rows"]],
        "physics_rows": {name: physics.get("about") for name, physics in table["physics"].items()},
    }
    if args.json:
        print(json.dumps(value, indent=2))
    else:
        for row in value["init_sources"]:
            print(f"source  {row['id']:10s} {row['status']:9s} {row['display_name']}")
        for row in value["global_rows"]:
            print(f"global  {row['id']:24s} {row['mesh']:12s} physics={row['physics']}")
    return 0


def add_sources_parser(commands: Any) -> None:
    parser = commands.add_parser("sources", help="list init sources and global rows (--json for machines)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--table", type=Path, default=None, help="alternative global table (tests)")
    parser.set_defaults(handler=run_sources)


def add_run_plan_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "run-plan", help="print a global row's run plan as JSON (paths, commands, card estimate) and exit")
    _add_common(parser)
    parser.set_defaults(handler=run_plan_command, skip=[], rebuild_init=False, json=True)
