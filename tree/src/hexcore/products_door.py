"""The product door: ``gpuwm-hex products``.

One command after a forecast -- and, by default, the last stage of the
forecast door itself -- turns a run's history on ANY mesh (uniform global,
variable-resolution, refined patch) into the standard product set:

* maps through the Rust renderer, every view filed
  ``<out>/<case>/<domain>/<product>/<valid-day>/``:
  the whole sphere at the mesh's own spacing, the mesh's refined core at its
  own spacing (and a context view around it) when the mesh has one, a
  default region when it does not, and any ``--region`` asked for;
* soundings (skew-T log-p) and meteograms at points, filed
  ``<out>/<case>/points/<product>/<day>/``.

Python here is orchestration only (the project's boundary for 2.5.0): every
field the pictures show is read, remapped and drawn by the Rust binaries
``rw_mpas_convert`` (``--remap product``: linear state, area-conservative
precipitation), ``rw_wrfbatch`` and ``rw_hex_points``.  This module chooses
views, names folders and writes one manifest.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as _dt
import html as _html
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time

from .errors import MpasPortError
from . import engines
from . import render_door


class ProductDoorError(MpasPortError):
    """A product-door refusal; the message names the breakage."""


MANIFEST_SCHEMA = "gpuwm-hex.products-manifest/v1"

#: The standard map set.  The renderer's own slugs: a member a history
#: cannot draw is named in the manifest with the catalog's reason, never
#: dropped silently (``render_door`` standard-set mode).
GLOBAL_PRODUCTS = (
    "2m_temperature_10m_winds",
    "10m_wind_speed_and_direction",
    "mslp_10m_winds",
    "500mb_absolute_vorticity_height_winds",
    "250mb_height_winds",
    "total_qpf",
    "precipitable_water",
    # The model-column vapour transport this door's history feeds.  THE BREAKAGE THIS PREVENTS (rolling 2.9.4
    # run, hex global-60km, 2026-10-21): the renderer's plain "ivt" became the 1000-300 hPa product of the
    # isobaric lanes when the capability landing joined both, and a hex history stores no isobaric IVT
    # ("not stored: ivt_1000_300_entire_atmosphere"), so the default IVT map was never drawn.
    "ivt_column",
    "ivt_vectors",
    "isotherm_height_0c",
    "300mb_height_winds",
    "simulated_ir_satellite",
    "sbcape",
    "composite_reflectivity",
    "precipitation_type",
)
#: Interval products: each is the difference (or the extreme) of several
#: stored frames, so each needs the frames its window spans.  The value is
#: (the spacing in hours every stored frame inside the window must keep, the
#: run length in hours the window needs): ``qpf_1h`` and the run/daily extremes
#: need every hour stored, ``qpf_6h`` every 6 h, ``qpf_24h`` a day of run.
#: THE BREAKAGE THIS PREVENTS (rolling 2.9.4 run, hex global-60km 24 h,
#: 2026-10-21): the default set asked for hourly windows from the row's
#: 3-hourly history, so every global run reported qpf_1h, 10m_wind_run_max
#: and the 0-24 h temperature extremes unavailable.  The door now requests a
#: window only when the stored frames can close it (windowed_for_history) and
#: names every window it did not request, with the cadence that rules it out,
#: in the manifest; the global rows store hourly where that is cheap.
WINDOWED_REQUIREMENTS = {
    "qpf_1h": (1, 1),
    "qpf_6h": (6, 6),
    "qpf_24h": (24, 24),
    "10m_wind_run_max": (1, 1),
    "2m_temp_0_24h_max": (1, 24),
    "2m_temp_0_24h_min": (1, 24),
}
WINDOWED_PRODUCTS = tuple(WINDOWED_REQUIREMENTS)
BASE_GLOBAL_PRODUCTS = GLOBAL_PRODUCTS
REGIONAL_EXTRA_PRODUCTS = (
    "2m_dewpoint_10m_winds",
    "500mb_height_winds",
    "850mb_temperature_height_winds",
    "850mb_height_winds",
    "925mb_height_winds",
    "mucape",
    "cloud_cover_levels",
)
GLOBAL_PRODUCTS = BASE_GLOBAL_PRODUCTS + WINDOWED_PRODUCTS
REGIONAL_PRODUCTS = GLOBAL_PRODUCTS + REGIONAL_EXTRA_PRODUCTS


def windowed_for_history(histories: list[Path], start: str) -> tuple[tuple[str, ...], dict[str, str]]:
    """The window products the stored frames can close, and why each other one is not requested.

    A frame's lead hour is its valid time minus the run start.  A window with spacing S and length L is
    requested when the run reaches L hours and every multiple of S up to the last stored lead is stored,
    so the renderer's every-hour / every-6-hour differencing never meets a gap.  Frames without a
    valid-time stamp leave the full set requested (the renderer names its own reasons).
    """

    parse = _parse_stamp
    stamps = [frame_stamp(path) for path in histories]
    if not histories or any(stamp is None for stamp in stamps):
        return WINDOWED_PRODUCTS, {}
    origin = parse(start)
    leads = {round((parse(stamp) - origin).total_seconds() / 3600.0, 6) for stamp in stamps}
    whole = {int(lead) for lead in leads if float(lead).is_integer() and lead >= 0}
    end = max(whole) if whole else 0
    requested, withheld = [], {}
    for slug, (spacing, length) in WINDOWED_REQUIREMENTS.items():
        missing = [h for h in range(0, end + 1, spacing) if h not in whole]
        if end < length:
            withheld[slug] = f"the run's stored frames reach F{end:03d}; this window needs {length} h"
        elif missing:
            withheld[slug] = (f"the history does not store every {spacing} h (first missing F{missing[0]:03d}); "
                              "store hourly frames (--history-every-minutes 60) to draw it")
        else:
            requested.append(slug)
    return tuple(requested), withheld

#: Where a mesh WITHOUT a refined region is also drawn regionally.  A view,
#: not a case: ground extents in km, dx from the mesh cells inside it.
DEFAULT_REGIONS = {
    "north-america": "region:40,-97,7000,4600",
    "europe": "region:50,12,5200,3800",
}

#: Points a sounding and meteogram are drawn at when none are named.  A
#: refined mesh takes those inside its refined core; a uniform mesh takes the
#: first ``DEFAULT_GLOBAL_POINT_COUNT``.
DEFAULT_POINTS = (
    ("Oklahoma City", 35.47, -97.52),
    ("Wichita", 37.69, -97.34),
    ("Dodge City", 37.76, -99.97),
    ("Kansas City", 39.10, -94.58),
    ("Amarillo", 35.22, -101.83),
    ("Tulsa", 36.15, -95.99),
    ("Denver", 39.74, -104.99),
    ("Chicago", 41.88, -87.63),
    ("Miami", 25.76, -80.19),
    ("London", 51.51, -0.13),
    ("Tokyo", 35.68, 139.69),
    ("Sydney", -33.87, 151.21),
    ("Sao Paulo", -23.55, -46.63),
    ("Nairobi", -1.29, 36.82),
    ("New Delhi", 28.61, 77.21),
)
DEFAULT_GLOBAL_POINT_COUNT = 6
EARTH_RADIUS_KM = 6371.229

POINTS_ENV = engines.POINTS.env_names[0]


def add_label_arguments(parser) -> None:
    """``--model-label`` and ``--brand`` on a door that ends in the product
    stage, forwarded to ``gpuwm-hex products`` by :func:`label_argv`.

    THE BREAKAGE THIS PREVENTS (WOOF sounding labels, 2026-10-23): the
    products door took ``--model-label`` but ``gpuwm-hex global`` (and the
    region, variable-resolution, storm-following and ensemble doors) had no
    way to pass it, so every point sounding a host launched through a door
    said "hex 121 km" top left with nothing top right, whatever the host
    called the model.  Unset, nothing is forwarded and the products door
    keeps its own defaults."""

    parser.add_argument(
        "--model-label", default=None, metavar="NAME",
        help="the model the maps and soundings name, top left (e.g. 'WOOF Hex'; "
             "the mesh spacing is added); forwarded to the product stage. Default: hex")
    parser.add_argument(
        "--brand", default=None, metavar="TEXT",
        help="bold text at the top right of every sounding and meteogram (e.g. "
             "'WOOF Hex · Global 120 km · Init GFS 22 Oct 2026 00Z'); forwarded "
             "to the product stage. Default: none")


def label_argv(args) -> list[str]:
    """The ``gpuwm-hex products`` flags a door forwards for its labels."""

    argv: list[str] = []
    if getattr(args, "model_label", None):
        argv += ["--model-label", str(args.model_label)]
    if getattr(args, "brand", None):
        argv += ["--brand", str(args.brand)]
    return argv


def points_take_brand(points_exe: Path) -> bool:
    """Whether this ``rw_hex_points`` draws ``--brand``.

    THE BREAKAGE THIS PREVENTS: a ``rw_hex_points`` built before the flag
    refuses any flag it does not know ("unknown flag --brand"), so passing
    it unconditionally would cost every point picture on an older engine.
    Such a build draws without the brand and the products log says so."""

    try:
        probe = subprocess.run([str(points_exe), "--help"], capture_output=True,
                               text=True, errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return "--brand" in (probe.stdout or "") + (probe.stderr or "")


def great_circle_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    c = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return EARTH_RADIUS_KM * math.acos(max(-1.0, min(1.0, c)))


#: The request the MPAS suite route (hexcore.mpasphys.forecast_v842) writes beside its stream files.
NATIVE_REQUEST = "native-physics-request.json"


def native_request(forecast_dir: Path | None) -> dict:
    """The MPAS suite route's own request for a run folder, or ``{}``."""

    if forecast_dir is None:
        return {}
    try:
        doc = json.loads((Path(forecast_dir) / NATIVE_REQUEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def native_output_frames(forecast_dir: Path) -> list[Path]:
    """The frames of the MPAS suite route's ``output`` stream, from the template its request names.

    THE BREAKAGE THIS PREVENTS: ``gpuwm-hex products RUN_DIR`` refused every MPAS suite run, because that
    route writes MPAS's own stream files (``output.$Y-$M-$D_$h.$m.$s.nc`` in a native deck,
    ``history.$Y-...`` from the Registry default) and the door only looked for ``cuda-history.*``/``history.*``
    (hexmx proof, 2026-10-16).  The stream's filename template is read from the run's own request, so any
    deck's naming is found without a guess.
    """

    for stream in native_request(forecast_dir).get("streams") or ():
        if not isinstance(stream, dict) or stream.get("name") != "output":
            continue
        template = str(stream.get("filename") or "")
        if not template or "output" not in (stream.get("directions") or ()):
            continue
        pattern = re.sub(r"\$[YMDhms]", "*", template)
        return sorted(p for p in Path(forecast_dir).glob(pattern) if p.is_file())
    return []


def histories_without_surface_pressure(histories: list[Path]) -> list[Path]:
    """The source's frames when its first carries neither ``surface_pressure`` nor ``PSFC``, else ``[]``.

    One header read by the Rust reader: every frame of a run comes from the same stream, so the first
    answers for all of them.
    """

    from gpuwm.netcdf_bridge import open_dataset

    if not histories:
        return []
    try:
        names = set(open_dataset(Path(histories[0])).variables)
    except Exception:
        return []  # unreadable here: rw_hex_points names the defect itself
    return [] if names & {"surface_pressure", "PSFC"} else [Path(p) for p in histories]


def discover_histories(source: Path) -> list[Path]:
    """History frames in a forecast output folder, valid-time order."""

    source = Path(source)
    found = (sorted(source.glob("cuda-history.*.nc")) or sorted(source.glob("history.*.nc"))
             or native_output_frames(source))
    if not found:
        raise ProductDoorError(
            f"{source} holds no cuda-history.*.nc or history.*.nc frame, nor the "
            f"output stream a {NATIVE_REQUEST} names; the "
            f"product door draws history frames and there are none to draw. "
            f"Point --forecast at the forecast door's --out folder or pass "
            f"--history.")
    return found


_STAMP = re.compile(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})[.:](\d{2})[.:](\d{2})")


def frame_stamp(path: Path) -> str | None:
    m = _STAMP.search(Path(path).name)
    if not m:
        return None
    y, mo, d, h, mi, s = m.groups()
    return f"{y}-{mo}-{d}_{h}:{mi}:{s}"


#: The maps a viewer opens on, in this order: fields that have content in every frame, the start included.
#: THE BREAKAGE THIS PREVENTS (hex 1 km LAM fast start, GFS 2025-05-19 18Z over Oklahoma City, 2026-10-23): a
#: cold start carries no hydrometeors, so the f000 composite reflectivity -- drawn 12.7 s into the forecast and
#: the map a viewer showed first -- was an empty map.  Reflectivity joins the opening maps from the first frame
#: that has echoes (``first_maps``), and a viewer that opens it starts on that frame.
FIRST_MAPS = ("2m_temperature_10m_winds", "10m_wind_speed_and_direction", "sbcape")
ECHO_MAP = "composite_reflectivity"
#: The weakest reflectivity that counts as an echo (dBZ): the first step of the renderer's reflectivity scale.
ECHO_DBZ = 10.0
#: Reflectivity as a history stores it: the 2-D column maximum where the run writes one, else the 3-D field.
ECHO_FIELDS = ("refl10cm_max", "refl10cm")
#: Values per read of the reflectivity scan, so a fine global mesh is read in cell blocks, not in one array.
ECHO_BLOCK_VALUES = 16_000_000


def _parse_stamp(stamp: str) -> _dt.datetime:
    return _dt.datetime.strptime(stamp.replace("T", "_").replace(".", ":")[:19], "%Y-%m-%d_%H:%M:%S")


def frame_echo_dbz(path: Path) -> float | None:
    """The strongest reflectivity in one history frame (dBZ), read by the Rust reader; None when none is stored."""

    import numpy as np
    from gpuwm.netcdf_bridge import open_dataset

    dataset = open_dataset(Path(path))
    for name in ECHO_FIELDS:
        variable = dataset.variables.get(name)
        if variable is None or variable.size == 0:
            continue
        if variable.ndim < 2 or variable.size <= ECHO_BLOCK_VALUES:
            blocks = [np.asarray(variable[...], dtype=float)]
        else:
            # (Time, nCells, nVertLevels): blocks of cells, every level, so the peak is the column maximum.
            rows, columns = variable.shape[-2], variable.shape[-1]
            outer = max(1, variable.size // (rows * columns))
            step = max(1, ECHO_BLOCK_VALUES // (columns * outer))
            blocks = (np.asarray(variable.read_window(i0=0, ni=columns, j0=j0, nj=min(step, rows - j0)), dtype=float)
                      for j0 in range(0, rows, step))
        peak = None
        for block in blocks:
            finite = block[np.isfinite(block)]
            if finite.size:
                peak = float(finite.max()) if peak is None else max(peak, float(finite.max()))
        return peak
    return None


def first_maps(histories: list[Path], start: str, drawn) -> dict:
    """The maps a viewer opens on: ``FIRST_MAPS`` that were drawn, then reflectivity from its first echo.

    Frames are read in valid-time order and the scan stops at the first frame with an echo of ``ECHO_DBZ`` or
    more, so a run with storms at the start reads one frame.  A frame that cannot be read ends the scan with a
    note; nothing here can fail the products.
    """

    from .render_door import lead_token

    drawn = set(drawn or ())
    record: dict = {"maps": [slug for slug in FIRST_MAPS if slug in drawn], "echo_map": ECHO_MAP,
                    "echo_dbz": ECHO_DBZ, "reflectivity_from": None, "reflectivity_frame": None,
                    "reflectivity_peak_dbz": None}
    if ECHO_MAP not in drawn:
        return record
    for path in histories:
        try:
            peak = frame_echo_dbz(path)
        except Exception as error:  # the scan only orders the opening maps; the maps themselves are drawn
            record["note"] = f"reflectivity scan stopped at {Path(path).name}: {type(error).__name__}: {error}"
            break
        if peak is None or peak < ECHO_DBZ:
            continue
        stamp = frame_stamp(path)
        if stamp is None:
            record["note"] = f"{Path(path).name} has echoes but no valid-time stamp in its name"
            break
        seconds = (_parse_stamp(stamp) - _parse_stamp(start)).total_seconds()
        record.update(reflectivity_from=lead_token(int(round(seconds)), Path(path).name),
                      reflectivity_frame=Path(path).name, reflectivity_peak_dbz=round(peak, 1))
        record["maps"].append(ECHO_MAP)
        break
    return record


#: The receipt the forecast door writes beside its frames.
FORECAST_RECEIPT = "forecast-receipt.json"


def forecast_receipt(forecast_dir: Path | None) -> dict:
    """The forecast door's own receipt for a run folder, or ``{}``.

    It names the grid the run was bound to, the init it started from and its
    start time, which is everything this door needs besides the frames -- so
    after a forecast the command is ``gpuwm-hex products RUN/fc`` and nothing
    the forecast already knew has to be typed again."""

    if forecast_dir is None:
        return {}
    try:
        doc = json.loads((Path(forecast_dir) / FORECAST_RECEIPT).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        doc = None
    if isinstance(doc, dict):
        return doc
    # An MPAS suite run writes no forecast-receipt.json; its request names the same three things: the grid
    # (else the init, which carries the whole mesh), the init its input stream read, and its start.
    request = native_request(forecast_dir)
    if not request:
        return {}
    init = next((s.get("input_file") for s in request.get("streams") or ()
                 if isinstance(s, dict) and s.get("name") == "input" and s.get("input_file")), None)
    init = request.get("init") or init
    grid = request.get("grid") or init
    start = request.get("start")
    synthesized: dict = {}
    if grid:
        synthesized["mesh"] = {"grid": str(grid)}
    if init:
        synthesized["init"] = {"path": str(init)}
    if start:
        synthesized["schedule"] = {"start_time": str(start)}
    return synthesized


def mesh_is_limited_area(mesh) -> bool:
    """A mesh with boundary rings (bdyMaskCell > 0) is a limited-area mesh.

    THE BREAKAGE THIS PREVENTS (hex fast start, 2026-10-23): the first map of a 937.5 m LAM, drawn
    from its f000 frame with --history while the forecast ran (no receipt yet), also rendered a 2,030 km
    context view of a 405 km domain: 152 s (a 10.9 GB conversion) beside the 19 s mesh view.
    """
    if mesh is None:
        return False
    try:
        from netCDF4 import Dataset

        from .netcdf_lock import NETCDF
        with NETCDF, Dataset(mesh) as grid:
            return "bdyMaskCell" in grid.variables and int(grid["bdyMaskCell"][:].max()) > 0
    except (OSError, ValueError, KeyError):
        return False


def receipt_text(receipt: dict, *keys: str) -> str | None:
    """A nested string of the receipt (``mesh`` -> ``grid``), or ``None``."""

    value = receipt
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value if isinstance(value, str) and value else None


def simulation_start(explicit: str | None, forecast_dir: Path | None,
                     histories: list[Path]) -> str:
    """Run start: as given, else the forecast receipt's, else the earliest
    frame (a forecast writes its initial state as its first frame)."""

    if explicit:
        return explicit
    if forecast_dir is not None:
        value = receipt_text(forecast_receipt(forecast_dir), "schedule", "start_time")
        if value and _STAMP.search(value.replace("T", "_")):
            return value
        for name in (FORECAST_RECEIPT, "cuda-v841-forecast-receipt.json"):
            path = Path(forecast_dir) / name
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for key in ("start_time", "simulation_start"):
                value = doc.get(key) or (doc.get("request") or {}).get(key)
                if isinstance(value, str) and _STAMP.search(value.replace("T", "_")):
                    return value
    stamp = frame_stamp(histories[0])
    if stamp is None:
        raise ProductDoorError(
            f"no run start: {histories[0].name} carries no valid-time stamp and "
            f"no --simulation-start was given, so lead hours cannot be numbered")
    return stamp


def case_token(explicit: str | None, mesh: Path, start: str,
               mesh_name: str | None = None) -> str:
    """``<mesh-name>-<YYYYMMDDHH>`` unless named: read from the data, never a
    hardcoded case.  The mesh name is the one the forecast ran under when its
    receipt gives one (``mesh_name``), else the grid file's own name, else
    the folder that holds it -- a grid stored as ``ordered.grid.nc`` has no
    name of its own, and every such run was filed as ``mesh-<date>``
    (measured 2026-10-12 on the 30 km global case)."""

    if explicit:
        token = explicit
    else:
        stem = Path(mesh).name
        for suffix in (".nc", ".grid", ".static", ".init", ".ordered"):
            stem = stem.replace(suffix, "")
        stem = stem.replace("ordered", "").strip("._-")
        stem = (mesh_name or "").strip() or stem or Path(mesh).parent.name or "mesh"
        day = re.sub(r"[^0-9]", "", start)[:10]
        token = f"{stem}-{day}"
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", token).strip("-")
    if not token:
        raise ProductDoorError("--case reduces to an empty folder name")
    return token


def resolve_points_exe(explicit: str | None, convert_exe: Path) -> Path:
    """``rw_hex_points``: as named, else the build that sits beside the
    converter in use (the three are built together, and a sounding should
    come from the same build as the map under it), else the shared engine
    ladder every door of this distribution reads."""

    named = explicit or any(os.environ.get(name) for name in
                            (*engines.POINTS.env_names, engines.POINTS.gpuwm_env))
    if not named:
        beside = Path(convert_exe).with_name(engines.POINTS.name)
        if beside.is_file():
            return beside
    try:
        return engines.resolve(engines.POINTS, explicit)
    except engines.EngineRefusal as error:
        raise ProductDoorError(
            f"{error}\n  # or draw the maps without the point pictures:\n"
            f"  gpuwm-hex products ... --no-points") from None


def parse_point_text(text: str) -> list[tuple[str, float, float]]:
    out = []
    for item in [s.strip() for s in text.split(";") if s.strip()]:
        name, _, coords = item.partition("@")
        lat, _, lon = coords.partition(",")
        try:
            out.append((name.strip(), float(lat), float(lon)))
        except ValueError:
            raise ProductDoorError(f"--points item {item!r} is not NAME@LAT,LON") from None
    return out


def window_centre(manifest: dict) -> tuple[float, float, float] | None:
    """(lat, lon, half-extent km) of a derived window, from the converter's
    own window spec as the render manifest recorded it (Lambert:
    centre/dx/nx/ny; the near-equatorial lat-lon fallback: its bounds)."""

    spec = (manifest.get("convert") or {}).get("window_spec")
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except ValueError:
            return None
    if not isinstance(spec, dict):
        return None
    try:
        if spec.get("kind") == "lambert":
            half = 0.5 * float(spec["dx_metres"]) * min(int(spec["nx"]), int(spec["ny"])) / 1000.0
            return float(spec["centre_lat"]), float(spec["centre_lon"]), half
        if spec.get("kind") == "latlon":
            south, north = float(spec["south"]), float(spec["north"])
            west, east = float(spec["west"]), float(spec["east"])
            lat = 0.5 * (south + north)
            half = 0.5 * min(north - south, (east - west) * math.cos(math.radians(lat))) * 111.195
            return lat, 0.5 * (west + east), half
    except (KeyError, TypeError, ValueError):
        return None
    return None


#: Memory one frame in flight is budgeted, as a multiple of its history
#: file.  MEASURED (2026-10-12, eight frames in flight): 78.2 GB resident
#: for 30 km global frames whose history files are 1.24 GB (7.9 files per
#: frame, G1) and 31.2 GB for 48-3 km patch frames of 0.57 GB (6.8);
#: the renderer's import of the converted frame is the peak.
FRAME_MEMORY_FACTOR = 8.0
#: The share of the host's available memory frames in flight may plan on.
FRAME_MEMORY_SHARE = 0.5
MAX_FRAME_JOBS = 8


def available_memory_bytes() -> int | None:
    """``MemAvailable`` from ``/proc/meminfo``; ``None`` where there is no
    such file (the door then runs one frame at a time)."""

    try:
        with open("/proc/meminfo", encoding="ascii", errors="replace") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def auto_frame_jobs(histories: list[Path], view_jobs: int, *,
                    available: int | None = None, cpus: int | None = None) -> int:
    """Frames each view converts and imports at a time, from the host.

    Bounded by memory (half of what is available, shared by the views that
    draw side by side, at :data:`FRAME_MEMORY_FACTOR` history files per
    frame in flight), by cores (two per frame per view), by the number of
    frames and by :data:`MAX_FRAME_JOBS`.  One when the host's memory cannot
    be read: a guess that is too high is an out-of-memory kill at the end of
    a forecast."""

    if available is None:
        available = available_memory_bytes()
    if available is None or not histories:
        return 1
    cpus = cpus or os.cpu_count() or 1
    view_jobs = max(1, view_jobs)
    per_frame = FRAME_MEMORY_FACTOR * max(Path(h).stat().st_size for h in histories)
    by_memory = int(FRAME_MEMORY_SHARE * available // max(1.0, per_frame * view_jobs))
    by_cores = cpus // (2 * view_jobs)
    return max(1, min(by_memory, by_cores, len(histories), MAX_FRAME_JOBS))


def _render_view(*, histories, mesh, init, start, out_case, scratch_root, view,
                 window, products, size, model_label, remap, exes, basemap_dir,
                 keep_scratch, frame_jobs=1, source_label=None) -> tuple[int, dict | None, str | None]:
    """One view through the render door in standard-set mode."""

    args = argparse.Namespace(
        history=[str(p) for p in histories], mesh=str(mesh),
        init=str(init) if init else None, compose=None, compose_base_only=False,
        out=str(out_case), products=",".join(products), frames="all",
        window=window, view=view, model_label=model_label, source_label=source_label,
        field_set="full",
        label=None, simulation_start=start, size=size, heavy=False,
        nc_format="cdf5", scratch=str(Path(scratch_root) / f"render-{view}"),
        keep_scratch=keep_scratch, basemap_dir=basemap_dir,
        convert_exe=exes[0], renderer_exe=exes[1], remap=remap,
        standard_set=True, series=True, frame_jobs=frame_jobs)
    print(f"PRODUCTS view={view} window={window}", flush=True)
    try:
        rc = render_door.run_render(args)
    except MpasPortError as error:
        return 2, None, str(error)
    except (OSError, ValueError) as error:
        # Views draw side by side: one view's file error is that view's
        # failure, named, and the others still deliver.
        return 2, None, f"{type(error).__name__}: {error}"
    manifests = sorted((Path(out_case) / view).glob("render-manifest*.json"),
                       key=lambda p: p.stat().st_mtime)
    manifest = None
    if manifests:
        try:
            manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = None
    return rc, manifest, None


def _place_points(raw: Path, out_case: Path, start: str) -> list[str]:
    """``<case>/points/<sounding-slug|meteogram-slug>/<day>/file.png``."""

    placed = []
    init_day = start[:10]
    for png in sorted(Path(raw).rglob("*.png")):
        name = png.name
        m = re.match(r"(sounding|meteogram)_(.+?)_(\d{8})_(\d{2,4})", name)
        if not m:
            continue
        kind, slug, ymd = m.group(1), m.group(2), m.group(3)
        day = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}" if kind == "sounding" else init_day
        dest = Path(out_case) / "points" / f"{kind}-{slug}" / day / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(png), str(dest))
        placed.append(dest.relative_to(out_case).as_posix())
    return placed


_LEAD_IN_NAME = re.compile(r"_(f\d{3}(?:h\d{2}m)?)\.png$")


_INDEX_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f4f5f7;--panel:#ffffff;--ink:#1c2026;--dim:#5d6772;--line:#d5d9df;--pick:#1f5fbf;--pickink:#ffffff}
@media (prefers-color-scheme: dark){:root{--bg:#14171a;--panel:#1d2126;--ink:#e8eaed;--dim:#9aa3ab;--line:#353b42;--pick:#5b9bf0;--pickink:#0d1014}}
*{box-sizing:border-box}
body{margin:0;font:15px/1.4 system-ui,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:12px 16px;border-bottom:1px solid var(--line)}
h1{font-size:18px;margin:0 0 2px}
header p{margin:0;color:var(--dim);font-size:13px}
main{display:flex;flex-wrap:wrap;gap:12px;padding:12px 16px}
nav{flex:0 0 250px;max-width:100%}
nav select{width:100%;padding:6px;font:inherit;margin-bottom:8px;background:var(--panel);color:var(--ink);border:1px solid var(--line)}
#products{max-height:70vh;overflow:auto;border:1px solid var(--line);background:var(--panel)}
#products button{display:block;width:100%;text-align:left;padding:6px 8px;font:inherit;font-size:13px;border:0;border-bottom:1px solid var(--line);background:none;color:var(--ink);cursor:pointer;overflow-wrap:anywhere}
#products button[aria-pressed=true]{background:var(--pick);color:var(--pickink)}
#stage{flex:1 1 520px;min-width:0}
#bar{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:8px}
#bar button{padding:5px 12px;font:inherit;border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer}
#slider{flex:1 1 160px;min-width:120px}
#lead{font-variant-numeric:tabular-nums;min-width:8ch}
#frame{display:block;max-width:100%;height:auto;background:#fff;border:1px solid var(--line)}
#open{font-size:13px;color:var(--dim)}
</style></head><body>
<header><h1>__TITLE__</h1><p>gpuwm-hex products. Run start __START__. __TOTAL__ pictures. Choose a view and a product; the slider and the arrow keys step through its frames.</p></header>
<main>
<nav><label for="view">View</label><select id="view"></select><div id="products"></div></nav>
<section id="stage">
<div id="bar"><button id="prev" type="button">Previous</button><button id="play" type="button">Play</button><button id="next" type="button">Next</button>
<input id="slider" type="range" min="0" max="0" value="0" aria-label="Frame"><span id="lead"></span><a id="open" href="#">open picture</a></div>
<img id="frame" alt="">
</section>
</main>
<script>
const FIRST = __FIRST__;
const DATA = __DATA__;
const el = id => document.getElementById(id);
let view = Object.keys(DATA)[0], product = null, index = 0, timer = null;
function frames(){ return (DATA[view] || {})[product] || []; }
function show(){
  const f = frames(); if (!f.length) return;
  index = Math.max(0, Math.min(index, f.length - 1));
  el('frame').src = f[index][1]; el('frame').alt = product + ' ' + f[index][0];
  el('open').href = f[index][1]; el('lead').textContent = f[index][0] + '  ' + (index + 1) + '/' + f.length;
  el('slider').max = f.length - 1; el('slider').value = index;
}
function pick(name){
  const before = frames()[index];
  const keepLead = before ? before[0] : null;
  product = name;
  const at = frames().findIndex(x => x[0] === keepLead);
  index = at >= 0 ? at : 0;
  // Reflectivity opens on its first frame with echoes: an earlier one (a cold start's f000) is an empty map.
  if (name === FIRST.echo_map && FIRST.reflectivity_from) {
    const echo = frames().findIndex(x => x[0] === FIRST.reflectivity_from);
    if (echo > index) index = echo;
  }
  for (const b of el('products').children) b.setAttribute('aria-pressed', b.dataset.name === name);
  show();
}
function list(){
  const box = el('products'); box.textContent = '';
  for (const name of Object.keys(DATA[view])) {
    const b = document.createElement('button'); b.type = 'button'; b.dataset.name = name;
    b.textContent = name + ' (' + DATA[view][name].length + ')'; b.onclick = () => pick(name); box.appendChild(b);
  }
  pick(Object.keys(DATA[view]).includes(product) ? product : Object.keys(DATA[view])[0]);
}
function step(d){ const n = frames().length; if (n) { index = (index + d + n) % n; show(); } }
function stop(){ if (timer) { clearInterval(timer); timer = null; el('play').textContent = 'Play'; } }
el('prev').onclick = () => { stop(); step(-1); };
el('next').onclick = () => { stop(); step(1); };
el('play').onclick = () => { if (timer) stop(); else { timer = setInterval(() => step(1), 600); el('play').textContent = 'Pause'; } };
el('slider').oninput = e => { stop(); index = +e.target.value; show(); };
document.addEventListener('keydown', e => { if (e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowLeft') { stop(); step(-1); } if (e.key === 'ArrowRight') { stop(); step(1); } });
const sel = el('view');
for (const name of Object.keys(DATA)) { const o = document.createElement('option'); o.value = o.textContent = name; sel.appendChild(o); }
sel.onchange = () => { stop(); view = sel.value; list(); };
list();
</script></body></html>
"""


def index_tree(out_case: Path) -> dict[str, dict[str, list[list[str]]]]:
    """``{view: {product: [[caption, relative path], ...]}}`` for every
    picture under a case folder, frames in valid-day then lead order."""

    tree: dict[str, dict[str, list[list[str]]]] = {}
    out_case = Path(out_case)
    for view_dir in sorted(p for p in out_case.iterdir() if p.is_dir()):
        products: dict[str, list[list[str]]] = {}
        for product_dir in sorted(p for p in view_dir.iterdir() if p.is_dir()):
            pictures = sorted(product_dir.rglob("*.png"),
                              key=lambda p: (p.parent.name, p.name))
            rows = []
            for png in pictures:
                lead = _LEAD_IN_NAME.search(png.name)
                rows.append([lead.group(1) if lead else png.parent.name,
                             png.relative_to(out_case).as_posix()])
            if rows:
                products[product_dir.name] = rows
        if products:
            tree[view_dir.name] = products
    return tree


def write_index(out_case: Path, case: str, start: str, first: dict | None = None) -> Path | None:
    """``<case>/index.html``: a one-file viewer over every delivered
    picture -- view, product, a frame slider and play.  Built from the tree
    as it is on disk, so it shows what was delivered and nothing else; the
    links are relative and the page loads nothing from the network, so the
    case folder can be copied anywhere and still open.  It lists and opens
    on the ``first`` maps (``first_maps``) and opens reflectivity on its
    first frame with echoes."""

    tree = index_tree(out_case)
    if not tree:
        return None
    first = first or {"maps": list(FIRST_MAPS), "echo_map": ECHO_MAP, "reflectivity_from": None}
    leading = list(first.get("maps") or [])
    if first.get("echo_map") and first["echo_map"] not in leading:
        leading.append(first["echo_map"])  # no echoes yet: still listed right after the opening maps
    rank = {slug: position for position, slug in enumerate(leading)}
    # "global" first, then the mesh's own views, points last.
    order = sorted(tree, key=lambda name: (name == "points", name != "global", name))
    data = {name: {product: tree[name][product]
                   for product in sorted(tree[name], key=lambda p: (rank.get(p, len(rank)), p))}
            for name in order}
    opening = {"echo_map": first.get("echo_map"), "reflectivity_from": first.get("reflectivity_from")}
    total = sum(len(rows) for products in tree.values() for rows in products.values())
    page = (_INDEX_PAGE
            .replace("__TITLE__", _html.escape(case))
            .replace("__START__", _html.escape(start))
            .replace("__TOTAL__", str(total))
            .replace("__FIRST__", json.dumps(opening).replace("</", "<\\/"))
            .replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    path = Path(out_case) / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def run_products(args) -> int:
    started = time.monotonic()
    named = getattr(args, "forecast", None)
    positional = getattr(args, "forecast_dir", None)
    if named and positional and Path(named) != Path(positional):
        raise ProductDoorError(
            f"two forecast folders were given ({positional} and --forecast "
            f"{named}); the door draws one run")
    forecast_dir = Path(named or positional) if (named or positional) else None
    receipt = forecast_receipt(forecast_dir)
    from .chem_transport import product_slugs
    chemistry_products = product_slugs(forecast_dir)
    if args.history:
        histories = [Path(p) for p in args.history]
    elif forecast_dir is not None:
        histories = discover_histories(forecast_dir)
    else:
        raise ProductDoorError("give --forecast DIR (the forecast door's --out) or --history FILE...")
    for path in histories:
        if not path.is_file():
            raise ProductDoorError(f"--history names a missing file: {path}")
    # What the forecast already recorded is read from its receipt; a flag
    # still wins, and a path the receipt names that is no longer there (the
    # run folder was copied without its mesh) is refused as the receipt's.
    mesh_text = args.mesh or receipt_text(receipt, "mesh", "grid")
    if not mesh_text:
        raise ProductDoorError(
            "no mesh: --mesh was not given and the forecast folder carries no "
            f"{FORECAST_RECEIPT} naming the grid the run was bound to. The "
            "converter places the mesh's cells on the render grid from it; "
            "pass --mesh GRID.nc.")
    mesh = Path(mesh_text)
    mesh_source = "--mesh" if args.mesh else f"{FORECAST_RECEIPT} (mesh.grid)"
    if not mesh.is_file():
        raise ProductDoorError(
            f"{mesh_source} names a missing file: {mesh}. The converter places the "
            f"mesh's cells on the render grid from it; give the run's grid file "
            f"with --mesh.")
    init_text = args.init or receipt_text(receipt, "init", "path")
    init = Path(init_text) if init_text else None
    if init is not None and not init.is_file():
        init_source = "--init" if args.init else f"{FORECAST_RECEIPT} (init.path)"
        raise ProductDoorError(
            f"{init_source} names a missing file: {init}. zgrid, the column "
            f"heights behind the pressure-level maps and the soundings, lives "
            f"there; pass --init INIT.nc.")
    start = simulation_start(args.simulation_start, forecast_dir, histories)
    windowed, windows_not_requested = windowed_for_history(histories, start)
    global_products = BASE_GLOBAL_PRODUCTS + windowed
    regional_products = global_products + REGIONAL_EXTRA_PRODUCTS
    for slug, reason in windows_not_requested.items():
        print(f"PRODUCTS window not requested: {slug}: {reason}", flush=True)
    case = case_token(args.case, mesh, start,
                      receipt_text(receipt, "mesh_binding", "mesh")
                      or receipt_text(receipt, "admission", "mesh"))
    if args.out:
        out_root = Path(args.out)
    elif forecast_dir is not None:
        out_root = forecast_dir / "products"
    else:
        raise ProductDoorError(
            "no --out: with --history there is no forecast folder to file "
            "the products beside, so name the delivered tree's root")
    args.out = str(out_root)
    out_case = Path(args.out) / case
    scratch_root = Path(args.scratch) if args.scratch else Path(args.out).parent / (Path(args.out).name + ".products-scratch")
    render_door.refuse_scratch_inside_out(scratch_root, Path(args.out))
    out_case.mkdir(parents=True, exist_ok=True)
    convert_exe = render_door.resolve_convert_exe(args.convert_exe)
    renderer_exe = render_door.resolve_renderer_exe(args.renderer_exe)
    exes = (str(convert_exe), str(renderer_exe))
    print(f"PRODUCTS case={case} frames={len(histories)} start={start} out={out_case}", flush=True)
    # The opening maps (first_maps): the reflectivity scan reads frames beside the views, off their path.
    opening: dict = {}
    opening_scan = threading.Thread(
        target=lambda: opening.update(first_maps(histories, start, (*FIRST_MAPS, ECHO_MAP))),
        name="first-maps", daemon=True)
    opening_scan.start()

    views: list[dict] = []

    def view(name, window, products, size):
        view_started = time.monotonic()
        rc, manifest, refusal = _render_view(
            histories=histories, mesh=mesh, init=init, start=start,
            out_case=out_case, scratch_root=scratch_root, view=name,
            window=window, products=tuple(products) + chemistry_products, size=size,
            model_label=args.model_label, remap=args.remap, exes=exes,
            basemap_dir=args.basemap_dir, keep_scratch=args.keep_scratch,
            frame_jobs=frame_jobs, source_label=getattr(args, "source_label", None))
        record = {"view": name, "window": window, "rc": rc, "refusal": refusal,
                  "elapsed_seconds": round(time.monotonic() - view_started, 3)}
        if refusal is not None and "no refined region" not in refusal:
            print(f"PRODUCTS view={name} refused: {refusal}", flush=True)
        if manifest is not None:
            record.update({
                "counts": manifest.get("counts"),
                "products_drawn": manifest.get("products_drawn"),
                "products_unavailable": manifest.get("products_unavailable"),
                "window_description": (manifest.get("convert") or {}).get("window_description"),
                "remap": (manifest.get("convert") or {}).get("remap"),
                "product_remap": (manifest.get("convert") or {}).get("product_remap"),
                "precipitation_budget": (manifest.get("convert") or {}).get("precipitation_budget"),
            })
        views.append(record)
        return rc, manifest, refusal

    # Views are independent passes over the same frames (their own window,
    # their own scratch), so they draw side by side.  MEASURED (G1,
    # 2026-10-12, nine 30 km frames): one after another the three views took
    # 25 min 54 s, most of it each view's own import.
    # What the point pictures need is asked for BEFORE any view is drawn: a
    # missing rw_hex_points or --init used to be discovered after every map
    # was on disk, minutes into the command.
    points_exe = None
    if not args.no_points:
        points_exe = resolve_points_exe(args.points_exe, convert_exe)
        if init is None:
            raise ProductDoorError(
                "soundings need the run's --init (zgrid, the column heights, "
                "lives there and nowhere else); pass --init or --no-points")
    extra_regions: dict[str, str] = {}
    for item in args.region or []:
        name, _, spec = item.partition("=")
        if not spec:
            raise ProductDoorError(f"--region {item!r} is not NAME=region:CLAT,CLON,W_KM,H_KM")
        extra_regions[render_door.checked_view(name)] = spec

    jobs = max(1, int(getattr(args, "jobs", 3) or 1))
    requested = getattr(args, "frame_jobs", None)
    if requested in (None, "auto"):
        frame_jobs = auto_frame_jobs(histories, jobs)
    else:
        frame_jobs = render_door.checked_frame_jobs(requested)
    print(f"PRODUCTS views side by side={jobs} frames at a time per view={frame_jobs}",
          flush=True)
    pool = ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="view")
    pending = []
    # A LIMITED-AREA run holds data inside its own domain only, and the mesh
    # view frames all of it.  THE BREAKAGE THIS PREVENTS (gpuwm-hex regional,
    # 2026-10-20, a 43,360-cell cull, 3 h, 4 frames): the
    # global view converted every frame onto a whole-globe window at the
    # cull's own spacing (35 GB per frame, 1,444 s for the conversion) and
    # the 1,500 km context view another 550 s, together 131 GB of scratch,
    # to draw maps that are empty outside a 270 km domain; the mesh view of
    # the same frames took 61 s.  Named --region views are still drawn.
    limited_area = (receipt_text(receipt, "admission", "configuration") == "limited-area"
                    or mesh_is_limited_area(mesh))
    if limited_area:
        print("PRODUCTS limited-area run: the mesh view frames the whole domain; "
              "no global or context view", flush=True)
    if not args.no_global and not limited_area:
        pending.append(pool.submit(view, "global", "global", global_products, args.global_size))
    refined_centre = None
    if not args.no_mesh_view:
        rc, manifest, refusal = pool.submit(
            view, "refined-core", "mesh", regional_products, args.regional_size).result()
        if refusal is not None and "no refined region" in refusal:
            # A uniform mesh has no refined region: the converter refuses
            # --window mesh by that name.  Not a failure of this run.
            core = next(v for v in views if v["view"] == "refined-core")
            core["rc"] = 0
            core["note"] = "uniform mesh: no refined region to frame"
            print("PRODUCTS uniform mesh: no refined-core view; default regions instead", flush=True)
            regions = dict(DEFAULT_REGIONS)
        else:
            regions = {}
            if manifest is not None:
                refined_centre = window_centre(manifest)
            if refined_centre is not None and not limited_area:
                lat, lon, half = refined_centre
                extent = max(1500.0, 8.0 * half)
                regions["refined-context"] = f"region:{lat:.3f},{lon:.3f},{extent:.0f},{0.7 * extent:.0f}"
    else:
        regions = {}
    regions.update(extra_regions)
    for name, spec in regions.items():
        pending.append(pool.submit(view, name, spec, regional_products, args.regional_size))

    def draw_points() -> dict:
        points_started = time.monotonic()
        lacking = histories_without_surface_pressure(histories)
        if lacking:
            # THE BREAKAGE THIS PREVENTS (hexmx proof, 2026-10-16): an MPAS suite run's history carries no
            # surface_pressure (the suite route does not write that diag field yet; its receipt lists it under
            # missing_fields), so rw_hex_points refused and the whole products command exited 1 with every map
            # drawn.  A sounding and a meteogram need PSFC, so the point pictures are skipped for such a source,
            # said once, and the maps stand.
            reason = (f"{lacking[0].name} carries no surface_pressure (PSFC), which a sounding and a meteogram "
                      f"need; point pictures skipped for this source ({len(lacking)} frames)")
            print(f"POINTS skipped: {reason}", flush=True)
            return {"rc": 0, "skipped": reason, "points": [],
                    "elapsed_seconds": round(time.monotonic() - points_started, 3)}
        if args.points:
            points = parse_point_text(args.points)
        elif refined_centre is not None:
            lat, lon, half = refined_centre
            radius = max(150.0, 0.9 * half)
            points = [p for p in DEFAULT_POINTS if great_circle_km(lat, lon, p[1], p[2]) <= radius]
            if not points:
                points = [("Refined core centre", lat, lon)]
        else:
            points = list(DEFAULT_POINTS[:3]) + [p for p in DEFAULT_POINTS if p[0] in ("London", "Tokyo", "Sydney")]
            points = points[:DEFAULT_GLOBAL_POINT_COUNT]
        raw = scratch_root / "points"
        raw.mkdir(parents=True, exist_ok=True)
        receipt = raw / "points.json"
        command = [str(points_exe), "--mesh", str(mesh), "--init", str(init),
                   "--out-dir", str(raw), "--simulation-start", start,
                   "--model-label", args.model_label,
                   "--sounding-every-hours", str(args.sounding_every_hours),
                   "--points", ";".join(f"{n}@{la},{lo}" for n, la, lo in points),
                   "--json", str(receipt)]
        brand = getattr(args, "brand", None)
        if brand and points_take_brand(points_exe):
            command += ["--brand", brand]
        elif brand:
            print(f"POINTS brand not drawn: {points_exe} predates --brand; "
                  "install the gpuwm release that carries it", flush=True)
        command += ["--history", *[str(p) for p in histories]]
        print("POINTS $ " + " ".join(command), flush=True)
        result = subprocess.run(command, capture_output=True, text=True, errors="replace")
        for line in (result.stdout or "").splitlines():
            if line.startswith(("POINT\t", "METEOGRAM\t", "FINISHED\t", "POINTS MISSING\t")):
                print(f"POINTS {line}", flush=True)
        points_record = {"rc": result.returncode, "points": points, "invocation": command,
                         "elapsed_seconds": round(time.monotonic() - points_started, 3)}
        if result.returncode != 0:
            tail = [l for l in (result.stderr or "").splitlines() if l.strip()]
            points_record["refusal"] = tail[-1] if tail else f"exit {result.returncode}"
            print(f"POINTS refused: {points_record['refusal']}", flush=True)
        else:
            try:
                points_record["receipt"] = json.loads(receipt.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                points_record["receipt"] = None
            points_record["placed"] = _place_points(raw, out_case, start)
            (out_case / "points").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(receipt, out_case / "points" / "points.json")
            points_record["points_exe_sha256"] = render_door.sha256_file(points_exe)
        return points_record

    # The point pictures read the same frames as the views and wait on none
    # of them, so they draw beside the regional views.
    points_future = None if args.no_points else pool.submit(draw_points)
    for future in pending:
        future.result()
    points_record = points_future.result() if points_future is not None else None
    pool.shutdown()
    order = ["global", "refined-core", *regions]
    views.sort(key=lambda v: order.index(v["view"]) if v["view"] in order else len(order))
    if not args.keep_scratch:
        shutil.rmtree(scratch_root, ignore_errors=True)

    rendered = sum((v.get("counts") or {}).get("rendered", 0) for v in views)
    opening_scan.join()
    drawn = set()
    for v in views:
        listed = v.get("products_drawn") or ()
        drawn.update(p.strip() for p in (listed.split(",") if isinstance(listed, str) else listed) if p)
    if opening:
        opening["maps"] = [slug for slug in opening["maps"] if slug in drawn]
        if opening["reflectivity_from"]:
            print(f"PRODUCTS reflectivity has echoes from {opening['reflectivity_from']} "
                  f"({opening['reflectivity_peak_dbz']} dBZ)", flush=True)
        print(f"PRODUCTS opening maps: {', '.join(opening['maps']) or 'none'}", flush=True)
    failed_views = [v["view"] for v in views if v.get("rc") not in (0, None)]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "case": case,
        "simulation_start": start,
        "mesh": str(mesh),
        "init": str(init) if init else None,
        "histories": [str(p) for p in histories],
        "layout": "<out>/<case>/<domain>/<product>/<valid-day>/ ; points: <out>/<case>/points/<sounding|meteogram>-<point>/<day>/",
        "remap": args.remap,
        "view_jobs": jobs,
        "frame_jobs": frame_jobs,
        "views": views,
        "points": points_record,
        "chemistry_products": list(chemistry_products),
        "windows_not_requested": windows_not_requested,
        "maps_rendered": rendered,
        "first_maps": opening or None,
        "failed_views": failed_views,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    index = write_index(out_case, case, start, opening or None)
    manifest["index"] = index.name if index is not None else None
    path = out_case / "products-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if index is not None:
        print(f"PRODUCTS index {index}", flush=True)
    soundings = len((points_record or {}).get("placed") or [])
    print(f"PRODUCTS done maps={rendered} point_pictures={soundings} "
          f"failed_views={','.join(failed_views) or 'none'} manifest={path}", flush=True)
    if failed_views or (points_record is not None and points_record.get("rc")):
        return 1
    return 0 if rendered > 0 else 1


def add_products_arguments(parser) -> None:
    parser.add_argument("forecast_dir", nargs="?", default=None, metavar="RUN_DIR",
                        help="the forecast door's --out folder: its history frames, and its "
                             "receipt, which names the mesh, the init and the start time, so "
                             "after a forecast this one argument is the whole command")
    parser.add_argument("--forecast", default=None, metavar="DIR",
                        help="the same folder, by flag")
    parser.add_argument("--history", nargs="+", default=None, metavar="FILE",
                        help="history frames, instead of --forecast")
    parser.add_argument("--mesh", default=None, metavar="FILE",
                        help="the run's grid file (cell centres, connectivity, areas). "
                             "Default: the grid the forecast receipt names")
    parser.add_argument("--init", default=None, metavar="FILE",
                        help="the run's init file: zgrid for pressure-level maps and soundings. "
                             "Default: the init the forecast receipt names")
    parser.add_argument("--out", default=None, metavar="DIR",
                        help="delivered tree root: <out>/<case>/<domain>/<product>/<valid-day>/. "
                             "Default: <RUN_DIR>/products")
    parser.add_argument("--case", default=None,
                        help="case folder name (default: <mesh>-<YYYYMMDDHH> from the data)")
    parser.add_argument("--simulation-start", default=None, metavar="YYYY-MM-DD_HH:MM:SS")
    parser.add_argument("--region", action="append", metavar="NAME=region:CLAT,CLON,W_KM,H_KM",
                        help="an extra regional view (repeatable); dx from the mesh cells inside it")
    parser.add_argument("--points", default=None, metavar="'Name@LAT,LON;...'",
                        help="sounding/meteogram points (default: cities inside a refined core, "
                             "else a world set)")
    parser.add_argument("--sounding-every-hours", type=float, default=6.0)
    parser.add_argument("--remap", default=render_door.REMAP_AUTO, choices=render_door.REMAP_CHOICES,
                        help="see `gpuwm-hex render --help`; default: product (linear state, "
                             "conservative precipitation)")
    parser.add_argument("--model-label", default=render_door.DEFAULT_MODEL_LABEL,
                        help="the model the maps and soundings name, top left (the mesh "
                             "spacing is added, e.g. 'hex 121 km'); default: hex")
    parser.add_argument("--brand", default=None, metavar="TEXT",
                        help="bold text at the top right of every sounding and meteogram, "
                             "e.g. 'WOOF Hex · Global 120 km · Init GFS 22 Oct 2026 00Z' "
                             "(at most 120 characters); default: none")
    parser.add_argument("--source-label", default=None, metavar="TEXT",
                        help="source credit printed on each map (1-160 printable ASCII "
                             "characters), e.g. which ensemble member it came from; default: "
                             "Hex forecast where the renderer supports it")
    parser.add_argument("--global-size", default="1800x1000")
    parser.add_argument("--regional-size", default="1400x1000")
    parser.add_argument("--jobs", type=int, default=3, metavar="N",
                        help="views drawn side by side (default 3; each is its own "
                             "converter and renderer pass over the frames)")
    parser.add_argument("--frame-jobs", default="auto", metavar="N|auto",
                        help="frames each view converts and imports at a time. auto "
                             "(default): from the host's available memory and cores "
                             "(8 history files of memory per frame in flight, half of "
                             "what is available, at most 8); 1 where memory cannot be read")
    parser.add_argument("--no-global", action="store_true")
    parser.add_argument("--no-mesh-view", action="store_true")
    parser.add_argument("--no-points", action="store_true")
    parser.add_argument("--scratch", default=None, metavar="DIR")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--basemap-dir", default=None, metavar="DIR")
    parser.add_argument("--convert-exe", default=None, metavar="FILE")
    parser.add_argument("--renderer-exe", default=None, metavar="FILE")
    parser.add_argument("--points-exe", default=None, metavar="FILE")


def add_products_parser(commands) -> None:
    parser = commands.add_parser(
        "products",
        help="forecast history -> the standard product set (maps, soundings, "
             "meteograms) through the Rust renderer, on any mesh")
    add_products_arguments(parser)
    parser.set_defaults(handler=run_products)
