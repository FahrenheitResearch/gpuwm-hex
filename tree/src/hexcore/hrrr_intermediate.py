"""A projected regional source onto the regular lat-lon WPS intermediate the
init and boundary engines read: ``gpuwm-hex intermediate``.

THE GAP THIS CLOSES, MEASURED.  ``docs/source-matrix.md`` (2026-08-24) drove
every registered source through the hex chain and recorded that a Lambert
regional product (HRRR, RAP, RRFS) is refused at TWO independent layers:
the engine's ``met_intermediate`` writer refuses to mint a projected grid as
``iproj 0`` ("would silently mis-georeference every point"), and
``rw_mpas_init`` / ``rw_mpas_lbc`` invert projection code 0 only.  So the
HRRR bytes the engine already fetches, decodes and interpolates for its own
WRF road never reached a hex init.

THE ROAD, and what is borrowed from where.  Everything with numbers in it
is the engine's own, driven, not re-implemented:

* DECODE -- ``hrrr_grib2_bridge`` (Rust, staged by the engine's bundle,
  resolved through ``gpuwm.bridges.resolve_source_decoder``), the same
  fail-closed decoder ``gpuwm prep`` runs, over the same ``wrfnat`` + soil
  pair ``gpuwm fetch --source hrrr`` writes, into the engine's own native
  window format that ``gpuwm.ingest.hrrr`` maps;
* REGRID -- the rw_wps projected-intermediate entry builds the engine's
  FP64 Lambert donor geometry and keeps donors separate from FP32 fractions.
  It calls the selected engine library's exact-donor WPS overlapping-parabolic
  operator. Rust rotates source winds to earth coordinates before movement;
* SOIL -- Rust interpolates the source row's depth nodes to the engine's
  declared layer midpoints in FP64, rounds once to FP32 and moves by nearest;
* WRITE -- Rust writes WPS version-5 records and checks the complete file with
  the engine Rust reader before publishing the per-hour completion marker.

WHAT THE RESULT IS, STATED PLAINLY.  The intermediate is HRRR RESAMPLED onto
a 0.025-degree regular lat-lon grid over the domain (2.8 km meridionally,
finer zonally at mid-latitudes -- the source is not coarsened).  Fifty
hybrid levels are carried as level-indexed slabs with the 3-D ``PRESSURE``
field beside them, which is the MPAS convention for model-level first
guesses (ERA5 ML uses it) and the branch ``rw_mpas_init`` takes when a 3-D
pressure is present.  Hydrometeors are not carried, although the engine's
window decodes them.  The init stream has slots for cloud water and rain
(``qc``, ``qr``), but the init and boundary writers read no hydrometeor
record from an intermediate file and write both as zero; it has no slot for
cloud ice, snow or graupel, so the README's hour-zero-has-no-ice sentence
applies.  Carrying them starts in the engine's writers, not in this row.

THE ARBITRARY ACCEPTANCE TEST.  The source is a ROW (:data:`SOURCE_ROWS`):
the engine decoder key, the loader, the level count, the field map, the
soil nodes, the file names ``gpuwm fetch`` writes.  Nothing below this
docstring branches on the string ``hrrr``; a second projected source with an
engine decoder and loader is a second row.  The module carries the name of
the one row it ships because the plan of record names the file that way.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import socket
import sys
from threading import Condition, Event
import time
import warnings
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .errors import MpasPortError

INTERMEDIATE_SCHEMA = "gpuwm-hex.intermediate/v1"
WPS_VERSION = 5
SURFACE_LEVEL = 200100.0
#: Meridional 3 km at the equator, a hair finer than HRRR's 3 km cells, so a
#: 3 km source is never coarsened by the target.
DEFAULT_SPACING_DEG = 0.025
#: Ground kept past the cull cut, so the four-point parabolic stencil and
#: the seven boundary rings never reach the edge of the intermediate.
DEFAULT_MARGIN_KM = 30.0
KM_PER_DEG = 111.195


class IntermediateRefusal(MpasPortError):
    """The intermediate cannot be written, and the message says why."""


# ---------------------------------------------------------------------------
# the source table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceRow:
    """One projected source: how the engine decodes and maps it, what it carries."""

    name: str
    #: The key ``gpuwm.bridges.resolve_source_decoder`` resolves.
    decoder_key: str
    #: The engine module whose loader maps the decoder's output.
    loader_module: str
    #: How the fetch names the two GRIB files per hour: (atmosphere, soil).
    grib_names: tuple[str, str]
    #: 3-D fields, by the engine's window name -> (WPS name, units, description).
    atmosphere_3d: Mapping[str, tuple[str, str, str]]
    #: The 3-D wind pair on the source grid basis: (u, v).
    wind_3d: tuple[str, str]
    #: Surface fields at the WPS surface level tag.
    surface: Mapping[str, tuple[str, str, str, str]]
    #: The 10 m wind pair.
    wind_10m: tuple[str, str]
    #: Categorical surface fields, moved by nearest neighbour.
    nearest: tuple[str, ...]
    #: Soil nodes and the Noah layer names they become.
    soil_temperature: str
    soil_moisture: str
    soil_layer_names: tuple[tuple[str, str], ...]
    levels: int
    map_source: str
    #: How long the decoder holds its output open per hour, for the halo.
    parabolic_halo_cells: int = 2
    #: What a refusal calls the product.
    description: str = ""
    #: The engine's fetch transport, used for a missing hour while decode runs.
    fetch_function: str = ""
    source_grid_function: str = ""
    loader_function: str = ""
    producer_class: str = ""
    decoder_workers_max: int = 1
    decoded_directories: tuple[str, str] = ("", "")
    soil_depth_metadata: tuple[str, str] = ("", "")

    def hour_files(self, root: Path, cycle: datetime, hour: int) -> tuple[Path, Path]:
        atmosphere, soil = self.grib_names
        return (
            root / atmosphere.format(cycle=cycle, hour=hour),
            root / soil.format(cycle=cycle, hour=hour),
        )


SOURCE_ROWS: Mapping[str, SourceRow] = {
    "hrrr": SourceRow(
        name="hrrr",
        decoder_key="hrrr",
        loader_module="gpuwm.ingest.hrrr",
        grib_names=(
            "hrrr.t{cycle:%H}z.wrfnatf{hour:02d}.grib2",
            "hrrr.t{cycle:%H}z.soilf{hour:02d}.grib2",
        ),
        atmosphere_3d={
            "PRES": ("PRESSURE", "Pa", "Pressure"),
            "HGT": ("GHT", "m", "Height"),
            "TT": ("TT", "K", "Temperature"),
            "SPFH": ("SPECHUMD", "kg kg-1", "Specific humidity"),
        },
        wind_3d=("U_MASS", "V_MASS"),
        surface={
            "T2": ("TT", "K", "Temperature", "parabolic"),
            "Q2": ("SPECHUMD", "kg kg-1", "Specific humidity", "parabolic"),
            "PSFC": ("PSFC", "Pa", "Surface pressure", "parabolic"),
            "SKINTEMP": ("SKINTEMP", "K", "Skin temperature", "parabolic"),
            "SOILHGT": ("SOILHGT", "m", "Terrain field of source analysis", "parabolic"),
            "SNOW": ("SNOW", "kg m-2", "Water equivalent snow depth", "bilinear"),
            "LANDSEA": ("LANDSEA", "proprtn", "Land/Sea flag (1=land, 0 or 2=sea)", "nearest"),
            "XICE": ("SEAICE", "proprtn", "Ice flag", "nearest"),
        },
        wind_10m=("U10_MASS", "V10_MASS"),
        nearest=("LANDSEA", "XICE"),
        soil_temperature="SOILT",
        soil_moisture="SOILW",
        soil_layer_names=(
            ("ST000010", "SM000010"),
            ("ST010040", "SM010040"),
            ("ST040100", "SM040100"),
            ("ST100200", "SM100200"),
        ),
        levels=50,
        map_source="NCEP HRRR via gpuwm-hex latlon",
        description="the 3 km Lambert CONUS HRRR (native hybrid wrfnat + wrfprs soil)",
        fetch_function="gpuwm.fetch:fetch_hrrr",
        source_grid_function="hrrr_source_grid",
        loader_function="load_hrrr_pipeline_ready_window",
        producer_class="tools.hrrr_pipeline:HrrrPipelineProducer",
        decoder_workers_max=13,
        decoded_directories=("atmosphere-f{hour:02d}", "soil-f{hour:02d}"),
        soil_depth_metadata=("HRRR_SOIL_NODE_DEPTHS_M", "NOAH_LAYER_MIDPOINTS_M"),
    ),
}


def source_row(name: str) -> SourceRow:
    key = str(name).strip().lower()
    if key not in SOURCE_ROWS:
        raise IntermediateRefusal(
            f"--source {name!r} is not a projected source this door knows.  "
            f"Known rows: {', '.join(sorted(SOURCE_ROWS))}.  A source is a row "
            f"in hexcore.hrrr_intermediate.SOURCE_ROWS naming the engine's "
            f"decoder, its loader and its field map; a uniform lat-lon source "
            f"does not need this door at all (docs/source-matrix.md)"
        )
    return SOURCE_ROWS[key]


# ---------------------------------------------------------------------------
# the target grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LatLonTarget:
    """A regular lat-lon grid, south-west corner first, x fastest."""

    south: float
    west: float
    dlat: float
    dlon: float
    nx: int
    ny: int

    @property
    def north(self) -> float:
        return self.south + self.dlat * (self.ny - 1)

    @property
    def east(self) -> float:
        return self.west + self.dlon * (self.nx - 1)

    def latitudes(self) -> np.ndarray:
        return self.south + self.dlat * np.arange(self.ny, dtype=np.float64)

    def longitudes(self) -> np.ndarray:
        return self.west + self.dlon * np.arange(self.nx, dtype=np.float64)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """``(lat, lon)`` as ``(ny, nx)`` arrays, the engine's row-major shape."""

        lat = np.repeat(self.latitudes()[:, None], self.nx, axis=1)
        lon = np.repeat(self.longitudes()[None, :], self.ny, axis=0)
        return lat, lon

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "regular-latlon",
            "south": self.south, "west": self.west, "north": self.north, "east": self.east,
            "dlat": self.dlat, "dlon": self.dlon, "nx": self.nx, "ny": self.ny,
        }


def target_for_cap(
    centre: tuple[float, float], radius_km: float, *,
    margin_km: float = DEFAULT_MARGIN_KM, spacing_deg: float = DEFAULT_SPACING_DEG,
) -> LatLonTarget:
    """The lat-lon box that covers a cap plus a margin, on a fixed spacing."""

    lat, lon = centre
    if not spacing_deg > 0.0:
        raise IntermediateRefusal(f"--spacing-deg {spacing_deg} is not positive")
    reach_km = float(radius_km) + float(margin_km)
    half_lat = reach_km / KM_PER_DEG
    cos_lat = max(math.cos(math.radians(lat)), 0.05)
    half_lon = reach_km / (KM_PER_DEG * cos_lat)
    ny = int(math.ceil(2.0 * half_lat / spacing_deg)) + 1
    nx = int(math.ceil(2.0 * half_lon / spacing_deg)) + 1
    south = round(lat - 0.5 * (ny - 1) * spacing_deg, 6)
    west = round(lon - 0.5 * (nx - 1) * spacing_deg, 6)
    if south < -90.0 or south + (ny - 1) * spacing_deg > 90.0:
        raise IntermediateRefusal(
            f"a {reach_km:g} km reach around {lat:.4f} N crosses a pole; a "
            f"regular lat-lon intermediate cannot carry it"
        )
    return LatLonTarget(south=south, west=west, dlat=spacing_deg, dlon=spacing_deg, nx=nx, ny=ny)


def target_from_plan(plan_path: Path, *, margin_km: float, spacing_deg: float) -> tuple[LatLonTarget, dict[str, Any]]:
    """The target grid a point plan's cull region implies."""

    try:
        document = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntermediateRefusal(f"--from-plan {plan_path} is not readable JSON: {error}") from error
    plan = document.get("plan", document)
    region = plan.get("cull_region") or {}
    if region.get("kind") != "cap":
        raise IntermediateRefusal(
            f"--from-plan {plan_path} carries no cap cull_region; it is not a "
            f"gpuwm-hex.point-plan/v1 or point-generate/v1 document"
        )
    centre = (float(region["center_deg"][0]), float(region["center_deg"][1]))
    radius = float(region["radius_km"])
    cull = plan.get("cull") or {}
    halo = float(cull.get("halo_km") or 0.0)
    target = target_for_cap(centre, radius + halo, margin_km=margin_km, spacing_deg=spacing_deg)
    return target, {"centre_deg": list(centre), "cut_radius_km": radius, "halo_km": halo}


# ---------------------------------------------------------------------------
# the source window
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceWindow:
    i_start: int
    i_end: int
    j_start: int
    j_end: int

    @property
    def nx(self) -> int:
        return self.i_end - self.i_start + 1

    @property
    def ny(self) -> int:
        return self.j_end - self.j_start + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "zero_based_inclusive": {"i": [self.i_start, self.i_end], "j": [self.j_start, self.j_end]},
            "shape": [self.ny, self.nx],
        }


def _projection_seed(grid, target, window=None):
    seed = {name: float(getattr(grid, name)) for name in
            ("hemi", "cone", "rebydx", "polei", "polej", "truelat1", "stand_lon")}
    seed.update(native_nx=int(grid.e_we)-1, native_ny=int(grid.e_sn)-1, target=target.as_dict())
    seed.update(i_start=window.i_start if window else 0, j_start=window.j_start if window else 0,
                source_nx=window.nx if window else seed["native_nx"],
                source_ny=window.ny if window else seed["native_ny"])
    return seed


def source_window_for(target: LatLonTarget, row: SourceRow, *, extra_cells: int = 2) -> SourceWindow:
    """Rust projects the target and bounds the source crop before decode."""
    import importlib
    loader = importlib.import_module(row.loader_module)
    grid = getattr(loader, row.source_grid_function)()
    seed = _projection_seed(grid, target)
    seed.update(halo=row.parabolic_halo_cells, extra_cells=extra_cells)
    encoded = json.dumps(seed, separators=(",", ":")).encode("utf-8")
    library, _ = _native_projected_entry()
    entry = library.gpuwm_projected_window_v1
    entry.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int64),
                      ctypes.c_void_p, ctypes.c_size_t]
    entry.restype = ctypes.c_int
    bounds = (ctypes.c_int64 * 4)()
    error = ctypes.create_string_buffer(4096)
    code = entry(encoded, len(encoded), bounds, error, len(error))
    if code:
        raise IntermediateRefusal(f"the target interpolation halo leaves {row.name} coverage: "
                                  + error.value.decode("utf-8", errors="replace"))
    return SourceWindow(*bounds)

def loader_envelope(loader: Any) -> tuple[float, float, float, float]:
    try:
        from gpuwm.ingest.hrrr_target import hrrr_coverage_envelope

        return tuple(float(v) for v in hrrr_coverage_envelope())  # type: ignore[return-value]
    except Exception:  # pragma: no cover - only the refusal text degrades
        return (float("nan"),) * 4


# ---------------------------------------------------------------------------
# the decoder
# ---------------------------------------------------------------------------
def parse_cycle(text: str) -> datetime:
    for pattern in ("%Y-%m-%dT%H", "%Y-%m-%dT%H:%M", "%Y-%m-%d_%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d%H"):
        try:
            return datetime.strptime(str(text).strip(), pattern)
        except ValueError:
            continue
    raise IntermediateRefusal(
        f"--cycle {text!r} is not a cycle; write it as YYYY-MM-DDTHH (UTC)"
    )


def parse_hours(text: str) -> tuple[int, ...]:
    hours: set[int] = set()
    for piece in str(text).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            first, last = piece.split("-", 1)
            hours.update(range(int(first), int(last) + 1))
        else:
            hours.add(int(piece))
    ordered = tuple(sorted(hours))
    if not ordered:
        raise IntermediateRefusal(f"--hours {text!r} names no forecast hour")
    if ordered != tuple(range(ordered[0], ordered[-1] + 1)):
        raise IntermediateRefusal(
            f"--hours {text!r} is not contiguous; the boundary producer admits "
            f"files by valid time and a gap would freeze the boundary across it"
        )
    return ordered


def resolve_decoder(row: SourceRow, explicit: Path | None = None) -> Path:
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if not candidate.is_file():
            raise IntermediateRefusal(f"--decoder {candidate} is not a file")
        return candidate
    try:
        from gpuwm import bridges
    except ImportError as error:
        raise IntermediateRefusal(
            f"gpuwm is not importable ({error}); this door drives the engine's "
            f"own {row.decoder_key} decoder and cannot find it without the "
            f"engine installed"
        ) from error
    try:
        return Path(bridges.resolve_source_decoder(row.decoder_key))
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise IntermediateRefusal(
            f"the engine's {row.decoder_key} decoder could not be resolved: "
            f"{error}.  `gpuwm fetch-bridges` stages it"
        ) from error


def decode_window(
    row: SourceRow,
    *,
    decoder: Path,
    grib_dir: Path,
    cycle: datetime,
    hours: Sequence[int],
    window: SourceWindow,
    out_dir: Path,
    workers: int,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run the engine's decoder over the fetched hours into its native window."""

    grib_dir = Path(grib_dir).expanduser().absolute()
    series = out_dir / "series.tsv"
    rows: list[str] = []
    inputs: list[dict[str, Any]] = []
    for hour in hours:
        atmosphere, soil = row.hour_files(grib_dir, cycle, int(hour))
        for path in (atmosphere, soil):
            if not path.is_file():
                raise IntermediateRefusal(
                    f"{path} is not a file.  `gpuwm fetch --source {row.name} "
                    f"--cycle {cycle:%Y-%m-%dT%H} --hours {max(hours)} --out "
                    f"{grib_dir}` writes it"
                )
        rows.append(f"{int(hour)}\t{atmosphere}\t{soil}\n")
        inputs.append({"hour": int(hour), "atmosphere": str(atmosphere), "soil": str(soil)})
    series.write_text("".join(rows), encoding="utf-8", newline="\n")
    native = out_dir / "native"
    if native.exists():
        raise IntermediateRefusal(
            f"{native} exists; the decoder refuses to overwrite a published "
            f"window.  Pass a fresh --out-dir"
        )
    workers = max(1, min(13, int(workers)))
    argv = [
        str(decoder), "--series-workers", str(workers), str(series), str(native),
        f"{cycle:%Y-%m-%d %H:00:00}",
        str(window.i_start), str(window.i_end), str(window.j_start), str(window.j_end),
    ]
    log(f"DECODE {row.decoder_key}: {len(hours)} hour(s), window "
        f"{window.ny}x{window.nx}, {workers} worker(s) ...")
    started = time.perf_counter()
    completed = subprocess.run(argv, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs" / "decode.log").write_text(
        f"$ {' '.join(argv)}\n[{elapsed:.1f} s, exit {completed.returncode}]\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise IntermediateRefusal(
            f"{decoder.name} exited {completed.returncode}; its own message is the "
            f"reason and it is in {out_dir / 'logs' / 'decode.log'}"
            + (f".  Last line: {tail[-1]}" if tail else "")
        )
    log(f"DECODE done in {elapsed:.1f} s -> {native}")
    manifest = _write_manifest(native)
    return {
        "decoder": str(decoder),
        "decoder_sha256": sha256_file(decoder),
        "argv": argv,
        "seconds": round(elapsed, 2),
        "series": str(series),
        "inputs": inputs,
        "native": str(native),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
    }


def _write_manifest(root: Path) -> Path:
    manifest = root / "SHA256SUMS"
    payloads = sorted(p for p in root.rglob("*") if p.is_file() and p != manifest)
    lines = [f"{sha256_file(p)}  ./{p.relative_to(root).as_posix()}" for p in payloads]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# the regrid
# ---------------------------------------------------------------------------
def _engine_pieces(row: SourceRow) -> dict[str, Any]:
    """The engine functions this road drives, resolved once and named once."""

    import importlib

    try:
        loader = importlib.import_module(row.loader_module)
        soil = importlib.import_module("gpuwm.ingest.soil")
        backend_module = importlib.import_module("gpuwm.ingest.preprocess_backend")
        producer_path, producer_name = row.producer_class.split(":", 1)
        producer_module = importlib.import_module(producer_path)
    except ImportError as error:
        raise IntermediateRefusal(
            f"the engine's ingest is not importable ({error}); this door drives "
            f"the engine's native decoder, projected metadata and interpolation operator and "
            f"has no road without them"
        ) from error
    needed = {
        "source_grid": getattr(loader, row.source_grid_function, None),
        "soil_nodes_m": getattr(soil, row.soil_depth_metadata[0], None),
        "noah_midpoints_m": getattr(soil, row.soil_depth_metadata[1], None),
        "resolve_backend": getattr(backend_module, "resolve_preprocess_backend", None),
        "producer": getattr(producer_module, producer_name, None),
    }
    missing = sorted(name for name, value in needed.items() if value is None)
    if missing:
        raise IntermediateRefusal(
            f"the installed engine's ingest does not expose {missing}; this "
            f"road drives the operator the engine ships and refuses to "
            f"substitute one of its own"
        )
    return needed


def _native_projected_entry():
    from gpuwm.bridges import artifact_candidates

    name = ("rw_projected_intermediate.dll" if os.name == "nt" else
            "librw_projected_intermediate.dylib" if sys.platform == "darwin" else
            "librw_projected_intermediate.so")
    candidates = list(artifact_candidates("GPUWM_PROJECTED_INTERMEDIATE_BRIDGE", name))
    # The entry is built in rw_wps, alongside its projected-source frontend.
    import gpuwm
    root = Path(gpuwm.__file__).resolve().parent.parent
    candidates.extend(root / "tools" / "rw_wps" / "target" / arm / name
                      for arm in ("release", "debug"))
    explicit = os.environ.get("GPUWM_PROJECTED_INTERMEDIATE_BRIDGE")
    if explicit and not Path(explicit).is_file():
        raise IntermediateRefusal(f"GPUWM_PROJECTED_INTERMEDIATE_BRIDGE={explicit} is not a file")
    for path in candidates:
        if path.is_file():
            try:
                library = ctypes.CDLL(str(path))
                entry = library.gpuwm_projected_intermediate_v1
                library.gpuwm_projected_geometry_v1
                library.gpuwm_projected_geometry_release_v1
                library.gpuwm_projected_window_v1
            except (OSError, AttributeError) as error:
                raise IntermediateRefusal(
                    f"{path} has no usable projected MET entry ({error}); "
                    "wind rotation, soil layering and MET writing cannot run. "
                    "Build tools/rw_wps with cargo build --release -p projected-intermediate"
                ) from error
            entry.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            entry.restype = ctypes.c_int
            return library, entry
    raise IntermediateRefusal(
        f"{name} was not found; projected MET fields cannot be regridded or written. "
        "Build tools/rw_wps with cargo build --release -p projected-intermediate; "
        "stage the library with the bridge bundle or name GPUWM_PROJECTED_INTERMEDIATE_BRIDGE"
    )


def _field_spec(source: str, name: str, units: str, description: str, method: str) -> dict[str, str]:
    transform = {"SPECHUMD": "nonnegative", "SNOW": "nonnegative",
                 "LANDSEA": "land", "SEAICE": "fraction"}.get(name, "none")
    return dict(source=source, field=name, units=units, description=description,
                method=method, transform=transform)


class ProjectedMetPlan:
    """Retain the engine's immutable donor geometry across hourly native calls.

    Only geometry and table metadata cross Python. Rust reads window fields,
    rotates winds, interpolates soil nodes, regrids, clips, writes and verifies.
    The engine's FP64 donor decisions are kept separate from FP32 fractions.
    """
    def __init__(self, pieces, snapshot, target, backend):
        import weakref
        self.library, self.entry = _native_projected_entry()
        if not backend.indexed_donor_interp:
            raise IntermediateRefusal("the CPU bridge lacks gpuwm_indexed_interp_f32; exact projected MET donors cannot be evaluated. Rebuild tools/grib1_bridge")
        self.backend = backend
        self.key = (snapshot.i_start, snapshot.j_start, snapshot.ny, snapshot.nx, target)
        grid = pieces["source_grid"]()
        seed = _projection_seed(grid, target,
                                SourceWindow(snapshot.i_start, snapshot.i_start+snapshot.nx-1,
                                             snapshot.j_start, snapshot.j_start+snapshot.ny-1))
        encoded = json.dumps(seed, separators=(",", ":")).encode("utf-8")
        error = ctypes.create_string_buffer(4096)
        interpolate = ctypes.cast(backend._native._library.gpuwm_indexed_interp_f32, ctypes.c_void_p)
        create = self.library.gpuwm_projected_geometry_v1
        create.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        create.restype = ctypes.c_void_p
        self.geometry = create(encoded, len(encoded), interpolate, error, len(error))
        if not self.geometry:
            raise IntermediateRefusal(error.value.decode("utf-8", errors="replace"))
        release = self.library.gpuwm_projected_geometry_release_v1
        release.argtypes = [ctypes.c_void_p]
        release.restype = None
        self._release = weakref.finalize(self, release, self.geometry)
        self.workers = int(backend.workers)

    def close(self):
        self._release()


@dataclass(frozen=True)
class WindowFile:
    filename: Path


@dataclass(frozen=True)
class DecodedWindowFiles:
    valid_time: datetime
    forecast_hour: int
    i_start: int
    j_start: int
    ny: int
    nx: int
    fields: Mapping[str, WindowFile]


def decoded_window_files(row, root, cycle, hour, window):
    """Describe producer-owned files; Rust opens and validates every field."""
    if not all(row.decoded_directories):
        raise IntermediateRefusal("the source row has no decoded directory metadata; MET fields cannot be located")
    atmosphere, soil = (Path(root) / pattern.format(hour=hour) for pattern in row.decoded_directories)
    names = set(row.atmosphere_3d) | set(row.surface) | set(row.wind_3d) | set(row.wind_10m)
    fields = {name: WindowFile(atmosphere / f"{name}.f32le") for name in names}
    fields.update({name: WindowFile(soil / f"{name}.f32le")
                   for name in (row.soil_temperature, row.soil_moisture)})
    for name in ("QNWFA", "QNIFA"):
        path = atmosphere / f"{name}.f32le"
        if path.is_file():
            fields[name] = WindowFile(path)
    return DecodedWindowFiles(cycle + timedelta(hours=hour), hour, window.i_start, window.j_start,
                              window.j_end - window.j_start + 1, window.i_end - window.i_start + 1, fields)

def regrid_hour(
    row: SourceRow, pieces: Mapping[str, Any], snapshot: Any, target: LatLonTarget,
    *, backend: Any, path: Path, plan: ProjectedMetPlan | None = None,
) -> tuple[dict[str, Any], ProjectedMetPlan]:
    """Drive the native projected MET entry. There is no Python field route."""
    if plan is None:
        plan = ProjectedMetPlan(pieces, snapshot, target, backend)
    if plan.key != (snapshot.i_start, snapshot.j_start, snapshot.ny, snapshot.nx, target):
        raise IntermediateRefusal("source window changed between hours; reusing its donors would misplace MET fields")
    if not plan._release.alive:
        raise IntermediateRefusal("projected geometry has been released; it cannot supply MET donors")
    required = set(row.atmosphere_3d) | set(row.surface) | set(row.wind_3d) | set(row.wind_10m)
    required.update((row.soil_temperature, row.soil_moisture))
    aerosol = [name for name in ("QNWFA", "QNIFA") if name in snapshot.fields]
    required.update(aerosol)
    sources = {}
    for name in sorted(required):
        value = snapshot.fields[name]
        filename = getattr(value, "filename", None)
        if filename is None:
            raise IntermediateRefusal(f"{name} is not a decoder window file; the native entry reads authenticated window bytes")
        sources[name] = str(filename)
    receipt_path = path.with_name(path.name + ".native.json")
    spec = {
        "output": str(path), "receipt": str(receipt_path),
        "valid_time": snapshot.valid_time.strftime("%Y-%m-%d_%H:%M:%S"),
        "forecast_hour": float(snapshot.forecast_hour), "map_source": row.map_source,
        "target": target.as_dict(), "levels": row.levels, "workers": plan.workers,
        "sources": sources,
        "atmosphere": [_field_spec(source, *metadata, "parabolic")
                       for source, metadata in row.atmosphere_3d.items()] +
                      [_field_spec(name, name, "kg-1", "Analysis aerosol number", "parabolic") for name in aerosol],
        "surface": [_field_spec(source, *metadata) for source, metadata in row.surface.items()],
        "wind_3d": row.wind_3d, "wind_10m": row.wind_10m,
        "soil": {"temperature": row.soil_temperature, "moisture": row.soil_moisture,
                 "nodes": pieces["soil_nodes_m"].tolist(),
                 "midpoints": pieces["noah_midpoints_m"].tolist(), "names": row.soil_layer_names,
                 "preserve_nodes": True},
        "admission": {"land": next(k for k, v in row.surface.items() if v[0] == "LANDSEA"),
                      "ice": next(k for k, v in row.surface.items() if v[0] == "SEAICE"),
                      "soil_temperature": row.soil_temperature, "soil_moisture": row.soil_moisture,
                      "humidity": [k for k, v in row.atmosphere_3d.items() if v[0] == "SPECHUMD"]
                                  + [k for k, v in row.surface.items() if v[0] == "SPECHUMD"]},
    }
    encoded = json.dumps(spec, separators=(",", ":")).encode("utf-8")
    error = ctypes.create_string_buffer(4096)
    code = plan.entry(encoded, len(encoded), plan.geometry, error, len(error))
    if code:
        raise IntermediateRefusal(error.value.decode("utf-8", errors="replace"))
    return json.loads(receipt_path.read_text(encoding="utf-8")), plan

# ---------------------------------------------------------------------------
# the writer
# ---------------------------------------------------------------------------
def _text(value: str, width: int) -> bytes:
    raw = value.encode("ascii", errors="strict")
    if len(raw) > width:
        raise IntermediateRefusal(f"{value!r} does not fit a {width}-character WPS field")
    return raw.ljust(width, b" ")


def _record(payload: bytes) -> bytes:
    marker = struct.pack(">i", len(payload))
    return marker + payload + marker


def wps_record_bytes(
    *,
    hdate: str,
    xfcst: float,
    map_source: str,
    field_name: str,
    units: str,
    description: str,
    level: float,
    target: LatLonTarget,
    values: np.ndarray,
    earth_radius_km: float = 6371.229,
) -> bytes:
    """One WPS version-5 field group on a regular lat-lon grid, big-endian."""

    values = np.asarray(values, dtype=np.float32)
    if values.shape != (target.ny, target.nx):
        raise IntermediateRefusal(
            f"{field_name} slab is {values.shape}, the target is ({target.ny}, {target.nx})"
        )
    if not np.isfinite(values).all():
        raise IntermediateRefusal(
            f"{field_name} at level {level:g} carries a non-finite value; a "
            f"structurally perfect record with NaN in it is exactly the file "
            f"this door refuses to write"
        )
    header = (
        _text(hdate, 24) + struct.pack(">f", float(xfcst)) + _text(map_source, 32)
        + _text(field_name, 9) + _text(units, 25) + _text(description, 46)
        + struct.pack(">f", float(level)) + struct.pack(">iii", target.nx, target.ny, 0)
    )
    projection = (
        _text("SWCORNER", 8)
        + struct.pack(">fffff", float(target.south), float(target.west),
                      float(target.dlat), float(target.dlon), float(earth_radius_km))
    )
    # The record is x-fastest (Fortran ``(nx, ny)``), which is exactly the
    # C-order byte stream of the row-major ``(ny, nx)`` slab; the reader
    # reshapes it as ``(nx, ny), order="F"`` and gets this array's transpose.
    slab = np.ascontiguousarray(values, dtype=">f4").tobytes(order="C")
    return (
        _record(struct.pack(">i", WPS_VERSION))
        + _record(header)
        + _record(projection)
        + _record(struct.pack(">i", 0))
        + _record(slab)
    )


def write_intermediate(
    path: Path, records: Sequence[Mapping[str, Any]], *, valid_time: datetime,
    forecast_hour: float, map_source: str, target: LatLonTarget,
) -> dict[str, Any]:
    hdate = valid_time.strftime("%Y-%m-%d_%H:%M:%S")
    with open(path, "wb") as handle:
        for item in records:
            handle.write(wps_record_bytes(
                hdate=hdate, xfcst=forecast_hour, map_source=map_source,
                field_name=str(item["field"]), units=str(item["units"]),
                description=str(item["description"]), level=float(item["level"]),
                target=target, values=item["values"],
            ))
    # Read back through the tree's own reader before claiming anything.
    from .wps_intermediate import inventory

    seen = inventory(path)
    if int(seen["field_records"]) != len(records):
        raise IntermediateRefusal(
            f"{path} reads back as {seen['field_records']} records after "
            f"{len(records)} were written; the writer and the reader disagree"
        )
    for got, want in zip(seen["records"], records):
        if got["field"] != want["field"] or got["nx"] != target.nx or got["ny"] != target.ny \
                or abs(float(got["level"]) - float(want["level"])) > 1e-3:
            raise IntermediateRefusal(
                f"{path}: record {want['field']}@{want['level']:g} read back as "
                f"{got['field']}@{got['level']:g} ({got['nx']}x{got['ny']})"
            )
    return {
        "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path),
        "valid_time": hdate, "forecast_hour": forecast_hour,
        "field_records": int(seen["field_records"]), "fields": seen["fields"],
        "record_endian": seen["record_endian"],
    }


# ---------------------------------------------------------------------------
# the door
# ---------------------------------------------------------------------------
@dataclass
class IntermediateRequest:
    source: SourceRow
    grib_dir: Path
    cycle: datetime
    hours: tuple[int, ...]
    target: LatLonTarget
    out_dir: Path
    decoder: Path | None = None
    workers: int = 4
    prefix: str = "MET"
    target_basis: Mapping[str, Any] = field(default_factory=dict)


_LEDGER_WARNED: set[str] = set()


def _ledger_emit(path: Path, stage: str, phase: str, attrs: Mapping[str, Any]) -> None:
    """C1 v1, one O_APPEND write, never a model failure on telemetry failure."""
    try:
        from gpuwm.timeline.ledger import emit
    except ImportError:
        pass  # The frozen hex bootstrap predates the separately delivered C1 package.
    else:
        try:
            emit(stage, phase, path=path, product="hex", attrs=attrs)
        except BaseException:
            pass
        return
    destination = os.environ.get("GPUWM_TIME_LEDGER") or str(path)
    try:
        event = {"v": 1, "t_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                 "t_mono": time.monotonic(), "host": socket.gethostname(),
                 "rental_id": os.environ.get("GPUWM_RENTAL_ID"), "run_id": os.environ.get("GPUWM_RUN_ID"),
                 "product": "hex", "stage": stage, "phase": phase, "card": None,
                 "pid": os.getpid(), "attrs": dict(attrs)}
        data = (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            if os.write(fd, data) != len(data):
                raise OSError("short append")
        finally:
            os.close(fd)
    except Exception as error:
        if destination not in _LEDGER_WARNED:
            _LEDGER_WARNED.add(destination)
            # A warnings filter set to 'error' must not turn telemetry into a
            # forecast failure either.
            try:
                warnings.warn(f"timeline append failed for {destination}: {error}", RuntimeWarning)
            except Exception:
                pass


def _ensure_hour(request: IntermediateRequest, hour: int, log: Callable[[str], None]) -> Path:
    """Fetch I/O only. This runs in the one prefetch thread, never a process pool."""
    row = request.source
    root = Path(request.grib_dir).expanduser().absolute()
    if all(path.is_file() for path in row.hour_files(root, request.cycle, hour)):
        return root
    if not row.fetch_function:
        raise IntermediateRefusal(f"source row {row.name} has no fetch transport for missing hour {hour}")
    import importlib
    from gpuwm.fetch import select_fetch_engine

    module, function = row.fetch_function.split(":", 1)
    fetch = getattr(importlib.import_module(module), function)
    choice = select_fetch_engine("rust", progress=log)
    root = Path(request.out_dir).absolute() / "fetch" / f"f{hour:02d}"
    root.mkdir(parents=True, exist_ok=True)
    ledger = Path(request.out_dir).absolute() / "timeline.jsonl"
    _ledger_emit(ledger, "fetch.intermediate.hour", "begin", {"hour": hour})
    started = time.perf_counter()
    try:
        fetch(cycle=request.cycle, hours=(hour,), area=None, out=root, workers=request.workers,
              file_workers=1, engine=choice.engine, engine_bin=choice.binary,
              engine_selection=choice.selection, transport="s3", progress=log,
              cache_dir=Path(request.out_dir).absolute() / "fetch-cache")
    except Exception:
        _ledger_emit(ledger, "fetch.intermediate.hour", "end",
                     {"hour": hour, "seconds": time.perf_counter() - started, "status": "failed"})
        raise
    _ledger_emit(ledger, "fetch.intermediate.hour", "end",
                 {"hour": hour, "seconds": time.perf_counter() - started, "status": "complete"})
    if not all(path.is_file() for path in row.hour_files(root, request.cycle, hour)):
        raise IntermediateRefusal(f"fetch completed without both source files for hour {hour}; no decode can start")
    return root


def _admit_hour(row, grib_root, cycle, hour, admissions):
    """Bind the bytes admitted to the one live Rust producer before it reads them."""
    atmosphere, soil = row.hour_files(grib_root, cycle, hour)
    inputs = {"hour": hour, "atmosphere": str(atmosphere), "soil": str(soil),
              "atmosphere_sha256": sha256_file(atmosphere), "soil_sha256": sha256_file(soil)}
    path = admissions / f"f{hour:02d}.admitted"
    temporary = path.with_suffix(".partial")
    temporary.write_text(f"forecast_hour\t{hour}\natmosphere\t{atmosphere}\nsoil\t{soil}\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, path)
    return inputs


def _cancel_producer(producer):
    """Use a caller's owned-process stop helper when its host requires one."""
    command = os.environ.get("GPUWM_CHILD_STOP_COMMAND")
    process = getattr(producer, "process", None)
    if command and process is not None and process.poll() is None:
        argv = json.loads(command)
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
            raise IntermediateRefusal("GPUWM_CHILD_STOP_COMMAND must name an argv list for the owned decoder stop helper")
        subprocess.run([*argv, str(process.pid)], check=True, capture_output=True, text=True)
        process.wait(timeout=30)
    producer.cancel()


def build_intermediates(request: IntermediateRequest, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    row = request.source
    out_dir = Path(request.out_dir).expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob(f"{request.prefix}:*")):
        raise IntermediateRefusal(f"{out_dir} holds intermediate hours; pass a fresh --out-dir so completion markers cannot describe an earlier run")
    pieces = _engine_pieces(row)
    decoder = resolve_decoder(row, request.decoder)
    window = source_window_for(request.target, row)
    if not request.hours or tuple(sorted(set(request.hours))) != tuple(request.hours):
        raise IntermediateRefusal("forecast hours must be unique and ordered; per-hour completion cannot be ambiguous")
    # Resolve the native entry before starting fetch. No fallback keeps the
    # Python field path alive when the native artifact is absent.
    _native_projected_entry()
    backend = pieces["resolve_backend"]("cpu", workers=request.workers)
    files: list[dict[str, Any]] = []
    decoded_hours: list[dict[str, Any]] = []
    plan = None
    operator = None
    ledger = out_dir / "timeline.jsonl"
    if len(request.hours) < 2 or request.hours != tuple(range(request.hours[0], request.hours[-1]+1)):
        raise IntermediateRefusal("the decoder series requires at least two contiguous hours; a missing boundary interval cannot be published")
    admissions = out_dir / "admissions"
    admissions.mkdir()
    series = out_dir / "series.tsv"
    predicted = {}
    for hour in request.hours:
        root = Path(request.grib_dir).expanduser().absolute()
        if not all(path.is_file() for path in row.hour_files(root, request.cycle, hour)):
            root = out_dir / "fetch" / f"f{hour:02d}"
        predicted[hour] = root
    series.write_text("".join(f"{hour}\t{a}\t{s}\n" for hour in request.hours
                              for a, s in [row.hour_files(predicted[hour], request.cycle, hour)]),
                      encoding="utf-8", newline="\n")
    native = out_dir / "native"
    producer = pieces["producer"](
        decoder=decoder, series=series, output=native, signals=out_dir / "signals",
        admissions=admissions, cycle=request.cycle.strftime("%Y-%m-%d %H:00:00"),
        window=(window.i_start, window.i_end, window.j_start, window.j_end),
        workers=str(max(1, min(row.decoder_workers_max, request.workers))), log=out_dir / "logs" / "decode.log")
    # One I/O thread fetches one missing lead ahead of ordered consumption.
    # Existing immutable input pairs are admitted immediately so inventory and
    # decode stay parallel on warm packs. Rust owns the shared field budget.
    condition = Condition()
    stop = Event()
    admitted = {}
    consuming = [0]

    def fetch_series():
        try:
            for index, hour in enumerate(request.hours):
                if predicted[hour] != Path(request.grib_dir).expanduser().absolute():
                    with condition:
                        while index > consuming[0] + 1 and not stop.is_set():
                            condition.wait(0.02)
                if stop.is_set():
                    return
                root = _ensure_hour(request, int(hour), log)
                if root != predicted[hour]:
                    raise IntermediateRefusal("fetched root differs from the admitted series; refuses a different source window")
                started = time.perf_counter()
                attrs = {"hour": int(hour), "decode_s": 0.0, "regrid_s": 0.0}
                _ledger_emit(ledger, "prep.intermediate.hour", "begin", attrs)
                try:
                    inputs = _admit_hour(row, root, request.cycle, int(hour), admissions)
                except Exception:
                    _ledger_emit(ledger, "prep.intermediate.hour", "end", {**attrs, "status": "failed"})
                    raise
                with condition:
                    admitted[hour] = (inputs, started)
                    condition.notify_all()
        finally:
            with condition:
                condition.notify_all()
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="intermediate-fetch") as prefetch:
            producer.start()
            fetched = prefetch.submit(fetch_series)
            for index, hour in enumerate(request.hours):
                hour = int(hour)
                with condition:
                    consuming[0] = index
                    condition.notify_all()
                    while hour not in admitted and not fetched.done():
                        condition.wait(0.02)
                    if hour not in admitted:
                        fetched.result()
                        raise IntermediateRefusal(f"source hour {hour} was never admitted; no completion can be published")
                    inputs, started = admitted[hour]
                attrs = {"hour": hour, "decode_s": 0.0, "regrid_s": 0.0}
                try:
                    if index == 0:
                        producer.wait_preflight()
                    staging = producer.wait_hour(hour)
                    attrs["decode_s"] = time.perf_counter() - started
                    decoded_hours.append({"native": str(native), "native_during_consume": str(staging),
                                          "seconds": attrs["decode_s"], "inputs": [inputs]})
                    snapshot = decoded_window_files(row, staging, request.cycle, hour, window)
                    valid = request.cycle + timedelta(hours=hour)
                    path = out_dir / f"{request.prefix}:{valid:%Y-%m-%d_%H}"
                    regrid_start = time.perf_counter()
                    written, plan = regrid_hour(row, pieces, snapshot, request.target,
                                               backend=backend, path=path, plan=plan)
                    attrs["regrid_s"] = time.perf_counter() - regrid_start
                    written["regrid_seconds"] = attrs["regrid_s"]
                    operator = written["operator"]
                    marker = path.with_name(path.name + ".ready.json")
                    temporary = marker.with_name(marker.name + ".partial")
                    completion = {"schema": "gpuwm-hex.intermediate.hour/v1", "hour": hour,
                                  **written, **attrs}
                    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                        handle.write(json.dumps(completion, sort_keys=True) + "\n")
                    os.replace(temporary, marker)
                    written["completion_marker"] = str(marker)
                    files.append(written)
                    _ledger_emit(ledger, "prep.intermediate.hour", "end", {**attrs, "status": "complete"})
                    _ledger_emit(ledger, "prep.intermediate.hour", "mark",
                                 {**attrs, "path": str(path), "marker": str(marker), "sha256": written["sha256"]})
                    log(f"WROTE {path.name}: {written['field_records']} records, "
                        f"{written['bytes'] / 1e6:.1f} MB, {attrs['regrid_s']:.3f} s")
                    del snapshot
                except Exception:
                    _ledger_emit(ledger, "prep.intermediate.hour", "end", {**attrs, "status": "failed"})
                    stop.set()
                    raise
            fetched.result()
        # Record the producer's generated staging paths before its normal cleanup.
        staging_files = [{"path": str(p), "bytes": p.stat().st_size}
                         for p in staging.rglob("*") if p.is_file() and not p.is_symlink()]
        (out_dir / "staging-cleanup.json").write_text(json.dumps(staging_files, indent=2)+"\n", encoding="utf-8")
        pipeline = producer.finish()
    except Exception as error:
        # This is the existing producer's owned-process failure unwind. Its
        # in-memory tail and exact PID-owned staging are captured before cleanup.
        _cancel_producer(producer)
        if plan is not None:
            plan.close()
        raise IntermediateRefusal(f"posted decoder/MET pipeline failed: {error}; log {out_dir / 'logs' / 'decode.log'}") from error
    manifest = _write_manifest(native)
    decoded = {"seconds": sum(item["seconds"] for item in decoded_hours), "hours": decoded_hours,
               "native": str(native), "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
               "pipeline": pipeline, "decoder": str(decoder), "decoder_sha256": sha256_file(decoder)}
    if plan is not None:
        plan.close()
    receipt = {
        "schema": INTERMEDIATE_SCHEMA,
        "source": row.name,
        "source_description": row.description,
        "cycle": f"{request.cycle:%Y-%m-%d_%H:%M:%S}",
        "hours": list(request.hours),
        "grib_dir": str(Path(request.grib_dir).absolute()),
        "target": request.target.as_dict(),
        "target_basis": dict(request.target_basis),
        "source_window": window.as_dict(),
        "decode": decoded,
        "regrid": {
            "operator": operator,
            "engine": "rw_wps projected-intermediate, using gpuwm_indexed_interp_f32 "
                      "and the engine projected plan's FP64 donors and rotation seed",
            "soil": "Rust FP64 node interpolation to the declared layer midpoints, "
                    "moved by nearest neighbour",
            "categorical": "nearest neighbour (LANDSEA, SEAICE)",
        },
        "levels": {
            "count": row.levels + 1,
            "convention": "level-indexed hybrid slabs (xlvl 1..N) with the 3-D PRESSURE "
                          "field carried, plus the surface level 200100.0",
            "init_switches": {
                "--nfglevels": row.levels + 1, "--nfgsoillevels": len(row.soil_layer_names),
                "--use-spechumd": "yes", "--extrap-airtemp": "constant",
                "why_constant": "the source column tops below the 30 km model top; "
                                "lapse-rate extrapolation above the first-guess top is "
                                "the Fortran's own fatal (docs/source-matrix.md, IFS row)",
            },
        },
        "files": files,
        "not_carried": ["QC", "QI", "QR", "QS", "QG", "SNOWH", "PMSL"],
        "why_not_carried": "the init and boundary writers read no hydrometeor record "
                           "from an intermediate file (the init stream's qc and qr "
                           "are written as zero) and the init stream has no slot for "
                           "cloud ice, snow or graupel (hour zero has no ice; README, "
                           "limited-area lane); the engine's window carries no "
                           "mean-sea-level pressure and the init does not require one",
    }
    receipt_path = out_dir / "intermediate-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    log(f"RECEIPT {receipt_path}")
    return receipt


def add_intermediate_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "intermediate",
        help="write regular lat-lon WPS intermediates from a projected regional "
             "source (HRRR) through the engine's own decoder and operator",
        description=(
            "Resample a projected regional source onto a regular lat-lon WPS "
            "intermediate the init and boundary engines read.  The decode, the "
            "interpolation operator and the soil layering are gpuwm's own; the "
            "record layout is the inverse of this tree's reader and every file "
            "is read back before the receipt is written."
        ),
    )
    parser.add_argument("--source", default="hrrr", metavar="ROW",
                        help=f"a row of SOURCE_ROWS ({', '.join(sorted(SOURCE_ROWS))})")
    parser.add_argument("--grib-dir", type=Path, required=True, metavar="DIR",
                        help="where `gpuwm fetch --source hrrr` wrote the cycle")
    parser.add_argument("--cycle", required=True, metavar="YYYY-MM-DDTHH",
                        help="the cycle the files came from (UTC)")
    parser.add_argument("--hours", default="0-3", metavar="SPEC",
                        help="forecast hours to write, contiguous (default 0-3)")
    parser.add_argument("--from-plan", type=Path, default=None, metavar="JSON",
                        help="derive the target box from a mesh-plan --point receipt's cull region")
    parser.add_argument("--point", default=None, metavar="LAT,LON",
                        help="centre of the target box (with --radius-km) instead of --from-plan")
    parser.add_argument("--radius-km", type=float, default=None, metavar="KM",
                        help="half-reach of the target box around --point")
    parser.add_argument("--margin-km", type=float, default=DEFAULT_MARGIN_KM, metavar="KM",
                        help=f"ground kept past the reach (default {DEFAULT_MARGIN_KM:g})")
    parser.add_argument("--spacing-deg", type=float, default=DEFAULT_SPACING_DEG, metavar="DEG",
                        help=f"target lat-lon spacing (default {DEFAULT_SPACING_DEG:g})")
    parser.add_argument("--out-dir", type=Path, required=True, metavar="DIR")
    parser.add_argument("--prefix", default="MET", metavar="TEXT",
                        help="file prefix, ungrib style PREFIX:YYYY-MM-DD_HH (default MET)")
    parser.add_argument("--decoder", type=Path, default=None, metavar="FILE",
                        help="the engine's decoder binary (default: gpuwm's bridge ladder)")
    parser.add_argument("--workers", type=int, default=4, metavar="N",
                        help="decoder workers, 1..13 (default 4)")
    parser.set_defaults(handler=run_intermediate)


def request_from_arguments(arguments: argparse.Namespace) -> IntermediateRequest:
    row = source_row(arguments.source)
    if arguments.from_plan is not None and arguments.point is not None:
        raise IntermediateRefusal("--from-plan and --point were both given; the box comes from one")
    if arguments.from_plan is not None:
        target, basis = target_from_plan(
            arguments.from_plan, margin_km=arguments.margin_km, spacing_deg=arguments.spacing_deg,
        )
        basis["from_plan"] = str(arguments.from_plan)
    elif arguments.point is not None:
        from .mesh_point import parse_point

        if arguments.radius_km is None:
            raise IntermediateRefusal("--point needs --radius-km: the box has no reach without one")
        centre = parse_point(arguments.point)
        target = target_for_cap(centre, float(arguments.radius_km),
                                margin_km=arguments.margin_km, spacing_deg=arguments.spacing_deg)
        basis = {"centre_deg": list(centre), "reach_km": float(arguments.radius_km)}
    else:
        raise IntermediateRefusal(
            "neither --from-plan nor --point was given; the intermediate covers a "
            "box and nothing here guesses one"
        )
    basis["margin_km"] = float(arguments.margin_km)
    return IntermediateRequest(
        source=row, grib_dir=arguments.grib_dir, cycle=parse_cycle(arguments.cycle),
        hours=parse_hours(arguments.hours), target=target, out_dir=arguments.out_dir,
        decoder=arguments.decoder, workers=int(arguments.workers), prefix=str(arguments.prefix),
        target_basis=basis,
    )


def run_intermediate(arguments: argparse.Namespace) -> int:
    request = request_from_arguments(arguments)
    receipt = build_intermediates(request)
    print(json.dumps({
        "files": [item["path"] for item in receipt["files"]],
        "target": receipt["target"],
        "source_window": receipt["source_window"],
        "next": (
            f"gpuwm-hex init --met {receipt['files'][0]['path']} ... --nfglevels "
            f"{receipt['levels']['count']} --nfgsoillevels "
            f"{receipt['levels']['init_switches']['--nfgsoillevels']} "
            f"--use-spechumd yes --extrap-airtemp constant"
        ),
    }, indent=2))
    return 0


# ---------------------------------------------------------------------------
# the boundary door: rw_mpas_lbc over the hourly intermediates
# ---------------------------------------------------------------------------
def intermediate_valid_times(met_dir: Path) -> list[tuple[datetime, Path]]:
    """Every WPS intermediate in a directory, by the valid time in its own header."""

    from .init_door import _probe_wps_header
    from .wps_intermediate import WpsIntermediateReader

    found: list[tuple[datetime, Path]] = []
    for path in sorted(Path(met_dir).iterdir()):
        if not path.is_file() or not _probe_wps_header(path):
            continue
        with WpsIntermediateReader(path) as reader:
            first = next(reader.iter_fields(load_values=False), None)
        if first is None:
            continue
        found.append((datetime.strptime(first.valid_time[:19], "%Y-%m-%d_%H:%M:%S"), path))
    if not found:
        raise IntermediateRefusal(
            f"{met_dir} holds no WPS intermediate; `gpuwm-hex intermediate` writes them"
        )
    found.sort()
    return found


def add_lbc_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "lbc",
        help="build lateral boundary files with rw_mpas_lbc from hourly WPS intermediates",
        description=(
            "Drive rw_mpas_lbc on its wps-intermediate route: one boundary file per "
            "intermediate valid time between --start-time and --stop-time.  The "
            "first-guess switches have no defaults, exactly as the engine says."
        ),
    )
    parser.add_argument("--grid", type=Path, required=True, metavar="INIT.nc",
                        help="the child's initial-conditions file (the culled init)")
    parser.add_argument("--met-dir", type=Path, required=True, metavar="DIR",
                        help="directory of WPS intermediates, one per boundary time")
    parser.add_argument("--out-dir", type=Path, required=True, metavar="DIR")
    parser.add_argument("--start-time", required=True, metavar="YYYY-MM-DD_HH:MM:SS")
    parser.add_argument("--stop-time", required=True, metavar="YYYY-MM-DD_HH:MM:SS")
    parser.add_argument("--nfglevels", type=int, required=True)
    parser.add_argument("--extrap-airtemp", choices=("constant", "linear", "lapse-rate"), required=True)
    parser.add_argument("--use-spechumd", choices=("yes", "no"), required=True)
    parser.add_argument("--theta-adv-order", type=int, default=3)
    parser.add_argument("--coef-3rd-order", type=float, default=0.25)
    parser.add_argument("--oned-underflow", choices=("preserve", "reproduce-ifx-ftz"), default="preserve")
    parser.add_argument("--physics-mode", choices=("correct", "exact"), default="correct",
                        help="correct (default): boundary winds extrapolate like the initial conditions "
                             "(IN4) and every boundary file must carry its valid time (IN5); exact: "
                             "native init_atmosphere v8.4.2 case 9.  An rw_mpas_lbc without "
                             "--physics-mode writes case 9 and the door receipt names IN4/IN5 as gaps")
    parser.add_argument("--lbc-exe", type=Path, default=None, metavar="FILE")
    parser.add_argument("--stream", action="store_true",
                        help="publish each hash-checked LBC hour as its MET marker lands")
    parser.set_defaults(handler=run_lbc)


def run_lbc(arguments: argparse.Namespace) -> int:
    from .cycle.chain import resolve_lbc_engine
    from .cycle.errors import CycleRefusal

    try:
        engine = resolve_lbc_engine(arguments.lbc_exe)
    except CycleRefusal as error:
        raise IntermediateRefusal(str(error)) from error
    grid = Path(arguments.grid).expanduser().absolute()
    if not grid.is_file():
        raise IntermediateRefusal(f"--grid {grid} is not a file")
    if getattr(arguments, "stream", False):
        from .lbc_producer import run_stream
        return run_stream(arguments, engine)
    start = datetime.strptime(arguments.start_time, "%Y-%m-%d_%H:%M:%S")
    stop = datetime.strptime(arguments.stop_time, "%Y-%m-%d_%H:%M:%S")
    if stop <= start:
        raise IntermediateRefusal("--stop-time is not after --start-time")
    times = [(t, p) for t, p in intermediate_valid_times(arguments.met_dir) if start <= t <= stop]
    if not times or times[0][0] != start or times[-1][0] != stop:
        held = ", ".join(t.strftime("%Y-%m-%d_%H:%M:%S") for t, _ in intermediate_valid_times(arguments.met_dir))
        raise IntermediateRefusal(
            f"the intermediates in {arguments.met_dir} do not cover "
            f"{arguments.start_time}..{arguments.stop_time} at both ends; they hold "
            f"[{held}].  A boundary series with a missing end freezes the boundary "
            f"there without saying so"
        )
    gaps = {int((b - a).total_seconds()) for (a, _), (b, _) in zip(times, times[1:])}
    if len(gaps) != 1:
        raise IntermediateRefusal(
            f"the intermediates are not evenly spaced in time ({sorted(gaps)} s); "
            f"--fg-interval-seconds is one number"
        )
    interval = gaps.pop()
    from .lbc_producer import lbc_physics_mode
    physics, correct_record = lbc_physics_mode(engine, getattr(arguments, "physics_mode", "exact"))
    out_dir = Path(arguments.out_dir).expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt = out_dir / "lbc-receipt.json"
    argv = [
        str(engine), "--grid", str(grid), "--out-dir", str(out_dir),
        "--start-time", arguments.start_time, "--stop-time", arguments.stop_time,
    ]
    for moment, path in times:
        argv += ["--interval", f"{moment:%Y-%m-%d_%H:%M:%S}={path}"]
    argv += [
        "--nfglevels", str(int(arguments.nfglevels)),
        "--fg-interval-seconds", str(interval),
        "--extrap-airtemp", arguments.extrap_airtemp,
        "--use-spechumd", arguments.use_spechumd,
        "--theta-adv-order", str(int(arguments.theta_adv_order)),
        "--coef-3rd-order", repr(float(arguments.coef_3rd_order)),
        "--oned-underflow", arguments.oned_underflow,
        *physics,
        "--receipt", str(receipt),
    ]
    started = time.perf_counter()
    completed = subprocess.run(argv, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    (out_dir / "lbc.log").write_text(
        f"$ {' '.join(argv)}\n[{elapsed:.1f} s, exit {completed.returncode}]\n--- stdout ---\n"
        f"{completed.stdout}\n--- stderr ---\n{completed.stderr}\n", encoding="utf-8",
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise IntermediateRefusal(
            f"rw_mpas_lbc exited {completed.returncode}; its log is {out_dir / 'lbc.log'}"
            + (f".  Last line: {tail[-1]}" if tail else "")
        )
    files = sorted(out_dir.glob("lbc.*.nc"))
    if not files:
        raise IntermediateRefusal(f"rw_mpas_lbc exited 0 and wrote no lbc.*.nc into {out_dir}")
    door_receipt = {
        "schema": "gpuwm-hex.lbc-door/v1",
        "engine": str(engine), "engine_sha256": sha256_file(engine),
        "grid": str(grid), "argv": argv, "seconds": round(elapsed, 2),
        "interval_seconds": interval,
        "intermediates": [{"valid_time": f"{t:%Y-%m-%d_%H:%M:%S}", "path": str(p), "sha256": sha256_file(p)}
                          for t, p in times],
        "files": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files],
        "engine_receipt": str(receipt) if receipt.is_file() else None,
        # Correct mode only (an exact door receipt is the one written before the key existed).
        **({"correct_mode": correct_record} if correct_record else {}),
    }
    (out_dir / "lbc-door-receipt.json").write_text(
        json.dumps(door_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n",
    )
    print(json.dumps({"files": [str(p) for p in files], "seconds": round(elapsed, 2),
                      "receipt": str(out_dir / "lbc-door-receipt.json")}, indent=2))
    print(f"NEXT gpuwm-hex forecast ... --lbc-dir {out_dir}")
    return 0


__all__ = [
    "DEFAULT_MARGIN_KM",
    "DEFAULT_SPACING_DEG",
    "INTERMEDIATE_SCHEMA",
    "SOURCE_ROWS",
    "SURFACE_LEVEL",
    "IntermediateRefusal",
    "IntermediateRequest",
    "LatLonTarget",
    "SourceRow",
    "SourceWindow",
    "add_intermediate_parser",
    "add_lbc_parser",
    "build_intermediates",
    "decode_window",
    "intermediate_valid_times",
    "parse_cycle",
    "parse_hours",
    "regrid_hour",
    "resolve_decoder",
    "run_intermediate",
    "run_lbc",
    "source_row",
    "source_window_for",
    "target_for_cap",
    "target_from_plan",
    "wps_record_bytes",
    "write_intermediate",
]
