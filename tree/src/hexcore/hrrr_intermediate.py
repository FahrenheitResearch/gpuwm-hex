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
* REGRID -- the engine's projected-source interpolation plan
  (``gpuwm.ingest.hrrr._ProjectedCpuPlan``: WPS's overlapping-parabolic
  operator with FP64 donor selection, executed by the engine's CPU bridge
  when it exports the indexed-donor entry), aimed at a regular lat-lon
  target instead of a WRF Lambert one.  Winds are rotated to the earth
  basis on the source grid with the engine's own rotation
  (``_source_window_rotation``) before they are moved, because an
  ``iproj 0`` record is earth-relative by definition;
* SOIL -- the engine's own node-to-Noah-layer interpolation
  (``gpuwm.ingest.soil._interp_nodes`` over ``HRRR_SOIL_NODE_DEPTHS_M`` and
  ``NOAH_LAYER_MIDPOINTS_M``), so the four ``ST/SM`` layers the init reads
  are the four the engine's own WRF road would have produced;
* WRITE -- the WPS version-5 record layout, the inverse of this tree's own
  reader (:mod:`hexcore.wps_intermediate`, itself a transcription of the
  frozen MPAS reader), and every file written is read back through that
  reader before the receipt is signed.

WHAT THE RESULT IS, STATED PLAINLY.  The intermediate is HRRR RESAMPLED onto
a 0.025-degree regular lat-lon grid over the domain (2.8 km meridionally,
finer zonally at mid-latitudes -- the source is not coarsened).  Fifty
hybrid levels are carried as level-indexed slabs with the 3-D ``PRESSURE``
field beside them, which is the MPAS convention for model-level first
guesses (ERA5 ML uses it) and the branch ``rw_mpas_init`` takes when a 3-D
pressure is present.  Hydrometeors are not carried: the init stream has no
slot for them, and the README's hour-zero-has-no-ice sentence applies.

THE ARBITRARY ACCEPTANCE TEST.  The source is a ROW (:data:`SOURCE_ROWS`):
the engine decoder key, the loader, the level count, the field map, the
soil nodes, the file names ``gpuwm fetch`` writes.  Nothing below this
docstring branches on the string ``hrrr``; a second projected source with an
engine decoder and loader is a second row.  The module carries the name of
the one row it ships because the plan of record names the file that way.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time
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


def source_window_for(target: LatLonTarget, row: SourceRow, *, extra_cells: int = 2) -> SourceWindow:
    """The decoder crop every target point's stencil needs, on the source grid."""

    import importlib

    loader = importlib.import_module(row.loader_module)
    grid = loader.hrrr_source_grid()
    lat, lon = target.mesh()
    x, y = grid.latlon_to_ij(lat, lon)
    zero_x = np.asarray(x, dtype=np.float64) - 1.0
    zero_y = np.asarray(y, dtype=np.float64) - 1.0
    if not (np.isfinite(zero_x).all() and np.isfinite(zero_y).all()):
        raise IntermediateRefusal(
            f"the target box maps to non-finite {row.name} source coordinates; "
            f"it lies outside the projection's domain"
        )
    halo = int(row.parabolic_halo_cells)
    window = SourceWindow(
        i_start=int(np.floor(zero_x.min())) - 1 - extra_cells,
        i_end=int(np.floor(zero_x.max())) + halo + extra_cells,
        j_start=int(np.floor(zero_y.min())) - 1 - extra_cells,
        j_end=int(np.floor(zero_y.max())) + halo + extra_cells,
    )
    limit_x = int(grid.e_we) - 1
    limit_y = int(grid.e_sn) - 1
    if window.i_start < 0 or window.j_start < 0 or window.i_end > limit_x - 1 or window.j_end > limit_y - 1:
        south, west, north, east = loader_envelope(loader)
        raise IntermediateRefusal(
            f"the target box ({target.south:.3f}..{target.north:.3f} N, "
            f"{target.west:.3f}..{target.east:.3f} E) plus its interpolation "
            f"halo leaves {row.name} coverage: it needs source cells "
            f"i={window.i_start}..{window.i_end}, j={window.j_start}..{window.j_end} "
            f"and the native mass grid is i=0..{limit_x - 1}, j=0..{limit_y - 1} "
            f"(envelope {south:.2f}..{north:.2f} N, {west:.2f}..{east:.2f} E).  "
            f"Move the point inland, shrink the core, or use a source that "
            f"covers it"
        )
    return window


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
    except ImportError as error:
        raise IntermediateRefusal(
            f"the engine's ingest is not importable ({error}); this door drives "
            f"gpuwm 2.7.4's own decoder, loader and interpolation operator and "
            f"has no road without them"
        ) from error
    needed = {
        "load": getattr(loader, "load_hrrr_pipeline_ready_window", None),
        "plan": getattr(loader, "_ProjectedCpuPlan", None),
        "rotation": getattr(loader, "_source_window_rotation", None),
        "ranges": getattr(loader, "_require_source_physical_ranges", None),
        "interp_nodes": getattr(soil, "_interp_nodes", None),
        "soil_nodes_m": getattr(soil, "HRRR_SOIL_NODE_DEPTHS_M", None),
        "noah_midpoints_m": getattr(soil, "NOAH_LAYER_MIDPOINTS_M", None),
        "resolve_backend": getattr(backend_module, "resolve_preprocess_backend", None),
    }
    missing = sorted(name for name, value in needed.items() if value is None)
    if missing:
        raise IntermediateRefusal(
            f"the installed engine's ingest does not expose {missing}; this "
            f"road is pinned to the operator gpuwm 2.7.4 ships and refuses to "
            f"substitute one of its own"
        )
    return needed


def regrid_hour(
    row: SourceRow,
    pieces: Mapping[str, Any],
    snapshot: Any,
    target: LatLonTarget,
    *,
    backend: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every WPS record for one valid time, from the engine's own operator."""

    lat, lon = target.mesh()
    plan = pieces["plan"](snapshot, lat, lon, backend)
    source = snapshot.fields
    pieces["ranges"](source)
    sina, cosa = pieces["rotation"](snapshot)
    sina = np.asarray(sina, dtype=np.float32)
    cosa = np.asarray(cosa, dtype=np.float32)

    def earth_wind(u_name: str, v_name: str) -> tuple[np.ndarray, np.ndarray]:
        u = np.asarray(source[u_name], dtype=np.float32)
        v = np.asarray(source[v_name], dtype=np.float32)
        u_earth = u * cosa - v * sina
        v_earth = v * cosa + u * sina
        return (
            np.asarray(plan.apply(u_earth, method="parabolic"), dtype=np.float32),
            np.asarray(plan.apply(v_earth, method="parabolic"), dtype=np.float32),
        )

    records: list[dict[str, Any]] = []

    def record(name: str, units: str, description: str, level: float, values: np.ndarray) -> None:
        records.append({
            "field": name, "units": units, "description": description,
            "level": float(level), "values": np.asarray(values, dtype=np.float32),
        })

    # 3-D: level-indexed slabs with the pressure beside them.
    three_d = {
        window_name: np.asarray(plan.apply(source[window_name], method="parabolic"), dtype=np.float32)
        for window_name in row.atmosphere_3d
    }
    u3, v3 = earth_wind(*row.wind_3d)
    levels = int(three_d[next(iter(row.atmosphere_3d))].shape[0])
    if levels != row.levels:
        raise IntermediateRefusal(
            f"the {row.name} window carries {levels} levels and the row declares "
            f"{row.levels}; the level table this door writes would be wrong"
        )
    for k in range(levels):
        tag = float(k + 1)
        for window_name, (wps_name, units, description) in row.atmosphere_3d.items():
            values = three_d[window_name][k]
            if wps_name == "SPECHUMD":
                values = np.maximum(values, np.float32(0.0))
            record(wps_name, units, description, tag, values)
        record("UU", "m s-1", "U", tag, u3[k])
        record("VV", "m s-1", "V", tag, v3[k])

    # Surface.
    for window_name, (wps_name, units, description, method) in row.surface.items():
        values = np.asarray(plan.apply(source[window_name], method=method), dtype=np.float32)
        if wps_name in ("SPECHUMD", "SNOW"):
            values = np.maximum(values, np.float32(0.0))
        if wps_name == "LANDSEA":
            values = np.where(values >= 0.5, np.float32(1.0), np.float32(0.0)).astype(np.float32)
        if wps_name == "SEAICE":
            values = np.clip(values, 0.0, 1.0).astype(np.float32)
        record(wps_name, units, description, SURFACE_LEVEL, values)
    u10, v10 = earth_wind(*row.wind_10m)
    record("UU", "m s-1", "U", SURFACE_LEVEL, u10)
    record("VV", "m s-1", "V", SURFACE_LEVEL, v10)

    # Soil: the engine's own node-to-layer interpolation, moved by nearest
    # neighbour so land values stay land values; the init masks by LANDSEA.
    soil_t = np.asarray(source[row.soil_temperature], dtype=np.float64)
    soil_m = np.asarray(source[row.soil_moisture], dtype=np.float64)
    layers_t = pieces["interp_nodes"](soil_t, pieces["soil_nodes_m"], pieces["noah_midpoints_m"])
    layers_m = pieces["interp_nodes"](soil_m, pieces["soil_nodes_m"], pieces["noah_midpoints_m"])
    for index, (t_name, m_name) in enumerate(row.soil_layer_names):
        t_values = plan.apply(np.asarray(layers_t[index], dtype=np.float32), method="nearest")
        m_values = plan.apply(np.asarray(layers_m[index], dtype=np.float32), method="nearest")
        record(t_name, "K", f"T {t_name[2:5]}-{t_name[5:8]} cm below ground layer (Upper)",
               SURFACE_LEVEL, np.asarray(t_values, dtype=np.float32))
        record(m_name, "m3 m-3", f"Soil moisture of {m_name[2:5]}-{m_name[5:8]} cm ground layer",
               SURFACE_LEVEL, np.clip(np.asarray(m_values, dtype=np.float32), 0.0, 1.0))
    return records, {
        "operator": getattr(plan, "operator", "unknown"),
        "levels": levels,
        "records": len(records),
    }


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


def build_intermediates(request: IntermediateRequest, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    row = request.source
    out_dir = Path(request.out_dir).expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    pieces = _engine_pieces(row)
    decoder = resolve_decoder(row, request.decoder)
    window = source_window_for(request.target, row)
    decoded = decode_window(
        row, decoder=decoder, grib_dir=request.grib_dir, cycle=request.cycle,
        hours=request.hours, window=window, out_dir=out_dir, workers=request.workers, log=log,
    )
    backend = pieces["resolve_backend"]("cpu")
    files: list[dict[str, Any]] = []
    operator = None
    for hour in request.hours:
        started = time.perf_counter()
        snapshot = pieces["load"](Path(decoded["native"]), int(hour))
        records, measured = regrid_hour(row, pieces, snapshot, request.target, backend=backend)
        operator = measured["operator"]
        valid = request.cycle + timedelta(hours=int(hour))
        path = out_dir / f"{request.prefix}:{valid:%Y-%m-%d_%H}"
        written = write_intermediate(
            path, records, valid_time=valid, forecast_hour=float(hour),
            map_source=row.map_source, target=request.target,
        )
        written["regrid_seconds"] = round(time.perf_counter() - started, 2)
        written["levels"] = measured["levels"]
        files.append(written)
        log(f"WROTE {path.name}: {written['field_records']} records, "
            f"{written['bytes'] / 1e6:.1f} MB, {written['regrid_seconds']} s")
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
            "engine": "gpuwm.ingest.hrrr._ProjectedCpuPlan (WPS overlapping-parabolic, "
                      "FP64 donors); winds rotated to the earth basis on the source grid",
            "soil": "gpuwm.ingest.soil._interp_nodes over HRRR depth nodes to the Noah "
                    "layer midpoints, moved by nearest neighbour",
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
        "why_not_carried": "the init stream has no slot for hydrometeors (hour zero has "
                           "no ice; README, limited-area lane); the engine's window "
                           "carries no mean-sea-level pressure and the init does not "
                           "require one",
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
    parser.add_argument("--lbc-exe", type=Path, default=None, metavar="FILE")
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
