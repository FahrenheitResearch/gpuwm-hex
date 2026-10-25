"""The MPAS render door: history files in, product PNGs out, Rust path only.

``mpas-port render`` is the user-reachable command over two Rust binaries:

* ``rw_mpas_convert`` (gpuwm ``tools/rustwx/crates/rw-mpas``) resamples one or
  more MPAS history frames onto a named render window and writes
  wrfout-shaped netCDF frames, with a JSON receipt carrying the mesh,
  weights and per-frame output digests.
* ``rw_wrfbatch`` (gpuwm ``tools/rustwx/crates/rw-wrfbatch``) imports each
  frame and draws the product PNGs.  Product availability is proven by the
  renderer's own ``--list-products`` catalog against the stored fields,
  never guessed from filenames.

This module is deliberately subprocess-and-stdlib only: every byte of field
data is decoded, resampled and drawn in Rust.  Python here is orchestration --
resolving executables, sequencing the two engines, filing the finished PNGs
into the ruled ``<domain>/<product>/<valid-day>/`` layout and writing the
manifest.  There is no fallback plotter behind it.

Native resolution and names (2026-10-10).  Every picture is drawn at the
spacing of the MESH, never a fixed grid: ``--window global`` is the whole
sphere at the mesh's own median spacing and ``--window region:...`` takes
its dx from the mesh cells inside it, both derived by the converter.  Each
PNG is filed as ``<view>/<product>/<valid-day>/<mesh>_<view>_<product>_<lead>.png``
-- ``hex15km_global_500mb_height_winds_f006.png`` -- where ``<mesh>`` is the
grid label the converter read off the mesh.  THE BREAKAGE THIS PREVENTS,
MEASURED: the live global hex gallery drew a 15 km mesh on 20 km and 28 km
grids, headlined every map ``(d01 20 km)`` and named every file
``rustwx_wrf_20261010_0z_f000_d01-20km_<product>.png`` -- the renderer's
generic WRF template, which says neither which model nor which mesh.

Scratch discipline: conversion frames, per-frame render stores and the raw
engine output live in a scratch directory that is a SIBLING of ``--out``,
never inside it.  A ``.rwstore-*`` scratch tree inside a delivered png tree
is a shipped defect this door refuses to repeat; an explicit ``--scratch``
inside ``--out`` is refused by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as _dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from . import engines
from .errors import MpasPortError


class RenderDoorError(MpasPortError):
    """A named render-door refusal: what breaks, and the remedy."""


#: Environment variables naming the two engine binaries.  The renderer also
#: honors gpuwm's own spelling so a box with a gpuwm install needs no second
#: variable pointed at the same file.
# The preferred spellings carry the distribution's own name.  The
# MPAS_PORT_* pair came from the import package's ORIGINAL name -- the
# package was `mpas_port` through 0.1.1 and is `hexcore` from 0.2.0 -- and it
# predates the decision that the brand and the spellings a user types never
# carry the MPAS token.  They are kept AHEAD of nothing and BEHIND the new
# names, never removed: an install line that already works must not be
# invalidated by a rename, and a variable that silently stops being read is
# the worst possible way to learn about one.  That rule is why the 0.2.0
# package rename did not touch this pair either: these two strings are the
# one place the retired name is still load-bearing for a user.
#
# The spellings are stated here and CONSUMED by :mod:`hexcore.engines`,
# which owns the one resolution ladder every door in this distribution
# uses.  Both halves matter: the names live beside the door that
# documents them, and the search itself is not reimplemented per door --
# which is how this door came to read three variables and PATH while the
# init door read one variable and PATH, and neither read the directory
# `gpuwm fetch-bridges` actually stages into.
CONVERT_ENV = "GPUWM_HEX_RW_MPAS_CONVERT"
CONVERT_ENV_LEGACY = "MPAS_PORT_RW_MPAS_CONVERT"
RENDERER_ENV = "GPUWM_HEX_RW_WRFBATCH"
RENDERER_ENV_LEGACY = "MPAS_PORT_RW_WRFBATCH"
RENDERER_ENV_GPUWM = "GPUWM_RW_WRFBATCH"

# The ladder reads them in this order, preferred first, legacy behind,
# gpuwm's own spelling last of the three.  A test binds these tuples to
# the constants above so a rename cannot drop a variable from the search.
assert engines.CONVERT.env_names == (CONVERT_ENV, CONVERT_ENV_LEGACY)
assert engines.RENDERER.env_names == (RENDERER_ENV, RENDERER_ENV_LEGACY)
assert engines.RENDERER.gpuwm_env == RENDERER_ENV_GPUWM

#: The renderer reads its basemap root from this variable first, ahead of
#: everything it can find beside itself.  The door always sets it.
BASEMAP_ENV = "RUSTWX_BASEMAP_DIR"
BASEMAP_ASSETS_ENV = "RUSTWX_ASSETS_DIR"

#: What makes a directory a basemap root: one of the Natural Earth sets the
#: renderer draws coastlines, borders and state lines from.
BASEMAP_MARKERS = (
    ("natural_earth_10m", "ne_10m_coastline.shp"),
    ("natural_earth_110m", "ne_110m_coastline.shp"),
)

#: The manifest schema written beside the delivered tree.
MANIFEST_SCHEMA = "mpas-port.render-door-manifest/v1"

#: Layout tokens for a frame whose facts could not be read (same spellings
#: as gpuwm.render_layout, so a reader of either tree learns one vocabulary).
NATIVE_GRID = "native_grid"
UNDATED = "undated"
UNCLASSIFIED = "unclassified"

#: ``--products`` tokens the renderer expands itself; they name groups, not
#: single catalog rows, so per-slug coverage refusal does not apply to them.
_GROUP_KEYWORDS = frozenset({"all", "direct", "derived", "heavy", "windowed",
                             "variables"})

#: The door's window names and the converter window each one asks for.
#: ``global`` is the whole sphere AT THE MESH'S OWN SPACING
#: (``global-mesh``): the frozen 0.25 degree overview drew a 15 km mesh on a
#: 27.8 km grid, so it is no longer what a bare ``--window global`` means.
#: It stays reachable, unchanged, as ``global-0.25`` -- its weights digests
#: are recorded in evidence and its geometry does not move.
WINDOW_TO_CONVERTER = {
    "mesh": "mesh",
    "focus": "focus",
    "global": "global-mesh",
    "global-0.25": "global",
    "composite": "composite",
}

#: Window specifications passed to the converter as written: ``region:``
#: (ground extent, dx from the mesh) and ``lambert:`` (an explicit grid).
WINDOW_SPEC_PREFIXES = ("region:", "region-latlon:", "lambert:")

#: ``--remap`` vocabulary (see :func:`resolve_remap`).
REMAP_AUTO = "auto"
REMAP_CHOICES = (REMAP_AUTO, "product", "nearest")


def resolve_remap(requested: str, composite: bool) -> str:
    """The converter remap for this render.

    ``auto`` is ``product`` -- linear state, conservative precipitation --
    for a single run, the door default because a nearest-cell map steps at
    every cell edge and carries no accumulation budget; and ``nearest`` for
    a composite, which switches source point by point and has no single mesh
    to interpolate across.  An explicit ``product`` with ``--compose`` is
    refused by that name rather than quietly downgraded."""

    requested = str(requested or REMAP_AUTO)
    if requested not in REMAP_CHOICES:
        raise RenderDoorError(
            f"--remap {requested!r} is not one of {', '.join(REMAP_CHOICES)}")
    if requested == REMAP_AUTO:
        return "nearest" if composite else "product"
    if requested == "product" and composite:
        raise RenderDoorError(
            "--remap product interpolates across one mesh, and a --compose "
            "frame takes each point from whichever run covers it, so there "
            "is no single mesh to interpolate across; use --remap nearest "
            "(or auto) for a composite")
    return requested


#: The model this door renders.  Every history it reads was written by
#: gpuwm-hex, so its plots say ``hex`` where the wrfout-shaped frames would
#: otherwise import as ``WRF``.
DEFAULT_MODEL_LABEL = "hex"

#: A view name: the layout's domain folder and the filename's second token.
_VIEW = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")

#: The renderer's output filename, as rustwx-products formats it::
#:
#:     rustwx_<model>_<YYYYMMDD>_<H>z_f<NNN>_<domain-slug>_<product-slug>.png
_LEAD_SEGMENT = re.compile(r"^f\d{3}$")

#: ``YYYY-MM-DD`` at the head of an ISO instant or WRF Times stamp.
_DAY = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[T_ ]|$)")


@dataclass
class FrameResult:
    """What one history frame became."""

    history: str
    wrfout: str
    valid_time: str
    valid_day: str
    output_sha256: str
    absent_wrf_fields: list[str]
    rendered: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    placed: list[str] = field(default_factory=list)
    # Composed frames only: how far the runs disagree where the composite
    # changes source, per surface field.  Empty for a single-run render,
    # which has no seam to measure.
    seam: list[dict] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(spec: engines.EngineSpec, explicit: str | None) -> Path:
    """Drive the shared ladder, re-raising as this door's own refusal type.

    The refusal TEXT is the ladder's, unedited: it already names every
    rung it searched and both commands that supply the binary, and a
    second paraphrase here would be a second thing to keep true.  Only
    the exception CLASS changes, so a caller catching
    :class:`RenderDoorError` still catches it.
    """

    try:
        return engines.resolve(spec, explicit)
    except engines.EngineRefusal as error:
        raise RenderDoorError(str(error)) from None


def resolve_convert_exe(explicit: str | None = None) -> Path:
    """The ``rw_mpas_convert`` binary, or a named refusal.

    Reads ``--convert-exe``, ``$CONVERT_ENV``, ``$CONVERT_ENV_LEGACY``,
    gpuwm's ``$GPUWM_RW_MPAS_CONVERT`` and its bridge directories, then
    ``PATH``.
    """

    return _resolve(engines.CONVERT, explicit)


def resolve_renderer_exe(explicit: str | None = None) -> Path:
    """The ``rw_wrfbatch`` binary, or a named refusal.

    Reads ``--renderer-exe``, ``$RENDERER_ENV``, ``$RENDERER_ENV_LEGACY``,
    ``$RENDERER_ENV_GPUWM`` and gpuwm's bridge directories, then ``PATH``.
    """

    return _resolve(engines.RENDERER, explicit)


def is_basemap_root(path: Path) -> bool:
    """Whether ``path`` holds a Natural Earth set the renderer can draw."""

    return any((Path(path) / folder / shapefile).is_file()
               for folder, shapefile in BASEMAP_MARKERS)


def basemap_candidates(renderer_exe: Path,
                       explicit: str | None = None) -> list[Path]:
    """Where the door looks for the renderer's map geometry, best first.

    1. ``--basemap-dir``;
    2. ``$RUSTWX_BASEMAP_DIR`` and ``$RUSTWX_ASSETS_DIR/basemap`` -- the
       renderer's own variables, which stay the user's to set;
    3. ``assets/basemap`` above the renderer binary, which is where a
       renderer built in a gpuwm checkout already finds it;
    4. gpuwm's own ladder (``gpuwm.rustwx.wrapper_basemap_candidates``:
       the checkout's assets, the ``gpuwm-data`` companion every install
       pulls, the ones ``gpuwm fetch-bridges`` staged) when gpuwm imports;
    5. ``~/.gpuwm/bridges/assets/basemap``, which needs no gpuwm import.
    """

    found: list[Path] = []

    def push(path) -> None:
        path = Path(path)
        if path not in found:
            found.append(path)

    if explicit:
        push(explicit)
    if os.environ.get(BASEMAP_ENV):
        push(os.environ[BASEMAP_ENV])
    if os.environ.get(BASEMAP_ASSETS_ENV):
        push(Path(os.environ[BASEMAP_ASSETS_ENV]) / "basemap")
    try:
        parent = Path(renderer_exe).resolve().parent
    except OSError:
        parent = Path(renderer_exe).parent
    for ancestor in (parent, *parent.parents)[:8]:
        push(ancestor / "assets" / "basemap")
    try:
        from gpuwm import rustwx  # noqa: PLC0415 - lazy on purpose
        for path in rustwx.wrapper_basemap_candidates():
            push(path)
    except Exception:  # noqa: BLE001 - gpuwm absent, or older than the ladder
        pass
    push(engines.USER_BRIDGE_DIR / "assets" / "basemap")
    return found


def resolve_basemap_dir(renderer_exe: Path,
                        explicit: str | None = None) -> Path:
    """The basemap root every renderer call of this door is given.

    THE BREAKAGE THIS PREVENTS, MEASURED (2026-10-11): the renderer finds
    its coastlines, borders and state lines beside its own build tree, and
    the door launched it bare.  A renderer binary copied to another folder
    therefore found none, drew every field with no geography under it,
    exited 0 and reported every product RENDERED: 59 delivered maps had no
    basemap.  The regional path never had this because it launches the
    same binary through ``gpuwm.rustwx``, which pins the root.  So the door
    resolves the root itself, hands it to the renderer by name, and refuses
    before converting anything when there is none.
    """

    candidates = basemap_candidates(renderer_exe, explicit)
    if explicit and not is_basemap_root(Path(explicit)):
        raise RenderDoorError(
            f"--basemap-dir {explicit} holds no Natural Earth set "
            f"(natural_earth_10m/ne_10m_coastline.shp or "
            f"natural_earth_110m/ne_110m_coastline.shp), so every map would "
            f"be drawn without coastlines, borders or state lines. Point it "
            f"at a gpuwm checkout's tools/rustwx/assets/basemap."
        )
    for candidate in candidates:
        if is_basemap_root(candidate):
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise RenderDoorError(
        f"no basemap geometry was found for the renderer {renderer_exe}, so "
        f"every map would be drawn without coastlines, borders or state "
        f"lines; nothing is rendered. Searched: {searched}. Pass "
        f"--basemap-dir (or set ${BASEMAP_ENV}) to a directory holding "
        f"natural_earth_10m/ or natural_earth_110m/ -- a gpuwm checkout's "
        f"tools/rustwx/assets/basemap -- or install gpuwm-data, which "
        f"carries them."
    )


def renderer_environment(basemap_dir: Path) -> dict[str, str]:
    """The renderer's environment: this process's, with the basemap root
    named.  The explicit root outranks ``$RUSTWX_ASSETS_DIR`` inside the
    renderer, so one resolved answer is the one that is drawn."""

    env = dict(os.environ)
    env[BASEMAP_ENV] = str(basemap_dir)
    return env


def valid_day(stamp: str | None) -> str:
    """``YYYY-MM-DD`` from an ISO instant, or :data:`UNDATED` -- a wrong
    date is worse than an honest ``undated`` because a reader believes it."""

    if not stamp:
        return UNDATED
    match = _DAY.match(str(stamp).strip())
    if match is None:
        return UNDATED
    try:
        _dt.date.fromisoformat(match.group(1))
    except ValueError:
        return UNDATED
    return match.group(1)


def split_engine_name(filename: str, slug: str) -> tuple[str, str]:
    """``(domain_token, product_dir)`` for one engine output filename.

    The RENDERED event names the product ``slug``, but the FILENAME spells
    it the engine's filesystem-safe way: a generic ``var:wrf_psfc`` event
    becomes ``..._var_wrf_psfc_<hash>.png`` on disk (colon to underscore,
    plus a content-hash suffix).  The product directory uses that safe
    spelling too -- a ``var:`` directory is unrepresentable on Windows and
    would strand the delivered tree on the render node.  The domain token
    is what sits between the ``fNNN`` lead segment and the slug's first
    appearance.  A name that does not parse files under
    (:data:`NATIVE_GRID`, safe slug) -- somewhere nameable, never dropped.
    """

    safe_slug = slug.replace(":", "_")
    stem = Path(filename).stem
    parts = stem.split("_")
    for index, part in enumerate(parts):
        if _LEAD_SEGMENT.match(part):
            tail = "_".join(parts[index + 1 :])
            marker = tail.find(f"_{safe_slug}")
            if marker > 0:
                return tail[:marker], safe_slug
            return NATIVE_GRID, safe_slug
    return NATIVE_GRID, safe_slug


#: The engine's own filename head: ``rustwx_<model>_<YYYYMMDD>_<H>z_f<NNN>_``.
_ENGINE_HEAD = re.compile(r"^rustwx_[^_]+_(?P<day>\d{8})_(?P<hour>\d{1,2})z_f(?P<fhr>\d{3})_")
#: The engine's lead suffix: ``..._lead_<HHH>h<MM>m<SS>s``.
_LEAD_SUFFIX = re.compile(r"_lead_(?P<h>\d+)h(?P<m>\d{2})m(?P<s>\d{2})s$")


def default_label(mesh: Path | None) -> str:
    """``hex_<mesh>`` from the mesh file name: ``x1.2621442.static.nc`` ->
    ``hex_x1.2621442``.  The role tokens (``ordered``, ``static``, ``grid``,
    ``nc``) are dropped; a name that is nothing but role tokens falls back to
    its directory's name, so a label is never empty."""

    if mesh is None:
        return "hex"
    roles = {"ordered", "static", "grid", "nc", "init"}
    tokens = [token for token in Path(mesh).name.split(".") if token and token not in roles]
    stem = ".".join(tokens) or Path(mesh).parent.name
    return f"hex_{stem}" if stem else "hex"


def legacy_product_file_name(label: str, engine_name: str, product: str) -> str:
    """``<label>_<YYYYMMDD>_<HH>z_f<lead>_<product>.png`` for one engine PNG.

    THE BREAKAGE THIS PREVENTS (the owner's naming ruling, restated 2026-10-10):
    the engine names every frame it imports from a wrfout-shaped file
    ``rustwx_wrf_<date>_<H>z_f<NNN>_d01-<dx>km_...``, so a hex global
    forecast's maps were filed under another model's name and a resolution
    token.  The delivered name says the model and mesh (``label``), the
    cycle, the lead and the product.  A sub-hour lead keeps its minutes
    (``f000h30m``) so two frames in one hour never share a name.  An engine
    name that does not parse is kept as it is rather than guessed at.
    """

    stem = Path(engine_name).stem
    head = _ENGINE_HEAD.match(stem)
    if head is None:
        return engine_name
    lead = f"f{head['fhr']}"
    tail = _LEAD_SUFFIX.search(stem)
    if tail is not None:
        hours, minutes, seconds = int(tail["h"]), int(tail["m"]), int(tail["s"])
        lead = f"f{hours:03d}" + (f"h{minutes:02d}m" if minutes or seconds else "")
        if seconds:
            lead += f"{seconds:02d}s"
    return f"{label}_{head['day']}_{int(head['hour']):02d}z_{lead}_{product}.png"
def window_argument(text: str) -> str:
    """argparse ``type`` for ``--window``: a door window name or a spec."""

    import argparse

    value = str(text).strip()
    if value in WINDOW_TO_CONVERTER or value.startswith(WINDOW_SPEC_PREFIXES):
        return value
    raise argparse.ArgumentTypeError(
        f"unknown window {value!r}: choose {', '.join(WINDOW_TO_CONVERTER)}, "
        f"or region:CLAT,CLON,WIDTH_KM,HEIGHT_KM[,TRUELAT1,TRUELAT2] (dx from "
        f"the mesh), region-latlon:CLAT,CLON,WIDTH_KM,HEIGHT_KM (the same box on a "
        f"regular lat-lon grid: what an ArWen domain reads as its parent), or "
        f"lambert:CLAT,CLON,DX_M,NX,NY,TRUELAT1,TRUELAT2,STAND_LON")


def converter_window(window: str) -> str:
    """The converter window a door window asks for (specs pass through)."""

    if window.startswith(WINDOW_SPEC_PREFIXES):
        return window
    try:
        return WINDOW_TO_CONVERTER[window]
    except KeyError:
        raise RenderDoorError(
            f"--window {window!r} is not a render window this door knows; "
            f"choose {', '.join(WINDOW_TO_CONVERTER)} or a region:/lambert: "
            f"spec.") from None


def default_view(window: str) -> str:
    """The view name a window files under when ``--view`` is not given."""

    if window.startswith(WINDOW_SPEC_PREFIXES):
        return "region"
    return "global" if window.startswith("global") else window


def checked_view(view: str) -> str:
    if not _VIEW.match(view or ""):
        raise RenderDoorError(
            f"--view {view!r} must be lower-case letters, digits and '-', "
            f"starting with a letter or digit (at most 40): it becomes a "
            f"folder and part of every filename, so a space or a slash would "
            f"split one picture's name into a path.")
    return view


def grid_token(grid_label: str | None, engine_domain: str) -> str:
    """``hex 15 km`` -> ``hex15km``; the engine's own domain token when the
    converter named no grid (an older engine), so a name is never empty."""

    if grid_label:
        token = "".join(ch for ch in str(grid_label).lower()
                        if ch.isascii() and (ch.isalnum() or ch in "-."))
        if token:
            return token
    return engine_domain or NATIVE_GRID


def lead_token(lead_seconds, filename: str) -> str:
    """``f006`` for a whole-hour lead, ``f000h30m`` for a sub-hourly one.

    Read from the converter's own ``lead_seconds`` when it reported one;
    otherwise the engine filename's ``fNNN`` segment (whole hours only)."""

    try:
        seconds = int(lead_seconds)
    except (TypeError, ValueError):
        seconds = None
    if seconds is not None and seconds >= 0:
        hours, rest = divmod(seconds, 3600)
        minutes, seconds = divmod(rest, 60)
        lead = f"f{hours:03d}"
        if minutes or seconds:
            lead += f"h{minutes:02d}m"
        if seconds:
            lead += f"{seconds:02d}s"
        return lead
    for part in Path(filename).stem.split("_"):
        if _LEAD_SEGMENT.match(part):
            return part
    return "f-unknown"


def product_file_name(token: str, view: str, product: str, lead: str | None = None) -> str:
    """``hex15km_global_500mb_height_winds_f006.png``: model and mesh,
    view, product, lead."""

    if lead is None:
        return legacy_product_file_name(token, view, product)
    return f"{token}_{view}_{product}_{lead}.png"


def carry_georef(raw_out: Path, moved: dict[str, str], folder: Path) -> int:
    """Re-key the renderer's ``render-georef.json`` to the filed names.

    The renderer writes one georeference per picture, keyed by the name it
    gave the picture inside its own output folder.  Those names are gone
    once the door files the pictures, so the records are carried to
    ``<folder>/render-georef.json`` (the view's folder) keyed by paths
    relative to that folder, the renderer's own convention; without this
    every delivered picture lost the transform that puts a latitude and
    longitude on its pixels.  Returns the records carried."""

    source = Path(raw_out) / "render-georef.json"
    if not moved or not source.is_file():
        return 0
    try:
        held = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    target = Path(folder) / "render-georef.json"
    try:
        merged = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        merged = {key: value for key, value in held.items()
                  if key not in ("panels", "without_georeference")}
        merged["panels"] = {}
        merged["without_georeference"] = []
    merged.setdefault("panels", {})
    merged.setdefault("without_georeference", [])
    carried = 0
    for key, record in (held.get("panels") or {}).items():
        new = moved.get(Path(key).as_posix())
        if new is not None:
            merged["panels"][new] = record
            carried += 1
    for absence in held.get("without_georeference") or []:
        new = moved.get(Path(str(absence.get("path", ""))).as_posix())
        if new is not None:
            merged["without_georeference"].append(dict(absence, path=new))
            carried += 1
    for key in ("generated_utc", "schema"):
        if key in held:
            merged[key] = held[key]
    target.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return carried


def place_path(out_root: Path, domain: str, product: str, day: str,
               filename: str) -> Path:
    """``<out>/<domain>/<product>/<valid-day>/<filename>`` -- the ruled
    layout, and the only one this door writes.  Nothing may invert it back
    to flat."""

    return (Path(out_root) / (domain or NATIVE_GRID)
            / (product or UNCLASSIFIED) / (day or UNDATED) / filename)


def default_scratch(out_root: Path) -> Path:
    """A scratch sibling of ``--out``: same parent, never inside the tree."""

    out_root = Path(out_root)
    return out_root.parent / (out_root.name + ".render-scratch")


def refuse_scratch_inside_out(scratch: Path, out_root: Path) -> None:
    scratch_abs = Path(os.path.abspath(scratch))
    out_abs = Path(os.path.abspath(out_root))
    if scratch_abs == out_abs or out_abs in scratch_abs.parents:
        raise RenderDoorError(
            f"--scratch {scratch} is inside the delivered png tree "
            f"{out_root}. Renderer scratch inside a delivered tree is a "
            f"shipped defect (readers publish half-finished temporaries as "
            f"products); give --scratch a directory outside --out, or omit "
            f"it for the sibling default."
        )


def _require_file(path: Path, flag: str, remedy: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise RenderDoorError(
            f"{flag} names a missing file: {path}. {remedy}"
        )
    return path


def _run(command: list[str], *, log_prefix: str,
         env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    printable = " ".join(str(part) for part in command)
    print(f"{log_prefix} $ {printable}", flush=True)
    return subprocess.run(
        [str(part) for part in command],
        capture_output=True, text=True, errors="replace", env=env,
    )


def _run_renderer(command: list[str], *, log_prefix: str,
                  env: dict[str, str] | None) -> subprocess.CompletedProcess:
    """A renderer call.  ``env`` carries the basemap root; ``None`` (a
    caller that resolved none, as the unit tests of one stage do) leaves
    the renderer to its own search."""

    if env is None:
        return _run(command, log_prefix=log_prefix)
    return _run(command, log_prefix=log_prefix, env=env)


def convert_frames(convert_exe: Path, *, histories: list[Path] | None,
                   mesh: Path | None, scratch: Path, window: str,
                   field_set: str, simulation_start: str | None,
                   nc_format: str, compose: Path | None = None,
                   compose_base_only: bool = False,
                   init: Path | None = None,
                   diagnostics: Path | None = None,
                   model_label: str | None = None,
                   remap: str = "nearest",
                   jobs: int = 1) -> tuple[dict, list[str]]:
    """Run ``rw_mpas_convert`` once over the whole series.

    Returns ``(report, invocation)`` where ``report`` is the converter's own
    JSON receipt (mesh/weights digests, per-frame output digests and absent
    WRF fields).  A converter refusal is relayed verbatim -- it already
    names the file and the field.

    With ``compose``, the source list in that file owns the meshes and the
    histories and ``histories``/``mesh`` are not passed: two source lists
    that can disagree about which run is the base is a worse failure than
    one flag being unavailable, so the converter refuses the pair and this
    door does not construct it."""

    if diagnostics is not None and (compose is not None or init is None or len(histories or []) != 1):
        raise RenderDoorError("--diagnostics requires one history, --init, and no --compose")
    frames_dir = Path(scratch) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(scratch) / "convert-report.json"
    command: list[str] = [str(convert_exe)]
    if compose is not None:
        command.extend(("--compose", str(compose)))
        if compose_base_only:
            command.append("--compose-base-only")
    else:
        command.extend(("--history", *[str(item) for item in histories or []]))
        command.extend(("--mesh", str(mesh)))
        if init is not None:
            # The converter sources PHB (g * zgrid) from the run's init file
            # and nothing else: a history frame carries no zgrid, so without
            # this every pressure-level product (500 hPa height and winds,
            # 850 hPa) is refused as missing-fields on every run.
            command.extend(("--init", str(init)))
        if diagnostics is not None:
            command.extend(("--diagnostics", str(diagnostics),
                            "--diagnostics-sha256", sha256_file(diagnostics),
                            "--init-sha256", sha256_file(init)))
    command.extend((
        "--out-dir", str(frames_dir),
        "--window", window,
        "--field-set", field_set,
        "--format", nc_format,
        "--clobber",
        "--json", str(report_path),
    ))
    if simulation_start:
        command.extend(("--simulation-start", simulation_start))
    if model_label:
        # The plot's metadata row names this model instead of the generic
        # WRF identity every wrfout-shaped frame imports under.
        command.extend(("--model-label", model_label[:32]))
    if remap != "nearest":
        # Linear state, conservative accumulations, nearest categories
        # (rw_mpas_convert --help).  Passed only when asked so an older
        # converter that predates the flag still runs a nearest render.
        command.extend(("--remap", remap))
    if jobs > 1 and compose is None:
        # Frames are independent, so the converter takes several at a time;
        # passed only when asked, so an older converter still runs.
        command.extend(("--jobs", str(jobs)))
    result = _run(command, log_prefix="CONVERT")
    for line in (result.stdout or "").splitlines():
        if line.startswith(("WINDOW\t", "WEIGHTS\t", "CONVERTED\t",
                            "COMPOSITE\t", "SEAM\t", "FINISHED\t",
                            "REMAP\t", "BUDGET\t")):
            print(f"CONVERT {line}", flush=True)
    # A composite drops any hour a source does not carry, and says so on
    # stderr.  Relayed rather than swallowed: a frame silently missing from
    # a delivered series reads as a gap in the weather.
    for line in (result.stderr or "").splitlines():
        if "dropping" in line:
            print(f"CONVERT {line.strip()}", flush=True)
    if result.returncode != 0:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip()]
        raise RenderDoorError(
            f"rw_mpas_convert exited {result.returncode}: "
            f"{tail[-1] if tail else 'no stderr'}"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RenderDoorError(
            f"rw_mpas_convert exited 0 but its JSON receipt at "
            f"{report_path} is unreadable ({error}); the receipt is the "
            f"digest evidence, so the render does not proceed without it."
        ) from error
    return report, [str(part) for part in command]


def list_catalog(renderer_exe: Path, wrfout: Path | list[Path], *, scratch: Path,
                 heavy: bool, store: Path | None = None,
                 products: str | None = None, env: dict[str, str] | None = None,
                 import_jobs: int = 1
                 ) -> tuple[dict[str, tuple[str, str, str]], str, list[str]]:
    """The renderer's own product catalog for one converted frame, or for
    a whole series imported into one store (a list of frames): only a
    store that holds several frames can close an accumulation window, so
    only the series listing can prove ``qpf_6h`` and its kin.

    Returns ``(rows, summary, invocation)``; ``rows`` maps slug ->
    (kind, status, detail).  Statuses are proven against the real stored
    import, never guessed.

    With ``store`` and ``products`` the listing imports the frame into the
    store its render will use, exactly as that render imports it, so the
    render finds the import there and does not repeat it.  MEASURED (8-card RTX PRO 6000 host,
    2026-10-10, one 15 km global frame at native spacing): the import is
    about two minutes of the render and drawing ten products 1.3 s, and the
    door imported every frame twice -- once in full into a catalog store,
    once for the render."""

    store = Path(store) if store is not None else Path(scratch) / "catalog-store"
    store.mkdir(parents=True, exist_ok=True)
    command = [
        str(renderer_exe),
        "--store-root", str(store),
        "--out-dir", str(store),
        "--list-products",
    ]
    if products is not None:
        command.extend(("--products", products))
    if heavy:
        command.append("--heavy")
    if import_jobs > 1:
        command.extend(("--import-jobs", str(import_jobs)))
    if isinstance(wrfout, (list, tuple)):
        command.extend(str(item) for item in wrfout)
        wrfout = wrfout[0]
    else:
        command.append(str(wrfout))
    result = _run_renderer(command, log_prefix="CATALOG", env=env)
    if result.returncode != 0:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip()]
        raise RenderDoorError(
            f"rw_wrfbatch --list-products exited {result.returncode} on "
            f"{wrfout}: {tail[-1] if tail else 'no stderr'}"
        )
    rows: dict[str, tuple[str, str, str]] = {}
    summary = ""
    for line in (result.stdout or "").splitlines():
        if line.startswith("PRODUCT\t"):
            # Five fields or more, never exactly five. The renderer's row
            # grew a sixth field (a machine reason code) in gpuwm 2.8.0; a
            # parser that demanded exactly five dropped every row of that
            # build and refused every render with "no catalog rows".
            parts = line.split("\t")
            if len(parts) >= 5:
                rows[parts[1]] = (parts[2], parts[3], parts[4])
        elif line.startswith("CATALOG "):
            summary = line[len("CATALOG "):]
    if not rows:
        raise RenderDoorError(
            f"rw_wrfbatch produced no catalog rows for {wrfout}; the import "
            f"failed silently, so product availability cannot be proven and "
            f"the render does not proceed."
        )
    return rows, summary, [str(part) for part in command]


def refuse_uncovered_products(products: str, rows: dict, history: Path,
                              absent: list[str]) -> None:
    """Refuse, by product name, when a requested product cannot be drawn.

    Group keywords (``all``, ``derived``, ...) expand to whatever IS
    renderable, so they never refuse; an explicitly named product that the
    catalog marks non-renderable stops the run before anything is drawn,
    so a partial gallery cannot pass as the requested one."""

    requested = [token.strip() for token in products.split(",")
                 if token.strip()]
    for token in requested:
        if token in _GROUP_KEYWORDS:
            continue
        row = rows.get(token)
        if row is None:
            known = ", ".join(sorted(slug for slug, (_k, status, _d)
                                     in rows.items()
                                     if status == "renderable"))
            raise RenderDoorError(
                f"product '{token}' is not in the renderer catalog for "
                f"{history}; nothing by that name can be drawn. Renderable "
                f"products from this history: {known}."
            )
        kind, status, detail = row
        if status != "renderable":
            hint = (f" (fields absent from the source history: "
                    f"{', '.join(absent)})" if absent else "")
            raise RenderDoorError(
                f"product '{token}' is not renderable from {history}: "
                f"{status} -- {detail}{hint}. The render stops before "
                f"drawing anything so a partial gallery cannot pass as the "
                f"requested one; drop the product from --products or rerun "
                f"the model with the fields it needs."
            )


#: The one store and raw folder a series render uses (:func:`run_render`).
SERIES_TAG = "series"


#: Graticule spacings, degrees, smallest first.
GRATICULE_STEPS = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0)
#: Most lines across a regional window's longer side.
GRATICULE_MAX_LINES = 6
GRATICULE_LINE_COLOUR = "#4a525c80"


def _wrap180(value):
    return ((value + 180.0) % 360.0) - 180.0


def _degree_text(value: float, step: float, positive: str, negative: str) -> str:
    number = f"{abs(value):.0f}" if step >= 1.0 else f"{abs(value):g}"
    if abs(value) < 1e-9 or abs(abs(value) - 180.0) < 1e-9:
        return f"{number}°"
    return f"{number}°{positive if value > 0 else negative}"


def graticule_overlays(lat, lon) -> dict | None:
    """Latitude/longitude lines and their labels for one converted frame,
    in the renderer's ``--overlays`` schema (degrees).

    THE BREAKAGE THIS PREVENTS (hex storm-following AL10, GFS 2026-10-22
    18Z, f012): a 3 km moving region over open ocean was drawn as
    reflectivity on ocean fill with no line of geography at all, so a
    reader could not tell where the storm was.  The renderer draws
    coastlines and borders, but a window with no land in it had nothing
    else; every hex map now carries a labelled graticule.  ``lat``/``lon``
    are the frame's XLAT/XLONG (2-D, south row first)."""

    import numpy as np

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.ndim != 2 or lat.shape != lon.shape or min(lat.shape) < 4:
        return None
    ny, nx = lat.shape
    centre = float(lon[ny // 2, nx // 2])
    offset = _wrap180(lon - centre)
    if not (np.isfinite(lat).all() and np.isfinite(offset).all()):
        return None
    south, north = float(lat.min()), float(lat.max())
    west, east = float(offset.min()), float(offset.max())
    whole_globe = east - west > 300.0
    if whole_globe:
        step = 30.0
    else:
        mid = math.radians(0.5 * (south + north))
        span = max(north - south, (east - west) * max(0.2, math.cos(mid)))
        step = next((s for s in GRATICULE_STEPS if span / s <= GRATICULE_MAX_LINES),
                    GRATICULE_STEPS[-1])
    edge = 1e-3
    west, east = west + edge, east - edge
    parallels = [k * step for k in range(math.ceil(south / step), math.floor(north / step) + 1)
                 if south < k * step < north and abs(k * step) < 90.0]
    meridians = [k * step for k in range(math.ceil((centre + west) / step),
                                         math.floor((centre + east) / step) + 1)
                 if west < k * step - centre < east]
    densify = 2.0 if whole_globe else step / 10.0
    lines, labels = [], []
    for value in parallels:
        count = max(2, int(math.ceil((east - west) / densify)) + 1)
        points = [(value, _wrap180(centre + west + (east - west) * i / (count - 1)))
                  for i in range(count)]
        lines.append({"points": points, "color": GRATICULE_LINE_COLOUR, "width": 1})
    for value in meridians:
        count = max(2, int(math.ceil((north - south) / densify)) + 1)
        points = [(south + (north - south) * i / (count - 1), _wrap180(value))
                  for i in range(count)]
        lines.append({"points": points, "color": GRATICULE_LINE_COLOUR, "width": 1})
    # Labels sit on the lines just inside the west and south edges (on a
    # whole globe: between the first two meridians and two parallels).
    if whole_globe:
        label_lon = _wrap180(centre - 165.0)
        label_lat = -45.0
        for value in parallels:
            labels.append({"lat": value, "lon": label_lon,
                           "text": _degree_text(value, step, "N", "S")})
        for value in meridians:
            labels.append({"lat": label_lat, "lon": _wrap180(value),
                           "text": _degree_text(_wrap180(value), step, "E", "W")})
    else:
        column = min(nx - 1, max(1, nx // 20))
        row = min(ny - 1, max(1, ny // 20))
        for value in parallels:
            j = int(np.argmin(np.abs(lat[:, column] - value)))
            if abs(lat[j, column] - value) > 0.5 * step or j in (0, ny - 1):
                continue
            labels.append({"lat": value, "lon": float(lon[j, column]),
                           "text": _degree_text(value, step, "N", "S")})
        for value in meridians:
            i = int(np.argmin(np.abs(offset[row, :] - (value - centre))))
            if abs(offset[row, i] - (value - centre)) > 0.5 * step or i in (0, nx - 1):
                continue
            labels.append({"lat": float(lat[row, i]), "lon": _wrap180(value),
                           "text": _degree_text(_wrap180(value), step, "E", "W")})
    if not lines:
        return None
    return {"lines": lines, "labels": labels, "step_deg": step}


def write_graticule_overlays(frame: Path, scratch: Path) -> tuple[Path | None, dict | None]:
    """Write :func:`graticule_overlays` for ``frame`` (a converted
    wrfout-shaped file) into ``scratch``; ``(None, None)`` when the frame
    carries no usable XLAT/XLONG."""

    import netCDF4

    from .netcdf_lock import NETCDF

    try:
        with NETCDF, netCDF4.Dataset(str(frame)) as dataset:
            if "XLAT" not in dataset.variables or "XLONG" not in dataset.variables:
                return None, None
            lat = dataset.variables["XLAT"][:]
            lon = dataset.variables["XLONG"][:]
    except OSError as error:
        # The renderer reads the same file and names its own failure.
        print(f"GRATICULE unreadable frame {frame}: {error}", flush=True)
        return None, None
    while getattr(lat, "ndim", 0) > 2:
        lat, lon = lat[0], lon[0]
    overlays = graticule_overlays(lat, lon)
    if overlays is None:
        return None, None
    step = overlays.pop("step_deg")
    path = Path(scratch) / "graticule-overlays.json"
    path.write_text(json.dumps(overlays) + "\n", encoding="utf-8")
    return path, {"step_deg": step, "lines": len(overlays["lines"]),
                  "labels": len(overlays["labels"])}


def render_frame(renderer_exe: Path, wrfout: Path | list[Path], *, scratch: Path,
                 index: int, products: str, frames: str, width: int,
                 height: int, heavy: bool,
                 env: dict[str, str] | None = None,
                 tag: str | None = None,
                 import_jobs: int = 1,
                 source_label: str | None = None,
                 overlays: Path | None = None
                 ) -> tuple[list[tuple[str, Path]], list[dict[str, str]],
                            list[str], list[str]]:
    """One renderer invocation over one store: a per-frame store (campaign
    convention), or with ``tag`` and a list of frames the series store.

    Returns ``(rendered, skipped, failed, invocation)`` where ``rendered``
    is ``[(slug, png_path), ...]``.  Only FAILED is a failure; an honest
    SKIPPED is relayed, never hidden."""

    tag = tag or f"{index:03d}"
    store = Path(scratch) / f"store-{tag}"
    raw_out = Path(scratch) / f"raw-{tag}"
    store.mkdir(parents=True, exist_ok=True)
    raw_out.mkdir(parents=True, exist_ok=True)
    command = [
        str(renderer_exe),
        "--store-root", str(store),
        "--out-dir", str(raw_out),
        "--products", products,
        "--frames", frames,
        "--width", str(width),
        "--height", str(height),
    ]
    if heavy:
        command.append("--heavy")
    if import_jobs > 1:
        command.extend(("--import-jobs", str(import_jobs)))
    if source_label is not None:
        command.extend(("--source-label", source_label))
    if overlays is not None:
        command.extend(("--overlays", str(overlays)))
    if isinstance(wrfout, (list, tuple)):
        command.extend(str(item) for item in wrfout)
    else:
        command.append(str(wrfout))
    result = _run_renderer(command, log_prefix="RENDER", env=env)
    rendered: list[tuple[str, Path]] = []
    skipped: list[dict[str, str]] = []
    failed: list[str] = []
    for line in (result.stdout or "").splitlines():
        if line.startswith("RENDERED "):
            slug, _, path = line[len("RENDERED "):].partition(" ")
            if path:
                rendered.append((slug, Path(path)))
        elif line.startswith("SKIPPED "):
            slug, _, reason = line[len("SKIPPED "):].partition(" ")
            skipped.append({"product": slug,
                            "reason": reason or "no reason given"})
    for line in (result.stderr or "").splitlines():
        if line.startswith("FAILED "):
            failed.append(line[len("FAILED "):])
    if result.returncode != 0 and not failed:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip()]
        failed.append(tail[-1] if tail else f"exit {result.returncode}")
    return rendered, skipped, failed, [str(part) for part in command]


def engine_lead_seconds(filename: str) -> int | None:
    """The lead one engine PNG is valid at, from its own name: the
    ``_lead_HHHhMMmSSs`` suffix a sub-hourly frame carries, else the
    ``fNNN`` segment.  ``None`` when the name is not the engine's."""

    stem = Path(filename).stem
    tail = _LEAD_SUFFIX.search(stem)
    if tail is not None:
        return int(tail["h"]) * 3600 + int(tail["m"]) * 60 + int(tail["s"])
    head = _ENGINE_HEAD.match(stem)
    if head is not None:
        return int(head["fhr"]) * 3600
    return None


def series_leads(frames_meta: list[dict]) -> list[int] | None:
    """Each frame's lead in seconds when a series can share one store:
    every frame reports a lead and no two frames share one.  ``None``
    otherwise, and the render falls back to a store per frame -- a picture
    is attributed to its frame by its lead, so two frames at one lead (or a
    converter that reported none) cannot be told apart in a shared store."""

    leads: list[int] = []
    for meta in frames_meta:
        try:
            lead = int(meta.get("lead_seconds"))
        except (TypeError, ValueError):
            return None
        if lead < 0 or lead in leads:
            return None
        leads.append(lead)
    return leads if len(leads) > 1 else None


def engine_accepts(exe: Path, flag: str) -> bool:
    """Whether an engine binary's own usage text names ``flag``.

    The frame-parallel flags (``rw_mpas_convert --jobs``, ``rw_wrfbatch
    --import-jobs``) arrived with the product door; an engine built before
    them refuses an unknown flag, so the door asks the binary instead of
    assuming, and runs one frame at a time on a build that does not say
    yes."""

    try:
        result = subprocess.run([str(exe), "--help"], capture_output=True,
                                text=True, errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return flag in (result.stdout or "") + (result.stderr or "")


def checked_frame_jobs(value) -> int:
    try:
        jobs = int(value)
    except (TypeError, ValueError):
        jobs = 0
    if not 1 <= jobs <= 64:
        raise RenderDoorError(
            f"--frame-jobs {value!r} must be a whole number from 1 to 64: it "
            f"is how many frames are converted and imported at a time")
    return jobs


def run_render(args) -> int:
    """The ``mpas-port render`` handler.  Returns the process exit code."""

    started = time.monotonic()
    out_root = Path(args.out)
    scratch = Path(args.scratch) if args.scratch else default_scratch(out_root)
    refuse_scratch_inside_out(scratch, out_root)

    # Argument coherence FIRST, before any binary is looked for: an
    # incoherent command is incoherent whether or not an engine is
    # installed, and making the user install one to be told so wastes
    # their time and hides the real complaint behind a second one.
    compose = getattr(args, "compose", None)
    compose_base_only = bool(getattr(args, "compose_base_only", False))
    if compose is not None:
        if args.history or args.mesh:
            raise RenderDoorError(
                "--compose owns the mesh and history list; passing --mesh "
                "or --history as well would leave two source lists that "
                "can disagree about which run is the base. Put every run "
                "in the compose file."
            )
    else:
        if compose_base_only:
            raise RenderDoorError(
                "--compose-base-only describes what to leave OUT of a "
                "composite, and there is no composite without --compose. "
                "A single-source render is already the base alone."
            )
        if args.window == "composite":
            raise RenderDoorError(
                "--window composite is sized to the fine regions named in "
                "--compose, and no source list was given, so there is no "
                "fine region to frame. Use --window mesh for one run's own "
                "refined core, or add --compose."
            )
        if not args.mesh or not args.history:
            raise RenderDoorError(
                "--history and --mesh are required unless --compose names "
                "them instead."
            )

    remap = resolve_remap(getattr(args, "remap", REMAP_AUTO), compose is not None)
    frame_jobs = checked_frame_jobs(getattr(args, "frame_jobs", 1) or 1)

    convert_exe = resolve_convert_exe(args.convert_exe)
    renderer_exe = resolve_renderer_exe(args.renderer_exe)
    source_label = getattr(args, "source_label", None)
    # These are hex forecast fields. The renderer's generic local-file
    # default names the WOOF engine version, not this forecast's source.
    # Default the credit only where the renderer takes --source-label: a
    # renderer that predates it keeps its own credit instead of refusing
    # every default render. A label the caller names still refuses there.
    if source_label is None and engine_accepts(renderer_exe, "--source-label"):
        source_label = "Hex forecast"
    if source_label is not None:
        if not source_label or len(source_label) > 160 or any(
                ord(c) < 32 or ord(c) > 126 for c in source_label):
            raise RenderDoorError("--source-label needs 1-160 printable ASCII characters")
        if not engine_accepts(renderer_exe, "--source-label"):
            raise RenderDoorError(f"{renderer_exe} does not support --source-label")
    convert_jobs = import_jobs = 1
    if frame_jobs > 1:
        if engine_accepts(convert_exe, "--jobs"):
            convert_jobs = frame_jobs
        else:
            print(f"FRAME-JOBS {convert_exe} predates --jobs; it converts one "
                  f"frame at a time", flush=True)
        if engine_accepts(renderer_exe, "--import-jobs"):
            import_jobs = frame_jobs
        else:
            print(f"FRAME-JOBS {renderer_exe} predates --import-jobs; it "
                  f"imports one frame at a time", flush=True)
    # Before the converter runs: a render with no map geometry is refused
    # here, by name, not discovered after minutes of resampling.
    basemap_dir = resolve_basemap_dir(
        renderer_exe, getattr(args, "basemap_dir", None))
    renderer_env = renderer_environment(basemap_dir)
    print(f"BASEMAP {basemap_dir}", flush=True)
    if compose is not None:
        compose = _require_file(
            Path(compose), "--compose",
            "Give a JSON file with a 'base' entry and an 'overlays' list, "
            "each carrying 'label', 'mesh' and 'history'.")
        mesh, histories = None, None
    else:
        mesh = _require_file(
            Path(args.mesh), "--mesh",
            "The converter cannot place Voronoi cells on the render grid "
            "without the mesh's cell coordinates; point --mesh at the run's "
            "grid/static netCDF (the file the run was initialized on).")
        histories = [
            _require_file(
                Path(item), "--history",
                "Give the run's history.*.nc output files; diag files do "
                "not carry the 3-D state the converter reads.")
            for item in args.history
        ]
    init = None
    if getattr(args, "init", None):
        if compose is not None:
            raise RenderDoorError(
                "--init belongs to a single-source render; a compose file "
                "names each run's own files.")
        init = _require_file(
            Path(args.init), "--init",
            "Point --init at the init netCDF the run started from; the "
            "converter reads zgrid from it and from nothing else.")

    diagnostics = None
    if getattr(args, "diagnostics", None):
        diagnostics = _require_file(Path(args.diagnostics), "--diagnostics",
                                    "Give the native diagnostics file at this history valid time.")

    label = getattr(args, "label", None) or default_label(mesh)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", label):
        raise RenderDoorError(
            f"--label {label!r} must be letters, digits, '.', '_' or '-' so it "
            "can lead a filename on every platform the tree is copied to")
    window = str(getattr(args, "window", "mesh") or "mesh")
    convert_window = converter_window(window)
    view = checked_view(getattr(args, "view", None) or default_view(window))
    model_label = getattr(args, "label", None) or getattr(args, "model_label", DEFAULT_MODEL_LABEL)

    try:
        width_text, height_text = str(args.size).lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except ValueError:
        raise RenderDoorError(
            f"--size must be WIDTHxHEIGHT (e.g. 1200x900), got {args.size!r}"
        ) from None

    scratch.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    invocations: list[list[str]] = []

    report, convert_invocation = convert_frames(
        convert_exe, histories=histories, mesh=mesh, scratch=scratch,
        window=convert_window, field_set=args.field_set,
        simulation_start=args.simulation_start, nc_format=args.nc_format,
        compose=compose, compose_base_only=compose_base_only, init=init, diagnostics=diagnostics,
        model_label=model_label, remap=remap, jobs=convert_jobs)
    grid_label = report.get("grid_label")
    invocations.append(convert_invocation)

    frames_meta = report.get("frames", [])
    if compose is not None:
        # A composite converts only the hours EVERY source carries, so its
        # frame count is an intersection and cannot be checked against one
        # history list.  What still has to hold is that it produced
        # something: an empty series would otherwise reach the renderer as
        # a clean run that delivered no images.
        if not frames_meta:
            raise RenderDoorError(
                f"rw_mpas_convert composed no frames from {compose}; no "
                f"valid time is carried by every source in the list, so "
                f"there is nothing to render."
            )
        histories = [Path(meta["output"]) for meta in frames_meta]
    elif len(frames_meta) != len(histories):
        raise RenderDoorError(
            f"rw_mpas_convert reported {len(frames_meta)} frames for "
            f"{len(histories)} history files; the receipt does not cover "
            f"the request, so the render does not proceed."
        )

    # Coverage gate: one real catalog per distinct absent-field signature.
    # Frames from one converter run differ only when the histories
    # themselves carry different fields.
    catalogs: dict[tuple[str, ...], dict] = {}
    catalog_summary = ""
    # One store for the whole series (--series): the frames are imported
    # once, side by side, which is the only arrangement in which the
    # renderer can close an accumulation window (qpf_6h is the stored total
    # at f012 minus the one at f006) and which costs one renderer start per
    # view, not two per frame.  THE BREAKAGE THIS PREVENTS, MEASURED
    # (2026-10-12): with a store per frame every windowed product listed as
    # "windowed accumulations need more than one stored whole-hour frame",
    # so a 48 h run could draw precipitation since init and nothing else.
    leads = None
    if getattr(args, "series", False) and compose is None:
        leads = series_leads(frames_meta)
        if leads is None:
            print("SERIES off: the frames do not carry distinct leads (or "
                  "there is one frame); one store per frame, so no windowed "
                  "product", flush=True)
    series_outputs = [Path(meta["output"]) for meta in frames_meta]
    # Every map carries a labelled latitude/longitude graticule (see
    # graticule_overlays).  One window per render, so the first frame's
    # coordinates hold for the series.
    overlays_path, graticule = None, None
    if frames_meta and engine_accepts(renderer_exe, "--overlays"):
        overlays_path, graticule = write_graticule_overlays(
            Path(frames_meta[0]["output"]), scratch)
    print(f"GRATICULE {graticule if graticule else 'none'}", flush=True)
    if leads is not None:
        rows, catalog_summary, catalog_invocation = list_catalog(
            renderer_exe, series_outputs, scratch=scratch, heavy=args.heavy,
            store=Path(scratch) / f"store-{SERIES_TAG}",
            products=args.products, env=renderer_env, import_jobs=import_jobs)
        catalogs[(SERIES_TAG,)] = rows
        invocations.append(catalog_invocation)
        if not getattr(args, "standard_set", False):
            absent = sorted({name for meta in frames_meta
                             for name in meta.get("absent_wrf_fields", [])})
            refuse_uncovered_products(args.products, rows, histories[0], absent)
    for index, (meta, history) in enumerate(zip(frames_meta, histories)):
        if leads is not None:
            break
        signature = tuple(meta.get("absent_wrf_fields", []))
        if signature not in catalogs:
            rows, catalog_summary, catalog_invocation = list_catalog(
                renderer_exe, Path(meta["output"]), scratch=scratch,
                heavy=args.heavy, store=Path(scratch) / f"store-{index:03d}",
                products=args.products, env=renderer_env)
            catalogs[signature] = rows
            invocations.append(catalog_invocation)
        if not getattr(args, "standard_set", False):
            refuse_uncovered_products(
                args.products, catalogs[signature], history,
                list(meta.get("absent_wrf_fields", [])))

    unavailable: dict[str, str] = {}
    products_effective = args.products
    if getattr(args, "standard_set", False):
        # A standard set is a GROUP the door owns, like the renderer's own
        # 'all': it expands to the members this history can draw, and every
        # member it cannot is named with the catalog's own reason in stdout
        # and the manifest -- never silently dropped.
        keep: list[str] = []
        for token in [t.strip() for t in args.products.split(",") if t.strip()]:
            statuses = [rows.get(token) for rows in catalogs.values()]
            if all(row is not None and row[1] == "renderable" for row in statuses):
                keep.append(token)
            else:
                row = next((r for r in statuses if r is not None and r[1] != "renderable"), None)
                unavailable[token] = (f"{row[1]}: {row[2]}" if row is not None
                                      else "not in the renderer catalog")
                print(f"UNAVAILABLE {token} {unavailable[token]}", flush=True)
        if not keep:
            raise RenderDoorError(
                f"none of the standard products can be drawn from this "
                f"history: {'; '.join(f'{k} ({v})' for k, v in unavailable.items())}")
        products_effective = ",".join(keep)

    results: list[FrameResult] = []
    total_rendered = total_skipped = total_failed = 0
    series_parts: dict[int, tuple[list, list, list]] = {}
    if leads is not None:
        # One renderer start draws every frame; each picture, skip and
        # failure is then handed to the frame it belongs to: a picture by
        # the lead in its own name, a skip or failure by the frame file the
        # renderer names in it (the last frame when it names none).
        rendered, skipped, failed, invocation = render_frame(
            renderer_exe, series_outputs, scratch=scratch, index=0,
            products=products_effective, frames=args.frames, width=width,
            height=height, heavy=args.heavy, env=renderer_env, tag=SERIES_TAG,
            import_jobs=import_jobs, source_label=source_label,
            overlays=overlays_path)
        invocations.append(invocation)
        by_lead = {lead: index for index, lead in enumerate(leads)}
        names = sorted(((str(path), index) for index, path in enumerate(series_outputs)),
                       key=lambda item: -len(item[0]))
        last = len(leads) - 1
        series_parts = {index: ([], [], []) for index in range(len(leads))}

        def owner(text: str) -> int:
            for name, index in names:
                if name in text:
                    return index
            return last

        for slug, png in rendered:
            index = by_lead.get(engine_lead_seconds(png.name))
            if index is None:
                # A picture no frame owns is a failure, said by name: filed
                # under a guessed frame it would carry a wrong valid day.
                series_parts[last][2].append(
                    f"{slug} {png.name}: its lead matches no frame of the series")
                continue
            series_parts[index][0].append((slug, png))
        for item in skipped:
            series_parts[owner(item["reason"])][1].append(item)
        for item in failed:
            series_parts[owner(item)][2].append(item)
    for index, (meta, history) in enumerate(zip(frames_meta, histories)):
        frame = FrameResult(
            history=str(history),
            wrfout=str(meta["output"]),
            valid_time=str(meta.get("valid_time", "")),
            valid_day=valid_day(meta.get("valid_time")),
            output_sha256=str(meta.get("output_sha256", "")),
            absent_wrf_fields=list(meta.get("absent_wrf_fields", [])),
            seam=list(meta.get("seam", [])),
        )
        if leads is not None:
            rendered, skipped, failed = series_parts[index]
            raw_out = Path(scratch) / f"raw-{SERIES_TAG}"
        else:
            rendered, skipped, failed, invocation = render_frame(
                renderer_exe, Path(meta["output"]), scratch=scratch, index=index,
                products=products_effective, frames=args.frames, width=width,
                height=height, heavy=args.heavy, env=renderer_env,
                source_label=source_label, overlays=overlays_path)
            invocations.append(invocation)
            raw_out = Path(scratch) / f"raw-{index:03d}"
        frame.skipped = skipped
        frame.failed = failed
        moved: dict[str, str] = {}
        lead = None
        for slug, png in rendered:
            domain, product = split_engine_name(png.name, slug)
            lead = lead_token(meta.get("lead_seconds"), png.name)
            name = product_file_name((getattr(args, "label", None) or grid_token(grid_label, domain)), view,
                                     product, lead)
            destination = place_path(out_root, view, product,
                                     frame.valid_day, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(png), str(destination))
            placed = str(destination.relative_to(out_root)).replace(os.sep, "/")
            in_view = destination.relative_to(out_root / view).as_posix()
            try:
                moved[png.relative_to(raw_out).as_posix()] = in_view
            except ValueError:
                moved[png.name] = in_view
            frame.rendered.append(slug)
            frame.placed.append(placed)
        carry_georef(raw_out, moved, out_root / view)
        results.append(frame)
        total_rendered += len(frame.rendered)
        total_skipped += len(skipped)
        total_failed += len(failed)
        print(f"FRAME {frame.valid_time} rendered={len(frame.rendered)} "
              f"skipped={len(skipped)} failed={len(failed)}", flush=True)
        for item in skipped:
            print(f"  SKIPPED {item['product']} {item['reason']}",
                  flush=True)
        for item in failed:
            print(f"  FAILED {item}", flush=True)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "engine": {
            "convert_exe": str(convert_exe),
            "convert_exe_sha256": sha256_file(convert_exe),
            "renderer_exe": str(renderer_exe),
            "renderer_exe_sha256": sha256_file(renderer_exe),
            "basemap_dir": str(basemap_dir),
        },
        "view": view,
        "window": window,
        "model_label": model_label,
        "source_label": source_label,
        "grid_label": grid_label,
        "graticule": graticule,
        "convert": {
            "schema": report.get("schema"),
            "engine": report.get("engine"),
            "window": report.get("window"),
            "window_spec": report.get("window_spec"),
            "window_description": report.get("window_description"),
            "mesh_spacing_km": report.get("mesh_spacing_km"),
            "mesh": str(mesh) if mesh is not None else None,
            "mesh_sha256": report.get("mesh_sha256"),
            "weights_sha256": report.get("weights_sha256"),
            "resample_method": report.get("resample_method"),
            "remap": remap,
            "product_remap": report.get("product_remap"),
            "precipitation_budget": [frame.get("precipitation_budget")
                                     for frame in frames_meta],
            "field_set": report.get("field_set"),
            # Present only on a composed render: which runs went in, how
            # many grid points each won, and how wide the seam is.  A
            # composite whose manifest did not say which runs it came from
            # would be indistinguishable from a single-run render.
            "compose_spec": str(compose) if compose is not None else None,
            "compose_base_only": compose_base_only,
            "composite": report.get("composite"),
        },
        "catalog_summary": catalog_summary,
        "label": label,
        "products_requested": args.products,
        "products_drawn": products_effective,
        "series_store": leads is not None,
        "frame_jobs": {"requested": frame_jobs, "convert": convert_jobs,
                       "import": import_jobs if leads is not None else 1},
        "products_unavailable": unavailable,
        "frames_requested": args.frames,
        "size": f"{width}x{height}",
        "counts": {"rendered": total_rendered, "skipped": total_skipped,
                   "failed": total_failed},
        "frames": [vars(frame) for frame in results],
        "invocations": invocations,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    # The any-point column store, every output hour (column_store.py), from the converted frames before the
    # scratch they sit in is cleared. A composite blends several runs, so it is no run's own column: none.
    if compose is None:
        from .column_store import write as write_columns
        manifest["columns"] = write_columns(sorted((scratch / "frames").glob("wrfout_*")),
                                            out_root / view / "columns", field_set=args.field_set)
    # A second run into the same delivered tree is legitimate (one full
    # catalog pass plus a restricted pass is the proof shape); its receipt
    # must not overwrite the first run's.  Each view keeps its receipts in
    # its own folder, so views of one run rendered side by side never race
    # for one file.
    receipt_dir = out_root / view
    receipt_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = receipt_dir / "render-manifest.json"
    if manifest_path.exists():
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        manifest_path = receipt_dir / f"render-manifest-{stamp}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"MANIFEST {manifest_path}", flush=True)

    ok = total_failed == 0 and total_rendered > 0
    if ok and not args.keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)
        print(f"SCRATCH cleared {scratch}", flush=True)
    else:
        print(f"SCRATCH kept {scratch}", flush=True)
    print(f"DOOR rendered={total_rendered} skipped={total_skipped} "
          f"failed={total_failed} out={out_root}", flush=True)
    if total_rendered == 0:
        print("DOOR nothing rendered: every requested product was skipped "
              "or failed; see the manifest's frames[] for each reason",
              flush=True)
        return 1
    return 0 if total_failed == 0 else 1


def add_render_arguments(parser) -> None:
    """Wire the ``render`` subcommand's arguments onto ``parser``."""

    parser.add_argument(
        "--history", nargs="+", default=None, metavar="FILE",
        help="MPAS history netCDF file(s), one frame each. Required unless "
             "--compose names them instead")
    parser.add_argument(
        "--mesh", default=None, metavar="FILE",
        help="the run's grid/static netCDF with cell coordinates. Required "
             "unless --compose names it instead")
    parser.add_argument(
        "--init", default=None, metavar="FILE",
        help="the run's init netCDF (the file the forecast started from). "
             "Optional: it is the converter's only source of zgrid, so the "
             "pressure-level products (500mb_height_winds, "
             "850mb_temperature_height_winds) render only when it is given")
    parser.add_argument(
        "--diagnostics", default=None, metavar="FILE",
        help="native MPAS diagnostics accompanying exactly one --history; requires --init. "
             "Both inputs are SHA-256 pinned for the Rust split-history reader.")
    parser.add_argument(
        "--compose", default=None, metavar="FILE",
        help="compose SEVERAL runs of the same hours into one frame: the "
             "coarse run everywhere, each fine run inside the ground that "
             "fine run actually resolves. The file lists a base and any "
             "number of overlays, each with its own mesh and history "
             "files. Pair it with --window composite, which frames all the "
             "fine regions at once. THE BREAKAGE THIS PREVENTS, MEASURED "
             "(2026-08-26): a cascade produces a coarse global forecast and "
             "one fine forecast per placed grid, and the only way to look "
             "at both was two separate images at two different scales. "
             "Compositing them as PICTURES would need every pixel's "
             "geographic transform and would still leave two projections "
             "and two colour scales to reconcile; compositing the DATA is "
             "one resample onto one grid and one render, so no seam can "
             "come from the drawing")
    parser.add_argument(
        "--compose-base-only", action="store_true",
        help="with --compose, frame the same window from the same sources "
             "but convert the BASE alone. The coarse counterpart to a "
             "composite, over identical ground, so a comparison between "
             "them cannot be confounded by a different map")
    parser.add_argument(
        "--out", required=True, metavar="DIR",
        help="delivered PNG tree root (<domain>/<product>/<valid-day>/)")
    parser.add_argument(
        "--products", default="all",
        help="renderer product selection: all|direct|derived|heavy|windowed "
             "or comma-separated slugs (default: all)")
    parser.add_argument(
        "--frames", default="all",
        help="renderer frame selection within each store (default: all)")
    parser.add_argument(
        "--window", default="mesh", type=window_argument,
        help="render window. 'mesh' is DERIVED from the mesh you "
             "pass: the cells within 2x the minimum spacing are its refined "
             "region, and the window is centred on them at their median "
             "spacing. 'global' is the whole sphere at the MESH'S OWN "
             "spacing (a 15 km mesh draws at 15 km, a 60 km mesh at 60 km). "
             "'region:CLAT,CLON,WIDTH_KM,HEIGHT_KM[,TRUELAT1,TRUELAT2]' is a "
             "Lambert zoom of that ground whose dx is the spacing of the mesh "
             "cells inside it; 'lambert:...' is an explicit WRF-style grid. "
             "'focus' is a fixed 22 km Lambert box over the CONUS and "
             "'global-0.25' the old fixed 0.25 degree overview; both are "
             "frozen geometry whose weights digests are recorded in evidence. "
             "THE BREAKAGE 'mesh' PREVENTS, MEASURED (2026-08-26): a mesh "
             "this project PLACED had its refined 4.5 km core in the "
             "tropical Atlantic, and the only windows that existed were a "
             "CONUS box 5,000 km away -- which samples nothing but 75 km "
             "background cells -- and a 0.25 degree (about 28 km) global "
             "grid, which cannot resolve a 4.5 km core at all. There was no "
             "way to render the run's own fine region, and every future "
             "placement would have needed another hardcoded box. "
             "'composite' needs --compose and frames every fine region in "
             "the source list at once (default: mesh)")
    parser.add_argument(
        "--view", default=None, metavar="NAME",
        help="what the window shows, as the folder and filename token "
             "(north-america, polar, ...); lower case, digits and '-'. "
             "Default: 'global' for the global windows, 'region' for a "
             "region:/lambert: spec, else the window name")
    parser.add_argument(
        "--model-label", default=DEFAULT_MODEL_LABEL, metavar="NAME",
        help="the model the plots name (metadata row and headline grid "
             "label, e.g. 'hex 15 km'); default: hex")
    parser.add_argument(
        "--source-label", default=None, metavar="TEXT",
        help="source credit printed on each picture (1-160 printable ASCII "
             "characters); default: Hex forecast where the renderer supports it")
    parser.add_argument(
        "--field-set", default="full", choices=("full", "surface"),
        help="converter field set (default: full)")
    parser.add_argument(
        "--remap", default=REMAP_AUTO, choices=REMAP_CHOICES,
        help="how mesh fields reach the render grid. 'product' (the "
             "default for a single run): linear barycentric on the mesh's "
             "own Delaunay triangles for state, first-order area-conservative "
             "for accumulated precipitation (the frame records the window "
             "integral against the mesh's), nearest cell for categories and "
             "soil. 'nearest': every field the value of the nearest cell. "
             "'auto' is product for a single run and nearest for --compose, "
             "which switches source point by point and has no single mesh to "
             "interpolate across")
    parser.add_argument(
        "--series", action="store_true",
        help="import every frame into ONE store and draw the series with one "
             "renderer start. Needed for the windowed products (qpf_1h, "
             "qpf_6h, qpf_24h, 2m_temp_0_24h_max, ...): an accumulation "
             "window is the difference of two stored frames, and a store "
             "per frame holds one. Off by default so an existing render "
             "keeps its per-frame stores; the product door turns it on. "
             "Frames without distinct leads fall back to a store per frame")
    parser.add_argument(
        "--frame-jobs", type=int, default=1, metavar="N",
        help="frames converted at a time and, with --series, imported at a "
             "time (rw_mpas_convert --jobs, rw_wrfbatch --import-jobs). The "
             "pictures are the same for every N; each frame in flight holds "
             "its fields in memory, so the default is 1 and the product "
             "door sizes N from the host's free memory. An engine built "
             "before these flags runs one frame at a time and the door says "
             "so")
    parser.add_argument(
        "--label", default=None, metavar="MODEL_MESH",
        help="leading token of every delivered filename, which reads "
             "<label>_<YYYYMMDD>_<HH>z_f<lead>_<product>.png (default: hex_ "
             "plus the mesh file's name without its role tokens, e.g. "
             "hex_x1.2621442)")
    parser.add_argument(
        "--simulation-start", default=None, metavar="YYYY-MM-DD_HH:MM:SS",
        help="run start, for true lead hours in filenames")
    parser.add_argument(
        "--size", default="1200x900",
        help="output pixels as WIDTHxHEIGHT (default: 1200x900)")
    parser.add_argument(
        "--heavy", action="store_true",
        help="include the heavy product family (off by default)")
    parser.add_argument(
        "--nc-format", default="cdf2", choices=("cdf2", "cdf5"),
        help="converted frame netCDF variant (default: cdf2)")
    parser.add_argument(
        "--scratch", default=None, metavar="DIR",
        help="conversion/render scratch dir OUTSIDE --out "
             "(default: sibling <out>.render-scratch)")
    parser.add_argument(
        "--keep-scratch", action="store_true",
        help="keep the scratch dir after a clean run")
    parser.add_argument(
        "--basemap-dir", default=None, metavar="DIR",
        help=f"the renderer's map geometry (natural_earth_10m/, "
             f"natural_earth_110m/, us_counties_5m/). Default: "
             f"${BASEMAP_ENV}, then assets/basemap above the renderer, then "
             f"gpuwm's own assets. A render with none is refused: it would "
             f"draw every map without coastlines, borders or state lines")
    parser.add_argument(
        "--convert-exe", default=None, metavar="FILE",
        help=f"rw_mpas_convert binary (else ${CONVERT_ENV}, then PATH)")
    parser.add_argument(
        "--renderer-exe", default=None, metavar="FILE",
        help=f"rw_wrfbatch binary (else ${RENDERER_ENV}, "
             f"${RENDERER_ENV_GPUWM}, then PATH)")
