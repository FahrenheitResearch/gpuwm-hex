"""Runtime mesh rows: a generated mesh registers itself from its own bytes.

THE BLOCKER THIS REMOVES.  ``gpuwm-hex forecast`` resolves ``--mesh``
against ``hexcore.drivers.mpas_mesh_binding.MESH_BINDINGS`` and refuses a name it
does not hold.  Every row in that file was typed by a person after the mesh
existed.  ``gpuwm-hex mesh-plan --point`` generates a mesh at a point nobody
typed a row for -- the whole point of the door is that the point is
arbitrary -- so the registry could never hold its row in advance, and a
checkout edit per point would be the hand-written-row-per-swath defect
``hexcore.cascade_row`` retired for the cascade, wearing a different hat.

WHAT THIS MODULE IS, AND HOW IT DIFFERS FROM ``cascade_row``.  A cascade row
describes a CULL of a row somebody registered, and it insists on that:
lineage stops at a person.  A runtime row describes a mesh the GENERATOR
made from a spec the door wrote, so its lineage stops at the spec and at the
generator's own receipt:

* a ``generated-global`` row names the spec by SHA-256, the generator
  receipt, the static receipt, and the pair's bytes;
* a ``generated-parent-grid`` row is a global parent described by its grid
  alone: the door built geography only around the cull it cut from it (see
  :mod:`hexcore.mesh_point`, the halo route), so the parent has no static,
  is not a runnable mesh, and is registered as LINEAGE for its culls only;
* a ``generated-cull`` row names the ``generated-global`` row it was cut
  from (which must be in the same file, or already registered), the cull
  receipt, the region, the boundary-mask digest and the boundary source.

Everything a registry row BUYS is kept, and it is worth restating which
half is a review and which is a measurement (``cascade_row`` says the same):
the bytes are pinned by count and SHA-256 and re-hashed at bind; the
Courant, dual-edge, cell-coordination and regional admissions are
measurements of the files at bind; the timestep is refused unless an anchor
covers the configuration; the regional forecast opener re-measures the class
key and requires a contract deck on THESE rings.  None of those become
weaker because the row was written by the door that generated the mesh.
What is genuinely lost is a person reading the row; what stands in for that
is the door's own admission pass on the pair it just built (dual edge, cell
coordination, Courant at the declared timestep), recorded in the row file
and refused at write time when it fails.

WHAT THIS MODULE REFUSES, by name:

* a name that collides with a shipped registry row (a runtime row may never
  shadow one);
* a cull row whose parent is neither in the same file nor registered;
* a row of any kind but ``generated-parent-grid`` that names no static, and a
  ``generated-parent-grid`` row that names one;
* a row whose declared bytes do not match the files on disk now;
* a cull row carrying a boundary zone and no ``lbc_source``;
* a file that is not a ``gpuwm-hex.mesh-rows/v1`` document.

Rows are supplied through a FILE, like cascade rows, because the forecast
door runs the driver in another process.  ``hexcore.drivers.mpas_mesh_binding``
applies them where it builds ``MESH_BINDINGS`` -- before the cascade rows,
so a cascade may cull a generated parent -- and the door's by-path
re-execution of that module picks them up as the first copy did.  The file
is named by ``$GPUWM_HEX_MESH_ROWS`` (``os.pathsep``-separated when there
are several) and, by convention, lives beside the mesh it describes as
``mesh-rows.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .errors import MpasPortError

#: The document schema this module writes and reads.
ROWS_SCHEMA = "gpuwm-hex.mesh-rows/v1"

#: Where the forecast door is told to look for runtime rows.
MESH_ROWS_ENVIRONMENT = "GPUWM_HEX_MESH_ROWS"

#: The file name a generating door leaves beside the mesh.
MESH_ROWS_FILENAME = "mesh-rows.json"

#: Stamped into a row's ``notes`` so the bind log and the run receipt say
#: what the row is, in the row's own words.
GENERATED_ROW_MARKER = "GENERATED-AT-POINT"
CULL_ROW_MARKER = "GENERATED-CULL"

#: A global parent described by its grid alone.  Its static was built only
#: for the neighbourhood of its cull, so nothing can run it; it is the
#: lineage a ``generated-cull`` row names, verified by its grid's bytes.
PARENT_GRID_KIND = "generated-parent-grid"

ARCHIVED_GLOBAL_KIND = "archived-global"
ROW_KINDS = ("generated-global", "generated-cull", PARENT_GRID_KIND, ARCHIVED_GLOBAL_KIND)


class MeshRowRefusal(MpasPortError):
    """A runtime mesh row is refused, and the message says why."""


@dataclass(frozen=True, slots=True)
class MeshRow:
    """One generated mesh, described by the receipts that made it."""

    kind: str
    name: str
    n_cells: int
    n_edges: int
    n_levels: int
    n_interfaces: int
    n_soil_levels: int
    nominal_dx_m: float
    dt_seconds: float
    grid: str
    grid_bytes: int
    grid_sha256: str
    #: ``None`` on a ``generated-parent-grid`` row, and only there.
    static: str | None
    static_bytes: int | None
    static_sha256: str | None
    #: The resolution spec the generator was handed, by digest, and the
    #: receipts it and the static builder wrote.  Lineage stops here.
    spec_sha256: str
    generator_receipt: str
    static_receipt: str | None
    #: The point, fine spacing and core radius the spec was built from.
    point_deg: tuple[float, float]
    fine_dx_m: float
    core_radius_km: float
    background_km: float
    #: The door's own admission pass over the pair, recorded so the row
    #: says what was measured before it was written.
    admission: Mapping[str, Any]
    # -- cull rows only ---------------------------------------------------
    parent_row: str | None = None
    boundary_zone_width: int | None = None
    bdy_mask_sha256: str | None = None
    lbc_source: str | None = None
    cull_receipt: str | None = None
    cull_region: Mapping[str, Any] | None = None
    cull_pad_scale: float | None = None
    resolution_spec: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, Mapping):
                value = json.loads(json.dumps(dict(value)))
            elif isinstance(value, tuple):
                value = list(value)
            out[item.name] = value
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MeshRow":
        names = {item.name for item in fields(cls)}
        missing = sorted(name for name in names if name not in raw
                         and name in _REQUIRED_KEYS)
        if missing:
            raise MeshRowRefusal(
                f"a mesh row is missing {missing}; a row that does not say "
                f"what it pins cannot be bound.  Rows are written by "
                f"`gpuwm-hex mesh-plan --point --generate`, never by hand"
            )
        unknown = sorted(name for name in raw if name not in names)
        if unknown:
            raise MeshRowRefusal(
                f"a mesh row carries fields this build does not know "
                f"({unknown}); a row from a newer door is refused rather than "
                f"read with its unknown fields ignored"
            )
        values = dict(raw)
        values["point_deg"] = tuple(float(v) for v in values["point_deg"])
        values["admission"] = MappingProxyType(dict(values.get("admission") or {}))
        if values.get("resolution_spec") is not None:
            values["resolution_spec"] = MappingProxyType(dict(values["resolution_spec"]))
        if values.get("cull_region") is not None:
            values["cull_region"] = MappingProxyType(dict(values["cull_region"]))
        return cls(**{name: values.get(name) for name in names})

    def notes(self) -> str:
        lat, lon = self.point_deg
        if self.resolution_spec is not None:
            return (
                f"GENERATED-GLOBAL-VR: resolution spec SHA-256 {self.spec_sha256}; "
                f"generator receipt {self.generator_receipt}; static receipt {self.static_receipt}. "
                f"{len(self.resolution_spec.get('regions', []))} refinement regions, "
                f"{self.background_km:g} km background, {self.fine_dx_m:g} m finest requested. "
                "Grid and static bytes are pinned and admissions are measured at bind. "
                "Generation is not an observation-skill qualification."
            )
        if self.kind == ARCHIVED_GLOBAL_KIND:
            return (
                f"ARCHIVED-GLOBAL: external mesh archive and preprocessing provenance "
                f"at {self.generator_receipt}, source specification digest "
                f"{self.spec_sha256}; static provenance at {self.static_receipt}. "
                f"Grid and static bytes are pinned independently. Geometry and "
                f"Courant admissions are measured at bind for declared dt "
                f"{self.dt_seconds:g} s. Archive registration earns no native "
                f"endpoint or observation-skill qualification."
            )
        if self.kind == "generated-cull":
            return (
                f"{CULL_ROW_MARKER}: a limited-area cull of the runtime row "
                f"{self.parent_row!r} (a {self.fine_dx_m:g} m core of "
                f"{self.core_radius_km:g} km radius at {lat:.4f} N {lon:.4f} E "
                f"on a {self.background_km:g} km background), cut at pad "
                f"{float(self.cull_pad_scale or 0.0):g} with rw_mpas_mesh "
                f"--cull-parent.  Nobody hand-wrote this row: it is written "
                f"from the cull receipt at {self.cull_receipt}, the parent's "
                f"generator receipt at {self.generator_receipt}, and the spec "
                f"digest {self.spec_sha256[:16]}...; every admission behind "
                f"it is a measurement of these files at bind, and the "
                f"regional opener requires a contract deck on these rings "
                f"and re-measures the class key.  Boundary series: "
                f"{self.lbc_source}."
            )
        if self.kind == PARENT_GRID_KIND:
            return (
                f"{GENERATED_ROW_MARKER}: the global parent of a {self.fine_dx_m:g} m "
                f"core of {self.core_radius_km:g} km radius at {lat:.4f} N "
                f"{lon:.4f} E on a {self.background_km:g} km background, generated "
                f"by rw_mpas_mesh from the spec digest {self.spec_sha256[:16]}... "
                f"(receipt {self.generator_receipt}).  Described by its grid "
                f"alone: its geography was built only around the cull cut from "
                f"it, so this row is that cull's lineage and not a runnable "
                f"mesh.  Dual edges, cell coordination and Courant at dt "
                f"{self.dt_seconds:g} s were measured on its grid before it was "
                f"written."
            )
        return (
            f"{GENERATED_ROW_MARKER}: a {self.fine_dx_m:g} m core of "
            f"{self.core_radius_km:g} km radius at {lat:.4f} N {lon:.4f} E on a "
            f"{self.background_km:g} km global background, generated by "
            f"rw_mpas_mesh from the spec digest {self.spec_sha256[:16]}... "
            f"(receipt {self.generator_receipt}); static by rw_mpas_static "
            f"(receipt {self.static_receipt}).  Nobody hand-wrote this row: "
            f"the door that generated the pair measured dual edges, cell "
            f"coordination and Courant at dt {self.dt_seconds:g} s before "
            f"writing it, and the bind measures them again."
        )


_REQUIRED_KEYS = {
    "kind", "name", "n_cells", "n_edges", "n_levels", "n_interfaces",
    "n_soil_levels", "nominal_dx_m", "dt_seconds", "grid", "grid_bytes",
    "grid_sha256", "static", "static_bytes", "static_sha256", "spec_sha256",
    "generator_receipt", "static_receipt", "point_deg", "fine_dx_m",
    "core_radius_km", "background_km", "admission",
}


def sha256_file(path: Path) -> str:
    # The real SHA-256, computed once per file version (hexcore.file_digest):
    # every forecast re-verified the same grid and static bytes.
    from hexcore.file_digest import sha256_file as file_version_sha256
    return file_version_sha256(path)


# ---------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------
def write_rows(path: Path, rows: Sequence[MeshRow]) -> Path:
    """Write (replace) the runtime row document at ``path``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [row.name for row in rows]
    if len(set(names)) != len(names):
        raise MeshRowRefusal(
            f"two rows in one document share a name ({names}); a name must "
            f"resolve to exactly one pair of files"
        )
    path.write_text(
        json.dumps(
            {"schema": ROWS_SCHEMA, "rows": [row.as_dict() for row in rows]},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def read_rows(path: Path) -> list[MeshRow]:
    """The rows one document holds, refused by name when it is not one."""

    location = Path(path)
    if not location.is_file():
        raise MeshRowRefusal(
            f"{location} is not a file.  A door told to read runtime mesh "
            f"rows from a file that is not there would fall back to the "
            f"shipped registry and run the wrong mesh"
        )
    try:
        document = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MeshRowRefusal(f"{location} is not readable JSON: {error}") from error
    if not isinstance(document, dict) or document.get("schema") != ROWS_SCHEMA:
        raise MeshRowRefusal(
            f"{location} is not a {ROWS_SCHEMA} document (schema "
            f"{document.get('schema') if isinstance(document, dict) else None!r})"
        )
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise MeshRowRefusal(f"{location} carries no 'rows' list")
    return [MeshRow.from_dict(raw) for raw in rows]


def append_row(path: Path, row: MeshRow) -> Path:
    """Add one row to a document, creating it when absent."""

    location = Path(path)
    existing = read_rows(location) if location.is_file() else []
    if any(item.name == row.name for item in existing):
        raise MeshRowRefusal(
            f"{location} already holds a row named {row.name!r}; a generated "
            f"pair is registered once, and re-registering it would let two "
            f"sets of bytes claim one name"
        )
    return write_rows(location, [*existing, row])


def row_files(path: Path | str | None = None) -> list[Path]:
    """The documents to read: an explicit path, or ``$GPUWM_HEX_MESH_ROWS``."""

    if path is not None:
        return [Path(path)]
    raw = os.environ.get(MESH_ROWS_ENVIRONMENT, "")
    return [Path(piece.strip()) for piece in raw.split(os.pathsep) if piece.strip()]


def load_rows(path: Path | str | None = None) -> list[MeshRow]:
    rows: list[MeshRow] = []
    for location in row_files(path):
        rows.extend(read_rows(location))
    return rows


# ---------------------------------------------------------------------------
# describing a pair
# ---------------------------------------------------------------------------
def _dimensions(path: Path) -> dict[str, int]:
    from netCDF4 import Dataset

    with Dataset(str(path)) as dataset:
        return {key: int(value.size) for key, value in dataset.dimensions.items()}


def describe_generated(
    *,
    name: str,
    grid: Path,
    static: Path | None,
    spec_path: Path,
    generator_receipt: Path,
    static_receipt: Path | None,
    point_deg: tuple[float, float],
    fine_dx_m: float,
    core_radius_km: float,
    background_km: float,
    dt_seconds: float,
    n_levels: int,
    admission: Mapping[str, Any],
    n_soil_levels: int = 4,
) -> MeshRow:
    """Measure a freshly-generated pair and write the row that describes it.

    ``static=None`` describes the parent by its grid alone: a
    ``generated-parent-grid`` row, the lineage of a cull whose geography was
    built around the cull only.
    """

    grid = Path(grid)
    dims = _dimensions(grid)
    for key in ("nCells", "nEdges"):
        if key not in dims:
            raise MeshRowRefusal(
                f"{grid} declares no {key}; it is not an MPAS grid and cannot "
                f"be registered"
            )
    if static is None:
        return MeshRow(
            kind=PARENT_GRID_KIND,
            name=name,
            n_cells=int(dims["nCells"]),
            n_edges=int(dims["nEdges"]),
            n_levels=int(n_levels),
            n_interfaces=int(n_levels) + 1,
            n_soil_levels=int(n_soil_levels),
            nominal_dx_m=float(fine_dx_m),
            dt_seconds=float(dt_seconds),
            grid=str(grid),
            grid_bytes=grid.stat().st_size,
            grid_sha256=sha256_file(grid),
            static=None,
            static_bytes=None,
            static_sha256=None,
            spec_sha256=sha256_file(Path(spec_path)),
            generator_receipt=str(generator_receipt),
            static_receipt=None,
            point_deg=(float(point_deg[0]), float(point_deg[1])),
            fine_dx_m=float(fine_dx_m),
            core_radius_km=float(core_radius_km),
            background_km=float(background_km),
            admission=MappingProxyType(dict(admission)),
        )
    static = Path(static)
    static_dims = _dimensions(static)
    if static_dims.get("nCells") != dims["nCells"]:
        raise MeshRowRefusal(
            f"{grid} has nCells={dims['nCells']} and {static} has "
            f"nCells={static_dims.get('nCells')}; the pair was not generated "
            f"together and a row pinning both would bind a static to the "
            f"wrong mesh"
        )
    return MeshRow(
        kind="generated-global",
        name=name,
        n_cells=int(dims["nCells"]),
        n_edges=int(dims["nEdges"]),
        n_levels=int(n_levels),
        n_interfaces=int(n_levels) + 1,
        n_soil_levels=int(n_soil_levels),
        nominal_dx_m=float(fine_dx_m),
        dt_seconds=float(dt_seconds),
        grid=str(grid),
        grid_bytes=grid.stat().st_size,
        grid_sha256=sha256_file(grid),
        static=str(static),
        static_bytes=static.stat().st_size,
        static_sha256=sha256_file(static),
        spec_sha256=sha256_file(Path(spec_path)),
        generator_receipt=str(generator_receipt),
        static_receipt=str(static_receipt),
        point_deg=(float(point_deg[0]), float(point_deg[1])),
        fine_dx_m=float(fine_dx_m),
        core_radius_km=float(core_radius_km),
        background_km=float(background_km),
        admission=MappingProxyType(dict(admission)),
    )


def describe_cull(
    *,
    name: str,
    parent: MeshRow,
    grid: Path,
    static: Path,
    cull_receipt: Path,
    cull_region: Mapping[str, Any],
    cull_pad_scale: float,
    lbc_source: str,
    admission: Mapping[str, Any],
    dt_seconds: float | None = None,
    static_receipt: Path | None = None,
) -> MeshRow:
    """Measure a freshly-cut pair and write the cull row that describes it.

    ``static_receipt`` names the receipt of the static this cull's static was
    cut from when that is not the parent's own (the halo route builds it on a
    wider cut of the parent, which has none).
    """

    from netCDF4 import Dataset
    import numpy as np

    from .mesh import REGIONAL_BOUNDARY_MASK_NAMES, regional_boundary_mask_digest

    grid = Path(grid)
    static = Path(static)
    with Dataset(str(grid)) as dataset:
        dataset.set_auto_maskandscale(False)
        dims = {key: int(value.size) for key, value in dataset.dimensions.items()}
        masks = {
            item: dataset.variables[item][:]
            for item in REGIONAL_BOUNDARY_MASK_NAMES
            if item in dataset.variables
        }
        if len(masks) != len(REGIONAL_BOUNDARY_MASK_NAMES):
            raise MeshRowRefusal(
                f"{grid} carries {sorted(masks)} of the boundary-mask triple "
                f"{list(REGIONAL_BOUNDARY_MASK_NAMES)}; a cull with an "
                f"incomplete triple cannot identify its own rings"
            )
        zone_width = int(np.max(np.asarray(masks["bdyMaskCell"])))
        digest = regional_boundary_mask_digest(masks)
    if zone_width <= 0:
        raise MeshRowRefusal(
            f"{grid} carries an all-zero boundary mask, so it is not a "
            f"limited-area cull; register it as a global row instead"
        )
    if not lbc_source:
        raise MeshRowRefusal(
            f"a cull row for {name!r} declares a {zone_width}-ring boundary "
            f"zone and no lbc_source, so nothing is registered to force it "
            f"and a run would integrate an unforced boundary inward"
        )
    return MeshRow(
        kind="generated-cull",
        name=name,
        n_cells=int(dims["nCells"]),
        n_edges=int(dims["nEdges"]),
        n_levels=parent.n_levels,
        n_interfaces=parent.n_interfaces,
        n_soil_levels=parent.n_soil_levels,
        nominal_dx_m=parent.nominal_dx_m,
        dt_seconds=float(parent.dt_seconds if dt_seconds is None else dt_seconds),
        grid=str(grid),
        grid_bytes=grid.stat().st_size,
        grid_sha256=sha256_file(grid),
        static=str(static),
        static_bytes=static.stat().st_size,
        static_sha256=sha256_file(static),
        spec_sha256=parent.spec_sha256,
        generator_receipt=parent.generator_receipt,
        static_receipt=(
            str(static_receipt) if static_receipt is not None else parent.static_receipt
        ),
        point_deg=parent.point_deg,
        fine_dx_m=parent.fine_dx_m,
        core_radius_km=parent.core_radius_km,
        background_km=parent.background_km,
        admission=MappingProxyType(dict(admission)),
        parent_row=parent.name,
        boundary_zone_width=zone_width,
        bdy_mask_sha256=digest,
        lbc_source=str(lbc_source),
        cull_receipt=str(cull_receipt),
        cull_region=MappingProxyType(json.loads(json.dumps(dict(cull_region)))),
        cull_pad_scale=float(cull_pad_scale),
    )


# ---------------------------------------------------------------------------
# applying rows to the registry
# ---------------------------------------------------------------------------
def _verify_bytes(row: MeshRow) -> None:
    grid_only = row.kind == PARENT_GRID_KIND
    if grid_only and row.static is not None:
        raise MeshRowRefusal(
            f"the {PARENT_GRID_KIND} row {row.name!r} names a static at "
            f"{row.static}; that kind describes a parent with no static, so a "
            f"row carrying one is not what it says it is"
        )
    if not grid_only and (row.static is None or row.static_sha256 is None
                          or row.static_bytes is None):
        raise MeshRowRefusal(
            f"the {row.kind} row {row.name!r} names no static; a runnable row "
            f"pins its static by bytes and digest, and only a "
            f"{PARENT_GRID_KIND} row may go without one"
        )
    checks = [("grid", Path(row.grid), row.grid_bytes, row.grid_sha256)]
    if not grid_only:
        checks.append(("static", Path(row.static), row.static_bytes, row.static_sha256))
    for role, file_path, want_bytes, want_sha in checks:
        if not file_path.is_file():
            raise MeshRowRefusal(
                f"the mesh row {row.name!r} names a {role} at {file_path} "
                f"that is not a file"
            )
        size = file_path.stat().st_size
        if size != want_bytes:
            raise MeshRowRefusal(
                f"the mesh row {row.name!r} declares a {role} of {want_bytes} "
                f"bytes and {file_path} is {size}.  The row was written from "
                f"the generator's receipt; a mismatch means the file moved "
                f"under it"
            )
        measured = sha256_file(file_path)
        if measured != want_sha:
            raise MeshRowRefusal(
                f"the mesh row {row.name!r} declares a {role} SHA-256 "
                f"{want_sha[:16]}... and {file_path} digests to "
                f"{measured[:16]}..."
            )


def apply_rows(
    registry: Mapping[str, Any], binding_type: Any, path: Path | str | None = None
) -> Mapping[str, Any]:
    """Return ``registry`` with the runtime rows added.

    Returns the mapping UNCHANGED when no runtime row file is named, which
    is every ordinary run.
    """

    rows = load_rows(path)
    if not rows:
        return registry
    patched = dict(registry)
    #: Parents described by their grid alone: verified, never bound.
    lineage: dict[str, MeshRow] = {}
    for row in rows:
        if row.kind not in ROW_KINDS:
            raise MeshRowRefusal(
                f"the mesh row {row.name!r} is of kind {row.kind!r}; this "
                f"build knows {list(ROW_KINDS)}"
            )
        if row.name in registry:
            raise MeshRowRefusal(
                f"the runtime row {row.name!r} collides with a shipped "
                f"registry row.  A runtime row may never shadow one: the "
                f"shipped row pins bytes a person reviewed and this one "
                f"would replace them silently"
            )
        if row.name in patched or row.name in lineage:
            raise MeshRowRefusal(
                f"two runtime row files both register {row.name!r}; a name "
                f"must resolve to exactly one pair of files"
            )
        _verify_bytes(row)
        if row.kind == PARENT_GRID_KIND:
            lineage[row.name] = row
            continue
        regional: dict[str, Any] = {}
        if row.kind == "generated-cull":
            if not row.parent_row or (
                row.parent_row not in patched and row.parent_row not in lineage
            ):
                raise MeshRowRefusal(
                    f"the cull row {row.name!r} names parent "
                    f"{row.parent_row!r}, which is neither in its own row "
                    f"file nor a registered mesh.  Lineage stops at a "
                    f"generated-global row: a cull of a mesh nobody described "
                    f"is how a shape nobody looked at becomes a forecast"
                )
            if not row.lbc_source:
                raise MeshRowRefusal(
                    f"the cull row {row.name!r} declares a "
                    f"{row.boundary_zone_width}-ring boundary zone and no "
                    f"lbc_source, so nothing is registered to force it"
                )
            regional = {
                "boundary_zone_width": int(row.boundary_zone_width or 0),
                "bdy_mask_sha256": row.bdy_mask_sha256,
                "lbc_source": row.lbc_source,
            }
        # A generated global row's background is its coarsest cell, which
        # the convection decision needs from the row alone at preflight (a
        # variable mesh is decided per cell; hexcore.convection_admission).
        # A cull keeps only part of its parent, so it carries no claim.
        if row.kind != "generated-cull" and "coarsest_dx_m" in getattr(
            binding_type, "__dataclass_fields__", {}
        ):
            regional["coarsest_dx_m"] = float(row.background_km) * 1000.0
        patched[row.name] = binding_type(
            name=row.name,
            n_cells=row.n_cells,
            n_edges=row.n_edges,
            n_levels=row.n_levels,
            n_interfaces=row.n_interfaces,
            n_soil_levels=row.n_soil_levels,
            nominal_dx_m=row.nominal_dx_m,
            dt_seconds=row.dt_seconds,
            grid_bytes=row.grid_bytes,
            grid_sha256=row.grid_sha256,
            static_bytes=row.static_bytes,
            static_sha256=row.static_sha256,
            drop_carried_deformation=True,
            notes=row.notes(),
            **regional,
        )
    return MappingProxyType(patched)


__all__ = [
    "CULL_ROW_MARKER",
    "GENERATED_ROW_MARKER",
    "MESH_ROWS_ENVIRONMENT",
    "MESH_ROWS_FILENAME",
    "ROWS_SCHEMA",
    "ROW_KINDS",
    "MeshRow",
    "MeshRowRefusal",
    "PARENT_GRID_KIND",
    "append_row",
    "apply_rows",
    "describe_cull",
    "describe_generated",
    "load_rows",
    "read_rows",
    "row_files",
    "sha256_file",
    "write_rows",
]


def describe_resolution_spec(*, spec: Mapping[str, Any], **kwargs: Any) -> MeshRow:
    """Register an authored global spec without a hand registry row."""
    from dataclasses import replace

    if kwargs["admission"].get("regional"):
        raise MeshRowRefusal("a global resolution spec produced a regional mesh")
    row = describe_generated(**kwargs)
    if row.kind != "generated-global":
        raise MeshRowRefusal("a runnable global VR row requires the generated static")
    return replace(row, resolution_spec=MappingProxyType(dict(spec)))
