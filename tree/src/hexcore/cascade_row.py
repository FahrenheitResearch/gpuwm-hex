"""Registering a cull the cascade placed this cycle, from its own bytes.

THE BLOCKER THIS REMOVES, and it is not the anchor.  ``gpuwm-hex forecast``
resolves ``--mesh`` against ``tools/mpas_mesh_binding.MESH_BINDINGS`` and
refuses a name it does not hold, by name:

    running an unregistered mesh would run an unproved shape at an unadmitted
    timestep

That refusal is right about a mesh somebody downloaded.  It is unanswerable
for a mesh the cascade CUT four seconds ago: a storm-following swath
re-places itself every cycle, so a cycling cascade would need a hand-written
Python row per swath per cycle, forever, and could never run unattended.

WHAT A REGISTRY ROW ACTUALLY BUYS, separated.  The row does two different
things and only one of them needs a human:

1. **It pins bytes.**  Byte count and SHA-256, so the file you run is the file
   that was reviewed.  For a cull the cascade made and hashed in the same
   chain, that guarantee is available by construction -- the row is written
   FROM the cull receipt, and ``bind_mesh`` re-hashes both files anyway, so a
   row that lies about its bytes refuses at bind exactly as a hand-written one
   would.

2. **It carries a declared shape and timestep** -- and every gate behind those
   is a MEASUREMENT, not a review.  The Courant admission reads the file's own
   ``dcEdge``; the dual-edge admission reads its own ``dvEdge/dcEdge``; the
   cell-coordination admission counts its own polygon sides; the regional
   admission re-measures its own ``bdyMask`` digest and ring shell; the
   timestep is refused unless an anchor covers it; the device admission
   measures the card.  None of those become weaker because nobody typed the
   row.

WHAT IS GENUINELY LOST, and it is stated rather than argued away: a human
reading the row is a real check, and a cascade row does not have one.  What
replaces it is the per-geometry CONTRACT DECK -- 8 decks bitwise against the
v8.4.1 CPU authority on this cull's own zone geometry, every mutation control
with teeth -- which the cascade runs before the forecast and which is a
stronger statement about the rings than a person reading a cell count.  A
cascade row with no deck behind it is refused at the regional gate regardless
of what this module does.

WHAT THIS MODULE REFUSES:

* a row whose cull receipt does not name a parent that is itself a REGISTERED
  row.  Lineage stops at something a person did register, always -- a cull of
  a cull of an unregistered mesh is how a shape nobody ever looked at becomes
  a forecast;
* a row whose declared bytes do not match the files on disk right now;
* a row carrying a boundary zone and no ``lbc_source``, which is the existing
  registry refusal and is not relaxed here;
* a name that collides with a shipped row.  A cascade row may never shadow one.

Rows are supplied through a FILE, not a context manager, because the door
runs the driver in another process and a process-local override would be
invisible to it.  ``tools/mpas_mesh_binding`` applies them where it builds
``MESH_BINDINGS``, the same hook ``mesh_row_candidate`` uses, so the door's
by-path re-execution of that module picks them up as the first copy did.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .errors import MpasPortError

#: Stamped into a cascade row's ``notes`` so the bind log and the run receipt
#: both say what the row is, in the row's own words.
CASCADE_ROW_MARKER = "CASCADE-CULL"

#: Where the cascade leaves the rows it registered this cycle.
CASCADE_ROWS_ENVIRONMENT = "GPUWM_HEX_CASCADE_ROWS"


class CascadeRowRefusal(MpasPortError):
    """A cascade-registered mesh row is refused, and the message says why."""


@dataclass(frozen=True, slots=True)
class CascadeRow:
    """One cull, described by the chain that made it."""

    name: str
    parent_row: str
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
    static: str
    static_bytes: int
    static_sha256: str
    boundary_zone_width: int
    bdy_mask_sha256: str
    lbc_source: str
    cull_receipt: str
    cycle_index: int
    slot_id: str
    cull_pad_scale: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parent_row": self.parent_row,
            "n_cells": self.n_cells,
            "n_edges": self.n_edges,
            "n_levels": self.n_levels,
            "n_interfaces": self.n_interfaces,
            "n_soil_levels": self.n_soil_levels,
            "nominal_dx_m": self.nominal_dx_m,
            "dt_seconds": self.dt_seconds,
            "grid": self.grid,
            "grid_bytes": self.grid_bytes,
            "grid_sha256": self.grid_sha256,
            "static": self.static,
            "static_bytes": self.static_bytes,
            "static_sha256": self.static_sha256,
            "boundary_zone_width": self.boundary_zone_width,
            "bdy_mask_sha256": self.bdy_mask_sha256,
            "lbc_source": self.lbc_source,
            "cull_receipt": self.cull_receipt,
            "cycle_index": self.cycle_index,
            "slot_id": self.slot_id,
            "cull_pad_scale": self.cull_pad_scale,
        }

    def notes(self) -> str:
        return (
            f"{CASCADE_ROW_MARKER}: cut by the cycling cascade at cycle "
            f"{self.cycle_index}, slot {self.slot_id}, from the REGISTERED "
            f"parent row {self.parent_row!r} at cull_pad_scale "
            f"{self.cull_pad_scale:g}. Nobody hand-wrote this row: it is "
            f"written from the cull receipt at {self.cull_receipt}, and every "
            f"admission behind it is a measurement of these files rather than "
            f"a declaration about them -- Courant against this grid's own "
            f"dcEdge, dual-edge and cell-coordination against its own "
            f"geometry, the boundary zone against its own bdyMask digest, the "
            f"timestep against the anchor table, and the 22 regional kernels "
            f"against the v8.4.1 CPU authority on this cull's own rings. The "
            f"one thing it does NOT have is a person who read it; what stands "
            f"in for that is the contract deck, which is a stronger statement "
            f"about the rings than a reader is. Boundary series: "
            f"{self.lbc_source}."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_cull(
    *,
    name: str,
    parent_row: str,
    grid: Path,
    static: Path,
    cull_receipt: Path,
    init: Path,
    dt_seconds: float,
    nominal_dx_m: float,
    lbc_source: str,
    cycle_index: int,
    slot_id: str,
    cull_pad_scale: float,
) -> CascadeRow:
    """Measure a freshly-cut pair and write the row that describes it."""

    from netCDF4 import Dataset

    grid = Path(grid)
    static = Path(static)
    from .mesh import REGIONAL_BOUNDARY_MASK_NAMES, regional_boundary_mask_digest

    with Dataset(str(grid)) as dataset:
        dataset.set_auto_maskandscale(False)
        dimensions = {key: int(value.size) for key, value in dataset.dimensions.items()}
        masks = {
            item: dataset.variables[item][:]
            for item in REGIONAL_BOUNDARY_MASK_NAMES
            if item in dataset.variables
        }
        if len(masks) != len(REGIONAL_BOUNDARY_MASK_NAMES):
            raise CascadeRowRefusal(
                f"the cascade cannot register {name!r}: {grid} carries "
                f"{sorted(masks)} of the boundary-mask triple "
                f"{list(REGIONAL_BOUNDARY_MASK_NAMES)}.  A cull with an "
                f"incomplete triple cannot identify its own rings, and a row "
                f"declaring a zone it cannot measure would admit an anchor "
                f"nobody checked"
            )
        import numpy as np

        zone_width = int(np.max(np.asarray(masks["bdyMaskCell"])))
        digest = regional_boundary_mask_digest(masks)
    # THE VERTICAL STRUCTURE COMES FROM THE INIT, and it is worth saying why.
    # A static file is a GEOGRAPHY file: it carries the horizontal mesh and
    # the surface fields and, on a culled pair, no ``nVertLevels`` at all.
    # Reading levels off it produced a row declaring ZERO levels, which the
    # bind refused as ``module N_LEVELS=55, registry=0`` -- a refusal that is
    # correct and whose cause was here.  The init is the file that carries the
    # column, so the row's column count is measured off the file the run will
    # actually read its state from.
    with Dataset(str(init)) as dataset:
        init_dimensions = {
            key: int(value.size) for key, value in dataset.dimensions.items()
        }

    def dimension(name: str, fallback: int) -> int:
        for source in (init_dimensions, dimensions):
            if name in source:
                return int(source[name])
        return fallback

    levels = dimension("nVertLevels", 0)
    if levels <= 0:
        raise CascadeRowRefusal(
            f"the cascade cannot register {name!r}: neither {init} nor {grid} "
            f"declares nVertLevels, so the row would claim a column height it "
            f"never measured and the bind would refuse it against the frozen "
            f"module's own 55"
        )
    interfaces = dimension("nVertLevelsP1", levels + 1)
    soil = dimension("nSoilLevels", 4)
    return CascadeRow(
        name=name,
        parent_row=parent_row,
        n_cells=int(dimensions["nCells"]),
        n_edges=int(dimensions["nEdges"]),
        n_levels=levels,
        n_interfaces=interfaces,
        n_soil_levels=soil,
        nominal_dx_m=float(nominal_dx_m),
        dt_seconds=float(dt_seconds),
        grid=str(grid),
        grid_bytes=grid.stat().st_size,
        grid_sha256=_sha256(grid),
        static=str(static),
        static_bytes=static.stat().st_size,
        static_sha256=_sha256(static),
        boundary_zone_width=zone_width,
        bdy_mask_sha256=digest,
        lbc_source=str(lbc_source),
        cull_receipt=str(cull_receipt),
        cycle_index=int(cycle_index),
        slot_id=str(slot_id),
        cull_pad_scale=float(cull_pad_scale),
    )


def write_rows(path: Path, rows: Sequence[CascadeRow]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "gpuwm-hex.cascade-rows/v1",
                "rows": [row.as_dict() for row in rows],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def load_rows(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Rows the cascade registered, from ``path`` or the environment."""

    target = path if path is not None else os.environ.get(CASCADE_ROWS_ENVIRONMENT)
    if not target:
        return []
    location = Path(target)
    if not location.is_file():
        raise CascadeRowRefusal(
            f"{CASCADE_ROWS_ENVIRONMENT} names {location}, which is not a "
            f"file.  A cascade that cannot read the rows it registered would "
            f"fall back to the shipped registry and run the wrong mesh"
        )
    document = json.loads(location.read_text(encoding="utf-8"))
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise CascadeRowRefusal(
            f"{location} carries no 'rows' list; it is not a cascade-row "
            f"document"
        )
    return rows


def apply_rows(
    registry: Mapping[str, Any], binding_type: Any, path: Path | str | None = None
) -> Mapping[str, Any]:
    """Return ``registry`` with this cycle's culls added.

    Returns the mapping UNCHANGED when the cascade registered nothing, which
    is every ordinary run.
    """

    rows = load_rows(path)
    if not rows:
        return registry
    patched = dict(registry)
    for raw in rows:
        row = CascadeRow(**{key: raw[key] for key in CascadeRow.__slots__})
        if row.name in registry:
            raise CascadeRowRefusal(
                f"the cascade tried to register {row.name!r}, which is "
                f"already a shipped registry row.  A cascade row may never "
                f"shadow one: the shipped row pins bytes a person reviewed "
                f"and this one would replace them silently"
            )
        if row.parent_row not in registry:
            raise CascadeRowRefusal(
                f"the cascade tried to register {row.name!r} as a cull of "
                f"{row.parent_row!r}, which is not a registered mesh row.  "
                f"Lineage stops at something a person registered: a cull of "
                f"an unregistered parent is how a shape nobody ever looked at "
                f"becomes a forecast.  Registered rows: {sorted(registry)}"
            )
        for role, file_path, want_bytes, want_sha in (
            ("grid", Path(row.grid), row.grid_bytes, row.grid_sha256),
            ("static", Path(row.static), row.static_bytes, row.static_sha256),
        ):
            if not file_path.is_file():
                raise CascadeRowRefusal(
                    f"the cascade row {row.name!r} names a {role} at "
                    f"{file_path} that is not a file"
                )
            size = file_path.stat().st_size
            if size != want_bytes:
                raise CascadeRowRefusal(
                    f"the cascade row {row.name!r} declares a {role} of "
                    f"{want_bytes} bytes and {file_path} is {size}.  The row "
                    f"was written from the cull receipt; a mismatch means the "
                    f"file moved under it"
                )
            measured = _sha256(file_path)
            if measured != want_sha:
                raise CascadeRowRefusal(
                    f"the cascade row {row.name!r} declares a {role} SHA-256 "
                    f"{want_sha[:16]}... and {file_path} digests to "
                    f"{measured[:16]}..."
                )
        if row.boundary_zone_width and not row.lbc_source:
            raise CascadeRowRefusal(
                f"the cascade row {row.name!r} declares a "
                f"{row.boundary_zone_width}-ring boundary zone and no "
                f"lbc_source, so nothing is registered to force it and a run "
                f"would integrate an unforced boundary inward"
            )
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
            boundary_zone_width=row.boundary_zone_width,
            bdy_mask_sha256=row.bdy_mask_sha256,
            lbc_source=row.lbc_source,
            notes=row.notes(),
        )
    return MappingProxyType(patched)


__all__ = [
    "CASCADE_ROWS_ENVIRONMENT",
    "CASCADE_ROW_MARKER",
    "CascadeRow",
    "CascadeRowRefusal",
    "apply_rows",
    "describe_cull",
    "load_rows",
    "write_rows",
]
