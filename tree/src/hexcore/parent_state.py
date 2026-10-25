"""A kept hex run's state at one of its later frames, as the analysis file the remaps read.

THE GAP THIS CLOSES.  A hex region cut from a kept hex parent (``gpuwm-hex regional --parent-*``, the WOOF "keep this
run usable as a start" route) starts at any of the parent's stored frames, not only at its start.  The two remaps that
build the region (``hexcore.analysis_remap`` for its start, ``hexcore.nest_boundary`` for its boundaries) read the
parent's state at the region's start as an *analysis*: an initial-conditions file with the parent's geometry, vertical
grid and reference state.  A forecast history frame is not one (it carries no reconstruction coefficients, no vertical
grid and no base state), and the parent's init is the state at the parent's start, hours earlier.

What this does, and nothing more: it rehydrates the frame on the parent's own grid with the native cold loader (the same
``nest_boundary.load_history`` the boundaries use), writes its prognostic state (dry density, modified potential
temperature, normal and vertical velocity, all six water species, and the region's number species --
carried from the frame or diagnosed, :mod:`hexcore.handover_numbers`) into a copy of the parent's init by dimension name
(``analysis_remap.write_fields``), carries every other time-varying field the frame and the init both hold with the same
shape (the land and sea state, snow, ice, number concentrations), and stamps the frame's valid time on all three MPAS
clocks (``analysis_remap.stamp_analysis_time``).  Geometry, vertical grid and reference state stay the init's own: the
frame is checked to be on the init's mesh (indexToCellID and indexToEdgeID) before anything is written.

No numerics of its own beyond the number diagnosis: the state is the frame's, moved into the init's layout.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

SCHEMA = "gpuwm-hex.parent-analysis/v1"

#: Never copied from the frame: the clocks (stamped from the frame's own xtime) and the mesh's index maps (checked
#: equal, and the init's are the analysis geometry's).
NOT_CARRIED = frozenset({"xtime", "Time", "initial_time", "indexToCellID", "indexToEdgeID", "indexToVertexID"})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def carry_time_state(target, frame, written: set[str]) -> list[str]:
    """Copy every time-varying variable the frame and the target both hold, with the same dimensions, shape and type,
    except the ones already written and NOT_CARRIED.  Returns the names copied."""
    carried = []
    for name, var in frame.variables.items():
        if name in written or name in NOT_CARRIED or name not in target.variables:
            continue
        into = target[name]
        if "Time" not in var.dimensions or var.dimensions != into.dimensions:
            continue
        if var.shape[1:] != into.shape[1:] or var.dtype != into.dtype or var.shape[0] < 1:
            continue
        into[0] = var[0]
        carried.append(name)
    return sorted(carried)


def analysis_at(parent_grid: Path, parent_static: Path, parent_init: Path, frame: Path, out: Path, *,
                numbers: tuple[str, ...] = (), scheme: int | None = None) -> dict:
    """Write the parent's state at `frame` as an analysis file `out` (a copy of `parent_init`), and return its receipt.

    ``numbers`` are the number species the region carries (:func:`hexcore.handover_numbers.region_number_species`):
    each is taken from the frame when the frame publishes it, else diagnosed from its mass with ``scheme``'s
    relations (None: left at the cold-start +0), and written into the analysis even though the parent's init holds
    no such variable -- the init never does, which is how the numbers used to be dropped here."""
    from netCDF4 import Dataset

    from . import analysis_remap as transfer
    from . import handover_numbers
    from . import nest_boundary

    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    numbers = tuple(numbers)
    published = handover_numbers.frame_number_species(frame)
    species = transfer.SPECIES + tuple(name for name in numbers if name in published)
    source = transfer.load(parent_grid, parent_static, parent_init, transfer.SPECIES)
    state = nest_boundary.load_history(frame, parent_init, source, species)
    number_receipt = None
    if numbers:
        species, number_receipt = handover_numbers.complete_state_numbers(
            state, species, numbers, scheme=scheme, xland=handover_numbers.read_xland(parent_init))
    values = transfer.native_fields(state.state, state.saved_diagnostics, source.vertical, species)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    shutil.copyfile(parent_init, partial)
    with Dataset(partial, "r+") as d, Dataset(frame) as f:
        transfer.write_fields(d, values)
        carried = carry_time_state(d, f, set(values))
        valid = transfer.stamp_analysis_time(d, f)
        d.setncattr("hex_parent_frame_sha256", _sha256(frame))
        d.setncattr("hex_parent_init_sha256", _sha256(parent_init))
    partial.replace(out)
    receipt = {"schema": SCHEMA, "valid_time": valid, "frame": str(frame), "init": str(parent_init),
               "output": str(out), "written": sorted(values), "carried": carried, "numbers": number_receipt,
               "output_sha256": _sha256(out)}
    out.with_name(out.name + ".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


__all__ = ["NOT_CARRIED", "SCHEMA", "analysis_at", "carry_time_state"]
