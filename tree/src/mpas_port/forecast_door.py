"""The forecast front door: an init and a registered mesh in, history out.

``gpuwm-hex forecast`` is the user-reachable command over the same execution
path the measurement harness drives.  It does not reimplement the model: it
binds a REGISTERED mesh through ``tools/mpas_mesh_binding.py``, hands the
bound shape to ``tools/run_cuda_v841_forecast.py`` -- whose ``execute_forecast``
is the one integration loop this project has -- and owns everything a
measurement harness deliberately does not:

* fail-closed argument resolution, with every refusal naming the wrong
  result it prevents and the command that fixes it;
* device-memory ADMISSION before CUDA is touched, against memory measured at
  the moment of the decision;
* destination discipline (the run's scratch never lands inside its output);
* one run receipt that states what ran, what it claims, and the exact
  ``gpuwm-hex render`` command that consumes the output.

WHY ADMISSION IS THE POINT OF THIS DOOR.  The driver's own floor
(``MIN_FREE_DEVICE_BYTES``, scaled per mesh from the native 24 GiB row) is a
FLOOR, not a footprint: on the published 40,962-cell mesh it admits at about
6.0 GiB free while the measured peak is 9,948 MiB.  A card between those two
numbers passes the floor, spends minutes loading a mesh and compiling
kernels, and then dies inside a CuPy allocation part-way through a run.  That
is the concrete breakage this gate prevents, and it is why the door refuses
on the FITTED model rather than on the driver's floor.

WHY THE MODEL IS NOT A FROZEN BUDGET.  Free device memory is read from the
driver at the moment of the decision, never carried from a previous run or a
previous card, because the memory a desktop session is already holding is
part of the answer.  The fitted row is a measurement with a provenance
string, and a card whose own ledger has been run supplies its own row with
``--device-fixed-mib`` / ``--device-bytes-per-cell``.

This module is stdlib-only at import.  numpy, netCDF4 and cupy are pulled
inside the handlers that need them, so ``gpuwm-hex --help`` and every
argument refusal work on a box with no CUDA lane at all.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as _dt
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

from .errors import MpasPortError

__all__ = [
    "AdmissionVerdict",
    "DEFAULT_HEADROOM_BYTES",
    "FOOTPRINT_MODEL",
    "FootprintModel",
    "ForecastDoorRefusal",
    "ForecastRequest",
    "MeshRow",
    "PROJECT_ROOT",
    "RECEIPT_SCHEMA",
    "add_forecast_arguments",
    "admission_verdict",
    "admit_device",
    "build_receipt",
    "checkout_root",
    "load_registry",
    "measure_device_memory",
    "render_command",
    "require_admitted",
    "resolve_repo",
    "resolve_request",
    "run_forecast",
]

MIB = 1024**2

#: The receipt this door writes beside the history files.
RECEIPT_SCHEMA = "gpuwm-hex.forecast-run/v1"

#: The receipt the driver writes into the same directory.  Read back and
#: embedded whole, so the door's receipt never restates a driver number.
DRIVER_RECEIPT_NAME = "cuda-v841-forecast-receipt.json"

#: Headroom the decision holds back from a card, matching the capacity
#: tooling's own final-gate headroom so one number governs both.  A device
#: with exactly the predicted peak free is not a device the run fits on:
#: the driver, the display server and the allocator all want a margin.
DEFAULT_HEADROOM_BYTES = 512 * MIB


class ForecastDoorRefusal(MpasPortError):
    """A named forecast-door refusal: what breaks, then the remedy."""


def _refuse(message: str) -> "ForecastDoorRefusal":
    return ForecastDoorRefusal(message)


def _mib(value: float) -> str:
    return f"{value / MIB:,.1f} MiB"


# ---------------------------------------------------------------------------
# where the drivers live
# ---------------------------------------------------------------------------
def checkout_root(start: Path | None = None) -> Path | None:
    """The source checkout this package was imported from, or ``None``.

    ``parents[2]`` is the project root of a ``src/`` layout checkout; from an
    installed wheel it is whatever directory happens to sit above
    ``site-packages``, which is not a place drivers or mesh assets live.
    Probing for a file only a checkout has keeps the convenience for a
    checkout and produces a named refusal for an install.
    """

    root = (start or Path(__file__).resolve().parents[2])
    if (root / "tools" / "run_cuda_v841_full_physics_x4.py").is_file():
        return root
    return None


PROJECT_ROOT = checkout_root()

#: What a repo must carry for this door to open.  Named individually so the
#: refusal can say WHICH file is absent rather than "not a checkout".
_REQUIRED_DRIVERS = (
    Path("tools") / "run_cuda_v841_full_physics_x4.py",
    Path("tools") / "run_cuda_v841_forecast.py",
    Path("tools") / "mpas_mesh_binding.py",
)


def _checkout_from_cwd() -> Path | None:
    """The documented second gesture: run the door from inside a checkout.

    From an installed wheel the package's own ancestry proves nothing
    (``PROJECT_ROOT`` is ``None``), but the refusal below and quickstart
    2.6 both promise that standing inside a checkout is enough.  Keeping
    that promise means probing the working directory and its parents for
    the tree that carries the drivers: either the ``tree/`` directory
    itself or a repository root holding one.  The staleness risk of
    "whatever checkout you are standing in" is the risk the drivers'
    own SHA-256 self-verification exists to catch, so nothing here is
    silently trusted.
    """

    here = Path.cwd()
    for candidate in (here, *here.parents):
        if checkout_root(candidate) is not None:
            return candidate
        if checkout_root(candidate / "tree") is not None:
            return candidate / "tree"
    return None


def resolve_repo(explicit: Path | None) -> Path:
    """The gpuwm-hex checkout holding the drivers, or a named refusal.

    The wheel deliberately does not carry ``tools/``: those drivers verify
    their own executing modules by SHA-256 and pin a gpuwm seam by the
    digests of sixteen individual source files, one of which no wheel
    places in site-packages.  A wheel-only forecast would therefore be a
    door that opens onto a missing floor, so the absence is NAMED here
    instead of failing as an ImportError three steps later.
    """

    if explicit is not None:
        candidate = Path(explicit).expanduser().absolute()
    elif PROJECT_ROOT is not None:
        candidate = PROJECT_ROOT
    else:
        candidate = _checkout_from_cwd()
    if candidate is None:
        raise _refuse(
            "the forecast lane needs the gpuwm-hex SOURCE CHECKOUT and this "
            "is an installed wheel: the drivers live in tools/, which the "
            "wheel does not carry, because they verify their own executing "
            "modules by SHA-256 and pin the gpuwm physics seam by the digests "
            "of sixteen individual source files -- one of them a repository "
            "document no wheel places in site-packages.  Obtain the "
            "gpuwm-hex repository and pass --repo <checkout>/tree, or run "
            "the door from inside it.  `gpuwm-hex doctor` reports the same "
            "gap under 'gpuwm source checkout (the forecast lane only)'."
        )
    if not candidate.is_dir():
        raise _refuse(
            f"--repo {candidate} is not a directory; it must be the gpuwm-hex "
            "checkout's tree/ directory, the one holding src/ and tools/"
        )
    missing = [str(name) for name in _REQUIRED_DRIVERS if not (candidate / name).is_file()]
    if missing:
        raise _refuse(
            f"--repo {candidate} is not a gpuwm-hex checkout: it is missing "
            f"{', '.join(missing)}.  The forecast lane runs from the "
            "repository's tree/ directory; point --repo at it."
        )
    return candidate


def _load_module(name: str, path: Path):
    """Load one driver by path, the way the registered-mesh runner does."""

    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:  # pragma: no cover
        raise _refuse(f"{path} could not be loaded as a Python module")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# the mesh registry, read rather than restated
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MeshRow:
    """One registered mesh, reduced to what the door decides with."""

    name: str
    cells: int
    dt_seconds: float


def load_registry(repo: Path) -> dict[str, MeshRow]:
    """The registered meshes, read out of ``tools/mpas_mesh_binding.py``.

    Read, never restated.  A second copy of the cell counts in this file
    would be a second thing to keep true, and the number it would drift on
    is the one every admission decision divides by.
    """

    module = _load_module("mpas_mesh_binding", Path(repo) / "tools" / "mpas_mesh_binding.py")
    return {
        name: MeshRow(name=name, cells=int(row.n_cells), dt_seconds=float(row.dt_seconds))
        for name, row in module.MESH_BINDINGS.items()
    }


# ---------------------------------------------------------------------------
# the fitted footprint, and the decision it feeds
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FootprintModel:
    """An affine process-footprint row: fixed term plus a per-cell slope."""

    fixed_bytes: float
    bytes_per_cell: float
    provenance: str

    def predict_bytes(self, cells: int) -> float:
        if cells <= 0:
            raise ValueError("cells must be positive")
        return self.fixed_bytes + self.bytes_per_cell * cells

    def max_cells(self, budget_bytes: int, headroom_bytes: int) -> int:
        usable = float(budget_bytes) - float(headroom_bytes) - self.fixed_bytes
        if usable <= 0.0 or self.bytes_per_cell <= 0.0:
            return 0
        return int(usable // self.bytes_per_cell)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fixed_bytes": self.fixed_bytes,
            "fixed_mib": self.fixed_bytes / MIB,
            "bytes_per_cell": self.bytes_per_cell,
            "provenance": self.provenance,
        }


#: The measured row of record.  Both coefficients come from one session that
#: ran BOTH published meshes at BOTH engine pins on one card, so the fit is
#: two measured points rather than an extrapolation from one.  The card was a
#: 170-SM part; the FIXED term is a property of the card (it is dominated by
#: the CUDA local-memory backing store, which scales with resident warps),
#: and both smaller parts previously measured carried smaller fixed terms.
#: That direction matters for how this row is used: on a smaller card the
#: prediction is an OVER-estimate, so a refusal it produces is conservative
#: and the remedy is to measure this card, not to widen the gate.
FOOTPRINT_MODEL = FootprintModel(
    fixed_bytes=6296.5 * MIB,
    bytes_per_cell=93_474.0,
    provenance=(
        "measured 2026-08-24 on an RTX 5090 (170 SM) at engine pin 0d04db712, "
        "both published meshes in one session; "
        "evidence/gf-pin-move-measured-20260824/"
    ),
)

#: The instrument that produces a row for a DIFFERENT card.  Named in the
#: refusal, because "your card may be smaller" is only useful beside the
#: command that settles it.
LEDGER_PROBE = "tools/device_memory_ledger/hex_ledger_probe.py"


@dataclass(frozen=True)
class AdmissionVerdict:
    """One admission decision, with every number it was made from."""

    admitted: bool
    mesh: str
    cells: int
    predicted_bytes: int
    headroom_bytes: int
    required_bytes: int
    free_bytes: int
    total_bytes: int
    shortfall_bytes: int
    fitted_cells: int
    alternatives: tuple[str, ...]
    model_provenance: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "mesh": self.mesh,
            "cells": self.cells,
            "predicted_peak_bytes": self.predicted_bytes,
            "predicted_peak_mib": self.predicted_bytes / MIB,
            "headroom_bytes": self.headroom_bytes,
            "required_free_bytes": self.required_bytes,
            "measured_free_bytes": self.free_bytes,
            "measured_total_bytes": self.total_bytes,
            "shortfall_bytes": self.shortfall_bytes,
            "largest_fitted_cells": self.fitted_cells,
            "fitted_registered_meshes": list(self.alternatives),
            "model_provenance": self.model_provenance,
            "measured_at_decision_time": True,
        }


def admission_verdict(
    *,
    mesh: str,
    cells: int,
    free_bytes: int,
    total_bytes: int,
    headroom_bytes: int,
    registry: Mapping[str, MeshRow],
    model: FootprintModel | None = None,
) -> AdmissionVerdict:
    """Decide whether this card, as it is right now, holds this mesh."""

    model = model or FOOTPRINT_MODEL
    predicted = int(round(model.predict_bytes(cells)))
    required = predicted + int(headroom_bytes)
    admitted = int(free_bytes) >= required
    fitted_cells = model.max_cells(int(free_bytes), int(headroom_bytes))
    alternatives = tuple(
        sorted(
            row.name
            for row in registry.values()
            if row.name != mesh and row.cells <= fitted_cells
        )
    )
    return AdmissionVerdict(
        admitted=admitted,
        mesh=mesh,
        cells=int(cells),
        predicted_bytes=predicted,
        headroom_bytes=int(headroom_bytes),
        required_bytes=required,
        free_bytes=int(free_bytes),
        total_bytes=int(total_bytes),
        shortfall_bytes=0 if admitted else required - int(free_bytes),
        fitted_cells=fitted_cells,
        alternatives=alternatives,
        model_provenance=model.provenance,
    )


def require_admitted(verdict: AdmissionVerdict) -> None:
    """Refuse an unadmitted request, naming the fitted alternative."""

    if verdict.admitted:
        return
    if verdict.alternatives:
        remedy = (
            "This card fits the registered mesh(es) "
            f"{', '.join(verdict.alternatives)} at this moment "
            f"({verdict.fitted_cells:,} cells fit); re-run --mesh with one of "
            "them, or free the memory the shortfall names and re-run."
        )
    else:
        remedy = (
            "no registered mesh fits this card at this moment: the fixed term "
            f"alone is {_mib(FOOTPRINT_MODEL.fixed_bytes)} before a single "
            "cell is allocated, and this card has "
            f"{_mib(verdict.free_bytes)} free.  Free device memory (close "
            "other CUDA processes and any desktop compositor holding the "
            "card) or run on a larger device."
        )
    raise _refuse(
        f"device memory admission refused --mesh {verdict.mesh}: the fitted "
        f"footprint for {verdict.cells:,} cells is "
        f"{_mib(verdict.predicted_bytes)} and the decision holds back "
        f"{_mib(verdict.headroom_bytes)}, so it needs "
        f"{_mib(verdict.required_bytes)} free; this device reports "
        f"{_mib(verdict.free_bytes)} free of {_mib(verdict.total_bytes)}, "
        f"short by {_mib(verdict.shortfall_bytes)}.  Running anyway does not "
        "fail a check -- it loads the mesh, compiles the kernels, and then "
        "dies inside a CuPy allocation part-way through the integration, "
        f"which is what this gate exists to prevent.  {remedy}  The fitted "
        f"row is {verdict.model_provenance}; the fixed term is a property of "
        "the card and smaller parts measure smaller, so if this card's own "
        f"ledger has been run, supply its row with --device-fixed-mib and "
        f"--device-bytes-per-cell.  {LEDGER_PROBE} is the instrument that "
        "measures it."
    )


def measure_device_memory() -> tuple[int, int]:
    """``(free, total)`` device bytes, read from the driver right now.

    Never cached and never carried between decisions: the memory another
    process is holding is part of the answer, and it changes.
    """

    for variable in ("GPUWM_HEX_NO_LOCAL_GPU", "GPUWM_NO_LOCAL_GPU"):
        if os.environ.get(variable, "") not in ("", "0"):
            raise _refuse(
                f"{variable} is set, so this box has declared that no GPU work "
                "happens here and the door will not open a device to measure "
                "it.  Unset it to run a forecast on this machine, or run the "
                "forecast on the machine that owns the card."
            )
    try:
        import cupy
    except Exception as error:  # pragma: no cover - depends on the box
        raise _refuse(
            f"the CUDA lane is not importable ({error}), so free device memory "
            "cannot be measured and a forecast cannot be admitted.  Install "
            "the CuPy wheel matching the CUDA major your driver reports: "
            'pip install "gpuwm-hex[gpu-cu12]" or "gpuwm-hex[gpu-cu13]" -- '
            "exactly one.  `gpuwm-hex doctor` reports which is missing."
        ) from None
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            raise _refuse(
                "cupy imported but reports no CUDA device; there is nothing "
                "here to run a forecast on.  Check `nvidia-smi` and that "
                "CUDA_VISIBLE_DEVICES does not exclude every device."
            )
        free, total = cupy.cuda.runtime.memGetInfo()
    except ForecastDoorRefusal:
        raise
    except Exception as error:  # pragma: no cover - driver present but unusable
        raise _refuse(
            f"the CUDA driver refused a memory query ({error}); the device is "
            "present but unusable, so no admission decision can be made.  "
            "Check the driver and that no other process has the card in an "
            "exclusive compute mode."
        ) from None
    return int(free), int(total)


def admit_device(
    *,
    mesh: str,
    cells: int,
    headroom_bytes: int,
    registry: Mapping[str, MeshRow],
    model: FootprintModel | None = None,
) -> AdmissionVerdict:
    """Measure the card NOW, then decide.  Module-level lookup of
    :func:`measure_device_memory` is deliberate: it is the seam a test
    substitutes to exercise the decision at a free-memory value no card
    here has."""

    free, total = measure_device_memory()
    return admission_verdict(
        mesh=mesh,
        cells=cells,
        free_bytes=free,
        total_bytes=total,
        headroom_bytes=headroom_bytes,
        registry=registry,
        model=model,
    )


# ---------------------------------------------------------------------------
# the request
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ForecastRequest:
    """One resolved, admitted-shaped forecast request."""

    repo: Path
    mesh: str
    cells: int
    dt_seconds: float
    grid: Path
    static: Path
    init: Path
    init_source: str
    start_time: str | None
    hours: float
    history_every_minutes: int
    steps: int
    capture_count: int
    out: Path
    scratch: Path
    receipt: Path
    gpuwm_checkout: Path
    case_label: str | None
    horiz_mixing: str
    local_timestep: bool
    local_timestep_rates: tuple[int, ...]
    local_timestep_buffer_rings: int
    headroom_bytes: int
    model: FootprintModel
    stop_on_refusal: bool
    preflight: bool
    #: Input problems collected rather than raised, preflight only.  Always
    #: empty on a real run, because there the same conditions refuse.
    input_problems: tuple[str, ...] = ()

    @property
    def inputs_present(self) -> bool:
        return not self.input_problems


def default_scratch(out: Path) -> Path:
    """A scratch sibling of ``--out``: same parent, never inside the tree.

    The render door already rules that a scratch tree inside a delivered
    tree is a shipped defect; the forecast lane's kernel cache is the same
    hazard against a directory a user is about to hand to the renderer.
    """

    out = Path(out)
    return out.parent / (out.name + ".forecast-scratch")


def _require(value: Any, flag: str, breakage: str) -> Any:
    if value is None:
        raise _refuse(f"{flag} was not given and cannot be inferred: {breakage}")
    return value


def _require_file(
    path: Path, flag: str, remedy: str, collected: list[str] | None = None
) -> Path:
    """Resolve one input file.

    ``collected`` is the preflight seam.  A preflight answers "will this
    run?", and a user asking that has often not built every input yet; if a
    missing init stopped the answer, the card question -- the one no file
    fixes -- could not be asked until every file existed.  So in preflight
    the problem is RECORDED and the remaining checks still run.  In a real
    run ``collected`` is ``None`` and the same condition refuses.
    """

    resolved = Path(path).expanduser().absolute()
    problem: str | None = None
    if resolved.is_symlink():
        problem = (
            f"{flag} {resolved} is a symbolic link.  The drivers pin their "
            "inputs by byte count and SHA-256 and refuse a link, because the "
            "bytes a receipt names must be the bytes that were read."
        )
    elif not resolved.is_file():
        problem = f"{flag} names a missing file: {resolved}.  {remedy}"
    if problem is None:
        return resolved
    if collected is None:
        raise _refuse(problem)
    collected.append(problem)
    return resolved


def _parse_rates(raw: Any, flag: str) -> tuple[int, ...]:
    try:
        rates = tuple(int(piece) for piece in str(raw).split(",") if piece.strip())
    except ValueError:
        raise _refuse(
            f"{flag} must be comma-separated integers; got {raw!r}"
        ) from None
    if not rates or rates[0] != 1 or list(rates) != sorted(set(rates)):
        raise _refuse(
            f"{flag} must be strictly increasing and start at 1; got {raw!r}.  "
            "Rate 1 is the class that carries the finest columns, and without "
            "it every column would be advanced on a longer acoustic sub-step "
            "than its own edge length admits."
        )
    return rates


def _resolve_model(arguments: argparse.Namespace) -> FootprintModel:
    fixed = getattr(arguments, "device_fixed_mib", None)
    slope = getattr(arguments, "device_bytes_per_cell", None)
    if fixed is None and slope is None:
        return FOOTPRINT_MODEL
    if fixed is None or slope is None:
        raise _refuse(
            "--device-fixed-mib and --device-bytes-per-cell are one measured "
            "row and must be given together.  Half a row silently mixes this "
            "card's fixed term with another card's slope, which is a "
            "footprint nothing ever measured.  Run "
            f"{LEDGER_PROBE} on this card and pass both numbers it reports."
        )
    if float(fixed) <= 0.0 or float(slope) <= 0.0:
        raise _refuse(
            "--device-fixed-mib and --device-bytes-per-cell must both be "
            "positive; a non-positive coefficient is not a measurement of a "
            "footprint."
        )
    return FootprintModel(
        fixed_bytes=float(fixed) * MIB,
        bytes_per_cell=float(slope),
        provenance="supplied on the command line as this card's measured row",
    )


def _schedule(hours: float, history_every_minutes: int, dt_seconds: float,
              mesh: str) -> tuple[int, int]:
    """``(steps, capture_count)``, or a refusal naming the mesh's timestep.

    The timestep is the REGISTERED row's, not a module constant: the
    generated 15 km row declares 60 s where the published rows declare 120 s,
    and a schedule checked against one constant would admit a half-step run
    on the other.
    """

    total_seconds = float(hours) * 3600.0
    if total_seconds <= 0.0:
        raise _refuse(f"--hours {hours} must be positive")
    raw_steps = total_seconds / dt_seconds
    if abs(raw_steps - round(raw_steps)) > 1e-9:
        raise _refuse(
            f"--hours {hours} is not a whole number of steps on mesh {mesh}, "
            f"whose registered timestep is {dt_seconds:.0f} s: it is "
            f"{raw_steps:.4f} steps.  A partial step cannot be integrated, so "
            f"choose a length that divides by {dt_seconds:.0f} s (for example "
            f"{max(1, int(raw_steps)) * dt_seconds / 3600.0:g} hours)."
        )
    steps = int(round(raw_steps))
    history_seconds = int(history_every_minutes) * 60
    if history_seconds <= 0 or history_seconds % int(dt_seconds) != 0:
        raise _refuse(
            f"--history-every-minutes {history_every_minutes} is not a whole "
            f"number of {dt_seconds:.0f} s steps on mesh {mesh}; a history "
            "frame can only be written at a step boundary."
        )
    stride = history_seconds // int(dt_seconds)
    if steps % stride != 0:
        raise _refuse(
            f"--history-every-minutes {history_every_minutes} does not divide "
            f"the {hours} h run on mesh {mesh} ({steps} steps, stride "
            f"{stride}); the last frame would fall inside the run rather than "
            "at its end, so the output would silently not cover the length "
            "asked for."
        )
    return steps, len(range(0, steps + 1, stride))


def resolve_request(
    arguments: argparse.Namespace,
    *,
    registry: Mapping[str, MeshRow] | None = None,
) -> ForecastRequest:
    """Every check that can be made before an expensive byte is read."""

    repo = resolve_repo(getattr(arguments, "repo", None))
    if registry is None:
        registry = load_registry(repo)

    mesh = _require(
        getattr(arguments, "mesh", None),
        "--mesh",
        "the mesh registry pins each row's grid and static bytes, its "
        "dimensions and its Courant-admitted timestep, and the door will not "
        "guess which row the supplied files are. Registered meshes: "
        + ", ".join(sorted(registry)),
    )
    if mesh not in registry:
        raise _refuse(
            f"--mesh {mesh!r} is not a registered mesh.  Registered meshes are "
            f"{', '.join(sorted(registry))}.  A row pins its grid and static "
            "by byte count and SHA-256 and declares the timestep its real "
            "dcEdge admits; running an unregistered mesh would run an "
            "unproved shape at an unadmitted timestep.  Register the row in "
            "tools/mpas_mesh_binding.py before running it."
        )
    row = registry[mesh]
    preflight = bool(getattr(arguments, "preflight", False))
    collected: list[str] | None = [] if preflight else None

    # ORDER.  Everything a user can fix from the command line alone is
    # checked before anything that depends on the filesystem, because those
    # are properties of the REQUEST and no file changes them.  The order
    # this replaced put --init's existence first, which meant a user with a
    # mistyped --hours and a not-yet-built init was told about the init,
    # built it, and only then learned the schedule was wrong.
    init_source = _require(
        getattr(arguments, "init_source", None),
        "--init-source",
        "the run receipt records which meteorology produced the init, and "
        "the init file's own bytes cannot say where they came from.  A "
        "forecast whose provenance sentence is blank is a forecast nobody "
        "can attribute later; state it, e.g. --init-source \"GFS 2026-08-24 "
        "00Z\"",
    )

    hours = float(_require(
        getattr(arguments, "hours", None), "--hours",
        "the forecast length has no default; the door will not choose how "
        "much compute a run spends"))
    history_every_minutes = int(_require(
        getattr(arguments, "history_every_minutes", None),
        "--history-every-minutes",
        "the history cadence has no default; it decides how much of the run "
        "survives it, and a wrong guess is discovered only after the run"))
    steps, captures = _schedule(hours, history_every_minutes, row.dt_seconds, mesh)

    out = Path(_require(
        getattr(arguments, "out", None), "--out",
        "the history files and the run receipt need a destination, and the "
        "door will not write into the current directory by default")
    ).expanduser().absolute()
    if not out.parent.is_dir():
        raise _refuse(
            f"--out {out} cannot be created: its parent {out.parent} does not "
            "exist.  Create the parent first; the door refuses to build a "
            "deep path for an expensive run because a mistyped one then looks "
            "like a successful new directory."
        )
    if out.exists() and not preflight:
        raise _refuse(
            f"--out {out} exists.  A forecast writes its history frames and "
            "its receipt into a fresh directory so a second run cannot be "
            "read as the first one's continuation, or silently mix frames "
            "from two trajectories.  Give an unused path, or move the "
            "existing one aside."
        )

    scratch = Path(
        getattr(arguments, "scratch", None) or default_scratch(out)
    ).expanduser().absolute()
    if scratch == out or out in scratch.parents:
        raise _refuse(
            f"--scratch {scratch} is inside the output tree {out}.  The "
            "forecast's kernel cache is not output: leaving it inside the "
            "directory a user hands to `gpuwm-hex render` publishes "
            "half-finished temporaries as products.  Give --scratch a "
            "directory outside --out, or omit it for the sibling default."
        )
    if scratch.exists() and not preflight:
        raise _refuse(
            f"--scratch {scratch} exists.  The driver builds its kernel cache "
            "in a fresh directory so a stale cache from another engine pin "
            "cannot be loaded into this run.  Give an unused path, or remove "
            "the existing one."
        )

    horiz_mixing = getattr(arguments, "horiz_mixing", "2d_smagorinsky")
    local_timestep = bool(getattr(arguments, "local_timestep", False))
    rates = _parse_rates(
        getattr(arguments, "local_timestep_rates", "1,3"), "--local-timestep-rates"
    )
    rings = int(getattr(arguments, "local_timestep_buffer_rings", 1))
    if rings < 1:
        raise _refuse(
            "--local-timestep-buffer-rings must be at least 1; with no buffer "
            "ring a class boundary has no cells demoted to the finer rate and "
            "the coarse class reads values one sub-step stale."
        )

    headroom_mib = float(getattr(arguments, "headroom_mib", DEFAULT_HEADROOM_BYTES / MIB))
    if headroom_mib < 0.0:
        raise _refuse("--headroom-mib must not be negative")
    model = _resolve_model(arguments)

    # Filesystem-dependent from here down.
    grid = _require_file(
        _require(getattr(arguments, "grid", None), "--grid",
                 "the mesh grid file carries the topology the registry "
                 "cross-examines; it has no default outside a checkout"),
        "--grid",
        "Pass the registered grid file for this mesh; `gpuwm-hex mesh-check` "
        "prints the digests a row is pinned by.", collected)
    static = _require_file(
        _require(getattr(arguments, "static", None), "--static",
                 "the static file carries terrain, land use and the physical "
                 "dcEdge the timestep is admitted against"),
        "--static",
        "Pass the static file generated with this grid; a mismatched pair is "
        "refused by name at bind.", collected)
    init = _require_file(
        _require(getattr(arguments, "init", None), "--init",
                 "the initial condition is the state the forecast starts "
                 "from and there is no synthetic default"),
        "--init",
        "Build one with `gpuwm-hex init` (chapter 5 of the manual).", collected)

    checkout_argument = getattr(arguments, "gpuwm_checkout", None)
    if checkout_argument is None and preflight:
        # The card question does not depend on the seam checkout, so a
        # preflight answers it and records the gap instead of refusing.
        checkout = Path("(not given)")
        collected.append(  # type: ignore[union-attr]
            "--gpuwm-checkout was not given; the forecast lane needs a gpuwm "
            "SOURCE CHECKOUT at the pinned commit, because the physics seam "
            "is pinned by the SHA-256 of sixteen individual gpuwm source "
            "files and one of them is a repository document no wheel places "
            "in site-packages."
        )
    else:
        checkout = Path(_require(
            checkout_argument,
            "--gpuwm-checkout",
            "the physics seam is pinned by the SHA-256 of sixteen individual "
            "gpuwm source files, one of which no wheel places in "
            "site-packages, so an installed gpuwm satisfies pip and does not "
            "satisfy the pin.  The forecast lane needs a gpuwm SOURCE "
            "CHECKOUT at the pinned commit")
        ).expanduser().absolute()
        if not checkout.is_dir():
            problem = (
                f"--gpuwm-checkout {checkout} is not a directory.  The "
                "forecast lane needs a gpuwm SOURCE CHECKOUT at the pinned "
                "commit: the seam is pinned by the digests of sixteen "
                "individual source files, one of them a repository document "
                "no wheel installs, so the installed distribution cannot "
                "stand in for it.  The run verifies the checkout's git state "
                "and those digests before CUDA is touched and refuses a "
                "mismatch by name."
            )
            if collected is None:
                raise _refuse(problem)
            collected.append(problem)


    receipt = getattr(arguments, "receipt", None)
    receipt_path = (
        Path(receipt).expanduser().absolute() if receipt is not None
        else out / "forecast-receipt.json"
    )
    if receipt is not None and not receipt_path.parent.is_dir():
        raise _refuse(
            f"--receipt directory {receipt_path.parent} does not exist; create "
            "it before the run rather than losing the receipt at the end of one."
        )

    return ForecastRequest(
        repo=repo,
        mesh=mesh,
        cells=row.cells,
        dt_seconds=row.dt_seconds,
        grid=grid,
        static=static,
        init=init,
        init_source=str(init_source),
        start_time=getattr(arguments, "start_time", None),
        hours=hours,
        history_every_minutes=history_every_minutes,
        steps=steps,
        capture_count=captures,
        out=out,
        scratch=scratch,
        receipt=receipt_path,
        gpuwm_checkout=checkout,
        case_label=getattr(arguments, "case_label", None),
        horiz_mixing=horiz_mixing,
        local_timestep=local_timestep,
        local_timestep_rates=rates,
        local_timestep_buffer_rings=rings,
        headroom_bytes=int(round(headroom_mib * MIB)),
        model=model,
        stop_on_refusal=bool(getattr(arguments, "stop_on_refusal", False)),
        preflight=preflight,
        input_problems=tuple(collected or ()),
    )


# ---------------------------------------------------------------------------
# the driver invocation and the hand-off
# ---------------------------------------------------------------------------
def build_driver_argv(request: ForecastRequest) -> list[str]:
    """The argument vector handed to the engineering forecast driver."""

    argv = [
        "--grid", str(request.grid),
        "--static", str(request.static),
        "--init", str(request.init),
        "--init-source", request.init_source,
        "--hours", repr(request.hours),
        "--history-every-minutes", str(request.history_every_minutes),
        "--arwen-checkout", str(request.gpuwm_checkout),
        "--horiz-mixing", request.horiz_mixing,
    ]
    if request.start_time:
        argv += ["--start-time", request.start_time]
    if request.case_label:
        argv += ["--case-label", request.case_label]
    if request.local_timestep:
        argv += [
            "--local-timestep",
            "--local-timestep-rates",
            ",".join(str(rate) for rate in request.local_timestep_rates),
            "--local-timestep-buffer-rings",
            str(request.local_timestep_buffer_rings),
        ]
    if request.stop_on_refusal:
        argv.append("--stop-on-refusal")
    if request.preflight:
        argv.append("--preflight-only")
    else:
        argv += [
            "--cache-root", str(request.scratch),
            "--output", str(request.out),
        ]
    return argv


def render_command(request: ForecastRequest, history: Sequence[Path]) -> list[str] | None:
    """The exact ``gpuwm-hex render`` command that consumes this output.

    The forecast door's job ends at files the NEXT door can take.  Printing
    the command is what makes that true for a reader rather than only for a
    manual: the converter needs the run's own grid file and the run's start
    time to number lead hours, and both are known here and nowhere else.
    """

    if not history:
        return None
    command = ["gpuwm-hex", "render", "--history"]
    command += [str(path) for path in history]
    command += [
        "--mesh", str(request.grid),
        "--out", str(request.out / "png"),
    ]
    if request.start_time:
        command += ["--simulation-start", request.start_time]
    return command


def history_files(out: Path) -> list[Path]:
    """The history frames the driver wrote, in valid-time order."""

    return sorted(Path(out).glob("cuda-history.*.nc"))


def build_receipt(
    *,
    request: ForecastRequest,
    admission: AdmissionVerdict | None,
    bind_receipt: Mapping[str, Any] | None,
    driver_receipt: Mapping[str, Any] | None,
    history: Sequence[Path],
    driver_argv: Sequence[str],
    seconds: float,
    status: str,
) -> dict[str, Any]:
    """The door's own receipt: what ran, on what, and what comes next."""

    from . import __version__

    return {
        "schema": RECEIPT_SCHEMA,
        "tool": "gpuwm-hex forecast",
        "door_version": __version__,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "host": platform.node(),
        "status": status,
        "mesh": {
            "name": request.mesh,
            "cells": request.cells,
            "dt_seconds": request.dt_seconds,
            "grid": str(request.grid),
            "static": str(request.static),
        },
        "init": {"path": str(request.init), "source": request.init_source},
        "schedule": {
            "hours": request.hours,
            "steps": request.steps,
            "history_every_minutes": request.history_every_minutes,
            "expected_frames": request.capture_count,
            "start_time": request.start_time,
        },
        "configuration": {
            "horiz_mixing": request.horiz_mixing,
            "local_timestep": request.local_timestep,
            "local_timestep_rates": list(request.local_timestep_rates),
            "local_timestep_buffer_rings": request.local_timestep_buffer_rings,
            "stop_on_refusal": request.stop_on_refusal,
        },
        "admission": admission.as_dict() if admission is not None else None,
        "footprint_model": request.model.as_dict(),
        "mesh_binding": bind_receipt,
        "driver_receipt": driver_receipt,
        "driver_argv": list(driver_argv),
        "gpuwm_checkout": str(request.gpuwm_checkout),
        "repo": str(request.repo),
        "out": str(request.out),
        "scratch": str(request.scratch),
        "history": [str(path) for path in history],
        "render_command": render_command(request, history),
        "door_seconds": round(float(seconds), 3),
    }


# ---------------------------------------------------------------------------
# the handler
# ---------------------------------------------------------------------------
def _write_receipt(request: ForecastRequest, receipt: Mapping[str, Any]) -> Path | None:
    path = request.receipt
    if not path.parent.is_dir():
        return None
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _load_drivers(repo: Path):
    """The two driver modules, loaded from the checkout by path."""

    for entry in (str(repo / "tools"), str(repo / "src")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    binding = _load_module("mpas_mesh_binding", repo / "tools" / "mpas_mesh_binding.py")
    driver = _load_module(
        "v841_forecast", repo / "tools" / "run_cuda_v841_forecast.py"
    )
    return binding, driver


_BIND_REMEDY = (
    "  The registry pins each row's grid and static by byte count and "
    "SHA-256 and admits its timestep against the file's own dcEdge, so this "
    "is the supplied files disagreeing with the named row, not a transient "
    "failure.  `gpuwm-hex mesh-check --grid ... --static ...` prints the "
    "digests the files actually carry."
)


def _bind(binding, driver, request: ForecastRequest) -> dict[str, Any]:
    try:
        return binding.bind_mesh(
            driver.proof,
            request.mesh,
            grid=request.grid,
            static=request.static,
            forecast=driver,
        )
    except binding.MeshBindingError as error:
        raise _refuse(f"the mesh bind refused: {error}{_BIND_REMEDY}") from None


def _print_admission(request: ForecastRequest, admission: AdmissionVerdict) -> None:
    print(
        "ADMISSION mesh={mesh} cells={cells:,} predicted={predicted} "
        "headroom={headroom} free={free} of {total} -> {verdict}".format(
            mesh=request.mesh,
            cells=request.cells,
            predicted=_mib(admission.predicted_bytes),
            headroom=_mib(admission.headroom_bytes),
            free=_mib(admission.free_bytes),
            total=_mib(admission.total_bytes),
            verdict="admitted" if admission.admitted else "REFUSED",
        ),
        flush=True,
    )


def _run_preflight(request: ForecastRequest, registry: Mapping[str, MeshRow],
                   started: float) -> int:
    """Answer the whole question, not the first half of it.

    A preflight that stopped at the first blocker would send a user round the
    loop once per blocker, on a lane where one loop is a mesh bind and a
    device query.  So every check runs, every result is reported, and the
    EXIT CODE carries the verdict.  This is the one mode where an
    unadmitted request is reported rather than raised.
    """

    problems: list[str] = list(request.input_problems)
    for problem in request.input_problems:
        print(f"INPUT MISSING {problem}", flush=True)

    bind_receipt: dict[str, Any] | None = None
    if request.grid.is_file() and request.static.is_file():
        binding, driver = _load_drivers(request.repo)
        try:
            bind_receipt = _bind(binding, driver, request)
            print(
                f"BIND mesh={request.mesh} rebound={bind_receipt['rebound']} "
                f"dt={bind_receipt['timestep_admission']['requested_dt_seconds']} s",
                flush=True,
            )
        except ForecastDoorRefusal as error:
            problems.append(str(error))
            print(f"BIND REFUSED {error}", flush=True)
    else:
        driver = None
        print(
            "BIND NOT ATTEMPTED the mesh pair is not both present, and the "
            "bind is a cross-examination of those exact bytes",
            flush=True,
        )

    admission: AdmissionVerdict | None = None
    try:
        admission = admit_device(
            mesh=request.mesh, cells=request.cells,
            headroom_bytes=request.headroom_bytes, registry=registry,
            model=request.model,
        )
        _print_admission(request, admission)
        if not admission.admitted:
            try:
                require_admitted(admission)
            except ForecastDoorRefusal as error:
                problems.append(str(error))
    except ForecastDoorRefusal as error:
        problems.append(str(error))
        print(f"ADMISSION NOT MEASURED {error}", flush=True)

    driver_receipt: dict[str, Any] | None = None
    driver_argv: list[str] = []
    if not problems and driver is not None:
        driver_argv = build_driver_argv(request)
        try:
            rc = int(driver.main(driver_argv))
        except Exception as error:
            problems.append(
                f"the driver's preflight stopped with "
                f"{type(error).__name__}: {error}"
            )
            rc = 1
        if rc != 0:
            problems.append(f"the driver's preflight exited {rc}")

    status = "preflight_passed" if not problems else "preflight_refused"
    receipt = build_receipt(
        request=request, admission=admission, bind_receipt=bind_receipt,
        driver_receipt=driver_receipt, history=[], driver_argv=driver_argv,
        seconds=time.monotonic() - started, status=status,
    )
    receipt["preflight_problems"] = problems
    written = _write_receipt(request, receipt)
    if written is not None:
        print(f"RECEIPT {written}", flush=True)
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    for problem in problems:
        print(f"PREFLIGHT REFUSED {problem}", file=sys.stderr)
    print(
        f"PREFLIGHT mesh={request.mesh} problems={len(problems)} status={status}",
        flush=True,
    )
    return 1 if problems else 0


def run_forecast(arguments: argparse.Namespace) -> int:
    """The ``gpuwm-hex forecast`` handler.  Returns the process exit code."""

    started = time.monotonic()
    request = resolve_request(arguments)
    registry = load_registry(request.repo)
    if request.preflight:
        return _run_preflight(request, registry, started)

    # Admission first on a real run: it is the cheapest of the remaining
    # checks and the most consequential, and refusing here spends neither
    # the mesh hashing nor a CUDA context on a run that cannot start.
    admission = admit_device(
        mesh=request.mesh,
        cells=request.cells,
        headroom_bytes=request.headroom_bytes,
        registry=registry,
        model=request.model,
    )
    _print_admission(request, admission)
    if not admission.admitted:
        receipt = build_receipt(
            request=request, admission=admission, bind_receipt=None,
            driver_receipt=None, history=[], driver_argv=[],
            seconds=time.monotonic() - started, status="refused_by_admission",
        )
        written = _write_receipt(request, receipt)
        if written is not None:
            print(f"RECEIPT {written}", flush=True)
        require_admitted(admission)

    binding, driver = _load_drivers(request.repo)
    bind_receipt = _bind(binding, driver, request)
    driver_argv = build_driver_argv(request)
    # The driver's validate_destination requires BOTH roots absent and creates
    # them itself; a door that pre-creates them kills every admitted run with
    # FileExistsError immediately after admission (measured on the RTX 3080,
    # 2026-08-24 -- the door's success leg had never run on hardware).  The
    # door's own earlier refusals still catch a pre-existing --out/--scratch
    # by name; creation belongs to the driver alone.
    try:
        rc = int(driver.main(driver_argv))
    except MemoryError as error:
        raise _refuse(
            f"the driver's own device-memory floor refused this run: {error}  "
            "The door's admission passed and the driver's floor did not, "
            "which means the two disagree: report the pair of numbers, "
            "because the floor is a proof constant and the fitted model is a "
            "measurement, and only one of them can be wrong."
        ) from None
    except ForecastDoorRefusal:
        raise
    except Exception as error:
        raise _refuse(
            f"the forecast driver stopped with {type(error).__name__}: {error}"
            f"  The scratch tree is kept at {request.scratch} and any frames "
            f"already written are in {request.out}."
        ) from None

    history = history_files(request.out)
    driver_receipt: dict[str, Any] | None = None
    driver_receipt_path = request.out / DRIVER_RECEIPT_NAME
    if driver_receipt_path.is_file():
        try:
            driver_receipt = json.loads(
                driver_receipt_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            driver_receipt = {"unreadable": str(driver_receipt_path)}

    status = "passed" if rc == 0 else "driver_rc_nonzero"
    if driver_receipt is not None and driver_receipt.get("status"):
        status = str(driver_receipt["status"])
    receipt = build_receipt(
        request=request,
        admission=admission,
        bind_receipt=bind_receipt,
        driver_receipt=driver_receipt,
        history=history,
        driver_argv=driver_argv,
        seconds=time.monotonic() - started,
        status=status,
    )
    written = _write_receipt(request, receipt)
    if written is not None:
        print(f"RECEIPT {written}", flush=True)
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True, default=str))

    print(
        f"DOOR mesh={request.mesh} steps={request.steps} "
        f"frames={len(history)} status={status} out={request.out}",
        flush=True,
    )
    command = render_command(request, history)
    if command is not None:
        print("NEXT " + " ".join(command), flush=True)
    elif rc == 0:
        print(
            "DOOR the driver exited 0 but wrote no history frame; the "
            "receipt's driver_receipt block carries the reason",
            flush=True,
        )
        return 1
    return rc


# ---------------------------------------------------------------------------
# the argument surface
# ---------------------------------------------------------------------------
def add_forecast_arguments(parser: argparse.ArgumentParser) -> None:
    """Wire the ``forecast`` subcommand's arguments onto ``parser``.

    Nothing is ``required=True``.  argparse's own missing-argument error is a
    usage block, and this project's doors state the WRONG RESULT a missing
    argument would cause; the checks live in :func:`resolve_request` so every
    one of them can be exercised without a card.
    """

    parser.add_argument(
        "--mesh", default=None, metavar="NAME",
        help="registered mesh row (tools/mpas_mesh_binding.py)")
    parser.add_argument(
        "--grid", type=Path, default=None, metavar="FILE",
        help="the row's grid netCDF; pinned by byte count and SHA-256")
    parser.add_argument(
        "--static", type=Path, default=None, metavar="FILE",
        help="the row's static netCDF, generated with that grid")
    parser.add_argument(
        "--init", type=Path, default=None, metavar="FILE",
        help="initial conditions from `gpuwm-hex init`")
    parser.add_argument(
        "--init-source", default=None, metavar="TEXT",
        help="provenance sentence for the init, recorded in the receipt "
             '(e.g. "GFS 2026-08-24 00Z")')
    parser.add_argument(
        "--start-time", default=None, metavar="YYYY-MM-DD_HH:MM:SS",
        help="asserted against the init's config_start_time; the init is "
             "the authority")
    parser.add_argument(
        "--hours", type=float, default=None, metavar="H",
        help="forecast length; must be a whole number of the row's timesteps")
    parser.add_argument(
        "--history-every-minutes", type=int, default=None, metavar="M",
        help="history cadence; must divide the run into whole steps")
    parser.add_argument(
        "--out", type=Path, default=None, metavar="DIR",
        help="fresh directory for history frames and the run receipt")
    parser.add_argument(
        "--scratch", type=Path, default=None, metavar="DIR",
        help="kernel-cache dir OUTSIDE --out "
             "(default: sibling <out>.forecast-scratch)")
    parser.add_argument(
        "--receipt", type=Path, default=None, metavar="FILE",
        help="run receipt path (default: <out>/forecast-receipt.json)")
    parser.add_argument(
        "--gpuwm-checkout", type=Path, default=None, metavar="DIR",
        help="gpuwm SOURCE checkout at the pinned commit; the physics seam "
             "is pinned by sixteen file digests, one of which no wheel ships")
    parser.add_argument(
        "--repo", type=Path, default=None, metavar="DIR",
        help="gpuwm-hex checkout tree/ holding tools/ "
             "(default: this checkout, if the door was imported from one)")
    parser.add_argument(
        "--case-label", default=None, metavar="TEXT",
        help="label recorded in the receipt")
    parser.add_argument(
        "--horiz-mixing", choices=("2d_smagorinsky", "off"),
        default="2d_smagorinsky",
        help="horizontal mixing lane; the default is the native Registry "
             "deformation-based 2-D Smagorinsky. 'off' is the pre-mixing "
             "control lane, reported as the configuration native itself "
             "cannot integrate on convective cases (default: 2d_smagorinsky)")
    parser.add_argument(
        "--local-timestep", action="store_true",
        help="OPT-IN, default off: advance coarse columns on fewer, longer "
             "acoustic sub-steps. Native v8.4.1 has none, so this is a "
             "declared divergence and does not pay on the published "
             "variable-resolution mesh (0.988x measured)")
    parser.add_argument(
        "--local-timestep-rates", default="1,3", metavar="LADDER",
        help="acoustic rate ladder; strictly increasing, starts at 1 "
             "(default: 1,3)")
    parser.add_argument(
        "--local-timestep-buffer-rings", type=int, default=1, metavar="N",
        help="rings demoted to the finer rate at a class boundary (default: 1)")
    parser.add_argument(
        "--headroom-mib", type=float, default=DEFAULT_HEADROOM_BYTES / MIB,
        metavar="MIB",
        help="memory the admission decision holds back from the card "
             f"(default: {DEFAULT_HEADROOM_BYTES / MIB:g})")
    parser.add_argument(
        "--device-fixed-mib", type=float, default=None, metavar="MIB",
        help="this card's measured fixed footprint term; supply with "
             "--device-bytes-per-cell to replace the shipped row")
    parser.add_argument(
        "--device-bytes-per-cell", type=float, default=None, metavar="BYTES",
        help="this card's measured per-cell slope; supply with "
             "--device-fixed-mib")
    parser.add_argument(
        "--stop-on-refusal", action="store_true",
        help="when the model refuses to publish a step, stop and write the "
             "receipt for the frames already committed instead of aborting "
             "with none. No validation is relaxed")
    parser.add_argument(
        "--preflight", action="store_true",
        help="resolve, bind and admit without touching the integration")


def add_forecast_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "forecast",
        help="run the model on a registered mesh from an init "
             "(GPU; needs the source checkout)",
        description=(
            "Drive the v8.4.1 CUDA forecast on a registered mesh. The door "
            "binds the mesh against the registry's pinned bytes, admits the "
            "request against device memory measured at that moment, runs the "
            "integration, and writes history frames the render door consumes "
            "plus one run receipt."
        ),
    )
    add_forecast_arguments(parser)
    parser.set_defaults(handler=run_forecast)
