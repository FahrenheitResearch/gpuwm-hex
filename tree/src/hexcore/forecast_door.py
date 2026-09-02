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

WHY ADMISSION IS THE POINT OF THIS DOOR.  The driver's floor and this door
answer from ONE surface, ``hexcore.device_admission``: the same measured
affine row, the same headroom, the same :func:`required_free_bytes` sum.
Before that surface existed the driver held a separate linear floor scaled
from an asserted 24 GiB constant, which on the published 40,962-cell mesh
admitted at about 6.0 GiB free while the measured peak was 9,948 MiB -- a
card between those numbers passed the floor, spent minutes loading a mesh
and compiling kernels, and then died inside a CuPy allocation part-way
through a run.  That is the concrete breakage this gate prevents.  When the
door admits on a card's OWN measured row it forwards the same requirement to
the driver (``--required-free-bytes``), so the two gates cannot disagree.

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

from .device_admission import (
    DEFAULT_HEADROOM_BYTES,
    FOOTPRINT_MODEL,
    REFERENCE_CARD,
    CardProfile,
    FootprintModel,
    ShapedFootprintModel,
    card_profile_from_attributes,
    model_for_card,
    required_free_bytes,
)
from .physics_backend_admission import (
    DEFAULT_BACKEND,
    registered_backend_names,
    resolve_backend,
)
from .errors import ConfigurationRefusal, MpasPortError

__all__ = [
    "AdmissionVerdict",
    "CardProfile",
    "DEFAULT_HEADROOM_BYTES",
    "FOOTPRINT_MODEL",
    "FootprintModel",
    "ShapedFootprintModel",
    "read_card_profile",
    "request_configuration",
    "resolve_admission_model",
    "ForecastDoorRefusal",
    "ForecastRequest",
    "MeshRow",
    "PROJECT_ROOT",
    "RECEIPT_SCHEMA",
    "add_forecast_arguments",
    "admission_verdict",
    "admit_timestep",
    "admit_architecture",
    "admit_device",
    "build_receipt",
    "checkout_root",
    "load_registry",
    "measure_device_memory",
    "read_device_compute",
    "render_command",
    "require_admitted",
    "resolve_repo",
    "resolve_request",
    "run_forecast",
    "seam_pin_problem",
]

MIB = 1024**2

#: ``DEFAULT_HEADROOM_BYTES``, ``FOOTPRINT_MODEL`` and ``FootprintModel``
#: live in :mod:`hexcore.device_admission` -- the one admission surface
#: the driver's floor also reads -- and are re-exported here unchanged for
#: every existing caller.

#: The receipt this door writes beside the history files.
RECEIPT_SCHEMA = "gpuwm-hex.forecast-run/v1"

#: The receipt the driver writes into the same directory.  Read back and
#: embedded whole, so the door's receipt never restates a driver number.
DRIVER_RECEIPT_NAME = "cuda-v841-forecast-receipt.json"

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
    their own executing modules by SHA-256, and a module cannot verify a
    copy of itself that was never shipped.  A wheel-only forecast would
    therefore be a door that opens onto a missing floor, so the absence is
    NAMED here instead of failing as an ImportError three steps later.

    This is the gpuwm-HEX checkout and it is a separate requirement from the
    gpuwm one below; do not fold the two reasons together again.  The gpuwm
    seam pin used to be part of this sentence because one of its sixteen
    paths reached no wheel at all; at 2.5.8 all sixteen resolve from an
    install and that clause is gone from here.
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
            "modules by SHA-256 and a module cannot verify a copy of itself "
            "that was never shipped.  Obtain the gpuwm-hex repository and "
            "pass --repo <checkout>/tree, or run the door from inside it.  "
            "`gpuwm-hex doctor` does NOT report this gap -- measured on a pure "
            "wheel install it prints 'Every required check passed' and "
            "exits 0 while this door is unreachable, so this refusal is "
            "the only surface that names it.  Doctor does report the "
            "separate gpuwm one under 'gpuwm git checkout (the forecast "
            "lane only)'."
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
    #: The row's declared nominal spacing, in metres.  Carried because the
    #: convection ruling (ruling, 2026-08-26: off below 3 km) is answerable
    #: from the row alone -- no card, no file -- exactly as the timestep
    #: anchor is, and preflight's answer and the run's must not disagree.
    #: ``MeshBinding`` requires it, so :func:`load_registry` always fills it;
    #: the ``None`` default exists for constructed doubles, and a row that
    #: offers no spacing is recorded as ``source: "unknown-spacing"`` rather
    #: than having one guessed for it.
    nominal_dx_m: float | None = None


def load_registry(repo: Path) -> dict[str, MeshRow]:
    """The registered meshes, read out of ``tools/mpas_mesh_binding.py``.

    Read, never restated.  A second copy of the cell counts in this file
    would be a second thing to keep true, and the number it would drift on
    is the one every admission decision divides by.
    """

    module = _load_module("mpas_mesh_binding", Path(repo) / "tools" / "mpas_mesh_binding.py")
    return {
        name: MeshRow(
            name=name,
            cells=int(row.n_cells),
            dt_seconds=float(row.dt_seconds),
            nominal_dx_m=float(row.nominal_dx_m),
        )
        for name, row in module.MESH_BINDINGS.items()
    }


# ---------------------------------------------------------------------------
# the fitted footprint, and the decision it feeds
# ---------------------------------------------------------------------------
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
    model: ShapedFootprintModel | None = None

    def as_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "admitted": self.admitted,
            "mesh": self.mesh,
            "cells": self.cells,
            "predicted_peak_bytes": self.predicted_bytes,
            "predicted_peak_mib": self.predicted_bytes / MIB,
            # The margin, under the name every earlier receipt used.  It is
            # no longer a flat 512 MiB: it is the model's own two named,
            # measured components, and ``margin_terms`` below says what each
            # one absorbs.
            "headroom_bytes": self.headroom_bytes,
            "margin_bytes": self.headroom_bytes,
            "required_free_bytes": self.required_bytes,
            "measured_free_bytes": self.free_bytes,
            "measured_total_bytes": self.total_bytes,
            "shortfall_bytes": self.shortfall_bytes,
            "largest_fitted_cells": self.fitted_cells,
            "fitted_registered_meshes": list(self.alternatives),
            "model_provenance": self.model_provenance,
            "measured_at_decision_time": True,
        }
        if self.model is not None:
            record["footprint_model"] = self.model.as_dict()
            record["margin_terms"] = self.model.margin_terms()
            record["tiled_workspace_bytes"] = self.model.tiled_breakdown(self.cells)
            record["card"] = self.model.card.as_dict()
            record["configuration"] = self.model.configuration
            record["row_is_measured"] = self.model.measured
        return record


def admission_verdict(
    *,
    mesh: str,
    cells: int,
    free_bytes: int,
    total_bytes: int,
    headroom_bytes: int | None,
    registry: Mapping[str, MeshRow],
    model: ShapedFootprintModel | None = None,
) -> AdmissionVerdict:
    """Decide whether this card, as it is right now, holds this mesh."""

    model = model or FOOTPRINT_MODEL
    margin = model.margin_bytes() if headroom_bytes is None else int(headroom_bytes)
    predicted = int(round(model.predict_bytes(cells)))
    # THE one admission sum, shared with the driver's floor: a byte of
    # divergence here is a card that passes one gate and dies on the other.
    required = required_free_bytes(cells, model, margin)
    admitted = int(free_bytes) >= required
    fitted_cells = model.max_cells(int(free_bytes), margin)
    headroom_bytes = margin
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
        model=model,
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
        core = (
            verdict.model.core_bytes
            if verdict.model is not None
            else FOOTPRINT_MODEL.core_bytes
        )
        remedy = (
            "no registered mesh fits this card at this moment: this card's "
            f"own core is {_mib(core)} before a single cell is allocated, and "
            f"this card has {_mib(verdict.free_bytes)} free.  Free device "
            "memory (close other CUDA processes and any desktop compositor "
            "holding the card) or run on a larger device."
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
        f"which is what this gate exists to prevent.  {remedy}  The margin "
        "held back is this card's own, and it is not padding: "
        f"{_mib(verdict.headroom_bytes)} = the RRTMG shortwave workspace "
        "(the largest block in the footprint that does not scale with the "
        "mesh, measured to move the pool high-water by 1,707.2 MiB when it "
        "stops being servable from the free list) plus 11.2 MiB of "
        f"instrument convention.  The row is {verdict.model_provenance}.  If "
        "this card's own ledger has been run, supply its row with "
        f"--device-fixed-mib and --device-bytes-per-cell.  {LEDGER_PROBE} is "
        "the instrument that measures it."
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
            'the CUDA-13 CuPy: pip install "gpuwm-hex[gpu]".  Every GPU door '
            "here refuses a CUDA runtime below 13000, so the cu12 wheel is "
            "not an alternative -- it imports, probes clean, and is then "
            "refused by name at launch.  `gpuwm-hex doctor` reads your "
            "driver's CUDA major and says whether this box can open the lane "
            "at all."
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


def read_device_compute() -> tuple[int, int]:
    """The card's compute capability, read from the driver right now."""

    for variable in ("GPUWM_HEX_NO_LOCAL_GPU", "GPUWM_NO_LOCAL_GPU"):
        if os.environ.get(variable, "") not in ("", "0"):
            raise _refuse(
                f"{variable} is set, so this box has declared that no GPU work "
                "happens here and the door will not open a device to read its "
                "architecture."
            )
    import cupy

    properties = cupy.cuda.runtime.getDeviceProperties(0)
    return (int(properties["major"]), int(properties["minor"]))


def admit_architecture() -> dict[str, Any]:
    """Answer the architecture half of "will this run?" for this card.

    The run's own gate is ``require_cuda`` — per-architecture admission below
    the proven contract floor — but it lives past the driver's CUDA import,
    which preflight deliberately never performs.  That is how an sm_86 card
    once heard ``preflight_passed`` from a door whose real run then refused it
    by architecture (evidence/small-card-3080-20260824, rung 4).  This check
    asks the same admission question from a device-properties read alone, so
    preflight's answer and the run's answer can no longer disagree about the
    architecture.  Module-level lookup of :func:`read_device_compute` is the
    test seam.
    """

    from .cuda_backend.arch_admission import (
        PROVEN_COMPUTE,
        admitted_architecture,
        below_floor_refusal,
    )

    try:
        compute = read_device_compute()
    except ForecastDoorRefusal:
        raise
    except Exception as error:  # pragma: no cover - depends on the box
        raise _refuse(
            f"the CUDA driver refused a device-properties query ({error}), so "
            "the architecture half of admission cannot be answered on this "
            "box."
        ) from None
    record: dict[str, Any] = {
        "compute_capability": f"{compute[0]}.{compute[1]}",
        "sm": f"sm_{compute[0]}{compute[1]}",
    }
    if tuple(compute) >= PROVEN_COMPUTE:
        record.update(admitted=True, basis="at or above the proven contract floor")
        return record
    anchor = admitted_architecture(compute)
    if anchor is not None:
        record.update(
            admitted=True,
            basis=(
                f"per-architecture anchor of {anchor.admitted_on} "
                f"({anchor.contract_receipt})"
            ),
        )
        return record
    raise _refuse(below_floor_refusal(compute, PROVEN_COMPUTE))


def admit_timestep(
    mesh: str,
    dt_seconds: float,
    *,
    nominal_dx_m: float | None = None,
    convection: str = "auto",
    pbl_cadence: str = "auto",
) -> dict[str, Any]:
    """Answer the timestep half of "will this run?" from the row alone.

    The run's own gate is ``V841MpasColumnPhysicsConfig.validate``, which
    admits ``config_dt`` from the earned-anchor registry
    (:mod:`hexcore.dt_admission`) -- but that lives past the driver's CUDA
    import and past the mesh bind, so a row with an unanchored timestep used
    to reach a card before anything said so.  MEASURED (2026-08-26, the proving RTX 5090
    RTX 5090): 18,820 MiB reserved and 285 s spent before the refusal.

    This needs no card and no file: a registered row carries its declared
    timestep, and whether that timestep is anchored is a table lookup.  It
    is the same shape as :func:`admit_architecture`, and for the same
    reason -- preflight's answer and the run's answer must not disagree.
    """

    from . import convection_admission, dt_admission
    from . import pbl_cadence as pbl_cadence_module

    # An anchor certifies a CONFIGURATION at a timestep, and the project's
    # 2026-08-26 ruling made the cumulus selection part of that
    # configuration.  Answered from the row alone: the row carries its
    # declared nominal spacing, and the ruling is a comparison against one
    # threshold.  A row with no spacing to offer keeps the historical
    # Grell-Freitas answer rather than guessing at one.
    if nominal_dx_m is None:
        decision = {
            "scheme": convection_admission.SCHEME_GRELL_FREITAS,
            "constructor_scheme": "gf",
            "source": "unknown-spacing",
        }
    else:
        decision = convection_admission.convection_decision(
            nominal_dx_m=nominal_dx_m, requested=convection
        )
    cumulus_scheme = decision["constructor_scheme"]

    # The surface/PBL cadence is part of the same configuration, answerable
    # from the row alone for the same reason: the row declares the timestep
    # and the cadence is either welded to it or held at a stated number.
    pbl_decision = pbl_cadence_module.pbl_cadence_decision(
        dt_seconds=dt_seconds, requested=pbl_cadence
    )
    surface_pbl_seconds = float(pbl_decision["surface_pbl_seconds"])

    anchor = dt_admission.admitted_timestep(
        dt_seconds, cumulus_scheme, surface_pbl_seconds
    )
    if anchor is None:
        raise _refuse(
            f"--mesh {mesh} declares dt={float(dt_seconds):g} s and selects "
            f"{convection_admission.label(decision['scheme'])} "
            f"({decision['source']}) with {pbl_decision['label']} "
            f"({pbl_decision['source']}).  "
            + dt_admission.unanchored_refusal(
                dt_seconds, cumulus_scheme, surface_pbl_seconds
            )
        )
    return {
        "dt_seconds": float(dt_seconds),
        "admitted": True,
        "basis": f"timestep anchor of {anchor.admitted_on}",
        "schedule_receipt": anchor.schedule_receipt,
        "integration_anchor": anchor.integration_anchor,
        "cumulus_scheme": anchor.cumulus_scheme,
        "surface_pbl_seconds": anchor.surface_pbl_seconds,
        "physics_health": anchor.physics_health,
        "convection": decision,
        "pbl_cadence": pbl_decision,
    }


def admit_device(
    *,
    mesh: str,
    cells: int,
    headroom_bytes: int | None,
    registry: Mapping[str, MeshRow],
    model: ShapedFootprintModel | None = None,
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
    convection: str
    pbl_cadence: str
    local_timestep: bool
    local_timestep_rates: tuple[int, ...]
    local_timestep_buffer_rings: int
    #: ``None`` means "the model's own margin" -- the default.
    headroom_bytes: int | None
    #: ``None`` means "read the card and select its row at decision time",
    #: which is the default and the fix for ledger #366.  A ``_SuppliedRow``
    #: is the user's ``--device-fixed-mib``/``--device-bytes-per-cell`` pair,
    #: still waiting for the card.  A ``ShapedFootprintModel`` is a row
    #: already resolved against a card.
    model: Any
    stop_on_refusal: bool
    preflight: bool
    #: The lateral-boundary series a limited-area grid is driven by.  ``None``
    #: on a global grid; the driver refuses either mismatch by name.
    lbc_dir: Path | None = None
    #: The column-physics backend ROW this run selects
    #: (``hexcore.physics_backend_admission``).  The default is the frozen
    #: lane; a provider's row runs its own pinned column batch and may
    #: declare extra prognostic scalars and a point source.
    physics_backend: str = "wsm6_column"
    #: A point-source table the selected row's seam releases from.  ``None``
    #: is no release; a row that carries no point source refuses one.
    source_table: Path | None = None
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


def _resolve_model(arguments: argparse.Namespace) -> ShapedFootprintModel | None:
    """The user's own measured row, or ``None`` to read the card instead.

    ``None`` is the DEFAULT and it is the fix for ledger #366.  The retired
    door returned the 170 SM card's row here whenever the user did not type
    two numbers, so a 10 GiB desktop was priced with a 32 GiB card's fixed
    term and refused ``x1.40962`` and ``v15.150.38857`` -- two meshes it had
    been measured running with 2,244 MiB to spare.  The card is now read at
    decision time by :func:`resolve_admission_model`.
    """

    fixed = getattr(arguments, "device_fixed_mib", None)
    slope = getattr(arguments, "device_bytes_per_cell", None)
    if fixed is None and slope is None:
        return None
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
    return _SuppliedRow(core_bytes=float(fixed) * MIB, bytes_per_cell=float(slope))


@dataclass(frozen=True)
class _SuppliedRow:
    """``--device-fixed-mib`` / ``--device-bytes-per-cell``, before the card.

    Held rather than turned into a model immediately because the model's
    tiled-workspace terms are arithmetic on the CARD, and the card is not
    read until the admission decision.
    """

    core_bytes: float
    bytes_per_cell: float

    def with_card(self, card: CardProfile, configuration: str) -> ShapedFootprintModel:
        return ShapedFootprintModel(
            core_bytes=self.core_bytes,
            bytes_per_cell=self.bytes_per_cell,
            card=card,
            configuration=configuration,
            measured=True,
            provenance=(
                "supplied on the command line as this card's measured row; "
                "the tiled physics workspaces are charged on top of it from "
                f"this card's own {card.multiprocessors} multiprocessors, "
                "because those terms are arithmetic on the card and are not "
                "part of any measured core"
            ),
        )


def read_card_profile() -> CardProfile:
    """The card's shape, read from the driver right now.

    Two numbers -- multiprocessor count and maximum resident threads per
    multiprocessor -- and they are what every card-scaled term in the
    footprint model is priced from: the CUDA per-context local-memory
    backing store and the RRTMG column-chunk width.  Read here, never
    carried between decisions and never inferred from a card's name.
    """

    for variable in ("GPUWM_HEX_NO_LOCAL_GPU", "GPUWM_NO_LOCAL_GPU"):
        if os.environ.get(variable, "") not in ("", "0"):
            raise _refuse(
                f"{variable} is set, so this box has declared that no GPU work "
                "happens here and the door will not open a device to read its "
                "shape.  Unset it to run a forecast on this machine."
            )
    try:
        import cupy

        properties = cupy.cuda.runtime.getDeviceProperties(0)
    except Exception as error:  # pragma: no cover - depends on the box
        raise _refuse(
            f"the CUDA driver refused a device-properties query ({error}), so "
            "this card's multiprocessor count cannot be read and no footprint "
            "row can be selected for it.  The footprint's largest fixed term "
            "is the per-context local-memory backing store, which CUDA prices "
            "at the card's resident-thread capacity, so a row chosen without "
            "the SM count is another card's row."
        ) from None
    # The runtime returns the device name as BYTES.  str() on bytes yields
    # "b'NVIDIA GeForce RTX 3080'", and that string goes straight into the
    # run receipt and the admission refusal -- the same defect class as a
    # refusal that prints a sentinel's repr instead of something a reader can
    # act on.  Decode it, and keep working if a driver ever returns str.
    raw_name = properties.get("name", b"")
    if isinstance(raw_name, (bytes, bytearray)):
        raw_name = raw_name.decode("utf-8", "replace")
    return card_profile_from_attributes(
        str(raw_name or "").strip() or "this device",
        {
            "MultiProcessorCount": properties["multiProcessorCount"],
            "MaxThreadsPerMultiProcessor": properties.get(
                "maxThreadsPerMultiProcessor", 1536
            ),
        },
    )


def request_configuration(request: "ForecastRequest") -> str:
    """Which device stack this run builds -- the model's second key.

    A limited-area run builds a padded atmosphere, two levels of
    lateral-boundary state and 22 more kernels, and measures 85 %
    mesh-independent at cull cell counts.  The retired affine row had one
    configuration and over-predicted every measured cull by 1.2-1.9 GiB.
    """

    return "limited-area" if request.lbc_dir is not None else "global"


def _with_resolved_model(request: "ForecastRequest") -> "ForecastRequest":
    """The request with its footprint row pinned to the card that is here.

    Done ONCE, before the admission decision, so the verdict, the receipt and
    the ``--required-free-bytes`` the driver's own floor is handed all come
    from the same object.  Two resolutions could disagree if the card changed
    between them, and a card that passes one gate and dies on the other is
    the breakage this whole surface exists to prevent.
    """

    from dataclasses import replace

    return replace(request, model=resolve_admission_model(request))


def resolve_admission_model(request: "ForecastRequest") -> ShapedFootprintModel:
    """The footprint row this request is decided on.

    The user's supplied row if there is one, otherwise the row for the card
    the driver reports right now.  Module-level lookup of
    :func:`read_card_profile` is the seam a test substitutes to exercise the
    decision on a card this box does not have.
    """

    configuration = request_configuration(request)
    card = read_card_profile()
    supplied = request.model
    if isinstance(supplied, _SuppliedRow):
        return supplied.with_card(card, configuration)
    if isinstance(supplied, ShapedFootprintModel):
        return supplied
    return model_for_card(card, configuration)


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


def resolve_physics_backend(arguments: argparse.Namespace):
    """The backend row a request selects, or a refusal naming what exists."""

    name = getattr(arguments, "physics_backend", None) or DEFAULT_BACKEND
    try:
        return resolve_backend(str(name))
    except ConfigurationRefusal as error:
        raise _refuse(
            f"--physics-backend {name}: {error}"
        ) from error


def backend_pin_problem(row, checkout: Path) -> str | None:
    """The seam pin the ROW is bound by, checked at the door.

    The frozen row is pinned by this port's engine manifest
    (:func:`seam_pin_problem`).  A provider's row binds a different column
    batch through its own manifest, so its adapter module publishes
    ``pin_problem(checkout)`` and answers for itself; a provider row whose
    adapter publishes none is refused here rather than checked against a
    pin it was never bound by.
    """

    if row.name == DEFAULT_BACKEND:
        return seam_pin_problem(checkout)
    try:
        module = row.load_adapter()
    except ConfigurationRefusal as error:
        return str(error)
    probe = getattr(module, "pin_problem", None)
    if probe is None:
        return (
            f"--physics-backend {row.name} names adapter module "
            f"{row.adapter_module}, which publishes no pin_problem(checkout); "
            "the door cannot say which bytes this row would run and refuses "
            "to run unpinned physics"
        )
    return probe(checkout)


def seam_pin_problem(checkout: Path) -> str | None:
    """The seam pin, checked at the door, as a refusal that NAMES A VERSION.

    THE BREAKAGE THIS PREVENTS, measured on a real install on 2026-08-27
    (``evidence/userwalk-20260827/``): a user whose pip resolved gpuwm 2.5.7
    met this::

        gpuwm/core/microphysics.py does not match the proven manifest
        (expected a127585c..., found 4df5b7a3...) - the seam's executed
        source moved; re-prove before running.  That is not one of this
        program's named refusals ...

    Two digests, no version, an instruction addressed to a developer of this
    port rather than a user of it, and the door's own wrapper telling the
    reader its message should not be trusted.  Under the refusal law a
    refusal must name the breakage AND its remedy; a manifest holds digests
    and the user has no way to invert one into a version.  So the door
    inverts it here: which engine is on disk, which engine this port pins,
    and the two commands that close the gap.

    The driver's own sixteen-file check is NOT retired by this and must not
    be -- it is the wall, it runs against the checkout's git state as well as
    its bytes, and it is inside the frozen proof harness.  This is the sign
    on the wall, placed where the user meets it first.
    """

    from . import engine_pin

    inspection = engine_pin.inspect_seam(checkout)
    if not inspection.moved and not inspection.absent:
        return None

    found = engine_pin.checkout_version(checkout)
    if found is None:
        found = engine_pin.version_from_moved(inspection.moved)
    wanted = engine_pin.wanted_version()
    names = ", ".join(inspection.moved or inspection.absent)
    what = "have moved" if inspection.moved else "are missing"
    identity = (
        f"is gpuwm {found}" if found is not None
        else "declares no version this door could read"
    )
    return (
        f"--gpuwm-checkout {checkout} {identity}, and this port's physics "
        f"seam is pinned to gpuwm {wanted}.  "
        f"{len(inspection.moved or inspection.absent)} of "
        f"{len(engine_pin.seam_manifest())} pinned files {what}: {names}.  "
        "The seam is the whole physics of this model -- the port owns no "
        "parameterization arithmetic -- so running against moved bytes would "
        "be running physics nobody proved.  Point --gpuwm-checkout at a "
        f"checkout of v{wanted}:\n"
        f"{engine_pin.remedy(found)}"
    )


def resolve_request(
    arguments: argparse.Namespace,
    *,
    registry: Mapping[str, MeshRow] | None = None,
) -> ForecastRequest:
    """Every check that can be made before an expensive byte is read."""
    backend_row = resolve_physics_backend(arguments)

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
    # Before the schedule is built from the row's timestep, ask whether that
    # timestep is one the frozen lane may execute at all.  Card-free,
    # file-free, and ahead of every expensive check.
    #
    # In PREFLIGHT the refusal is collected, not raised.  THE BREAKAGE THIS
    # PREVENTS (measured 2026-08-26, the proving RTX 5090): asked whether the
    # 224,210-cell row fits the card, the door exited on the timestep and
    # never printed the memory verdict at all -- so the answer to "will this
    # run?" was half an answer, which is the one thing preflight promises not
    # to be.  The two blockers are independent and a user fixing one wants to
    # know about the other in the same pass.
    convection = getattr(arguments, "convection", "auto")
    pbl_cadence = getattr(arguments, "pbl_cadence", "auto")
    if collected is None:
        admit_timestep(
            mesh, row.dt_seconds,
            nominal_dx_m=row.nominal_dx_m, convection=convection,
            pbl_cadence=pbl_cadence,
        )
    else:
        try:
            admit_timestep(
                mesh, row.dt_seconds,
                nominal_dx_m=row.nominal_dx_m, convection=convection,
                pbl_cadence=pbl_cadence,
            )
        except ForecastDoorRefusal as error:
            collected.append(str(error))
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

    # None means "the model's own margin", which is now the DEFAULT: the
    # margin is two named, measured terms priced from the card, not a flat
    # constant every card shares.  A user who names a number still gets it.
    headroom_raw = getattr(arguments, "headroom_mib", None)
    headroom_bytes: int | None
    if headroom_raw is None:
        headroom_bytes = None
    else:
        headroom_mib = float(headroom_raw)
        if headroom_mib < 0.0:
            raise _refuse("--headroom-mib must not be negative")
        headroom_bytes = int(round(headroom_mib * MIB))
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
            "GIT CHECKOUT at the pinned commit.  The seam pin itself is "
            "satisfied by an install at the pinned engine -- all sixteen "
            "pinned paths resolve from site-packages there -- but the run's "
            "proof harness records the checkout's HEAD, tree and dirty paths "
            "into every receipt, and an install has no commit to name the "
            "executed bytes by."
        )
    else:
        checkout = Path(_require(
            checkout_argument,
            "--gpuwm-checkout",
            "the run's proof harness verifies the seam against a gpuwm GIT "
            "WORKING TREE and records its HEAD, tree and dirty paths into "
            "every receipt, so the executed bytes can be named by commit; an "
            "installed distribution has no commit and site-packages is not a "
            "git tree.  The forecast lane therefore needs a gpuwm checkout at "
            "the pinned commit.  This is NOT the old reason -- until engine "
            "2.5.7 one of the sixteen pinned paths reached no wheel at all, "
            "and at 2.5.8 all sixteen resolve from an install")
        ).expanduser().absolute()
        if not checkout.is_dir():
            problem = (
                f"--gpuwm-checkout {checkout} is not a directory.  The "
                "forecast lane needs a gpuwm GIT CHECKOUT at the pinned "
                "commit: the run verifies the checkout's git state and the "
                "sixteen seam digests before CUDA is touched, and writes "
                "HEAD, tree and dirty paths into the receipt so the executed "
                "bytes can be named by commit.  An installed distribution "
                "carries the bytes and no commit, so it cannot stand in for "
                "the checkout here."
            )
            if collected is None:
                raise _refuse(problem)
            collected.append(problem)
        else:
            problem = backend_pin_problem(backend_row, checkout)
            if problem is not None:
                if collected is None:
                    raise _refuse(problem)
                collected.append(problem)


    # The point-source table.  A row that carries no point source refuses
    # one by name: the frozen batch would accept the table and never read
    # it, which is configuration the run does not have.
    source_argument = getattr(arguments, "source_table", None)
    source_table: Path | None = None
    if source_argument is not None:
        if not backend_row.appended_scalar_names:
            problem = (
                f"--source-table {source_argument} was given, but "
                f"--physics-backend {backend_row.name} carries no point "
                "source and no appended scalars to release into; the table "
                "would be accepted and never read.  A point source selects "
                "a row that declares one: "
                f"{[name for name in registered_backend_names() if resolve_backend(name).appended_scalar_names]}"
            )
            if collected is None:
                raise _refuse(problem)
            collected.append(problem)
        source_table = _require_file(
            Path(source_argument), "--source-table",
            "the point-source table file (one waypoint per line) the "
            "treatment run releases from",
            collected,
        )

    # The lateral-boundary series.  The door checks only that the directory
    # exists and holds files: WHETHER this grid needs one is decided by the
    # grid's own bdyMask triple, in the driver, where the grid is read -- one
    # decision, one source.
    lbc_argument = getattr(arguments, "lbc_dir", None)
    lbc_dir: Path | None = None
    if lbc_argument is not None:
        lbc_dir = Path(lbc_argument).expanduser().absolute()
        if not lbc_dir.is_dir():
            problem = (
                f"--lbc-dir {lbc_dir} is not a directory.  It must be the "
                "--out-dir rw_mpas_lbc wrote its boundary files into, and it "
                "must hold lbc.*.nc: a limited-area domain integrated with no "
                "boundary series empties from its outer ring inward."
            )
            if collected is None:
                raise _refuse(problem)
            collected.append(problem)
        elif not sorted(lbc_dir.glob("lbc.*.nc")):
            problem = (
                f"--lbc-dir {lbc_dir} holds no lbc.*.nc.  rw_mpas_lbc names "
                "its output lbc.<valid-time>.nc, one file per boundary time; "
                "a directory with none of them drives nothing."
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
        convection=convection,
        pbl_cadence=pbl_cadence,
        local_timestep=local_timestep,
        local_timestep_rates=rates,
        local_timestep_buffer_rings=rings,
        headroom_bytes=headroom_bytes,
        model=model,
        stop_on_refusal=bool(getattr(arguments, "stop_on_refusal", False)),
        preflight=preflight,
        lbc_dir=lbc_dir,
        physics_backend=backend_row.name,
        source_table=source_table,
        input_problems=tuple(collected or ()),
    )


# ---------------------------------------------------------------------------
# the driver invocation and the hand-off
# ---------------------------------------------------------------------------
def build_driver_argv(request: ForecastRequest) -> list[str]:
    """The argument vector handed to the engineering forecast driver.

    ``--required-free-bytes`` carries the door's own admission requirement --
    :func:`required_free_bytes` over the request's resolved model and
    headroom -- into the driver's floor, so when a card's own measured row
    replaces the shipped one at the door, the driver admits on the same
    number instead of refusing on the default model's larger fixed term.
    One number, computed once, enforced twice.
    """

    argv = [
        "--grid", str(request.grid),
        "--static", str(request.static),
        "--init", str(request.init),
        "--init-source", request.init_source,
        "--hours", repr(request.hours),
        "--history-every-minutes", str(request.history_every_minutes),
        "--arwen-checkout", str(request.gpuwm_checkout),
        "--horiz-mixing", request.horiz_mixing,
        "--convection", request.convection,
        "--pbl-cadence", request.pbl_cadence,
        "--required-free-bytes",
        str(required_free_bytes(request.cells, request.model,
                                request.headroom_bytes)),
    ]
    if request.lbc_dir is not None:
        argv += ["--lbc-dir", str(request.lbc_dir)]
    if request.physics_backend != DEFAULT_BACKEND:
        argv += ["--physics-backend", request.physics_backend]
    if request.source_table is not None:
        argv += ["--source-table", str(request.source_table)]
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
        "physics": {
            "backend": request.physics_backend,
            "source_table": (
                None if request.source_table is None
                else str(request.source_table)
            ),
        },
        "schedule": {
            "hours": request.hours,
            "steps": request.steps,
            "history_every_minutes": request.history_every_minutes,
            "expected_frames": request.capture_count,
            "start_time": request.start_time,
        },
        "configuration": {
            "horiz_mixing": request.horiz_mixing,
            "convection": request.convection,
            "pbl_cadence": request.pbl_cadence,
            "local_timestep": request.local_timestep,
            "local_timestep_rates": list(request.local_timestep_rates),
            "local_timestep_buffer_rings": request.local_timestep_buffer_rings,
            "stop_on_refusal": request.stop_on_refusal,
        },
        "admission": admission.as_dict() if admission is not None else None,
        # ``None`` when the card could not be read at all, which is the one
        # case where the door has no row to report rather than the wrong one.
        "footprint_model": (
            request.model.as_dict()
            if isinstance(request.model, ShapedFootprintModel)
            else None
        ),
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
            convection=request.convection,
            pbl_cadence=request.pbl_cadence,
        )
    except binding.MeshBindingError as error:
        raise _refuse(f"the mesh bind refused: {error}{_BIND_REMEDY}") from None


def _print_admission(request: ForecastRequest, admission: AdmissionVerdict) -> None:
    model = admission.model
    card = (
        f"{model.card.multiprocessors} SM"
        if model is not None
        else "unknown card"
    )
    row = (
        ("measured" if model.measured else "DERIVED")
        + f" {model.configuration}"
        if model is not None
        else "default"
    )
    print(
        "ADMISSION mesh={mesh} cells={cells:,} card={card} row={row} "
        "predicted={predicted} margin={margin} free={free} of {total} "
        "-> {verdict}".format(
            mesh=request.mesh,
            cells=request.cells,
            card=card,
            row=row,
            predicted=_mib(admission.predicted_bytes),
            margin=_mib(admission.headroom_bytes),
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
        request = _with_resolved_model(request)
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

    try:
        architecture = admit_architecture()
        print(
            f"ARCHITECTURE sm={architecture['sm']} -> admitted "
            f"({architecture['basis']})",
            flush=True,
        )
    except ForecastDoorRefusal as error:
        problems.append(str(error))
        print(f"ARCHITECTURE REFUSED {error}", flush=True)

    driver_receipt: dict[str, Any] | None = None
    driver_argv: list[str] = []
    if not problems and driver is not None:
        driver_argv = build_driver_argv(request)
        try:
            rc = int(driver.main(driver_argv))
        except MpasPortError as error:
            # A refusal from this program names the breakage it prevents (the
            # gate law), so it is relayed verbatim: wrapping it in "stopped
            # with <ExceptionName>" buries the sentence a user needs behind a
            # Python type they cannot act on.
            problems.append(str(error))
            rc = 1
        except Exception as error:
            # An exception that is NOT one of this program's named refusals
            # reached the door.  Relaying it alone was a gate-law defect in
            # its own right and it had a measured victim: before 2026-08-26 a
            # limited-area mesh handed to this door reported
            # "RuntimeError: cellsOnEdge must contain two valid one-based
            # endpoints" -- an array-index sentence -- when what was actually
            # true is that the driver had no lateral-boundary route at all.
            # Say both: what stopped, and that its silence about the reason
            # is a defect rather than the user's error.
            problems.append(
                f"the driver's preflight stopped with "
                f"{type(error).__name__}: {error}.  That is not one of this "
                f"program's named refusals, so it does not say which "
                f"configuration it is refusing or why; treat it as a defect "
                f"in the driver rather than a statement about this run."
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
    request = _with_resolved_model(request)
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

    # The architecture half of admission, answered before a CUDA context or
    # a kernel compile is spent on a card the run would refuse by name.
    architecture = admit_architecture()
    print(
        f"ARCHITECTURE sm={architecture['sm']} -> admitted "
        f"({architecture['basis']})",
        flush=True,
    )

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
        # The CAUSE survives: a transactional driver raises its own summary
        # ("composite step aborted without publication") over a chained
        # exception, and a refusal that carries only the summary is a wall
        # with no sign.  The whole chain is written beside the run.
        import traceback

        trace_path = request.out / "forecast-traceback.log"
        cause_note = ""
        try:
            request.out.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(traceback.format_exc(), encoding="utf-8")
            cause = error.__cause__ or error.__context__
            if cause is not None:
                cause_note = f"  Underlying cause: {type(cause).__name__}: {cause}"
            cause_note += f"  Full traceback: {trace_path}."
        except OSError:
            pass
        raise _refuse(
            f"the forecast driver stopped with {type(error).__name__}: {error}"
            f"{cause_note}"
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
        "--lbc-dir", type=Path, default=None, metavar="DIR",
        help="lateral-boundary files (lbc.*.nc from `rw_mpas_lbc`) for a "
             "LIMITED-AREA grid. THE BREAKAGE IT PREVENTS: a grid cut with "
             "`rw_mpas_mesh --cull-parent` carries a seven-ring boundary "
             "zone whose outermost cells hold no atmosphere of their own; "
             "integrated with nothing driving them the domain empties from "
             "the edge inward. Required when --grid carries the bdyMask "
             "triple, refused when it does not")
    parser.add_argument(
        "--physics-backend", default=DEFAULT_BACKEND, metavar="ROW",
        help="the column-physics backend row this run selects "
             "(hexcore.physics_backend_admission). The default is the "
             "frozen lane. A provider's row runs its own pinned column "
             "batch and may declare extra prognostic scalars and a point "
             "source; an unregistered name is refused naming the rows "
             "this installation carries")
    parser.add_argument(
        "--source-table", type=Path, default=None, metavar="FILE",
        help="a point-source table the selected row's seam releases from "
             "(one waypoint per line); recorded in the receipt and pinned "
             "by its bytes into the run's identity. Refused on a row that "
             "carries no point source, because the table would be "
             "accepted and never read")
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
        help="gpuwm GIT checkout at the pinned commit; the run's proof "
             "harness records its HEAD, tree and dirty paths into the "
             "receipt, and an install has no commit")
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
        "--convection", choices=("auto", "off", "gf"), default="auto",
        help="cumulus selection. The default APPLIES DREW'S 2026-08-26 "
             "RULING -- convection is switched off below 3 km -- decided "
             "from the bound mesh's own finest spacing with no flag passed. "
             "'off' and 'gf' are explicit A/B arms and record themselves as "
             "explicit (default: auto)")
    parser.add_argument(
        "--pbl-cadence", default="auto", metavar="{auto,SECONDS}",
        help="surface/PBL cadence (config_bldt_seconds). The default 'auto' "
             "is the PROVEN CONFIGURATION: welded to config_dt, so the "
             "surface layer, the land-surface model and the PBL run every "
             "model step, as the native x4 v8.4.1 reference ran. An explicit "
             "number of seconds HOLDS the cadence while config_dt moves -- "
             "an A/B instrument for whether a forcing scales with call count "
             "rather than elapsed time. It records itself as explicit and "
             "earns its own timestep anchor (default: auto)")
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
        "--headroom-mib", type=float, default=None, metavar="MIB",
        help="override the margin the admission decision holds back from the "
             "card.  The default is the model's own margin, priced from this "
             "card: the RRTMG shortwave workspace (1,745.6 MiB on a 170 SM "
             "part, 872.8 MiB on a 68 or 70 SM part) plus 11.2 MiB of "
             "instrument convention.  The retired flat 512 MiB is still "
             "computable at device_admission.RETIRED_FLAT_HEADROOM_BYTES; it "
             "named no breakage and failed by 96 MiB on a real run")
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
