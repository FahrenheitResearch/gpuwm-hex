"""Lateral-boundary-condition files: the reader and the two-level pool.

The producer side of this seam is ``rw_mpas_lbc`` (and, for reference mints,
native ``init_atmosphere_model`` case 9): one CDF-5 file per boundary time,
each carrying FULL-MESH fields — ``lbc_u`` over every edge, the cell fields
over every cell — regardless of how few of those elements the consuming model
will read.  This module is the consumer's first half, a transcription of the
admission and timekeeping in ``mpas_atm_boundaries.F`` (MPAS v8.4.1):

* **admission** — the first admitted file is the LATEST file at-or-before
  the model time (``MPAS_STREAM_LATEST_BEFORE``); every subsequent one is the
  EARLIEST file STRICTLY after it (``MPAS_STREAM_EARLIEST_STRICTLY_AFTER``).
  A missing interval refuses by name: which rule, which model time, what the
  inventory actually holds.
* **the two-level pool** — level 2 holds the state read at the current
  interval's END; level 1 holds the TENDENCY ``(new - old) / dt`` computed in
  float32 exactly as the Fortran computes it (``dt`` formed in REAL(4), then
  reciprocal-multiplied).
* **time interpolation** — the state at model time ``t`` inside an interval
  is ``state(end) - (end - t) * tendency``: linear, anchored BACKWARD from
  the interval end, which is ``mpas_atm_get_bdy_state`` verbatim.

What this module deliberately does NOT do: derive the coupled fields
(``lbc_rho_zz = lbc_rho / zz``, edge-averaged ``lbc_rho_edge``, ``lbc_ru``,
``lbc_rtheta_m``) or touch the integration loop.  Those need mesh state and
live in :class:`hexcore.regional_v841.RegionalDrivingState`, which wraps
this pool and computes them at every admission exactly as
``mpas_atm_update_bdy_tend`` does; the pool stays generic over field names
so that wrapper carries derived fields beside the file fields without this
module growing a mesh dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .errors import MpasPortError

__all__ = [
    "LBC_REQUIRED_VARIABLES",
    "LbcAdmissionError",
    "LbcFile",
    "LbcFileError",
    "LbcInventory",
    "LbcPool",
    "read_lbc_file",
    "read_lbc_valid_time",
]

#: The v8.4.1 ``lbc`` stream under the ``lbcs`` package with the ``moist``
#: scalar group: variable name -> the dimension names its file slab carries.
#: Measured on the native case-9 reference files and pinned here; a file
#: missing one of these, or carrying it on other dimensions, is not an lbc
#: file this port understands.
LBC_REQUIRED_VARIABLES: Dict[str, Tuple[str, ...]] = {
    "lbc_qv": ("Time", "nCells", "nVertLevels"),
    "lbc_qc": ("Time", "nCells", "nVertLevels"),
    "lbc_qr": ("Time", "nCells", "nVertLevels"),
    "lbc_u": ("Time", "nEdges", "nVertLevels"),
    "lbc_w": ("Time", "nCells", "nVertLevelsP1"),
    "lbc_rho": ("Time", "nCells", "nVertLevels"),
    "lbc_theta": ("Time", "nCells", "nVertLevels"),
}

_XTIME_FORMAT = "%Y-%m-%d_%H:%M:%S"


class LbcFileError(MpasPortError):
    """An lbc file is missing something the stream contract promises."""


class LbcAdmissionError(MpasPortError):
    """No file in the inventory satisfies the admission rule being applied."""


def _parse_xtime(raw: object, path: Path) -> datetime:
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", "replace").strip()
    else:
        text = str(raw).strip()
    try:
        return datetime.strptime(text[:19], _XTIME_FORMAT)
    except ValueError as error:
        raise LbcFileError(
            f"{path} carries xtime {text!r}, which is not YYYY-MM-DD_HH:MM:SS; "
            "without a parseable valid time the file cannot be placed on the "
            "boundary timeline at all"
        ) from error


def _read_xtime(dataset, path: Path) -> datetime:
    if "xtime" not in dataset.variables:
        raise LbcFileError(
            f"{path} has no xtime variable; an lbc file with no valid time "
            "cannot be admitted against any model time"
        )
    slab = np.asarray(dataset.variables["xtime"][0])
    raw = slab.tobytes() if slab.dtype.kind in ("S", "U") else bytes(slab)
    return _parse_xtime(raw, path)


def read_lbc_valid_time(path: str | Path) -> datetime:
    """The file's valid time, from its own ``xtime`` — never the filename."""

    from netCDF4 import Dataset

    p = Path(path)
    with Dataset(p) as dataset:
        dataset.set_auto_maskandscale(False)
        return _read_xtime(dataset, p)


@dataclass(frozen=True)
class LbcFile:
    """One boundary time, fields squeezed of their record axis, float32."""

    path: Path
    valid_time: datetime
    fields: Mapping[str, np.ndarray]

    @property
    def n_cells(self) -> int:
        return int(self.fields["lbc_theta"].shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.fields["lbc_u"].shape[0])

    @property
    def n_vert_levels(self) -> int:
        return int(self.fields["lbc_theta"].shape[1])


def read_lbc_file(path: str | Path) -> LbcFile:
    """Read and validate one lbc file.

    Refusals name the missing piece: an absent variable, a variable on the
    wrong dimensions, or a non-float32 payload.  Each would otherwise surface
    as a shape error several frames inside the driver, long after the wrong
    file was accepted.
    """

    from netCDF4 import Dataset

    p = Path(path)
    with Dataset(p) as dataset:
        # Raw bytes, not the mask-and-scale view: the tendency arithmetic is
        # pinned to the reference's float32, so the values must be the file's.
        dataset.set_auto_maskandscale(False)
        valid_time = _read_xtime(dataset, p)
        fields: Dict[str, np.ndarray] = {}
        for name, want_dims in LBC_REQUIRED_VARIABLES.items():
            if name not in dataset.variables:
                raise LbcFileError(
                    f"{p} carries no {name}; the v8.4.1 lbc stream always "
                    f"writes it, so this is not a complete lbc file and "
                    f"admitting it would leave that field frozen at whatever "
                    f"the previous interval held"
                )
            variable = dataset.variables[name]
            got_dims = tuple(variable.dimensions)
            if got_dims != want_dims:
                raise LbcFileError(
                    f"{p} carries {name} on dimensions {got_dims}, not "
                    f"{want_dims}; reading it anyway would transpose the "
                    f"field into plausible-looking wrong numbers"
                )
            if variable.dtype != np.float32:
                raise LbcFileError(
                    f"{p} carries {name} as {variable.dtype}, not float32; "
                    f"the reference stream is single precision and a silent "
                    f"widening here would make every tendency differ from "
                    f"the reference arithmetic"
                )
            data = np.array(variable[0][:], dtype=np.float32)
            data.setflags(write=False)
            fields[name] = data
    return LbcFile(path=p, valid_time=valid_time, fields=fields)


class LbcInventory:
    """The boundary files available to a run, indexed by their own xtime.

    The inventory holds ``(valid_time, path)`` and applies the two stream-
    manager admission rules.  It reads only each file's ``xtime`` at
    construction; field payloads are read on admission.
    """

    def __init__(self, paths: Sequence[str | Path]):
        if not paths:
            raise LbcAdmissionError(
                "the lbc inventory is empty; a regional run with no boundary "
                "files has nothing to relax toward and would integrate on a "
                "frozen boundary"
            )
        stamped: list[Tuple[datetime, Path]] = []
        for path in paths:
            p = Path(path)
            stamped.append((read_lbc_valid_time(p), p))
        stamped.sort(key=lambda pair: pair[0])
        for (t1, p1), (t2, p2) in zip(stamped, stamped[1:]):
            if t1 == t2:
                raise LbcAdmissionError(
                    f"two lbc files carry the same valid time "
                    f"{t1.strftime(_XTIME_FORMAT)}: {p1} and {p2}; the "
                    f"admission rules cannot order them and picking one "
                    f"silently would make the run depend on directory order"
                )
        self._entries: Tuple[Tuple[datetime, Path], ...] = tuple(stamped)

    @property
    def valid_times(self) -> Tuple[datetime, ...]:
        return tuple(t for t, _ in self._entries)

    def _timeline(self) -> str:
        return ", ".join(t.strftime(_XTIME_FORMAT) for t, _ in self._entries)

    def latest_before(self, when: datetime) -> Path:
        """``MPAS_STREAM_LATEST_BEFORE``: the latest file at-or-before ``when``."""

        candidates = [(t, p) for t, p in self._entries if t <= when]
        if not candidates:
            raise LbcAdmissionError(
                f"no lbc file is valid at or before "
                f"{when.strftime(_XTIME_FORMAT)} (LATEST_BEFORE); the "
                f"inventory holds [{self._timeline()}].  This is the native "
                f"'Could not read from lbc_in stream on or before the "
                f"current date' failure, refused before the run starts "
                f"instead of inside it"
            )
        return candidates[-1][1]

    def earliest_strictly_after(self, when: datetime) -> Path:
        """``MPAS_STREAM_EARLIEST_STRICTLY_AFTER``: the earliest file after ``when``."""

        candidates = [(t, p) for t, p in self._entries if t > when]
        if not candidates:
            raise LbcAdmissionError(
                f"no lbc file is valid strictly after "
                f"{when.strftime(_XTIME_FORMAT)} (EARLIEST_STRICTLY_AFTER); "
                f"the inventory holds [{self._timeline()}].  The interval "
                f"the model is about to integrate has no far end, and "
                f"integrating past the last boundary file would freeze the "
                f"boundary at its final state without saying so"
            )
        return candidates[0][1]


def _interval_seconds_f32(start: datetime, end: datetime) -> np.float32:
    """Interval length in seconds, formed the way the Fortran forms it.

    ``mpas_atm_boundaries.F`` splits the interval into whole days and
    seconds and combines them in REAL(RKIND) arithmetic:
    ``86400.0 * dd + s`` in single precision.
    """

    delta = end - start
    dd = np.float32(delta.days)
    s = np.float32(delta.seconds)
    return np.float32(np.float32(86400.0) * dd + s)


class LbcPool:
    """The two-level value/tendency pool of ``mpas_atm_boundaries.F``.

    Level 2 holds the field values read at the CURRENT interval's end;
    level 1 holds the tendencies over the current interval.  ``start`` is
    ``mpas_atm_update_bdy_tend(firstCall=.true.)``; ``advance`` is every
    later call: shift, read the next file, form tendencies.
    """

    def __init__(self, inventory: LbcInventory):
        self._inventory = inventory
        self._state: LbcFile | None = None
        self._tendencies: Dict[str, np.ndarray] | None = None
        self._interval_start: datetime | None = None

    # -- admission ---------------------------------------------------------

    def start(self, when: datetime) -> LbcFile:
        """Admit the latest file at-or-before ``when`` into level 2."""

        admitted = read_lbc_file(self._inventory.latest_before(when))
        self._state = admitted
        self._tendencies = None
        self._interval_start = None
        return admitted

    def advance(self, when: datetime | None = None) -> LbcFile:
        """Shift level 2 to level 1, admit the next file, form tendencies.

        ``when`` defaults to the current interval end, which is what the
        native driver passes: the clock sits at the boundary it just
        reached.  The tendency arithmetic is float32 throughout, matching
        the reference: ``dt`` in REAL(4), inverted once, multiplied through.
        """

        if self._state is None:
            raise LbcAdmissionError(
                "advance was called on a pool that never started; the first "
                "admission is LATEST_BEFORE the run start and must happen "
                "before any interval can have two ends"
            )
        old = self._state
        clock = when if when is not None else old.valid_time
        admitted = read_lbc_file(self._inventory.earliest_strictly_after(clock))
        dt = _interval_seconds_f32(old.valid_time, admitted.valid_time)
        inv_dt = np.float32(np.float32(1.0) / dt)
        tendencies: Dict[str, np.ndarray] = {}
        for name in LBC_REQUIRED_VARIABLES:
            tendency = (admitted.fields[name] - old.fields[name]) * inv_dt
            tendency = np.asarray(tendency, dtype=np.float32)
            tendency.setflags(write=False)
            tendencies[name] = tendency
        self._interval_start = old.valid_time
        self._state = admitted
        self._tendencies = tendencies
        return admitted

    # -- what the pool holds ----------------------------------------------

    @property
    def interval_start(self) -> datetime:
        if self._interval_start is None:
            raise LbcAdmissionError(
                "the pool holds no complete interval yet; start then advance "
                "must both have run before an interval has two ends"
            )
        return self._interval_start

    @property
    def interval_end(self) -> datetime:
        if self._state is None:
            raise LbcAdmissionError("the pool has not started")
        return self._state.valid_time

    def tendency(self, name: str) -> np.ndarray:
        """The level-1 tendency for ``name``, per second."""

        self._require_field(name)
        if self._tendencies is None:
            raise LbcAdmissionError(
                f"the pool holds no tendencies yet, so the {name} tendency "
                f"does not exist; advance must admit the far end of the "
                f"first interval before any tendency is defined"
            )
        return self._tendencies[name]

    def state_at(self, name: str, when: datetime) -> np.ndarray:
        """The field at model time ``when``: linear, backward from interval end.

        ``mpas_atm_get_bdy_state`` verbatim: ``state(end) - dt * tend`` with
        ``dt = seconds(end - when)`` formed in float32.
        """

        self._require_field(name)
        if self._state is None or self._tendencies is None:
            raise LbcAdmissionError(
                f"state_at({name}) was asked of a pool with no complete "
                f"interval; start admits one end and advance the other, and "
                f"both are needed before boundary state exists at any time"
            )
        end = self._state.valid_time
        if when <= end:
            dt = _interval_seconds_f32(when, end)
        else:
            # Past the interval end the same linear form extrapolates
            # forward; the native driver never asks for this (it advances
            # first), and a consumer that does gets the arithmetic the
            # formula defines rather than a silent clamp.
            dt = np.float32(-_interval_seconds_f32(end, when))
        return np.asarray(
            self._state.fields[name] - dt * self._tendencies[name],
            dtype=np.float32,
        )

    def _require_field(self, name: str) -> None:
        if name not in LBC_REQUIRED_VARIABLES:
            raise LbcFileError(
                f"{name} is not an lbc stream field; the stream carries "
                f"exactly {sorted(LBC_REQUIRED_VARIABLES)}.  Derived coupled "
                f"fields (lbc_rho_zz, lbc_ru, lbc_rho_edge, lbc_rtheta_m) "
                f"are computed by "
                f"hexcore.regional_v841.RegionalDrivingState, which is "
                f"the pool the regional driver consumes"
            )
