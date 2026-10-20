"""A registry row's timestep, overridden for the duration of an anchor mint.

SUPERSEDED 2026-10-10 by ``gpuwm-hex forecast --dt``: a run takes any
timestep directly, so this instrument is no longer the only way to reach one.
It is kept for the mint tools that use it, and its anchor refusals are
retired (no candidate scope or authorization sentence is required, and a
configuration that already holds an anchor may be overridden too).  What it
still refuses is what would make a receipt ambiguous: an override with no
mesh, and two overrides of one row at once.

:mod:`hexcore.dt_admission` resolves half of the mint's chicken-and-egg
problem: ``candidate_mint`` admits ONE unanchored ``config_dt`` while a mint
run is in flight, so the sealed v8.4.1 configuration will build at a timestep
nobody has proven yet.

The other half is the mesh registry.  ``src/hexcore/drivers/mpas_mesh_binding.py`` rows
carry a *declared* timestep, the forecast door computes its step schedule
from that field, and ``bind_mesh`` admits it against the earned-anchor
registry.  So an admitted candidate timestep is not reachable at all unless
some row declares it -- and the rows that declare the interesting values are
exactly the meshes an anchor would unblock, which is backwards: those meshes
are large, and an anchor is a property of the TIMESTEP, not of the mesh.  The
cheap and correct place to earn one is the smallest registered mesh, whose
own Courant limit is an upper bound that every candidate sits far beneath.

Registering four permanent rows for the same mesh files at four timesteps
would be the alternative, and it would leave the shipped registry carrying
rows that exist only because a mint once needed them.  This module is the
narrower thing: one row's ``dt_seconds`` is replaced for the duration of one
mint, on the same files, under the same authorization sentence, and it is
withdrawn on exit.

Everything about the path is loud and it cannot be entered by accident:

* the caller repeats :data:`hexcore.dt_admission.CANDIDATE_MINT_AUTHORIZATION`
  verbatim, the same sentence the timestep half requires;
* the candidate timestep must ALREADY be admitted as a candidate -- a live
  ``dt_admission.candidate_mint`` whose row is stamped ``CANDIDATE``.  A mesh
  row can therefore never declare a timestep the timestep gate has not itself
  been opened for, in the same breath, by the same caller;
* a timestep holding a REAL anchor is refused here, because a proven timestep
  needs a registered row and not a temporary one;
* the row's ``notes`` are stamped so the string lands in the bind log and the
  run receipt; and
* the override is removed when the mint finishes.

It admits nothing on its own.  What a mint earns is evidence; registering the
anchor that evidence supports is a decision of record.

The state lives here rather than in ``mpas_mesh_binding`` itself because the
forecast door loads that file by path with ``importlib`` and re-executes it
per run, which would reset any module-level state it held.  This module is
imported normally, so it is one object for the life of the process -- the
same property ``dt_admission`` relies on.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Any, Mapping

from hexcore import dt_admission
from hexcore.dt_admission import DtAdmissionError

__all__ = [
    "CANDIDATE_ROW_MARKER",
    "active_overrides",
    "apply_overrides",
    "candidate_mesh_dt",
]

#: Stamped into an overridden row's ``notes`` so the bind log and the run
#: receipt both say what the row is, in the row's own words.
CANDIDATE_ROW_MARKER = "CANDIDATE-UNANCHORED-DT"

_OVERRIDES: dict[str, float] = {}


def active_overrides() -> Mapping[str, float]:
    """The mesh rows whose timestep is currently overridden, if any."""

    return MappingProxyType(dict(_OVERRIDES))


def apply_overrides(rows: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return ``rows`` with any live candidate timestep applied.

    Returns the mapping unchanged when no mint is in flight, which is every
    ordinary run.  ``src/hexcore/drivers/mpas_mesh_binding.py`` calls this once, where it
    builds ``MESH_BINDINGS``, so a re-executed copy of that module picks the
    override up exactly as the first copy did.
    """

    if not _OVERRIDES:
        return rows
    patched = dict(rows)
    for mesh, dt_seconds in _OVERRIDES.items():
        row = patched.get(mesh)
        if row is None:
            continue
        patched[mesh] = replace(
            row,
            dt_seconds=float(dt_seconds),
            notes=(
                f"{CANDIDATE_ROW_MARKER}: this row's declared timestep is "
                f"overridden to {float(dt_seconds):g} s for the duration of one "
                f"anchor mint, on unchanged grid/static bytes.  The registered "
                f"row declares {float(row.dt_seconds):g} s and is restored when "
                f"the mint exits.  Nothing this run produces is an anchored "
                f"forecast; it is evidence for registering an anchor.  || {row.notes}"
            ),
        )
    return MappingProxyType(patched)


class _CandidateMeshDt:
    """Context manager overriding ONE registered row's timestep, for a mint."""

    def __init__(
        self,
        mesh: str,
        dt_seconds: float,
        *,
        authorization: str,
        cumulus_scheme: str | None = "gf",
        surface_pbl_seconds: float | None = None,
    ) -> None:
        self.authorization = authorization
        if not mesh:
            raise DtAdmissionError(
                "a candidate row override must name the registered mesh it "
                "applies to: the anchor it feeds records which mesh carried "
                "the integration, and evidence with no mesh on it is not "
                "evidence"
            )
        # RETIRED 2026-10-10: the refusals that required a live candidate
        # mint for this configuration and refused one that already held a
        # registered anchor.  No timestep is refused for its anchor any more,
        # so neither guarded a breakage.
        self.mesh = str(mesh)
        self.dt_seconds = float(dt_seconds)
        self._restore: tuple[bool, float] | None = None

    def __enter__(self) -> "_CandidateMeshDt":
        if self.mesh in _OVERRIDES:
            raise DtAdmissionError(
                f"mesh {self.mesh!r} already carries a candidate timestep "
                f"override at {_OVERRIDES[self.mesh]:g} s; one mint at a time "
                f"per row, or the receipt cannot say which timestep it measured"
            )
        self._restore = (False, 0.0)
        _OVERRIDES[self.mesh] = self.dt_seconds
        return self

    def __exit__(self, *exception: object) -> None:
        if self._restore is not None:
            _OVERRIDES.pop(self.mesh, None)
            self._restore = None


def candidate_mesh_dt(
    mesh: str,
    dt_seconds: float,
    *,
    authorization: str,
    cumulus_scheme: str | None = "gf",
    surface_pbl_seconds: float | None = None,
) -> _CandidateMeshDt:
    """Override one registered row's timestep for the duration of a mint."""

    return _CandidateMeshDt(
        mesh,
        dt_seconds,
        authorization=authorization,
        cumulus_scheme=cumulus_scheme,
        surface_pbl_seconds=surface_pbl_seconds,
    )
