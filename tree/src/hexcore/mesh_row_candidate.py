"""A registry row's timestep, overridden for the duration of an anchor mint.

:mod:`hexcore.dt_admission` resolves half of the mint's chicken-and-egg
problem: ``candidate_mint`` admits ONE unanchored ``config_dt`` while a mint
run is in flight, so the sealed v8.4.1 configuration will build at a timestep
nobody has proven yet.

The other half is the mesh registry.  ``tools/mpas_mesh_binding.py`` rows
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
anchor that evidence supports is a ruling.

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
    ordinary run.  ``tools/mpas_mesh_binding.py`` calls this once, where it
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
                f"forecast; it is evidence for a ruling.  || {row.notes}"
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
        if authorization != dt_admission.CANDIDATE_MINT_AUTHORIZATION:
            raise DtAdmissionError(
                "overriding a registered row's timestep requires the "
                "candidate-mint authorization verbatim "
                f"({dt_admission.CANDIDATE_MINT_AUTHORIZATION!r}); got "
                f"{authorization!r}.  This path exists ONLY to let a mint run "
                "reach a card at a timestep no mesh row declares, and every "
                "receipt it produces says so"
            )
        if not mesh:
            raise DtAdmissionError(
                "a candidate row override must name the registered mesh it "
                "applies to: the anchor it feeds records which mesh carried "
                "the integration, and evidence with no mesh on it is not "
                "evidence"
            )
        # The question is asked of the CONFIGURATION, not of the timestep
        # alone.  RETIRED 2026-08-26 under the fix-retires-guards law: this
        # guard read admitted_timestep(dt) with the Grell-Freitas default, so
        # once dt_admission was keyed by (dt, cumulus selection) it refused
        # every convection-off mint at an already-GF-anchored timestep --
        # "20 s holds a real anchor" -- when the configuration being earned
        # held none.  MEASURED: it killed the first convection-off arm on
        # the proving RTX 5070 Ti in 0.1 s.
        # WIDENED 2026-08-26 under the fix-retires-guards law, for the
        # second time and for the identical reason: once dt_admission was
        # keyed by the surface/PBL cadence as well, a guard reading the
        # WELDED default would refuse every held-cadence mint at a timestep
        # whose welded row is anchored -- "5 s with convection off holds a
        # real anchor" -- when the configuration being earned holds none.
        # The configuration is the whole triple or the guard asks the wrong
        # question.
        configuration = (
            "convection off" if cumulus_scheme is None else str(cumulus_scheme).upper()
        )
        if (
            dt_admission.surface_pbl_key(dt_seconds, surface_pbl_seconds)
            != "dt"
        ):
            configuration += (
                f" and the surface/PBL cadence held at "
                f"{float(surface_pbl_seconds):g} s"
            )
        anchor = dt_admission.admitted_timestep(
            dt_seconds, cumulus_scheme, surface_pbl_seconds
        )
        if anchor is None:
            raise DtAdmissionError(
                f"dt={float(dt_seconds):g} s with {configuration} is not "
                f"admitted at all, so a mesh "
                f"row declaring it would be refused at bind anyway.  Open "
                f"dt_admission.candidate_mint({float(dt_seconds):g}, ...) around "
                f"this override: the timestep gate and the row must be opened "
                f"together, by the same caller, or the row is the only thing "
                f"that moved"
            )
        if not anchor.admitted_on.startswith("CANDIDATE"):
            raise DtAdmissionError(
                f"dt={float(dt_seconds):g} s with {configuration} holds a real "
                f"anchor (admitted {anchor.admitted_on}), so it needs a "
                f"REGISTERED row and not a temporary one.  This override "
                f"exists only for configurations that are still trying to "
                f"earn an anchor"
            )
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
