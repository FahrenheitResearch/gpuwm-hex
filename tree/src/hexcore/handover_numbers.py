"""Number concentrations handed from a hex parent to a hex region: carried when published, else diagnosed.

THE BREAKAGE THIS PREVENTS (init-from-any-run proof, 2026-10-24).  A child handed hydrometeor MASS with number
concentration 0 sizes that mass at its scheme's PSD limits: Thompson's entry block makes 2.5 mm rain and
drizzle-sized droplets of it, reflectivity from it reads 60-70 dBZ rings or NaN, and in the ArWen 1 km child it
seeded spurious early cells.  On the hex-region route (``gpuwm-hex regional --parent-*``) the numbers were lost
twice: ``parent_state`` copied a frame variable only when the parent's init already held it (an init holds no
ni/nr/nc), and the start and boundary remaps moved the six water masses only.  A region running a number-carrying
row (aerosol-aware Thompson, the MYNN/RUC/GSL suite) then either refused the partial species row at the forecast
door or started its numbers from +0 against carried mass.

The rule, the same one the engine applies on its own parent -> child routes:

* a number the parent's frames publish is CARRIED (remapped like every scalar, into the start and every boundary
  frame);
* a number the region's row carries and the parent does not publish is DIAGNOSED from the mass with the region
  scheme's own relations, only where the mass is above zero and the number is not -- for the Thompson family the
  scheme's cold-start functions (WRF module_mp_thompson.F make_RainNumber :9163-9194, make_IceNumber :9044-9114,
  make_DropletNumber :9119-9158, ported bitwise in gpuwm.core.thompson_entry), per volume at the moist density
  MPAS's physics interface forms (``rho_p = zz rho_zz (1 + qv)``), divided back to per kilogram;
* the two aerosol numbers (nwfa, nifa) have no mass to diagnose from; absent from the parent they start at the
  same +0 a cold start gives them today, and the droplet number is sized with no aerosol (make_DropletNumber's
  qnwfa <= 0 branch, by the surface type).

An exact physics mode (``--physics-mode exact``, or a strict process, GPUWM_WRF_EXACT=1) keeps today's numbers: it
diagnoses nothing.  Carrying a published number is not a correction and runs in every mode.

A number that is neither carried nor diagnosed is never PUBLISHED (:func:`complete_state_numbers`): written as a +0
plane it would be driven at the rings as number 0 under the driven mass, the very breakage above, and an exact or
no-diagnosis region under a numberless parent (the default WSM6 one) keeps the six-mass boundary it had before.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import numpy as np

#: The number species a hex boundary file can carry (hexcore.lbc.LBC_OPTIONAL_SCALAR_VARIABLES), in the order the
#: number-carrying rows declare them.
NUMBER_SPECIES = ("ni", "nr", "nc", "nwfa", "nifa")

#: Number -> the mass it is diagnosed from.  The aerosol numbers have none.
DIAGNOSED_FROM = {"ni": "qi", "nr": "qr", "nc": "qc"}

#: Rows whose numbers follow Thompson's relations (classic, aerosol-aware, TEMPO).
THOMPSON_FAMILY = frozenset({8, 28, 900})

_R_D, _CP, _P0 = 287.0, 1004.5, 100000.0
_CV = _CP - _R_D


def exact_requested(forecast_args: Sequence[str] | None) -> bool:
    """True when the region runs exact physics (no correction may change its start)."""
    if os.environ.get("GPUWM_WRF_EXACT", "").strip() not in ("", "0"):
        return True
    from .global_door import forecast_arg_physics
    return forecast_arg_physics(forecast_args).get("mode") == "exact"


def frame_number_species(path) -> tuple[str, ...]:
    """The number species a hex history/analysis file carries on (Time, nCells, nVertLevels)."""
    from netCDF4 import Dataset
    with Dataset(path) as d:
        return tuple(name for name in NUMBER_SPECIES
                     if name in d.variables and d[name].dimensions == ("Time", "nCells", "nVertLevels"))


def _forecast_flag(forecast_args: Sequence[str] | None, flag: str) -> str | None:
    items, found, index = [str(item) for item in forecast_args or ()], None, 0
    while index < len(items):
        name, equals, value = items[index].partition("=")
        if name == flag:
            if not equals:
                if index + 1 >= len(items):
                    break
                index += 1
                value = items[index]
            found = value
        index += 1
    return found


def region_row_names(forecast_args: Sequence[str] | None, *, nominal_dx_m: float | None = None) -> tuple[str, ...]:
    """The scalar names of the row the region's forecast will run (``--forecast-arg``), default the frozen WSM6 row.

    A suite or backend the forecast door would refuse resolves to the default here; the forecast door then refuses
    it by name, so nothing is diagnosed for a run that never starts.
    """
    from .physics_backend_admission import DEFAULT_BACKEND, resolve_backend
    backend = _forecast_flag(forecast_args, "--physics-backend")
    suite = _forecast_flag(forecast_args, "--physics-suite")
    name = backend or DEFAULT_BACKEND
    if suite and not backend:
        try:
            from .physics_suites import resolve_physics
            name = resolve_physics(suite, {}, mesh=None, nominal_dx_m=nominal_dx_m).backend_row
        except Exception:  # the forecast door refuses the suite by name; nothing to diagnose for it here
            name = DEFAULT_BACKEND
    try:
        return tuple(resolve_backend(str(name)).model_scalar_names())
    except Exception:
        return tuple(resolve_backend(DEFAULT_BACKEND).model_scalar_names())


def region_number_species(forecast_args: Sequence[str] | None, parent_frame,
                          *, nominal_dx_m: float | None = None) -> tuple[str, ...]:
    """Numbers the region's start and boundaries carry: the region row's, plus any the parent publishes."""
    row = region_row_names(forecast_args, nominal_dx_m=nominal_dx_m)
    published = set(frame_number_species(parent_frame))
    return tuple(name for name in NUMBER_SPECIES if name in row or name in published)


def _row_mp(names: Sequence[str]) -> int | None:
    try:
        from .species_row import species_row_for_names
        return int(species_row_for_names(tuple(names)).mp_physics)
    except Exception:
        return None


def diagnose_scheme(forecast_args: Sequence[str] | None, *, nominal_dx_m: float | None = None) -> int | None:
    """The mp_physics whose relations size a missing number, or None (diagnose nothing).

    None for an exact run, and for a row outside the Thompson family (its own entry closure is not ported here,
    so its numbers keep the cold-start value rather than another scheme's sizes).
    """
    if exact_requested(forecast_args):
        return None
    mp = _row_mp(region_row_names(forecast_args, nominal_dx_m=nominal_dx_m))
    return mp if mp in THOMPSON_FAMILY else None


def _temperature_and_density(fields: Mapping[str, np.ndarray]):
    """Absolute temperature and moist density from the native fields (MPAS's equation of state)."""
    rho_d = np.asarray(fields["rho"], dtype=np.float64)
    theta_m = np.asarray(fields["theta_m"], dtype=np.float64)
    theta = np.asarray(fields["theta"], dtype=np.float64)
    qv = np.asarray(fields["qv"], dtype=np.float64)
    exner = np.power(_R_D * rho_d * theta_m / _P0, _R_D / _CV)
    return (theta * exner).astype(np.float32), (rho_d * (1.0 + qv)).astype(np.float32)


def diagnose(name: str, fields: Mapping[str, np.ndarray], *, xland: np.ndarray | None) -> np.ndarray:
    """Number ``name`` (per kg) from its mass in ``fields`` with Thompson's cold-start relation; 0 where no mass."""
    from gpuwm.core.thompson_entry import make_droplet_number, make_ice_number, make_rain_number
    mass = np.asarray(fields[DIAGNOSED_FROM[name]], dtype=np.float32)
    out = np.zeros(mass.shape, dtype=np.float32)
    seed = mass > 0
    if not seed.any():
        return out
    temperature, rho = _temperature_and_density(fields)
    q_m3 = mass[seed] * rho[seed]
    if name == "nr":
        n_m3 = make_rain_number(q_m3, temperature[seed])
    elif name == "ni":
        n_m3 = make_ice_number(q_m3, temperature[seed])
    else:
        nwfa = fields.get("nwfa")
        nwfa_m3 = (np.zeros_like(q_m3) if nwfa is None
                   else np.asarray(nwfa, dtype=np.float32)[seed] * rho[seed])
        land = np.ones(mass.shape[-1], dtype=np.float32) if xland is None else np.asarray(xland, dtype=np.float32)
        surface = np.broadcast_to(land[None, :], mass.shape)[seed]
        n_m3 = make_droplet_number(q_m3, nwfa_m3, surface)
    values = np.asarray(n_m3, dtype=np.float32) / rho[seed]
    if not np.isfinite(values).all() or (values < 0).any():
        raise FloatingPointError(f"diagnosed {name} is not finite and non-negative")
    out[seed] = values
    return out


def complete_numbers(fields: dict[str, np.ndarray], numbers: Sequence[str], *, scheme: int | None,
                     xland: np.ndarray | None) -> dict[str, Any]:
    """Make ``fields`` hold every name in ``numbers``: carried as published, diagnosed where missing, else +0.

    ``fields`` are native fields (:func:`hexcore.analysis_remap.native_fields`), (nVertLevels, nCells) planes.  A
    carried number at or below zero under mass above zero is filled too, so a published-but-empty plane is never
    handed on with mass.  Returns the receipt.
    """
    shape = np.asarray(fields["qv"]).shape
    receipt: dict[str, Any] = {"scheme": scheme, "carried": [], "diagnosed": {}, "cold_zero": []}
    for name in numbers:
        present = name in fields
        if not present:
            fields[name] = np.zeros(shape, dtype=np.float32)
        else:
            receipt["carried"].append(name)
        source = DIAGNOSED_FROM.get(name)
        if scheme is None or source is None or source not in fields:
            if not present:
                receipt["cold_zero"].append(name)
            continue
        plane = np.asarray(fields[name], dtype=np.float32)
        holes = (np.asarray(fields[source]) > 0) & (plane <= 0)
        if holes.any():
            diagnosed = diagnose(name, fields, xland=xland)
            plane = np.where(holes, diagnosed, plane).astype(np.float32)
            fields[name] = plane
            receipt["diagnosed"][name] = int(np.count_nonzero(holes))
    return receipt


def published_numbers(numbers: Sequence[str], receipt: Mapping[str, Any], *, prefix: bool) -> tuple[str, ...]:
    """The numbers a frame publishes: carried or diagnosed ones, never a cold +0 plane.

    ``prefix`` (a boundary series) stops at the first cold one, so the published numbers stay the contiguous prefix of
    the row's block that the forecast door drives (``select_lbc_scalar_names``).
    THE BREAKAGE THIS PREVENTS (release review 2.9.7, MEDIUM 21): a cold number written as a +0 boundary plane was
    driven at the rings as number 0 under the driven mass (an exact-mode Thompson region under the default WSM6
    parent: lbc_ni/nr/nc all zero, Thompson sizing the ring's mass at its limits).
    """
    cold = set(receipt.get("cold_zero", ()))
    if not prefix:
        return tuple(name for name in numbers if name not in cold)
    out = []
    for name in numbers:
        if name in cold:
            break
        out.append(name)
    return tuple(out)


def complete_state_numbers(loaded, species: Sequence[str], numbers: Sequence[str], *, scheme: int | None,
                           xland: np.ndarray | None, prefix: bool = False) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Extend a cold-loaded state's scalar block from ``species`` to ``species + numbers`` (missing ones completed).

    ``loaded`` is what :func:`hexcore.analysis_remap.load` or :func:`hexcore.nest_boundary.load_history` returns.
    Returns the new species order (``species`` without the numbers, then the published numbers in their given order,
    so every frame of a series has one order) and the receipt; the state's scalars are replaced in that order.  A
    number neither carried nor diagnosed is left out (:func:`published_numbers`; ``prefix`` for a boundary series).
    """
    from .analysis_remap import native_fields
    species = tuple(species)
    fields = native_fields(loaded.state, loaded.saved_diagnostics, loaded.vertical, species)
    receipt = complete_numbers(fields, numbers, scheme=scheme, xland=xland)
    kept = published_numbers(numbers, receipt, prefix=prefix)
    receipt["published"] = list(kept)
    order = tuple(name for name in species if name not in numbers) + kept
    block = np.stack([np.asarray(fields[name], dtype=loaded.state.scalars.dtype) for name in order])
    loaded.state.scalars = np.ascontiguousarray(block)
    return order, receipt


def read_xland(path) -> np.ndarray | None:
    """The land/water flag a droplet number is sized by (MPAS ``xland``: 1 land, 2 water), or None."""
    from netCDF4 import Dataset
    with Dataset(path) as d:
        if "xland" not in d.variables:
            return None
        value = np.asarray(d["xland"][:], dtype=np.float32)
    return value[0] if value.ndim == 2 else value


__all__ = ["DIAGNOSED_FROM", "NUMBER_SPECIES", "THOMPSON_FAMILY", "complete_numbers", "complete_state_numbers",
           "diagnose", "diagnose_scheme", "exact_requested", "frame_number_species", "published_numbers",
           "read_xland", "region_number_species", "region_row_names"]
