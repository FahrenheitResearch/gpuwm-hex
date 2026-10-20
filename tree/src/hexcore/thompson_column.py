"""Adapter for the ``thompson_aero_column`` backend row.

Aerosol-aware Thompson (mp=28) through the SAME frozen engine column batch
and the SAME frozen adapter class as the default ``wsm6_column`` row.  The
microphysics choice travels as data -- the configuration's
``config_microp_scheme`` (:func:`scheme_alias_for`) selects the species row,
and the sealed constructor's ``microphysics_scheme`` comes from that row --
so this module owns no physics and no second pin: it answers the door's
row hooks with the frozen adapter's own answers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigurationRefusal

SCHEME_ALIAS = "mp_thompson_aerosols"


def scheme_alias_for(backend: str) -> str:
    """The ``config_microp_scheme`` this row's runs carry."""

    return SCHEME_ALIAS


def seam_options_for(*, source_table: Any = None, **_: Any) -> Mapping[str, Any]:
    """No seam options: this row carries no point source."""

    if source_table is not None:
        raise ConfigurationRefusal(
            "source_table",
            str(source_table),
            "the thompson_aero_column row carries no point source; the table "
            "would be accepted and never read",
            "a row whose adapter reads a source table",
        )
    return {}


def pin_problem(checkout: str | Path | None = None) -> str | None:
    """The frozen row's own pin check: the same engine, the same 16 files."""

    from .forecast_door import seam_pin_problem

    return seam_pin_problem(Path(checkout) if checkout is not None else checkout)


def pin_column_batch(checkout: str | Path | None = None) -> Mapping[str, Any]:
    """The frozen pin, plus the ``pin.files`` view the door's git verifier reads."""

    from .cuda_arwen_physics_v841 import pin_arwen_physics_v841

    pin = dict(pin_arwen_physics_v841(checkout))
    pin["pin"] = {
        "files": {
            relative: {"sha256": digest}
            for relative, digest in dict(pin["source_manifest"]).items()
        },
        "frozen_batch_cross_pin": {},
    }
    return pin


def build_column_backend(row: Any, *, constructor: Any, prep_geometry: Any,
                         kernel_cache: Any, gwdo_static: Any = None,
                         gwdo_kernel_cache: Any = None, checkout: Any = None,
                         cell_area_m2: Any = None,
                         seam_options: Mapping[str, Any] | None = None) -> Any:
    """The frozen backend class; cell area is not read by this batch."""

    if seam_options:
        raise ConfigurationRefusal(
            "physics_backend", row.name,
            f"seam options {sorted(seam_options)} would be accepted and never read",
            "no seam options on this row",
        )
    from .cuda_arwen_physics_v841 import PersistentTwoPhaseCudaPhysicsBackendV841

    return PersistentTwoPhaseCudaPhysicsBackendV841(
        constructor=constructor, prep_geometry=prep_geometry,
        kernel_cache=kernel_cache, gwdo_static=gwdo_static,
        gwdo_kernel_cache=gwdo_kernel_cache, arwen_checkout=checkout)
