"""Which column-physics backend a configuration selects, as table work.

The port owns no parameterization arithmetic.  Every physics column runs
through a gpuwm column-batch seam, and until now there was exactly one: the
frozen ``gpuwm.core.mpas_column_batch.run_mpas_column_batch``, bound by
:mod:`hexcore.cuda_arwen_physics_v841` and byte-pinned by that adapter's
sixteen-file ``ARWEN_SOURCE_MANIFEST``.

Adding a second physics variant must not be a second code path through the
first.  ``mpas_column_batch.py`` is one of the sixteen files the production
adapter pins by SHA-256, so a dispatch arm inside it moves bytes that the
execution anchors are keyed to and forces a re-mint campaign.  This module is
therefore the TABLE the variants plug into: a backend is a ROW naming the
adapter module, the batch module it binds, the scalar registry it integrates,
and the anchor configuration class its evidence is filed under.  Adding a
variant is one row; the frozen row's bytes never move.

THE BREAKAGE THIS PREVENTS, and why the table refuses a rebind.  A registry
whose rows can be replaced is a registry where importing a module changes what
the DEFAULT configuration runs.  The default row here is the frozen lane, and
:func:`register_backend_row` refuses any name already present -- so no
optional module, however imported, can rebind ``wsm6_column`` to something
else.  ``tests/mod/test_mod_seam.py`` measures that the default row's
module resolution is identical before and after every optional provider is
imported.

WHAT A ROW DOES NOT DO.  A row does not admit a run.  Admission is still the
existing machinery -- :mod:`hexcore.dt_admission` for the timestep and the
physics cadences, :mod:`hexcore.device_admission` for the card,
:mod:`hexcore.cuda_backend.arch_admission` for the architecture.  A row that
names an anchor configuration class holding no anchors refuses through
:meth:`PhysicsBackendRow.require_anchored`, with the remedy named.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import MappingProxyType
from typing import Mapping

from .errors import ConfigurationRefusal


__all__ = [
    "DEFAULT_BACKEND",
    "FROZEN_WSM6_COLUMN_ROW",
    "PhysicsBackendRow",
    "registered_backend_names",
    "register_backend_row",
    "resolve_backend",
]


#: The name of the row every configuration in this tree selects today.  It is
#: a constant rather than "whatever was registered first": a default that is
#: an accident of import order is not a default.
DEFAULT_BACKEND = "wsm6_column"


@dataclass(frozen=True, slots=True)
class PhysicsBackendRow:
    """One column-physics backend, named by its table row.

    ``driven_scalar_names`` is the LEADING block of the model's scalar array
    -- the species a lateral-boundary stream is allowed to drive.
    ``appended_scalar_names`` sits strictly after it and is never driven at
    the boundary; see :meth:`model_scalar_names`.
    """

    #: The capability this row names.  Never a case name, a programme, a site
    #: or a customer (project law): a row is what the physics DOES.
    name: str
    #: The hex adapter module that binds the batch and owns the call contract.
    adapter_module: str
    #: The gpuwm module published as this row's column-batch seam.
    batch_module: str
    #: The callable in ``batch_module`` that constructs the seam.
    batch_entrypoint: str
    #: The species a boundary stream may drive, in model array order.
    driven_scalar_names: tuple[str, ...]
    #: Species carried after the driven block and never driven at the rings.
    appended_scalar_names: tuple[str, ...]
    #: The key this row's dt/execution anchors are filed under.  Two rows
    #: sharing a key would let one quote the other's measurements.
    anchor_configuration_class: str
    #: Whether that class holds any earned anchor at all.
    anchored: bool
    #: What earning them takes.  Required when ``anchored`` is False.
    unanchored_remedy: str
    #: True when this row's implementation is deliberately excluded from the
    #: published distribution, so a public install cannot resolve it.
    excluded_from_distribution: bool = False

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip().lower():
            raise ValueError(
                "a backend row name is a lowercase capability name with no "
                f"surrounding space, got {self.name!r}"
            )
        overlap = sorted(
            set(self.driven_scalar_names) & set(self.appended_scalar_names)
        )
        if overlap:
            raise ValueError(
                f"row {self.name!r} lists {overlap} as both driven and "
                "appended; a species is one or the other, and the boundary "
                "law reads the driven block by POSITION"
            )
        names = self.model_scalar_names()
        if len(set(names)) != len(names):
            raise ValueError(
                f"row {self.name!r} repeats a scalar name: {names}"
            )
        if not self.anchored and not self.unanchored_remedy.strip():
            raise ValueError(
                f"row {self.name!r} holds no anchor and names no remedy; a "
                "refusal that does not say what would lift it is a dead end"
            )

    def model_scalar_names(self) -> tuple[str, ...]:
        """The model's scalar array order: driven block, then appended."""

        return tuple(self.driven_scalar_names) + tuple(self.appended_scalar_names)

    def require_anchored(self) -> None:
        """Refuse a row whose configuration class has earned no anchors."""

        if self.anchored:
            return
        raise ConfigurationRefusal(
            "physics_backend",
            self.name,
            (
                f"the {self.name!r} physics backend is a NEW configuration "
                f"class in the anchor registry ({self.anchor_configuration_class}) "
                "and that class holds no timestep anchor and no execution "
                "anchor.  The concrete breakage this prevents: a run would "
                "integrate an outer step, a physics call cadence and a clock "
                "that nobody has measured for THIS backend, and would produce "
                "a receipt with nothing to check it against -- the registered "
                "anchors were all earned on the frozen WSM6 column backend "
                "and say nothing about this one.  " + self.unanchored_remedy
            ),
            f"a registered anchor for configuration class "
            f"{self.anchor_configuration_class!r}",
        )

    def build_column_backend(
        self,
        *,
        constructor: Any,
        prep_geometry: Any,
        kernel_cache: Any,
        gwdo_static: Any = None,
        gwdo_kernel_cache: Any = None,
        checkout: Any = None,
        cell_area_m2: Any = None,
        seam_options: Mapping[str, Any] | None = None,
    ) -> Any:
        """Construct this row's column-physics backend -- ONE call site for
        every row.

        A provider publishes ``build_column_backend(row, **kwargs)`` on its
        adapter module and receives everything, the per-column cell area
        and the seam options included.  The frozen adapter publishes its
        class instead, and a row naming it constructs that class with the
        production keyword set alone: the frozen batch weighs nothing by
        cell area and carries no point source, so either extra is refused
        by name rather than accepted and never read.
        """

        module = self.load_adapter()
        builder = getattr(module, "build_column_backend", None)
        base = {
            "constructor": constructor,
            "prep_geometry": prep_geometry,
            "kernel_cache": kernel_cache,
            "gwdo_static": gwdo_static,
            "gwdo_kernel_cache": gwdo_kernel_cache,
        }
        if builder is not None:
            return builder(
                self,
                **base,
                checkout=checkout,
                cell_area_m2=cell_area_m2,
                seam_options=seam_options,
            )
        extras = [
            name
            for name, value in (
                ("cell_area_m2", cell_area_m2),
                ("seam_options", seam_options),
            )
            if value is not None and (name != "seam_options" or len(value))
        ]
        if extras:
            raise ConfigurationRefusal(
                "physics_backend",
                self.name,
                (
                    f"row {self.name!r} binds the frozen column batch, which "
                    "weighs nothing by cell area and carries no point "
                    f"source; {extras} would be accepted and never read, "
                    "which is configuration the run does not have.  A "
                    "point source selects a seeded row"
                ),
                f"a row whose adapter publishes build_column_backend, or no {extras}",
            )
        cls = getattr(module, "PersistentTwoPhaseCudaPhysicsBackendV841", None)
        if cls is None:
            raise ConfigurationRefusal(
                "physics_backend",
                self.name,
                (
                    f"row {self.name!r} names adapter module "
                    f"{self.adapter_module!r}, which publishes neither "
                    "build_column_backend nor "
                    "PersistentTwoPhaseCudaPhysicsBackendV841; there is "
                    "nothing to construct"
                ),
                "an adapter module publishing one of the two",
            )
        return cls(**base, arwen_checkout=checkout)

    def load_adapter(self):
        """Import this row's adapter module, or refuse by name."""

        try:
            return importlib.import_module(self.adapter_module)
        except ImportError as error:
            raise ConfigurationRefusal(
                "physics_backend",
                self.name,
                (
                    f"row {self.name!r} names adapter module "
                    f"{self.adapter_module!r}, which this installation does "
                    "not carry"
                    + (
                        ".  That module is deliberately excluded from the "
                        "published distribution, so this row is reachable "
                        "only from a source checkout that carries it"
                        if self.excluded_from_distribution
                        else ""
                    )
                    + f" ({error})"
                ),
                f"an installation carrying {self.adapter_module!r}",
            ) from error


#: The frozen lane.  Everything about it is a statement of what is already
#: true at this tip -- ``cuda_arwen_physics_v841`` binds
#: ``gpuwm.core.mpas_column_batch.run_mpas_column_batch`` and refuses any
#: scalar order other than ``WSM6_SCALAR_NAMES`` -- restated here as a row so
#: that "which backend does this configuration select" has ONE answer surface
#: instead of being implicit in an import.
FROZEN_WSM6_COLUMN_ROW = PhysicsBackendRow(
    name=DEFAULT_BACKEND,
    adapter_module="hexcore.cuda_arwen_physics_v841",
    batch_module="gpuwm.core.mpas_column_batch",
    batch_entrypoint="run_mpas_column_batch",
    driven_scalar_names=("qv", "qc", "qr", "qi", "qs", "qg"),
    appended_scalar_names=(),
    anchor_configuration_class="v841-arwen-column-wsm6",
    anchored=True,
    unanchored_remedy="",
)


_ROWS: dict[str, PhysicsBackendRow] = {FROZEN_WSM6_COLUMN_ROW.name: FROZEN_WSM6_COLUMN_ROW}

#: Optional modules that register further rows on import.  A row whose
#: implementation is not part of the published distribution cannot be declared
#: here as data -- the data would ship and the implementation would not -- so
#: the table names the PROVIDER and the provider owns its own row.  Import
#: failure is not an error: it is what a public install looks like.
_ROW_PROVIDERS: tuple[str, ...] = ("hexcore.mod",)

_providers_loaded = False


def _load_providers() -> None:
    global _providers_loaded
    if _providers_loaded:
        return
    # Set first: a provider that imports this module back must not recurse.
    _providers_loaded = True
    for module in _ROW_PROVIDERS:
        try:
            importlib.import_module(module)
        except ImportError:
            continue


def register_backend_row(row: PhysicsBackendRow) -> PhysicsBackendRow:
    """Add one row.  A name already present is REFUSED, never replaced.

    THE BREAKAGE THIS PREVENTS: a replaceable row means importing a module
    changes which batch the DEFAULT configuration runs, silently and by import
    order.  The frozen lane's bytes are pinned by execution anchors; what
    selects it must be equally unmovable.
    """

    if not isinstance(row, PhysicsBackendRow):
        raise TypeError("a backend row must be a PhysicsBackendRow")
    existing = _ROWS.get(row.name)
    if existing is not None:
        if existing == row:
            return existing
        raise ConfigurationRefusal(
            "physics_backend",
            row.name,
            (
                f"a backend row named {row.name!r} is already registered and "
                "rows are not replaceable; rebinding one would change which "
                "column-batch seam an existing configuration runs, by import "
                "order rather than by decision"
            ),
            "a new row name, or a ruling that retires the registered row",
        )
    _ROWS[row.name] = row
    return row


def registered_backend_names() -> tuple[str, ...]:
    """Every row this installation can resolve, sorted."""

    _load_providers()
    return tuple(sorted(_ROWS))


def resolve_backend(name: str = DEFAULT_BACKEND) -> PhysicsBackendRow:
    """The row a configuration selects, or a refusal naming what exists."""

    key = str(name).strip().lower()
    row = _ROWS.get(key)
    if row is not None:
        return row
    _load_providers()
    row = _ROWS.get(key)
    if row is not None:
        return row
    raise ConfigurationRefusal(
        "physics_backend",
        name,
        (
            "no column-physics backend row carries that name in this "
            f"installation.  Registered: {sorted(_ROWS)}"
        ),
        f"physics_backend in {sorted(_ROWS)}",
    )


def backend_rows() -> Mapping[str, PhysicsBackendRow]:
    """A read-only view of every resolvable row."""

    _load_providers()
    return MappingProxyType(dict(_ROWS))
