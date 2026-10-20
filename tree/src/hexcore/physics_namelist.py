"""Read the physics of an MPAS ``namelist.atmosphere``, option by option.

A user hands hex the ``namelist.atmosphere`` a model was run with -- RRFS v2's,
say -- and gets back either the physics hex would run for it or one named
refusal per option hex cannot honour.  Every option of the namelist ends in
exactly one disposition:

* ``admitted``: carried as written (a scheme the column seam runs, a switch
  whose value is what hex does);
* ``ignored``: read and ignored exactly as MPAS ignores it (a TEMPO knob when
  the microphysics is not TEMPO, a stochastic-physics parameter when stochastic
  physics is off), with the reason;
* ``dycore``: a dynamics option, handed to the v8.4.1 dycore namelist door
  (:func:`hexcore.config.dry_dycore_config_from_namelist`), whose verdict is
  recorded.  The forecast door then holds the run to it
  (:func:`dycore_disagreements`): an option whose value the run's own
  configuration does not hold refuses the run, naming the flag that carries
  it, because a run that took the namelist's physics and the door's own
  dynamics would be reported as the namelist's run and be another one;
* ``outside``: a run-control option of a group this importer does not read
  (I/O, decomposition, restart, printout, soundings, assimilation), named so a
  reader sees it was not silently dropped;
* ``refused``: an option hex cannot honour, with the breakage accepting it
  would cause and the follow-up that would lift it.

The scheme selectors resolve as MPAS resolves them: ``config_physics_suite``
names an MPAS suite (``hrrrv5`` ...) or a hex suite, options left at
``'suite'`` take the suite's value, and the result goes through the same
resolver every other door uses (:func:`hexcore.physics_suites.resolve_physics`).
Which options exist, and what each one means, is the ``namelist`` block of the
suite table: the code reads rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Mapping

from .config import parse_fortran_namelist
from .errors import ConfigurationRefusal
from .physics_suites import (
    OptionRefusal,
    PhysicsSelectionRefusal,
    ResolvedPhysics,
    load_table,
    resolve_physics,
)

__all__ = [
    "Disposition",
    "NamelistPhysics",
    "dycore_disagreements",
    "import_physics_namelist",
    "read_physics_namelist",
]

#: The &nhyd_model options that are the run's calendar, not the dycore's
#: arithmetic: the dycore door refuses them as outside DryDycoreConfig, and
#: the forecast door takes them from --start-time/--hours.
_CALENDAR = ("config_start_time", "config_run_duration", "config_stop_time",
             "config_calendar_type")

_INTERVAL = re.compile(r"^(?:(?P<d>\d+)_)?(?P<h>\d+):(?P<m>\d+):(?P<s>\d+)$")


@dataclass(frozen=True)
class Disposition:
    group: str
    option: str
    value: Any
    status: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"group": self.group, "option": self.option, "value": self.value,
                "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class NamelistPhysics:
    """The outcome of one import: every option's disposition, and the
    resolved physics when nothing was refused."""

    dispositions: tuple[Disposition, ...]
    physics: ResolvedPhysics | None
    refusals: tuple[OptionRefusal, ...]
    radiation_seconds: float | None
    dycore_config: Any = None
    suite: str | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)
    #: ``(group, option, value)`` of every option dispositioned ``dycore``:
    #: what :func:`dycore_disagreements` holds the run's configuration to.
    dycore_options: tuple[tuple[str, str, Any], ...] = ()
    stochastic: Mapping[str, Any] = field(default_factory=dict)

    @property
    def admitted(self) -> bool:
        return not self.refusals

    def require_admitted(self) -> ResolvedPhysics:
        if self.refusals:
            raise PhysicsSelectionRefusal(self.refusals, context="namelist.atmosphere")
        assert self.physics is not None
        return self.physics

    def report(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "suite": self.suite,
            "overrides": dict(self.overrides),
            "radiation_seconds": self.radiation_seconds,
            "stochastic": dict(self.stochastic),
            "physics": None if self.physics is None else self.physics.receipt(),
            "refusals": [
                {"option": item.knob, "value": item.value, "reason": item.reason,
                 "follow_up": item.follow_up}
                for item in self.refusals
            ],
            "dispositions": [item.as_dict() for item in self.dispositions],
        }


def _interval_seconds(text: Any) -> float | None:
    match = _INTERVAL.match(str(text).strip())
    if match is None:
        return None
    days = int(match.group("d") or 0)
    return float(days * 86400 + int(match.group("h")) * 3600
                 + int(match.group("m")) * 60 + int(match.group("s")))


def _coerce_override(row: Mapping[str, Any] | None, value: Any) -> Any:
    """A command-line override's text, typed as the option's row expects."""

    if not isinstance(value, str) or row is None:
        return value
    text = value.strip().strip("'\"")
    wanted = row.get("value")
    values = row.get("values")
    # The loaded table holds ``values`` as a tuple, and ``(0, 1) ==
    # (False, True)`` in Python, so a logical row is told apart from an
    # integer row by the type of its values, not by equality.
    logical_values = isinstance(values, (list, tuple)) and bool(values) and all(
        isinstance(v, bool) for v in values)
    if (isinstance(wanted, bool) or row.get("kind") == "frac_seaice"
            or logical_values):
        lowered = text.lower().strip(".")
        if lowered in ("true", "t"):
            return True
        if lowered in ("false", "f"):
            return False
        return text
    integer_values = isinstance(values, (list, tuple)) and bool(values) and all(
        isinstance(v, int) and not isinstance(v, bool) for v in values)
    if (isinstance(wanted, int) or row.get("kind") == "soil_levels"
            or integer_values):
        try:
            return int(text)
        except ValueError:
            return text
    return text


def _same(value: Any, wanted: Any) -> bool:
    if isinstance(wanted, bool):
        return isinstance(value, bool) and value is wanted
    if isinstance(wanted, int):
        return isinstance(value, int) and not isinstance(value, bool) and value == wanted
    return value == wanted


def import_physics_namelist(
    text: str,
    *,
    mesh: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> NamelistPhysics:
    """Every option of ``text`` dispositioned, and the physics it resolves to.

    ``overrides`` (the forecast door's ``--physics KEY=VALUE``) replace the
    namelist's value of an option in a physics group, or add a selector to
    ``&physics``, BEFORE anything is dispositioned: an override is read
    exactly as if the namelist had said it, so it lifts a refusal only by
    giving a value the table admits.
    """

    table = load_table()
    spec = table["namelist"]
    rows = spec["options"]
    physics_groups = {name.casefold() for name in spec["physics_groups"]}
    dycore_groups = {name.casefold() for name in spec["dycore_groups"]}
    dycore_ignored = {name.casefold(): why for name, why in spec["dycore_ignored"].items()}
    outside = {name.casefold(): why for name, why in spec["outside_groups"].items()}
    stochastic = set(spec["stochastic_parameters"])

    parsed = {group: dict(values) for group, values in parse_fortran_namelist(text).items()}
    for key, value in (overrides or {}).items():
        value = _coerce_override(rows.get(str(key).casefold()), value)
        placed = False
        for group in physics_groups & set(parsed):
            for option in list(parsed[group]):
                if option.casefold() == str(key).casefold():
                    parsed[group][option] = value
                    placed = True
        if not placed:
            parsed.setdefault("physics", {})[str(key)] = value
    defaulted, registry_name = _apply_registry_defaults(parsed, spec, physics_groups)
    dispositions: list[Disposition] = []
    refusals: list[OptionRefusal] = []
    suite: str | None = None
    overrides: dict[str, Any] = {}
    soil_levels: tuple[str, Any] | None = None
    intervals: dict[str, float] = {}
    scheme_options: list[tuple[str, str, Any, Mapping[str, Any]]] = []
    stochastic_seen: list[tuple[str, str, Any]] = []
    dycore_text: list[str] = []

    conditional: list[tuple[str, str, Any, Mapping[str, Any]]] = []

    def refuse(group: str, option: str, value: Any, reason: str, follow_up: str) -> None:
        if option.casefold() in defaulted:
            reason = _default_note(registry_name) + reason
            follow_up = f"state {option} in the namelist; " + follow_up
        refusals.append(OptionRefusal(option, value, reason, follow_up))
        dispositions.append(Disposition(group, option, value, "refused", reason))

    for group, assignments in parsed.items():
        if group in dycore_groups:
            kept = []
            iau_off = group == "iau" and str(
                {k.casefold(): v for k, v in assignments.items()}.get("config_iau_option", "off")
            ).strip().lower() == "off"
            for option, value in assignments.items():
                if iau_off and option.casefold() != "config_iau_option":
                    dispositions.append(Disposition(
                        group, option, value, "ignored",
                        "an IAU parameter; MPAS reads it only when "
                        "config_IAU_option is not 'off'"))
                elif option.casefold() in _CALENDAR:
                    dispositions.append(Disposition(
                        group, option, value, "outside",
                        "the run's calendar: the forecast door takes it from "
                        "--start-time and --hours"))
                else:
                    kept.append((option, value))
            if kept:
                dycore_text.append(_render_group(group, kept))
            continue
        if group in outside:
            for option, value in assignments.items():
                dispositions.append(Disposition(group, option, value, "outside", outside[group]))
            continue
        if group not in physics_groups:
            for option, value in assignments.items():
                refuse(group, option, value,
                       f"&{group} is neither a physics group this importer reads "
                       "nor a group it knows to be outside the physics, so its "
                       "options would be accepted and never read",
                       "a row in the suite table's namelist block")
            continue
        for option, value in assignments.items():
            key = option.casefold()
            row = rows.get(key)
            if key in stochastic:
                stochastic_seen.append((group, option, value))
                continue
            if row is None:
                refuse(group, option, value,
                       "the suite table has no row for this physics option, so it "
                       "would be accepted and never read",
                       "a row in the suite table's namelist.options")
                continue
            kind = row["kind"]
            if kind == "suite":
                suite = str(value)
                dispositions.append(Disposition(group, option, value, "admitted",
                                                "the suite the selectors resolve from"))
            elif kind == "selector":
                overrides[key] = str(value)
                dispositions.append(Disposition(group, option, value, "admitted",
                                                "scheme selector (resolved below)"))
            elif kind == "soil_levels":
                soil_levels = (group, value)
            elif kind == "frac_seaice":
                if not isinstance(value, bool):
                    refuse(group, option, value, "the MPAS Registry type is logical",
                           "config_frac_seaice = true or false")
                else:
                    overrides["config_frac_seaice"] = value
                    dispositions.append(Disposition(group, option, value, "admitted",
                                                    "the land surface's sea-ice switch"))
            elif kind == "surface_update":
                if not isinstance(value, bool):
                    refuse(group, option, value, "the MPAS Registry type is logical",
                           f"{option} = true or false")
                else:
                    overrides[key] = value
                    dispositions.append(Disposition(group, option, value, "admitted",
                        "shared GPU surface service; automatically enabled by a covering surface series"))
            elif kind == "radiation_interval":
                seconds = _interval_seconds(value)
                if seconds is None or seconds <= 0:
                    refuse(group, option, value,
                           "not a DD_HH:MM:SS radiation interval, so no cadence can be "
                           "read from it",
                           f"{option} = 'HH:MM:SS'")
                else:
                    intervals[key] = seconds
            elif kind == "must_equal" and "only_with" in row:
                # MPAS reads it only with the scheme it configures (the MYNN
                # mixing switches, mpas_atmphys_driver_pbl.F); decided once
                # the selectors are resolved, below.
                conditional.append((group, option, value, row))
            elif kind == "must_equal":
                if _same(value, row["value"]):
                    dispositions.append(Disposition(group, option, value, "admitted",
                                                    "the value hex runs"))
                else:
                    refuse(group, option, value, row["reason"], row["follow_up"])
            elif kind == "stochastic":
                if type(value) is not bool:
                    refuse(group, option, value, "stochastic switches are logical", "true or false")
                else:
                    dispositions.append(Disposition(group, option, value, "admitted", row["reason"]))
            elif kind == "scheme_option":
                scheme_options.append((group, option, value, row))
            else:  # pragma: no cover - the table is validated
                raise AssertionError(kind)

    radiation_seconds = None
    if intervals:
        values = set(intervals.values())
        if len(values) != 1:
            refuse("physics", "config_radtlw_interval", dict(intervals),
                   "longwave and shortwave on different intervals: the column seams "
                   "call one radiation pair on one cadence, so one interval would be "
                   "honoured and the other recorded and never applied",
                   "the same config_radtlw_interval and config_radtsw_interval")
        else:
            radiation_seconds = values.pop()
            for key, seconds in intervals.items():
                dispositions.append(Disposition(
                    "physics", key, seconds, "admitted",
                    f"radiation every {seconds:g} s (the forecast door's "
                    "--radiation-seconds)"))

    # Scheme-specific metadata may map a native option to a carried engine
    # selector. Resolve it before constructing the profile, so the receipt
    # and executable configuration share the same selection.
    # A scheme_option row belongs to the scheme its ``slot`` selects
    # (default the microphysics); ``scheme`` names one scheme or a list of
    # the spellings MPAS accepts for it (bl_mynn and bl_mynnedmf).
    def _row_slot(row: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
        schemes = row["scheme"]
        if isinstance(schemes, str):
            schemes = (schemes,)
        return row.get("slot", "config_microp_scheme"), tuple(schemes)

    for group, option, value, row in scheme_options:
        slot, schemes = _row_slot(row)
        # A value equal to the row's declared engine default selects
        # nothing: carrying it would give the same physics a second
        # generation identity (and a synthesized backend row).
        if "default" in row and _same(value, row["default"]):
            continue
        if ("selector" in row and any(_same(value, v) for v in row["values"])
                and _effective_selector(slot, None, suite, overrides, mesh) in schemes):
            overrides[row["selector"]] = value
    physics = None
    try:
        physics = resolve_physics(suite, overrides, mesh=mesh)
    except PhysicsSelectionRefusal as error:
        for item in error.refusals:
            refusals.append(item)
            dispositions.append(Disposition("physics", item.knob, item.value, "refused", item.reason))

    for group, option, value, row in conditional:
        unmet = [
            (key, allowed, _effective_selector(key, physics, suite, overrides, mesh))
            for key, allowed in row["only_with"].items()
        ]
        unmet = [item for item in unmet if item[2] not in item[1]]
        if unmet:
            key, allowed, selected = unmet[0]
            dispositions.append(Disposition(
                group, option, value, "ignored",
                f"MPAS reads it only with {key} in {list(allowed)}, and this "
                f"run's is {selected!r}"))
        elif _same(value, row["value"]):
            dispositions.append(Disposition(group, option, value, "admitted",
                                            "the value hex runs"))
        else:
            refuse(group, option, value, row["reason"], row["follow_up"])

    for group, option, value, row in scheme_options:
        slot, schemes = _row_slot(row)
        selected = _effective_selector(slot, physics, suite, overrides, mesh)
        scheme = " or ".join(schemes)
        role = "microphysics" if slot == "config_microp_scheme" else slot
        if selected in schemes:
            if (("value" in row and _same(value, row["value"]))
                    or ("selector" in row and any(_same(value, v) for v in row["values"]))):
                dispositions.append(Disposition(group, option, value, "admitted",
                                                f"the selected {role} carries this value"))
            else:
                refuse(group, option, value,
                       row.get("reason", f"the selected {scheme} interface does not carry this option"),
                       row.get("follow_up", f"implement and verify this {role} option"))
        else:
            dispositions.append(Disposition(
                group, option, value, "ignored",
                f"configures {scheme}; MPAS reads it only when {scheme} is the "
                f"{role}, and this run's is {selected!r}"))
    from dataclasses import asdict
    from .stochastic import resolve_options
    stochastic_config = {}
    stochastic_on = any(values.get(name) is True for values in parsed.values()
                        for name in ("do_sppt", "do_skeb"))
    try:
        stochastic_config = asdict(resolve_options(namelist=parsed))
    except ValueError as error:
        refuse("nam_stochy", "stochastic_configuration", None, str(error),
               "supported spherical stochastic controls")
    for group, option, value in stochastic_seen:
        dispositions.append(Disposition(
            group, option, value, "admitted" if stochastic_on else "ignored",
            "GPU spherical stochastic control" if stochastic_on else
            "MPAS reads it only when do_sppt or do_skeb is true"))

    if soil_levels is not None:
        group, value = soil_levels
        land = _effective_selector("config_lsm_scheme", physics, suite, overrides, mesh)
        lsm = table["options"]["config_lsm_scheme"].get(land or "", {})
        if "carried" not in lsm:
            dispositions.append(Disposition(group, "num_soil_layers", value, "ignored",
                                            f"the land surface {land!r} is refused above"))
        else:
            wanted = int(lsm["carried"]["n_soil_levels"])
            if _same(value, wanted):
                dispositions.append(Disposition(group, "num_soil_layers", value, "admitted",
                                                "the land surface's own soil levels"))
            else:
                refuse(group, "num_soil_layers", value,
                       f"{land} integrates {wanted} soil "
                       "levels; a soil column of another size would be refused at the "
                       "start or read at the wrong depths",
                       f"num_soil_layers = {wanted}")

    dycore = None
    dycore_options: list[tuple[str, str, Any]] = []
    if dycore_text:
        from .config import dry_dycore_config_from_namelist

        try:
            dycore = dry_dycore_config_from_namelist(
                "\n".join(dycore_text), source_release="v8.4.1")
            for group, assignments in parsed.items():
                if group in dycore_groups:
                    iau_off = group == "iau" and str(
                        {k.casefold(): v for k, v in assignments.items()}.get(
                            "config_iau_option", "off")).strip().lower() == "off"
                    for option, value in assignments.items():
                        if iau_off and option.casefold() != "config_iau_option":
                            continue
                        if option.casefold() in _CALENDAR:
                            continue
                        if option.casefold() in dycore_ignored:
                            dispositions.append(Disposition(
                                group, option, value, "ignored",
                                dycore_ignored[option.casefold()]))
                            continue
                        dycore_options.append((group, option, value))
                        dispositions.append(Disposition(
                            group, option, value, "dycore",
                            "admitted by the v8.4.1 dycore namelist door; the "
                            "forecast door holds the run's configuration to it"))
        except ConfigurationRefusal as error:
            refusals.append(OptionRefusal(error.knob, error.value, error.reason,
                                          error.declaration))
            dispositions.append(Disposition("dycore", error.knob, error.value, "refused",
                                            error.reason))

    if defaulted:
        dispositions = [
            Disposition(item.group, item.option, item.value, item.status,
                        _default_note(registry_name) + item.detail)
            if item.option.casefold() in defaulted and item.status != "refused" else item
            for item in dispositions
        ]

    return NamelistPhysics(
        dispositions=tuple(dispositions),
        physics=physics if not refusals else None,
        refusals=tuple(refusals),
        radiation_seconds=radiation_seconds,
        dycore_config=dycore,
        suite=suite,
        overrides=dict(overrides),
        dycore_options=tuple(dycore_options),
        stochastic=stochastic_config,
    )


def _apply_registry_defaults(
    parsed: dict[str, dict[str, Any]], spec: Mapping[str, Any], physics_groups: set[str],
) -> tuple[frozenset[str], str | None]:
    """Write MPAS's Registry default of every option the namelist leaves out.

    THE BREAKAGE THIS PREVENTS, MEASURED (2026-10-07, hex suite review): a
    namelist with no ``config_physics_suite`` resolved to hex's default suite
    (WSM6, YSU, Noah-MP, the auto cumulus rule) where MPAS runs
    ``mesoscale_reference`` (New Tiedtke, Noah); an absent radiation
    interval ran hex's 600 s where MPAS runs 30 min; an absent
    ``config_frac_seaice`` ran the suite's switch where MPAS runs true.
    Every one ran with no refusal and a receipt naming the namelist.  The
    defaults are rows (``namelist.registry_defaults``), read here, in place:
    the dispositioning below then treats a defaulted option exactly as a
    written one.  A namelist naming a hex suite reads no registry -- that
    suite's own row decides, as at every other door.  Returns the options
    written and the registry they came from.
    """

    from .physics_suites import suite_names

    table = spec.get("registry_defaults")
    if table is None:
        return frozenset(), None
    present = {
        option.casefold()
        for group in physics_groups & set(parsed)
        for option in parsed[group]
    }
    named = next(
        (str(value) for group in physics_groups & set(parsed)
         for option, value in parsed[group].items()
         if option.casefold() == "config_physics_suite"),
        None,
    )
    if named is not None and named.strip() in suite_names():
        return frozenset(), None
    registry_name = (
        table["default_registry"] if named is None
        else table["by_suite"].get(named.strip(), table["default_registry"])
    )
    written: set[str] = set()
    for option, value in table["registries"][registry_name]["values"].items():
        if option.casefold() not in present:
            parsed.setdefault("physics", {})[option] = value
            written.add(option.casefold())
    return frozenset(written), registry_name


def _default_note(registry_name: str | None) -> str:
    return (f"not in the namelist, so MPAS's Registry default ({registry_name}), "
            "read as if written: ")


#: The forecast-door flag that carries each dynamics option a namelist can
#: set, for the refusal's remedy.  An option missing here has no flag: the
#: run holds the configuration type's own value.
_DYCORE_FLAGS = {
    "config_dt": "--dt",
    "config_time_integration_order": "--time-integration-order",
    "config_number_of_sub_steps": "--acoustic-substeps",
    "config_monotonic": "--monotonic",
    "config_horiz_mixing": "--horiz-mixing",
    "config_mpas_cam_coef": "--cam-coefficient",
    "config_number_cam_damping_levels": "--cam-damping-levels",
    "config_rayleigh_damp_u": "--rayleigh-u",
    "config_zd": "--damping-start-m",
    "config_apply_lbcs": "--lbc-dir (the grid's bdyMask decides it)",
}


def dycore_disagreements(imported: "NamelistPhysics", run_config: Any) -> list[str]:
    """Every dynamics option of the namelist the run would not execute.

    THE BREAKAGE THIS PREVENTS, MEASURED (2026-10-07, hex suite review):
    ``--physics-namelist`` took the physics of RRFS v2's own namelist and
    dispositioned its &nhyd_model and &damping options "admitted by the
    v8.4.1 dycore namelist door", and the forecast door then ran its own
    defaults -- RK3 not RK2, 6 acoustic substeps not 2, no monotonic
    limiter, no CAM, no Rayleigh, damping from 22 km not 16 km -- with
    nothing on the console or in the receipt saying so.  ``run_config`` is
    the configuration the driver builds for this request
    (``build_forecast_config``); an option it holds at another value is one
    line here.  An attribute the run's configuration type does not carry
    agrees only when the namelist asks for MPAS's own default of it.
    """

    from dataclasses import MISSING, fields

    from .config import _V841_DYCORE_BINDINGS

    config = imported.dycore_config
    if config is None:
        return []
    defaults = {
        item.name: item.default for item in fields(type(config))
        if item.default is not MISSING
    }
    problems: list[str] = []
    for group, option, value in imported.dycore_options:
        binding = _V841_DYCORE_BINDINGS.get(option.casefold())
        if binding is None:  # pragma: no cover - the dycore door bound it
            continue
        attribute = binding.attribute
        wanted = getattr(config, attribute)
        if hasattr(run_config, attribute):
            held = getattr(run_config, attribute)
        elif attribute in defaults:
            held = defaults[attribute]
        else:
            held = "(not carried)"
        if _dycore_equal(wanted, held):
            continue
        flag = _DYCORE_FLAGS.get(attribute)
        if flag is not None:
            remedy = f"give {flag} the namelist's value"
        elif hasattr(run_config, attribute):
            # The option travels as --dynamics; the two sources disagree.
            remedy = (
                f"--dynamics {attribute} names another value; drop it or "
                "give it the namelist's"
            )
        else:
            remedy = (
                "this run's configuration carries no such option; remove "
                "it from the namelist or leave it at MPAS's default"
            )
        problems.append(
            f"&{group} {option} = {value!r}: the run's configuration holds "
            f"{held!r}, so the namelist's dynamics would be recorded and not "
            f"run ({remedy})")
    return problems


def _dycore_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


def _effective_selector(key, physics, suite, overrides, mesh) -> str | None:
    """The value the namelist selects for one selector, refused or not."""

    if physics is not None:
        return physics.selectors[key]
    value = overrides.get(key)
    if value not in (None, "suite"):
        return str(value)
    from .physics_suites import _starting_point

    try:
        return _starting_point(suite, mesh)[2][key]
    except PhysicsSelectionRefusal:
        return None


def _render_group(group: str, assignments: list[tuple[str, Any]]) -> str:
    def literal(value: Any) -> str:
        if isinstance(value, bool):
            return ".true." if value else ".false."
        if isinstance(value, str):
            return "'" + value.replace("'", "''") + "'"
        return repr(value)

    body = "\n".join(f"    {option} = {literal(value)}" for option, value in assignments)
    return f"&{group}\n{body}\n/"


def read_physics_namelist(
    path: str | Path,
    *,
    mesh: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> NamelistPhysics:
    """Read one UTF-8 ``namelist.atmosphere`` and import its physics."""

    source = Path(path).expanduser().resolve(strict=True)
    return import_physics_namelist(
        source.read_text(encoding="utf-8"), mesh=mesh, overrides=overrides)
