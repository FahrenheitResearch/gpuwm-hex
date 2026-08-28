"""Canonical dry configuration and fail-closed Fortran namelist ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, TypeAlias

import numpy as np

from .driver import DryDycoreConfig
from .errors import ConfigurationRefusal


NamelistScalar: TypeAlias = bool | int | float | str
ParsedNamelist: TypeAlias = Mapping[str, Mapping[str, NamelistScalar]]

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INTEGER = re.compile(r"^(?P<number>[+-]?\d+)(?:_[A-Za-z0-9_]+)?$")
_REAL = re.compile(
    r"^(?P<number>[+-]?(?:(?:\d+\.\d*|\.\d+)"
    r"(?:[EeDdQq][+-]?\d+)?|\d+[EeDdQq][+-]?\d+))"
    r"(?:_[A-Za-z0-9_]+)?$"
)


def _refuse(knob: str, value: object, reason: str, declaration: str) -> None:
    raise ConfigurationRefusal(knob, value, reason, declaration)


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str
    text: str
    line: int
    column: int


def _lex_namelist(text: str) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    index = 0
    line = 1
    column = 1
    length = len(text)

    def advance(character: str) -> None:
        nonlocal line, column
        if character == "\n":
            line += 1
            column = 1
        else:
            column += 1

    while index < length:
        character = text[index]
        if character in " \t\f\v\r":
            advance(character)
            index += 1
            continue
        if character == "\n":
            tokens.append(_Token("newline", character, line, column))
            advance(character)
            index += 1
            continue
        if character == "!":
            while index < length and text[index] != "\n":
                advance(text[index])
                index += 1
            continue
        if character in "'\"":
            quote = character
            start_line, start_column = line, column
            index += 1
            advance(quote)
            value: list[str] = []
            while index < length:
                character = text[index]
                if character == quote:
                    if index + 1 < length and text[index + 1] == quote:
                        value.append(quote)
                        advance(quote)
                        advance(quote)
                        index += 2
                        continue
                    advance(quote)
                    index += 1
                    break
                value.append(character)
                advance(character)
                index += 1
            else:
                _refuse(
                    "namelist_syntax",
                    f"line {start_line}, column {start_column}",
                    "an unterminated quoted string was found",
                    "a closed Fortran character literal",
                )
            tokens.append(
                _Token("string", "".join(value), start_line, start_column)
            )
            continue
        if character in "&$":
            start_line, start_column = line, column
            advance(character)
            index += 1
            start = index
            while index < length and (
                text[index].isalnum() or text[index] == "_"
            ):
                advance(text[index])
                index += 1
            name = text[start:index]
            if not name or not _IDENTIFIER.fullmatch(name):
                _refuse(
                    "namelist_syntax",
                    f"line {start_line}, column {start_column}",
                    "a group delimiter is not followed by a Fortran identifier",
                    "&group_name",
                )
            tokens.append(_Token("group", name, start_line, start_column))
            continue
        punctuation = {"=": "equals", ",": "comma", "/": "end"}
        if character in punctuation:
            tokens.append(_Token(punctuation[character], character, line, column))
            advance(character)
            index += 1
            continue

        start_line, start_column = line, column
        start = index
        while index < length and text[index] not in " \t\f\v\r\n!&$=,/\"'":
            advance(text[index])
            index += 1
        if start == index:
            _refuse(
                "namelist_syntax",
                f"line {line}, column {column}",
                f"unsupported character {text[index]!r}",
                "Fortran namelist scalar syntax",
            )
        tokens.append(_Token("atom", text[start:index], start_line, start_column))

    return tuple(tokens)


def _scalar(token: _Token, knob: str) -> NamelistScalar:
    if token.kind == "string":
        return token.text
    if token.kind != "atom":
        _refuse(
            knob,
            token.text,
            "the namelist value is missing or is not scalar",
            f"one scalar value for {knob}",
        )

    folded = token.text.casefold()
    logicals = {
        ".true.": True,
        ".t.": True,
        "true": True,
        "t": True,
        ".false.": False,
        ".f.": False,
        "false": False,
        "f": False,
    }
    if folded in logicals:
        return logicals[folded]
    integer = _INTEGER.fullmatch(token.text)
    if integer is not None:
        return int(integer.group("number"), 10)
    real = _REAL.fullmatch(token.text)
    if real is not None:
        return float(
            real.group("number").replace("D", "e").replace("d", "e").replace("Q", "e").replace("q", "e")
        )
    _refuse(
        knob,
        token.text,
        "only logical, integer, real, and quoted character scalars are admitted",
        f"one supported scalar value for {knob}",
    )


def parse_fortran_namelist(text: str) -> ParsedNamelist:
    """Parse scalar Fortran namelist records without silently dropping syntax.

    Identifiers are case-insensitive. Group and knob insertion order is
    retained, while duplicate groups/knobs and all non-scalar values refuse.
    """

    if not isinstance(text, str):
        _refuse(
            "namelist",
            type(text).__name__,
            "the parser input must be text",
            "a str containing Fortran namelist records",
        )
    tokens = _lex_namelist(text)
    groups: dict[str, Mapping[str, NamelistScalar]] = {}
    index = 0

    def skip_separators(position: int) -> int:
        while position < len(tokens) and tokens[position].kind in {
            "newline",
            "comma",
        }:
            position += 1
        return position

    index = skip_separators(index)
    while index < len(tokens):
        opener = tokens[index]
        if opener.kind != "group" or opener.text.casefold() == "end":
            _refuse(
                "namelist_syntax",
                opener.text,
                f"line {opener.line} is outside a named namelist group",
                "&group_name ... /",
            )
        group_key = opener.text.casefold()
        if group_key in groups:
            _refuse(
                opener.text,
                "duplicate group",
                "duplicate namelist records are ambiguous in this port contract",
                f"one &{opener.text} record",
            )
        index += 1
        assignments: dict[str, NamelistScalar] = {}
        seen_names: set[str] = set()
        closed = False
        while index < len(tokens):
            index = skip_separators(index)
            if index >= len(tokens):
                break
            token = tokens[index]
            if token.kind == "end" or (
                token.kind == "group" and token.text.casefold() == "end"
            ):
                index += 1
                closed = True
                break
            if token.kind != "atom" or not _IDENTIFIER.fullmatch(token.text):
                _refuse(
                    token.text if token.kind == "atom" else "namelist_syntax",
                    token.text,
                    f"line {token.line} does not begin a scalar assignment",
                    "config_name = scalar_value",
                )
            knob = token.text
            key = knob.casefold()
            if key in seen_names:
                _refuse(
                    knob,
                    "duplicate assignment",
                    "duplicate knob assignments are ambiguous in this port contract",
                    f"one assignment for {knob}",
                )
            if index + 1 >= len(tokens) or tokens[index + 1].kind != "equals":
                _refuse(
                    knob,
                    token.text,
                    f"line {token.line} omits '=' after the knob name",
                    f"{knob} = scalar_value",
                )
            if index + 2 >= len(tokens):
                _refuse(
                    knob,
                    None,
                    "the namelist ends before the scalar value",
                    f"{knob} = scalar_value",
                )
            value_token = tokens[index + 2]
            value = _scalar(value_token, knob)
            assignments[knob] = value
            seen_names.add(key)
            index += 3

            probe = skip_separators(index)
            if probe < len(tokens):
                following = tokens[probe]
                starts_assignment = (
                    following.kind == "atom"
                    and _IDENTIFIER.fullmatch(following.text) is not None
                    and probe + 1 < len(tokens)
                    and tokens[probe + 1].kind == "equals"
                )
                ends_group = following.kind == "end" or (
                    following.kind == "group"
                    and following.text.casefold() == "end"
                )
                if not starts_assignment and not ends_group:
                    _refuse(
                        knob,
                        following.text,
                        "array, repetition, complex, and multi-value assignments are not admitted",
                        f"one scalar value for {knob}",
                    )
            index = probe

        if not closed:
            _refuse(
                opener.text,
                "unterminated group",
                f"&{opener.text} has no '/' or &end terminator",
                f"&{opener.text} ... /",
            )
        groups[group_key] = MappingProxyType(assignments)
        index = skip_separators(index)

    if not groups:
        _refuse(
            "namelist",
            "empty",
            "no Fortran namelist group was found",
            "at least one &group_name ... / record",
        )
    return MappingProxyType(groups)


@dataclass(frozen=True, slots=True)
class _KnobBinding:
    group: str
    attribute: str


def _bindings(group: str, *attributes: str) -> dict[str, _KnobBinding]:
    return {
        attribute.casefold(): _KnobBinding(group=group, attribute=attribute)
        for attribute in attributes
    }


_DRY_DYCORE_BINDINGS: Mapping[str, _KnobBinding] = MappingProxyType(
    _bindings(
        "nhyd_model",
        "config_time_integration_order",
        "config_dt",
        "config_split_dynamics_transport",
        "config_number_of_sub_steps",
        "config_dynamics_split_steps",
        "config_h_mom_eddy_visc2",
        "config_h_mom_eddy_visc4",
        "config_v_mom_eddy_visc2",
        "config_h_theta_eddy_visc2",
        "config_h_theta_eddy_visc4",
        "config_v_theta_eddy_visc2",
        "config_horiz_mixing",
        "config_len_disp",
        "config_visc4_2dsmag",
        "config_del4u_div_factor",
        "config_scalar_adv_order",
        "config_scalar_vadv_order",
        "config_scalar_advection",
        "config_positive_definite",
        "config_monotonic",
        "config_coef_3rd_order",
        "config_smagorinsky_coef",
        "config_epssm",
        "config_smdiv",
        "config_apvm_upwinding",
        "config_h_ScaleWithMesh",
    )
    | _bindings(
        "damping",
        "config_zd",
        "config_xnutr",
        "config_mpas_cam_coef",
        "config_rayleigh_damp_u",
    )
    | _bindings("limited_area", "config_apply_lbcs")
    | {"config_iau_option": _KnobBinding("iau", "config_iau_option")}
    | _bindings("physics", "config_physics_suite")
)
_V841_DYCORE_BINDINGS: Mapping[str, _KnobBinding] = MappingProxyType(
    dict(_DRY_DYCORE_BINDINGS)
    | _bindings(
        "nhyd_model",
        "config_les_model",
        "config_les_surface",
        "config_surface_heat_flux",
        "config_surface_moisture_flux",
        "config_surface_drag_coefficient",
        "config_mix_scalars",
    )
    | _bindings(
        "damping",
        "config_epssm_minimum",
        "config_epssm_maximum",
        "config_epssm_transition_bottom_z",
        "config_epssm_transition_top_z",
    )
    | _bindings("development", "config_gpu_aware_mpi")
)


_KNOWN_UNSUPPORTED: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        name.casefold(): (name, reason)
        for name, reason in {
            "config_time_integration": "the alternative integrator selector is not linked",
            "config_calendar_type": "calendar and alarm scheduling are outside DryDycoreConfig",
            "config_start_time": "calendar and alarm scheduling are outside DryDycoreConfig",
            "config_stop_time": "calendar and alarm scheduling are outside DryDycoreConfig",
            "config_run_duration": "calendar and alarm scheduling are outside DryDycoreConfig",
            "config_w_adv_order": "the independent w horizontal-order knob is not linked",
            "config_theta_adv_order": "the independent theta horizontal-order knob is not linked",
            "config_u_vadv_order": "the independent u vertical-order knob is not linked",
            "config_w_vadv_order": "the independent w vertical-order knob is not linked",
            "config_theta_vadv_order": "the independent theta vertical-order knob is not linked",
            "config_mix_full": "the alternate perturbation-only mixing branch is not linked",
            "config_num_halos": "halo allocation is outside the serial dry driver",
            "config_pio_num_iotasks": "PIO topology is outside DryDycoreConfig",
            "config_pio_stride": "PIO topology is outside DryDycoreConfig",
            "config_block_decomp_file_prefix": "decomposition is outside DryDycoreConfig",
            "config_do_restart": "restart orchestration is outside DryDycoreConfig",
            "config_print_global_minmax_vel": "printout controls are outside DryDycoreConfig",
            "config_print_detailed_minmax_vel": "printout controls are outside DryDycoreConfig",
            "config_sst_update": "the production physics update is not linked",
            "config_sstdiurn_update": "the production physics update is not linked",
            "config_deepsoiltemp_update": "the production physics update is not linked",
            "config_bucket_update": "the production physics update is not linked",
            "config_sounding_interval": "sounding output is outside DryDycoreConfig",
        }.items()
    }
)


def _coerce_binding(
    name: str,
    value: NamelistScalar,
    attribute: str,
    *,
    config_type: type[DryDycoreConfig] = DryDycoreConfig,
) -> object:
    default = getattr(config_type(), attribute)
    if isinstance(default, (bool, np.bool_)):
        if not isinstance(value, bool):
            _refuse(name, value, "the MPAS Registry type is logical", f"{name}=true or false")
        return value
    if isinstance(default, int) and not isinstance(default, bool):
        if not isinstance(value, int) or isinstance(value, bool):
            _refuse(name, value, "the MPAS Registry type is integer", f"an integer {name}")
        return value
    if isinstance(default, float):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            _refuse(name, value, "the MPAS Registry type is real", f"a real {name}")
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            _refuse(name, value, "the MPAS Registry type is character", f"a quoted {name}")
        return value
    _refuse(
        name,
        value,
        f"{attribute} has no admitted scalar namelist projection",
        f"a supported DryDycoreConfig mapping for {name}",
    )


def dry_dycore_config_from_namelist(
    text: str,
    *,
    source_release: str = "v8.2.3",
) -> DryDycoreConfig:
    """Map a scalar atmosphere namelist to the canonical dry config.

    Every encountered knob must have an explicit admitted binding. Known
    framework/physics options and unknown options refuse by their MPAS names.
    """

    if source_release == "v8.2.3":
        config_type: type[DryDycoreConfig] = DryDycoreConfig
        bindings = _DRY_DYCORE_BINDINGS
    elif source_release == "v8.4.1":
        # Lazy import avoids the config_v841 -> driver -> config import cycle.
        from .config_v841 import V841DryDycoreConfig

        config_type = V841DryDycoreConfig
        bindings = _V841_DYCORE_BINDINGS
    else:
        _refuse(
            "source_release",
            source_release,
            "only source-pinned v8.2.3 and v8.4.1 namelist schemas are admitted",
            "source_release='v8.2.3' or 'v8.4.1'",
        )
    admitted_groups = frozenset(binding.group for binding in bindings.values())
    parsed = parse_fortran_namelist(text)
    values: dict[str, object] = {}
    seen_attributes: set[str] = set()
    for group, assignments in parsed.items():
        if group not in admitted_groups and not assignments:
            _refuse(
                group,
                "empty namelist group",
                "this namelist group has no DryDycoreConfig binding",
                f"remove &{group} or port its behavior explicitly",
            )
        for original_name, value in assignments.items():
            key = original_name.casefold()
            binding = bindings.get(key)
            if binding is None:
                unsupported = _KNOWN_UNSUPPORTED.get(key)
                canonical = original_name if unsupported is None else unsupported[0]
                reason = (
                    "this MPAS namelist option has no DryDycoreConfig binding"
                    if unsupported is None
                    else unsupported[1]
                )
                _refuse(
                    canonical,
                    value,
                    reason,
                    f"remove {canonical} or port its behavior explicitly",
                )
            if binding.group != group:
                _refuse(
                    original_name,
                    value,
                    f"the MPAS Registry places this knob in &{binding.group}",
                    f"&{binding.group} {original_name} = ... /",
                )
            if binding.attribute in seen_attributes:
                _refuse(
                    original_name,
                    value,
                    "multiple namelist spellings map to the same canonical knob",
                    f"one assignment for {binding.attribute}",
                )
            values[binding.attribute] = _coerce_binding(
                original_name,
                value,
                binding.attribute,
                config_type=config_type,
            )
            seen_attributes.add(binding.attribute)

    mixing = values.get("config_horiz_mixing")
    if mixing == "2d_smagorinsky":
        # These are the frozen v8.2.3 Registry defaults for the active branch.
        values.setdefault("config_visc4_2dsmag", 0.05)
        values.setdefault("config_smagorinsky_coef", 0.125)
    if "config_smdiv" in values:
        values["config_divergence_damping"] = bool(values["config_smdiv"] != 0.0)
    return config_type.from_mapping(values)


def read_dry_dycore_config(
    path: str | Path,
    *,
    source_release: str = "v8.2.3",
) -> DryDycoreConfig:
    """Read and map one UTF-8 ``namelist.atmosphere`` file."""

    source = Path(path).expanduser().resolve(strict=True)
    return dry_dycore_config_from_namelist(
        source.read_text(encoding="utf-8"),
        source_release=source_release,
    )


__all__ = [
    "DryDycoreConfig",
    "NamelistScalar",
    "ParsedNamelist",
    "dry_dycore_config_from_namelist",
    "parse_fortran_namelist",
    "read_dry_dycore_config",
]
