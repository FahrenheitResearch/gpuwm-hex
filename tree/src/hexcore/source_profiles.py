"""Source profiles, as DATA: the outside model configurations hex is measured against.

A profile row pins one outside configuration: the model source commits (with
their physics submodules), the analysis code, the dated mesh files, the
vertical level file, the namelist, and how hex stands against each dycore
setting that namelist asks for.  It also carries the audit of the outside
model's dycore against the upstream release hex is a port of, each
difference classified at that configuration's settings.

THE TABLE IS THE INTERFACE.  Adding or re-pinning an outside configuration is
an edit to ``data/profiles/source-profiles.v1.json`` plus the stage proofs.
No code path keys on a row id, and nothing here selects a kernel: the rows
are the record a claim is checked against, and the tests hold each setting
row to the tree per hex lane: the refusal that actually fires on that lane
(knob, reason, file and function) and every refusal behind it a port must
also lift.  Lifting any of them fails the tests until the row moves too.

This module only loads and validates the table.  It refuses a malformed
table by name, because a row with a short or misspelled commit would pin
nothing while looking like a pin.
"""

from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .errors import MpasPortError

__all__ = [
    "AUDIT_CLASSES",
    "LANE_STATES",
    "SCHEMA",
    "SETTING_STATES",
    "SourceProfileError",
    "load_source_profiles",
    "source_profile",
]

SCHEMA = "gpuwm-hex.source-profiles/v1"

#: How a dycore difference is classified at its profile's own settings.
#: ``inert``: the outside model's arithmetic at those settings is the pinned
#: release's.  ``port``: matching it needs new code here.  ``declare``: it
#: stays a declared difference (rounding order, or outside the dycore).  A
#: trailing ``(VERIFY)`` marks an inert call that waits on a value in the
#: outside model's files.
AUDIT_CLASSES = frozenset(
    {
        "inert",
        "inert (VERIFY)",
        "port",
        "declare",
    }
)

#: Where hex stands against one requested dycore setting today, summed over
#: its lanes: ``refused`` exactly when some lane refuses it.
SETTING_STATES = frozenset({"admitted", "refused"})

#: Where one hex lane stands against it.  ``no_field``: the lane has no such
#: setting at all (its door takes the value from elsewhere, or the setting
#: has no meaning there yet).
LANE_STATES = frozenset({"admitted", "refused", "no_field"})

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SourceProfileError(MpasPortError):
    """The source profile table violates its schema."""


def _default_table_text() -> str:
    return (
        resources.files("hexcore")
        .joinpath("data", "profiles", "source-profiles.v1.json")
        .read_text(encoding="utf-8")
    )


def _walk_commits(node: Any, where: str) -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            here = f"{where}.{key}"
            if key == "commit":
                if not isinstance(value, str) or not _COMMIT.match(value):
                    raise SourceProfileError(
                        f"{here} must be a full 40-character lowercase commit, got {value!r}"
                    )
            elif key.endswith("sha256") and value is not None:
                if not isinstance(value, str) or not _SHA256.match(value):
                    raise SourceProfileError(
                        f"{here} must be a 64-character lowercase SHA-256, got {value!r}"
                    )
            else:
                _walk_commits(value, here)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _walk_commits(value, f"{where}[{index}]")


def _validate_setting(
    ident: str, setting: Mapping[str, Any], lanes: Mapping[str, Any]
) -> None:
    where = f"profile {ident!r} setting {setting.get('knob')!r}"
    if setting.get("mesh"):
        where += f" ({setting['mesh']})"
    summary = setting.get("hex_today")
    if summary not in SETTING_STATES:
        raise SourceProfileError(f"{where}: hex_today must be one of {sorted(SETTING_STATES)}")
    per_lane = setting.get("lanes")
    if not isinstance(per_lane, Mapping):
        raise SourceProfileError(f"{where}: needs a 'lanes' mapping")
    if setting.get("kind") != "declaration" and set(per_lane) != set(lanes):
        raise SourceProfileError(
            f"{where}: names lanes {sorted(per_lane)}, the profile declares {sorted(lanes)}"
        )
    refused_anywhere = False
    for lane, entry in per_lane.items():
        state = entry.get("state") if isinstance(entry, Mapping) else None
        if state not in LANE_STATES:
            raise SourceProfileError(
                f"{where} lane {lane!r}: state must be one of {sorted(LANE_STATES)}"
            )
        if state == "refused":
            refused_anywhere = True
            for key in ("reason", "site", "function"):
                if not isinstance(entry.get(key), str) or not entry[key]:
                    raise SourceProfileError(f"{where} lane {lane!r}: a refusal needs {key!r}")
            if entry.get("match", "exact") not in ("exact", "prefix"):
                raise SourceProfileError(f"{where} lane {lane!r}: match is 'exact' or 'prefix'")
        elif entry.get("reason") or entry.get("site"):
            raise SourceProfileError(f"{where} lane {lane!r} is {state} but names a refusal")
    if (summary == "refused") != refused_anywhere:
        raise SourceProfileError(
            f"{where}: hex_today {summary!r} disagrees with its lanes; it is "
            "'refused' exactly when some lane refuses"
        )
    for layer in setting.get("further_layers", []):
        if layer.get("lane") not in lanes:
            raise SourceProfileError(f"{where}: a further layer names unknown lane {layer.get('lane')!r}")
        for key in ("site", "function", "reason"):
            if not isinstance(layer.get(key), str) or not layer[key]:
                raise SourceProfileError(f"{where}: a further layer needs {key!r}")


def _validate(table: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if not isinstance(table, Mapping) or table.get("schema") != SCHEMA:
        raise SourceProfileError(f"source profile table schema must be {SCHEMA!r}")
    rows = table.get("profiles")
    if not isinstance(rows, list) or not rows:
        raise SourceProfileError("source profile table needs a non-empty 'profiles' list")
    out: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SourceProfileError(f"profiles[{index}] must be an object")
        ident = row.get("id")
        if not isinstance(ident, str) or not ident:
            raise SourceProfileError(f"profiles[{index}] needs a string id")
        if ident in out:
            raise SourceProfileError(f"profile id {ident!r} appears twice")
        for key in ("describes", "role"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise SourceProfileError(f"profile {ident!r} needs a {key!r} string")
        _walk_commits(row, ident)
        candidates = row.get("model_candidates", [])
        candidate_ids = {c.get("id") for c in candidates}
        if len(candidate_ids) != len(candidates):
            raise SourceProfileError(f"profile {ident!r} repeats a model candidate id")
        lanes = row.get("hex_lanes", {})
        for setting in row.get("hex_dycore_settings", []):
            _validate_setting(ident, setting, lanes)
        audit = row.get("dycore_audit")
        if audit is not None:
            for audit_row in audit.get("rows", []):
                if audit_row.get("class") not in AUDIT_CLASSES:
                    raise SourceProfileError(
                        f"profile {ident!r} audit item {audit_row.get('item')!r}: "
                        f"class must be one of {sorted(AUDIT_CLASSES)}"
                    )
                if audit_row.get("candidate") not in candidate_ids:
                    raise SourceProfileError(
                        f"profile {ident!r} audit item {audit_row.get('item')!r} "
                        "names a candidate the row does not pin"
                    )
        out[ident] = MappingProxyType(dict(row))
    return out


def load_source_profiles(path: str | Path | None = None) -> Mapping[str, Mapping[str, Any]]:
    """Load and validate the table; the packaged table when ``path`` is None."""

    try:
        text = (
            _default_table_text()
            if path is None
            else Path(path).expanduser().read_text(encoding="utf-8")
        )
        table = json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceProfileError(f"cannot read the source profile table: {error}") from error
    return MappingProxyType(_validate(table))


def source_profile(ident: str, path: str | Path | None = None) -> Mapping[str, Any]:
    """One profile row by id; refuses an id the table does not carry."""

    profiles = load_source_profiles(path)
    try:
        return profiles[ident]
    except KeyError:
        raise SourceProfileError(
            f"no source profile {ident!r}; the table carries {sorted(profiles)}"
        ) from None
