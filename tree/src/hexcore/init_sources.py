"""One table-driven path from any initial-condition source to a hex init.

Every source hex can start from is a row of ``data/init-sources.v1.json``
or the generated-model extension ``earth2_init_rows.json``.
The path is the same for all of them:

1. **acquire** the row's declared products (whole files, or the records an
   index selects, optionally a member's records only), or take them from a
   staging directory, or generate the row's forecast through the shared
   Earth-2 inference runner;
2. **decode** them into ONE WPS intermediate through the Rust GRIB writer
   or the engine's mapped composer, as declared by the row. Generated forecasts
   use the shared Rust NetCDF composition and WPS export instead;
3. **init** through the source-agnostic init door (:mod:`hexcore.init_door`)
   with the row's first-guess declarations and the table's run switches;
4. optionally **order** the init for a mesh case and write a case JSON that
   names it.

There is no branch on a source id anywhere in this module.  Adding a model
is a table row and, when its parameter codes are new, a Vtable: the
arbitrary acceptance test (a project ruling, 2026-08-16).  A row may declare
``status: refused`` with a ``refusal`` sentence when its bytes cannot travel
this path; asking for it refuses with that sentence.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .errors import MpasPortError

TABLE_SCHEMA = "gpuwm-hex.init-sources/v1"
TABLE_PATH = Path(__file__).with_name("data") / "init-sources.v1.json"
GENERATED_TABLE_PATH = Path(__file__).with_name("earth2_init_rows.json")
#: Vtable paths in the table are relative to the package's ``data/``
#: directory, where ``vtables/`` lives and from where the wheel and the sdist
#: carry it.  THE BREAKAGE THIS PREVENTS, measured on a clean pip install of
#: the 0.4.0 candidate (g16 fresh-install QA, 2026-10-12): the root was the
#: REPOSITORY root (``parents[3]`` of this file), the Vtables lived outside
#: ``tree/`` and shipped in no artifact, so every row refused
#: "Vtable <venv>/lib/vtables/Vtable.GFS does not exist" after its download.
VTABLE_ROOT = Path(__file__).resolve().with_name("data")
RECEIPT_SCHEMA = "gpuwm-hex.init-source-receipt/v1"

_ROW_KEYS = {"id", "aliases", "engine_source_id", "display_name", "role", "license", "grid",
             "products", "decode", "vertical", "first_guess", "surface", "limits", "members",
             "status", "refusal", "sets", "remap", "archive", "posting", "forecast", "surface_series",
             "generation", "attribution", "coverage", "intermediate"}
_PRODUCT_KEYS = {"role", "transport", "filename", "url", "index", "cds", "codec", "fields", "reason", "fetch_args",
                 "id", "expand", "remap", "streams", "request"}
_REMAP_KEYS = {"tool", "mesh_cells", "centre", "target", "coordinates", "about"}
_REMAP_METHODS = {"idw4", "surface-idw4", "nearest", "soil-nearest", "seaice-nearest"}
_CODECS = {"bz2"}
_TRANSPORTS = {"https", "cds", "engine-fetch", "zarr"}
_INDEX_FORMATS = {"ecmwf-jsonl", "ncep-idx"}
#: Parallel range requests a product takes when neither ``--jobs`` nor the
#: product's own ``streams`` says otherwise (open-data hosts that throttle
#: bursts answer 503 Slow Down, which the retry ladder absorbs).
DEFAULT_STREAMS = 3
# The inference runner derives its Torch/BLAS pools from CPU affinity, ignoring
# OMP_NUM_THREADS. Bound its child affinity so a container exposing more CPUs
# than its allocation cannot create hundreds of competing loader threads.
DEFAULT_GENERATION_WORKERS = 14
#: Largest single range request.  A selected or whole object is cut into
#: pieces of this size so every stream stays busy; the bytes written are the
#: same bytes in the same order whatever the piece size.
RANGE_PIECE_BYTES = 4 << 20
#: Where a source's meteorology exists.  A ``limited_area`` row (a projected
#: regional product) initialises a limited-area cull through the regional door
#: and its intermediate row; it never initialises a global mesh, which would
#: have no meteorology outside the source's domain.
COVERAGES = ("global", "limited_area")
_RUN_SWITCHES = ("extrap_airtemp", "theta_adv_order", "coef_3rd_order", "virtual_factor",
                 "deep_soil_moisture", "landuse_table", "frac_seaice", "tsk_seaice_threshold",
                 "oned_underflow")


class InitSourceRefusal(MpasPortError):
    """A named refusal: what would go wrong, then the remedy."""


def _refuse(message: str) -> None:
    raise InitSourceRefusal(message)


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------
def load_table(path: Path | None = None) -> dict[str, Any]:
    """The source table, schema-checked; every row is validated on load."""

    table_path = Path(path) if path is not None else TABLE_PATH
    table = json.loads(table_path.read_text(encoding="utf-8"))
    if table.get("schema") != TABLE_SCHEMA:
        _refuse(f"{table_path} declares schema {table.get('schema')!r}, not {TABLE_SCHEMA}; "
                "a reader of another schema would misread its rows")
    missing = [key for key in _RUN_SWITCHES if key not in table.get("run_switches", {})]
    if missing:
        _refuse(f"{table_path} run_switches lacks {', '.join(missing)}; the init door has no default "
                "for them and two sources would be initialised under different run settings")
    # Extension rows leave the shared physical-source registry append-only.
    # Explicit alternate tables are self-contained (including test fixtures).
    if path is None:
        generated = json.loads(GENERATED_TABLE_PATH.read_text(encoding="utf-8"))
        if generated.get("schema") != "gpuwm-hex.earth2-init-rows/v1":
            _refuse(f"{GENERATED_TABLE_PATH}: unknown generated-source schema")
        table["sources"].extend(generated["sources"])
        table["_extensions"] = [{"path": str(GENERATED_TABLE_PATH),
                                 "sha256": _sha256_file(GENERATED_TABLE_PATH)}]
    seen: dict[str, str] = {}
    for row in table["sources"]:
        validate_row(row)
        for name in (row["id"], *row.get("aliases", ())):
            key = _norm(name)
            if key in seen:
                _refuse(f"source name {name!r} is claimed by both {seen[key]!r} and {row['id']!r}; "
                        "a request by that name would pick one silently")
            seen[key] = row["id"]
    table["_path"] = str(table_path)
    table["_sha256"] = hashlib.sha256(table_path.read_bytes()).hexdigest()
    return table


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def validate_row(row: dict[str, Any]) -> None:
    unknown = set(row) - _ROW_KEYS
    if unknown:
        _refuse(f"source row {row.get('id')!r} carries unknown keys {sorted(unknown)}; a misspelt key "
                "would be ignored and its declaration lost")
    posting = row.get("posting")
    if posting is not None:
        hours = posting.get("cycle_hours")
        if (not isinstance(hours, list) or not hours
                or any(type(h) is not int or not 0 <= h <= 23 for h in hours)):
            _refuse(f"source row {row.get('id')!r}: posting.cycle_hours must list the UTC hours (0-23) the "
                    "source runs; `--cycle latest` would otherwise pick a cycle that never exists")
        if not isinstance(posting.get("lag_minutes"), (int, float)) or posting["lag_minutes"] < 0:
            _refuse(f"source row {row.get('id')!r}: posting.lag_minutes must be the measured minutes from "
                    "cycle time to the step-0 products answering")
    if row.get("status") == "refused":
        if not str(row.get("refusal", "")).strip():
            _refuse(f"source row {row['id']!r} is refused without naming the breakage")
        return
    first_guess = row.get("first_guess", {})
    for key in ("nfglevels", "nfgsoillevels", "use_spechumd"):
        if key not in first_guess:
            _refuse(f"{row['id']}: first_guess.{key} is required; the init door refuses without it")
    if first_guess["use_spechumd"] not in ("yes", "no"):
        _refuse(f"{row['id']}: first_guess.use_spechumd must be yes or no for the init door")
    generation = row.get("generation")
    if generation is not None:
        if generation.get("backend") != "earth2studio":
            _refuse(f"{row['id']}: unknown generation backend")
        for key in ("model", "profile", "channels", "bindings", "weights_revision"):
            if not generation.get(key):
                _refuse(f"{row['id']}: generation.{key} is required")
        step = generation.get("step_hours")
        spinup = generation.get("spinup_hours")
        if type(step) is not int or step < 1 or type(spinup) is not int or spinup < step or spinup % step:
            _refuse(f"{row['id']}: generation needs positive whole forecast steps")
        if [binding.get("channel") for binding in generation["bindings"]] != generation["channels"]:
            _refuse(f"{row['id']}: every output channel needs one explicit hex binding or exclusion")
        for key in ("decode", "first_guess", "license"):
            if not row.get(key):
                _refuse(f"{row['id']}: {key} is required")
        donor = generation.get("global_donor")
        if donor is not None and (donor.get("area") != [-90, -180, 90, 180]
                or any(not donor.get(key) for key in ("provider", "acquisition_receipt", "data_role"))):
            _refuse(f"{row['id']}: global_donor must declare the complete global extent and acquisition role")
        return
    coverage = row.get("coverage", "global")
    if coverage not in COVERAGES:
        _refuse(f"source row {row.get('id')!r}: coverage {coverage!r} is not one of {list(COVERAGES)}")
    if coverage == "limited_area":
        # A projected regional source travels its intermediate row, not the
        # Vtable path, so it declares that row instead of products and decode.
        if not str((row.get("intermediate") or {}).get("row", "")).strip():
            _refuse(f"source row {row.get('id')!r} covers a limited area and names no intermediate.row; "
                    "the regional door has no way to write its WPS intermediates")
        return
    for key in ("products", "decode", "first_guess"):
        if not row.get(key):
            _refuse(f"source row {row.get('id')!r} declares no {key}; the path cannot run without it")
    for product in row["products"]:
        unknown = set(product) - _PRODUCT_KEYS
        if unknown:
            _refuse(f"{row['id']}: product keys {sorted(unknown)} are unknown")
        if product.get("transport") not in _TRANSPORTS:
            _refuse(f"{row['id']}: transport {product.get('transport')!r} is not one of {sorted(_TRANSPORTS)}")
        if product["transport"] == "https" and not product.get("url"):
            _refuse(f"{row['id']}: an https product needs a url template")
        if product["transport"] == "cds" and not product.get("cds", {}).get("dataset"):
            _refuse(f"{row['id']}: a cds product needs cds.dataset")
        if product["transport"] == "engine-fetch":
            args = product.get("fetch_args")
            if not isinstance(args, list) or not args or not all(isinstance(a, str) for a in args):
                _refuse(f"{row['id']}: an engine-fetch product needs a nonempty fetch_args string list")
        if product["transport"] == "zarr":
            request = product.get("request", {})
            if not request.get("store") or not request.get("fields"):
                _refuse(f"{row['id']}: a zarr product needs a native request with store and fields")
        if product.get("codec") is not None and product["codec"] not in _CODECS:
            _refuse(f"{row['id']}: codec {product['codec']!r} is not one of {sorted(_CODECS)}")
        if product.get("expand") is not None and product["expand"] not in row.get("sets", {}):
            _refuse(f"{row['id']}: product expands over set {product['expand']!r}, which the row does not declare")
        if product.get("remap") is not None:
            if "remap" not in row:
                _refuse(f"{row['id']}: a product declares a remap but the row declares no remap stage")
            if product["remap"].get("method") not in _REMAP_METHODS:
                _refuse(f"{row['id']}: remap method {product['remap'].get('method')!r} is not one of "
                        f"{sorted(_REMAP_METHODS)}")
            if len(str(product["remap"].get("selector", "")).split(",")) != 7:
                _refuse(f"{row['id']}: a remap selector is seven comma-separated values (discipline, "
                        "category, parameter, first level type and value, second level type and value)")
        index = product.get("index")
        if index is not None and index.get("format") not in _INDEX_FORMATS:
            _refuse(f"{row['id']}: index format {index.get('format')!r} is not one of {sorted(_INDEX_FORMATS)}")
        if index is not None and index.get("select") == "member" and "members" not in row:
            _refuse(f"{row['id']}: a product selects by member but the row declares no members")
        if index is not None and index.get("format") == "ncep-idx":
            select = index.get("select")
            records = select.get("records") if isinstance(select, dict) else None
            if (not isinstance(records, list) or not records
                    or any(not isinstance(r, list) or len(r) != 2 or not all(isinstance(x, str) for x in r)
                           for r in records)):
                _refuse(f"{row['id']}: an ncep-idx index selects with select.records, a list of "
                        "[variable regex, level regex] pairs; without it no record would be fetched")
            for variable, level in records:
                try:
                    re.compile(variable)
                    re.compile(level)
                except re.error as error:
                    _refuse(f"{row['id']}: ncep-idx selector {[variable, level]} is not a valid regex ({error})")
        streams = product.get("streams")
        if streams is not None and (type(streams) is not int or streams < 1):
            _refuse(f"{row['id']}: product streams must be a positive integer (parallel range requests)")
    blocks = [("forecast", row.get("forecast")), ("surface_series", row.get("surface_series")),
              ("surface_series.forecast", row.get("surface_series", {}).get("forecast"))]
    for block_name, block in blocks:
        if block is None:
            continue
        is_forecast = block_name.endswith("forecast")
        allowed = {"interval_hours", "products", "decode"} | ({"max_hours"} if is_forecast else {"forecast"})
        if set(block) - allowed:
            _refuse(f"{row['id']}: unknown {block_name} keys {sorted(set(block) - allowed)}")
        if type(block.get("interval_hours")) is not int or block["interval_hours"] < 1:
            _refuse(f"{row['id']}: {block_name}.interval_hours must be a positive integer")
        if is_forecast and (type(block.get("max_hours")) is not int or block["max_hours"] < 1):
            _refuse(f"{row['id']}: {block_name}.max_hours must be a positive integer")
        if "products" in block:
            nested = {key: value for key, value in row.items() if key not in ("forecast", "surface_series")}
            nested["products"] = block["products"]
            if "decode" in block:
                nested["decode"] = block["decode"]
            validate_row(nested)
        elif block_name == "surface_series":
            _refuse(f"{row['id']}: surface_series needs products")
    remap = row.get("remap")
    if remap is not None:
        unknown = set(remap) - _REMAP_KEYS
        if unknown:
            _refuse(f"{row['id']}: remap keys {sorted(unknown)} are unknown")
        ids = {product.get("id") for product in row["products"]}
        for role in ("latitude", "longitude", "land_fraction"):
            coordinate = remap.get("coordinates", {}).get(role)
            if not coordinate or coordinate.get("product") not in ids:
                _refuse(f"{row['id']}: remap.coordinates.{role} must name a product id of this row; the "
                        "plan is built from the mesh's own coordinate and land-fraction records")
    decode = row["decode"]
    if "mapping" in decode:
        if Path(decode["mapping"]).name != decode["mapping"] or not decode.get("composition"):
            _refuse(f"{row['id']}: mapped decode needs a packaged mapping filename and composition")
        if type(decode.get("workers")) is not int or decode["workers"] < 1:
            _refuse(f"{row['id']}: mapped decode workers must be a positive integer")
    if decode.get("format") == "regular-netcdf":
        fields = decode.get("fields")
        if (not isinstance(fields, list) or not fields
                or any(not all(isinstance(f.get(k), str) and f[k] for k in ("name", "field", "units"))
                       for f in fields)):
            _refuse(f"{row['id']}: regular NetCDF decode requires name, field and units for each field")
    else:
        for key in ("vtable", "map_source"):
            if not decode.get(key):
                _refuse(f"{row['id']}: decode.{key} is required; met_intermediate has no default for it")
    first_guess = row["first_guess"]
    for key in ("nfglevels", "nfgsoillevels", "use_spechumd"):
        if key not in first_guess:
            _refuse(f"{row['id']}: first_guess.{key} is required; the init door refuses without it")


def resolve_source(table: dict[str, Any], name: str, accepted_licences=(), *, coverage: str = "global") -> dict[str, Any]:
    wanted = _norm(name)
    for row in table["sources"]:
        if wanted in {_norm(n) for n in (row["id"], *row.get("aliases", ()))}:
            if row.get("status") == "refused":
                _refuse(f"source {row['id']!r} is declared refused: {row['refusal']}")
            require_source_licences(row, accepted_licences)
            held = row.get("coverage", "global")
            if held != coverage:
                if held == "limited_area":
                    _refuse(f"source {row['id']!r} covers a limited area ({row.get('display_name')}); a global "
                            "mesh initialised from it would have no meteorology outside that domain.  Run it "
                            f"with `gpuwm-hex regional --source {row['id']} --point LAT,LON ...`")
                _refuse(f"source {row['id']!r} is a global source; the regional door takes a limited-area "
                        "row (`gpuwm-hex sources` lists each row's coverage)")
            return row
    known = ", ".join(row["id"] for row in table["sources"])
    _refuse(f"no init source is named {name!r}; known: {known}")
    raise AssertionError("unreachable")


def require_source_licences(row: dict[str, Any], accepted=()) -> None:
    """Check model code AND weights before acquiring inputs or opening a card."""
    if not row.get("generation"):
        return
    policy = row["license"]
    permissive = {"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "public domain"}
    for role in ("code", "weights"):
        grant = policy[role]
        if not grant.get("source"):
            _refuse(f"{row['id']}: {role} licence has no source citation")
        restricted = grant["name"] not in permissive or (role == "weights" and not policy["default_enabled"])
        if restricted and grant["name"] not in accepted:
            _refuse(f"{row['id']}: research-only {role}, licence {grant['name']}; "
                    f"explicitly opt in with --accept-licence {grant['name']!r}")


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Cycle:
    when: datetime

    @classmethod
    def parse(cls, text: str) -> "Cycle":
        for fmt in ("%Y-%m-%dT%H", "%Y-%m-%dT%H:%M", "%Y-%m-%d_%H:%M:%S", "%Y%m%d%H"):
            try:
                return cls(datetime.strptime(text, fmt).replace(tzinfo=timezone.utc))
            except ValueError:
                continue
        _refuse(f"--cycle {text!r} is not YYYY-MM-DDTHH; the products and the init clock both derive from it")
        raise AssertionError("unreachable")

    @property
    def start_time(self) -> str:
        return self.when.strftime("%Y-%m-%d_%H:%M:%S")

    @property
    def wps_date(self) -> str:
        return self.when.strftime("%Y-%m-%d_%H:%M:%S")


def template_context(cycle: Cycle, row: dict[str, Any], member: int | None, lead_hours: int = 0) -> dict[str, Any]:
    if lead_hours < 0:
        _refuse("forecast lead must be nonnegative")
    forecast = row.get("forecast", {})
    if lead_hours and (not forecast or lead_hours > forecast["max_hours"]
                       or lead_hours % forecast["interval_hours"]):
        _refuse(f"{row['id']}: lead {lead_hours} is outside the row's forecast cadence/range")
    valid = cycle.when + timedelta(hours=lead_hours)
    context: dict[str, Any] = {
        "lead": lead_hours, "valid_yyyymmddhh": valid.strftime("%Y%m%d%H"),
        "yyyymmddhh": cycle.when.strftime("%Y%m%d%H"),
        "yyyymmdd": cycle.when.strftime("%Y%m%d"),
        "hh": cycle.when.strftime("%H"),
        "yyyy": cycle.when.strftime("%Y"),
        "mm": cycle.when.strftime("%m"),
        "dd": cycle.when.strftime("%d"),
    }
    members = row.get("members")
    if members is None:
        if member is not None:
            _refuse(f"source {row['id']!r} declares no members; --member {member} would be ignored and the "
                    "receipt would name a member that does not exist")
        context["member"] = 0
        return context
    control = members.get("control")
    if member is None and row.get("generation") and control is not None:
        member = control["number"]
    declared = ((f"control is {control['number']}, " if control else "no control member is published, ")
                + f"perturbed {members['perturbed']['first']}..{members['perturbed']['last']}")
    if member is None:
        _refuse(f"source {row['id']!r} is an ensemble; pass --member ({declared}) so the receipt "
                "names the state it starts from")
    if control is not None and member == control["number"]:
        block = control
    elif members["perturbed"]["first"] <= member <= members["perturbed"]["last"]:
        block = members["perturbed"]
    else:
        _refuse(f"member {member} is outside {row['id']!r}'s declared members ({declared})")
    context["member"] = member
    context["member_tag"] = str(block.get("tag", "")).format(member=member)
    context["member_select"] = {key: str(value).format(member=member)
                                for key, value in block.get("index_select", {}).items()}
    return context


def _templates(value: str | list[str] | None, context: dict[str, Any]) -> list[str]:
    if value is None:
        return []
    return [template.format(**context) for template in ([value] if isinstance(value, str) else value)]


@dataclass
class ProductPlan:
    role: str
    transport: str
    filename: str
    url: str | None
    index_url: str | None
    index_select: dict[str, str] | None = None
    cds: dict[str, Any] | None = None
    codec: str | None = None
    fields: list[str] | None = None
    reason: str | None = None
    #: Further addresses of the same bytes, tried in order when the first
    #: refuses or throttles (a row declares url as a list of templates).
    url_mirrors: tuple[str, ...] = ()
    index_mirrors: tuple[str, ...] = ()
    product_id: str | None = None
    remap: dict[str, str] | None = None
    index_format: str | None = None
    streams: int | None = None
    fetch_args: list[str] | None = None
    request: dict[str, Any] | None = None


def plan_products(row: dict[str, Any], cycle: Cycle, member: int | None, lead_hours: int = 0) -> list[ProductPlan]:
    base_context = template_context(cycle, row, member, lead_hours)
    plans = []
    expanded = []
    for product in (row.get("forecast", {}).get("products", row["products"]) if lead_hours else row["products"]):
        entries = row.get("sets", {}).get(product["expand"], []) if product.get("expand") else [{}]
        for entry in entries:
            expanded.append((product, {**base_context, **entry}))
    for product, context in expanded:
        index = product.get("index")
        select = None
        if index is not None:
            select = context["member_select"] if index["select"] == "member" else dict(index["select"])
        urls = _templates(product.get("url"), context)
        index_urls = _templates(index["url"] if index else None, context)
        plans.append(ProductPlan(
            role=product["role"],
            fetch_args=_templates(product.get("fetch_args"), context),
            request=product.get("request"),
            transport=product["transport"],
            filename=product["filename"].format(**context),
            url=urls[0] if urls else None,
            index_url=index_urls[0] if index_urls else None,
            url_mirrors=tuple(urls[1:]),
            index_mirrors=tuple(index_urls[1:]),
            index_select=select,
            cds=product.get("cds"),
            codec=product.get("codec"),
            fields=product.get("fields"),
            reason=product.get("reason"),
            product_id=product.get("id"),
            remap=({key: str(value).format(**context) for key, value in product["remap"].items()}
                   if product.get("remap") else None),
            index_format=index["format"] if index else None,
            streams=product.get("streams"),
        ))
    names = [plan.filename for plan in plans]
    if len(names) != len(set(names)):
        _refuse(f"{row['id']}: two products expand to the same filename; one would overwrite the other")
    return plans


# ---------------------------------------------------------------------------
# acquisition
# ---------------------------------------------------------------------------
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _http_get(url: str, *, byte_range: tuple[int, int | None] | None = None, attempts: int = 10) -> bytes:
    delay = 2.0
    last: Exception | None = None
    for _ in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": "gpuwm-hex-init-source/1"})
        if byte_range is not None:
            last_byte = "" if byte_range[1] is None else str(byte_range[1])
            request.add_header("Range", f"bytes={byte_range[0]}-{last_byte}")
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body = response.read()
            if (byte_range is not None and byte_range[1] is not None
                    and len(body) != byte_range[1] - byte_range[0] + 1):
                raise OSError(f"range {byte_range} of {url} returned {len(body)} bytes")
            return body
        except urllib.error.HTTPError as error:
            if error.code not in _RETRY_STATUS:
                _refuse(f"{url} answered HTTP {error.code}; the product is not published at that address")
            last = error
        except OSError as error:
            last = error
        time.sleep(delay)
        delay = min(delay * 1.6, 30.0)
    _refuse(f"{url} could not be read after {attempts} attempts ({last}); no partial product is kept")
    raise AssertionError("unreachable")


def select_index_records(index_text: str, select: dict[str, str]) -> list[dict[str, Any]]:
    records = []
    for line in index_text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if all(str(record.get(key)) == value for key, value in select.items()):
            records.append(record)
    return records


def select_ncep_idx_records(index_text: str, select: dict[str, Any]) -> list[dict[str, Any]]:
    """The records of an NCEP ``.idx`` inventory (``n:offset:d=...:VAR:LEVEL:...``) a row's
    ``select.records`` pairs match, as ``{"_offset", "_length", "record"}`` (the last record of the
    file has ``_length`` None: it runs to the end of the object).  A pair is two regexes matched in
    full against VAR and LEVEL, so the selection is table data, never code."""

    rules = [(re.compile(v), re.compile(lv)) for v, lv in select["records"]]
    rows = [line.split(":") for line in index_text.splitlines() if line.strip()]
    for position, row in enumerate(rows):
        if len(row) < 5:
            _refuse(f"index line {position + 1} has {len(row)} fields; an NCEP inventory line has at least five")
    offsets = [int(row[1]) for row in rows]
    if any(a > b for a, b in zip(offsets, offsets[1:])):
        _refuse("NCEP inventory offsets decrease")
    chosen = []
    position = 0
    while position < len(rows):
        end = position + 1
        while end < len(rows) and offsets[end] == offsets[position]:
            end += 1
        selected = [row for row in rows[position:end]
                    if any(v.fullmatch(row[3]) and lv.fullmatch(row[4]) for v, lv in rules)]
        if selected:
            following = offsets[end] if end < len(rows) else None
            chosen.append({"_offset": offsets[position],
                           "_length": None if following is None else following - offsets[position],
                           "record": ";".join(f"{row[3]}:{row[4]}" for row in selected)})
        position = end
    return chosen


def coalesce_spans(spans: list[tuple[int, int | None]],
                   piece: int = RANGE_PIECE_BYTES) -> list[tuple[int, int | None]]:
    """Byte ranges merged where they touch, then cut into pieces of at most ``piece`` bytes.
    Concatenating the pieces in order yields exactly the bytes of the input ranges in order."""

    merged: list[list[int | None]] = []
    for first, last in sorted(spans, key=lambda span: span[0]):
        if merged and merged[-1][1] is not None and merged[-1][1] + 1 == first:
            merged[-1][1] = last
        elif merged and (merged[-1][1] is None or first <= merged[-1][1]):
            _refuse(f"index records overlap at byte {first}; an inventory with overlapping records is corrupt")
        else:
            merged.append([first, last])
    pieces: list[tuple[int, int | None]] = []
    for first, last in merged:
        if last is None:
            pieces.append((first, None))
            continue
        start = first
        while start <= last:
            end = min(last, start + piece - 1)
            pieces.append((start, end))
            start = end + 1
    return pieces


def _fetch_selected(plan: ProductPlan, target: Path, jobs: int) -> dict[str, Any]:
    pairs = list(zip([plan.url, *plan.url_mirrors], [plan.index_url, *plan.index_mirrors]))
    for position, (data_url, index_url) in enumerate(pairs):
        try:
            return _fetch_selected_from(plan, target, jobs, data_url, index_url)
        except InitSourceRefusal:
            if position == len(pairs) - 1:
                raise
    raise AssertionError("unreachable")


def _fetch_selected_from(plan: ProductPlan, target: Path, jobs: int, data_url: str,
                         index_url: str) -> dict[str, Any]:
    index_text = _http_get(index_url).decode("utf-8")
    if plan.index_format == "ncep-idx":
        records = select_ncep_idx_records(index_text, plan.index_select or {})
    else:
        records = select_index_records(index_text, plan.index_select or {})
    if not records:
        _refuse(f"the index {index_url} has no record matching {plan.index_select}; "
                "an empty product would decode to a missing field")
    records.sort(key=lambda record: int(record["_offset"]))
    spans = [(int(r["_offset"]),
              None if r["_length"] is None else int(r["_offset"]) + int(r["_length"]) - 1) for r in records]
    pieces = coalesce_spans(spans)
    partial = target.with_name(target.name + ".part")
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool, partial.open("wb") as handle:
        # map() yields in submission order: the pieces land in file order while
        # every stream keeps working ahead.
        for body in pool.map(lambda span: _http_get(data_url, byte_range=span), pieces):
            handle.write(body)
    with partial.open("rb") as handle:
        head = handle.read(4)
    if head != b"GRIB":
        partial.unlink(missing_ok=True)
        _refuse(f"the records {index_url} selects do not start a GRIB message; that index does not "
                f"describe {data_url}")
    partial.replace(target)
    return {"url": data_url, "index_url": index_url, "index_format": plan.index_format or "ecmwf-jsonl",
            "select": plan.index_select, "records": len(records), "requests": len(pieces), "streams": jobs,
            "index_sha256": hashlib.sha256(index_text.encode("utf-8")).hexdigest()}


def _fetch_whole(plan: ProductPlan, target: Path, jobs: int = 1) -> dict[str, Any]:
    urls = [plan.url, *plan.url_mirrors]
    for position, url in enumerate(urls):
        try:
            if plan.codec is None and jobs > 1:
                parallel = _fetch_whole_parallel(target, url, jobs)
                if parallel is not None:
                    return {"url": url, **parallel}
            return {"url": url, **_fetch_whole_from(plan, target, url)}
        except InitSourceRefusal:
            if position == len(urls) - 1:
                raise
    raise AssertionError("unreachable")


def _object_size(url: str) -> int | None:
    """The object's byte count when the host serves byte ranges, else None (one stream then)."""

    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "gpuwm-hex-init-source/1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            size = response.headers.get("Content-Length")
            ranges = (response.headers.get("Accept-Ranges") or "").lower()
    except (urllib.error.HTTPError, OSError):
        return None
    return int(size) if size and size.isdigit() and int(size) > 0 and ranges == "bytes" else None


def _fetch_whole_parallel(target: Path, url: str, jobs: int) -> dict[str, Any] | None:
    """One whole object as parallel range pieces written at their offsets; None when the host
    does not say it serves ranges (the single-stream path takes it then)."""

    size = _object_size(url)
    if size is None:
        return None
    pieces = coalesce_spans([(0, size - 1)])
    partial = target.with_name(target.name + ".part")
    with partial.open("wb") as handle:
        handle.truncate(size)
    descriptor = os.open(partial, os.O_WRONLY)
    try:
        def one(span: tuple[int, int | None]) -> int:
            body = _http_get(url, byte_range=span)
            os.pwrite(descriptor, body, span[0])
            return len(body)
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            written = sum(pool.map(one, pieces))
    except BaseException:
        os.close(descriptor)
        partial.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    if written != size:
        partial.unlink(missing_ok=True)
        _refuse(f"{url}: {written} of {size} bytes arrived; no partial product is kept")
    partial.replace(target)
    return {"mode": "parallel-range", "requests": len(pieces), "streams": jobs}


def _fetch_whole_from(plan: ProductPlan, target: Path, url: str) -> dict[str, Any]:
    partial = target.with_name(target.name + ".part")
    delay = 2.0
    for attempt in range(10):
        request = urllib.request.Request(url, headers={"User-Agent": "gpuwm-hex-init-source/1"})
        try:
            with urllib.request.urlopen(request, timeout=600) as response, partial.open("wb") as handle:
                stream = bz2.BZ2File(response) if plan.codec == "bz2" else response
                shutil.copyfileobj(stream, handle, length=1 << 22)
            partial.replace(target)
            return {"attempts": attempt + 1}
        except urllib.error.HTTPError as error:
            if error.code not in _RETRY_STATUS:
                _refuse(f"{url} answered HTTP {error.code}; the product is not published at that address")
        except OSError:
            pass
        time.sleep(delay)
        delay = min(delay * 1.6, 30.0)
    partial.unlink(missing_ok=True)
    _refuse(f"{url} could not be read after 10 attempts; no partial product is kept")
    raise AssertionError("unreachable")


def _fetch_cds(plan: ProductPlan, target: Path, cycle: Cycle) -> dict[str, Any]:
    try:
        import cdsapi  # noqa: PLC0415 - optional provider client
    except ImportError:
        _refuse(f"{plan.filename} is a Copernicus CDS product and the cdsapi client is not installed here; "
                "stage the file with --staged DIR (fetched where the CDS key lives) instead")
    request = {key: value for key, value in plan.cds.items() if key != "dataset"}
    request.update({"product_type": ["reanalysis"], "year": [cycle.when.strftime("%Y")],
                    "month": [cycle.when.strftime("%m")], "day": [cycle.when.strftime("%d")],
                    "time": [cycle.when.strftime("%H:00")], "data_format": "grib",
                    "download_format": "unarchived"})
    partial = target.with_name(target.name + ".part")
    cdsapi.Client(quiet=True).retrieve(plan.cds["dataset"], request, str(partial))
    partial.replace(target)
    return {"dataset": plan.cds["dataset"]}


def _streams(plan: ProductPlan, jobs: int | None) -> int:
    """``--jobs`` when given, else the product's declared ``streams``, else the default."""

    return max(1, jobs if jobs is not None else (plan.streams or DEFAULT_STREAMS))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire(plans: list[ProductPlan], dest: Path, cycle: Cycle, *, staged: list[Path] = (),
            jobs: int | None = None) -> list[dict[str, Any]]:
    """Every product as a local file, with how it got here and its digest."""

    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for plan in plans:
        target = dest / plan.filename
        how: dict[str, Any]
        found = next((directory / plan.filename for directory in staged
                      if (directory / plan.filename).is_file()), None)
        if found is not None:
            if found.resolve() != target.resolve():
                target.unlink(missing_ok=True)
                os.symlink(found.resolve(), target)
            how = {"staged_from": str(found.resolve())}
        elif target.is_file() and plan.transport != "engine-fetch":
            how = {"reused": True}
        elif plan.transport == "zarr":
            from gpuwm.zarr_bridge import extract_regular_zarr
            request = {**plan.request, "times": [cycle.when.strftime("%Y-%m-%dT%H:%M:%SZ")]}
            native = extract_regular_zarr(request,
                request_path=dest / (plan.filename + ".request.json"), output=target)
            how = {"request": request, "native": native}
        elif plan.transport == "engine-fetch":
            workers = _streams(plan, jobs)
            argv = [sys.executable, "-m", "gpuwm", "fetch", *plan.fetch_args,
                    "--out", str(dest)]
            env = {**os.environ, "GPUWM_RW_ZARR_WORKERS": str(workers)}
            completed = subprocess.run(argv, env=env, text=True, capture_output=True)
            log = dest / (plan.filename + ".fetch.log")
            log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
            if completed.returncode:
                _refuse(f"engine fetch refused (rc {completed.returncode}); see {log}: "
                        f"{completed.stderr.strip()[-2000:]}")
            if not target.is_file():
                _refuse(f"engine fetch did not publish declared product {target}; see {log}")
            how = {"argv": argv, "workers": workers, "log": str(log)}
        elif plan.transport == "cds":
            how = _fetch_cds(plan, target, cycle)
        elif plan.index_url is not None:
            how = _fetch_selected(plan, target, _streams(plan, jobs))
        else:
            how = _fetch_whole(plan, target, _streams(plan, jobs))
        if target.stat().st_size == 0:
            _refuse(f"{target} is empty; an empty product decodes to missing fields")
        out.append({"role": plan.role, "file": str(target), "bytes": target.stat().st_size,
                    "sha256": _sha256_file(target), "url": plan.url, "transport": plan.transport,
                    "fields": plan.fields, "reason": plan.reason, "id": plan.product_id,
                    "remap": plan.remap, **how})
    return out


# ---------------------------------------------------------------------------
# remap (unstructured source grids)
# ---------------------------------------------------------------------------
def resolve_remapper(explicit: Path | None) -> Path:
    """gdt101_remap through the one engine ladder (flag, env, gpuwm's bridges, PATH)."""

    from .engines import GDT101_REMAP, resolve  # noqa: PLC0415 - the shared ladder

    return resolve(GDT101_REMAP, explicit)


def remap(row: dict[str, Any], products: list[dict[str, Any]], cycle: Cycle, out_dir: Path,
          remapper: Path) -> dict[str, Any]:
    """Put every remap-declared product on the row's regular target grid.

    The mesh's own coordinate and land-fraction records build one plan; each
    declared record is then remapped with the method its row names.  The
    tool refuses a record whose grid identity, centre, reference time,
    parameter or level disagrees with what the row declared.
    """

    spec = row["remap"]
    by_id = {product["id"]: product for product in products if product.get("id")}
    work = out_dir / "remap"
    if work.exists():
        shutil.rmtree(work)
    (work / "out").mkdir(parents=True)
    target = work / "target.txt"
    target.write_text(spec["target"] + "\n", encoding="utf-8")
    plan = work / "plan.bin"
    stamp = cycle.when.strftime("%Y%m%d%H")
    coordinates = spec["coordinates"]
    plan_argv = [str(remapper), "plan"]
    for role in ("latitude", "longitude", "land_fraction"):
        plan_argv += [by_id[coordinates[role]["product"]]["file"], coordinates[role]["selector"]]
    plan_argv += [str(target), str(plan), stamp, str(spec["mesh_cells"]), str(spec["centre"])]
    completed = subprocess.run(plan_argv, capture_output=True, text=True)
    if completed.returncode != 0:
        _refuse(f"gdt101_remap plan refused {row['id']}: {completed.stderr.strip()[-2000:]}")
    jobs = []
    outputs = []
    for product in products:
        if not product.get("remap"):
            continue
        output = work / "out" / Path(product["file"]).name
        jobs.append("\t".join([product["file"], str(output), *product["remap"]["selector"].split(","),
                               product["remap"]["method"], stamp, "0"]))
        outputs.append(str(output))
    (work / "jobs.tsv").write_text("\n".join(jobs) + "\n", encoding="utf-8")
    completed = subprocess.run([str(remapper), "apply", str(plan), str(work / "jobs.tsv")],
                               capture_output=True, text=True)
    if completed.returncode != 0:
        _refuse(f"gdt101_remap apply refused {row['id']}: {completed.stderr.strip()[-2000:]}")
    receipt = {"plan_argv": plan_argv, "plan_sha256": _sha256_file(plan), "jobs": len(jobs),
               "remapper_sha256": _sha256_file(remapper), "target": spec["target"]}
    plan.unlink()
    return {**receipt, "outputs": outputs}


# ---------------------------------------------------------------------------
# decode and init
# ---------------------------------------------------------------------------
def resolve_writer(explicit: Path | None) -> Path:
    """met_intermediate through the one engine ladder (flag, env, gpuwm's bridges, PATH)."""

    from .engines import MET_INTERMEDIATE, resolve  # noqa: PLC0415 - the shared ladder

    return resolve(MET_INTERMEDIATE, explicit)


def _resolve_mapped_writer(spec: dict[str, Any], explicit: Path | None) -> Path:
    """Check the packaged authority and WPS capability before acquiring bytes."""
    import gpuwm
    from gpuwm.mapped_engine_bridge import require_engine

    mapping = Path(gpuwm.__file__).with_name("authorities") / spec["mapping"]
    if not mapping.is_file():
        _refuse(f"mapped authority {mapping} is missing; install an engine carrying this mapping")
    writer = Path(explicit) if explicit else require_engine()
    completed = subprocess.run([str(writer), "capabilities"], capture_output=True, text=True)
    if completed.returncode or not json.loads(completed.stdout.splitlines()[-1]).get("wps_intermediate_v5"):
        _refuse(f"{writer} lacks WPS intermediate v5 export; install a current mapped engine")
    return writer


def resolve_vtable(row: dict[str, Any], vtable_root: Path = VTABLE_ROOT) -> Path:
    vtable = (vtable_root / row["decode"]["vtable"]).resolve()
    if not vtable.is_file():
        _refuse(f"{row['id']}: Vtable {vtable} does not exist")
    return vtable


def generation_preflight(row: dict[str, Any], accepted=()) -> None:
    """Resolve the shared inference and Rust composition contracts before I/O."""
    from gpuwm.earth2_run import model_declaration
    from gpuwm.mapped_engine_bridge import require_engine
    from gpuwm.source_authorities import packaged_authorities

    require_source_licences(row, accepted)
    declaration = row["generation"]
    _, installed = model_declaration(declaration["model"])
    for key, actual in (("channels", installed.get("channels")),
                        ("profile", installed.get("profile")),
                        ("weights_revision", installed.get("weights", {}).get("revision"))):
        if declaration[key] != actual:
            _refuse(f"{row['id']}: installed engine {key} differs from the hex binding; "
                    "update the metadata against the installed model contract before inference")
    if installed.get("weights", {}).get("licence") != row["license"]["weights"]["name"]:
        _refuse(f"{row['id']}: installed weight licence differs from the cited hex grant")
    mapping = json.loads(Path(packaged_authorities(declaration["profile"])["mapping"]).read_text())
    for binding in declaration["bindings"]:
        if binding.get("field") is None:
            continue
        source_field = mapping["fields"].get(binding["source_field"], {})
        selectors = source_field.get("selectors", [])
        selected = [selector for selector in selectors if binding["channel"] in selector.get("name", [])]
        if not selected or ("pressure_pa" in binding and not any(
                selector.get("level_value", -1) * 100 == binding["pressure_pa"] for selector in selected)):
            _refuse(f"{row['id']}: {binding['channel']} differs from the installed Rust field/level mapping")
    # run_model checks the inference runtime before acquisition. Verified
    # forecast reuse needs the model contract and Rust reader, not that runtime.
    engine = require_engine()
    probe = subprocess.run([str(engine), "capabilities"], capture_output=True, text=True, check=True)
    if not json.loads(probe.stdout.splitlines()[-1]).get("wps_intermediate_v5"):
        _refuse(f"{engine}: missing Rust WPS intermediate export; refusing before inference")


def generate(row: dict[str, Any], source_cycle: Cycle, lead: int, member: int | None,
             out_dir: Path, accepted=(), workers: int | None = None) -> dict[str, Any]:
    """One table-driven inference call; the requested hex clock is its valid time.

    The positive spinup is intentional: a bare global source request consumes a
    real forecast, including when the caller asks for source lead zero.
    """
    from gpuwm.earth2_licensing import licence_scope
    from gpuwm.earth2_run import run_model

    template_context(source_cycle, row, member, lead)
    declaration = row["generation"]
    spinup = declaration["spinup_hours"]
    inference_cycle = source_cycle.when - timedelta(hours=spinup)
    workers = workers or int(os.environ.get("RAYON_NUM_THREADS", DEFAULT_GENERATION_WORKERS))
    if workers < 1:
        _refuse("generated initialization needs a positive worker budget")
    affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    selected = set(sorted(affinity)[:workers]) if affinity else None
    reused = False
    cached = out_dir / "earth2-run.json"
    if cached.is_file():
        prior = json.loads(cached.read_text())
        expected_members = [0 if member is None else member]
        model = prior.get("model_row", {})
        if (prior.get("model") != declaration["model"]
                or prior.get("cycle") != inference_cycle.isoformat()
                or prior.get("hours") != spinup + lead
                or prior.get("members") != expected_members
                or prior.get("initial_condition_source") != declaration["ic_source"]
                or prior.get("forecast_start_hour") != 0
                or model.get("channels") != declaration["channels"]
                or model.get("profile") != declaration["profile"]
                or model.get("weights", {}).get("revision") != declaration["weights_revision"]
                or model.get("weights", {}).get("licence") != row["license"]["weights"]["name"]):
            _refuse(f"{cached}: existing generation belongs to a different request")
        files = prior["runner"]["files"]
        if (not files or set(prior["forecast"]) != {f["path"] for f in files}
                or not any(f.get("lead_hours") == spinup + lead
                           and f.get("member") == expected_members[0] for f in files)):
            _refuse(f"{cached}: forecast inventory differs from the runner receipt")
        for product in files:
            path = Path(product["path"])
            if (not path.is_file() or path.stat().st_size != product["bytes"]
                    or _sha256_file(path) != product["sha256"]):
                _refuse(f"{cached}: generated forecast bytes changed: {path}")
        receipt, reused = prior, True
    try:
        if selected is not None:
            os.sched_setaffinity(0, selected)
        if not reused:
            with licence_scope(accepted):
                receipt = run_model(model=declaration["model"], cycle=inference_cycle,
                                    hours=spinup + lead, area=None, out=out_dir,
                                    ic_source=declaration["ic_source"], member=member)
    finally:
        if affinity is not None:
            os.sched_setaffinity(0, affinity)
    receipt["hex_binding"] = {"inference_cycle": inference_cycle.isoformat(),
                              "forecast_lead_hours": spinup + lead,
                              "valid_time": Cycle(source_cycle.when + timedelta(hours=lead)).start_time,
                              "workers": len(selected) if selected else workers,
                              "cpu_affinity": sorted(selected) if selected else None,
                              "reused_verified_forecast": reused,
                              "bindings": declaration["bindings"]}
    return receipt


def global_surface_donor(row: dict[str, Any], generated: dict[str, Any], out_dir: Path) -> None:
    """Keep a global donor global, without a string corner parser's dateline heuristic.

    The engine's historical default (-180, 179.75) is interpreted as a narrow
    dateline box by its corner parser. Reuse its declared native acquisition
    request, selecting surface fields and the row's explicit full-band extent.
    Rust owns all acquisition, field reads, derivation and writing.
    """
    spec = row["generation"].get("global_donor")
    donor = generated["donor"]
    if not spec or donor.get("provider") != spec["provider"]:
        return
    from gpuwm.zarr_bridge import extract_regular_zarr

    acquisition = Path(donor["analysis"]).parent / spec["acquisition_receipt"]
    original = json.loads(acquisition.read_text())
    request = dict(original["request"]["native_request"])
    request["area"] = spec["area"]
    request["fields"] = [f for f in request["fields"] if not f.get("pressure")]
    fields = {f["output"] for f in request["fields"]}
    request["derived"] = [f for f in request.get("derived", [])
                          if f.get("inputs") and set(f["inputs"]) <= fields]
    root = out_dir / "global-donor"
    root.mkdir(parents=True, exist_ok=True)
    target = root / "surface.nc"
    receipt_path = root / "receipt.json"
    started = time.perf_counter()
    if receipt_path.is_file():
        prior = json.loads(receipt_path.read_text())
        if (prior["request"] != request or not target.is_file()
                or target.stat().st_size != prior["bytes"]
                or _sha256_file(target) != prior["sha256"]):
            _refuse(f"{receipt_path}: existing global donor differs from the requested bytes")
        result = prior
    else:
        native = extract_regular_zarr(request, request_path=root / "request.json", output=target)
        result = {"request": request, "native": native, "path": str(target),
                  "bytes": target.stat().st_size, "sha256": _sha256_file(target),
                  "seconds": round(time.perf_counter() - started, 3),
                  "parallelism": donor.get("parallelism")}
        receipt_path.write_text(json.dumps(result, indent=2) + "\n")
    donor["supplements"][spec["data_role"]] = str(target)
    generated["hex_binding"]["global_donor"] = result


def decode_generated(row: dict[str, Any], generated: dict[str, Any], cycle: Cycle,
                     out_dir: Path, workers: int, accepted=()) -> dict[str, Any]:
    """The existing mapped Rust engine owns NetCDF reads, transforms and WPS writes."""
    global_surface_donor(row, generated, out_dir)
    met = out_dir / f"MET:{cycle.when.strftime('%Y-%m-%d_%H')}"
    argv = [sys.executable, "-m", "gpuwm", "source-intermediate",
            "--source", row["engine_source_id"], "--profile", generated["profile"],
            "--valid-time", cycle.when.strftime("%Y-%m-%dT%H:%M:%S"),
            "--out", str(met), "--work-dir", str(out_dir / "composition"),
            "--workers", str(workers)]
    for licence in accepted:
        argv += ["--accept-licence", licence]
    for filename in generated["forecast"]:
        argv += ["--input", str(filename)]
    for role, filenames in generated["donor"]["supplements"].items():
        for filename in ([filenames] if isinstance(filenames, str) else filenames):
            argv += ["--supplement", f"{role}={filename}"]
    with (out_dir / "source-intermediate.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        _refuse(f"Rust source composition exited {completed.returncode}; "
                f"see {out_dir / 'source-intermediate.log'}")
    export = json.loads(met.with_name(met.name + ".receipt.json").read_text())
    return {"argv": argv, "met": str(met), "met_sha256": _sha256_file(met),
            "met_bytes": met.stat().st_size, "export": export}


def decode(row: dict[str, Any], inputs: list[str], cycle: Cycle, out_dir: Path,
           writer: Path, vtable_root: Path = VTABLE_ROOT) -> dict[str, Any]:
    if "mapping" in row["decode"]:
        return _decode_mapped(row, inputs, cycle, out_dir, writer)
    vtable = resolve_vtable(row, vtable_root)
    met = out_dir / f"MET:{cycle.when.strftime('%Y-%m-%d_%H')}"
    argv = [str(writer), "--vtable", str(vtable), "--date", cycle.wps_date,
            "--map-source", row["decode"]["map_source"], "--out", str(met), *inputs]
    completed = subprocess.run(argv, capture_output=True, text=True)
    (out_dir / "met_intermediate.stdout.json").write_text(completed.stdout, encoding="utf-8")
    (out_dir / "met_intermediate.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        _refuse(f"met_intermediate refused {row['id']} (rc {completed.returncode}): "
                f"{completed.stderr.strip()[-2000:]}")
    return {"argv": argv, "met": str(met), "vtable_sha256": _sha256_file(vtable),
            "writer_sha256": _sha256_file(writer), "met_sha256": _sha256_file(met),
            "met_bytes": met.stat().st_size}


def _decode_mapped(row: dict[str, Any], inputs: list[str], cycle: Cycle,
                   out_dir: Path, writer: Path) -> dict[str, Any]:
    """Compose a declared mapping and export WPS, with all arrays kept in Rust."""
    import gpuwm
    from gpuwm.mapped_authoring import author_input_manifest
    from gpuwm.mapped_engine_bridge import run_engine
    from gpuwm.ingest.preparation_workers import worker_environment

    spec = row["decode"]
    mapping = Path(gpuwm.__file__).with_name("authorities") / spec["mapping"]
    work = out_dir / "mapped"
    work.mkdir(exist_ok=True)
    packaged_mapping_sha256 = _sha256_file(mapping)
    if spec.get("mapping_fields"):
        metadata = json.loads(mapping.read_text(encoding="utf-8"))
        metadata["fields"].update(spec["mapping_fields"])
        mapping = work / "mapping.json"
        mapping.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    composition = work / "composition.json"
    composition.write_text(json.dumps(spec["composition"], indent=2) + "\n", encoding="utf-8")
    supplements = {}
    provenance = {}
    in_band = work / "in-band-provenance.json"
    if spec.get("in_band_supplements"):
        in_band.write_text(json.dumps({"schema": "gpuwm-hex.in-band-inputs/v1",
            "inputs": [{"path": name, "sha256": _sha256_file(Path(name))} for name in inputs],
            "mapping_sha256": _sha256_file(mapping)}, indent=2) + "\n", encoding="utf-8")
        for data_role, provenance_role in spec["in_band_supplements"].items():
            supplements[data_role] = inputs
            provenance[provenance_role] = in_band
    manifest = author_input_manifest(work / "inputs.json", mapping_path=mapping,
        composition_path=composition, primary_files=inputs,
        supplement_files=supplements, provenance_files=provenance)
    composed = run_engine("compose", mapping=mapping, files=inputs, composition=composition,
        supplements=supplements, provenance=provenance,
        input_manifest=manifest["manifest"]["path"],
        input_manifest_sha256=manifest["manifest"]["sha256"], output=work / "composed",
        engine=writer, threads=spec["workers"], lead_batch=True)
    met = out_dir / f"MET:{cycle.when.strftime('%Y-%m-%d_%H')}"
    pending = work / "intermediate.part"
    pending.unlink(missing_ok=True)
    contract = Path(gpuwm.__file__).with_name("authorities") / "wps-intermediate-fields.v1.json"
    argv = [str(writer), "wps", "--frameset", str(work / "composed"),
            "--composition", str(composition), "--contract", str(contract),
            "--valid-time", cycle.when.strftime("%Y-%m-%dT%H:%M:%S"), "--out", str(pending)]
    completed = subprocess.run(argv, capture_output=True, text=True,
                               env=worker_environment(spec["workers"]))
    (out_dir / "mapped-wps.stdout.json").write_text(completed.stdout, encoding="utf-8")
    (out_dir / "mapped-wps.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        _refuse(f"mapped WPS export refused {row['id']}: {completed.stderr.strip()[-2000:]}")
    native = json.loads(completed.stdout.splitlines()[-1])
    for key in ("nfglevels", "nfgsoillevels"):
        if native[key] != row["first_guess"][key]:
            _refuse(f"{row['id']}: WPS {key}={native[key]} disagrees with row {row['first_guess'][key]}")
    pending.replace(met)
    return {"argv": argv, "met": str(met), "mapping_sha256": _sha256_file(mapping),
            "packaged_mapping_sha256": packaged_mapping_sha256,
            "composition_sha256": _sha256_file(composition), "workers": spec["workers"],
            "compose": composed["receipt"], "native": native,
            "writer_sha256": _sha256_file(writer), "met_sha256": _sha256_file(met),
            "met_bytes": met.stat().st_size}


def init_argv(table: dict[str, Any], row: dict[str, Any], *, met: Path, grid: Path, static: Path,
              vertical_spec: Path, out: Path, cycle: Cycle) -> list[str]:
    switches = table["run_switches"]
    first_guess = row["first_guess"]
    return [
        "init", "--met", str(met), "--grid", str(grid), "--static", str(static),
        "--vertical-spec", str(vertical_spec), "--out", str(out), "--start-time", cycle.start_time,
        "--nfglevels", str(first_guess["nfglevels"]),
        "--nfgsoillevels", str(first_guess["nfgsoillevels"]),
        "--use-spechumd", first_guess["use_spechumd"],
        "--extrap-airtemp", switches["extrap_airtemp"],
        "--theta-adv-order", str(switches["theta_adv_order"]),
        "--coef-3rd-order", repr(float(switches["coef_3rd_order"])),
        "--virtual-factor", switches["virtual_factor"],
        "--deep-soil-moisture", switches["deep_soil_moisture"],
        "--landuse-table", switches["landuse_table"],
        "--frac-seaice", switches["frac_seaice"],
        "--tsk-seaice-threshold", repr(float(switches["tsk_seaice_threshold"])),
        "--oned-underflow", switches["oned_underflow"],
    ]


def run_init(argv: list[str]) -> int:
    from .cli import main as cli_main  # noqa: PLC0415 - the init door, unchanged

    return int(cli_main(argv))


def order_and_case(*, init: Path, order_map: Path, hex_order: Path, case_template: Path,
                   case_out: Path, row: dict[str, Any], cycle: Cycle, member: int | None) -> dict[str, Any]:
    ordered = init.with_name("ordered.init.nc")
    log = init.with_name("order.log")
    with log.open("w") as handle:
        for argv in ([str(hex_order), "--apply", str(order_map), str(init), str(ordered)],
                     [str(hex_order), "--record-donor", str(order_map), str(ordered)]):
            rc = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                _refuse(f"{' '.join(argv)} failed (rc {rc}); see {log}")
    case = json.loads(case_template.read_text(encoding="utf-8"))
    if case.get("start") not in (None, cycle.start_time):
        _refuse(f"case template {case_template} starts {case.get('start')}, not {cycle.start_time}; the "
                "forecast would run a clock its init does not carry")
    case["init"] = str(ordered)
    label = row["display_name"] + (f", member {member}" if member is not None else "")
    case["init_source"] = f"{label}; {cycle.start_time}; table row {row['id']}"
    case_out.write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    return {"ordered_init": str(ordered), "case": str(case_out)}


# ---------------------------------------------------------------------------
# door
# ---------------------------------------------------------------------------
def add_init_source_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "init-source",
        help="build a hex init from any source row of the init-source table",
        description=("Acquire, decode and initialise from one row of data/init-sources.v1.json. "
                     "Every source travels the same path; the row supplies its products, Vtable, "
                     "map-source label and first-guess declarations."),
    )
    parser.add_argument("--source", help="source id or alias (see --list)")
    parser.add_argument("--accept-licence", action="append", default=[], metavar="LICENCE",
                        help="research-only opt-in naming each exact model code/weight licence")
    parser.add_argument("--cycle", help="YYYY-MM-DDTHH")
    parser.add_argument("--lead-hours", type=int, default=0, help="forecast lead from source cycle; init clock is cycle + lead")
    parser.add_argument("--member", type=int, default=None, help="ensemble member (ensemble rows only)")
    parser.add_argument("--out-dir", type=Path, help="directory for products, intermediate, init and receipt")
    parser.add_argument("--grid", type=Path)
    parser.add_argument("--static", type=Path)
    parser.add_argument("--vertical-spec", type=Path)
    parser.add_argument("--staged", type=Path, action="append", default=[],
                        help="directory already holding product files by their table filename")
    parser.add_argument("--writer", type=Path, default=None, help="met_intermediate executable")
    parser.add_argument("--remapper", type=Path, default=None,
                        help="gdt101_remap executable (rows whose source grid is unstructured)")
    parser.add_argument("--table", type=Path, default=None, help="alternative table (tests)")
    parser.add_argument("--jobs", type=int, default=None,
                        help="parallel range requests per product (default: the product's declared streams, "
                             f"else {DEFAULT_STREAMS}; open-data hosts throttle bursts with 503 Slow Down)")
    parser.add_argument("--stop-after", choices=("acquire", "decode", "init"), default=None)
    parser.add_argument("--order-map", type=Path, default=None)
    parser.add_argument("--hex-order", type=Path, default=None)
    parser.add_argument("--case-template", type=Path, default=None)
    parser.add_argument("--keep-unordered", action="store_true",
                        help="keep init.nc after ordering (default: removed; its digest stays in the provenance)")
    parser.add_argument("--list", action="store_true", help="print the table's rows and exit")
    parser.set_defaults(handler=run_init_source_command)


def run_init_source_command(arguments: argparse.Namespace) -> int:
    table = load_table(arguments.table)
    if arguments.list:
        for row in table["sources"]:
            state = row.get("status", "runnable")
            print(f"{row['id']:10s} {state:9s} {row['display_name']}")
        return 0
    for flag in ("source", "cycle", "out_dir"):
        if getattr(arguments, flag) is None:
            _refuse(f"--{flag.replace('_', '-')} is required")
    started = time.time()
    accepted = getattr(arguments, "accept_licence", ())
    row = resolve_source(table, arguments.source, accepted)
    source_cycle = Cycle.parse(arguments.cycle)
    lead = getattr(arguments, "lead_hours", 0)
    cycle = Cycle(source_cycle.when + timedelta(hours=lead))
    # Everything the later stages need from this install is resolved BEFORE
    # the download: a missing writer, remapper or Vtable found after the
    # acquire stage costs the user the whole transfer and then refuses.
    stop = arguments.stop_after
    generation = row.get("generation")
    if generation:
        generation_preflight(row, accepted)
    if stop != "acquire" and not generation and "mapping" in row["decode"]:
        writer = _resolve_mapped_writer(row["decode"], arguments.writer)
    else:
        writer = resolve_writer(arguments.writer) if stop != "acquire" and not generation else None
    remapper = (resolve_remapper(arguments.remapper)
                if row.get("remap") and stop != "acquire" else None)
    if stop != "acquire" and not generation:
        resolve_vtable(row)
    out_dir = arguments.out_dir.expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    plans = plan_products(row, source_cycle, arguments.member, lead) if not generation else []
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "table": {"path": table["_path"], "sha256": table["_sha256"]},
        "source": row["id"], "member": arguments.member, "cycle": source_cycle.start_time,
        "source_cycle": source_cycle.start_time, "lead_hours": lead, "valid_time": cycle.start_time,
        "row": row, "run_switches": table["run_switches"],
        "table_extensions": table.get("_extensions", []),
    }
    if row.get("attribution"):
        # Licence attribution the source requires (ERA5: Licence to Use Copernicus Products).
        receipt["attribution"] = row["attribution"]
    stages: dict[str, float] = {}

    def write_receipt() -> None:
        receipt["stage_seconds"] = stages
        receipt["seconds"] = round(time.time() - started, 3)
        (out_dir / "init-source-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    t0 = time.time()
    if generation:
        if arguments.staged:
            _refuse("generated initialization owns inference provenance; --staged is for downloaded products")
        try:
            receipt["generation"] = generate(row, source_cycle, lead, arguments.member,
                                             out_dir / "generation", accepted, arguments.jobs)
        except Exception as error:
            stages["acquire"] = round(time.time() - t0, 3)
            receipt["failed"] = f"{type(error).__name__}: {error}"
            write_receipt()
            raise
        receipt["products"] = receipt["generation"]["runner"]["files"]
    else:
        receipt["products"] = acquire(plans, out_dir / "products", cycle,
                                      staged=[p.expanduser().absolute() for p in arguments.staged],
                                      jobs=arguments.jobs)
    stages["acquire"] = round(time.time() - t0, 3)
    if arguments.stop_after == "acquire":
        write_receipt()
        return 0
    inputs = [product["file"] for product in receipt["products"]] if not generation else []
    if row.get("remap"):
        t0 = time.time()
        receipt["remap"] = remap(row, receipt["products"], cycle, out_dir, remapper)
        inputs = receipt["remap"].pop("outputs")
        stages["remap"] = round(time.time() - t0, 3)
    t0 = time.time()
    if generation:
        workers = arguments.jobs or int(os.environ.get("RAYON_NUM_THREADS", DEFAULT_GENERATION_WORKERS))
        try:
            receipt["decode"] = decode_generated(row, receipt["generation"], cycle, out_dir, workers, accepted)
        except Exception as error:
            stages["decode"] = round(time.time() - t0, 3)
            receipt["failed"] = f"{type(error).__name__}: {error}"
            write_receipt()
            raise
        # These counts come from the actual Rust export, including its surface
        # slab. A nominal pressure-level count can silently drop the top level.
        export = receipt["decode"]["export"]
        row = {**row, "first_guess": {**row["first_guess"],
                                     "nfglevels": export["nfglevels"],
                                     "nfgsoillevels": export["nfgsoillevels"]}}
        receipt["effective_first_guess"] = row["first_guess"]
    else:
        receipt["decode"] = decode(row, inputs, cycle, out_dir, writer)
    if row.get("remap"):
        # The regular-grid copies are an intermediate step; their digests are
        # not needed once the WPS intermediate (hashed above) exists.
        shutil.rmtree(out_dir / "remap" / "out", ignore_errors=True)
    stages["decode"] = round(time.time() - t0, 3)
    if arguments.stop_after == "decode":
        write_receipt()
        return 0
    for flag in ("grid", "static", "vertical_spec"):
        if getattr(arguments, flag) is None:
            _refuse(f"--{flag.replace('_', '-')} is required to build the init")
    init = out_dir / "init.nc"
    argv = init_argv(table, row, met=Path(receipt["decode"]["met"]), grid=arguments.grid,
                     static=arguments.static, vertical_spec=arguments.vertical_spec, out=init, cycle=cycle)
    receipt["init_argv"] = argv
    t0 = time.time()
    rc = run_init(argv)
    stages["init"] = round(time.time() - t0, 3)
    if rc != 0:
        write_receipt()
        return rc
    receipt["init"] = {"path": str(init), "provenance": str(init.with_name(init.name + ".provenance.json"))}
    if arguments.stop_after != "init" and arguments.case_template is not None:
        for flag in ("order_map", "hex_order"):
            if getattr(arguments, flag) is None:
                _refuse(f"--{flag.replace('_', '-')} is required with --case-template")
        t0 = time.time()
        receipt["case"] = order_and_case(init=init, order_map=arguments.order_map,
                                         hex_order=arguments.hex_order,
                                         case_template=arguments.case_template,
                                         case_out=out_dir / "case.json", row=row, cycle=cycle,
                                         member=arguments.member)
        stages["order"] = round(time.time() - t0, 3)
        if not arguments.keep_unordered:
            # 6.2 GB per 30 km init: the ordered copy is what a forecast reads,
            # and the init provenance receipt keeps the unordered file's digest.
            receipt["init"]["unordered_removed_bytes"] = init.stat().st_size
            init.unlink()
    write_receipt()
    print(json.dumps({"source": row["id"], "member": arguments.member, "init": str(init),
                      "case": receipt.get("case", {}).get("case"), "seconds": receipt["seconds"]}, indent=2))
    return 0
