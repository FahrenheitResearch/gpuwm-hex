"""Table-driven chemistry inputs. Data decode, mapping and physics run in Rust.

The normalized frame is mesh-bound, SI, and explicit about time. An emission
frame covers [valid_time, valid_time + interval_s); a missing frame is an error.
Composition frames are dry-air mixing ratios on model mass levels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

SCHEMA = "gpuwm-hex.chem-source-frame/v1"
TABLE = Path(__file__).with_name("data") / "chem-sources.v1.json"


def instant(value):
    if isinstance(value, bool):
        raise ValueError("chemistry time cannot be boolean")
    if isinstance(value, (int, float)):
        if not float(value).is_integer():
            raise ValueError("chemistry valid time must be an integral UTC second")
        return int(value)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("chemistry time needs an explicit UTC offset")
    if parsed.microsecond:
        raise ValueError("chemistry valid time must be an integral UTC second")
    return int(parsed.timestamp())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _discard(path):
    path = Path(path)
    if path.exists():
        record = {"path":str(path.resolve()), "bytes":path.stat().st_size,
                  "reason":"failed source preparation, owned partial artifact"}
        with (path.parent / "deleted-files.jsonl").open("a") as stream:
            stream.write(json.dumps(record)+"\n")
        path.unlink()


def source_row(name, table=TABLE, *, research_license=None):
    raw = json.loads(Path(table).read_text())
    if raw.get("schema") != "gpuwm-hex.chem-sources/v1":
        raise ValueError("unknown chemistry source-table schema")
    rows = [row for row in raw["sources"] if row["id"] == name]
    if len(rows) != 1:
        raise ValueError(f"chemistry source {name!r} must identify exactly one table row")
    row = rows[0]
    if row.get("research_only") and research_license != row["license"]["id"]:
        raise ValueError(f"{name} requires --research-license {row['license']['id']}")
    return row


def binary():
    explicit = os.environ.get("GPUWM_HEX_RW_CHEM_SOURCES")
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"GPUWM_HEX_RW_CHEM_SOURCES={path} does not exist")
        return str(path)
    name = "rw_hex_chem_sources" + (".exe" if os.name == "nt" else "")
    bundled = Path(__file__).with_name("libexec") / name
    if bundled.is_file():
        return str(bundled)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        "rw_hex_chem_sources is not installed. Build rust/chem-sources/Cargo.toml "
        "and set GPUWM_HEX_RW_CHEM_SOURCES to its executable.")


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(v for k, v in attrs if k == "href" and v)


def discover(row, valid_time):
    """Select one public file by date and revision, entirely from a table row."""
    when = datetime.fromtimestamp(instant(valid_time) + row["download"].get("file_time_offset_s", 0), timezone.utc)
    remote = row["download"]
    base = when.strftime(remote["listing"])
    parsed = urlparse(base)
    if parsed.scheme != "https" or parsed.username or parsed.password:
        raise ValueError("chemistry listings require credential-free HTTPS")
    with urlopen(base, timeout=45) as stream:
        listing = stream.read(16 * 1024 * 1024 + 1)
    if len(listing) > 16 * 1024 * 1024:
        raise ValueError("source listing exceeds 16 MiB")
    parser = _Links()
    parser.feed(listing.decode("utf-8"))
    pattern = re.compile(remote["pattern"])
    wanted = when.strftime(remote["time_format"])
    found = []
    for link in parser.links:
        match = pattern.fullmatch(link)
        if match and match["valid"] == wanted:
            url = urljoin(base, link)
            if urlparse(url).netloc != parsed.netloc:
                raise ValueError("source listing redirected a file to another host")
            found.append((match.groupdict().get("creation", ""), url))
    if not found:
        raise FileNotFoundError(f"no {row['id']} file for {wanted} in {base}")
    return max(found)[1]


def fetch(row, valid_time, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = discover(row, valid_time)
    part = destination.with_name(destination.name + ".part")
    started = time.monotonic()
    # Claim the partial name before entering cleanup scope. An existing
    # partial belongs to its earlier producer and must never be discarded.
    target = part.open("xb")
    try:
        with target, urlopen(url, timeout=90) as stream:
            expected = getattr(stream, "headers", {}).get("Content-Length")
            if getattr(stream, "status", 200) != 200:
                raise ValueError("chemistry fetch requires a complete HTTP 200 response")
            shutil.copyfileobj(stream, target, 4 * 1024 * 1024)
            target.flush()
            if expected is not None and part.stat().st_size != int(expected):
                raise ValueError("truncated chemistry response: Content-Length differs from downloaded bytes")
        if part.stat().st_size == 0:
            raise ValueError("empty chemistry source response")
        # Publish without replacing a file another producer created while
        # the download was running. Both names are on the same filesystem.
        os.link(part, destination)
        part.unlink()
    except BaseException:
        if part.exists():
            _discard(part)
        raise
    receipt = {"url": url, "file": str(destination), "sha256": digest(destination),
               "bytes": destination.stat().st_size, "wall_s": time.monotonic() - started}
    destination.with_suffix(destination.suffix + ".download.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return receipt


def prepare(*, row, input, mesh, output, valid_time, time_index=0,
            pressure_file=None, humidity_input=None, pressure_variables=(), max_distance_m=200000,
            expected_species=(), threads=4, executable=None, map_output=None,
            map_spacing_deg=1., map_bounds=(-180,180,-90,90)):
    """Run the same Rust decoder for any compatible source-table row."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = output.with_suffix(output.suffix + ".receipt.json")
    request_path = output.with_suffix(output.suffix + ".request.json")
    if any(p.exists() for p in (output, receipt, request_path)) or (map_output and Path(map_output).exists()):
        raise FileExistsError(f"chemistry output or receipt already exists: {output}")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("chemistry Rust thread count must be a positive integer")
    paths = {"input": str(Path(input).resolve()), "mesh": str(Path(mesh).resolve())}
    if pressure_file is not None:
        paths["pressure_file"] = str(Path(pressure_file).resolve())
    if humidity_input is not None:
        paths["humidity_input"] = str(Path(humidity_input).resolve())
    if expected_species:
        row = json.loads(json.dumps(row))
        row["variables"] = [v for v in row["variables"] if v["species"] in expected_species]
        if {v["species"] for v in row["variables"]} != set(expected_species):
            raise ValueError("requested species are absent from the chemistry source row")
    with ThreadPoolExecutor(max_workers=min(threads, len(paths))) as pool:
        hashes = dict(zip(paths, pool.map(digest, paths.values())))
    request = {"row": row, **paths, "output": str(output.resolve()),
               "receipt": str(receipt.resolve()), "valid_time": instant(valid_time),
               "time_index": time_index, "max_distance_m": max_distance_m,
               "pressure_variables": list(pressure_variables),
               "expected_species": list(expected_species), "mesh_sha256": hashes["mesh"]}
    if map_output:
        request.update(map_output=str(Path(map_output).resolve()), map_spacing_deg=map_spacing_deg, map_bounds=list(map_bounds))
    request_path.write_text(json.dumps(request, indent=2)+"\n")
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    command = [executable or binary(), str(request_path.resolve())]
    started = time.monotonic()
    # One process per source frame, bounded CPU parallelism, no Python data path.
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    log = output.with_suffix(output.suffix + ".log")
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        # Failed partial products are never admissible chemistry frames.
        if output.exists():
            _discard(output)
        if map_output:
            _discard(map_output)
        raise RuntimeError(f"chemistry source decode failed ({result.returncode}); see {log}: {result.stderr.strip()}")
    record = json.loads(receipt.read_text())
    record.update({"command": command, "decoder_sha256":digest(command[0]),
                   "orchestrator_sha256":digest(__file__), "input_sha256": hashes,
                   "source_row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
                   "output_sha256": digest(output), "total_wall_s": time.monotonic()-started})
    if map_output:
        record["map"] = {"path":str(Path(map_output).resolve()), "sha256":digest(map_output), "sampling":"nearest native hex cell"}
    receipt.write_text(json.dumps(record, indent=2)+"\n")
    return record


def _frame(path, names, shape, mesh_sha256, units, valid_time, *, subset=False):
    # gpuwm.netcdf_bridge performs NetCDF reads in the Rust bridge.
    from gpuwm.netcdf_bridge import Dataset
    with Dataset(path) as dataset:
        for key, wanted in (("schema", SCHEMA), ("basis", "dry_air"), ("mesh_sha256", mesh_sha256)):
            if dataset.getncattr(key) != wanted:
                raise ValueError(f"{path}: chemistry {key} differs from the requested mesh/schema")
        if int(dataset.getncattr("valid_time_unix")) != valid_time:
            raise ValueError(f"{path}: chemistry frame valid time differs")
        present = [n for n in names if n in dataset.variables]
        if not subset and len(present) != len(names):
            raise ValueError(f"{path}: initial/boundary chemistry must populate every configured species")
        if not present:
            raise ValueError(f"{path}: no configured chemistry species")
        result = {}
        for name in present:
            variable = dataset.variables[name]
            if tuple(variable.shape) != shape or variable.getncattr("units") != units:
                raise ValueError(f"{path}: {name} shape or units differ from the transport contract")
            result[name] = variable[:]
    return result


def attach(backend, host):
    """Bind normalized source frames to the transport lane's committed state."""
    import cupy as cp
    runtime = backend.chemistry
    config = runtime.raw["inputs"]
    start = instant(config["valid_time"])
    mesh_hash = config["mesh_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", mesh_hash):
        raise ValueError("chemistry inputs need the prepared mesh SHA-256")
    prepared_hash = host["prepared"].input_bytes["grid"]["sha256"]
    if mesh_hash != prepared_hash:
        raise ValueError("chemistry source mesh differs from the forecast grid authority")
    regional = host.get("regional") is not None
    boundary_cells = None
    if regional:
        from .regional_v841 import N_RELAX_ZONE
        rings = cp.asarray(host["prepared"].mesh.arrays["bdyMaskCell"])
        boundary_cells = cp.flatnonzero(rings > N_RELAX_ZONE)
        if boundary_cells.size == 0:
            raise ValueError("regional chemistry has no declared specified boundary cells")
    names = [s.name for s in runtime.species]
    base = Path(runtime.path).parent

    def path(entry):
        return str((base / entry["path"]).resolve())

    shape = (runtime.nlev, runtime.ncells)
    initial_path = path(config["initial"])
    initial = _frame(initial_path, names, shape, mesh_hash, "kg kg-1", start)
    runtime.initial = {n: cp.asarray(v, dtype=cp.float32) for n, v in initial.items()}
    for name, values in runtime.initial.items():
        if not bool(cp.all(cp.isfinite(values) & (values >= 0))):
            raise ValueError(f"initial {name} contains invalid mixing ratio")
    emissions = []
    for entry in config.get("emissions", []):
        at = instant(entry["valid_time"])
        span = float(entry["interval_s"])
        if not math.isfinite(span) or not span > 0:
            raise ValueError("chemistry emission interval must be positive")
        values = _frame(path(entry), names, (runtime.ncells,), mesh_hash,
                        "kg m-2 s-1", at, subset=True)
        gpu = {n: cp.asarray(v, dtype=cp.float32) for n, v in values.items()}
        if not all(bool(cp.all(cp.isfinite(v) & (v >= 0))) for v in gpu.values()):
            raise ValueError("emission frame contains invalid source flux")
        emissions.append((at, at + span, gpu))
    if not emissions:
        raise ValueError("chemistry inputs need explicit emission coverage")
    # Per-species series may combine distinct source products, but duplicate
    # frames for a species and instant would accidentally double its source.
    for i, (a, b, values) in enumerate(emissions):
        for c, d, other in emissions[:i]:
            if max(a, c) < min(b, d) and set(values) & set(other):
                raise ValueError("overlapping emission frames would double count a species")

    boundary = []
    for entry in config.get("boundary", []):
        at = instant(entry["valid_time"])
        values = _frame(path(entry), names, shape, mesh_hash, "kg kg-1", at)
        gpu = {n: cp.asarray(v, dtype=cp.float32) for n, v in values.items()}
        if not all(bool(cp.all(cp.isfinite(v) & (v >= 0))) for v in gpu.values()):
            raise ValueError("boundary frame contains invalid mixing ratio")
        boundary.append((at, gpu))
    boundary.sort(key=lambda item: item[0])
    if len({at for at, _ in boundary}) != len(boundary):
        raise ValueError("duplicate chemistry boundary time")
    if regional and len(boundary) < 2:
        raise ValueError("regional chemistry needs bracketing lateral composition frames")

    def update(atmosphere, transport, elapsed, dt=0.):
        now = start + elapsed
        if not math.isfinite(elapsed) or not math.isfinite(dt) or dt < 0:
            raise ValueError("chemistry update needs finite elapsed time and nonnegative dt")
        end = now + dt
        active = [(a, b, v) for a, b, v in emissions
                  if (max(a, now) < min(b, end) if dt else a <= now < b)]
        # Species without sources in this recipe are explicit in the config.
        inactive = set(config.get("zero_emission_species", []))
        if not inactive <= set(names):
            raise ValueError("unknown zero-emission species")
        supplied = set().union(*(v.keys() for _, _, v in active))
        if supplied | inactive != set(names) or supplied & inactive:
            raise ValueError(f"chemistry emission coverage missing or contradictory at {now}")
        if dt:
            coverage = {n: 0. for n in names if n not in inactive}
            for a, b, values in active:
                for name in values:
                    coverage[name] += min(b, end)-max(a, now)
            if any(abs(span-dt) > 1e-6 for span in coverage.values()):
                raise ValueError(f"chemistry emissions do not cover the complete timestep at {now}")
        for name, flux in transport.surface_flux.items():
            flux.fill(0)
        for a, b, values in active:
            weight = (min(b, end)-max(a, now))/dt if dt else 1.
            for name, value in values.items():
                transport.surface_flux[name][...] += value*weight
        if regional:
            if not boundary or now < boundary[0][0] or now > boundary[-1][0]:
                raise ValueError(f"lateral chemistry does not bracket {now}")
            right = next((i for i, (at, _) in enumerate(boundary) if at >= now), len(boundary)-1)
            left = max(0, right-1)
            ta, a = boundary[left]
            tb, b = boundary[right]
            weight = 0. if ta == tb else (now-ta)/(tb-ta)
            for name in names:
                target = atmosphere.state.scalars[transport.row.index_of(name)]
                target[:, boundary_cells] = (
                    a[name][:, boundary_cells]*(1-weight)
                    + b[name][:, boundary_cells]*weight)
    runtime.boundary = update
    update(runtime.driver.atmosphere, runtime, float(runtime.driver.atmosphere.state.time_seconds))
    files = [initial_path] + [path(e) for e in config.get("emissions", [])+config.get("boundary", [])]
    return {"schema": "gpuwm-hex.chem-input-binding/v1", "start_unix": start,
            "mesh_sha256": mesh_hash, "files": {p: digest(p) for p in files},
            "initial_species": names, "emission_frames": len(emissions),
            "boundary_frames": len(boundary), "missing_coverage": "refuse"}


def add_parser(commands):
    parser = commands.add_parser("chem-sources", help="decode table-declared chemistry inputs in Rust")
    sub = parser.add_subparsers(dest="chem_source_action", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--table", type=Path, default=TABLE)
    listing.set_defaults(handler=lambda a: print(Path(a.table).read_text()) or 0)
    fetcher = sub.add_parser("fetch")
    decoder = sub.add_parser("prepare")
    for p in (fetcher, decoder):
        p.add_argument("--source", required=True)
        p.add_argument("--table", type=Path, default=TABLE)
        p.add_argument("--valid-time", required=True)
        p.add_argument("--output", required=True, type=Path)
        p.add_argument("--research-license")
    fetcher.set_defaults(handler=lambda a: print(json.dumps(fetch(
        source_row(a.source, a.table, research_license=a.research_license),
        a.valid_time, a.output), indent=2)) or 0)
    decoder.add_argument("--input", required=True, type=Path)
    decoder.add_argument("--mesh", required=True, type=Path)
    decoder.add_argument("--pressure-file", type=Path)
    decoder.add_argument("--humidity-input", type=Path)
    decoder.add_argument("--pressure-variables", nargs="+", default=[])
    decoder.add_argument("--time-index", type=int, default=0)
    decoder.add_argument("--threads", type=int, default=4)
    decoder.add_argument("--max-distance-m", type=float, default=200000.)
    decoder.add_argument("--map-output", type=Path)
    decoder.add_argument("--map-spacing-deg", type=float, default=1.)
    decoder.add_argument("--map-bounds", nargs=4, type=float, default=[-180,180,-90,90])
    decoder.add_argument("--species", nargs="+", default=[])
    decoder.set_defaults(handler=lambda a: print(json.dumps(prepare(
        row=source_row(a.source, a.table, research_license=a.research_license),
        input=a.input, mesh=a.mesh, output=a.output, valid_time=a.valid_time,
        pressure_file=a.pressure_file, humidity_input=a.humidity_input, pressure_variables=a.pressure_variables,
        time_index=a.time_index, threads=a.threads, max_distance_m=a.max_distance_m,
        expected_species=a.species, map_output=a.map_output,
        map_spacing_deg=a.map_spacing_deg, map_bounds=a.map_bounds), indent=2)) or 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers(required=True))
    args = parser.parse_args()
    raise SystemExit(args.handler(args))
