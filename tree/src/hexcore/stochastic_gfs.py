"""GFS analysis transport, existing Rust GRIB stacks and GPU mesh remapping."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import tempfile
import threading
import time
import urllib.request

from .stochastic_verification import utc_time

# Raw WMO identifiers used by the engine's GFS bridge. Pressure levels are Pa.
SELECTORS = {
    "z500": ("HGT", "500 mb", 3, 5, 100, 50000.),
    "t2m": ("TMP", "2 m above ground", 0, 0, 103, 2.),
    "psfc": ("PRES", "surface", 3, 0, 1, 0.),
}

# MPAS init stores coordinates as float32. Its encoded pole rounds above
# mathematical pi/2; the Rust reader widens that stored value to float64.
SINGLE_PRECISION_POLE_RADIANS = float.fromhex("0x1.921fb6p+0")


def selector(key):
    _, _, category, parameter, level_type, level = SELECTORS[key]
    return dict(key=key, discipline=0, category=category, parameter=parameter,
                level_type=level_type, levels=[level], constituent_type=None, aerosol_type=None)


def index_ranges(text, valid_time):
    """Select exactly three f000 analysis records from a GFS text index."""
    valid = utc_time(valid_time)
    rows = [line.split(":") for line in text.splitlines() if line.strip()]
    try:
        offsets = [int(row[1]) for row in rows]
    except (ValueError, IndexError) as error:
        raise ValueError("invalid GFS index offsets") from error
    if not offsets or offsets[0] < 0 or any(b <= a for a, b in zip(offsets, offsets[1:])):
        raise ValueError("GFS index offsets must increase strictly")
    result = {}
    for key, (tag, level, *_) in SELECTORS.items():
        matches = [i for i, row in enumerate(rows)
                   if len(row) >= 6 and row[3:6] == [tag, level, "anl"]
                   and row[2] == valid.strftime("d=%Y%m%d%H")]
        if len(matches) != 1 or matches[0]+1 == len(rows):
            raise ValueError(f"GFS index requires one bounded {key} analysis range")
        i = matches[0]
        begin, end = offsets[i], offsets[i+1]-1
        if end-begin+1 > 64*1024**2:
            raise ValueError("selected GFS range exceeds 64 MiB")
        result[key] = (begin, end)
    return result


def fetch_ranges(url, valid_time, folder, audit):
    """Public HTTPS byte transport only; field decoding stays in Rust."""
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("GFS object URL must use HTTPS")
    def get(address, limit, byte_range=None):
        headers = {"User-Agent": "hex-stochastic-verification/1"}
        if byte_range:
            headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
        before = time.perf_counter()
        with urllib.request.urlopen(urllib.request.Request(address, headers=headers), timeout=60) as response:
            if byte_range:
                expected = f"bytes {byte_range[0]}-{byte_range[1]}/"
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                    raise ValueError("GFS server did not honor the exact byte range")
            body = response.read(limit+1)
        if len(body) > limit or (byte_range and len(body) != limit):
            raise ValueError("GFS HTTP response length differs from request")
        return body, dict(url=address, byte_range=byte_range, bytes=len(body),
                          sha256=hashlib.sha256(body).hexdigest(),
                          wall_seconds=time.perf_counter()-before)
    raw, row = get(url+".idx", 4*1024**2)
    audit["index"] = row
    ranges = index_ranges(raw.decode("ascii"), valid_time)
    audit["downloads"] = {}
    def one(item):
        key, span = item
        body, row = get(url, span[1]-span[0]+1, span)
        audit["downloads"][key] = row
        path = folder/(key+".grib2")
        path.write_bytes(body)
        return key, path, row
    # Three independent fields are transported concurrently, with no full object download.
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(one, ranges.items()))
    return {key: path for key, path, _ in rows}


def regular_grid(grid):
    """Validate metadata for a global regular lat/lon grid in stored row order."""
    nx, ny, scan = grid["nx"], grid["ny"], grid["scan_mode"]
    if (type(nx) is not int or type(ny) is not int or nx < 2 or ny < 2
            or grid["template"] != 0 or scan not in (0, 64, 128, 192)):
        raise ValueError("GFS remap requires a global regular row-major lat/lon grid")
    lat1, lat2, lon1, lon2, dy, dx = [grid[k] for k in
        ("lat1", "lat2", "lon1", "lon2", "dlat", "dlon")]
    if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in (lat1, lat2, lon1, lon2, dy, dx)):
        raise ValueError("nonfinite GFS grid metadata")
    sx, sy = (-dx if scan & 128 else dx), (dy if scan & 64 else -dy)
    lon_error = (lon2-lon1-(nx-1)*sx+180.) % 360.-180.
    if (dx <= 0 or dy <= 0 or abs(nx*dx-360.) > 1e-6
            or abs(dy*(ny-1)-180.) > 1e-6
            or abs(lat1-(-90. if sy > 0 else 90.)) > 1e-6
            or abs(lat2-lat1-(ny-1)*sy) > 1e-6 or abs(lon_error) > 1e-6):
        raise ValueError("GFS grid endpoints/increments do not describe the whole sphere")
    return nx, ny, lat1, lon1, sy, sx


def remap_gpu(values, grid, latitude, longitude):
    """Bilinear point values at mesh cells; periodic longitude, no extrapolation.

    Positive-weight missing corners make the result missing. Zero-weight
    corners do not contaminate an exact grid point. All field work is GPU.
    Mesh coordinates are radians, matching latCell/lonCell in MPAS geometry.
    """
    import cupy as cp
    nx, ny, lat1, lon1, dy, dx = regular_grid(grid)
    if any(not isinstance(a, cp.ndarray) or a.dtype.kind != "f" for a in (values, latitude, longitude)):
        raise TypeError("GFS remap requires floating-point GPU arrays")
    if values.shape != (ny, nx) or latitude.ndim != 1 or not latitude.size or longitude.shape != latitude.shape:
        raise ValueError("GFS remap field/mesh shapes differ")
    if not bool(cp.all(cp.isfinite(latitude) & cp.isfinite(longitude)
                       & (cp.abs(latitude) <= SINGLE_PRECISION_POLE_RADIANS))):
        raise ValueError("mesh coordinates must be finite radians on the sphere")
    y = cp.clip((latitude*(180./math.pi)-lat1)/dy, 0., ny-1.)
    x = ((longitude*(180./math.pi)-lon1)*(1 if dx > 0 else -1) % 360.)/abs(dx)
    # Modulo also covers roundoff giving x == nx at the longitude seam.
    x %= nx
    j = cp.minimum(cp.floor(y).astype(cp.int64), ny-2)
    i = cp.floor(x).astype(cp.int64)
    wy, wx = y-j, x-i
    result = cp.zeros(latitude.shape, dtype=cp.float64)
    valid = cp.ones(latitude.shape, dtype=bool)
    for jj, ii, weight in ((j, i, (1-wy)*(1-wx)), (j, (i+1) % nx, (1-wy)*wx),
                           (j+1, i, wy*(1-wx)), (j+1, (i+1) % nx, wy*wx)):
        corner = values[jj, ii]
        valid &= (weight == 0) | cp.isfinite(corner)
        result += cp.where((weight > 0) & cp.isfinite(corner), corner, 0.)*weight
    return cp.where(valid, result, cp.nan)


def decode_stacks(paths, backend, folder, audit, valid_time):
    """Validate existing Rust stack transports, overlapping the three fields.

    The caller owns folder and payload cleanup. No field arithmetic occurs here.
    """
    def decode(key):
        before = time.perf_counter()
        specification = dict(schema="gpuwm-grib2-stack-v1", select=[selector(key)])
        spec_path, output = folder/(key+".json"), folder/key
        spec_path.write_text(json.dumps(specification, allow_nan=False))
        evidence = dict(input=str(paths[key]), specification=specification,
            output=str(output), native_thread_id=threading.get_native_id())
        audit.setdefault("decodes", {})[key] = evidence
        native_started = time.perf_counter()
        try:
            backend.grib2_stack(paths[key], spec_path, output)
        finally:
            evidence["native_call_wall_seconds"] = time.perf_counter()-native_started
        document = json.loads((output/"stack.json").read_text())
        rows = document.get("outputs", [])
        if document.get("schema") != specification["schema"] or len(rows) != 1:
            raise ValueError("GFS Rust stack schema or field roster differs")
        record = rows[0]
        if (record["key"] != key or record["levels"] != selector(key)["levels"]
                or record["pdt"] != 0 or record["forecast_step"] != 0
                or record["selector_values"] != [{k: v for k, v in selector(key).items() if k not in ("key", "levels")}]
                or utc_time(record["valid_time"]) != utc_time(valid_time)
                or utc_time(record["reference_time"]) != utc_time(valid_time)):
            raise ValueError("GFS stack must contain the requested instantaneous f000 analysis")
        nx, ny, *_ = regular_grid(record["grid"])
        filename = record["file"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("GFS decoded payload escapes its output directory")
        payload = output/filename
        if payload.is_symlink() or not payload.is_file() or payload.stat().st_size != nx*ny*4:
            raise ValueError("GFS decoded payload size/type differs")
        with payload.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                raise ValueError("GFS decoded payload checksum differs")
        evidence.update(wall_seconds=time.perf_counter()-before, manifest=record)
        return key, payload, record
    with ThreadPoolExecutor(max_workers=3) as pool:
        return list(pool.map(decode, SELECTORS))


def analysis_gpu(spec, valid_time, latitude, longitude, *, work, audit, bridge=None):
    """Decode three fields concurrently through the existing Rust stack ABI.

    The generic exporter is serial within a field. Three independent Rust
    calls run concurrently via ctypes (which releases the GIL). Actual CPU
    occupancy and GPU costs must still be measured on the execution box.
    Every owned raw download and decoded payload is removed on exit.
    """
    import cupy as cp
    from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
    started = time.perf_counter()
    row = dict(stage="GFS analysis", parallelism="3 concurrent Rust field workers; GPU remap",
               source={k: spec[k] for k in ("path", "url") if k in spec},
               valid_time=valid_time, observed_threads=None, decoded_files_deleted=[])
    audit.append(row)
    if bool(spec.get("path")) == bool(spec.get("url")):
        raise ValueError("GFS GRIB input requires exactly one local path or HTTPS object URL")
    try:
        backend = CpuPreprocessBackend(bridge)
    except Exception:
        row["wall_seconds"] = time.perf_counter()-started
        raise
    with Path(backend.path).open("rb") as stream:
        row["reader"] = dict(path=str(backend.path), sha256=hashlib.file_digest(stream, "sha256").hexdigest())
    temporary = tempfile.TemporaryDirectory(prefix="stochastic-gfs-", dir=work)
    with temporary as directory:
        folder = Path(directory)
        try:
            paths = (fetch_ranges(spec["url"], valid_time, folder, row) if spec.get("url")
                     else {key: Path(spec["path"]) for key in SELECTORS})
            decoded = decode_stacks(paths, backend, folder, row, valid_time)
            grid = decoded[0][2]["grid"]
            if any(record["grid"] != grid for _, _, record in decoded):
                raise ValueError("GFS analysis fields have different grids")
            fields = {key: cp.fromfile(payload, dtype="<f4").reshape(grid["ny"], grid["nx"])
                      for key, payload, _ in decoded}
            event_start, event_end = cp.cuda.Event(), cp.cuda.Event()
            event_start.record()
            # GFS may publish extrapolated isobaric height below terrain.
            # Exclude any positive-weight corner below 500 hPa for both arms.
            fields["z500"] = cp.where(cp.isfinite(fields["psfc"]) & (fields["psfc"] >= 50000.),
                                     fields["z500"], cp.nan)
            result = {key: remap_gpu(fields[key], grid, latitude, longitude) for key in ("z500", "t2m")}
            event_end.record()
            event_end.synchronize()
            row["remap_gpu_elapsed_ms"] = cp.cuda.get_elapsed_time(event_start, event_end)
            identity = hashlib.sha256(json.dumps([r for _, _, r in decoded], sort_keys=True).encode()).hexdigest()
            provenance = dict(method="GPU bilinear point remap; periodic longitude; no missing-corner renormalization",
                grid=grid, terrain_support="all positive-weight z500 corners have finite surface pressure >=50000 Pa",
                reader=row["reader"], fields={key: record["sha256"] for key, _, record in decoded})
            return result, "GFS-f000-"+identity, provenance
        finally:
            row["decoded_files_deleted"] = [dict(path=str(p), bytes=p.lstat().st_size)
                for p in folder.rglob("*") if p.is_file() or p.is_symlink()]
            before = time.perf_counter()
            temporary.cleanup()
            row["cleanup_wall_seconds"] = time.perf_counter()-before
            row["wall_seconds"] = time.perf_counter()-started
