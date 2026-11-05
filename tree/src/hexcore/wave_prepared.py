"""Transfer the shared Rust prepared contract to CUDA, without field decoding."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path


@dataclass
class Prepared:
    path: Path
    manifest: dict
    arrays: dict
    sha256: str


def coordinate_difference(mesh_radians, prepared_degrees, *, longitude=False, xp):
    """Compare two encodings of one coordinate within their rounding bounds.

    Hex stores mesh angles as float32 radians, while prepared/v1 exchanges
    float32 degrees. Compare in float64 and allow half an ulp from each
    stored encoding, instead of imposing a tolerance smaller than one ulp.
    This does not change either coordinate or regrid a field.
    """
    mesh = xp.rad2deg(mesh_radians.astype(xp.float64))
    prepared = prepared_degrees.astype(xp.float64)
    delta = mesh - prepared
    if longitude:
        delta = (delta + 180.) % 360. - 180.
    def ulp(value):
        away = xp.where(value < 0, -xp.inf, xp.inf).astype(value.dtype)
        return xp.abs(xp.nextafter(value, away) - value).astype(xp.float64)
    tolerance = (ulp(mesh_radians) * (.5 * 180. / 3.141592653589793)
                 + .5 * ulp(prepared_degrees) + 1.e-10)
    return xp.abs(delta), tolerance


def load(path):
    import cupy as cp
    from gpuwm.ocean.waves.prepared_run import NativeIO
    path = Path(path).resolve()
    raw = (path / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != "gpuwm.waves.prepared/v1" or manifest.get("status") != "complete":
        raise ValueError("hex waves require completed shared spectral preparation")
    required = {"lat", "lon", "depth", "wet", "area", "initial_partitions",
                "boundary_cells", "boundary_partitions", "u10_east", "v10_north"}
    if required - manifest.get("arrays", {}).keys():
        raise ValueError("spectral prepared arrays are incomplete")

    io = NativeIO()
    # Shared Rust byte reader, with independent verification/reads parallel.
    # Rust owns the decode, units, regridding, mask and partition preparation.
    with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 1, 24)) as pool:
        names = sorted(required)
        hosts = list(pool.map(lambda name:io.load(path,manifest["arrays"][name]), names))
    # Coupled stepping gets subsequent winds from the atmosphere. Only the
    # first prepared wind plane seeds initial partition diagnostics.
    arrays = {name:cp.asarray(host[0] if name in ("u10_east","v10_north") else host)
              for name,host in zip(names,hosts)}
    return Prepared(path, manifest, arrays, hashlib.sha256(raw).hexdigest())


def ensure(*, mesh, grid_path, start_time, end_time, output_root, options):
    path = options.get("prepared")
    if path is None:
        try:
            from gpuwm.ocean.waves.prepared import prepare_mesh
        except ImportError as exc:
            raise RuntimeError("hex wave automatic preparation requires the shared waves-data prepare_mesh entry point") from exc
        # That one shared entry point owns cache reuse, bathymetry, GFS-Wave,
        # stations and polygon-aware point-list preparation on Rust workers.
        preparation = options.get("preparation", {})
        if not isinstance(preparation, dict):
            raise ValueError("wave preparation options must be a JSON object")
        path = prepare_mesh(mesh=mesh, grid_path=grid_path, start_time=start_time,
            end_time=end_time, output_root=output_root, options=dict(options, **preparation))
    from .wave_stations import exposed_stations
    manifest = json.loads((Path(path) / "manifest.json").read_text())
    stations, mapping_sha = exposed_stations(path, manifest, output_root)
    prepared = load(path)
    prepared.manifest = dict(prepared.manifest, stations=stations, station_mapping_sha256=mapping_sha)
    return prepared
