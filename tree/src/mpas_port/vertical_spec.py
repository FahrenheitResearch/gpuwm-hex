"""Declarative native-free vertical specification and artifact producer.

There is one production numerical implementation: :mod:`mpas_port.vertical`.
This module validates a versioned JSON declaration, calls that implementation,
and materializes the same small NetCDF contract the existing Rust init engine
already consumes through its capsule ABI.  A native MPAS file is therefore an
optional oracle/compatibility input, never a default-path dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

from .errors import MpasPortError
from .vertical import (
    VerticalEdgeMetrics,
    VerticalGrid,
    build_edge_vertical_metrics,
    build_vertical_grid,
    runtime_vertical_vectors,
    validate_vertical_grid,
)

SCHEMA = "gpuwm-hex.vertical-spec/v1"
ARTIFACT_SCHEMA = "gpuwm-hex.native-free-vertical-artifact/v1"
RECEIPT_SCHEMA = "gpuwm-hex.native-free-vertical-receipt/v1"

#: The engine soil column is fixed: rw_mpas_init interpolates the met file's
#: first-guess soil layers onto MPAS-A v8.4.1's four Noah levels, whatever
#: the source published (nine RUC nodes, one ISBA layer, two ordinal layers).
MET_STATE_SOIL_LEVELS = 4

#: Length of the ``initial_time`` character field in an init-class file.
MET_STATE_STRLEN = 64

#: The met-state variables ``rw_mpas_init`` computes and writes.  The engine
#: lays out its output file from the capsule's schema VERBATIM and refuses by
#: name when a computed value has no variable to land in ("this run computed
#: N value(s) the init file has no variable for"), so a native-free vertical
#: artifact must DECLARE these variables even though it holds no meteorology:
#: the declarations are the landing sites, the engine supplies every number.
#: Names, dtypes, dimension orders and Registry attributes mirror a
#: native-lineage v8.4.1 init file field for field (units/long_name read off
#: the native x1.40962 init, 2026-08-24); ``initial_time`` is the one
#: non-record character field.  The emitter copies variable attributes from
#: the capsule verbatim, so the attributes declared here are the attributes
#: the mint carries.  A variable declared here but not computed for some
#: source is carried as explicit zeros and the engine's emit ledger records
#: it as carried, not computed -- the receipt, not the reader, says which.
MET_STATE_VARIABLES: tuple[tuple[str, tuple[str, ...], str, str], ...] = (
    ("dzs", ("Time", "nCells", "nSoilLevels"), "m", "soil layer thickness"),
    ("precipw", ("Time", "nCells"), "kg m^{-2}", "precipitable water"),
    ("q2", ("Time", "nCells"), "kg kg^{-1}", "2-meter specific humidity"),
    ("qc", ("Time", "nCells", "nVertLevels"), "kg kg^{-1}", "Cloud water mixing ratio"),
    ("qr", ("Time", "nCells", "nVertLevels"), "kg kg^{-1}", "Rain water mixing ratio"),
    ("qv", ("Time", "nCells", "nVertLevels"), "kg kg^{-1}", "Water vapor mixing ratio"),
    ("relhum", ("Time", "nCells", "nVertLevels"), "percent", "Relative humidity"),
    ("rh2", ("Time", "nCells"), "percent", "2-meter relative humidity"),
    ("rho", ("Time", "nCells", "nVertLevels"), "kg m^{-3}", "Dry air density"),
    ("rho_base", ("Time", "nCells", "nVertLevels"), "kg m^{-3}", "Base state dry air density"),
    ("seaice", ("Time", "nCells"), "unitless", "sea-ice flag (0=no seaice; =1 otherwise)"),
    ("sfc_albbck", ("Time", "nCells"), "unitless", "background surface albedo"),
    ("sh2o", ("Time", "nCells", "nSoilLevels"), "m3 m^{-3}", "soil equivalent liquid water"),
    ("skintemp", ("Time", "nCells"), "K", "ground or water surface temperature"),
    ("smois", ("Time", "nCells", "nSoilLevels"), "m3 m^{-3}", "soil moisture"),
    ("snow", ("Time", "nCells"), "kg m^{-2}", "snow water equivalent"),
    ("snowc", ("Time", "nCells"), "unitless", "flag for snow on ground (=0 no snow; =1,otherwise"),
    ("snowh", ("Time", "nCells"), "m", "physical snow depth"),
    ("sst", ("Time", "nCells"), "K", "sea-surface temperature"),
    ("surface_pressure", ("Time", "nCells"), "Pa", "Diagnosed surface pressure"),
    ("t2m", ("Time", "nCells"), "K", "2-meter temperature"),
    ("theta", ("Time", "nCells", "nVertLevels"), "K", "Potential temperature"),
    ("theta_base", ("Time", "nCells", "nVertLevels"), "K", "Base state potential temperature"),
    ("tke", ("Time", "nCells", "nVertLevels"), "m^2 s^{-2}", "Turbulent kinetic energy for the prognostic tke LES scheme"),
    ("tmn", ("Time", "nCells"), "K", "deep soil temperature"),
    ("tslb", ("Time", "nCells", "nSoilLevels"), "K", "soil layer temperature"),
    ("u", ("Time", "nEdges", "nVertLevels"), "m s^{-1}", "Horizontal normal velocity at edges"),
    ("u10", ("Time", "nCells"), "m s^{-1}", "10-meter zonal wind"),
    ("v10", ("Time", "nCells"), "m s^{-1}", "10-meter meridional wind"),
    ("vegfra", ("Time", "nCells"), "percent", "vegetation fraction"),
    ("w", ("Time", "nCells", "nVertLevelsP1"), "m s^{-1}", "Vertical velocity at vertical cell faces"),
    ("xice", ("Time", "nCells"), "unitless", "fractional area coverage of sea-ice"),
    ("xland", ("Time", "nCells"), "unitless", "land-ocean mask (1=land including sea-ice ; 2=ocean)"),
    ("zs", ("Time", "nCells", "nSoilLevels"), "m", "depth of centers of soil layers"),
)

#: Reference profiles the v8.4.1 real-data init carries as zeros (measured on
#: the native-lineage x4.163842 init: every value exactly 0.0, float32).  The
#: forecast driver's perturbation-Coriolis path refuses an init without
#: ``u_init``/``v_init``, so the artifact declares all four the way the
#: native file does.  These are NOT record variables.
REFERENCE_PROFILE_VARIABLES: tuple[tuple[str, tuple[str, ...], str, str], ...] = (
    ("u_init", ("nVertLevels",), "m s^{-1}", "u reference profile"),
    ("v_init", ("nVertLevels",), "m s^{-1}", "v reference profile"),
    ("qv_init", ("nVertLevels",), "kg kg^{-1}", "qv reference profile"),
    ("t_init", ("nCells", "nVertLevels"), "K", "theta reference profile"),
)

#: Init-stream slots the engine never computes and a real-data init carries
#: as exact zeros (measured on the native x1.40962 init, 2026-08-24: every
#: value 0.0).  ``Time`` is declared separately because its ``units``
#: attribute is minted from this run's start time.  Without these three a
#: native-free mint is three variables short of the native schema.
ZERO_STATE_VARIABLES: tuple[tuple[str, tuple[str, ...], str, str], ...] = (
    ("dz", ("Time", "nCells", "nSoilLevels"), "m", "depth of soil layer bottom"),
    ("h_oml_initial", ("Time", "nCells"), "m", "Initial depth of ocean mix layer"),
)


class VerticalSpecError(MpasPortError):
    """A vertical declaration or produced artifact violates a named contract."""


@dataclass(frozen=True, slots=True)
class VerticalSpec:
    schema: str = SCHEMA
    n_vert_levels: int = 55
    ztop_m: float = 30_000.0
    scheme: str = "tc"
    specified_interfaces_m: tuple[float, ...] | None = None
    interface_projection: str = "linear_interpolation"
    terrain_smoothing_passes: int = 1
    smooth_surfaces: bool = True
    surface_smoothing_passes: int = 30
    minimum_layer_fraction: float = 0.3
    hybrid_coordinate: bool = True
    hybrid_transition_height_m: float = 30_000.0
    rayleigh_xnutr: float = 0.0
    rayleigh_damping_start_m: float = 22_000.0
    theta_adv_order: int = 3
    coef_3rd_order: float = 0.25

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "VerticalSpec":
        if not isinstance(raw, Mapping):
            raise VerticalSpecError("vertical spec root must be a JSON object")
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise VerticalSpecError(
                f"vertical spec contains unknown key(s) {unknown}; a new option needs a versioned schema change"
            )
        values = dict(raw)
        if "specified_interfaces_m" in values and values["specified_interfaces_m"] is not None:
            if not isinstance(values["specified_interfaces_m"], (list, tuple)):
                raise VerticalSpecError("specified_interfaces_m must be a JSON array or null")
            values["specified_interfaces_m"] = tuple(
                float(value) for value in values["specified_interfaces_m"]
            )
        try:
            result = cls(**values)
        except TypeError as error:
            raise VerticalSpecError(f"vertical spec cannot be constructed: {error}") from error
        result.validate()
        return result

    @classmethod
    def from_file(cls, path: str | Path) -> "VerticalSpec":
        source = Path(path).expanduser().resolve(strict=True)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise VerticalSpecError(f"cannot read vertical spec {source}: {error}") from error
        return cls.from_mapping(raw)

    def validate(self) -> None:
        if self.schema != SCHEMA:
            raise VerticalSpecError(
                f"vertical spec schema {self.schema!r} is unsupported; expected {SCHEMA!r}"
            )
        if isinstance(self.n_vert_levels, bool) or not isinstance(self.n_vert_levels, int):
            raise VerticalSpecError("n_vert_levels must be an integer")
        if self.n_vert_levels < 3:
            raise VerticalSpecError("n_vert_levels must be at least 3")
        if self.scheme not in {"tc", "legacy", "specified"}:
            raise VerticalSpecError("scheme must be one of tc, legacy, specified")
        if self.interface_projection not in {"linear_interpolation", "layer_integral"}:
            raise VerticalSpecError(
                "interface_projection must be linear_interpolation or layer_integral"
            )
        for name in (
            "ztop_m",
            "minimum_layer_fraction",
            "hybrid_transition_height_m",
            "rayleigh_xnutr",
            "rayleigh_damping_start_m",
            "coef_3rd_order",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise VerticalSpecError(f"{name} must be finite")
        if self.ztop_m <= 0.0:
            raise VerticalSpecError("ztop_m must be positive")
        if not 0.0 < self.minimum_layer_fraction < 1.0:
            raise VerticalSpecError("minimum_layer_fraction must lie strictly between zero and one")
        if self.hybrid_coordinate and self.hybrid_transition_height_m <= 0.0:
            raise VerticalSpecError("hybrid_transition_height_m must be positive when hybrid_coordinate=true")
        if self.rayleigh_xnutr != 0.0 and not self.rayleigh_damping_start_m < self.ztop_m:
            raise VerticalSpecError(
                "nonzero rayleigh_xnutr requires rayleigh_damping_start_m < ztop_m"
            )
        for name in ("terrain_smoothing_passes", "surface_smoothing_passes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise VerticalSpecError(f"{name} must be a non-negative integer")
        if self.theta_adv_order not in (2, 3, 4):
            raise VerticalSpecError("theta_adv_order must be 2, 3, or 4")
        if self.scheme == "specified":
            if self.specified_interfaces_m is None:
                raise VerticalSpecError(
                    "scheme='specified' requires specified_interfaces_m"
                )
            expected = self.n_vert_levels + 1
            if len(self.specified_interfaces_m) != expected:
                raise VerticalSpecError(
                    f"specified_interfaces_m has {len(self.specified_interfaces_m)} values; expected {expected}"
                )
            values = np.asarray(self.specified_interfaces_m, dtype=np.float64)
            if not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0.0):
                raise VerticalSpecError(
                    "specified_interfaces_m must be finite and strictly increasing"
                )
            if values[0] != 0.0:
                raise VerticalSpecError("specified_interfaces_m[0] must be exactly 0.0 m")
            if not np.isclose(values[-1], self.ztop_m, rtol=0.0, atol=1.0e-9):
                raise VerticalSpecError(
                    f"specified_interfaces_m[-1]={values[-1]} does not equal ztop_m={self.ztop_m}"
                )
        elif self.specified_interfaces_m is not None:
            raise VerticalSpecError(
                "specified_interfaces_m is only valid when scheme='specified'; remove it or select that branch"
            )

    def to_mapping(self) -> dict[str, Any]:
        self.validate()
        raw = asdict(self)
        if self.specified_interfaces_m is not None:
            raw["specified_interfaces_m"] = list(self.specified_interfaces_m)
        return raw

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_mapping(), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        ).encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _raw_array_receipt(array: object) -> dict[str, Any]:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    selected = np.asarray(value, dtype=np.float64).ravel()
    finite = np.isfinite(selected)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "raw_sha256": digest.hexdigest(),
        "finite": bool(np.all(finite)),
        "minimum": float(np.min(selected[finite])) if np.any(finite) else None,
        "maximum": float(np.max(selected[finite])) if np.any(finite) else None,
    }


def build_vertical_from_spec(mesh: object, spec: VerticalSpec) -> tuple[VerticalGrid, VerticalEdgeMetrics]:
    """Construct the in-memory vertical contract through the sole numeric path."""

    spec.validate()
    terrain = np.asarray(getattr(mesh, "ter"))
    if terrain.dtype.kind != "f":
        terrain = terrain.astype(np.float64)
    vertical = build_vertical_grid(
        mesh,
        terrain,
        n_vert_levels=spec.n_vert_levels,
        ztop=spec.ztop_m,
        scheme=spec.scheme,  # type: ignore[arg-type]
        specified_zw=(
            None
            if spec.specified_interfaces_m is None
            else np.asarray(spec.specified_interfaces_m, dtype=terrain.dtype)
        ),
        interface_projection=spec.interface_projection,  # type: ignore[arg-type]
        terrain_smoothing_passes=spec.terrain_smoothing_passes,
        smooth_surfaces=spec.smooth_surfaces,
        surface_smoothing_passes=spec.surface_smoothing_passes,
        minimum_layer_fraction=spec.minimum_layer_fraction,
        hybrid_coordinate=spec.hybrid_coordinate,
        hybrid_transition_height=spec.hybrid_transition_height_m,
        xnutr=spec.rayleigh_xnutr,
        damping_start=spec.rayleigh_damping_start_m,
    )
    edge = build_edge_vertical_metrics(
        mesh, vertical, theta_adv_order=spec.theta_adv_order
    )
    return vertical, edge


def _ensure_dimension(dataset: Any, name: str, size: int) -> None:
    if name in dataset.dimensions:
        observed = len(dataset.dimensions[name])
        if observed != size:
            raise VerticalSpecError(
                f"vertical artifact template dimension {name}={observed}, constructed value requires {size}"
            )
    else:
        dataset.createDimension(name, size)


def _write_variable(
    dataset: Any,
    name: str,
    data: object,
    dimensions: tuple[str, ...],
    *,
    units: str,
    description: str,
) -> None:
    # The init-class file ABI is single precision: rw_mpas_init's emitter
    # carries Float/Int/Char and refuses Double by name ("carried variable
    # cf1 is Double, which this emitter does not copy" -- measured on real
    # x1.40962 assets, 2026-08-24), and every native-lineage init stores the
    # vertical contract as float32.  Construction stays float64; the on-disk
    # artifact is the native single-precision contract.
    value = np.asarray(data, dtype=np.float32)
    if name in dataset.variables:
        variable = dataset.variables[name]
        if tuple(variable.dimensions) != dimensions:
            raise VerticalSpecError(
                f"template variable {name} dimensions {variable.dimensions} != required {dimensions}"
            )
        if tuple(variable.shape) != tuple(value.shape):
            raise VerticalSpecError(
                f"template variable {name} shape {variable.shape} != constructed {value.shape}"
            )
    else:
        variable = dataset.createVariable(name, value.dtype, dimensions)
    variable[...] = value
    variable.units = units
    variable.description = description
    variable.gpuwm_hex_origin = ARTIFACT_SCHEMA


def _write_scalar(dataset: Any, name: str, value: float, description: str) -> None:
    # Same single-precision ABI as _write_variable: cf1..cf3 are float32 in
    # every native-lineage init and the engine's emitter refuses Double.
    data = np.asarray(value, dtype=np.float32)
    if name in dataset.variables:
        variable = dataset.variables[name]
        if variable.dimensions != ():
            raise VerticalSpecError(f"template variable {name} is not scalar")
    else:
        variable = dataset.createVariable(name, data.dtype, ())
    variable.assignValue(data)
    variable.units = "unitless"
    variable.description = description
    variable.gpuwm_hex_origin = ARTIFACT_SCHEMA


def _declare_met_state_schema(
    dataset: Any, *, start_time: str | None = None
) -> list[str]:
    """Declare the engine's met-state landing sites in the artifact.

    Returns the list of variable names declared by THIS call (a template
    already carrying a name keeps it, shape-checked).  Every declared
    record variable is materialized as one explicit zero record so a
    field the engine carries instead of computing reads as defined zeros,
    never as library fill values.

    ``start_time`` (``YYYY-MM-DD_hh:mm:ss``) mints the ``Time``
    coordinate's CF ``units``; when it is unknown the variable is still
    declared so the schema stays complete, without a units claim it
    cannot back.
    """

    if "Time" not in dataset.dimensions:
        dataset.createDimension("Time", None)
    elif not dataset.dimensions["Time"].isunlimited():
        raise VerticalSpecError(
            "vertical artifact template declares a fixed Time dimension; "
            "an init-class file needs Time unlimited"
        )
    _ensure_dimension(dataset, "nSoilLevels", MET_STATE_SOIL_LEVELS)
    _ensure_dimension(dataset, "StrLen", MET_STATE_STRLEN)

    def _declare(
        name: str,
        dims: tuple[str, ...],
        units: str,
        long_name: str,
    ) -> Any | None:
        if name in dataset.variables:
            observed = tuple(dataset.variables[name].dimensions)
            if observed != dims:
                raise VerticalSpecError(
                    f"template variable {name} dimensions {observed} != "
                    f"required {dims}"
                )
            return None
        variable = dataset.createVariable(name, np.float32, dims)
        variable.units = units
        variable.long_name = long_name
        variable.gpuwm_hex_origin = ARTIFACT_SCHEMA
        return variable

    declared: list[str] = []
    for name, dims, units, long_name in MET_STATE_VARIABLES + ZERO_STATE_VARIABLES:
        if _declare(name, dims, units, long_name) is not None:
            declared.append(name)
    if "Time" not in dataset.variables:
        variable = dataset.createVariable("Time", np.float32, ("Time",))
        if start_time is not None:
            variable.units = f"seconds since {start_time.replace('_', ' ')}"
        variable.long_name = "CF-compliant valid time"
        variable.standard_name = "time"
        variable.gpuwm_hex_origin = ARTIFACT_SCHEMA
        declared.append("Time")
    if "initial_time" not in dataset.variables:
        variable = dataset.createVariable("initial_time", "S1", ("StrLen",))
        variable.units = "YYYY-MM-DD_hh:mm:ss"
        variable.long_name = "Model initialization time"
        variable.gpuwm_hex_origin = ARTIFACT_SCHEMA
        variable[:] = np.frombuffer(
            b" " * MET_STATE_STRLEN, dtype="S1")
        declared.append("initial_time")
    for name, dims, units, long_name in REFERENCE_PROFILE_VARIABLES:
        variable = _declare(name, dims, units, long_name)
        if variable is None:
            continue
        variable[...] = np.zeros(
            tuple(len(dataset.dimensions[d]) for d in dims), dtype=np.float32)
        declared.append(name)
    # One explicit zero record: defined bytes for any carried field.
    for name, dims, _, _ in MET_STATE_VARIABLES + ZERO_STATE_VARIABLES:
        variable = dataset.variables[name]
        shape = tuple(len(dataset.dimensions[d]) for d in dims[1:])
        variable[0, ...] = np.zeros(shape, dtype=np.float32)
    dataset.variables["Time"][0] = np.float32(0.0)
    return declared


def _derived_geometry_arrays(
    mesh: Any, cache_file: Path | None, cache_key: str
) -> tuple[dict[str, Any], str]:
    """Compute (or reload) the derived geometry, returning (arrays, source).

    The four fields are pure functions of the grid/static pair and of the
    frozen transcriptions in :mod:`mpas_port.vector`; ``cache_key`` digests
    exactly those inputs, so a cache hit is the same numbers the compute
    branch would produce.  The RBF reconstruction solve is a per-cell
    Fortran-transcribed Gaussian elimination -- minutes of pure Python at
    real mesh sizes -- which is why the reload path exists at all.  A cache
    file whose recorded key differs is ignored and rewritten, never trusted.
    """

    from .vector import (
        initialize_reconstruction_coefficients,
        initialize_vector_geometry,
    )

    if cache_file is not None and cache_file.is_file():
        with np.load(cache_file) as stored:
            if str(stored["cache_key"]) == cache_key:
                return (
                    {
                        "edgeNormalVectors": stored["edgeNormalVectors"],
                        "localVerticalUnitVectors": stored[
                            "localVerticalUnitVectors"],
                        "cellTangentPlane": stored["cellTangentPlane"],
                        "coeffs_reconstruct": stored["coeffs_reconstruct"],
                    },
                    "cache",
                )
    geometry = initialize_vector_geometry(mesh)
    arrays = {
        "edgeNormalVectors": geometry.edge_normal_vectors,
        "localVerticalUnitVectors": geometry.local_vertical_unit_vectors,
        "cellTangentPlane": geometry.cell_tangent_plane,
        "coeffs_reconstruct": initialize_reconstruction_coefficients(mesh),
    }
    if cache_file is not None:
        tmp = cache_file.with_name(cache_file.name + f".tmp-{os.getpid()}")
        np.savez_compressed(tmp, cache_key=np.str_(cache_key), **arrays)
        # np.savez appends .npz to a name without it; normalize.
        produced = tmp if tmp.is_file() else tmp.with_name(tmp.name + ".npz")
        os.replace(produced, cache_file)
    return arrays, "computed"


def _complete_derived_geometry(
    dataset: Any,
    mesh: Any,
    *,
    cache_file: Path | None = None,
    cache_key: str = "",
) -> tuple[list[str], str]:
    """Fill the derived mesh-geometry fields MPAS computes at initialization.

    ``edgeNormalVectors``, ``localVerticalUnitVectors``, ``cellTangentPlane``
    and ``coeffs_reconstruct`` are declared by the published/generated static
    files but carried as zeros: native MPAS fills them inside initialization
    (see :func:`mpas_port.vector.initialize_reconstruction_coefficients`),
    and the forecast driver refuses an init whose edge normals are not unit
    vectors -- measured on real x1.40962 assets, 2026-08-24 ("initialized
    edge normals are not unit vectors: norm envelope (0.0, 0.0)").  On the
    native path the Fortran init supplied them through the capsule; on the
    native-free path this door is the initialization, so it computes them
    with the port's own frozen transcriptions.  A field that already holds
    nonzero values is left untouched and reported as carried.
    """

    values, source = _derived_geometry_arrays(mesh, cache_file, cache_key)
    completed: list[str] = []
    for name, value in values.items():
        variable = dataset.variables.get(name)
        if variable is None:
            raise VerticalSpecError(
                f"static template declares no {name} variable; the generated "
                "static writer emits it (as zeros) and this door fills it -- "
                "use a complete static file"
            )
        data = np.asarray(value, dtype=np.float32)
        if tuple(variable.shape) != data.shape:
            raise VerticalSpecError(
                f"derived geometry {name} shape {data.shape} != static "
                f"template {tuple(variable.shape)}"
            )
        if float(np.abs(variable[:]).max()) > 0.0:
            continue  # a future static writer may fill these; keep its bytes
        variable[...] = data
        variable.gpuwm_hex_origin = ARTIFACT_SCHEMA
        completed.append(name)
    return completed, source


def materialize_vertical_artifact(
    *,
    grid: str | Path,
    static: str | Path,
    spec_path: str | Path,
    output: str | Path,
    receipt_path: str | Path | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a durable vertical NetCDF artifact and provenance receipt.

    The static file is copied first so all mesh/statics carriers remain
    available to the existing Rust init ABI.  The generated vertical fields
    then replace/add the native ``vertical_stage_out`` contract.  No native
    initialization file is opened on this path.
    """

    try:
        from netCDF4 import Dataset
    except ImportError as error:  # pragma: no cover - environment-specific
        raise VerticalSpecError(
            f"netCDF4 is required to materialize a vertical artifact ({error}); install netCDF4"
        ) from error
    from .mesh import Mesh

    grid_path = Path(grid).expanduser().resolve(strict=True)
    static_path = Path(static).expanduser().resolve(strict=True)
    spec_file = Path(spec_path).expanduser().resolve(strict=True)
    out = Path(output).expanduser().resolve()
    receipt = (
        Path(receipt_path).expanduser().resolve()
        if receipt_path is not None
        else out.with_name(out.name + ".receipt.json")
    )
    if not out.parent.is_dir():
        raise VerticalSpecError(
            f"vertical artifact output directory {out.parent} does not exist; create it"
        )
    if not receipt.parent.is_dir():
        raise VerticalSpecError(
            f"vertical receipt directory {receipt.parent} does not exist; create it"
        )
    if out == static_path:
        raise VerticalSpecError("vertical artifact output must not overwrite the static authority")

    spec = VerticalSpec.from_file(spec_file)
    mesh = Mesh.from_netcdf(grid_path, static_path)
    grid_sha = _sha256_file(grid_path)
    static_sha = _sha256_file(static_path)
    geometry_producer_sha = _sha256_file(
        Path(__file__).with_name("vector.py"))
    geometry_cache_key = hashlib.sha256(
        f"{grid_sha}:{static_sha}:{geometry_producer_sha}".encode("ascii")
    ).hexdigest()
    geometry_cache_file = out.parent / (
        f"derived-geometry-{geometry_cache_key[:12]}.npz")
    vertical, edge = build_vertical_from_spec(mesh, spec)
    invariants = validate_vertical_grid(
        vertical,
        n_cells=int(mesh.dimensions["nCells"]),
        n_edges=int(mesh.dimensions["nEdges"]),
    )
    runtime = runtime_vertical_vectors(vertical)

    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    tmp.unlink(missing_ok=True)
    shutil.copyfile(static_path, tmp)
    try:
        with Dataset(tmp, mode="r+") as dataset:
            ncells = int(mesh.dimensions["nCells"])
            nedges = int(mesh.dimensions["nEdges"])
            nlev = spec.n_vert_levels
            _ensure_dimension(dataset, "nCells", ncells)
            _ensure_dimension(dataset, "nEdges", nedges)
            _ensure_dimension(dataset, "nVertLevels", nlev)
            _ensure_dimension(dataset, "nVertLevelsP1", nlev + 1)
            _ensure_dimension(dataset, "TWO", 2)

            # Native on-disk orientation is entity-major; logical authority is
            # level-major.  The explicit transposes are part of the contract.
            _write_variable(dataset, "ter", vertical.hx[0], ("nCells",), units="m", description="smoothed terrain height")
            _write_variable(dataset, "hx", vertical.hx.T, ("nCells", "nVertLevelsP1"), units="m", description="terrain influence in vertical coordinate")
            _write_variable(dataset, "zgrid", vertical.zgrid.T, ("nCells", "nVertLevelsP1"), units="m MSL", description="geometric height of layer interfaces")
            _write_variable(dataset, "zz", vertical.zz.T, ("nCells", "nVertLevels"), units="unitless", description="d(zeta)/dz vertical metric")
            _write_variable(dataset, "zxu", vertical.zxu.T, ("nEdges", "nVertLevels"), units="unitless", description="dz/dx on coordinate surfaces")
            _write_variable(dataset, "dss", vertical.dss.T, ("nCells", "nVertLevels"), units="unitless", description="w damping coefficient")
            _write_variable(dataset, "rdzw", runtime["rdzw"], ("nVertLevels",), units="unitless", description="reciprocal reference layer thickness")
            _write_variable(dataset, "dzu", runtime["dzu"], ("nVertLevels",), units="unitless", description="reference spacing at w levels")
            _write_variable(dataset, "rdzu", runtime["rdzu"], ("nVertLevels",), units="unitless", description="reciprocal dzu")
            _write_variable(dataset, "fzm", runtime["fzm"], ("nVertLevels",), units="unitless", description="lower interpolation weight")
            _write_variable(dataset, "fzp", runtime["fzp"], ("nVertLevels",), units="unitless", description="upper interpolation weight")
            _write_variable(dataset, "zb", np.transpose(edge.zb, (2, 1, 0)), ("nEdges", "TWO", "nVertLevelsP1"), units="unitless", description="u-to-omega edge metric")
            _write_variable(dataset, "zb3", np.transpose(edge.zb3, (2, 1, 0)), ("nEdges", "TWO", "nVertLevelsP1"), units="unitless", description="third-order u-to-omega correction")
            _write_scalar(dataset, "cf1", vertical.cf1, "surface extrapolation coefficient 1")
            _write_scalar(dataset, "cf2", vertical.cf2, "surface extrapolation coefficient 2")
            _write_scalar(dataset, "cf3", vertical.cf3, "surface extrapolation coefficient 3")
            met_state_declared = _declare_met_state_schema(
                dataset,
                start_time=(
                    str((run_config or {}).get("config_start_time"))
                    if (run_config or {}).get("config_start_time") is not None
                    else None
                ),
            )
            geometry_completed, geometry_source = _complete_derived_geometry(
                dataset, mesh,
                cache_file=geometry_cache_file,
                cache_key=geometry_cache_key,
            )
            # The emitter copies capsule global attributes verbatim, and the
            # forecast driver parses the init's config_start_time -- a static
            # template carries placeholders ('0000-01-01_00:00:00',
            # config_nfglevels=1), so the door stamps the values this run
            # actually declares.  Measured refusal without this: "init
            # config_start_time is unparseable: '0000-01-01_00:00:00'".
            for key, value in (run_config or {}).items():
                if not str(key).startswith("config_"):
                    raise VerticalSpecError(
                        f"run_config key {key!r} is not a config_* attribute"
                    )
                if isinstance(value, bool):
                    value = "YES" if value else "NO"
                if isinstance(value, int) and not isinstance(value, bool):
                    value = np.int32(value)
                elif isinstance(value, float):
                    value = np.float32(value)
                dataset.setncattr(str(key), value)
            dataset.setncattr("config_nvertlevels", np.int32(spec.n_vert_levels))
            dataset.setncattr("config_ztop", np.float32(spec.ztop_m))
            dataset.gpuwm_hex_vertical_artifact_schema = ARTIFACT_SCHEMA
            dataset.gpuwm_hex_vertical_spec_schema = spec.schema
            dataset.gpuwm_hex_vertical_spec_sha256 = spec.sha256()
            dataset.gpuwm_hex_vertical_source = "constructed-no-native-runtime-input"
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)

    arrays = {
        "ter": vertical.hx[0],
        "zgrid": vertical.zgrid.T,
        "zz": vertical.zz.T,
        "zxu": vertical.zxu.T,
        "dss": vertical.dss.T,
        "rdzw": runtime["rdzw"],
        "dzu": runtime["dzu"],
        "fzm": runtime["fzm"],
        "fzp": runtime["fzp"],
        "zb": np.transpose(edge.zb, (2, 1, 0)),
        "zb3": np.transpose(edge.zb3, (2, 1, 0)),
    }
    payload = {
        "schema": RECEIPT_SCHEMA,
        "mode": "constructed",
        "native_runtime_dependency": False,
        "inputs": {
            "grid": {"path": str(grid_path), "bytes": grid_path.stat().st_size, "sha256": grid_sha},
            "static": {"path": str(static_path), "bytes": static_path.stat().st_size, "sha256": static_sha},
            "vertical_spec": {"path": str(spec_file), "bytes": spec_file.stat().st_size, "file_sha256": _sha256_file(spec_file), "canonical_sha256": spec.sha256(), "declaration": spec.to_mapping()},
            "producer_sources": {
                "vertical.py": _sha256_file(Path(__file__).with_name("vertical.py")),
                "vertical_spec.py": _sha256_file(Path(__file__)),
            },
        },
        "invariants": invariants,
        "run_config_attributes": {
            str(k): (v if isinstance(v, (str, int, float, bool)) else str(v))
            for k, v in (run_config or {}).items()
        },
        "derived_geometry": {
            "completed": geometry_completed,
            "source": geometry_source,
            "cache_file": str(geometry_cache_file),
            "cache_key": geometry_cache_key,
            "reason": (
                "static files declare edgeNormalVectors/localVerticalUnit"
                "Vectors/cellTangentPlane/coeffs_reconstruct as zeros; native "
                "MPAS fills them at initialization, so the native-free door "
                "computes them with the port's frozen transcriptions "
                "(mpas_vector_operations.F, mpas_vector_reconstruction.F, "
                "mpas_rbf_interpolation.F)"
            ),
        },
        "met_state_schema": {
            "soil_levels": MET_STATE_SOIL_LEVELS,
            "variables": (
                [name for name, *_ in MET_STATE_VARIABLES]
                + [name for name, *_ in ZERO_STATE_VARIABLES]
                + [name for name, *_ in REFERENCE_PROFILE_VARIABLES]
                + ["Time", "initial_time"]
            ),
            "declared_by_this_run": met_state_declared,
            "reason": (
                "rw_mpas_init lays out its output from the capsule schema "
                "verbatim and refuses computed values with no variable to "
                "land in; these declarations are the landing sites, every "
                "value is engine-supplied"
            ),
        },
        "arrays": {name: _raw_array_receipt(value) for name, value in arrays.items()},
        "output": {"path": str(out), "bytes": out.stat().st_size, "sha256": _sha256_file(out)},
        "claim": "grid + static + declarative vertical specification produced the complete vertical contract without reading a native init file",
        "nonclaims": [
            "this receipt alone does not establish equality to a compiled MPAS-A v8.4.1 oracle",
            "this receipt alone does not establish that the current sibling rw_mpas_init binary accepts the artifact ABI",
            "this receipt does not establish forecast stability or skill",
        ],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    receipt.write_text(rendered, encoding="utf-8", newline="\n")
    return payload
