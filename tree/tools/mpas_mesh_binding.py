#!/usr/bin/env python
"""Mesh, geometry, timestep, and memory binding for the frozen v8.4.1 path.

The frozen x4 proof remains an asserted no-op.  Every non-native mesh is bound
at runtime from a registry entry whose grid/static bytes, dimensions, nominal
resolution, *declared timestep*, and Courant policy are explicit.  The actual
stability length is the finite positive ``dcEdge`` array in the supplied
Earth-scaled static file; ``nominalMinDc`` is never used as its substitute.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from mpas_port.dual_edge_admission import (
    DualEdgeAdmission,
    DualEdgeAdmissionError,
    DualEdgePolicy,
    admit_dual_edges,
)
from mpas_port.timestep_admission import (
    CourantPolicy,
    EdgeLengthAuthority,
    TimestepAdmissionError,
    admit_timestep,
    edge_length_authority,
)

__all__ = [
    "MeshBindingError",
    "MeshBindingMismatch",
    "MeshBinding",
    "MESH_BINDINGS",
    "constants_fingerprint",
    "binding_fingerprint",
    "bind_mesh",
]


class MeshBindingError(RuntimeError):
    """Base class: a run is refused rather than executed under an unproved bind."""


class MeshBindingMismatch(MeshBindingError):
    """The declared mesh and supplied bytes/geometry/timestep do not agree."""


@dataclass(frozen=True)
class MeshBinding:
    name: str
    n_cells: int
    n_edges: int
    n_levels: int
    n_interfaces: int
    n_soil_levels: int
    nominal_dx_m: float
    dt_seconds: float
    grid_bytes: int
    grid_sha256: str
    static_bytes: int
    static_sha256: str
    courant_wave_speed_m_s: float = 125.0
    courant_safety_factor: float = 0.90
    frozen_native: bool = False
    scale_admission_floor: bool = True
    drop_carried_deformation: bool = False
    notes: str = ""

    def courant_policy(self) -> CourantPolicy:
        return CourantPolicy(
            max_characteristic_speed_m_s=self.courant_wave_speed_m_s,
            safety_factor=self.courant_safety_factor,
        )

    def dual_edge_policy(self) -> DualEdgePolicy:
        """One floor for every row, deliberately with no per-row override.

        The Courant policy is per-row because a wave speed is a modelling
        choice. Dual-edge amplification is not: it is a property of the mesh
        the operators inherit, so a row cannot admit itself past it. Changing
        the floor means editing mpas_port.dual_edge_admission, where the
        measured anchors that set it live.
        """

        return DualEdgePolicy()


MESH_BINDINGS: Mapping[str, MeshBinding] = MappingProxyType(
    {
        "x4.163842": MeshBinding(
            name="x4.163842",
            n_cells=163_842,
            n_edges=491_520,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=25_000.0,
            dt_seconds=120.0,
            grid_bytes=224_139_172,
            grid_sha256="48e747157bb1f0b83b96505e268699dfb562b4c1428468cb91457fbb03b1be55",
            static_bytes=298_860_376,
            static_sha256="f064ee8f8d40085db4bf77a3d5fc6081cd92368b7d3dd32d98110b8b64d177e8",
            frozen_native=True,
            scale_admission_floor=False,
            notes=(
                "Frozen native proof shape. Binding must change no module constant or "
                "trajectory; geometry is still independently Courant-admitted. "
                "Grandfathered under the named floors NATIVE_DEVICE_FLOOR (24 GiB) and "
                "NATIVE_RESTART_FLOOR (22 GiB), which predate the GF frame cut; the "
                "measured peak at the current pin is 20,902 MiB (2026-08-24, RTX 5090), "
                "and re-deriving the floors is a re-proof decision, not an edit. "
                "Static provenance (read from the file, 2026-08-24): native MPAS-A "
                "v8.4.1 init_atmosphere with config_native_gwd_static=YES; no writer "
                "of ours ever touched it, so the retired writer's antipodal drag band "
                "does not apply. Rebuilding it would break the dycore byte-identity "
                "anchor and is refused."
            ),
        ),
        "x1.40962": MeshBinding(
            name="x1.40962",
            n_cells=40_962,
            n_edges=122_880,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=120_000.0,
            dt_seconds=120.0,
            grid_bytes=56_039_332,
            grid_sha256="9a9e1909a755dac209462ceb0bfffd77ac1b37503169568b7f296707ee612bb9",
            static_bytes=94_766_584,
            static_sha256="cf1a47d4168327f06a8403555d6ed8b2fe1aff7f8b916bb7f6a754c34a10ac82",
            drop_carried_deformation=True,
            notes=(
                "Published 120 km mesh. Device admission scales with columns; timestep "
                "is explicit and independently checked against physical dcEdge. "
                "Static provenance (read from the file, 2026-08-24): the NCAR-published "
                "static, built by native init_atmosphere v8.2.0 on glade with "
                "config_native_gwd_static=YES; no writer of ours produced it, so the "
                "retired writer's antipodal drag band does not apply and it is kept."
            ),
        ),
        "v15.150.38857": MeshBinding(
            name="v15.150.38857",
            n_cells=38_857,
            n_edges=116_565,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=15_000.0,
            dt_seconds=60.0,
            grid_bytes=53_162_368,
            grid_sha256="0e6ac7c46140b24010764e840f3b1b77d52adb4abb16cac7cae4ce384b84c3b6",
            static_bytes=74_304_272,
            static_sha256="199c16ca993edfca9335b9e63b63db0a67e0eb201179d3dd1df1f9510420635f",
            drop_carried_deformation=True,
            notes=(
                "Generated 15-to-150 km variable-resolution mesh. It deliberately does "
                "not inherit x4's 120 s timestep: registry declares 60 s and the actual "
                "finite positive dcEdge minimum must admit it before CUDA contact. "
                "Static rebuilt 2026-08-24 by the unified rw_mpas_static (82-variable "
                "union): the previously pinned static came from the retired writer, "
                "whose drag band sampled terrain 180 degrees of longitude from every "
                "cell (archive-origin assumption); measured corr(old var2d, new var2d) "
                "= +0.003 at the same cell and +0.697 at lon+180. The retired pin "
                "a326fad338a4 also omitted deriv_two, cell_gradient_coef_x/y, "
                "defc_a/defc_b and the soil-composition group; no file matching it "
                "survives on any reachable machine. The rebuilt static carries real "
                "defc tables, so like the published mesh this row drops them at "
                "attach: the frozen v8.4.1 path runs deformation inactive. "
                "REFUSED AT BIND since 2026-08-24 by the dual-edge admission, and "
                "the row is kept because it is the measurement that sets the floor: "
                "the density-biased Fibonacci seed this mesh was generated from is "
                "polycrystalline, so its Delaunay carries 3,447 heptagons beside "
                "3,459 pentagons and the near-cocircular dislocation quads collapse "
                "dvEdge to 6.514 m at edge 19786 (dcEdge 38,657 m, ratio 1.685e-04, "
                "TRiSK tangential amplification 5,935x). Measured consequence "
                "(the proving node, RTX 5070 Ti): every runaway magnitude in the first outer "
                "step sits on that edge and the run dies at composite step 0. It has "
                "never completed a forecast and no timestep makes it able to."
            ),
        ),
        "u96.64002": MeshBinding(
            name="u96.64002",
            n_cells=64_002,
            n_edges=192_000,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=96_000.0,
            dt_seconds=120.0,
            grid_bytes=87_560_436,
            grid_sha256="57f4965a81d25dbc16b4fcbdb06474ca1a4b39adf406a58b04852be72f93f305",
            static_bytes=122_381_512,
            static_sha256="005cb9e7363283ec98a0cb027956d0245978a3a9a7c12cb3032d31e815561e27",
            drop_carried_deformation=True,
            notes=(
                "Generated uniform 96 km mesh, and the first generated mesh to "
                "complete a full-physics forecast. Both files are ours: the grid "
                "from rw_mpas_mesh seeded on the icosahedral Goldberg subdivision "
                "GP(80,0) -- 64,002 cells exactly, snap 0.0 percent -- and the "
                "static from the unified rw_mpas_static against that same grid. "
                "The seed is what separates it from v15.150.38857: coordination is "
                "exactly 12 pentagons and no heptagons, so no dislocation quad "
                "exists for a collapsed dual edge to form around, and "
                "min(dvEdge/dcEdge) measures 0.394671 -- the published x1.40962's "
                "own class. Declared dt 120 s against a measured Courant limit of "
                "511.4 s (min dcEdge 71,031.6 m). The static carries real defc "
                "tables, so like the other non-native rows this one drops them at "
                "attach."
            ),
        ),
    }
)

NATIVE_MESH_NAME = "x4.163842"
NATIVE_DEVICE_FLOOR = 24 * 1024**3
NATIVE_RESTART_FLOOR = 22 * 1024**3

_FINGERPRINT_SCALARS = (
    "N_CELLS",
    "N_EDGES",
    "N_LEVELS",
    "N_INTERFACES",
    "N_SOIL_LEVELS",
    "DT_SECONDS",
    "NOMINAL_DX_M",
    "MIN_FREE_DEVICE_BYTES",
    "RESTART_WORKER_MIN_FREE_DEVICE_BYTES",
)
_FINGERPRINT_PINS = (
    "INIT_RECONSTRUCTION_COEFFICIENTS_PIN",
    "INIT_EDGE_NORMAL_VECTORS_PIN",
    "PHYSICS_GEOMETRY_CARRIER_PIN",
    "LANDMASK_CONSTRUCTOR_CAST_PIN",
    "AUTHORITY_PINS",
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return repr(value.item())
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def constants_fingerprint(
    proof: Any,
    *,
    edge_authority_sha256: str | None = None,
    edge_minimum_m: float | None = None,
) -> dict[str, Any]:
    """Fingerprint every shape/timestep-sensitive value plus real-edge authority."""

    body: dict[str, Any] = {}
    for name in _FINGERPRINT_SCALARS:
        body[name] = _plain(getattr(proof, name, None))
    for name in _FINGERPRINT_PINS:
        body[name] = _plain(getattr(proof, name, None))
    body["EDGE_LENGTH_AUTHORITY_SHA256"] = edge_authority_sha256
    body["EDGE_LENGTH_MINIMUM_M"] = (
        None if edge_minimum_m is None else repr(float(edge_minimum_m))
    )
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"fields": body, "sha256": hashlib.sha256(blob).hexdigest()}


def _static_edge_authority(static_path: Path) -> EdgeLengthAuthority:
    """Build the physical edge-length authority through its public constructor.

    Reading ``dcEdge`` here rather than accepting a previously built mapping is
    deliberate: a hand-assembled dict can carry a minimum nothing measured.
    """

    import netCDF4

    with netCDF4.Dataset(str(static_path)) as dataset:
        raw_dc = dataset.variables.get("dcEdge")
        if raw_dc is None:
            raise MeshBindingMismatch(
                f"{static_path}: static carries no physical dcEdge; there is no "
                "length authority to admit a timestep or to fingerprint against"
            )
        dc_edge = np.asarray(raw_dc[:])
    try:
        return edge_length_authority(dc_edge, source="static.dcEdge")
    except TimestepAdmissionError as error:
        raise MeshBindingMismatch(f"{static_path}: {error}") from error


def _static_dual_edges(static_path: Path) -> tuple[Any, Any, Any]:
    """Read the DUAL edge lengths the TRiSK tangential terms divide by.

    Read from the same file the timestep authority reads, for the same reason:
    a mapping assembled earlier can carry a minimum nothing measured.
    """

    import netCDF4

    with netCDF4.Dataset(str(static_path)) as dataset:
        missing = [
            name
            for name in ("dvEdge", "dcEdge")
            if dataset.variables.get(name) is None
        ]
        if missing:
            raise MeshBindingMismatch(
                f"{static_path}: static carries no physical {', '.join(missing)}; "
                "the dual-edge amplification the TRiSK operators inherit cannot be measured"
            )
        dv_edge = np.asarray(dataset.variables["dvEdge"][:])
        dc_edge = np.asarray(dataset.variables["dcEdge"][:])
        raw_cells = dataset.variables.get("cellsOnEdge")
        cells_on_edge = None if raw_cells is None else np.asarray(raw_cells[:])
    return dv_edge, dc_edge, cells_on_edge


def _fingerprint_with_authority(
    proof: Any, authority: EdgeLengthAuthority
) -> dict[str, Any]:
    """The single site that supplies the fingerprint's authority keywords.

    Both sides of every fingerprint comparison route through here, so a keyword
    added to :func:`constants_fingerprint` cannot reach one side only.
    """

    return constants_fingerprint(
        proof,
        edge_authority_sha256=authority.raw_sha256,
        edge_minimum_m=authority.minimum_m,
    )


def binding_fingerprint(proof: Any, static: Path) -> dict[str, Any]:
    """The digest a bind against ``static`` produces, computed as ``bind_mesh`` does.

    Any baseline compared against a bind receipt MUST come from here.  A bare
    :func:`constants_fingerprint` digests ``EDGE_LENGTH_AUTHORITY_SHA256=None``
    and can never equal a bound digest, which would make the x4 frozen no-op
    report a moved fingerprint on a run that rebound nothing.
    """

    return _fingerprint_with_authority(proof, _static_edge_authority(Path(static)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(role: str, path: Path, want_bytes: int, want_sha: str, mesh: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise MeshBindingMismatch(f"mesh {mesh!r}: {role} authority is missing: {resolved}")
    size = resolved.stat().st_size
    if size != want_bytes:
        raise MeshBindingMismatch(
            f"mesh {mesh!r}: {role} byte count {size} != declared {want_bytes} ({resolved}); "
            "the supplied file is not the registered authority"
        )
    digest = _sha256_file(resolved)
    if digest != want_sha:
        raise MeshBindingMismatch(
            f"mesh {mesh!r}: {role} SHA-256 {digest} != declared {want_sha} ({resolved}); "
            "the supplied file is not the registered authority"
        )
    return {"path": str(resolved), "bytes": size, "sha256": digest}


def _inspect_grid(grid_path: Path, binding: MeshBinding) -> dict[str, Any]:
    import netCDF4

    with netCDF4.Dataset(str(grid_path)) as dataset:
        dims = {key: len(value) for key, value in dataset.dimensions.items()}
        observed_cells = dims.get("nCells")
        observed_edges = dims.get("nEdges")
        if observed_cells != binding.n_cells or observed_edges != binding.n_edges:
            raise MeshBindingMismatch(
                f"mesh {binding.name!r}: grid declares nCells={observed_cells}, nEdges={observed_edges}; "
                f"registry declares nCells={binding.n_cells}, nEdges={binding.n_edges}"
            )
        radius = float(getattr(dataset, "sphere_radius", 0.0) or 0.0)
        raw_nominal = dataset.variables.get("nominalMinDc")
        nominal = (
            float(np.float32(np.asarray(raw_nominal[:]).ravel()[0]))
            if raw_nominal is not None
            else None
        )
        if "nEdgesOnCell" not in dataset.variables or "cellsOnEdge" not in dataset.variables:
            raise MeshBindingMismatch(
                f"mesh {binding.name!r}: grid lacks nEdgesOnCell/cellsOnEdge required to bind topology"
            )
        n_edges_on_cell = np.asarray(dataset.variables["nEdgesOnCell"][:], dtype=np.int64)
        cells_on_edge = np.ascontiguousarray(
            np.asarray(dataset.variables["cellsOnEdge"][:], dtype="<i4")
        )
    active_slots = int(n_edges_on_cell.sum())
    return {
        "nCells": observed_cells,
        "nEdges": observed_edges,
        "grid_sphere_radius": radius,
        "grid_nominalMinDc_f32": nominal,
        "active_slots": active_slots,
        "active_components": active_slots * 3,
        "cellsOnEdge_shape": [int(cells_on_edge.shape[0]), int(cells_on_edge.shape[1])],
        "cellsOnEdge_raw_sha256": hashlib.sha256(cells_on_edge.tobytes(order="C")).hexdigest(),
    }


def _inspect_static(
    static_path: Path,
    binding: MeshBinding,
    grid_observed: Mapping[str, Any],
) -> dict[str, Any]:
    import netCDF4

    with netCDF4.Dataset(str(static_path)) as dataset:
        dims = {key: len(value) for key, value in dataset.dimensions.items()}
        radius = float(getattr(dataset, "sphere_radius", 0.0) or 0.0)
        raw_nominal = dataset.variables.get("nominalMinDc")
        if raw_nominal is None:
            raise MeshBindingMismatch(
                f"mesh {binding.name!r}: static carries no nominalMinDc; nominal dx cannot be reported"
            )
        observed_dx = np.float32(np.asarray(raw_nominal[:]).ravel()[0])
        raw_dc = dataset.variables.get("dcEdge")
        if raw_dc is None:
            raise MeshBindingMismatch(
                f"mesh {binding.name!r}: static carries no physical dcEdge; timestep admission has no length authority"
            )
        dc_edge = np.asarray(raw_dc[:])
    want_dx = np.float32(binding.nominal_dx_m)
    if observed_dx.view(np.uint32) != want_dx.view(np.uint32):
        raise MeshBindingMismatch(
            f"mesh {binding.name!r}: static nominalMinDc={float(observed_dx)} is not FP32-exact "
            f"equal to registry nominal dx={float(want_dx)}"
        )
    grid_dc = grid_observed.get("grid_nominalMinDc_f32")
    if grid_dc and radius:
        implied = float(grid_dc) * radius
        if abs(implied - float(want_dx)) > 1.0e-3 * float(want_dx):
            raise MeshBindingMismatch(
                f"mesh {binding.name!r}: unit-sphere grid nominalMinDc={grid_dc} rad times "
                f"static sphere_radius={radius} implies {implied:.3f} m, not {float(want_dx)} m; "
                "grid and static are not the same mesh"
            )
    levels = dims.get("nVertLevels")
    soil = dims.get("nSoilLevels")
    if levels is not None and levels != binding.n_levels:
        raise MeshBindingMismatch(
            f"mesh {binding.name!r}: static nVertLevels={levels}, registry={binding.n_levels}"
        )
    if soil is not None and soil != binding.n_soil_levels:
        raise MeshBindingMismatch(
            f"mesh {binding.name!r}: static nSoilLevels={soil}, registry={binding.n_soil_levels}"
        )
    try:
        edge_authority = edge_length_authority(dc_edge, source="static.dcEdge")
    except TimestepAdmissionError as error:
        raise MeshBindingMismatch(f"mesh {binding.name!r}: {error}") from error
    return {
        "nVertLevels": levels,
        "nSoilLevels": soil,
        "static_sphere_radius": radius,
        "nominalMinDc_f32": float(observed_dx),
        "edge_length_authority": edge_authority.as_dict(),
    }


def _zero_digest(shape: tuple[int, ...]) -> str:
    return hashlib.sha256(np.zeros(shape, dtype="<f4").tobytes(order="C")).hexdigest()


def bind_mesh(
    proof: Any,
    mesh_name: str,
    *,
    grid: Path,
    static: Path,
    forecast: Any = None,
    verify_frozen_sources: bool = True,
    log=print,
) -> dict[str, Any]:
    """Cross-examine, Courant-admit, and bind one registered mesh before CUDA."""

    if mesh_name not in MESH_BINDINGS:
        raise MeshBindingMismatch(
            f"unknown mesh {mesh_name!r}; registered meshes are {sorted(MESH_BINDINGS)}. "
            "Register dimensions, file pins, nominal dx, timestep, and Courant policy before running"
        )
    binding = MESH_BINDINGS[mesh_name]

    frozen: dict[str, Any] | None = None
    if verify_frozen_sources:
        frozen = proof.require_frozen_execution_sources()
        log(
            f"[mesh-binding] frozen execution sources verified: {len(frozen['files'])} modules, "
            f"receipt {frozen['sha256'][:16]}"
        )

    files = {
        "grid": _require_file("grid", Path(grid), binding.grid_bytes, binding.grid_sha256, mesh_name),
        "static": _require_file("static", Path(static), binding.static_bytes, binding.static_sha256, mesh_name),
    }
    observed = _inspect_grid(Path(files["grid"]["path"]), binding)
    observed.update(_inspect_static(Path(files["static"]["path"]), binding, observed))
    # Reconstruct the immutable authority through its public constructor -- the
    # same read a fingerprint baseline performs -- rather than trusting the
    # mapping the inspection pass already built.
    authority = _static_edge_authority(Path(files["static"]["path"]))
    try:
        timestep = admit_timestep(
            binding.dt_seconds,
            authority,
            policy=binding.courant_policy(),
        )
    except TimestepAdmissionError as error:
        raise MeshBindingMismatch(f"mesh {mesh_name!r}: {error}") from error
    observed["timestep_admission"] = timestep.as_dict()
    # Dual-edge admission runs for EVERY mesh, published or generated, frozen
    # or not.  A mesh whose Voronoi edges collapse cannot be saved by a smaller
    # timestep, so it is refused here rather than inside step 0 on a validation
    # flag that names no array, no cell and no edge.
    dv_edge, dc_edge, cells_on_edge = _static_dual_edges(
        Path(files["static"]["path"])
    )
    try:
        dual_edges = admit_dual_edges(
            dv_edge,
            dc_edge,
            policy=binding.dual_edge_policy(),
            cells_on_edge=cells_on_edge,
            mesh_name=mesh_name,
        )
    except DualEdgeAdmissionError as error:
        raise MeshBindingMismatch(str(error)) from error
    observed["dual_edge_admission"] = dual_edges.as_dict()
    log(
        f"[mesh-binding] mesh {mesh_name}: nCells={observed['nCells']} "
        f"nEdges={observed['nEdges']} nominal={observed['nominalMinDc_f32']} m; "
        f"min(dcEdge)={authority.minimum_m:.3f} m; dt={binding.dt_seconds:.3f} s; "
        f"limit={timestep.maximum_admitted_dt_seconds:.3f} s"
    )
    log(
        f"[mesh-binding] mesh {mesh_name}: min(dvEdge/dcEdge)="
        f"{dual_edges.minimum_ratio:.6g} at edge {dual_edges.minimum_ratio_edge} "
        f"(dvEdge={dual_edges.minimum_ratio_dv_edge_m:.3f} m), "
        f"TRiSK tangential amplification {dual_edges.amplification:.4g}x; "
        f"floor {dual_edges.policy.minimum_dv_over_dc:.6g}"
    )

    edge_sha = authority.raw_sha256
    fingerprint_before = _fingerprint_with_authority(proof, authority)

    for attr, want in (
        ("N_LEVELS", binding.n_levels),
        ("N_INTERFACES", binding.n_interfaces),
        ("N_SOIL_LEVELS", binding.n_soil_levels),
    ):
        have = getattr(proof, attr)
        if have != want:
            raise MeshBindingMismatch(
                f"mesh {mesh_name!r}: module {attr}={have}, registry={want}; vertical structure "
                "is not rebound by mesh binding and must already agree"
            )

    native = MESH_BINDINGS[NATIVE_MESH_NAME]
    if binding.frozen_native:
        mismatches: list[str] = []
        for attr, want in (
            ("N_CELLS", binding.n_cells),
            ("N_EDGES", binding.n_edges),
            ("MIN_FREE_DEVICE_BYTES", NATIVE_DEVICE_FLOOR),
        ):
            have = getattr(proof, attr)
            if int(have) != int(want):
                mismatches.append(f"{attr}={have!r} != {want!r}")
        if float(getattr(proof, "DT_SECONDS")) != float(binding.dt_seconds):
            mismatches.append(
                f"DT_SECONDS={getattr(proof, 'DT_SECONDS')!r} != {binding.dt_seconds!r}"
            )
        have_dx = np.float32(proof.NOMINAL_DX_M)
        if have_dx.view(np.uint32) != np.float32(binding.nominal_dx_m).view(np.uint32):
            mismatches.append(f"NOMINAL_DX_M={float(have_dx)} != {binding.nominal_dx_m}")
        if mismatches:
            raise MeshBindingMismatch(
                f"mesh {mesh_name!r} is the frozen native mesh, so binding must be a no-op, "
                f"but constants already moved: {'; '.join(mismatches)}"
            )
        fingerprint_after = _fingerprint_with_authority(proof, authority)
        if fingerprint_after["sha256"] != fingerprint_before["sha256"]:
            raise MeshBindingError(
                "native x4 bind was required to change nothing, but fingerprint moved "
                f"{fingerprint_before['sha256'][:16]} -> {fingerprint_after['sha256'][:16]}"
            )
        log(
            f"[mesh-binding] {mesh_name} frozen no-op: fingerprint unchanged "
            f"at {fingerprint_after['sha256'][:16]}"
        )
        return {
            "mesh": mesh_name,
            "rebound": False,
            "files": files,
            "observed": observed,
            "frozen_execution_sources": frozen,
            "constants_fingerprint_before": fingerprint_before["sha256"],
            "constants_fingerprint_after": fingerprint_after["sha256"],
            "edge_length_authority_sha256": edge_sha,
            "timestep_admission": timestep.as_dict(),
            "rebindings": {},
            "notes": binding.notes,
        }

    rebindings: dict[str, Any] = {}
    nc, ne = binding.n_cells, binding.n_edges

    for role in ("grid", "static"):
        pin = dict(proof.AUTHORITY_PINS[role])
        pin["bytes"] = files[role]["bytes"]
        pin["sha256"] = files[role]["sha256"]
        pin["relative_path"] = files[role]["path"]
        proof.AUTHORITY_PINS[role] = pin
    rebindings["AUTHORITY_PINS"] = {
        role: files[role]["sha256"][:16] for role in ("grid", "static")
    }

    proof.N_CELLS, proof.N_EDGES = nc, ne
    rebindings["N_CELLS"] = [native.n_cells, nc]
    rebindings["N_EDGES"] = [native.n_edges, ne]

    before_dt = float(proof.DT_SECONDS)
    proof.DT_SECONDS = float(binding.dt_seconds)
    rebindings["DT_SECONDS"] = [before_dt, float(binding.dt_seconds)]

    rc_pin = dict(proof.INIT_RECONSTRUCTION_COEFFICIENTS_PIN)
    rc_pin["shape"] = (nc, 10, 3)
    rc_pin["static_placeholder_raw_sha256"] = _zero_digest((nc, 10, 3))
    rc_pin["active_slots"] = observed["active_slots"]
    rc_pin["active_components"] = observed["active_components"]
    rc_pin["nonzero_components"] = observed["active_components"]
    proof.INIT_RECONSTRUCTION_COEFFICIENTS_PIN = MappingProxyType(rc_pin)
    rebindings["INIT_RECONSTRUCTION_COEFFICIENTS_PIN"] = {
        "shape": [nc, 10, 3],
        "active_slots": observed["active_slots"],
        "active_components": observed["active_components"],
    }

    en_pin = dict(proof.INIT_EDGE_NORMAL_VECTORS_PIN)
    en_pin["shape"] = (ne, 3)
    en_pin["static_placeholder_raw_sha256"] = _zero_digest((ne, 3))
    proof.INIT_EDGE_NORMAL_VECTORS_PIN = MappingProxyType(en_pin)
    rebindings["INIT_EDGE_NORMAL_VECTORS_PIN"] = {"shape": [ne, 3]}

    geometry = {
        key: dict(value) if isinstance(value, Mapping) else value
        for key, value in proof.PHYSICS_GEOMETRY_CARRIER_PIN.items()
    }
    geometry["cellsOnEdge"]["shape"] = (ne, 2)
    geometry["cellsOnEdge"]["raw_sha256"] = observed["cellsOnEdge_raw_sha256"]
    proof.PHYSICS_GEOMETRY_CARRIER_PIN = MappingProxyType(geometry)
    rebindings["PHYSICS_GEOMETRY_CARRIER_PIN"] = {
        "cellsOnEdge_shape": [ne, 2],
        "cellsOnEdge_raw_sha256": observed["cellsOnEdge_raw_sha256"][:16],
    }

    if hasattr(proof, "LANDMASK_CONSTRUCTOR_CAST_PIN"):
        landmask = dict(proof.LANDMASK_CONSTRUCTOR_CAST_PIN)
        landmask["shape"] = (nc,)
        landmask["source_array_sha256"] = None
        landmask["rebound_for_mesh"] = mesh_name
        proof.LANDMASK_CONSTRUCTOR_CAST_PIN = MappingProxyType(landmask)
        rebindings["LANDMASK_CONSTRUCTOR_CAST_PIN"] = {"shape": [nc]}

    dx = np.float32(binding.nominal_dx_m)
    proof.NOMINAL_DX_M = dx
    rebindings["NOMINAL_DX_M"] = [float(native.nominal_dx_m), float(dx)]

    if binding.scale_admission_floor:
        floor = int(NATIVE_DEVICE_FLOOR * nc / native.n_cells)
        inner = proof.gpu_memory_admission

        def _admit(cp, *, minimum=None, _inner=inner, _floor=floor):
            return _inner(cp, minimum=_floor if minimum is None else minimum)

        proof.gpu_memory_admission = _admit
        proof.MIN_FREE_DEVICE_BYTES = floor
        if hasattr(proof, "RESTART_WORKER_MIN_FREE_DEVICE_BYTES"):
            proof.RESTART_WORKER_MIN_FREE_DEVICE_BYTES = int(
                NATIVE_RESTART_FLOOR * nc / native.n_cells
            )
        rebindings["MIN_FREE_DEVICE_BYTES"] = [
            NATIVE_DEVICE_FLOOR,
            floor,
            f"{floor / 1024**3:.2f} GiB",
        ]

    dropped: dict[str, Any] = {}
    if binding.drop_carried_deformation:
        attach = proof.attach_inactive_zero_deformation

        def _attach(mesh, _attach=attach, _dropped=dropped):
            arrays = getattr(mesh, "arrays", None)
            if isinstance(arrays, dict):
                for name in ("defc_a", "defc_b"):
                    if name in arrays:
                        array = np.ascontiguousarray(arrays.pop(name))
                        _dropped[name] = {
                            "shape": list(array.shape),
                            "nonzero": int(np.count_nonzero(array)),
                            "raw_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                        }
            return _attach(mesh)

        proof.attach_inactive_zero_deformation = _attach
        rebindings["attach_inactive_zero_deformation"] = "carried defc dropped"

    if forecast is not None:
        forecast.N_CELLS, forecast.N_EDGES = nc, ne
        forecast.NOMINAL_DX_M = dx
        forecast.DT_SECONDS = float(binding.dt_seconds)
        rebindings["forecast_reexports"] = [
            "N_CELLS",
            "N_EDGES",
            "NOMINAL_DX_M",
            "DT_SECONDS",
        ]
        # THE BREAKAGE THIS PREVENTS: a receipt for a run on this mesh that
        # says "x4.163842" in its own claim sentence and profile slug.  Both
        # constants are written verbatim into every receipt, and a receipt
        # that names the wrong mesh is worse than no receipt -- a reader
        # comparing two runs would take them for the same shape.  The native
        # row rebinds nothing, so its claim and profile stay byte-identical.
        native_label, mesh_label = native.name, binding.name
        if mesh_label != native_label:
            for module, attribute in (
                (forecast, "CLAIM"),
                (proof, "PROFILE"),
                (proof, "CLAIM"),
            ):
                text = getattr(module, attribute, None)
                if not isinstance(text, str) or native_label not in text:
                    continue
                setattr(
                    module,
                    attribute,
                    text.replace(native_label, mesh_label).replace(
                        f"dt = {native.dt_seconds:g} s",
                        f"dt = {binding.dt_seconds:g} s",
                    ),
                )
                rebindings[f"{attribute}_mesh_label"] = [native_label, mesh_label]
        prepare = forecast.prepare_forecast_host

        def _prepare(
            *args,
            _prepare=prepare,
            _dx=dx,
            _dt=np.float32(binding.dt_seconds),
            **kwargs,
        ):
            import mpas_port.cuda_gwdo_v841 as gwdo

            before = np.float32(gwdo.X4_NOMINAL_MIN_DC_M_F32)
            gwdo.X4_NOMINAL_MIN_DC_M_F32 = _dx
            # The GWDO run guard compares the composite dt against this module
            # constant bit-for-bit.  On the frozen native mesh nothing is
            # rebound and the guard still demands exactly 120 s; a registered
            # mesh runs at ITS declared, Courant-admitted timestep, so the
            # guard must demand exactly that value instead -- the kernel takes
            # dt as a runtime argument and is dt-general.  Without this
            # rebind every registered mesh whose dt is not 120 s dies at
            # step 0 in run_bl_ysu_gwdo_cuda_v841.
            before_dt = np.float32(gwdo.X4_DT_SECONDS_F32)
            gwdo.X4_DT_SECONDS_F32 = _dt
            log(
                f"[mesh-binding] GWD effective len_disp {float(before)} -> {float(_dx)} m; "
                f"GWDO dt guard {float(before_dt)} -> {float(_dt)} s"
            )
            return _prepare(*args, **kwargs)

        forecast.prepare_forecast_host = _prepare
        rebindings["cuda_gwdo_v841.X4_NOMINAL_MIN_DC_M_F32"] = [
            float(native.nominal_dx_m),
            float(dx),
        ]
        rebindings["cuda_gwdo_v841.X4_DT_SECONDS_F32"] = [
            float(native.dt_seconds),
            float(binding.dt_seconds),
        ]

    fingerprint_after = _fingerprint_with_authority(proof, authority)
    if fingerprint_after["sha256"] == fingerprint_before["sha256"]:
        raise MeshBindingError(
            f"mesh {mesh_name!r} was supposed to rebind shape/timestep constants, but "
            "the fingerprint did not move; the treatment never engaged"
        )
    log(
        f"[mesh-binding] bound {mesh_name}: nCells={nc}, nEdges={ne}, "
        f"dx={float(dx)} m, dt={binding.dt_seconds} s; fingerprint "
        f"{fingerprint_before['sha256'][:16]} -> {fingerprint_after['sha256'][:16]}"
    )
    return {
        "mesh": mesh_name,
        "rebound": True,
        "files": files,
        "observed": observed,
        "frozen_execution_sources": frozen,
        "constants_fingerprint_before": fingerprint_before["sha256"],
        "constants_fingerprint_after": fingerprint_after["sha256"],
        "edge_length_authority_sha256": edge_sha,
        "timestep_admission": timestep.as_dict(),
        "rebindings": rebindings,
        "deformation_dropped": dropped,
        "notes": binding.notes,
    }
