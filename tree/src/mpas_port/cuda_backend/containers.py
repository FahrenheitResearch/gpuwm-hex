"""One-time host/device transfer containers for the CUDA atmosphere port."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import numpy as np

from .runtime import CudaRefusal, require_cuda


def _cp() -> Any:
    import cupy as cp

    return cp


def require_resident_array(
    name: str,
    value: Any,
    *,
    dtype: Any | None = None,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
) -> Any:
    """Return one packed device array or refuse implicit transfer/layout drift.

    MPAS CUDA kernels address logical ``[level, entity]`` fields as
    ``level * n_entity + entity``.  Accepting a NumPy array, a strided CuPy
    view, or a different dtype would make that raw indexing silently wrong.
    Public resident APIs therefore share this exact fail-closed boundary.
    """

    cp = _cp()
    if not isinstance(value, cp.ndarray):
        raise TypeError(f"{name} must be a resident cupy.ndarray")
    if dtype is not None and value.dtype != cp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {cp.dtype(dtype)}")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.ndim}")
    if not bool(value.flags.c_contiguous):
        raise ValueError(f"{name} must be C-contiguous with entities fastest")
    return value


def _require_same_host_dtype(name: str, value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    """Reject silent precision conversion at prognostic authority boundaries."""

    array = np.asarray(value)
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")
    return _as_float(name, array, dtype)


def _mesh_value(mesh: object, name: str, default: Any = None) -> Any:
    if hasattr(mesh, name):
        return getattr(mesh, name)
    arrays = getattr(mesh, "arrays", None)
    if isinstance(arrays, Mapping) and name in arrays:
        return arrays[name]
    attrs = getattr(mesh, "attrs", None)
    if isinstance(attrs, Mapping) and name in attrs:
        return attrs[name]
    return default


def _as_float(name: str, value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def _as_index(name: str, value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must be integer")
    info = np.iinfo(dtype)
    if array.size and (np.min(array) < info.min or np.max(array) > info.max):
        raise OverflowError(f"{name} cannot be represented by {dtype}")
    return np.ascontiguousarray(array, dtype=dtype)


@dataclass(frozen=True, slots=True)
class TransferStats:
    bytes: int
    seconds: float

    @property
    def gib_per_second(self) -> float:
        return self.bytes / max(self.seconds, np.finfo(np.float64).tiny) / 2**30

    def as_dict(self) -> dict[str, float | int]:
        return {
            "bytes": self.bytes,
            "seconds": self.seconds,
            "gib_per_second": self.gib_per_second,
        }


@dataclass(slots=True)
class DeviceMesh:
    cells_on_edge: Any
    edges_on_cell: Any
    n_edges_on_cell: Any
    cells_on_cell: Any
    vertices_on_edge: Any
    edges_on_edge: Any
    n_edges_on_edge: Any
    vertices_on_cell: Any
    edges_on_vertex: Any
    cells_on_vertex: Any
    edge_sign_on_cell: Any
    weights_on_edge: Any
    dc_edge: Any
    dv_edge: Any
    area_cell: Any
    area_triangle: Any
    kite_areas_on_vertex: Any
    lat_cell: Any
    lon_cell: Any
    lat_edge: Any
    lon_edge: Any
    angle_edge: Any
    mesh_density: Any
    defc_a: Any
    defc_b: Any
    f_vertex: Any
    f_edge: Any
    spec_zone_mask_edge: Any
    n_cells: int
    n_edges: int
    n_vertices: int
    max_edges: int
    max_edges2: int
    vertex_degree: int
    nominal_min_dc: float
    dtype: np.dtype[Any]
    index_dtype: np.dtype[Any]
    h2d: TransferStats

    @classmethod
    def from_host(
        cls,
        mesh: object,
        *,
        dtype: Any = np.float32,
        index_dtype: Any = np.int32,
    ) -> "DeviceMesh":
        require_cuda()
        cp = _cp()
        float_dtype = np.dtype(dtype)
        integer_dtype = np.dtype(index_dtype)
        if float_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("DeviceMesh dtype must be float32 or float64")
        if integer_dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
            raise TypeError("DeviceMesh index_dtype must be int32 or int64")
        dimensions = getattr(mesh, "dimensions", {})
        n_cells = int(
            dimensions.get("nCells", np.asarray(_mesh_value(mesh, "latCell")).size)
        )
        n_edges = int(
            dimensions.get("nEdges", np.asarray(_mesh_value(mesh, "dcEdge")).size)
        )
        n_vertices = int(
            dimensions.get(
                "nVertices", np.asarray(_mesh_value(mesh, "areaTriangle")).size
            )
        )
        max_edges = int(
            dimensions.get(
                "maxEdges", np.asarray(_mesh_value(mesh, "edgesOnCell")).shape[1]
            )
        )
        max_edges2 = int(
            dimensions.get(
                "maxEdges2", np.asarray(_mesh_value(mesh, "edgesOnEdge")).shape[1]
            )
        )
        vertex_degree = int(
            dimensions.get(
                "vertexDegree", np.asarray(_mesh_value(mesh, "cellsOnVertex")).shape[1]
            )
        )
        index_names = (
            "cellsOnEdge",
            "edgesOnCell",
            "nEdgesOnCell",
            "cellsOnCell",
            "verticesOnEdge",
            "edgesOnEdge",
            "nEdgesOnEdge",
            "verticesOnCell",
            "edgesOnVertex",
            "cellsOnVertex",
        )
        host_index = {
            name: _as_index(name, _mesh_value(mesh, name), integer_dtype)
            for name in index_names
        }
        signs = np.zeros((n_cells, max_edges), dtype=float_dtype)
        eoc = host_index["edgesOnCell"]
        coe = host_index["cellsOnEdge"]
        counts = host_index["nEdgesOnCell"]
        for cell in range(n_cells):
            for slot in range(int(counts[cell])):
                edge = int(eoc[cell, slot])
                signs[cell, slot] = 1.0 if int(coe[edge, 0]) == cell else -1.0
        float_map = {
            "weights_on_edge": "weightsOnEdge",
            "dc_edge": "dcEdge",
            "dv_edge": "dvEdge",
            "area_cell": "areaCell",
            "area_triangle": "areaTriangle",
            "kite_areas_on_vertex": "kiteAreasOnVertex",
            "lat_cell": "latCell",
            "lon_cell": "lonCell",
            "lat_edge": "latEdge",
            "lon_edge": "lonEdge",
            "angle_edge": "angleEdge",
            "mesh_density": "meshDensity",
            "defc_a": "defc_a",
            "defc_b": "defc_b",
            "f_vertex": "fVertex",
            "f_edge": "fEdge",
        }
        host_float = {
            target: _as_float(source, _mesh_value(mesh, source), float_dtype)
            for target, source in float_map.items()
        }
        raw_zone = _mesh_value(mesh, "spec_zone_mask_edge")
        if raw_zone is None:
            raw_zone = np.zeros(n_edges, dtype=float_dtype)
        host_float["spec_zone_mask_edge"] = _as_float(
            "spec_zone_mask_edge", raw_zone, float_dtype
        )
        nominal = _mesh_value(mesh, "nominalMinDc")
        if nominal is None:
            nominal = _mesh_value(mesh, "nominal_min_dc")
        if nominal is None or not np.isfinite(float(np.asarray(nominal))):
            raise CudaRefusal(
                "mesh.nominalMinDc is required by CUDA scale-aware filters"
            )
        started = time.perf_counter()
        device_index = {name: cp.asarray(value) for name, value in host_index.items()}
        device_float = {name: cp.asarray(value) for name, value in host_float.items()}
        device_signs = cp.asarray(signs)
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        arrays = (*device_index.values(), *device_float.values(), device_signs)
        byte_count = int(sum(int(value.nbytes) for value in arrays))
        return cls(
            cells_on_edge=device_index["cellsOnEdge"],
            edges_on_cell=device_index["edgesOnCell"],
            n_edges_on_cell=device_index["nEdgesOnCell"],
            cells_on_cell=device_index["cellsOnCell"],
            vertices_on_edge=device_index["verticesOnEdge"],
            edges_on_edge=device_index["edgesOnEdge"],
            n_edges_on_edge=device_index["nEdgesOnEdge"],
            vertices_on_cell=device_index["verticesOnCell"],
            edges_on_vertex=device_index["edgesOnVertex"],
            cells_on_vertex=device_index["cellsOnVertex"],
            edge_sign_on_cell=device_signs,
            **device_float,
            n_cells=n_cells,
            n_edges=n_edges,
            n_vertices=n_vertices,
            max_edges=max_edges,
            max_edges2=max_edges2,
            vertex_degree=vertex_degree,
            nominal_min_dc=float(np.asarray(nominal)),
            dtype=float_dtype,
            index_dtype=integer_dtype,
            h2d=TransferStats(byte_count, elapsed),
        )

    def validate(self) -> None:
        """Validate the committed packed topology used by raw CUDA indexing."""

        index_shapes = {
            "cells_on_edge": (self.n_edges, 2),
            "edges_on_cell": (self.n_cells, self.max_edges),
            "n_edges_on_cell": (self.n_cells,),
            "cells_on_cell": (self.n_cells, self.max_edges),
            "vertices_on_edge": (self.n_edges, 2),
            "edges_on_edge": (self.n_edges, self.max_edges2),
            "n_edges_on_edge": (self.n_edges,),
            "vertices_on_cell": (self.n_cells, self.max_edges),
            "edges_on_vertex": (self.n_vertices, self.vertex_degree),
            "cells_on_vertex": (self.n_vertices, self.vertex_degree),
        }
        for name, shape in index_shapes.items():
            require_resident_array(
                f"mesh.{name}", getattr(self, name), dtype=self.index_dtype, shape=shape
            )
        float_shapes = {
            "edge_sign_on_cell": (self.n_cells, self.max_edges),
            "weights_on_edge": (self.n_edges, self.max_edges2),
            "dc_edge": (self.n_edges,),
            "dv_edge": (self.n_edges,),
            "area_cell": (self.n_cells,),
            "area_triangle": (self.n_vertices,),
            "kite_areas_on_vertex": (self.n_vertices, self.vertex_degree),
            "lat_cell": (self.n_cells,),
            "lon_cell": (self.n_cells,),
            "lat_edge": (self.n_edges,),
            "lon_edge": (self.n_edges,),
            "angle_edge": (self.n_edges,),
            "mesh_density": (self.n_cells,),
            "defc_a": (self.n_cells, self.max_edges),
            "defc_b": (self.n_cells, self.max_edges),
            "f_vertex": (self.n_vertices,),
            "f_edge": (self.n_edges,),
            "spec_zone_mask_edge": (self.n_edges,),
        }
        for name, shape in float_shapes.items():
            require_resident_array(
                f"mesh.{name}", getattr(self, name), dtype=self.dtype, shape=shape
            )

    def to_host(self) -> dict[str, np.ndarray | float | int]:
        cp = _cp()
        result: dict[str, np.ndarray | float | int] = {}
        for name in (
            "cells_on_edge",
            "edges_on_cell",
            "n_edges_on_cell",
            "cells_on_cell",
            "vertices_on_edge",
            "edges_on_edge",
            "n_edges_on_edge",
            "vertices_on_cell",
            "edges_on_vertex",
            "cells_on_vertex",
            "edge_sign_on_cell",
            "weights_on_edge",
            "dc_edge",
            "dv_edge",
            "area_cell",
            "area_triangle",
            "kite_areas_on_vertex",
            "lat_cell",
            "lon_cell",
            "lat_edge",
            "lon_edge",
            "angle_edge",
            "mesh_density",
            "defc_a",
            "defc_b",
            "f_vertex",
            "f_edge",
            "spec_zone_mask_edge",
        ):
            result[name] = cp.asnumpy(getattr(self, name))
        result.update(
            n_cells=self.n_cells,
            n_edges=self.n_edges,
            n_vertices=self.n_vertices,
            max_edges=self.max_edges,
            max_edges2=self.max_edges2,
            vertex_degree=self.vertex_degree,
            nominal_min_dc=self.nominal_min_dc,
        )
        return result


@dataclass(slots=True)
class DeviceVerticalGrid:
    zw: Any
    dzw: Any
    rdzw: Any
    zu: Any
    dzu: Any
    rdzu: Any
    rdzwp: Any
    rdzwm: Any
    fzp: Any
    fzm: Any
    ah: Any
    hx: Any
    zgrid: Any
    zz: Any
    zxu: Any
    dss: Any
    cf1: float
    cf2: float
    cf3: float
    first_height_level: int
    n_vert_levels: int
    dtype: np.dtype[Any]
    h2d: TransferStats

    @classmethod
    def from_host(
        cls, vertical: object, *, dtype: Any | None = None
    ) -> "DeviceVerticalGrid":
        require_cuda()
        cp = _cp()
        selected = np.dtype(
            np.asarray(getattr(vertical, "zz")).dtype if dtype is None else dtype
        )
        if selected not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("DeviceVerticalGrid dtype must be float32 or float64")
        names = (
            "zw",
            "dzw",
            "rdzw",
            "zu",
            "dzu",
            "rdzu",
            "rdzwp",
            "rdzwm",
            "fzp",
            "fzm",
            "ah",
            "hx",
            "zgrid",
            "zz",
            "zxu",
            "dss",
        )
        host = {
            name: np.ascontiguousarray(
                np.asarray(getattr(vertical, name), dtype=selected)
            )
            for name in names
        }
        started = time.perf_counter()
        device = {name: cp.asarray(value) for name, value in host.items()}
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        byte_count = int(sum(int(value.nbytes) for value in device.values()))
        return cls(
            **device,
            cf1=float(getattr(vertical, "cf1")),
            cf2=float(getattr(vertical, "cf2")),
            cf3=float(getattr(vertical, "cf3")),
            first_height_level=int(getattr(vertical, "first_height_level")),
            n_vert_levels=int(np.asarray(getattr(vertical, "zz")).shape[0]),
            dtype=selected,
            h2d=TransferStats(byte_count, elapsed),
        )

    def validate(self, *, n_cells: int, n_edges: int) -> None:
        one_dimensional = {
            "zw": self.n_vert_levels + 1,
            "dzw": self.n_vert_levels,
            "rdzw": self.n_vert_levels,
            "zu": self.n_vert_levels,
            "dzu": self.n_vert_levels,
            "rdzu": self.n_vert_levels,
            "rdzwp": self.n_vert_levels,
            "rdzwm": self.n_vert_levels,
            "fzp": self.n_vert_levels,
            "fzm": self.n_vert_levels,
            "ah": self.n_vert_levels + 1,
        }
        for name, length in one_dimensional.items():
            require_resident_array(
                f"vertical.{name}",
                getattr(self, name),
                dtype=self.dtype,
                shape=(length,),
            )
        for name, shape in {
            "hx": (self.n_vert_levels + 1, n_cells),
            "zgrid": (self.n_vert_levels + 1, n_cells),
            "zz": (self.n_vert_levels, n_cells),
            "zxu": (self.n_vert_levels, n_edges),
            "dss": (self.n_vert_levels, n_cells),
        }.items():
            require_resident_array(
                f"vertical.{name}", getattr(self, name), dtype=self.dtype, shape=shape
            )


@dataclass(slots=True)
class DeviceReferenceState:
    rho_base: Any
    rho_theta_base: Any
    pressure_base: Any
    exner_base: Any
    dtype: np.dtype[Any]
    h2d: TransferStats

    @classmethod
    def from_host(
        cls, reference: object, *, dtype: Any | None = None
    ) -> "DeviceReferenceState":
        require_cuda()
        cp = _cp()
        selected = np.dtype(
            np.asarray(getattr(reference, "rho_base")).dtype if dtype is None else dtype
        )
        names = ("rho_base", "rho_theta_base", "pressure_base", "exner_base")
        host = {
            name: _as_float(name, getattr(reference, name), selected) for name in names
        }
        started = time.perf_counter()
        device = {name: cp.asarray(value) for name, value in host.items()}
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        return cls(
            **device,
            dtype=selected,
            h2d=TransferStats(
                sum(int(value.nbytes) for value in device.values()), elapsed
            ),
        )

    def validate(self, *, n_vert_levels: int, n_cells: int) -> None:
        shape = (n_vert_levels, n_cells)
        for name in ("rho_base", "rho_theta_base", "pressure_base", "exner_base"):
            require_resident_array(
                f"reference.{name}", getattr(self, name), dtype=self.dtype, shape=shape
            )


@dataclass(slots=True)
class DevicePrognosticState:
    rho: Any
    rho_theta: Any
    rho_u: Any
    rho_w: Any
    scalars: Any
    time_seconds: float
    dtype: np.dtype[Any]
    h2d: TransferStats

    @classmethod
    def from_host(
        cls, state: object, *, dtype: Any | None = None
    ) -> "DevicePrognosticState":
        require_cuda()
        cp = _cp()
        selected = np.dtype(
            np.asarray(getattr(state, "rho")).dtype if dtype is None else dtype
        )
        names = ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
        host = {
            name: _require_same_host_dtype(name, getattr(state, name), selected)
            for name in names
        }
        started = time.perf_counter()
        device = {name: cp.asarray(value) for name, value in host.items()}
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        return cls(
            **device,
            time_seconds=float(getattr(state, "time_seconds", 0.0)),
            dtype=selected,
            h2d=TransferStats(
                sum(int(value.nbytes) for value in device.values()), elapsed
            ),
        )

    def validate(self, *, n_vert_levels: int, n_cells: int, n_edges: int) -> None:
        for name, shape in {
            "rho": (n_vert_levels, n_cells),
            "rho_theta": (n_vert_levels, n_cells),
            "rho_u": (n_vert_levels, n_edges),
            "rho_w": (n_vert_levels + 1, n_cells),
        }.items():
            require_resident_array(
                f"state.{name}", getattr(self, name), dtype=self.dtype, shape=shape
            )
        require_resident_array("state.scalars", self.scalars, dtype=self.dtype, ndim=3)
        if tuple(self.scalars.shape[1:]) != (n_vert_levels, n_cells):
            raise ValueError(
                "state.scalars must have shape "
                f"(nTracers, {n_vert_levels}, {n_cells}), got {tuple(self.scalars.shape)}"
            )

    def to_host(self) -> Any:
        from mpas_port.state import PrognosticState

        cp = _cp()
        return PrognosticState(
            rho=cp.asnumpy(self.rho),
            rho_theta=cp.asnumpy(self.rho_theta),
            rho_u=cp.asnumpy(self.rho_u),
            rho_w=cp.asnumpy(self.rho_w),
            scalars=cp.asnumpy(self.scalars),
            time_seconds=self.time_seconds,
        )

    def to_host_timed(self) -> tuple[Any, TransferStats]:
        started = time.perf_counter()
        state = self.to_host()
        elapsed = time.perf_counter() - started
        byte_count = int(
            sum(
                int(getattr(self, name).nbytes)
                for name in ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
            )
        )
        return state, TransferStats(byte_count, elapsed)


@dataclass(slots=True)
class DeviceSavedDiagnostics:
    theta_m: Any
    exner: Any
    density_perturbation: Any
    rho_theta_perturbation: Any
    pressure_perturbation: Any
    normal_velocity: Any
    vertical_velocity: Any
    dtype: np.dtype[Any]
    h2d: TransferStats

    @classmethod
    def empty(
        cls,
        *,
        n_vert_levels: int,
        n_cells: int,
        n_edges: int,
        dtype: Any = np.float32,
    ) -> "DeviceSavedDiagnostics":
        """Allocate a device-only target for self-derived t0 diagnostics."""

        require_cuda()
        cp = _cp()
        selected = np.dtype(dtype)
        if selected not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("DeviceSavedDiagnostics dtype must be float32 or float64")

        def cell_field() -> Any:
            return cp.empty((n_vert_levels, n_cells), dtype=selected)

        result = cls(
            theta_m=cell_field(),
            exner=cell_field(),
            density_perturbation=cell_field(),
            rho_theta_perturbation=cell_field(),
            pressure_perturbation=cell_field(),
            normal_velocity=cp.empty((n_vert_levels, n_edges), dtype=selected),
            vertical_velocity=cp.empty(
                (n_vert_levels + 1, n_cells), dtype=selected
            ),
            dtype=selected,
            h2d=TransferStats(0, 0.0),
        )
        result.validate(
            n_vert_levels=n_vert_levels,
            n_cells=n_cells,
            n_edges=n_edges,
        )
        return result

    @classmethod
    def from_host(
        cls, saved: object, *, dtype: Any | None = None
    ) -> "DeviceSavedDiagnostics":
        require_cuda()
        cp = _cp()
        selected = np.dtype(
            np.asarray(getattr(saved, "theta_m")).dtype if dtype is None else dtype
        )
        names = (
            "theta_m",
            "exner",
            "density_perturbation",
            "rho_theta_perturbation",
            "pressure_perturbation",
            "normal_velocity",
            "vertical_velocity",
        )
        host = {
            name: _require_same_host_dtype(name, getattr(saved, name), selected)
            for name in names
        }
        started = time.perf_counter()
        device = {name: cp.asarray(value) for name, value in host.items()}
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        return cls(
            **device,
            dtype=selected,
            h2d=TransferStats(
                sum(int(value.nbytes) for value in device.values()), elapsed
            ),
        )

    def validate(self, *, n_vert_levels: int, n_cells: int, n_edges: int) -> None:
        cell_shape = (n_vert_levels, n_cells)
        for name in (
            "theta_m",
            "exner",
            "density_perturbation",
            "rho_theta_perturbation",
            "pressure_perturbation",
        ):
            require_resident_array(
                f"saved.{name}", getattr(self, name), dtype=self.dtype, shape=cell_shape
            )
        require_resident_array(
            "saved.normal_velocity",
            self.normal_velocity,
            dtype=self.dtype,
            shape=(n_vert_levels, n_edges),
        )
        require_resident_array(
            "saved.vertical_velocity",
            self.vertical_velocity,
            dtype=self.dtype,
            shape=(n_vert_levels + 1, n_cells),
        )


@dataclass(slots=True)
class DeviceTerrainMetrics:
    zb_cell: Any
    zb3_cell: Any
    dtype: np.dtype[Any]
    h2d: TransferStats

    @classmethod
    def from_host(
        cls, terrain: object, *, dtype: Any | None = None
    ) -> "DeviceTerrainMetrics":
        require_cuda()
        cp = _cp()
        selected = np.dtype(
            np.asarray(getattr(terrain, "zb_cell")).dtype if dtype is None else dtype
        )
        host_zb = _as_float("zb_cell", getattr(terrain, "zb_cell"), selected)
        host_zb3 = _as_float("zb3_cell", getattr(terrain, "zb3_cell"), selected)
        started = time.perf_counter()
        zb = cp.asarray(host_zb)
        zb3 = cp.asarray(host_zb3)
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        return cls(
            zb, zb3, selected, TransferStats(int(zb.nbytes + zb3.nbytes), elapsed)
        )

    def validate(self, *, n_vert_levels: int, n_cells: int, max_edges: int) -> None:
        shape = (n_vert_levels + 1, n_cells, max_edges)
        require_resident_array(
            "terrain.zb_cell", self.zb_cell, dtype=self.dtype, shape=shape
        )
        require_resident_array(
            "terrain.zb3_cell", self.zb3_cell, dtype=self.dtype, shape=shape
        )


@dataclass(slots=True)
class DeviceAtmosphere:
    mesh: DeviceMesh
    state: DevicePrognosticState
    vertical: DeviceVerticalGrid
    reference: DeviceReferenceState
    saved: DeviceSavedDiagnostics
    terrain: DeviceTerrainMetrics | None
    h2d: TransferStats

    @classmethod
    def from_host(
        cls,
        mesh: object,
        state: object,
        vertical: object,
        reference: object,
        saved_diagnostics: object | None,
        terrain_metrics: object | None = None,
        *,
        dtype: Any | None = None,
        index_dtype: Any = np.int32,
    ) -> "DeviceAtmosphere":
        require_cuda()
        started = time.perf_counter()
        selected = np.asarray(getattr(state, "rho")).dtype if dtype is None else dtype
        device_mesh = DeviceMesh.from_host(
            mesh, dtype=selected, index_dtype=index_dtype
        )
        device_state = DevicePrognosticState.from_host(state, dtype=selected)
        device_vertical = DeviceVerticalGrid.from_host(vertical, dtype=selected)
        device_reference = DeviceReferenceState.from_host(reference, dtype=selected)
        device_saved = (
            DeviceSavedDiagnostics.empty(
                n_vert_levels=device_vertical.n_vert_levels,
                n_cells=device_mesh.n_cells,
                n_edges=device_mesh.n_edges,
                dtype=selected,
            )
            if saved_diagnostics is None
            else DeviceSavedDiagnostics.from_host(
                saved_diagnostics, dtype=selected
            )
        )
        device_terrain = (
            None
            if terrain_metrics is None
            else DeviceTerrainMetrics.from_host(terrain_metrics, dtype=selected)
        )
        _cp().cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        components = (
            device_mesh,
            device_state,
            device_vertical,
            device_reference,
            device_saved,
            device_terrain,
        )
        byte_count = int(sum(item.h2d.bytes for item in components if item is not None))
        result = cls(
            device_mesh,
            device_state,
            device_vertical,
            device_reference,
            device_saved,
            device_terrain,
            TransferStats(byte_count, elapsed),
        )
        result.validate()
        return result

    def validate(self) -> None:
        """Fail closed before any raw kernel can observe an inconsistent atmosphere."""

        self.mesh.validate()
        nlev = self.vertical.n_vert_levels
        ncells = self.mesh.n_cells
        nedges = self.mesh.n_edges
        expected_dtype = np.dtype(self.state.dtype)
        for component_name, component in (
            ("mesh", self.mesh),
            ("vertical", self.vertical),
            ("reference", self.reference),
            ("saved", self.saved),
        ):
            if np.dtype(component.dtype) != expected_dtype:
                raise TypeError(
                    f"{component_name}.dtype must match state dtype {expected_dtype}"
                )
        if self.terrain is not None and np.dtype(self.terrain.dtype) != expected_dtype:
            raise TypeError(f"terrain.dtype must match state dtype {expected_dtype}")
        self.vertical.validate(n_cells=ncells, n_edges=nedges)
        self.reference.validate(n_vert_levels=nlev, n_cells=ncells)
        self.state.validate(n_vert_levels=nlev, n_cells=ncells, n_edges=nedges)
        self.saved.validate(n_vert_levels=nlev, n_cells=ncells, n_edges=nedges)
        if self.terrain is not None:
            self.terrain.validate(
                n_vert_levels=nlev,
                n_cells=ncells,
                max_edges=self.mesh.max_edges,
            )


__all__ = [
    "DeviceAtmosphere",
    "DeviceMesh",
    "DevicePrognosticState",
    "DeviceReferenceState",
    "DeviceSavedDiagnostics",
    "DeviceTerrainMetrics",
    "DeviceVerticalGrid",
    "TransferStats",
    "require_resident_array",
]
