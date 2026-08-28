"""Device-only context for the additive MPAS-A v8.4.1 CUDA lane.

The frozen v8.2.3 CUDA containers intentionally do not grow release-specific
fields.  This sidecar owns the pre-rounded RKIND geometry inverses, released
height-dependent acoustic coefficients, and perturbation-Coriolis reference
profiles consumed only by the v8.4.1 mirror.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .cuda_backend.containers import TransferStats, require_resident_array
from .dynamics_v841 import V841ReferenceWindProfiles, precomputed_mesh_inverse_v841
from .offcentering_v841 import AcousticOffcenteringV841


def _mesh_array(mesh: object, name: str) -> np.ndarray[Any, Any]:
    try:
        return np.asarray(getattr(mesh, name))
    except AttributeError:
        arrays = getattr(mesh, "arrays", None)
        if arrays is None or name not in arrays:
            raise AttributeError(f"mesh has no MPAS field {name!r}") from None
        return np.asarray(arrays[name])


@dataclass(frozen=True, slots=True)
class CudaV841Context:
    """Resident arrays that have no v8.2.3 numerical meaning."""

    inv_area_cell: Any
    inv_area_triangle: Any
    inv_dc_edge: Any
    inv_dv_edge: Any
    etp: Any
    etm: Any
    ewp: Any
    ewm: Any
    u_init: Any
    v_init: Any
    h2d: TransferStats

    @classmethod
    def from_host(
        cls,
        mesh: object,
        offcentering: AcousticOffcenteringV841,
        reference_wind: V841ReferenceWindProfiles,
        *,
        n_vert_levels: int,
        dtype: Any = np.float32,
    ) -> "CudaV841Context":
        """Upload source-initialized v8.4.1 arrays after admission."""

        selected = np.dtype(dtype)
        if selected != np.dtype(np.float32):
            raise TypeError("the v8.4.1 CUDA mirror currently requires float32 RKIND")
        reference_wind.validate(n_vert_levels, selected)

        geometry = {
            "areaCell": _mesh_array(mesh, "areaCell"),
            "areaTriangle": _mesh_array(mesh, "areaTriangle"),
            "dcEdge": _mesh_array(mesh, "dcEdge"),
            "dvEdge": _mesh_array(mesh, "dvEdge"),
        }
        n_cells = int(geometry["areaCell"].size)
        n_vertices = int(geometry["areaTriangle"].size)
        n_edges = int(geometry["dcEdge"].size)
        geometry_shapes = {
            "areaCell": (n_cells,),
            "areaTriangle": (n_vertices,),
            "dcEdge": (n_edges,),
            "dvEdge": (n_edges,),
        }
        for name, shape in geometry_shapes.items():
            value = np.asarray(geometry[name])
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} != {shape}")
            rounded = value.astype(selected, copy=False)
            if not np.all(np.isfinite(rounded)) or np.any(rounded <= 0.0):
                raise ValueError(
                    f"{name} must remain finite and strictly positive in float32"
                )

        profile_shapes = {
            "etp": (n_vert_levels,),
            "etm": (n_vert_levels,),
            "ewp": (n_vert_levels + 1,),
            "ewm": (n_vert_levels + 1,),
        }
        profile_host: dict[str, np.ndarray[Any, Any]] = {}
        for name, shape in profile_shapes.items():
            value = np.asarray(getattr(offcentering, name))
            if value.dtype != selected:
                raise TypeError(
                    f"v8.4.1 CUDA offcentering {name} dtype {value.dtype} "
                    f"!= {selected}"
                )
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"v8.4.1 CUDA offcentering {name} must be finite with shape {shape}"
                )
            if not value.flags.c_contiguous:
                raise ValueError(
                    f"v8.4.1 CUDA offcentering {name} must be C-contiguous"
                )
            profile_host[name] = value

        for name in ("u_init", "v_init"):
            value = np.asarray(getattr(reference_wind, name))
            if not value.flags.c_contiguous:
                raise ValueError(
                    f"v8.4.1 CUDA reference profile {name} must be C-contiguous"
                )

        host = {
            "inv_area_cell": np.ascontiguousarray(
                precomputed_mesh_inverse_v841(mesh, "areaCell", selected)
            ),
            "inv_area_triangle": np.ascontiguousarray(
                precomputed_mesh_inverse_v841(mesh, "areaTriangle", selected)
            ),
            "inv_dc_edge": np.ascontiguousarray(
                precomputed_mesh_inverse_v841(mesh, "dcEdge", selected)
            ),
            "inv_dv_edge": np.ascontiguousarray(
                precomputed_mesh_inverse_v841(mesh, "dvEdge", selected)
            ),
            **profile_host,
            "u_init": np.asarray(reference_wind.u_init),
            "v_init": np.asarray(reference_wind.v_init),
        }
        expected = {
            "inv_area_cell": (n_cells,),
            "inv_area_triangle": (n_vertices,),
            "inv_dc_edge": (n_edges,),
            "inv_dv_edge": (n_edges,),
            "etp": (n_vert_levels,),
            "etm": (n_vert_levels,),
            "ewp": (n_vert_levels + 1,),
            "ewm": (n_vert_levels + 1,),
            "u_init": (n_vert_levels,),
            "v_init": (n_vert_levels,),
        }
        for name, shape in expected.items():
            value = np.asarray(host[name])
            if value.shape != shape or value.dtype != selected:
                raise ValueError(
                    f"v8.4.1 CUDA context {name} {value.shape}/{value.dtype} "
                    f"!= {shape}/{selected}"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(f"v8.4.1 CUDA context {name} must be finite")

        from .cuda_backend import require_cuda

        require_cuda(min_compute=(12, 0))
        import cupy as cp

        started = time.perf_counter()
        device = {name: cp.asarray(value) for name, value in host.items()}
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        transferred = int(sum(int(value.nbytes) for value in device.values()))
        return cls(**device, h2d=TransferStats(transferred, elapsed))

    def validate(
        self,
        *,
        n_vert_levels: int,
        n_cells: int,
        n_edges: int,
        n_vertices: int,
    ) -> None:
        expected = {
            "inv_area_cell": (n_cells,),
            "inv_area_triangle": (n_vertices,),
            "inv_dc_edge": (n_edges,),
            "inv_dv_edge": (n_edges,),
            "etp": (n_vert_levels,),
            "etm": (n_vert_levels,),
            "ewp": (n_vert_levels + 1,),
            "ewm": (n_vert_levels + 1,),
            "u_init": (n_vert_levels,),
            "v_init": (n_vert_levels,),
        }
        for name, shape in expected.items():
            require_resident_array(
                f"v841.{name}", getattr(self, name), dtype=np.float32, shape=shape
            )


__all__ = ["CudaV841Context"]
