"""MPAS-A v8.4.1 height-based hybrid vertical coordinate authority.

Logical floating fields use Fortran order ``(vertical_level, horizontal_entity)``.
The scalar implementation follows ``src/core_init_atmosphere/mpas_init_atm_cases.F``
from the pinned MPAS-A v8.4.1 source.  It deliberately keeps the native loop
ordering for terrain and coordinate-surface smoothing so a compiled native
oracle can referee the result.

Two details are load-bearing:

* terrain smoothing skips a cell when the current field is exactly zero, as the
  native first and second passes do; coordinate-surface smoothing does *not*
  use that guard;
* the native ``do k=2,kz-1`` surface loop maps to Python
  ``range(1, first_height_level)``.  Stopping at ``first_height_level - 1``
  omits one native interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .errors import ConfigurationRefusal

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]


def _mesh_array(mesh: object, name: str) -> NDArray[Any]:
    try:
        return np.asarray(getattr(mesh, name))
    except AttributeError:
        arrays = getattr(mesh, "arrays", None)
        if arrays is None or name not in arrays:
            raise AttributeError(f"mesh has no MPAS field {name!r}") from None
        return np.asarray(arrays[name])


@dataclass(frozen=True, slots=True)
class VerticalGrid:
    """Vertical coordinate and terrain metric fields consumed by the dycore."""

    zw: FloatArray
    dzw: FloatArray
    rdzw: FloatArray
    zu: FloatArray
    dzu: FloatArray
    rdzu: FloatArray
    rdzwp: FloatArray
    rdzwm: FloatArray
    fzp: FloatArray
    fzm: FloatArray
    ah: FloatArray
    hx: FloatArray
    zgrid: FloatArray
    zz: FloatArray
    zxu: FloatArray
    dss: FloatArray
    cf1: float
    cf2: float
    cf3: float
    first_height_level: int

    @property
    def n_vert_levels(self) -> int:
        return int(self.dzw.size)


@dataclass(frozen=True, slots=True)
class VerticalEdgeMetrics:
    """Edge-oriented omega-diagnosis metrics in logical Fortran order.

    Both arrays have shape ``(nVertLevels + 1, 2, nEdges)``.  MPAS computes
    rows 1..nVertLevels and leaves the top interface row at exact +0.
    """

    zb: FloatArray
    zb3: FloatArray
    theta_adv_order: int


def _refuse(knob: str, value: object, reason: str, remedy: str) -> None:
    raise ConfigurationRefusal(knob, value, reason, remedy)


def _vertical_interfaces(
    n_vert_levels: int,
    ztop: float,
    dtype: np.dtype[Any],
    scheme: Literal["tc", "legacy", "specified"],
    specified_zw: FloatArray | None,
) -> FloatArray:
    """Transcribe the three v8.4.1 interface-height branches."""

    nz = n_vert_levels + 1
    if scheme == "specified":
        if specified_zw is None:
            _refuse(
                "specified_zeta_levels",
                None,
                "scheme='specified' requires nVertLevels+1 interface heights",
                "specified_zeta_levels=[0, ..., ztop]",
            )
        zw = np.asarray(specified_zw, dtype=dtype)
        if zw.shape != (nz,):
            _refuse(
                "specified_zeta_levels",
                f"shape={zw.shape}",
                f"the native source requires exactly {nz} values",
                f"an array with shape ({nz},)",
            )
        return np.array(zw, copy=True)

    eta = np.arange(nz, dtype=dtype) / dtype.type(n_vert_levels)
    if scheme == "legacy":
        return eta ** dtype.type(1.5) * dtype.type(ztop)
    if scheme != "tc":
        _refuse(
            "vertical_grid_scheme",
            scheme,
            "the pinned source has only specified, tc, and legacy branches",
            "vertical_grid_scheme='tc'",
        )

    if n_vert_levels >= 55:
        als, alt, zetal = (0.075, 1.70, 0.75)
    else:
        als, alt, zetal = (0.075, 1.23, 0.31)
    als = dtype.type(als)
    alt = dtype.type(alt)
    zetal = dtype.type(zetal)
    zl = dtype.type(1.0) - alt * (dtype.type(1.0) - zetal)
    zw = np.empty(nz, dtype=dtype)
    low = eta < zetal
    scaled = eta / zetal
    zw[low] = (
        als * eta[low]
        + (
            dtype.type(3.0) * (dtype.type(1.0) - alt)
            + dtype.type(2.0) * (alt - als) * zetal
        )
        * scaled[low] ** 2
        - (
            dtype.type(2.0) * (dtype.type(1.0) - alt)
            + (alt - als) * zetal
        )
        * scaled[low] ** 3
    ) * dtype.type(ztop)
    zw[~low] = (zl + alt * (eta[~low] - zetal)) * dtype.type(ztop)
    return zw


def _laplacian_without_area(
    mesh: object,
    field: FloatArray,
    *,
    skip_zero_center: bool,
) -> FloatArray:
    """Native weighted neighbor sum, including the regional garbage-cell rule."""

    n_edges_on_cell = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
    edges_on_cell = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
    cells_on_cell = _mesh_array(mesh, "cellsOnCell").astype(np.int64, copy=False)
    dv_edge = _mesh_array(mesh, "dvEdge")
    dc_edge = _mesh_array(mesh, "dcEdge")
    out = np.zeros_like(field)
    for cell in range(field.size):
        if skip_zero_center and field[cell] == field.dtype.type(0.0):
            continue
        for slot in range(int(n_edges_on_cell[cell])):
            edge = int(edges_on_cell[cell, slot])
            neighbor = int(cells_on_cell[cell, slot])
            neighbor_value = field[cell] if neighbor < 0 else field[neighbor]
            out[cell] += (
                dv_edge[edge] / dc_edge[edge] * (neighbor_value - field[cell])
            )
    return out


def smooth_terrain(mesh: object, terrain: FloatArray, passes: int = 1) -> FloatArray:
    """Apply the native two-pass fourth-order terrain smoother."""

    if passes < 0:
        _refuse(
            "nsmterrain",
            passes,
            "terrain smoothing passes cannot be negative",
            "nsmterrain=1",
        )
    result = np.asarray(terrain).copy()
    for _ in range(passes):
        first = _laplacian_without_area(
            mesh, result, skip_zero_center=True
        )
        hs = result + result.dtype.type(0.216) * first
        second = _laplacian_without_area(mesh, hs, skip_zero_center=True)
        result = hs - hs.dtype.type(0.216) * second
    return result


def _validate_parameters(
    *,
    n_vert_levels: int,
    ztop: float,
    terrain_smoothing_passes: int,
    surface_smoothing_passes: int,
    minimum_layer_fraction: float,
    hybrid_coordinate: bool,
    hybrid_transition_height: float,
    xnutr: float,
    damping_start: float,
) -> None:
    if n_vert_levels < 3:
        _refuse(
            "nVertLevels",
            n_vert_levels,
            "the lower-boundary extrapolation coefficients require at least three layers",
            "nVertLevels>=3",
        )
    if not np.isfinite(ztop) or ztop <= 0.0:
        _refuse("ztop", ztop, "model-top height must be finite and positive", "ztop=30000.0")
    if terrain_smoothing_passes < 0:
        _refuse("nsmterrain", terrain_smoothing_passes, "passes cannot be negative", "nsmterrain>=0")
    if surface_smoothing_passes < 0:
        _refuse("nsm", surface_smoothing_passes, "passes cannot be negative", "nsm>=0")
    if not np.isfinite(minimum_layer_fraction) or not 0.0 < minimum_layer_fraction < 1.0:
        _refuse(
            "dzmin",
            minimum_layer_fraction,
            "the accepted physical layer fraction must lie strictly between zero and one",
            "0 < dzmin < 1",
        )
    if hybrid_coordinate and (
        not np.isfinite(hybrid_transition_height) or hybrid_transition_height <= 0.0
    ):
        _refuse(
            "hybrid_top_z",
            hybrid_transition_height,
            "a hybrid coordinate needs a finite positive transition height",
            "hybrid_top_z=30000.0",
        )
    if not np.isfinite(xnutr):
        _refuse("xnutr", xnutr, "the damping amplitude must be finite", "xnutr=0.0")
    if not np.isfinite(damping_start):
        _refuse("zd", damping_start, "the damping start must be finite", "zd=22000.0")
    if xnutr != 0.0 and not damping_start < ztop:
        _refuse(
            "zd",
            damping_start,
            "nonzero upper-level damping with zd>=ztop divides by a zero or negative layer depth",
            "zd < ztop or xnutr=0",
        )


def build_vertical_grid(
    mesh: object,
    terrain: FloatArray,
    *,
    n_vert_levels: int = 55,
    ztop: float = 30_000.0,
    scheme: Literal["tc", "legacy", "specified"] = "tc",
    specified_zw: FloatArray | None = None,
    interface_projection: Literal["linear_interpolation", "layer_integral"] = "linear_interpolation",
    terrain_smoothing_passes: int = 1,
    smooth_surfaces: bool = True,
    surface_smoothing_passes: int = 30,
    minimum_layer_fraction: float = 0.3,
    hybrid_coordinate: bool = True,
    hybrid_transition_height: float = 30_000.0,
    xnutr: float = 0.0,
    damping_start: float = 22_000.0,
) -> VerticalGrid:
    """Build all native vertical-coordinate fields for a closed/global mesh."""

    _validate_parameters(
        n_vert_levels=n_vert_levels,
        ztop=ztop,
        terrain_smoothing_passes=terrain_smoothing_passes,
        surface_smoothing_passes=surface_smoothing_passes,
        minimum_layer_fraction=minimum_layer_fraction,
        hybrid_coordinate=hybrid_coordinate,
        hybrid_transition_height=hybrid_transition_height,
        xnutr=xnutr,
        damping_start=damping_start,
    )
    if interface_projection not in ("linear_interpolation", "layer_integral"):
        _refuse(
            "interface_projection",
            interface_projection,
            "only the two pinned MPAS projection branches exist",
            "interface_projection='linear_interpolation'",
        )

    ter = np.asarray(terrain)
    if ter.ndim != 1:
        raise ValueError(f"terrain must be one-dimensional over nCells, got {ter.shape}")
    n_cells = int(_mesh_array(mesh, "areaCell").size)
    if ter.shape != (n_cells,):
        raise ValueError(f"terrain shape {ter.shape} does not match nCells={n_cells}")
    if ter.dtype.kind != "f":
        ter = ter.astype(np.float64)
    if not np.all(np.isfinite(ter)):
        raise ValueError("terrain contains non-finite values")
    dtype = ter.dtype

    ter = smooth_terrain(mesh, ter, terrain_smoothing_passes)
    zw = _vertical_interfaces(n_vert_levels, ztop, dtype, scheme, specified_zw)
    if not np.all(np.isfinite(zw)) or not np.all(np.diff(zw) > 0):
        _refuse(
            "specified_zeta_levels" if scheme == "specified" else "vertical_grid_scheme",
            "non-monotonic",
            "MPAS interface heights must be finite and strictly increasing",
            "strictly increasing heights from 0 to ztop",
        )
    zero_tolerance = np.finfo(dtype).eps * max(float(abs(zw[-1])), 1.0) * 8.0
    if abs(float(zw[0])) > zero_tolerance:
        _refuse(
            "specified_zeta_levels" if scheme == "specified" else "vertical_grid_scheme",
            float(zw[0]),
            "the first reference interface must be model height zero",
            "the first interface equal to 0.0 m",
        )
    if scheme == "specified":
        ztop = float(zw[-1])
        if xnutr != 0.0 and not damping_start < ztop:
            _refuse("zd", damping_start, "damping begins at or above the specified top", "zd < specified top")

    dzw = np.diff(zw)
    rdzw = np.reciprocal(dzw)
    zu = (zw[:-1] + zw[1:]) * dtype.type(0.5)
    dzu = np.full(n_vert_levels, np.nan, dtype=dtype)
    rdzu = np.full(n_vert_levels, np.nan, dtype=dtype)
    rdzwp = np.full(n_vert_levels, np.nan, dtype=dtype)
    rdzwm = np.full(n_vert_levels, np.nan, dtype=dtype)
    dzu[1:] = dtype.type(0.5) * (dzw[1:] + dzw[:-1])
    rdzu[1:] = np.reciprocal(dzu[1:])
    rdzwp[1:] = dzw[:-1] / (dzw[1:] * (dzw[1:] + dzw[:-1]))
    rdzwm[1:] = dzw[1:] / (dzw[:-1] * (dzw[1:] + dzw[:-1]))

    fzp = np.full(n_vert_levels, np.nan, dtype=dtype)
    fzm = np.full(n_vert_levels, np.nan, dtype=dtype)
    if interface_projection == "linear_interpolation":
        fzp[1:] = dtype.type(0.5) * dzw[1:] / dzu[1:]
        fzm[1:] = dtype.type(0.5) * dzw[:-1] / dzu[1:]
    else:
        fzm[1:] = dtype.type(0.5) * dzw[1:] / dzu[1:]
        fzp[1:] = dtype.type(0.5) * dzw[:-1] / dzu[1:]

    cof1 = (
        (dtype.type(2.0) * dzu[1] + dzu[2])
        / (dzu[1] + dzu[2])
        * dzw[0]
        / dzu[1]
    )
    cof2 = dzu[1] / (dzu[1] + dzu[2]) * dzw[0] / dzu[2]
    cf1 = float(fzp[1] + cof1)
    cf2 = float(fzm[1] - cof1 - cof2)
    cf3 = float(cof2)

    if hybrid_coordinate:
        ah = np.zeros(n_vert_levels + 1, dtype=dtype)
        below = zw < dtype.type(hybrid_transition_height)
        ah[below] = np.cos(
            dtype.type(0.5 * np.pi)
            * zw[below]
            / dtype.type(hybrid_transition_height)
        ) ** 6
        candidates = np.flatnonzero(~below)
        first_height_level = (
            int(candidates[0]) if candidates.size else n_vert_levels + 1
        )
    else:
        ah = dtype.type(1.0) - zw / dtype.type(ztop)
        first_height_level = n_vert_levels + 1

    hx = np.broadcast_to(ter, (n_vert_levels + 1, n_cells)).copy()
    if smooth_surfaces:
        mean_dc = np.empty(n_cells, dtype=dtype)
        n_edges_on_cell = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
        edges_on_cell = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
        dc_edge = _mesh_array(mesh, "dcEdge")
        for cell in range(n_cells):
            count = int(n_edges_on_cell[cell])
            if count <= 0:
                raise ValueError(f"cell {cell} has no active edges")
            active_edges = edges_on_cell[cell, :count]
            mean_dc[cell] = np.mean(dc_edge[active_edges], dtype=dtype)
        sm0 = np.maximum(
            dtype.type(0.01),
            dtype.type(0.125)
            * np.minimum(dtype.type(1.0), dtype.type(3000.0) / mean_dc),
        )
        stop = min(first_height_level, n_vert_levels)
        for level in range(1, stop):
            hx[level] = hx[level - 1]
            for _ in range(surface_smoothing_passes + level + 1):
                sm = sm0 * min((3.0 * float(zw[level]) / ztop) ** 2.0, 1.0)
                candidate = hx[level] + sm * _laplacian_without_area(
                    mesh, hx[level], skip_zero_center=False
                )
                thickness = (
                    zw[level] + ah[level] * candidate
                    - zw[level - 1]
                    - ah[level - 1] * hx[level - 1]
                )
                accept = thickness > dtype.type(minimum_layer_fraction) * dzw[level - 1]
                hx[level, accept] = candidate[accept]
        if first_height_level <= n_vert_levels:
            hx[first_height_level:] = dtype.type(0.0)

    zgrid = zw[:, None] + ah[:, None] * hx
    physical_dz = np.diff(zgrid, axis=0)
    if not np.all(np.isfinite(zgrid)) or np.any(physical_dz <= 0):
        location = np.argwhere(~np.isfinite(physical_dz) | (physical_dz <= 0))
        sample = location[0].tolist() if location.size else None
        raise ValueError(
            "hybrid coordinate produced a non-finite or non-positive physical "
            f"layer thickness; first bad [level,cell]={sample}"
        )
    zz = dzw[:, None] / physical_dz

    cells_on_edge = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    dc_edge = _mesh_array(mesh, "dcEdge")
    c0 = cells_on_edge[:, 0]
    c1 = cells_on_edge[:, 1]
    if np.any(c0 < 0) or np.any(c1 < 0):
        _refuse(
            "regional_boundary_metrics",
            "boundary edge",
            "the closed-sphere vertical authority does not invent exterior state",
            "a closed global mesh or an explicit regional boundary implementation",
        )
    zxu = dtype.type(0.5) * (
        zgrid[:-1, c1]
        - zgrid[:-1, c0]
        + zgrid[1:, c1]
        - zgrid[1:, c0]
    ) / dc_edge[None, :]

    lower_interfaces = zgrid[:-1]
    dss = np.zeros_like(lower_interfaces)
    if xnutr != 0.0:
        active = lower_interfaces > dtype.type(damping_start + 0.1)
        phase = (
            dtype.type(0.5 * np.pi)
            * (lower_interfaces - dtype.type(damping_start))
            / dtype.type(ztop - damping_start)
        )
        dss[active] = dtype.type(xnutr) * np.sin(phase[active]) ** 2

    result = VerticalGrid(
        zw=zw,
        dzw=dzw,
        rdzw=rdzw,
        zu=zu,
        dzu=dzu,
        rdzu=rdzu,
        rdzwp=rdzwp,
        rdzwm=rdzwm,
        fzp=fzp,
        fzm=fzm,
        ah=ah,
        hx=hx,
        zgrid=zgrid,
        zz=zz,
        zxu=zxu,
        dss=dss,
        cf1=cf1,
        cf2=cf2,
        cf3=cf3,
        first_height_level=first_height_level,
    )
    validate_vertical_grid(result, n_cells=n_cells, n_edges=int(cells_on_edge.shape[0]))
    return result


def _logical_deriv_two(mesh: object, n_edges: int) -> FloatArray:
    raw = np.asarray(_mesh_array(mesh, "deriv_two"))
    if raw.dtype.kind != "f" or raw.ndim != 3:
        raise TypeError(f"deriv_two must be a three-dimensional floating array, got {raw.shape} {raw.dtype}")
    if raw.shape[2] == n_edges and raw.shape[1] == 2:
        logical = raw
    elif raw.shape[0] == n_edges and raw.shape[1] == 2:
        logical = np.transpose(raw, (2, 1, 0))
    elif raw.shape[1] == n_edges and raw.shape[0] == 2:
        logical = np.transpose(raw, (2, 0, 1))
    else:
        raise ValueError(
            "deriv_two orientation is unsupported: expected logical "
            f"(nCoeff,2,{n_edges}) or native ({n_edges},2,nCoeff), got {raw.shape}"
        )
    if logical.shape[0] < 2 or not np.all(np.isfinite(logical)):
        raise ValueError("deriv_two is too short or contains non-finite values")
    return np.asarray(logical)


def build_edge_vertical_metrics(
    mesh: object,
    vertical: VerticalGrid,
    *,
    theta_adv_order: int,
) -> VerticalEdgeMetrics:
    """Build ``zb`` and ``zb3`` exactly where native init does.

    This is the missing piece that prevented the existing vertical-grid port
    from replacing a native init capsule.  ``theta_adv_order`` 2 uses the edge
    average; orders 3/4 use ``deriv_two`` and only order 3 retains the odd
    correction in ``zb3``.
    """

    if theta_adv_order not in (2, 3, 4):
        _refuse(
            "theta_adv_order",
            theta_adv_order,
            "the pinned init source implements only orders 2, 3, and 4",
            "theta_adv_order=3",
        )
    cells_on_edge = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    n_edges = int(cells_on_edge.shape[0])
    if cells_on_edge.shape != (n_edges, 2):
        raise ValueError(f"cellsOnEdge must have shape (nEdges,2), got {cells_on_edge.shape}")
    n_cells = int(vertical.zgrid.shape[1])
    dc_edge = np.asarray(_mesh_array(mesh, "dcEdge"))
    dv_edge = np.asarray(_mesh_array(mesh, "dvEdge"))
    area_cell = np.asarray(_mesh_array(mesh, "areaCell"))
    if dc_edge.shape != (n_edges,) or dv_edge.shape != (n_edges,):
        raise ValueError("dcEdge/dvEdge do not match nEdges")
    if area_cell.shape != (n_cells,):
        raise ValueError("areaCell does not match vertical nCells")
    if (
        not np.all(np.isfinite(dc_edge))
        or not np.all(np.isfinite(dv_edge))
        or not np.all(np.isfinite(area_cell))
        or np.any(dc_edge <= 0)
        or np.any(area_cell <= 0)
    ):
        raise ValueError("edge lengths and cell areas must be finite and positive")

    deriv_two: FloatArray | None = None
    cells_on_cell: IntArray | None = None
    n_edges_on_cell: IntArray | None = None
    if theta_adv_order != 2:
        deriv_two = _logical_deriv_two(mesh, n_edges)
        cells_on_cell = _mesh_array(mesh, "cellsOnCell").astype(np.int64, copy=False)
        n_edges_on_cell = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
        required = int(np.max(n_edges_on_cell, initial=0)) + 1
        if deriv_two.shape[0] < required:
            raise ValueError(
                f"deriv_two has {deriv_two.shape[0]} coefficients but topology requires {required}"
            )

    dtype = vertical.zgrid.dtype
    zb = np.zeros((vertical.n_vert_levels + 1, 2, n_edges), dtype=dtype)
    zb3 = np.zeros_like(zb)
    for edge in range(n_edges):
        cell1 = int(cells_on_edge[edge, 0])
        cell2 = int(cells_on_edge[edge, 1])
        if cell1 < 0 and cell2 < 0:
            raise ValueError(f"edge {edge} has no interior cell")
        if cell1 < 0:
            cell1 = cell2
        if cell2 < 0:
            cell2 = cell1
        if cell1 >= n_cells or cell2 >= n_cells:
            raise ValueError(f"edge {edge} references an out-of-range cell")
        scale1 = dtype.type(dv_edge[edge] / area_cell[cell1])
        scale2 = dtype.type(dv_edge[edge] / area_cell[cell2])
        dc2_over_12 = dtype.type(dc_edge[edge] ** 2 / 12.0)
        for level in range(vertical.n_vert_levels):
            z1 = vertical.zgrid[level, cell1]
            z2 = vertical.zgrid[level, cell2]
            z_edge3 = dtype.type(0.0)
            if theta_adv_order == 2:
                z_edge = dtype.type(0.5) * (z1 + z2)
            else:
                assert deriv_two is not None
                assert cells_on_cell is not None
                assert n_edges_on_cell is not None
                d2_1 = deriv_two[0, 0, edge] * z1
                d2_2 = deriv_two[0, 1, edge] * z2
                for slot in range(int(n_edges_on_cell[cell1])):
                    neighbor = int(cells_on_cell[cell1, slot])
                    if neighbor >= 0:
                        d2_1 += deriv_two[slot + 1, 0, edge] * vertical.zgrid[level, neighbor]
                for slot in range(int(n_edges_on_cell[cell2])):
                    neighbor = int(cells_on_cell[cell2, slot])
                    if neighbor >= 0:
                        d2_2 += deriv_two[slot + 1, 1, edge] * vertical.zgrid[level, neighbor]
                z_edge = dtype.type(0.5) * (z1 + z2) - dc2_over_12 * (d2_1 + d2_2)
                if theta_adv_order == 3:
                    z_edge3 = -dc2_over_12 * (d2_1 - d2_2)
            zb[level, 0, edge] = (z_edge - z1) * scale1
            zb[level, 1, edge] = (z_edge - z2) * scale2
            zb3[level, 0, edge] = z_edge3 * scale1
            zb3[level, 1, edge] = z_edge3 * scale2

    if not np.all(np.isfinite(zb)) or not np.all(np.isfinite(zb3)):
        raise ValueError("constructed zb/zb3 contain non-finite values")
    return VerticalEdgeMetrics(zb=zb, zb3=zb3, theta_adv_order=theta_adv_order)


def runtime_vertical_vectors(vertical: VerticalGrid) -> dict[str, FloatArray]:
    """Return native-file vectors with unused lower slots materialized as +0."""

    vectors: dict[str, FloatArray] = {}
    for name in ("dzu", "rdzu", "rdzwp", "rdzwm", "fzm", "fzp"):
        value = np.array(getattr(vertical, name), copy=True)
        value[0] = value.dtype.type(0.0)
        vectors[name] = value
    vectors["rdzw"] = np.array(vertical.rdzw, copy=True)
    return vectors


def validate_vertical_grid(
    vertical: VerticalGrid,
    *,
    n_cells: int | None = None,
    n_edges: int | None = None,
) -> dict[str, float | int]:
    """Independent structural and physical invariant gate."""

    nlev = vertical.n_vert_levels
    nc = int(vertical.zgrid.shape[1])
    ne = int(vertical.zxu.shape[1])
    if n_cells is not None and nc != n_cells:
        raise ValueError(f"vertical nCells={nc} != expected {n_cells}")
    if n_edges is not None and ne != n_edges:
        raise ValueError(f"vertical nEdges={ne} != expected {n_edges}")
    expected = {
        "zw": (nlev + 1,),
        "dzw": (nlev,),
        "rdzw": (nlev,),
        "zu": (nlev,),
        "dzu": (nlev,),
        "rdzu": (nlev,),
        "rdzwp": (nlev,),
        "rdzwm": (nlev,),
        "fzp": (nlev,),
        "fzm": (nlev,),
        "ah": (nlev + 1,),
        "hx": (nlev + 1, nc),
        "zgrid": (nlev + 1, nc),
        "zz": (nlev, nc),
        "zxu": (nlev, ne),
        "dss": (nlev, nc),
    }
    for name, shape in expected.items():
        value = np.asarray(getattr(vertical, name))
        if value.shape != shape:
            raise ValueError(f"{name} shape {value.shape} != {shape}")
    for name in ("zw", "dzw", "rdzw", "zu", "ah", "hx", "zgrid", "zz", "zxu", "dss"):
        if not np.all(np.isfinite(np.asarray(getattr(vertical, name)))):
            raise ValueError(f"{name} contains non-finite values")
    for name in ("dzu", "rdzu", "rdzwp", "rdzwm", "fzp", "fzm"):
        value = np.asarray(getattr(vertical, name))
        if not np.all(np.isfinite(value[1:])):
            raise ValueError(f"{name} contains a non-finite active coefficient")
    dz = np.diff(vertical.zgrid, axis=0)
    if np.any(dz <= 0.0):
        raise ValueError("zgrid has a non-positive physical layer")
    if not np.all(np.diff(vertical.zw) > 0.0):
        raise ValueError("zw is not strictly increasing")
    if not np.all(vertical.zz > 0.0):
        raise ValueError("zz must be strictly positive")
    return {
        "n_vert_levels": nlev,
        "n_cells": nc,
        "n_edges": ne,
        "minimum_physical_layer_m": float(np.min(dz)),
        "maximum_physical_layer_m": float(np.max(dz)),
        "minimum_zz": float(np.min(vertical.zz)),
        "maximum_zz": float(np.max(vertical.zz)),
    }
