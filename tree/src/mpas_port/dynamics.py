"""Large-step dry MPAS-A dynamics tendencies.

This module ports the horizontally local and finite-volume sections of frozen
``src/core_atmosphere/dynamics/mpas_atm_time_integration.F``
``atm_compute_dyn_tend_work`` (lines 4459-5391).  Transport limiting is kept in
``transport.py``; the vertically implicit small-step solve is in
``acoustic.py``.

Logical array order is ``(vertical_level, horizontal_entity)`` and loop order
is retained where the Fortran accumulates an unstructured stencil.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .acoustic import edge_signs_on_cells
from .diagnostics import edge_signs_on_vertices
from .errors import ConfigurationRefusal

FloatArray = NDArray[np.floating[Any]]


def _mesh_array(mesh: object, name: str) -> NDArray[Any]:
    try:
        return np.asarray(getattr(mesh, name))
    except AttributeError:
        arrays = getattr(mesh, "arrays", None)
        if arrays is None or name not in arrays:
            raise AttributeError(f"mesh has no MPAS field {name!r}") from None
        return np.asarray(arrays[name])


def _source_inverse(
    value: FloatArray,
    name: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> FloatArray:
    inverse = np.asarray(value)
    if inverse.dtype != dtype:
        raise TypeError(f"{name} dtype {inverse.dtype} != RKIND {dtype}")
    if inverse.shape != shape:
        raise ValueError(f"{name} shape {inverse.shape} != {shape}")
    if not np.all(np.isfinite(inverse)):
        raise ValueError(f"{name} must be finite")
    return inverse


def flux4(
    q_im2: FloatArray | float,
    q_im1: FloatArray | float,
    q_i: FloatArray | float,
    q_ip1: FloatArray | float,
    velocity: FloatArray | float,
) -> FloatArray:
    """Fourth-order flux statement function at frozen lines 4625-4626."""
    return np.asarray(velocity) * (
        7.0 * (np.asarray(q_i) + np.asarray(q_im1))
        - (np.asarray(q_ip1) + np.asarray(q_im2))
    ) / 12.0


def flux3(
    q_im2: FloatArray | float,
    q_im1: FloatArray | float,
    q_i: FloatArray | float,
    q_ip1: FloatArray | float,
    velocity: FloatArray | float,
    coefficient: float,
) -> FloatArray:
    """Third-order upwind-biased flux at frozen
    ``src/core_atmosphere/dynamics/mpas_atm_time_integration.F:4636-4638``.
    """
    vel = np.asarray(velocity)
    return flux4(q_im2, q_im1, q_i, q_ip1, vel) + coefficient * np.abs(vel) * (
        (np.asarray(q_ip1) - np.asarray(q_im2))
        - 3.0 * (np.asarray(q_i) - np.asarray(q_im1))
    ) / 12.0


def mass_flux_divergence(
    mesh: object,
    rho_u: FloatArray,
    *,
    inv_area_cell: FloatArray | None = None,
) -> FloatArray:
    """Horizontal mass-flux divergence (Fortran lines 4705-4728)."""
    ru = np.asarray(rho_u)
    if ru.ndim != 2 or ru.dtype.kind != "f":
        raise ValueError("rho_u must be floating with shape (nVertLevels, nEdges)")
    nlev, nedges = ru.shape
    ncells = _mesh_array(mesh, "areaCell").size
    if _mesh_array(mesh, "dvEdge").size != nedges:
        raise ValueError("rho_u edge dimension does not match mesh")
    counts = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
    edges = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
    signs = edge_signs_on_cells(mesh).astype(ru.dtype, copy=False)
    dv = _mesh_array(mesh, "dvEdge").astype(ru.dtype, copy=False)
    area = _mesh_array(mesh, "areaCell").astype(ru.dtype, copy=False)
    inverse_area = None
    if inv_area_cell is not None:
        inverse_area = _source_inverse(
            inv_area_cell, "inv_area_cell", ru.dtype, area.shape
        )
    result = np.zeros((nlev, ncells), dtype=ru.dtype)
    for cell in range(ncells):
        for slot in range(int(counts[cell])):
            edge = int(edges[cell, slot])
            result[:, cell] += signs[cell, slot] * dv[edge] * ru[:, edge]
        if inverse_area is None:
            result[:, cell] /= area[cell]
        else:
            result[:, cell] *= inverse_area[cell]
    return result


def density_tendency(
    mesh: object,
    rho_u: FloatArray,
    rho_w: FloatArray,
    rdzw: FloatArray,
    physics_tendency: FloatArray | None = None,
    *,
    inv_area_cell: FloatArray | None = None,
) -> FloatArray:
    """Complete density tendency at frozen lines 4732-4746."""
    horizontal = mass_flux_divergence(
        mesh,
        rho_u,
        inv_area_cell=inv_area_cell,
    )
    rw = np.asarray(rho_w)
    if rw.shape != (horizontal.shape[0] + 1, horizontal.shape[1]):
        raise ValueError("rho_w must have shape (nVertLevels+1, nCells)")
    rdzw = np.asarray(rdzw, dtype=horizontal.dtype)
    if rdzw.shape != (horizontal.shape[0],):
        raise ValueError("rdzw vertical dimension mismatch")
    result = -horizontal - rdzw[:, None] * np.diff(rw, axis=0)
    if physics_tendency is not None:
        if np.shape(physics_tendency) != result.shape:
            raise ValueError("density physics tendency shape mismatch")
        result += physics_tendency
    return result


def smagorinsky_diffusivity(
    mesh: object,
    normal_velocity: FloatArray,
    tangential_velocity: FloatArray,
    defc_a: FloatArray,
    defc_b: FloatArray,
    *,
    c_s: float,
    length_scale: float,
    dt: float,
) -> FloatArray:
    """2-D Smagorinsky coefficient at frozen lines 4647-4673."""
    u = np.asarray(normal_velocity)
    v = np.asarray(tangential_velocity)
    if u.shape != v.shape or u.ndim != 2:
        raise ValueError("u and v must share shape (nVertLevels, nEdges)")
    if dt <= 0.0 or length_scale <= 0.0:
        raise ValueError("dt and length_scale must be positive")
    counts = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
    edges = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
    ncells = counts.size
    if np.shape(defc_a) != edges.shape or np.shape(defc_b) != edges.shape:
        raise ValueError("defc coefficient shapes must equal edgesOnCell")
    result = np.zeros((u.shape[0], ncells), dtype=u.dtype)
    ceiling = u.dtype.type(0.01 * length_scale**2 / dt)
    scale2 = u.dtype.type((c_s * length_scale) ** 2)
    for cell in range(ncells):
        diagonal = np.zeros(u.shape[0], dtype=u.dtype)
        off_diagonal = np.zeros(u.shape[0], dtype=u.dtype)
        for slot in range(int(counts[cell])):
            edge = int(edges[cell, slot])
            diagonal += defc_a[cell, slot] * u[:, edge] - defc_b[cell, slot] * v[:, edge]
            off_diagonal += defc_b[cell, slot] * u[:, edge] + defc_a[cell, slot] * v[:, edge]
        result[:, cell] = np.minimum(scale2 * np.sqrt(diagonal**2 + off_diagonal**2), ceiling)
    return result


def pressure_gradient_euler_tendency(
    mesh: object,
    pressure_perturbation: FloatArray,
    dpdz: FloatArray,
    cqu: FloatArray,
    zz: FloatArray,
    zxu: FloatArray,
    *,
    inv_dc_edge: FloatArray | None = None,
) -> FloatArray:
    """Terrain-coordinate pressure gradient, frozen lines 4755-4767."""
    pp = np.asarray(pressure_perturbation)
    if pp.shape != np.shape(dpdz) or pp.shape != np.shape(zz):
        raise ValueError("pressure, dpdz, and zz cell fields must share shape")
    cells = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    if inv_dc_edge is None:
        inv_dc = np.reciprocal(_mesh_array(mesh, "dcEdge")).astype(
            pp.dtype, copy=False
        )
    else:
        inv_dc = _source_inverse(
            inv_dc_edge,
            "inv_dc_edge",
            pp.dtype,
            (cells.shape[0],),
        )
    result = np.zeros(np.shape(cqu), dtype=pp.dtype)
    if result.shape != (pp.shape[0], cells.shape[0]) or np.shape(zxu) != result.shape:
        raise ValueError("cqu/zxu edge field shape mismatch")
    for edge, (cell0, cell1) in enumerate(cells):
        result[:, edge] = -cqu[:, edge] * (
            (pp[:, cell1] - pp[:, cell0])
            * inv_dc[edge]
            / (pp.dtype.type(0.5) * (zz[:, cell1] + zz[:, cell0]))
            - pp.dtype.type(0.5) * zxu[:, edge] * (dpdz[:, cell0] + dpdz[:, cell1])
        )
    return result


def vertical_transport_u(
    mesh: object,
    normal_velocity: FloatArray,
    rho_w: FloatArray,
    *,
    fzm: FloatArray,
    fzp: FloatArray,
    rdzw: FloatArray,
) -> FloatArray:
    """Vertical transport of normal momentum, frozen lines 4770-4790."""
    u = np.asarray(normal_velocity)
    rw = np.asarray(rho_w)
    nlev, nedges = u.shape
    ncells = _mesh_array(mesh, "areaCell").size
    if nlev < 2 or rw.shape != (nlev + 1, ncells):
        raise ValueError("vertical transport requires >=2 levels and matching rho_w")
    cells = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    result = np.zeros_like(u)
    for edge in range(nedges):
        cell0, cell1 = cells[edge]
        wduz = np.zeros(nlev + 1, dtype=u.dtype)
        level = 1
        wduz[level] = (
            u.dtype.type(0.5)
            * (rw[level, cell0] + rw[level, cell1])
            * (fzm[level] * u[level, edge] + fzp[level] * u[level - 1, edge])
        )
        for level in range(2, nlev - 1):
            wduz[level] = flux3(
                u[level - 2, edge],
                u[level - 1, edge],
                u[level, edge],
                u[level + 1, edge],
                u.dtype.type(0.5) * (rw[level, cell0] + rw[level, cell1]),
                1.0,
            )
        level = nlev - 1
        wduz[level] = (
            u.dtype.type(0.5)
            * (rw[level, cell0] + rw[level, cell1])
            * (fzm[level] * u[level, edge] + fzp[level] * u[level - 1, edge])
        )
        result[:, edge] = -np.asarray(rdzw) * np.diff(wduz)
    return result


def vector_invariant_momentum_tendency(
    mesh: object,
    *,
    normal_velocity: FloatArray,
    rho_edge: FloatArray,
    pv_edge: FloatArray,
    kinetic_energy: FloatArray,
    horizontal_divergence: FloatArray,
) -> FloatArray:
    """Nonlinear PV/KE momentum terms at frozen lines 4792-4820."""
    u = np.asarray(normal_velocity)
    if np.shape(rho_edge) != u.shape or np.shape(pv_edge) != u.shape:
        raise ValueError("rho_edge and pv_edge must match velocity")
    cells = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    edges_on_edge = _mesh_array(mesh, "edgesOnEdge").astype(np.int64, copy=False)
    counts = _mesh_array(mesh, "nEdgesOnEdge").astype(np.int64, copy=False)
    weights = _mesh_array(mesh, "weightsOnEdge").astype(u.dtype, copy=False)
    inv_dc = np.reciprocal(_mesh_array(mesh, "dcEdge")).astype(u.dtype, copy=False)
    result = np.zeros_like(u)
    for edge, (cell0, cell1) in enumerate(cells):
        q = np.zeros(u.shape[0], dtype=u.dtype)
        for slot in range(int(counts[edge])):
            neighbor_edge = int(edges_on_edge[edge, slot])
            work_pv = u.dtype.type(0.5) * (pv_edge[:, edge] + pv_edge[:, neighbor_edge])
            q += weights[edge, slot] * u[:, neighbor_edge] * work_pv
        result[:, edge] = rho_edge[:, edge] * (
            q - (kinetic_energy[:, cell1] - kinetic_energy[:, cell0]) * inv_dc[edge]
        ) - u[:, edge] * u.dtype.type(0.5) * (
            horizontal_divergence[:, cell0] + horizontal_divergence[:, cell1]
        )
    return result


def rayleigh_damp_horizontal_momentum(
    tendency: FloatArray,
    rho_edge: FloatArray,
    normal_velocity: FloatArray,
    *,
    levels: int,
    timescale_days: float,
) -> FloatArray:
    """Apply the linearly ramped upper-level damping at lines 4994-5012."""
    out = np.asarray(tendency).copy()
    nlev = out.shape[0]
    if levels == 0:
        return out
    if levels < 0 or levels > nlev:
        raise ConfigurationRefusal(
            "config_number_rayleigh_damp_u_levels",
            levels,
            f"the damping layer must be between 0 and nVertLevels={nlev}",
            "config_number_rayleigh_damp_u_levels=0",
        )
    if timescale_days <= 0.0:
        raise ConfigurationRefusal(
            "config_rayleigh_damp_u_timescale_days",
            timescale_days,
            "enabled Rayleigh damping requires a positive timescale",
            "config_rayleigh_damp_u_timescale_days>0",
        )
    inverse = 1.0 / (levels * timescale_days * 86_400.0)
    start = nlev - levels
    for level in range(start, nlev):
        coefficient = (level - start + 1) * inverse
        out[level] -= rho_edge[level] * normal_velocity[level] * coefficient
    return out


@dataclass(frozen=True, slots=True)
class MomentumMixing:
    tendency: FloatArray
    laplacian_velocity: FloatArray
    laplacian_divergence: FloatArray
    laplacian_vorticity: FloatArray


def horizontal_momentum_mixing(
    mesh: object,
    *,
    normal_velocity: FloatArray,
    rho_edge: FloatArray,
    divergence: FloatArray,
    vorticity: FloatArray,
    kdiff: FloatArray,
    second_order_scaling: FloatArray | None = None,
    fourth_order_scaling: FloatArray | None = None,
    fourth_order_viscosity: float = 0.0,
    fourth_order_divergence_factor: float = 1.0,
) -> MomentumMixing:
    """Second/fourth-order vector diffusion at frozen lines 4828-4921."""
    u = np.asarray(normal_velocity)
    nlev, nedges = u.shape
    cells = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    vertices = _mesh_array(mesh, "verticesOnEdge").astype(np.int64, copy=False)
    dc = _mesh_array(mesh, "dcEdge").astype(u.dtype, copy=False)
    dv = _mesh_array(mesh, "dvEdge").astype(u.dtype, copy=False)
    inv_dc = np.reciprocal(dc)
    inv_dv_limited = np.minimum(np.reciprocal(dv), 4.0 * inv_dc)
    if second_order_scaling is None:
        second_order_scaling = np.ones(nedges, dtype=u.dtype)
    if fourth_order_scaling is None:
        fourth_order_scaling = np.ones(nedges, dtype=u.dtype)
    lap_u = np.zeros_like(u)
    tendency = np.zeros_like(u)
    for edge, (cell0, cell1) in enumerate(cells):
        vertex0, vertex1 = vertices[edge]
        lap_u[:, edge] = (
            (divergence[:, cell1] - divergence[:, cell0]) * inv_dc[edge]
            - (vorticity[:, vertex1] - vorticity[:, vertex0]) * inv_dv_limited[edge]
        )
        k_edge = u.dtype.type(0.5) * (kdiff[:, cell0] + kdiff[:, cell1])
        tendency[:, edge] += (
            rho_edge[:, edge] * k_edge * lap_u[:, edge] * second_order_scaling[edge]
        )

    ncells = _mesh_array(mesh, "areaCell").size
    nvertices = _mesh_array(mesh, "areaTriangle").size
    lap_div = np.zeros((nlev, ncells), dtype=u.dtype)
    lap_vort = np.zeros((nlev, nvertices), dtype=u.dtype)
    if fourth_order_viscosity > 0.0:
        counts_cell = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
        edges_cell = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
        signs_cell = edge_signs_on_cells(mesh).astype(u.dtype, copy=False)
        area_cell = _mesh_array(mesh, "areaCell").astype(u.dtype, copy=False)
        edges_vertex = _mesh_array(mesh, "edgesOnVertex").astype(np.int64, copy=False)
        signs_vertex = edge_signs_on_vertices(mesh).astype(u.dtype, copy=False)
        area_vertex = _mesh_array(mesh, "areaTriangle").astype(u.dtype, copy=False)
        for vertex in range(nvertices):
            for slot, edge in enumerate(edges_vertex[vertex]):
                if edge >= 0:
                    lap_vort[:, vertex] += (
                        dc[edge] * signs_vertex[vertex, slot] * lap_u[:, edge] / area_vertex[vertex]
                    )
        for cell in range(ncells):
            for slot in range(int(counts_cell[cell])):
                edge = int(edges_cell[cell, slot])
                lap_div[:, cell] += (
                    dv[edge] * signs_cell[cell, slot] * lap_u[:, edge] / area_cell[cell]
                )
        for edge, (cell0, cell1) in enumerate(cells):
            vertex0, vertex1 = vertices[edge]
            scale = fourth_order_scaling[edge] * fourth_order_viscosity
            diffusion = rho_edge[:, edge] * (
                (lap_div[:, cell1] - lap_div[:, cell0])
                * scale
                * fourth_order_divergence_factor
                * inv_dc[edge]
                - (lap_vort[:, vertex1] - lap_vort[:, vertex0])
                * scale
                * inv_dv_limited[edge]
            )
            tendency[:, edge] -= diffusion
    elif fourth_order_viscosity < 0.0:
        raise ConfigurationRefusal(
            "config_h_mom_eddy_visc4",
            fourth_order_viscosity,
            "negative fourth-order viscosity is not an admitted MPAS branch",
            "config_h_mom_eddy_visc4>=0",
        )
    return MomentumMixing(tendency, lap_u, lap_div, lap_vort)
