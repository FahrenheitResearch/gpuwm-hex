"""MPAS-A v8.4.1 deformation-based 2-D Smagorinsky horizontal mixing.

CPU authority for the v8.4.1 mixing formulation, mirrored from the MPAS-A
v8.4.1 native reference source tree that produced the 2026-08-17 reference
control (byte-identical across the gnu and intel build copies of that tree);
all source paths below are relative to that tree:

* deformation weights: ``src/core_atmosphere/mpas_atm_core.F:1620-1850``
  (``atm_initialize_deformation_weights``), spherical branch only, with the
  called helpers ``mpas_sphere_angle`` and ``mpas_arc_length`` from
  ``src/operators/mpas_geometry_utils.F:27-72,131-154``.  The block at core
  lines 1802-1812 (``mpas_plane_angle`` accumulation) is dead code -- every
  ``thetat`` entry it writes is overwritten by the ``atan2`` assignment inside
  the area loop at lines 1814-1822 before any use -- so it is not mirrored;
  this has no floating-point effect.
* eddy viscosity: ``src/core_atmosphere/dynamics/mpas_atm_dissipation_models.F
  :119-204`` (``smagorinsky_2d``), called from
  ``mpas_atm_time_integration.F:6346-6352`` on RK step 1 of every dynamics
  substep with the edge normal velocity ``u`` and the reconstructed edge
  tangential velocity ``v``.
* the u/w/theta applications (``u_dissipation_3d`` at dissipation-models
  lines 577-945, ``w_dissipation_3d`` at 949-1151, and the theta branch of
  ``scalar_dissipation_3d_les`` at 1155-1330) are, for the non-LES
  ``les_model_opt == LES_MODEL_NONE`` / ``v_*_eddy_visc2 == 0`` /
  ``config_mix_scalars = false`` regime this lane runs, term-for-term the
  stencils already mirrored by :mod:`hexcore.mixing` for v8.2.3, so those
  authorities are reused here.  Two verified-inert differences: (a) the v8.4.1
  ``u_diffusion_les`` extra divergence-gradient term carries
  ``tau_12_factor = 0`` outside LES (dissipation-models line 684-685,711), an
  exact multiply-by-zero add of zero; (b) the theta application multiplies by
  ``prandtl_inv`` (lines 1280,1310) where ``prandtl = 1.0_RKIND``
  (``src/framework/mpas_constants.F:56``), an exact multiply by one.  Neither
  can change any binary32 bit.

The reference native build runs single precision (``RKIND =
selected_real_kind(6)``; both 24-h reference logs print ``Default real
precision: single``), so the execution dtype of this authority is float32 with
identical operation order; the float64 mirror of the same code path is the
pinning scaffold, exactly as elsewhere in the port.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import ConfigurationRefusal
from .mixing import (
    DryMixingTendencies,
    _finite_scalar,
    _float_field,
    _mesh_array,
    _same_dtype,
    compute_mesh_mixing_scaling,
    momentum_horizontal_filter_tendency,
    resolve_config_len_disp,
    theta_horizontal_filter_tendency,
    vertical_momentum_horizontal_filter_tendency,
)

FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True, slots=True)
class V841MixingConfig:
    """Exact works-or-refuses contract for the v8.4.1 Smagorinsky branch.

    Defaults are the native Registry values the natB-24h reference
    integrated, verbatim from that run's ``namelist.atmosphere``.
    """

    config_horiz_mixing: str = "2d_smagorinsky"
    config_len_disp: float = 0.0
    config_visc4_2dsmag: float = 0.05
    config_smagorinsky_coef: float = 0.125
    config_del4u_div_factor: float = 10.0
    config_h_ScaleWithMesh: bool = True
    config_mpas_cam_coef: float = 0.0

    def validate(self) -> None:
        if self.config_horiz_mixing != "2d_smagorinsky":
            raise ConfigurationRefusal(
                "config_horiz_mixing",
                self.config_horiz_mixing,
                "this authority admits the v8.4.1 2-D Smagorinsky branch",
                "config_horiz_mixing='2d_smagorinsky'",
            )
        length = _finite_scalar("config_len_disp", self.config_len_disp)
        if length < 0.0:
            raise ConfigurationRefusal(
                "config_len_disp",
                self.config_len_disp,
                "a negative filter length is not an MPAS-admitted value",
                "config_len_disp>=0 (zero selects nominalMinDc)",
            )
        visc4 = _finite_scalar("config_visc4_2dsmag", self.config_visc4_2dsmag)
        if visc4 < 0.0:
            raise ConfigurationRefusal(
                "config_visc4_2dsmag",
                self.config_visc4_2dsmag,
                "the Registry admits only non-negative fourth-order scaling",
                "config_visc4_2dsmag>=0",
            )
        smag = _finite_scalar(
            "config_smagorinsky_coef", self.config_smagorinsky_coef
        )
        if smag < 0.0:
            raise ConfigurationRefusal(
                "config_smagorinsky_coef",
                self.config_smagorinsky_coef,
                "the admitted Smagorinsky coefficient is non-negative",
                "config_smagorinsky_coef>=0",
            )
        div_factor = _finite_scalar(
            "config_del4u_div_factor", self.config_del4u_div_factor
        )
        if div_factor <= 0.0:
            raise ConfigurationRefusal(
                "config_del4u_div_factor",
                self.config_del4u_div_factor,
                "the Registry requires a positive divergent hyperdiffusion factor",
                "config_del4u_div_factor>0",
            )
        cam = _finite_scalar("config_mpas_cam_coef", self.config_mpas_cam_coef)
        if cam != 0.0:
            raise ConfigurationRefusal(
                "config_mpas_cam_coef",
                self.config_mpas_cam_coef,
                "the CAM-SE upper-level coefficient floor is not ported",
                "config_mpas_cam_coef=0.0",
            )
        if not isinstance(self.config_h_ScaleWithMesh, (bool, np.bool_)):
            raise ConfigurationRefusal(
                "config_h_ScaleWithMesh",
                self.config_h_ScaleWithMesh,
                "the Registry option is logical",
                "config_h_ScaleWithMesh=True or False",
            )


@dataclass(frozen=True, slots=True)
class DeformationWeightsV841:
    """``deformation_coef_{c2,s2,cs}`` with shape ``(nCells, maxEdges)``."""

    coef_c2: FloatArray
    coef_s2: FloatArray
    coef_cs: FloatArray

    def validate(self, *, n_cells: int, max_edges: int) -> None:
        shape = (n_cells, max_edges)
        for name in ("coef_c2", "coef_s2", "coef_cs"):
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} != {shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")


def _sphere_arc_length(
    ax: Any, ay: Any, az: Any, bx: Any, by: Any, bz: Any, one: Any
) -> Any:
    """``mpas_arc_length`` (mpas_geometry_utils.F:131-154)."""

    cx = bx - ax
    cy = by - ay
    cz = bz - az
    r = np.sqrt(ax * ax + ay * ay + az * az)
    c = np.sqrt(cx * cx + cy * cy + cz * cz)
    two = one + one
    return r * two * np.arcsin(c / (two * r))


def _sphere_angle(
    ax: Any, ay: Any, az: Any,
    bx: Any, by: Any, bz: Any,
    cx: Any, cy: Any, cz: Any,
    one: Any,
) -> Any:
    """``mpas_sphere_angle`` (mpas_geometry_utils.F:27-72)."""

    zero = one - one
    half = one / (one + one)
    two = one + one
    a = _sphere_arc_length(bx, by, bz, cx, cy, cz, one)
    b = _sphere_arc_length(ax, ay, az, cx, cy, cz, one)
    c = _sphere_arc_length(ax, ay, az, bx, by, bz, one)
    ab_x = bx - ax
    ab_y = by - ay
    ab_z = bz - az
    ac_x = cx - ax
    ac_y = cy - ay
    ac_z = cz - az
    d_x = (ab_y * ac_z) - (ab_z * ac_y)
    d_y = -((ab_x * ac_z) - (ab_z * ac_x))
    d_z = (ab_x * ac_y) - (ab_y * ac_x)
    s = half * (a + b + c)
    ratio = (np.sin(s - b) * np.sin(s - c)) / (np.sin(b) * np.sin(c))
    sin_angle = np.sqrt(np.minimum(one, np.maximum(zero, ratio)))
    magnitude = two * np.arcsin(np.maximum(np.minimum(sin_angle, one), -one))
    if (d_x * ax + d_y * ay + d_z * az) >= zero:
        return magnitude
    return -magnitude


def initialize_deformation_weights_v841(
    mesh: object,
    *,
    dtype: np.dtype[Any] | type[Any] = np.float32,
) -> DeformationWeightsV841:
    """Mirror ``atm_initialize_deformation_weights`` (mpas_atm_core.F:1620-1850).

    Spherical branch only (``on_a_sphere`` true); planar meshes are refused
    fail-closed.  Executed in the requested dtype throughout: float32 is the
    execution mirror of the single-precision native reference build, float64
    is the pinning scaffold.
    """

    out_dtype = np.dtype(dtype)
    if out_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("deformation weights dtype must be float32 or float64")

    attrs = getattr(mesh, "attrs", {})
    on_sphere = str(attrs.get("on_a_sphere", "NO")).strip().upper() == "YES"
    if not on_sphere:
        raise ConfigurationRefusal(
            "on_a_sphere",
            attrs.get("on_a_sphere"),
            "only the spherical deformation-weight branch is ported",
            "a spherical MPAS mesh",
        )
    radius = out_dtype.type(float(attrs["sphere_radius"]))
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("sphere_radius must be finite and positive")

    counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
    edges_on_cell = np.asarray(_mesh_array(mesh, "edgesOnCell"), dtype=np.int64)
    cells_on_cell = np.asarray(_mesh_array(mesh, "cellsOnCell"), dtype=np.int64)
    vertices_on_cell = np.asarray(
        _mesh_array(mesh, "verticesOnCell"), dtype=np.int64
    )
    cells_on_edge = np.asarray(_mesh_array(mesh, "cellsOnEdge"), dtype=np.int64)
    x_cell = np.asarray(_mesh_array(mesh, "xCell"), dtype=out_dtype)
    y_cell = np.asarray(_mesh_array(mesh, "yCell"), dtype=out_dtype)
    z_cell = np.asarray(_mesh_array(mesh, "zCell"), dtype=out_dtype)
    x_vertex = np.asarray(_mesh_array(mesh, "xVertex"), dtype=out_dtype)
    y_vertex = np.asarray(_mesh_array(mesh, "yVertex"), dtype=out_dtype)
    z_vertex = np.asarray(_mesh_array(mesh, "zVertex"), dtype=out_dtype)

    n_cells = int(counts.size)
    max_edges = int(edges_on_cell.shape[1])
    if edges_on_cell.shape[0] != n_cells or vertices_on_cell.shape != edges_on_cell.shape:
        raise ValueError("edgesOnCell/verticesOnCell shapes are inconsistent")
    if cells_on_cell.shape != edges_on_cell.shape:
        raise ValueError("cellsOnCell shape is inconsistent with edgesOnCell")

    one = out_dtype.type(1.0)
    zero = out_dtype.type(0.0)
    quarter = out_dtype.type(0.25)
    half = one / (one + one)
    # pii = 2.*asin(1.0), evaluated in the working precision (core line 1698).
    pii = (one + one) * np.arcsin(one)

    coef_c2 = np.zeros((n_cells, max_edges), dtype=out_dtype)
    coef_s2 = np.zeros((n_cells, max_edges), dtype=out_dtype)
    coef_cs = np.zeros((n_cells, max_edges), dtype=out_dtype)

    for cell in range(n_cells):
        count = int(counts[cell])
        if count < 3 or count > max_edges:
            raise ValueError(f"cell {cell} has invalid nEdgesOnCell {count}")
        # Halo guard (core lines 1702-1716): native builds cell_list from
        # the cell and its neighbours and CYCLES -- leaving this cell's
        # weights at their zero initialization -- when any entry reaches
        # outside nCells.  On the serial global mesh that can only mean a
        # corrupt table; on a regional cull it is the ring-7 rows, whose
        # absent-neighbour slots map to the garbage cell, so those cells
        # keep zero weights exactly as the reference executable leaves them
        # (their specified-zone tendencies are overwritten regardless).
        neighbors = cells_on_cell[cell, :count]
        if np.any((neighbors < 0) | (neighbors >= n_cells)):
            continue
        verts = vertices_on_cell[cell, :count]
        if np.any((verts < 0) | (verts >= x_vertex.size)):
            raise ValueError(f"verticesOnCell reaches outside the mesh at cell {cell}")
        slot_edges = edges_on_cell[cell, :count]
        if np.any((slot_edges < 0) | (slot_edges >= cells_on_edge.shape[0])):
            raise ValueError(f"edgesOnCell reaches outside the mesh at cell {cell}")

        # Normalized Cartesian points (core lines 1725-1734).
        cx = x_cell[cell] / radius
        cy = y_cell[cell] / radius
        cz = z_cell[cell] / radius
        vx = x_vertex[verts] / radius
        vy = y_vertex[verts] / radius
        vz = z_vertex[verts] / radius

        # theta_abs (core lines 1742-1750).
        if cz == one:
            theta_abs = pii / (one + one)
        else:
            theta_abs = pii / (one + one) - _sphere_angle(
                cx, cy, cz, vx[0], vy[0], vz[0], zero, zero, one, one
            )

        # thetav / dl_sphere / thetat accumulation (core lines 1760-1772).
        thetat = np.zeros(count, dtype=out_dtype)
        dl_sphere = np.zeros(count, dtype=out_dtype)
        thetav = np.zeros(count, dtype=out_dtype)
        for j in range(count):
            jp1 = (j + 1) % count
            thetav[j] = _sphere_angle(
                cx, cy, cz,
                vx[j], vy[j], vz[j],
                vx[jp1], vy[jp1], vz[jp1],
                one,
            )
            dl_sphere[j] = radius * _sphere_arc_length(
                cx, cy, cz, vx[j], vy[j], vz[j], one
            )
        thetat[0] = theta_abs
        for j in range(1, count):
            thetat[j] = thetat[j - 1] + thetav[j - 1]

        # Tangent-plane vertices (core lines 1776-1779).
        xp = np.cos(thetat) * dl_sphere
        yp = np.sin(thetat) * dl_sphere

        # Cell area and edge-normal angles (core lines 1814-1822).  The
        # preceding mpas_plane_angle block (1802-1812) is dead code -- see the
        # module docstring.
        area_cell = zero
        theta_edge = np.zeros(count, dtype=out_dtype)
        for j in range(count):
            jp1 = (j + 1) % count
            dx = xp[jp1] - xp[j]
            dy = yp[jp1] - yp[j]
            area_cell = (
                area_cell
                + quarter * (xp[j] + xp[jp1]) * (yp[jp1] - yp[j])
                - quarter * (yp[j] + yp[jp1]) * (xp[jp1] - xp[j])
            )
            theta_edge[j] = np.arctan2(dy, dx) - pii / (one + one)

        # Coefficients (core lines 1826-1846).
        for j in range(count):
            jp1 = (j + 1) % count
            dx = xp[jp1] - xp[j]
            dy = yp[jp1] - yp[j]
            dl = np.sqrt(dx * dx + dy * dy)
            sin_t = np.sin(theta_edge[j])
            cos_t = np.cos(theta_edge[j])
            sint2 = sin_t * sin_t
            cost2 = cos_t * cos_t
            sint_cost = sin_t * cos_t
            c2 = dl * cost2 / area_cell
            s2 = dl * sint2 / area_cell
            cs = dl * sint_cost / area_cell
            if int(cells_on_edge[slot_edges[j], 0]) != cell:
                c2 = -c2
                s2 = -s2
                cs = -cs
            coef_c2[cell, j] = c2
            coef_s2[cell, j] = s2
            coef_cs[cell, j] = cs

    _ = half  # parity with the Fortran locals; no further use
    result = DeformationWeightsV841(
        coef_c2=coef_c2, coef_s2=coef_s2, coef_cs=coef_cs
    )
    result.validate(n_cells=n_cells, max_edges=max_edges)
    return result


@dataclass(frozen=True, slots=True)
class SmagorinskyCoefficientsV841:
    kdiff: FloatArray
    h_mom_eddy_visc4: np.floating[Any]
    h_theta_eddy_visc4: np.floating[Any]
    config_len_disp: np.floating[Any]


def compute_smagorinsky_coefficients_v841(
    mesh: object,
    normal_velocity: object,
    tangential_velocity: object,
    weights: DeformationWeightsV841,
    *,
    dt: float,
    config: V841MixingConfig | None = None,
) -> SmagorinskyCoefficientsV841:
    """Mirror ``smagorinsky_2d`` (mpas_atm_dissipation_models.F:119-204).

    ``u`` is the edge normal velocity and ``v`` the reconstructed edge
    tangential velocity, as at the native call site
    (mpas_atm_time_integration.F:6348-6352).
    """

    cfg = V841MixingConfig() if config is None else config
    cfg.validate()
    u = _float_field("normal_velocity", normal_velocity, ndim=2)
    v = _same_dtype(u, "tangential_velocity", tangential_velocity, ndim=2)
    if v.shape != u.shape:
        raise ValueError("normal_velocity and tangential_velocity shapes differ")
    timestep = _finite_scalar("config_dt", dt)
    if timestep <= 0.0:
        raise ConfigurationRefusal(
            "config_dt",
            dt,
            "the Smagorinsky stability ceiling requires a positive timestep",
            "config_dt>0",
        )

    edges = np.asarray(_mesh_array(mesh, "edgesOnCell"), dtype=np.int64)
    counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
    dtype = u.dtype
    coef_c2 = np.asarray(weights.coef_c2, dtype=dtype)
    coef_s2 = np.asarray(weights.coef_s2, dtype=dtype)
    coef_cs = np.asarray(weights.coef_cs, dtype=dtype)
    if edges.ndim != 2 or counts.shape != (edges.shape[0],):
        raise ValueError("edgesOnCell/nEdgesOnCell shapes are inconsistent")
    if coef_c2.shape != edges.shape:
        raise ValueError("deformation weights must have shape edgesOnCell")
    used = np.arange(edges.shape[1])[None, :] < counts[:, None]
    if np.any((edges[used] < 0) | (edges[used] >= u.shape[1])):
        raise ValueError("edgesOnCell contains an invalid used edge")

    length = resolve_config_len_disp(mesh, cfg.config_len_disp, dtype=dtype)
    cs_coef = dtype.type(cfg.config_smagorinsky_coef)
    # (c_s * config_len_disp)**2  (dissipation-models line 189)
    strain_scale = (cs_coef * length) * (cs_coef * length)
    # invDt = 1.0/dt; ceiling = (0.01*config_len_disp**2) * invDt
    # (time-integration line 6323; dissipation-models line 190)
    inv_dt = dtype.type(1.0) / dtype.type(timestep)
    ceiling = (dtype.type(0.01) * (length * length)) * inv_dt

    nlev = u.shape[0]
    n_cells = edges.shape[0]
    kdiff = np.zeros((nlev, n_cells), dtype=dtype)
    two = dtype.type(2.0)
    quarter = dtype.type(0.25)
    for cell in range(n_cells):
        dudx = np.zeros(nlev, dtype=dtype)
        dudy = np.zeros(nlev, dtype=dtype)
        dvdx = np.zeros(nlev, dtype=dtype)
        dvdy = np.zeros(nlev, dtype=dtype)
        for slot in range(int(counts[cell])):
            edge = int(edges[cell, slot])
            c2 = coef_c2[cell, slot]
            s2 = coef_s2[cell, slot]
            ccs = coef_cs[cell, slot]
            dudx += c2 * u[:, edge] - ccs * v[:, edge]
            dudy += ccs * u[:, edge] - s2 * v[:, edge]
            dvdx += ccs * u[:, edge] + c2 * v[:, edge]
            dvdy += s2 * u[:, edge] + ccs * v[:, edge]
        d_11 = two * dudx
        d_22 = two * dvdy
        d_12 = dudy + dvdx
        diff = d_11 - d_22
        strain = np.sqrt(quarter * (diff * diff) + d_12 * d_12)
        kdiff[:, cell] = np.minimum(strain_scale * strain, ceiling)

    visc4 = dtype.type(cfg.config_visc4_2dsmag)
    # h_mom_eddy_visc4 = config_visc4_2dsmag * config_len_disp**3;
    # h_theta_eddy_visc4 = h_mom_eddy_visc4 (dissipation-models lines 199-200)
    h4 = visc4 * ((length * length) * length)
    return SmagorinskyCoefficientsV841(
        kdiff=kdiff,
        h_mom_eddy_visc4=h4,
        h_theta_eddy_visc4=h4,
        config_len_disp=length,
    )


def compute_dry_mixing_tendencies_v841(
    mesh: object,
    weights: DeformationWeightsV841,
    *,
    normal_velocity: object,
    tangential_velocity: object,
    vertical_velocity: object,
    theta_m: object,
    rho_edge: object,
    divergence: object,
    vorticity: object,
    dt: float,
    config: V841MixingConfig | None = None,
) -> DryMixingTendencies:
    """v8.4.1 RK-stage-one horizontal mixing increments (saved for RK2/RK3).

    kdiff comes from the v8.4.1 ``smagorinsky_2d`` mirror; the u/w/theta
    applications reuse the :mod:`hexcore.mixing` authorities, which are the
    same non-LES stencils as ``u_dissipation_3d`` / ``w_dissipation_3d`` /
    ``scalar_dissipation_3d_les`` (see the module docstring for the two
    exact-inert differences).
    """

    cfg = V841MixingConfig() if config is None else config
    cfg.validate()
    coefficients = compute_smagorinsky_coefficients_v841(
        mesh,
        normal_velocity,
        tangential_velocity,
        weights,
        dt=dt,
        config=cfg,
    )
    dtype = coefficients.kdiff.dtype
    scaling = compute_mesh_mixing_scaling(
        mesh,
        config_h_ScaleWithMesh=cfg.config_h_ScaleWithMesh,
        dtype=dtype,
    )
    # Regional culls carry stored-0 (negative-sentinel) cellsOnEdge slots on
    # ring-7 rows.  Native runs these filters over the explicit garbage
    # elements -- delsq scratch garbage columns are zeroed by atm_srk3 and
    # theta_m's garbage cell is zeroed by the rk setup -- so the filters run
    # here on the same padded memory model and the pads are stripped after.
    # Ring-6/7 filter lanes are dead regardless: the specified-zone tendency
    # overwrite replaces them before anything reads them.
    filter_mesh: object = mesh
    kdiff_arg: object = coefficients.kdiff
    div_arg: object = divergence
    vort_arg: object = vorticity
    theta_arg: object = theta_m
    w_arg: object = vertical_velocity
    raw_coe = np.asarray(_mesh_array(mesh, "cellsOnEdge"), dtype=np.int64)
    regional = bool(np.any(raw_coe < 0))
    n_cells = int(np.asarray(_mesh_array(mesh, "areaCell")).size)
    if regional:
        def _pad(value: object, fill: float = 0.0) -> FloatArray:
            data = np.asarray(value)
            pad = np.full(data.shape[:-1] + (1,), fill, dtype=data.dtype)
            return np.concatenate([data, pad], axis=-1)

        arrays: dict[str, np.ndarray] = {
            "cellsOnEdge": np.where(raw_coe < 0, n_cells, raw_coe),
        }
        for name in (
            "verticesOnEdge",
            "edgesOnVertex",
            "dcEdge",
            "dvEdge",
            "areaTriangle",
            "meshDensity",
            "nominalMinDc",
        ):
            try:
                arrays[name] = np.asarray(_mesh_array(mesh, name))
            except AttributeError:
                pass
        # The garbage cell: no edges, unit area (native never divides by its
        # area -- its loops exclude it -- so the pad only has to be inert).
        eoc = np.asarray(_mesh_array(mesh, "edgesOnCell"), dtype=np.int64)
        arrays["edgesOnCell"] = np.concatenate(
            [eoc, np.zeros((1, eoc.shape[1]), dtype=np.int64)], axis=0
        )
        counts_real = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
        arrays["nEdgesOnCell"] = np.concatenate(
            [counts_real, np.zeros(1, dtype=np.int64)]
        )
        area_real = np.asarray(_mesh_array(mesh, "areaCell"))
        arrays["areaCell"] = np.concatenate(
            [area_real, np.ones(1, dtype=area_real.dtype)]
        )

        class _RegionalFilterMesh:
            def __init__(self, table: dict[str, np.ndarray]) -> None:
                self.arrays = table

        filter_mesh = _RegionalFilterMesh(arrays)
        kdiff_arg = _pad(coefficients.kdiff)
        div_arg = _pad(divergence)
        vort_arg = vorticity
        theta_arg = _pad(theta_m)
        w_arg = _pad(vertical_velocity)
    momentum = momentum_horizontal_filter_tendency(
        filter_mesh,
        rho_edge=rho_edge,
        divergence=div_arg,
        vorticity=vort_arg,
        kdiff=kdiff_arg,
        h_mom_eddy_visc4=coefficients.h_mom_eddy_visc4,
        config_del4u_div_factor=cfg.config_del4u_div_factor,
        mesh_scaling_del2=scaling.del2,
        mesh_scaling_del4=scaling.del4,
    )
    w_filter = vertical_momentum_horizontal_filter_tendency(
        filter_mesh,
        vertical_velocity=w_arg,
        rho_edge=rho_edge,
        kdiff=kdiff_arg,
        h_mom_eddy_visc4=coefficients.h_mom_eddy_visc4,
        mesh_scaling_del2=scaling.del2,
        mesh_scaling_del4=scaling.del4,
    )
    theta_filter = theta_horizontal_filter_tendency(
        filter_mesh,
        theta_m=theta_arg,
        rho_edge=rho_edge,
        kdiff=kdiff_arg,
        h_theta_eddy_visc4=coefficients.h_theta_eddy_visc4,
        mesh_scaling_del2=scaling.del2,
        mesh_scaling_del4=scaling.del4,
    )

    def _strip(value: FloatArray, count: int) -> FloatArray:
        return value[:, :count] if regional else value

    return DryMixingTendencies(
        kdiff=coefficients.kdiff,
        h_mom_eddy_visc4=coefficients.h_mom_eddy_visc4,
        h_theta_eddy_visc4=coefficients.h_theta_eddy_visc4,
        tend_u_euler=momentum.tendency,
        tend_w_euler=_strip(w_filter.tendency, n_cells),
        tend_theta_euler=_strip(theta_filter.tendency, n_cells),
        delsq_u=momentum.delsq_u,
        delsq_divergence=_strip(momentum.delsq_divergence, n_cells),
        delsq_vorticity=momentum.delsq_vorticity,
        delsq_w=_strip(w_filter.laplacian, n_cells),
        delsq_theta=_strip(theta_filter.laplacian, n_cells),
    )


__all__ = [
    "DeformationWeightsV841",
    "SmagorinskyCoefficientsV841",
    "V841MixingConfig",
    "compute_dry_mixing_tendencies_v841",
    "compute_smagorinsky_coefficients_v841",
    "initialize_deformation_weights_v841",
]
