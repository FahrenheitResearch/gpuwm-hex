"""Release-specific CUDA horizontal operators for MPAS-A v8.4.1."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .cuda_fp32 import CUDA_FTZ_HELPERS
from .cuda_horizontal import (
    CudaDryMixingTendencies,
    CudaHorizontal,
    CudaSolveDiagnostics,
)
from .cuda_v841 import CudaV841Context


_CUDA_SOURCE = CUDA_FTZ_HELPERS + r"""
__device__ __forceinline__ int lidx_v841(int level, int entity, int count) {
    return level * count + entity;
}
__device__ __forceinline__ float cell_sign_v841(
    int cell, int edge, const int *cells_on_edge)
{
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}
__device__ __forceinline__ float vertex_sign_v841(
    int vertex, int edge, const int *vertices_on_edge)
{
    return vertices_on_edge[2 * edge] == vertex ? -1.0f : 1.0f;
}
__device__ __forceinline__ int kite_slot_v841(
    int vertex, int cell, const int *cells_on_vertex, int vertex_degree)
{
    for (int slot = 0; slot < vertex_degree; ++slot) {
        if (cells_on_vertex[vertex * vertex_degree + slot] == cell) return slot;
    }
    return -1;
}

extern "C" __global__ void vertex_diagnostics_v841_f32(
    const float *normal_velocity, const float *ke_edge,
    const int *edges_on_vertex, const int *vertices_on_edge,
    const float *dc_edge, const float *inv_area_triangle,
    const float *f_vertex, const int nlev, const int nedges,
    const int nvertices, const int vertex_degree,
    float *vorticity, float *ke_vertex, float *pv_vertex)
{
    const int vertex = blockDim.x * blockIdx.x + threadIdx.x;
    if (vertex >= nvertices) return;
    for (int k = 0; k < nlev; ++k) {
        float vort = 0.0f;
        float ke = 0.0f;
        for (int slot = 0; slot < vertex_degree; ++slot) {
            const int edge = edges_on_vertex[vertex * vertex_degree + slot];
            if (edge < 0) continue;
            vort = mpas_add(vort, mpas_mul(mpas_mul(
                vertex_sign_v841(vertex, edge, vertices_on_edge), dc_edge[edge]),
                normal_velocity[lidx_v841(k, edge, nedges)]));
            ke = mpas_add(ke, ke_edge[lidx_v841(k, edge, nedges)]);
        }
        vort = mpas_mul(vort, inv_area_triangle[vertex]);
        const float r = mpas_mul(0.25f, inv_area_triangle[vertex]);
        ke = mpas_mul(ke, r);
        const int index = lidx_v841(k, vertex, nvertices);
        vorticity[index] = vort;
        ke_vertex[index] = ke;
        pv_vertex[index] = mpas_add(f_vertex[vertex], vort);
    }
}

extern "C" __global__ void cell_diagnostics_v841_f32(
    const float *normal_velocity, const float *ke_edge,
    const float *ke_vertex, const int *edges_on_cell,
    const int *n_edges_on_cell, const int *cells_on_edge,
    const int *vertices_on_cell, const int *cells_on_vertex,
    const float *dv_edge, const float *inv_area_cell,
    const float *kite_areas, const int nlev, const int ncells,
    const int nedges, const int nvertices, const int max_edges,
    const int vertex_degree, float *divergence, float *kinetic_energy)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const int count = n_edges_on_cell[cell];
    const float inv_area = inv_area_cell[cell];
    const float ke_fact = 0.625f;
    for (int k = 0; k < nlev; ++k) {
        float div = 0.0f;
        float ke = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            div = mpas_add(div, mpas_mul(mpas_mul(
                cell_sign_v841(cell, edge, cells_on_edge), dv_edge[edge]),
                normal_velocity[lidx_v841(k, edge, nedges)]));
            ke = mpas_add(ke, mpas_mul(
                0.25f, ke_edge[lidx_v841(k, edge, nedges)]));
        }
        div = mpas_mul(div, inv_area);
        ke = mpas_mul(ke, inv_area);
        ke = mpas_mul(ke, ke_fact);
        for (int slot = 0; slot < count; ++slot) {
            const int vertex = vertices_on_cell[cell * max_edges + slot];
            const int kite = kite_slot_v841(
                vertex, cell, cells_on_vertex, vertex_degree);
            if (kite >= 0) {
                float correction = mpas_mul(mpas_sub(1.0f, ke_fact),
                    kite_areas[vertex * vertex_degree + kite]);
                correction = mpas_mul(correction,
                    ke_vertex[lidx_v841(k, vertex, nvertices)]);
                correction = mpas_mul(correction, inv_area);
                ke = mpas_add(ke, correction);
            }
        }
        const int index = lidx_v841(k, cell, ncells);
        divergence[index] = div;
        kinetic_energy[index] = ke;
    }
}

extern "C" __global__ void smagorinsky_v841_f32(
    const float *normal_velocity, const float *tangential_velocity,
    const int *edges_on_cell, const int *n_edges_on_cell,
    const float *coef_c2, const float *coef_s2, const float *coef_cs,
    const float strain_scale, const float ceiling,
    const int nlev, const int ncells, const int nedges, const int max_edges,
    float *kdiff)
{
    // mpas_atm_dissipation_models.F:119-204 (smagorinsky_2d): full
    // velocity-gradient tensor from the precomputed v8.4.1 deformation
    // weights, D11 = 2 du/dx, D22 = 2 dv/dy, D12 = du/dy + dv/dx,
    // kdiff = (c_s*len)**2 * sqrt(0.25*(D11-D22)**2 + D12**2) capped at
    // (0.01*len**2)*invDt.
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const int count = n_edges_on_cell[cell];
    for (int k = 0; k < nlev; ++k) {
        float dudx = 0.0f;
        float dudy = 0.0f;
        float dvdx = 0.0f;
        float dvdy = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int offset = cell * max_edges + slot;
            const int edge = edges_on_cell[offset];
            const float u = normal_velocity[lidx_v841(k, edge, nedges)];
            const float v = tangential_velocity[lidx_v841(k, edge, nedges)];
            const float c2 = coef_c2[offset];
            const float s2 = coef_s2[offset];
            const float cs = coef_cs[offset];
            dudx = mpas_add(dudx,
                mpas_sub(mpas_mul(c2, u), mpas_mul(cs, v)));
            dudy = mpas_add(dudy,
                mpas_sub(mpas_mul(cs, u), mpas_mul(s2, v)));
            dvdx = mpas_add(dvdx,
                mpas_add(mpas_mul(cs, u), mpas_mul(c2, v)));
            dvdy = mpas_add(dvdy,
                mpas_add(mpas_mul(s2, u), mpas_mul(cs, v)));
        }
        const float d_11 = mpas_mul(2.0f, dudx);
        const float d_22 = mpas_mul(2.0f, dvdy);
        const float d_12 = mpas_add(dudy, dvdx);
        const float diff = mpas_sub(d_11, d_22);
        const float strain = mpas_sqrt(mpas_add(
            mpas_mul(0.25f, mpas_mul(diff, diff)),
            mpas_mul(d_12, d_12)));
        kdiff[lidx_v841(k, cell, ncells)] = mpas_min(
            mpas_mul(strain_scale, strain), ceiling);
    }
}

extern "C" __global__ void pv_cell_v841_f32(
    const float *pv_vertex, const int *vertices_on_cell,
    const int *n_edges_on_cell, const int *cells_on_vertex,
    const float *kite_areas, const float *inv_area_cell,
    const int nlev, const int ncells, const int nvertices,
    const int max_edges, const int vertex_degree, float *pv_cell)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const int count = n_edges_on_cell[cell];
    for (int k = 0; k < nlev; ++k) {
        float value = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int vertex = vertices_on_cell[cell * max_edges + slot];
            const int kite = kite_slot_v841(
                vertex, cell, cells_on_vertex, vertex_degree);
            if (kite >= 0) {
                float contribution = mpas_mul(
                    kite_areas[vertex * vertex_degree + kite],
                    pv_vertex[lidx_v841(k, vertex, nvertices)]);
                contribution = mpas_mul(contribution, inv_area_cell[cell]);
                value = mpas_add(value, contribution);
            }
        }
        pv_cell[lidx_v841(k, cell, ncells)] = value;
    }
}

extern "C" __global__ void pv_apvm_v841_f32(
    const float *normal_velocity, const float *tangential_velocity,
    const float *pv_vertex, const float *pv_cell,
    const int *vertices_on_edge, const int *cells_on_edge,
    const float *inv_dc_edge, const float *inv_dv_edge, const float scale,
    const int nlev, const int ncells, const int nedges, const int nvertices,
    float *pv_edge, float *grad_pv_normal, float *grad_pv_tangential)
{
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int vertex0 = vertices_on_edge[2 * edge];
    const int vertex1 = vertices_on_edge[2 * edge + 1];
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    for (int k = 0; k < nlev; ++k) {
        const int index = lidx_v841(k, edge, nedges);
        const float grad_t = mpas_mul(mpas_sub(
            pv_vertex[lidx_v841(k, vertex1, nvertices)],
            pv_vertex[lidx_v841(k, vertex0, nvertices)]), inv_dv_edge[edge]);
        const float grad_n = mpas_mul(mpas_sub(
            pv_cell[lidx_v841(k, cell1, ncells)],
            pv_cell[lidx_v841(k, cell0, ncells)]), inv_dc_edge[edge]);
        grad_pv_tangential[index] = grad_t;
        grad_pv_normal[index] = grad_n;
        pv_edge[index] = mpas_sub(pv_edge[index], mpas_mul(scale, mpas_add(
            mpas_mul(tangential_velocity[index], grad_t),
            mpas_mul(normal_velocity[index], grad_n))));
    }
}

extern "C" __global__ void mass_flux_divergence_v841_f32(
    const float *rho_u, const int *edges_on_cell,
    const int *n_edges_on_cell, const int *cells_on_edge,
    const float *dv_edge, const float *inv_area_cell,
    const int nlev, const int ncells, const int nedges,
    const int max_edges, float *mass_divergence)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const int count = n_edges_on_cell[cell];
    for (int k = 0; k < nlev; ++k) {
        float value = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            value = mpas_add(value, mpas_mul(mpas_mul(
                cell_sign_v841(cell, edge, cells_on_edge), dv_edge[edge]),
                rho_u[lidx_v841(k, edge, nedges)]));
        }
        mass_divergence[lidx_v841(k, cell, ncells)] = mpas_mul(
            value, inv_area_cell[cell]);
    }
}

extern "C" __global__ void pressure_gradient_v841_f32(
    const float *pressure_p, const float *dpdz, const float *cqu,
    const float *zz, const float *zxu, const int *cells_on_edge,
    const float *inv_dc_edge, const int nlev, const int ncells,
    const int nedges, float *tendency)
{
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    for (int k = 0; k < nlev; ++k) {
        const int index = lidx_v841(k, edge, nedges);
        float normal = mpas_mul(mpas_sub(
            pressure_p[lidx_v841(k, cell1, ncells)],
            pressure_p[lidx_v841(k, cell0, ncells)]), inv_dc_edge[edge]);
        normal = mpas_div(normal, mpas_mul(0.5f, mpas_add(
            zz[lidx_v841(k, cell1, ncells)],
            zz[lidx_v841(k, cell0, ncells)])));
        const float terrain = mpas_mul(mpas_mul(0.5f, zxu[index]), mpas_add(
            dpdz[lidx_v841(k, cell0, ncells)],
            dpdz[lidx_v841(k, cell1, ncells)]));
        tendency[index] = mpas_mul(mpas_sub(0.0f, cqu[index]),
            mpas_sub(normal, terrain));
    }
}
"""


class CudaHorizontalV841(CudaHorizontal):
    """v8.4.1 stored-inverse operators with inherited unchanged kernels."""

    def __init__(
        self,
        mesh: Any,
        n_vert_levels: int,
        context: CudaV841Context,
        *,
        kernel_cache: Any | None = None,
    ) -> None:
        super().__init__(mesh, n_vert_levels, kernel_cache=kernel_cache)
        mesh.validate()
        if self.vertex_degree != 3:
            raise ValueError("v8.4.1 CUDA requires MPAS vertexDegree=3")
        self.v841 = context
        context.validate(
            n_vert_levels=self.nlev,
            n_cells=self.ncells,
            n_edges=self.nedges,
            n_vertices=self.nvertices,
        )
        self._v841_kernels: dict[str, Any] = {}
        self._deformation_v841: dict[str, Any] | None = None

    def _kernel_v841(self, name: str) -> Any:
        result = self._v841_kernels.get(name)
        if result is None:
            result = self.kernel_cache.raw_kernel(
                name,
                _CUDA_SOURCE,
                module_key="hexcore.cuda_horizontal_v841",
            )
            self._v841_kernels[name] = result
        return result

    def _launch_v841(self, name: str, total: int, args: tuple[Any, ...]) -> None:
        if total < 1:
            return
        threads = 128
        blocks = (int(total) + threads - 1) // threads
        self._kernel_v841(name)((blocks,), (threads,), args)

    def mass_flux_divergence(self, rho_u: Any) -> Any:
        rho_u = self._edge_field("rho_u", rho_u)
        out = self.cp.empty((self.nlev, self.ncells), dtype=self.cp.float32)
        self._launch_v841(
            "mass_flux_divergence_v841_f32",
            self.ncells,
            (
                rho_u,
                self.mesh.edges_on_cell,
                self.mesh.n_edges_on_cell,
                self.mesh.cells_on_edge,
                self.mesh.dv_edge,
                self.v841.inv_area_cell,
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                np.int32(self.max_edges),
                out,
            ),
        )
        return out

    def solve_diagnostics(
        self,
        rho: Any,
        rho_u: Any,
        *,
        dt: float,
        apvm_upwinding: float = 0.0,
        normal_velocity: Any | None = None,
        cached_tangential_velocity: Any | None = None,
        rk_step: int | None = 3,
    ) -> CudaSolveDiagnostics:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(apvm_upwinding) or apvm_upwinding < 0.0:
            raise ValueError("apvm_upwinding must be finite and non-negative")
        cp = self.cp
        normal, rho_edge, ke_edge = self._edge_state(rho, rho_u, normal_velocity)
        reconstruct_v = rk_step is None or rk_step == 3
        if reconstruct_v:
            tangential = cp.empty_like(normal)
            self._launch(
                "tangential_velocity_f32",
                self.nedges,
                (
                    normal,
                    self.mesh.edges_on_edge,
                    self.mesh.n_edges_on_edge,
                    self.mesh.weights_on_edge,
                    np.int32(self.nlev),
                    np.int32(self.nedges),
                    np.int32(self.max_edges2),
                    tangential,
                ),
            )
        else:
            if cached_tangential_velocity is None:
                raise ValueError("RK1/RK2 diagnostics require cached tangential velocity")
            # Bound, not copied: every consumer of the tangential velocity in
            # this tree takes it as a const float* -- the APVM kernel, the
            # v8.4.1 mixing operator and the vertex diagnostics all read it and
            # write their own outputs -- so the RK1/RK2 reuse of the RK3 field
            # needs a reference, not a private nlev x nEdges image per stage.
            tangential = self._edge_field(
                "cached_tangential_velocity", cached_tangential_velocity
            )

        vorticity = cp.empty((self.nlev, self.nvertices), dtype=cp.float32)
        ke_vertex = cp.empty_like(vorticity)
        pv_vertex = cp.empty_like(vorticity)
        self._launch_v841(
            "vertex_diagnostics_v841_f32",
            self.nvertices,
            (
                normal,
                ke_edge,
                self.mesh.edges_on_vertex,
                self.mesh.vertices_on_edge,
                self.mesh.dc_edge,
                self.v841.inv_area_triangle,
                self.mesh.f_vertex,
                np.int32(self.nlev),
                np.int32(self.nedges),
                np.int32(self.nvertices),
                np.int32(self.vertex_degree),
                vorticity,
                ke_vertex,
                pv_vertex,
            ),
        )
        divergence = cp.empty((self.nlev, self.ncells), dtype=cp.float32)
        kinetic_energy = cp.empty_like(divergence)
        self._launch_v841(
            "cell_diagnostics_v841_f32",
            self.ncells,
            (
                normal,
                ke_edge,
                ke_vertex,
                self.mesh.edges_on_cell,
                self.mesh.n_edges_on_cell,
                self.mesh.cells_on_edge,
                self.mesh.vertices_on_cell,
                self.mesh.cells_on_vertex,
                self.mesh.dv_edge,
                self.v841.inv_area_cell,
                self.mesh.kite_areas_on_vertex,
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                np.int32(self.nvertices),
                np.int32(self.max_edges),
                np.int32(self.vertex_degree),
                divergence,
                kinetic_energy,
            ),
        )
        pv_edge = cp.empty_like(normal)
        self._launch(
            "pv_edge_base_f32",
            self.nedges,
            (
                pv_vertex,
                self.mesh.vertices_on_edge,
                np.int32(self.nlev),
                np.int32(self.nedges),
                np.int32(self.nvertices),
                pv_edge,
            ),
        )
        pv_cell = cp.zeros((self.nlev, self.ncells), dtype=cp.float32)
        grad_normal = cp.zeros_like(normal)
        grad_tangential = cp.zeros_like(normal)
        if apvm_upwinding > 0.0:
            self._launch_v841(
                "pv_cell_v841_f32",
                self.ncells,
                (
                    pv_vertex,
                    self.mesh.vertices_on_cell,
                    self.mesh.n_edges_on_cell,
                    self.mesh.cells_on_vertex,
                    self.mesh.kite_areas_on_vertex,
                    self.v841.inv_area_cell,
                    np.int32(self.nlev),
                    np.int32(self.ncells),
                    np.int32(self.nvertices),
                    np.int32(self.max_edges),
                    np.int32(self.vertex_degree),
                    pv_cell,
                ),
            )
            scale = np.float32(apvm_upwinding) * np.float32(dt)
            self._launch_v841(
                "pv_apvm_v841_f32",
                self.nedges,
                (
                    normal,
                    tangential,
                    pv_vertex,
                    pv_cell,
                    self.mesh.vertices_on_edge,
                    self.mesh.cells_on_edge,
                    self.v841.inv_dc_edge,
                    self.v841.inv_dv_edge,
                    scale,
                    np.int32(self.nlev),
                    np.int32(self.ncells),
                    np.int32(self.nedges),
                    np.int32(self.nvertices),
                    pv_edge,
                    grad_normal,
                    grad_tangential,
                ),
            )
        return CudaSolveDiagnostics(
            h_edge=rho_edge,
            normal_velocity=normal,
            tangential_velocity=tangential,
            vorticity=vorticity,
            divergence=divergence,
            kinetic_energy=kinetic_energy,
            pv_edge=pv_edge,
            pv_vertex=pv_vertex,
            pv_cell=pv_cell,
            grad_pv_normal=grad_normal,
            grad_pv_tangential=grad_tangential,
        )

    def attach_deformation_weights_v841(self, weights: Any) -> dict[str, Any]:
        """Upload host float32 v8.4.1 deformation weights onto the device.

        ``weights`` is a :class:`hexcore.mixing_v841.DeformationWeightsV841`
        computed by the CPU authority; the device copies are byte-identical
        H2D transfers (no device arithmetic touches them at upload).
        """

        cp = self.cp
        shape = (self.ncells, self.max_edges)
        arrays = {}
        receipts: dict[str, Any] = {}
        for name in ("coef_c2", "coef_s2", "coef_cs"):
            host = np.ascontiguousarray(
                np.asarray(getattr(weights, name), dtype=np.float32)
            )
            if host.shape != shape:
                raise ValueError(f"{name} shape {host.shape} != {shape}")
            if not np.all(np.isfinite(host)):
                raise ValueError(f"{name} contains non-finite values")
            arrays[name] = cp.asarray(host)
            receipts[name] = {
                "shape": list(shape),
                "dtype": "float32",
                "nonzero": int(np.count_nonzero(host)),
                "sha256": hashlib.sha256(
                    host.tobytes(order="C")
                ).hexdigest(),
            }
        self._deformation_v841 = arrays
        return receipts

    def compute_dry_mixing_tendencies_v841(
        self,
        normal_velocity: Any,
        tangential_velocity: Any,
        vertical_velocity: Any,
        theta_m: Any,
        rho_edge: Any,
        divergence: Any,
        vorticity: Any,
        *,
        dt: float,
        config: Any | None = None,
    ) -> CudaDryMixingTendencies:
        """v8.4.1 2-D Smagorinsky mixing on device.

        kdiff mirrors ``smagorinsky_2d`` (mpas_atm_dissipation_models.F:
        119-204) through the ``smagorinsky_v841_f32`` kernel with the
        v8.4.1 deformation weights; the u/w/theta applications reuse the
        inherited armored filter kernels, which implement the same non-LES
        stencils as ``u_dissipation_3d`` / ``w_dissipation_3d`` /
        ``scalar_dissipation_3d_les`` (see hexcore.mixing_v841).
        """

        from .mixing_v841 import V841MixingConfig

        cfg = V841MixingConfig() if config is None else config
        if not isinstance(cfg, V841MixingConfig):
            raise TypeError("config must be a mixing_v841.V841MixingConfig")
        cfg.validate()
        timestep = float(dt)
        if not np.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("dt must be finite and positive")
        deformation = getattr(self, "_deformation_v841", None)
        if deformation is None:
            raise RuntimeError(
                "v8.4.1 deformation weights are not attached; call "
                "attach_deformation_weights_v841 first"
            )

        normal = self._edge_field("normal_velocity", normal_velocity)
        tangential = self._edge_field(
            "tangential_velocity", tangential_velocity
        )
        vertical = self._interface_field("vertical_velocity", vertical_velocity)
        theta = self._cell_field("theta_m", theta_m)
        edge_rho = self._edge_field("rho_edge", rho_edge)
        div = self._cell_field("divergence", divergence)
        vort = self._field(
            "vorticity", vorticity, (self.nlev, self.nvertices)
        )
        cp = self.cp

        configured_length = float(cfg.config_len_disp)
        length = (
            float(self.mesh.nominal_min_dc)
            if configured_length == 0.0
            else configured_length
        )
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError(
                "config_len_disp or DeviceMesh.nominal_min_dc must be positive"
            )
        # Host-side scalar staging matches the float32 CPU authority bit for
        # bit: (c_s*len)**2, invDt = 1/dt, ceiling = (0.01*len**2)*invDt,
        # h4 = visc4*len**3 (dissipation-models lines 189-190, 199-200;
        # time-integration line 6323).
        length32 = np.float32(length)
        cs = np.float32(cfg.config_smagorinsky_coef)
        strain_scale = np.float32((cs * length32) * (cs * length32))
        inv_dt = np.float32(np.float32(1.0) / np.float32(timestep))
        ceiling = np.float32(
            (np.float32(0.01) * (length32 * length32)) * inv_dt
        )
        visc4 = np.float32(cfg.config_visc4_2dsmag)
        h4 = np.float32(visc4 * ((length32 * length32) * length32))

        del2 = cp.empty((self.nedges,), dtype=cp.float32)
        del4 = cp.empty_like(del2)
        self._launch(
            "mixing_scaling_f32",
            self.nedges,
            (
                self.mesh.cells_on_edge,
                self.mesh.mesh_density,
                np.int32(self.nedges),
                np.int32(bool(cfg.config_h_ScaleWithMesh)),
                del2,
                del4,
            ),
        )

        kdiff = cp.empty((self.nlev, self.ncells), dtype=cp.float32)
        self._launch_v841(
            "smagorinsky_v841_f32",
            self.ncells,
            (
                normal,
                tangential,
                self.mesh.edges_on_cell,
                self.mesh.n_edges_on_cell,
                deformation["coef_c2"],
                deformation["coef_s2"],
                deformation["coef_cs"],
                strain_scale,
                ceiling,
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                np.int32(self.max_edges),
                kdiff,
            ),
        )

        delsq_u = cp.empty_like(normal)
        tend_u = cp.empty_like(normal)
        self._launch(
            "momentum_filter_lap2_f32",
            self.nedges,
            (
                edge_rho,
                div,
                vort,
                kdiff,
                del2,
                self.mesh.cells_on_edge,
                self.mesh.vertices_on_edge,
                self.mesh.dc_edge,
                self.mesh.dv_edge,
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                np.int32(self.nvertices),
                delsq_u,
                tend_u,
            ),
        )
        delsq_div = cp.zeros((self.nlev, self.ncells), dtype=cp.float32)
        delsq_vort = cp.zeros((self.nlev, self.nvertices), dtype=cp.float32)
        if h4 > np.float32(0.0):
            self._launch(
                "laplacian_divergence_f32",
                self.ncells,
                (
                    delsq_u,
                    self.mesh.edges_on_cell,
                    self.mesh.n_edges_on_cell,
                    self.mesh.cells_on_edge,
                    self.mesh.dv_edge,
                    self.mesh.area_cell,
                    np.int32(self.nlev),
                    np.int32(self.ncells),
                    np.int32(self.nedges),
                    np.int32(self.max_edges),
                    delsq_div,
                ),
            )
            self._launch(
                "laplacian_vorticity_f32",
                self.nvertices,
                (
                    delsq_u,
                    self.mesh.edges_on_vertex,
                    self.mesh.vertices_on_edge,
                    self.mesh.dc_edge,
                    self.mesh.area_triangle,
                    np.int32(self.nlev),
                    np.int32(self.nedges),
                    np.int32(self.nvertices),
                    np.int32(self.vertex_degree),
                    delsq_vort,
                ),
            )
            self._launch(
                "momentum_filter_lap4_f32",
                self.nedges,
                (
                    edge_rho,
                    delsq_div,
                    delsq_vort,
                    del4,
                    self.mesh.cells_on_edge,
                    self.mesh.vertices_on_edge,
                    self.mesh.dc_edge,
                    self.mesh.dv_edge,
                    h4,
                    np.float32(cfg.config_del4u_div_factor),
                    np.int32(self.nlev),
                    np.int32(self.ncells),
                    np.int32(self.nedges),
                    np.int32(self.nvertices),
                    tend_u,
                ),
            )

        delsq_theta = cp.empty_like(theta)
        tend_theta = cp.empty_like(theta)
        self._launch(
            "theta_filter_lap2_f32",
            self.ncells,
            (
                theta,
                edge_rho,
                kdiff,
                del2,
                self.mesh.edges_on_cell,
                self.mesh.n_edges_on_cell,
                self.mesh.cells_on_edge,
                self.mesh.dc_edge,
                self.mesh.dv_edge,
                self.mesh.area_cell,
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                np.int32(self.max_edges),
                delsq_theta,
                tend_theta,
            ),
        )
        if h4 > np.float32(0.0):
            self._launch(
                "theta_filter_lap4_f32",
                self.ncells,
                (
                    delsq_theta,
                    del4,
                    self.mesh.edges_on_cell,
                    self.mesh.n_edges_on_cell,
                    self.mesh.cells_on_edge,
                    self.mesh.dc_edge,
                    self.mesh.dv_edge,
                    self.mesh.area_cell,
                    h4,
                    np.int32(self.nlev),
                    np.int32(self.ncells),
                    np.int32(self.max_edges),
                    tend_theta,
                ),
            )

        delsq_w = cp.empty((self.nlev, self.ncells), dtype=cp.float32)
        tend_w = cp.zeros((self.nlev + 1, self.ncells), dtype=cp.float32)
        self._launch(
            "w_filter_lap2_f32",
            self.ncells,
            (
                vertical,
                edge_rho,
                kdiff,
                del2,
                self.mesh.edges_on_cell,
                self.mesh.n_edges_on_cell,
                self.mesh.cells_on_edge,
                self.mesh.dc_edge,
                self.mesh.dv_edge,
                self.mesh.area_cell,
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                np.int32(self.max_edges),
                delsq_w,
                tend_w,
            ),
        )
        if h4 > np.float32(0.0):
            self._launch(
                "w_filter_lap4_f32",
                self.ncells,
                (
                    delsq_w,
                    del4,
                    self.mesh.edges_on_cell,
                    self.mesh.n_edges_on_cell,
                    self.mesh.cells_on_edge,
                    self.mesh.dc_edge,
                    self.mesh.dv_edge,
                    self.mesh.area_cell,
                    h4,
                    np.int32(self.nlev),
                    np.int32(self.ncells),
                    np.int32(self.max_edges),
                    tend_w,
                ),
            )

        return CudaDryMixingTendencies(
            kdiff=kdiff,
            h_mom_eddy_visc4=h4,
            h_theta_eddy_visc4=h4,
            tend_u_euler=tend_u,
            tend_w_euler=tend_w,
            tend_theta_euler=tend_theta,
            delsq_u=delsq_u,
            delsq_divergence=delsq_div,
            delsq_vorticity=delsq_vort,
            delsq_w=delsq_w,
            delsq_theta=delsq_theta,
        )

    def pressure_gradient_euler_tendency(
        self,
        pressure_perturbation: Any,
        dpdz: Any,
        cqu: Any,
        zz: Any,
        zxu: Any,
    ) -> Any:
        pressure = self._cell_field("pressure_perturbation", pressure_perturbation)
        dpdz = self._cell_field("dpdz", dpdz)
        cqu = self._edge_field("cqu", cqu)
        zz = self._cell_field("zz", zz)
        zxu = self._edge_field("zxu", zxu)
        out = self.cp.empty_like(cqu)
        self._launch_v841(
            "pressure_gradient_v841_f32",
            self.nedges,
            (
                pressure,
                dpdz,
                cqu,
                zz,
                zxu,
                self.mesh.cells_on_edge,
                self.v841.inv_dc_edge,
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                out,
            ),
        )
        return out


__all__ = ["CudaHorizontalV841"]
