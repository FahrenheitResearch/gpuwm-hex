"""RRFS vector-invariant momentum subsystem.

Source: RRFSx/MPAS-Model MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5494-5651 at 1d6e370e.
The fork omits reference-wind perturbation Coriolis. Shared theta/w finish and
split-flux reductions have the same non-LES arithmetic in both sources.
"""
from typing import Any
import numpy as np
from .cuda_fp32 import CUDA_FTZ_HELPERS
from .cuda_v841 import CudaV841Context
from .cuda_backend.containers import require_resident_array
from .cuda_dynamics_v841 import (_cupy, _shape, theta_finish_cuda_v841,
    w_finish_cuda_v841, enforce_rw_endpoints_cuda_v841,
    validate_positive_state_cuda_v841, validate_recovered_state_cuda_v841,
    accumulate_finite_array_cuda_v841, accumulate_split_flux_cuda_v841,
    finish_split_flux_cuda_v841)
_CUDA_SOURCE = CUDA_FTZ_HELPERS + r"""
#define E2(k,e,ne) ((k)*(ne)+(e))
#define C2(k,c,nc) ((k)*(nc)+(c))
#define DYNAMICS_ELEMENT_LOOP(total) for(int element=blockIdx.x*blockDim.x+threadIdx.x; element<(total);element+=gridDim.x*blockDim.x)
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5148-6409 at 1d6e370e.
extern "C" __global__ void vector_momentum_rrfs_f32(
    const int nlev, const int ncells, const int nedges, const int max_edges2,
    const float *u, const float *rho_edge, const float *pv_edge,
    const float *kinetic, const float *mass_divergence,
    const int *cells_on_edge, const int *edges_on_edge,
    const int *n_edges_on_edge, const float *weights_on_edge,
    const float *inv_dc_edge,
    const float *initial_tendency, const int has_initial,
    float *result)
{
    DYNAMICS_ELEMENT_LOOP(nlev * nedges) {
        const int edge = element % nedges;
        const int k = element / nedges;
        const int c0 = cells_on_edge[2 * edge];
        const int c1 = cells_on_edge[2 * edge + 1];
        const int index = E2(k, edge, nedges);
        float q = 0.0f;
        for (int slot = 0; slot < n_edges_on_edge[edge]; ++slot) {
            const int offset = edge * max_edges2 + slot;
            const int neighbor = edges_on_edge[offset];
            const float work_pv = mpas_mul(0.5f, mpas_add(
                pv_edge[index], pv_edge[E2(k, neighbor, nedges)]));
            q = mpas_add(q, mpas_mul(mpas_mul(
                weights_on_edge[offset], u[E2(k, neighbor, nedges)]), work_pv));
        }
        const float gradient = mpas_mul(mpas_sub(
            kinetic[C2(k, c1, ncells)], kinetic[C2(k, c0, ncells)]),
            inv_dc_edge[edge]);
        const float horizontal = mpas_mul(rho_edge[index], mpas_sub(q, gradient));
        result[index] = mpas_sub(has_initial ? mpas_add(initial_tendency[index], horizontal) : horizontal,
            mpas_mul(mpas_mul(u[index], 0.5f),
                mpas_add(mass_divergence[C2(k, c0, ncells)],
                    mass_divergence[C2(k, c1, ncells)])));
    }
}

// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:6070-6076 at 1d6e370e.
extern "C" __global__ void euler_w_accumulate_rrfs_f32(
    const int nl, const int nc, const int solve,
    const float *pp, const float *dpdz, const float *cqw,
    const float *rdzu, const float *fzm, const float *fzp, float *tend)
{
    DYNAMICS_ELEMENT_LOOP((nl-1)*solve) {
        const int c=element%solve, k=1+element/solve;
        const float pressure=mpas_mul(rdzu[k],mpas_sub(pp[C2(k,c,nc)],pp[C2(k-1,c,nc)]));
        const float buoyancy=mpas_add(mpas_mul(fzm[k],dpdz[C2(k,c,nc)]),
                                      mpas_mul(fzp[k],dpdz[C2(k-1,c,nc)]));
        tend[C2(k,c,nc)]=mpas_sub(tend[C2(k,c,nc)],mpas_mul(cqw[C2(k,c,nc)],mpas_sub(pressure,buoyancy)));
    }
}

"""

# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:6070-6076 at 1d6e370e.
def accumulate_euler_w_cuda_rrfs(tendency, pressure, dpdz, cqw, rdzu, fzm, fzp,
                                *, cell_solve_count=None, kernel_cache=None):
    """Subtract pressure/buoyancy from the already accumulated mixing pool."""
    nl,nc=pressure.shape
    solve=nc if cell_solve_count is None else int(cell_solve_count)
    _launch('euler_w_accumulate_rrfs_f32',(nl-1)*solve,
            (np.int32(nl),np.int32(nc),np.int32(solve),pressure,dpdz,cqw,rdzu,fzm,fzp,tendency),kernel_cache)
    return tendency

# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5494-5651 at 1d6e370e.
def _launch(name,count,args,cache):
    from .cuda_compile_rrfs import kernel
    if count:
        kernel(_CUDA_SOURCE,"hexcore.cuda_dynamics_rrfs",name,cache)(((count+127)//128,),(128,),args)

# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5494-5651 at 1d6e370e.
def vector_momentum_tendency_cuda_rrfs(
    mesh: Any,
    context: CudaV841Context,
    *,
    normal_velocity: Any,
    rho_edge: Any,
    pv_edge: Any,
    kinetic_energy: Any,
    horizontal_divergence: Any,
    initial_tendency: Any | None = None,
    kernel_cache: Any | None = None,
) -> Any:
    cp = _cupy()
    nlev, nedges = _shape(normal_velocity, "normal_velocity")
    ncells = int(mesh.n_cells)
    if nedges != int(mesh.n_edges):
        raise ValueError("normal_velocity edge extent differs from mesh")
    mesh.validate()
    context.validate(
        n_vert_levels=nlev,
        n_cells=ncells,
        n_edges=nedges,
        n_vertices=int(mesh.n_vertices),
    )
    for name, value, shape in (
        ("rho_edge", rho_edge, (nlev, nedges)),
        ("pv_edge", pv_edge, (nlev, nedges)),
        ("kinetic_energy", kinetic_energy, (nlev, ncells)),
        ("horizontal_divergence", horizontal_divergence, (nlev, ncells)),
    ):
        require_resident_array(name, value, dtype=np.float32, shape=shape)
    out = cp.empty_like(normal_velocity)
    if initial_tendency is not None:
        require_resident_array("initial_tendency", initial_tendency, dtype=np.float32, shape=(nlev, nedges))
    arguments = (
        np.int32(nlev), np.int32(ncells), np.int32(nedges),
        np.int32(mesh.max_edges2), normal_velocity, rho_edge, pv_edge,
        kinetic_energy, horizontal_divergence, mesh.cells_on_edge,
        mesh.edges_on_edge, mesh.n_edges_on_edge, mesh.weights_on_edge,
        context.inv_dc_edge,
        normal_velocity if initial_tendency is None else initial_tendency,
        np.int32(initial_tendency is not None),
    )
    _launch("vector_momentum_rrfs_f32", nlev * nedges, (*arguments, out), kernel_cache)
    return out
