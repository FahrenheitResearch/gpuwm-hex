"""Resident global convection-permitting driver for pristine MPAS v8.4.2.

Authority: mpas_atmphys_interface.F:86-557; todynamics.F:98-552.
Native initialization and scalar calendar state remain on the host. After
activation, mutable pool fields and prepared arrays belong to the GPU until
an explicit publication boundary. No native field operation may intervene.
"""
from __future__ import annotations

import os
from functools import lru_cache
from types import SimpleNamespace

import numpy as np


TARGET = dict(microphysics='mp_thompson', convection='cu_grell_freitas',
              pbl='bl_mynn', gwdo='bl_ysu_gwdo', radiation_lw='rrtmg_lw',
              radiation_sw='rrtmg_sw', cloudiness='cld_fraction',
              surface_layer='sf_mynn', land_surface='sf_noah')


def resident_requested(config, dimensions, *, partitioned=False):
    """Admit the global CUDA suite in either physics mode, or fail a request.

    ``partitioned``: a same-host rank's local mesh (halo cells and edges beyond
    nCellsSolve/nEdgesSolve) whose rank owner exchanges the physics halo groups.
    """
    setting = os.environ.get('GPUWM_HEX_MPAS_RESIDENT', 'auto')
    if setting not in ('auto', '0', '1'):
        raise ValueError('GPUWM_HEX_MPAS_RESIDENT must be auto, 0 or 1')
    supported = (getattr(config, 'table_key', 'mpasphys_v842') == 'mpasphys_v842'
                 and all(config.selectors.get(k) == v for k, v in TARGET.items())
                 and all(str(config.backends[k]) == 'hex-cuda'
                         for k in config.selectors if config.active(k))
                 and (partitioned or all(dimensions[e] == dimensions[e + 'Solve']
                                         for e in ('nCells', 'nEdges'))))
    if setting == '1' and not supported:
        raise ValueError('resident v8.4.2 requires the global convection_permitting CUDA suite')
    return setting != '0' and supported


def ported_route_refusal(config):
    """Refuse, by name, an MPAS suite the ported framework cannot run yet.

    THE BREAKAGE THIS PREVENTS: without the native MPAS library the ported
    framework (framework_backend_v842.FrameworkPhysics) implements only the
    resident interface glue, which this module admits for the global
    convection_permitting CUDA suite.  Every other v8.4.2 suite (MPAS's default
    mesoscale_reference among them) reaches the native-glue path and stops
    before step 1 with "MPAS framework operation has no Python implementation:
    mpas_to_physics".  The door names that before any work instead.
    Retire this refusal when the non-resident interface glue is ported.
    """
    from .runtime_v842 import framework_only
    if config is None or not framework_only(config):
        return None
    gaps = [f'{slot}={config.selectors.get(slot)} (needs {scheme})'
            for slot, scheme in TARGET.items() if config.selectors.get(slot) != scheme]
    gaps += [f'{slot} on {config.backends[slot]} (needs hex-cuda)'
             for slot in config.selectors
             if config.active(slot) and str(config.backends[slot]) != 'hex-cuda']
    if not gaps:
        return None
    options = getattr(config, 'options', None) or {}
    suite = str(options.get('config_physics_suite', '') or 'this selection')
    return (f'MPAS physics suite {suite} is not runnable in this release: its MPAS interface glue '
            '(MPAS_to_physics and the scheme driver entries) is ported to the GPU only for the '
            'convection_permitting suite, and without the native MPAS library the run would stop '
            'before step 1 (MPAS framework operation has no Python implementation: mpas_to_physics). '
            'Differs from convection_permitting in: ' + '; '.join(gaps) + '. '
            'Run --physics-suite convection_permitting, or the default ArWen physics.')


@lru_cache(maxsize=1)
def preparation_source():
    """Native surface-pressure sum and interpolation writes, interface.F:437,479-483.

    Shared carriers preserve the strict CUDA operation sequence. v8.4.2 adds
    pressure_p then pressure_base separately, rather than their rounded sum.
    """
    from ..cuda_physics_prep_v841 import _CUDA_SOURCE
    from .math_v842 import bind_common_carrier_math
    replacements = {
        'float *plrad, int *invalid)':
            'float *plrad, int *invalid, const float *pressure_perturbation, '
            'const float *pressure_base, float *weights_minus, float *weights_plus)',
        'psfc_p[cell] = mpas_add(hydro, pres_p[cell]);':
            'psfc_p[cell] = mpas_add(mpas_add(hydro, pressure_perturbation[cell]), pressure_base[cell]);',
        't2_p[i] = mpas_add(\n            mpas_mul(fzm, t_p[i]),':
            'weights_minus[i] = fzm; weights_plus[i] = fzp;\n'
            '        t2_p[i] = mpas_add(\n            mpas_mul(fzm, t_p[i]),',
    }
    source = _CUDA_SOURCE
    for before, after in replacements.items():
        # The interpolation expression also occurs in the validation kernel;
        # only the production kernel is extended with the extra arguments.
        if before not in source:
            raise RuntimeError('shared preparation source no longer matches the v8.4.2 binding')
        source = source.replace(before, after, 1)
    return bind_common_carrier_math(source)


_LATITUDE = r'''
extern "C" __global__ void resident_latitude(int n,float degrad,const float *lat,float *out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=__fdiv_rn(lat[i],degrad);
}
'''


def native_physics_geometry(provider, *, local_cells=None):
    """Retain initialized native vectors, mpas_atmphys_init.F:538-592.

    init_dirs_forphys normalizes its east/north vectors after evaluating
    native sin/cos. Reconstructing them from latitude/longitude loses those
    rounded words. Pool layouts are component,entity; projection consumes
    contiguous entity,component. Only connectivity changes value, from
    native one-based indices to the CUDA kernel's zero-based indices.
    """
    from ..cuda_backend.containers import TransferStats
    from ..cuda_physics_v841 import CudaPhysicsGeometryV841
    cp = provider.cp
    dimensions = provider.session.layout.dimensions
    if local_cells is None:
        east = provider.field('mesh', 'east').T
        north = provider.field('mesh', 'north').T
    else:
        # A same-host rank: owned edges reference halo cells, so the projected
        # cell vectors cover every local cell (the halo-exchanged momentum
        # tendencies are local-extent too), todynamics.F:501-552.
        east = cp.asarray(provider.session.native.get_field('mesh', 'east')[:, :local_cells]).T
        north = cp.asarray(provider.session.native.get_field('mesh', 'north')[:, :local_cells]).T
    geometry = CudaPhysicsGeometryV841(
        cells_on_edge=cp.ascontiguousarray(
            provider.field('mesh', 'cellsOnEdge').T - cp.int32(1)),
        east_cell=cp.ascontiguousarray(east),
        north_cell=cp.ascontiguousarray(north),
        edge_normal_vectors=cp.ascontiguousarray(
            provider.field('mesh', 'edgeNormalVectors').T),
        n_cells=dimensions['nCellsSolve'] if local_cells is None else local_cells,
        n_edges=dimensions['nEdgesSolve'],
        # Uploads belong to the persistent provider's transfer receipt;
        # this layout adapter adds no direct host/device transfers.
        h2d=TransferStats(0, 0.0),
    )
    geometry.validate()
    return geometry


def regional_physics_geometry(provider):
    """native_physics_geometry for a limited-area cull: native's garbage cell made explicit.

    Every ring-7 boundary edge of a cull names no second cell (stored 0, which native maps to the garbage cell
    nCells+1 at read time).  The projection kernel indexes cellsOnEdge directly, so the closed-sphere geometry would
    read one column outside the tendency arrays.  Here that endpoint is the garbage cell, the last column of a padded
    view whose east/north vectors are zero, and CudaMpasCouplingV842(regional_garbage=True) gives the tendencies a
    zero column there: the edge takes half its one real cell's tendency, as native's tend_toEdges does with its zero
    garbage tendency.  Those edges lie in the specified zone, whose tendencies the boundary driver replaces.
    """
    from ..cuda_backend.containers import TransferStats
    from ..cuda_physics_v841 import CudaPhysicsGeometryV841
    cp = provider.cp
    dimensions = provider.session.layout.dimensions
    n_cells = int(dimensions['nCellsSolve'])
    cells = provider.field('mesh', 'cellsOnEdge').T - cp.int32(1)
    cells = cp.where((cells >= 0) & (cells < n_cells), cells, cp.int32(n_cells)).astype(cp.int32)

    def padded(name):
        vectors = provider.field('mesh', name).T
        out = cp.zeros((n_cells + 1, vectors.shape[1]), dtype=vectors.dtype)
        out[:n_cells] = vectors[:n_cells]
        return out

    geometry = CudaPhysicsGeometryV841(
        cells_on_edge=cp.ascontiguousarray(cells),
        east_cell=padded('east'), north_cell=padded('north'),
        edge_normal_vectors=cp.ascontiguousarray(provider.field('mesh', 'edgeNormalVectors').T),
        n_cells=n_cells + 1, n_edges=dimensions['nEdgesSolve'],
        h2d=TransferStats(0, 0.0), regional_garbage_cell=n_cells)
    geometry.validate()
    return geometry


def _solved_edges_reference_dummy(native, n_cells_solve, n_edges_solve) -> bool:
    """Whether a solved edge names no second real cell (native 1-based: 0 or the dummy nCells+1)."""
    cells = np.asarray(native.get_field('mesh', 'cellsOnEdge'))
    cells = cells.reshape(-1, cells.shape[-1])[:, :int(n_edges_solve)]
    return bool(np.any((cells < 1) | (cells > int(n_cells_solve))))


class ResidentPhysicsGlue:
    """GPU field authority and native-order driver glue for the admitted suite."""

    def __init__(self, session, coupling, mesh, *, halo=None):
        from ..cuda_physics_v841 import CudaMpasCouplingV842
        from .surface_update_v842 import NativeSurfaceService
        from .sfclay_mm5_v842 import run_native_diagnostics
        self.session, self.coupling = session, coupling
        self.provider = p = session.device_pools
        self.cp, self.cache = p.cp, coupling.physics_cache
        self.dimensions = d = session.layout.dimensions
        self.nlev, self.ncells = d['nVertLevels'], d['nCellsSolve']
        self.names = tuple(next(row['components'] for row in session.layout.fields
                               if row['pool'] == 'state' and row['name'] == 'scalars'))
        if self.names[0] != 'qv':
            raise ValueError('resident MPAS preparation requires native qv as the first scalar')
        # These allocators only allocate guarded module arrays, with no field
        # arithmetic. Allocate once after physics_init established tile bounds.
        stems = ['forall_physics', 'cloudiness', 'radiation_sw', 'radiation_lw',
                 'sfclayer', 'lsm', 'pbl', 'gwdo', 'convection', 'microphysics']
        if session.config.active('seaice'):
            stems.append('seaice')
        for stem in stems:
            entry = 'allocate_' + stem
            available = dict(configs='configs', xtime_s=np.float32(0))
            args = session.native.abi['entries'][entry]['arguments']
            session.native.call(entry, **{name: available[name] for name in args})
        p.buffers.clear()
        p.values.clear()
        p.resident = True
        p.resident_buffers.clear()
        self.invalid = self.cp.zeros(1, dtype=self.cp.int32)
        self.rho_dry = self.cp.empty((self.nlev, self.ncells), dtype=self.cp.float32)
        self.number_scratch = {}
        self.halo = halo
        self.local_rates = {}
        local_cells = None
        if halo is not None:
            local_cells = d['nCells']
            # Native's full-extent momentum tendencies (nCells): owned columns
            # from the schemes, halo columns from their owners, before the edge
            # projection reads both cells of every owned edge.
            for group in ('physics:blten', 'physics:cuten'):
                for name in session.native._groups.get(group, ((),))[0]:
                    self.local_rates[name] = self.cp.zeros((self.nlev, local_cells), dtype=self.cp.float32)
        regional = bool(session.layout.configs.get('config_apply_lbcs', False))
        if regional and halo is not None:
            raise ValueError('a limited-area MPAS suite run runs on one card (bind_limited_area refuses more)')
        # A solved edge whose second cell is the native dummy (or 0) has only one real cell: a limited-area cull's
        # boundary ring, and an LTS rate class's outermost halo ring (lts_v842 solves every local entity).  The
        # closed-sphere geometry indexes that endpoint as column nCellsSolve, one past the tendency arrays.
        # THE BREAKAGE THIS PREVENTS: on an MPAS-suite LTS class the PBL edge projection read whatever followed
        # rublten/rvblten in device memory into those edges' momentum tendency, and the first step of some class
        # was refused by the dycore validation flag at time 0 (3 of 3 storm-following moving segments on box L
        # and on an RTX 4090 host; every class with 0x7F-filled fresh pool blocks: 1,159 edges of the rate-4
        # class at |tend| ~ 3e36, MPAS-ROUTE-SPEED-2026-10-24).  Such edges take the regional geometry's zero
        # garbage column: half their one real cell's tendency, native's tend_toEdges word.
        dummy_edges = (not regional and halo is None
                       and _solved_edges_reference_dummy(session.native, d['nCellsSolve'], d['nEdgesSolve']))
        geometry = (regional_physics_geometry(p) if regional or dummy_edges
                    else native_physics_geometry(p, local_cells=local_cells))
        self.assembler = CudaMpasCouplingV842(regional_garbage=regional or dummy_edges,
            selectors=session.config.execution_options, scalar_names=self.names,
            geometry=geometry, kernel_cache=self.cache,
            # Single-rank closed sphere: all cells and edges are owned, no
            # neighbor ranks and no boundary endpoint references the dummy.
            # A same-host rank exchanges the group's owner words before the
            # edge projection reads halo cells (todynamics.F:380-412).
            exchange_halo_group=(lambda group: None) if halo is None else self.exchange_group,
            pbl_edge=p.field('tend_physics', 'rublten_Edge'),
            # Owned edges, like the other edge pools (equal to nEdges on one card).
            convection_edge=self.cp.zeros((self.nlev, d['nEdgesSolve']), dtype=self.cp.float32),
            tend_u_phys=p.field('diag', 'tend_u_phys'))
        if session.surface_service is None:
            session.surface_service = NativeSurfaceService(session, self.cp)
        operations = dict(session.surface_service.callbacks())
        operations.update(MPAS_to_physics=self.prepare, physics_get_tend=self.get_tend,
                          post_transport_prepare=session.hooks.external['post_transport_prepare'],
                          pre_microphysics=self.noop, post_microphysics=self.noop,
                          update_convection_step1=self.noop,
                          snapshot_gwdo_tendencies=session.snapshot_gwdo,
                          atmphys_sfc_diagnostics=lambda call: run_native_diagnostics(p, call))
        # Native pre/post are empty without MPAS_OPENACC. GF has no step1
        # branch. Allocator/deallocator pairs now own one persistent lifetime.
        for stem in stems:
            operations['allocate_' + stem] = self.noop
            operations['deallocate_' + stem] = self.noop
        if session.config.options.get('config_o3climatology'):
            from .radshared_v842 import bind_native_ozone_pools, ozone_slot
            from dataclasses import replace
            def ozone(call):
                p.begin(call, {})
                result = ozone_slot(replace(call, pools=bind_native_ozone_pools(p, call)))
                p.commit()
                return result
            operations['o3climatology_from_MPAS'] = ozone
        if session.precipitation_window is not None:
            session.precipitation_window.device_pools = p
        self.operations = frozenset(operations)
        session.hooks.external.update(operations)
        session.hooks.resident_operations = self.operations
        session.resident_glue = self

    def exchange_group(self, group):
        """exchange_halo_group(domain, group) on a same-host rank, todynamics.F:380-412."""
        fields = []
        for name in self.session.native._groups[group][0]:
            local = self.local_rates[name]
            local[:, :self.ncells] = self.provider.field('tend_physics', name)
            fields.append((f'physics/{name}', local, 'cell'))
        self.halo.exchange(group, fields)

    @staticmethod
    def noop(call):
        """Persistent allocation lifetime, or an empty native conditional body."""
        return None

    def _launch(self, name, source, args):
        unit = 'latitude' if name == 'resident_latitude' else 'preparation'
        kernel = self.cache.raw_kernel(name, source, module_key='hexcore.mpasphys.resident_v842.' + unit)
        kernel(((self.ncells + 127) // 128,), (128,), args)

    def prepare(self, call):
        """MPAS_to_physics, interface.F:284-557, with native diagnostic winds."""
        from ..cuda_physics_prep_v841 import prepare_mpas_number_carriers_v842
        p, cp, nz, nc = self.provider, self.cp, self.nlev, self.ncells
        p.level = 1
        field = p.field
        prepared = p.prepared
        state = {name: field('state', name) for name in ('rho_zz', 'theta_m', 'scalars', 'w')}
        scalar = state['scalars']
        species = {}
        for name in self.names:
            if name in ('qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'nc', 'ni', 'nifa', 'nwfa'):
                try:
                    species[name] = prepared(name + '_p')[:nz]
                except RuntimeError:
                    if name not in self.number_scratch:
                        self.number_scratch[name] = cp.empty((nz, nc), dtype=cp.float32)
                    species[name] = self.number_scratch[name]
            else:
                if name not in self.number_scratch:
                    self.number_scratch[name] = cp.empty((nz, nc), dtype=cp.float32)
                species[name] = self.number_scratch[name]
        pointers = cp.asarray(np.array([species[name].data.ptr for name in self.names], dtype=np.uint64))
        mass_names = ('zz_p', 'rho_p', 'th_p', 't_p', 'pi_p', 'pres_p', 'zmid_p', 'dz_p')
        mass = {name: prepared(name)[:nz] for name in mass_names}
        constants = p['constants']
        source = preparation_source()
        self._launch('prep_mass_v841_f32', source,
            (np.int32(nz), np.int32(nc), np.int32(len(self.names)),
             np.float32(constants['R_v'] / constants['R_d']),
             state['rho_zz'], state['theta_m'], scalar, field('diag', 'exner'),
             field('diag', 'pressure_base'), field('diag', 'pressure_p'),
             field('mesh', 'zgrid'), field('mesh', 'zz'), pointers,
             mass['zz_p'], self.rho_dry, *(mass[name] for name in mass_names[1:]), self.invalid))
        for name, diagnostic in (('u_p', 'uReconstructZonal'), ('v_p', 'uReconstructMeridional')):
            prepared(name)[:nz] = field('diag', diagnostic)
        interface_names = ('w_p', 'z_p', 't2_p', 'pres2_p', 'pres_hyd_p', 'pres2_hyd_p',
                           'pres_hydd_p', 'pres2_hydd_p', 'psfc_p', 'psfc_hyd_p',
                           'psfc_hydd_p', 'znu_p', 'znu_hyd_p')
        self._launch('prep_interfaces_v841_f32', source,
            (np.int32(nz), np.int32(nc), np.float32(constants['gravity']),
             state['rho_zz'], field('mesh', 'zz'), species['qv'], mass['pres_p'], mass['t_p'],
             mass['rho_p'], mass['dz_p'], field('mesh', 'zgrid'), state['w'],
             *(prepared(name) for name in interface_names), field('diag_physics', 'plrad'), self.invalid,
             field('diag', 'pressure_p'), field('diag', 'pressure_base'),
             prepared('fzm_p'), prepared('fzp_p')))
        field('diag', 'surface_pressure')[...] = prepared('psfc_p')
        flags = {name: bool(p['scalar_state']['f_' + name]) for name in ('nc', 'ni', 'nifa', 'nwfa')}
        numbers = prepare_mpas_number_carriers_v842(SimpleNamespace(species_p=species),
            microphysics=call.options['config_microp_scheme'], pbl=call.options['config_pbl_scheme'],
            scalar_flags=flags, n_cells_solve=nc, kernel_cache=self.cache)
        for name, value in numbers.items():
            prepared(name + '_p')[:nz] = value
        # radiation_lw_from_MPAS.F:366. Other LW inputs are bound directly
        # by its CUDA slot; latitude is the one remaining prepared surface.
        self._launch('resident_latitude', _LATITUDE,
            (np.int32(nc), np.float32(constants['degrad']), field('mesh', 'latCell'), prepared('xlat_p')))
        p.commit()

    def get_tend(self, call):
        """Assemble outer tendencies once, todynamics.F:98-498."""
        p = self.provider
        p.level = 2
        mass = p.field('state', 'rho_zz')
        p.level = 1
        rates = {row['pool_name']: p.field('tend_physics', row['pool_name'])
                 for row in self.session.layout.fields if row['pool'] == 'tend_physics'
                 and (row['pool_name'].endswith('ten')
                      or row['pool_name'] in ('rthratenlw', 'rthratensw'))}
        rates.update(getattr(self, 'local_rates', {}))
        out = self.assembler.get_tend(mass=mass, mass_edge=p.field('diag', 'rho_edge'),
            theta_m=p.field('state', 'theta_m'), scalars=p.field('state', 'scalars'),
            tendencies=rates, n_cells_solve=self.ncells,
            n_edges_solve=self.dimensions['nEdgesSolve'], rk_step=call.pools['rk_step'],
            dynamics_substep=call.pools['dynamics_substep'], invalid_flag=self.invalid)
        p.field('tend', 'scalars_tend')[...] = out.scalars
        p.commit()
        return out
