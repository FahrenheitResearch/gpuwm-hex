"""Prepared CUDA slot views; mpas_atmphys_driver.F:185-384.

The native driver allocates/prepares these pools before a slot and consumes its
changed words after it. This adapter copies ABI storage and owns solve/halo
extents; schemes retain all arithmetic and from/to conversions.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import importlib
import json
from pathlib import Path
from typing import Any

from ..device_memory_release import phase_release
from ..physics_suites import load_table
from .driver_v842 import Backend


class NativeDevicePoolViews(Mapping):
    """Lazy solve-cell views of native ABI storage; mpas_atmphys_interface.F:284-425."""

    # MPAS driver allocates the native prepared arrays before slots: mpas_atmphys_driver.F:185-209.
    def __init__(self, session, device):
        self.session, self.native, self.cp = session, session.native, device
        self.buffers = {}
        self.values = {}
        self.level = 1
        self.resident = False
        self.resident_buffers = {}
        self._field_rows = {}
        self.transfers = {'initial_upload_bytes': 0, 'publication_download_bytes': 0}

    def __iter__(self):
        """Named pools follow Registry.xml:1360-4020."""
        return iter((*self.session.layout.pools,'configs','constants','scalar_state','prepared'))

    def __len__(self):
        """Registry pool enumeration; Registry.xml:1360-4020."""
        return len(self.session.layout.pools)+4

    def __getitem__(self, name):
        """Expose native fields or prepared globals; mpas_atmphys_vars.F:68-955."""
        if name in self.values:return self.values[name]
        if name=='configs':return self.session.layout.configs
        if name=='constants':return _ConstantViews(self.native)
        if name=='scalar_state':return _ScalarViews(self.native)
        if name=='prepared':return _PreparedViews(self)
        if name not in self.session.layout.pools:raise KeyError(name)
        if name not in self.values:self.values[name]=_PoolViews(self,name)
        return self.values[name]

    def field(self, pool, name, *, resident=False):
        """Pack solve cells or full storage; reject mixed cached extents.

        mpas_atmphys_interface.F:304-305. Both layouts must be contiguous:
        slicing a full multidimensional allocation cannot supply the packed
        solve layout while halo/dummy columns remain in its row stride.
        """
        if name in self.session.layout.dimensions:
            return self.session.layout.dimensions[name]
        row=self._field_rows.get((pool,name))
        if row is None:
            row=next((r for r in self.session.layout.fields
                      if r['pool']==pool and name in (r['name'],r['pool_name'])),None)
            if row is None:raise KeyError(f'{pool}.{name}')
            for alias in (row['name'],row['pool_name']):
                self._field_rows[pool,alias]=row
        key=pool,row['pool_name'],self.level if pool=='state' else 1
        entity=next((d for d in row['dimensions'] if d in ('nCells','nEdges','nVertices')),None)
        selection=(...,slice(0,self.session.layout.dimensions[entity+'Solve'])) if entity and not resident else (...,)
        cached=self.buffers.get(key)
        if cached is None and self.resident:
            cached=self.resident_buffers.get(key)
        if cached is not None:
            host,packed,cached_selection=cached
            requested_shape=host[selection].shape
            cached_shape=host[cached_selection].shape
            if cached_shape != requested_shape:
                raise ValueError(f'cached extent conflict for {pool}.{row["pool_name"]} level {key[2]}: '
                                 f'requested {requested_shape}, cached {cached_shape}; '
                                 'owned and full layouts cannot share a packed allocation')
            self.buffers[key]=cached
            return packed
        host=self.native.get_field(pool,row['pool_name'],level=key[2])
        selected=host[selection]
        # ascontiguousarray promotes rank-zero arrays to length one. Registry
        # scalars must keep rank zero through publication and scheme bindings.
        packed=self.cp.ascontiguousarray(self.cp.asarray(selected)).reshape(selected.shape)
        self.buffers[key]=(host,packed,selection)
        if self.resident:
            self.resident_buffers[key] = self.buffers[key]
            self.transfers['initial_upload_bytes'] += packed.nbytes
        return packed

    def prepared(self, name):
        """Native i,k,j globals become k,i views; mpas_atmphys_vars.F:179-217; mpas_atmphys_vars.F:934-942."""
        key=('prepared',name)
        if key in self.buffers:return self.buffers[key][1]
        if self.resident and key in self.resident_buffers:
            self.buffers[key] = self.resident_buffers[key]
            return self.buffers[key][1]
        value=self.native.get_variable(name)
        if value is None:raise RuntimeError(f'native prepared array {name} is not allocated before CUDA slot')
        host,lower=value
        if host.ndim not in (2,3,4) or host.shape[2 if host.ndim==4 else -1]!=1 or lower[0]!=1:
            raise ValueError(f'native prepared {name} must have its declared i[,k],one-j bounds')
        count=self.session.layout.dimensions['nCellsSolve']
        # Surface globals are i,j; profiles are i,k,j. MPAS vars.F:934-942.
        view=host[:count,:,0,:].transpose(2,1,0) if host.ndim==4 else host[:count,:,0].T if host.ndim==3 else host[:count,0]
        packed=self.cp.ascontiguousarray(self.cp.asarray(view))
        self.buffers[key]=(host,packed,lower)
        if self.resident:
            self.resident_buffers[key] = self.buffers[key]
            self.transfers['initial_upload_bytes'] += packed.nbytes
        return packed

    def write_field(self, pool, name, value, *, level=1):
        """Update the authoritative device allocation without a host round trip."""
        old = self.level
        self.level = level
        try:
            target = self.field(pool, name)
            source = self.cp.asarray(value)
            if source.shape == target.shape:
                target[...] = source
            elif (source.shape[:-1] == target.shape[:-1]
                  and (source.shape[-1] == target.shape[-1] + 1
                       or (getattr(self.session.layout, 'samehost_partition', False)
                           and source.shape[-1] > target.shape[-1]))):
                # The native dummy column, or on a same-host rank the halo
                # columns: owned entities come first in a rank's local order.
                target[...] = source[..., :target.shape[-1]]
            else:
                raise ValueError(f'resident {pool}.{name} extent differs: {source.shape} != {target.shape}')
        finally:
            self.level = old

    def publish(self):
        """Copy persistent pool words at an explicit output/checkpoint boundary.

        Prepared scheme scratch is not Registry output. Its native allocation
        is retained solely for ABI shape discovery and eventual finalization.
        """
        for key, (host, packed, selection) in self.resident_buffers.items():
            if key[0] == 'prepared':
                continue
            if packed.dtype in (self.cp.float32, self.cp.int32):
                value = packed.view(self.cp.uint32).get(order='F').view(packed.dtype)
            else:
                value = packed.get()
            host[selection] = value
            self.native.put_field(key[0], key[1], host, level=key[2],
                                  nlevels=self.native._fields[key[0], key[1]][2])
            self.transfers['publication_download_bytes'] += packed.nbytes

    def resident_pool(self, name):
        """Resident rings; RRFS fork MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:668-710."""
        return _ResidentPoolViews(self,name)

    def exchange_halo_group(self, name):
        """Native group transaction; RRFS fork MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:1336-1367."""
        members=self.halo_groups[name]
        self.cp.cuda.get_current_stream().synchronize()
        for field in members:
            pool,key=self.native._registered[field]
            host,packed,selection=self.buffers[pool,key,1]
            host[selection]=packed.get()
            self.native.put_field(pool,key,host,level=1,nlevels=self.native._fields[pool,key][2])
        self.native.exchange_halo_group(name)
        for field in members:
            pool,key=self.native._registered[field]
            host,packed,selection=self.buffers[pool,key,1]
            updated=self.native.get_field(pool,key,level=1)
            host[...]=updated
            packed[...] = self.cp.asarray(updated[selection])

    def begin(self, call, descriptor):
        """Bind a slot after native preparation; mpas_atmphys_driver.F:209-365."""
        self.buffers.clear();self.values.clear()
        self.halo_groups={g['name']:tuple(g['fields']) for g in descriptor.get('halo_groups',())}
        for group in descriptor.get('halo_groups',()):
            if group['name'] not in self.native._groups:
                self.native.exchange_group(group['name'],group['fields'],level=group['time_level'],
                                           halo_layers=group['halo_layers'])
        self.level=2 if call.slot=='microphysics' else 1
        self.values['constants']=_ConstantViews(self.native,descriptor.get('constant_aliases',{}))
        for pool, aliases in descriptor.get("profile_pools", {}).items():
            self.values[pool]=_ProfileViews(self,aliases)
        profiles=descriptor.get('profiles',{})
        if profiles:
            self.values['atm_input']=_ProfileViews(self,profiles)
        binder=descriptor.get('bind')
        if binder:
            module=importlib.import_module('.'+descriptor['module'],__package__)
            return getattr(module,binder)(self,call)
        return self

    def commit(self):
        """Return changed solve words without touching halos; mpas_atmphys_driver.F:209-384."""
        if self.resident:
            # Scheme adapters may retire input-only transaction entries. The
            # persistent authority retains those allocations independently.
            self.resident_buffers.update(self.buffers)
            return
        for key,(host,packed,selection) in self.buffers.items():
            if (key[0] != 'prepared' and isinstance(packed, getattr(self.cp, 'ndarray', ()))
                    and packed.dtype in (self.cp.float32, self.cp.int32)):
                # Ordinary pools retain Fortran order. Pack integer words on
                # the GPU before the existing transfer to preserve NaN payloads
                # and avoid a strided CPU assignment. Prepared globals below
                # use different transposed layouts and retain their old path.
                value=packed.view(self.cp.uint32).get(order='F').view(packed.dtype)
            else:
                value=packed.get() if hasattr(packed,'get') else packed
            if key[0]=='prepared':
                if host.ndim==4:
                    host[:value.shape[2],:,0,:]=value.transpose(2,1,0)
                elif host.ndim==3:
                    host[:value.shape[1],:,0]=value.T
                else:
                    host[:value.shape[0],0]=value
                self.native.put_variable(key[1],host,lower=selection)
            else:
                host[selection]=value
                self.native.put_field(key[0],key[1],host,level=key[2],
                                      nlevels=self.native._fields[key[0],key[1]][2])
        self.buffers.clear()


class _PoolViews(Mapping):
    """Native pool keys, Registry.xml:1360-4020."""
    def __init__(self, owner, pool):
        """Pool allocation ownership; mpas_pool_routines.F:2110-3143."""
        self.owner,self.pool=owner,pool
    def __iter__(self):
        """Enumerate actual Registry names; Registry.xml:1360-4020."""
        return iter(r['pool_name'] for r in self.owner.session.layout.fields if r['pool']==self.pool)
    def __len__(self):
        """Registry field count; Registry.xml:1360-4020."""
        return sum(1 for _ in self)
    def __getitem__(self,name):
        """Access solve-cell storage; mpas_atmphys_interface.F:304-305."""
        return self.owner.field(self.pool,name)


class _PreparedViews(Mapping):
    """Prepared global names; mpas_atmphys_vars.F:68-955."""
    def __init__(self,owner):
        """Globals are allocated by native glue; mpas_atmphys_vars.F:179-217; mpas_atmphys_vars.F:934-942."""
        self.owner=owner
    def __iter__(self):
        """Declared names; mpas_atmphys_vars.F:68-955."""
        return iter(self.owner.native.abi['variables'])
    def __len__(self):
        """Native declaration count; mpas_atmphys_vars.F:68-955."""
        return len(self.owner.native.abi['variables'])
    def __getitem__(self,name):
        """Contiguous k,i view; mpas_atmphys_vars.F:179-217; mpas_atmphys_vars.F:934-942."""
        return self.owner.prepared(name)


class _ResidentPoolViews(_PoolViews):
    """Full resident entity arrays for a halo-aware slot boundary."""
    def __getitem__(self,name):
        """Resident getters; RRFS fork MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:668-710."""
        return self.owner.field(self.pool,name,resident=True)


class _ProfileViews(_PreparedViews):
    """Driver argument aliases; mpas_atmphys_driver_sfclayer.F:1016-1047."""
    def __init__(self,owner,aliases):
        """Native argument spelling; mpas_atmphys_driver_sfclayer.F:1016-1047."""
        super().__init__(owner);self.aliases=aliases
    def __iter__(self):
        """Declared profile aliases; mpas_atmphys_driver_sfclayer.F:1016-1047."""
        return iter(self.aliases)
    def __len__(self):
        """Profile alias count; mpas_atmphys_driver_sfclayer.F:1016-1047."""
        return len(self.aliases)
    def __getitem__(self,name):
        """Reuse prepared native words; mpas_atmphys_driver_sfclayer.F:1016-1047."""
        return self.owner.prepared(self.aliases[name])


class _ScalarViews(Mapping):
    """Native manager scalar values; mpas_atmphys_vars.F:68-955."""
    def __init__(self,native):
        """Bind native manager globals; mpas_atmphys_vars.F:68-955."""
        self.native=native
    def __iter__(self):
        """Enumerate scalar globals; mpas_atmphys_vars.F:68-955."""
        return iter(n for n,r in self.native.abi['variables'].items() if r['rank']==0)
    def __len__(self):
        """Scalar declaration count; mpas_atmphys_vars.F:68-955."""
        return sum(1 for _ in self)
    def __getitem__(self,name):
        """Copy a native scalar, preserving its dtype; mpas_atmphys_vars.F:68-955."""
        value=self.native.get_variable(name)
        if value is None or value[0].ndim:raise KeyError(name)
        return value[0][()]


class _ConstantViews(Mapping):
    """Native resolved constants; src/core_atmosphere/physics/mpas_atmphys_constants.F:8-103."""
    def __init__(self,native,aliases=None):
        """No host re-evaluation of constant expressions; src/core_atmosphere/physics/mpas_atmphys_constants.F:8-103."""
        self.native=native
        self.aliases=aliases or {}
    def __iter__(self):
        """Provider's native constant names; src/core_atmosphere/physics/mpas_atmphys_constants.F:8-103."""
        return iter(dict.fromkeys((*self.native.abi.get('constants',{}),*self.aliases)))
    def __len__(self):
        """Provider constant count; src/core_atmosphere/physics/mpas_atmphys_constants.F:8-103."""
        return sum(1 for _ in self)
    def __getitem__(self,name):
        """Read compiled native words; src/core_atmosphere/physics/mpas_atmphys_constants.F:8-103."""
        if not callable(getattr(self.native,'get_constant',None)):
            raise RuntimeError('native constant export is absent: CUDA slot constants must match compiled MPAS words')
        return self.native.get_constant(self.aliases.get(name,name))


def _session_landuse(provider):
    """The hydrated static's land-use table name, for Noah's packet; mpas_atmphys_landuse.F:177-227."""
    try:
        value=provider.native.get_text_field('sfc_input','mminlu').strip()
    except (AttributeError,KeyError,RuntimeError,ValueError):
        return None
    return value or None


def register_cuda_slots(registry,config,provider,resources=None):
    """Register only table-selected CUDA callbacks; mpas_atmphys_driver.F:209-384."""
    rows=load_table()[getattr(config,'table_key','mpasphys_v842')]['slots']
    for slot,scheme in config.selectors.items():
        if not config.active(slot) or config.backends[slot]!=Backend.HEX_CUDA:continue
        descriptor=rows[slot].get('cuda_entries',{}).get(scheme)
        if descriptor is None:
            raise RuntimeError(f'MPAS CUDA slot {slot}/{scheme} has no published callback descriptor')
        try:
            module=importlib.import_module('.'+descriptor['module'],__package__)
        except ModuleNotFoundError as exc:
            raise RuntimeError(f'MPAS CUDA slot module {descriptor["module"]} is not in this merged tree') from exc
        callback=getattr(module,descriptor['call'])
        resource=descriptor.get('resources')
        if resource:
            if resources is None:
                # First use fetches (or reuses) the certified prebuilt packet.
                # Framework and scheme tables need no native compiler here.
                from .resources_build import ensure_packet
                needs_mynnedmf=any(selected=='bl_mynnedmf' and config.active(name)
                                  and config.backends[name]==Backend.HEX_CUDA
                                  for name,selected in config.selectors.items())
                resources=ensure_packet(landuse=_session_landuse(provider),
                                        include_mynnedmf=needs_mynnedmf)
            from .resources_build import load_packet
            packets=load_packet(resources)
            provider.resources_packet = resources
            if scheme not in packets:
                raise RuntimeError(f'MPAS CUDA slot {slot}/{scheme}: physics-resources packet {resources} has no '
                                   f'{scheme} entry (gpuwm-hex physics-resources build names what it does not build)')
            loader=importlib.import_module('.'+resource['module'],__package__)
            setattr(provider,resource['attribute'],getattr(loader,resource['call'])(packets[scheme],provider.cp))
        def invoke(call,callback=callback,descriptor=descriptor,slot=slot):
            """Slot copy transaction; mpas_atmphys_driver.F:209-384."""
            # Device memory the slot does not use is given back at its two
            # boundaries (hexcore.device_memory_release; knobs
            # GPUWM_HEX_POOL_RELEASE, GPUWM_HEX_LMEM_RELEASE).
            with phase_release('slot:'+slot):
                prepared=provider.begin(call,descriptor)
                result=callback(replace(call,pools=prepared))
                provider.commit()
            return result
        registry.register(slot,scheme,Backend.HEX_CUDA,invoke,
                          corrections=descriptor.get('corrections',()))
        diagnostics=descriptor.get('post_lsm_diagnostics')
        if diagnostics:
            # MPAS mpas_atmphys_driver.F:313-325; diagnostics after LSM/seaice.
            diagnostic_callback=getattr(module,diagnostics)
            provider.session.hooks.external['atmphys_sfc_diagnostics'] = (
                lambda call,callback=diagnostic_callback: callback(provider,call))
