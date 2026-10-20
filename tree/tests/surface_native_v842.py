"""Compile pristine pinned MPAS surface modules with pointer-pool stubs.

Only framework lookup is stubbed. Arithmetic routines are copied unchanged.
The caller invokes this build as make -j4 (through the build queue on a shared host).
"""
from pathlib import Path
import hashlib
import os
import ctypes
import numpy as np

SOURCE = Path(os.environ.get('MPAS_REF', 'MPAS-Model-v8.4.2')) / 'src/core_atmosphere/physics'
REAL = ('shdmin shdmax sfc_albbck sst snow tmn skintemp vegfra xice seaice snowc snowh xland '
        'sfc_albedo sfc_emiss sfc_emibck xicem sstsk sstsk_dtc sstsk_dtw glw gsw hfx qfx ust '
        'nsteps_accum ndays_accum tday_accum tyear_accum tyear_mean cuprec rainc rainnc rainncv').split()
RAD = [b+d+l+c for b in ('sw','lw') for d in ('dn','up') for l in ('b','t') for c in ('','c')]
REAL += RAD + ['ac'+k for k in RAD]
INT = ['landmask','isltyp','ivgtyp','i_rainc','i_rainnc']+['i_ac'+k for k in RAD]
TWO = ['albedo12m','greenfrac','tslb','sh2o','smois','tlag']

# Every carrier touched by each pristine routine, including prior output state.
SURFACE_FIELDS = {
    1: ['landmask','albedo12m','greenfrac','shdmin','shdmax','sfc_albbck','vegfra'],
    2: ('landmask isltyp ivgtyp vegfra sfc_albbck sst tmn skintemp tslb sh2o smois '
        'snow snowc snowh seaice xice xland sfc_albedo sfc_emiss sfc_emibck xicem').split(),
    3: ('sst skintemp xland sstsk sstsk_dtc sstsk_dtw sfc_emiss glw gsw hfx qfx ust').split(),
    4: ('nsteps_accum ndays_accum tday_accum tyear_accum tyear_mean tlag tmn skintemp').split(),
    5: RAD + ['ac'+k for k in RAD] + ['i_ac'+k for k in RAD],
    6: ['cuprec', 'rainc', 'i_rainc'],
    7: ['rainncv', 'rainnc', 'i_rainnc'],
}


def replay_fields(arrays):
    """Expose contiguous caller-owned buffers; Rust fills native file words."""
    rf, ii, monthly, soil, lag = arrays
    fields = {key: rf[:, i] for i, key in enumerate(REAL)}
    fields.update({key: ii[:, i] for i, key in enumerate(INT)})
    fields.update({key: monthly[:, :, i] for i, key in enumerate(TWO[:2])})
    fields.update({key: soil[:, :, i] for i, key in enumerate(TWO[2:5])})
    fields['tlag'] = lag
    return fields


def surface_reader(library):
    """Bind the compiled Rust decoder; Python supplies caller-owned buffers."""
    reader = ctypes.CDLL(str(library)).read_surface_field
    reader.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                       ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                       ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    reader.restype = ctypes.c_int
    return reader


def load_real_surface_fields(arrays, paths, library, mode, *, first=0, record=0):
    """Fill every touched carrier, refusing missing inputs without zero fill.

    Multiple native files may supply static, restart and diagnostic carriers.
    Verify the selected global cell IDs before combining their carriers.
    Decoding and dimension/type checks are Rust operations.
    """
    reader = surface_reader(library)
    reference = None
    for path in paths:
        ids = np.empty(arrays[0].shape[0], dtype='i4')
        rc = reader(os.fsencode(path), b'indexToCellID', 4, None, 1, first,
                    ids.size, record, ids.ctypes.data, ids.size)
        if rc:
            raise ValueError(f'Rust native surface reader: {path}:indexToCellID code {rc}')
        if reference is not None and not np.array_equal(reference, ids):
            raise ValueError(f'native surface files have different cell order: {path}')
        reference = ids
    fields = replay_fields(arrays)
    sources = {}
    for key in SURFACE_FIELDS[mode]:
        output = fields[key]
        axis = None if output.ndim == 1 else (
            b'nMonths' if key in TWO[:2] else b'nLags' if key == 'tlag' else b'nSoilLevels')
        levels = 1 if axis is None else output.shape[0]
        for path in paths:
            rc = reader(os.fsencode(path), key.encode(), 4 if key in INT else 5,
                        axis, levels, first, arrays[0].shape[0], record,
                        output.ctypes.data, output.size)
            if rc == 0:
                sources[key] = str(path)
                break
            if rc != -49:  # Only NC_ENOTVAR permits trying another supplied file.
                raise ValueError(f'Rust native surface reader: {path}:{key} code {rc}')
        else:
            raise ValueError(f'native surface input missing {key}; no synthetic replacement')
    return sources


def load_real_land_use(paths, library):
    """Read native scalar land-use categories used by the SST oracle."""
    reader = surface_reader(library)
    values, sources = {}, {}
    for key in ('isice_lu', 'iswater_lu'):
        output = np.empty(1, dtype='i4')
        for path in paths:
            rc = reader(os.fsencode(path), key.encode(), 4, None, 1, 0, 1, 0,
                        output.ctypes.data, 1)
            if rc == 0:
                values[key] = int(output[0])
                sources[key] = str(path)
                break
            if rc != -49:
                raise ValueError(f'Rust native land-use reader: {path}:{key} code {rc}')
        else:
            raise ValueError(f'native surface input missing {key}; no synthetic replacement')
    return values, sources


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    pins = {}
    for name in ('mpas_atmphys_update_surface.F','mpas_atmphys_date_time.F','mpas_atmphys_update.F'):
        data = (SOURCE/name).read_bytes()
        pins[name] = hashlib.sha256(data).hexdigest()
        (directory/name).write_bytes(data)
    for name in ('mpas_atmphys_driver_convection.F','mpas_atmphys_driver_microphysics.F','mpas_atmphys_manager.F'):
        pins[name] = hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
    conv = (SOURCE/'mpas_atmphys_driver_convection.F').read_text()
    routine = conv[conv.index(' subroutine update_convection_step2('):conv.index(' end subroutine update_convection_step2')+len(' end subroutine update_convection_step2')]
    (directory/'rain.f90').write_text('module rain_slice\nuse mpas_kind_types\nuse mpas_pool_routines\nuse mpas_atmphys_vars\ncontains\n'+routine+'\nend module\n')
    micro = (SOURCE/'mpas_atmphys_driver_microphysics.F').read_text()
    # Verbatim accumulated-rain slice; other microphysics diagnostics belong to p01/p02.
    slice_start = micro.index('    !accumulated precipitation:',micro.index(' subroutine precip_to_MPAS('))
    slice_end = micro.index('\n enddo',slice_start)
    rainnc = micro[slice_start:slice_end]
    (directory/'rainnc.f90').write_text('''module rainnc_slice
use mpas_kind_types
use mpas_atmphys_vars
contains
subroutine native_rainnc(n,rainnc,rainncv,i_rainnc,config_bucket_rainnc)
integer::n,i
real(RKIND):: rainnc(n),rainncv(n),config_bucket_rainnc
integer:: i_rainnc(n)
do i=1,n
'''+rainnc+'\nenddo\nend subroutine\nend module\n')
    base = '''module mpas_kind_types
use iso_c_binding
integer,parameter:: RKIND=c_float,StrKIND=64
end module
module mpas_log
contains
subroutine mpas_log_write(message)
character(*)::message
end subroutine
end module
module mpas_dmpar
type dm_info
integer::unused
end type
end module
module mpas_atmphys_constants
use mpas_kind_types
real(RKIND),parameter::stbolt=5.67051e-8
end module
module mpas_atmphys_vars
use mpas_kind_types
real(RKIND)::xice_threshold,dt_dyn
logical::l_acrain,l_acradt
integer::replay_isice=24,replay_iswater=16
end module
module mpas_atmphys_driver_convection
end module
module mpas_pool_routines
use mpas_kind_types
type mpas_pool_type
integer,pointer::nCellsSolve,nSoilLevels,nLags,isice,iswater
real(RKIND),pointer::config_bucket_radt,config_bucket_rainc
'''
    base += ''.join(f'real(RKIND),pointer::{k}(:)\n' for k in REAL)
    base += ''.join(f'integer,pointer::{k}(:)\n' for k in INT)
    base += ''.join(f'real(RKIND),pointer::{k}(:,:)\n' for k in TWO)
    base += '''end type
type block_type
type(block_type),pointer::next=>null()
end type
type domain_type
type(block_type),pointer::blocklist=>null()
end type
interface mpas_pool_get_array
module procedure get_r,get_i,get_r2,get_scalar
end interface
contains
'''
    groups = [('get_r','real(RKIND)',REAL,'(:)'),('get_i','integer',INT,'(:)'),
              ('get_r2','real(RKIND)',TWO,'(:,:)'),('get_scalar','integer',['isice','iswater'],'')]
    for fn,typ,keys,dim in groups:
        base += f'subroutine {fn}(p,name,a)\ntype(mpas_pool_type)::p\ncharacter(*)::name\n{typ},pointer::a{dim}\nselect case(name)\n'
        base += ''.join(f"case('{k}');a=>p%{k}\n" for k in keys)
        base += 'case default;error stop "unknown array"\nend select\nend subroutine\n'
    for fn,typ,keys in [('mpas_pool_get_dimension','integer',['nCellsSolve','nSoilLevels','nLags']),
                       ('mpas_pool_get_config','real(RKIND)',['config_bucket_radt','config_bucket_rainc'])]:
        base += f'subroutine {fn}(p,name,a)\ntype(mpas_pool_type)::p\ncharacter(*)::name\n{typ},pointer::a\nselect case(name)\n'
        base += ''.join(f"case('{k}');a=>p%{k}\n" for k in keys)
        base += 'case default;error stop "unknown scalar"\nend select\nend subroutine\n'
    base += 'end module\n'
    (directory/'stubs.f90').write_text(base)
    wrapper = f'''subroutine run_surface(mode,n,ns,nl,flag,nsolve,dt,jul,threshold,bucket,rf,ii,monthly,soil,lag,date) bind(c)
use iso_c_binding
use mpas_kind_types
use mpas_pool_routines
use mpas_dmpar
use mpas_atmphys_vars
use mpas_atmphys_update_surface
use mpas_atmphys_update
use rain_slice
use rainnc_slice
integer(c_int),value::mode,n,ns,nl,flag,nsolve
real(c_float),value::dt,jul,threshold,bucket
real(c_float),target::rf(n,{len(REAL)}),monthly(12,n,2),soil(ns,n,3),lag(nl,n)
integer(c_int),target::ii(n,{len(INT)})
character(c_char)::date(64)
character(64)::stamp
integer,target::nn,nss,nll,ice,water
real(RKIND),target::b
integer::k
type(mpas_pool_type)::p
type(dm_info)::dm
nn=nsolve;nss=ns;nll=nl;ice=replay_isice;water=replay_iswater;b=bucket
p%nCellsSolve=>nn;p%nSoilLevels=>nss;p%nLags=>nll;p%isice=>ice;p%iswater=>water
p%config_bucket_radt=>b;p%config_bucket_rainc=>b
'''
    wrapper += ''.join(f'p%{k}=>rf(:,{i+1})\n' for i,k in enumerate(REAL))
    wrapper += ''.join(f'p%{k}=>ii(:,{i+1})\n' for i,k in enumerate(INT))
    wrapper += 'p%albedo12m=>monthly(:,:,1);p%greenfrac=>monthly(:,:,2)\n'
    wrapper += 'p%tslb=>soil(:,:,1);p%sh2o=>soil(:,:,2);p%smois=>soil(:,:,3);p%tlag=>lag\n'
    wrapper += '''do k=1,64
stamp(k:k)=date(k)
enddo
xice_threshold=threshold;dt_dyn=dt;l_acrain=flag/=0;l_acradt=flag/=0
select case(mode)
case(1);call physics_update_surface(stamp,flag/=0,p,p)
case(2);call physics_update_sst(dm,flag/=0,p,p,p)
case(3);call physics_update_sstskin(dt,p,p,p)
case(4);call physics_update_deepsoiltemp(flag/=0,dt,jul,p,p,p)
case(5);call update_radiation_diagnostics(p,p,p,1,nsolve)
case(6);call update_convection_step2(p,p,1,nsolve)
case(7);call native_rainnc(nsolve,p%rainnc,p%rainncv,p%i_rainnc,b)
end select
end subroutine
subroutine set_surface_land_use(ice,water) bind(c)
use iso_c_binding
use mpas_atmphys_vars
integer(c_int),value::ice,water
replay_isice=ice;replay_iswater=water
end subroutine
'''
    (directory/'wrapper.f90').write_text(wrapper)
    (directory/'Makefile').write_text('''FC=gfortran
FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -fPIC -ffree-form -ffree-line-length-none
all: oracle.so
oracle.so: stubs.f90 mpas_atmphys_date_time.F mpas_atmphys_update_surface.F mpas_atmphys_update.F rain.f90 rainnc.f90 wrapper.f90
	$(FC) $(FLAGS) -shared $^ -o $@
''')
    return pins


def call(lib, mode, arrays, flag=1, dt=120., jul=100., threshold=.02, bucket=100., date='2001-03-31_12:00:00', nsolve=None, *, isice=24, iswater=16):
    rf,ii,monthly,soil,lag=arrays
    setter=getattr(lib,'set_surface_land_use',None)
    if setter is not None:
        setter.argtypes=[ctypes.c_int,ctypes.c_int]
        setter.restype=None
        setter(isice,iswater)
    elif (isice,iswater)!=(24,16):
        raise ValueError('rebuild the standalone framework wrapper for native land-use metadata')
    fn=lib.run_surface
    fn.argtypes=[ctypes.c_int]*6+[ctypes.c_float]*4+[ctypes.c_void_p]*6
    stamp=ctypes.create_string_buffer(date.ljust(64).encode())
    fn(mode,rf.shape[0],soil.shape[0],lag.shape[0],flag,nsolve or rf.shape[0],dt,jul,threshold,bucket,
       *[a.ctypes.data for a in arrays],ctypes.cast(stamp,ctypes.c_void_p))


if __name__ == '__main__':
    import argparse
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(prepare(args.directory),indent=2)+'\n')
