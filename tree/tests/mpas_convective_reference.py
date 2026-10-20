"""Standalone pristine v8.4.2 module replay; build with make -j4 (through HEX_BUILD_QUEUE on a shared host).

Framework stubs provide pools, streams and log signatures only. The complete
MPAS diagnostic module is copied unchanged; no numerical Fortran is rewritten.
The optional C++ host build checks the CUDA source with live host libm, and is
explicitly a debugging tier, never a substitute for CUDA/whole-model proof.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

import numpy as np

STUBS = r'''
module mpas_kind_types
 integer,parameter :: RKIND=kind(1.0),R8KIND=kind(1.0d0)
end module
module mpas_derived_types
 type MPAS_pool_type
 integer :: unused
 end type
 type MPAS_clock_type
 integer :: unused
 end type
 integer,parameter :: MPAS_LOG_ERR=1,MPAS_LOG_CRIT=2
 integer,parameter :: MPAS_STREAM_OUTPUT=1,MPAS_STREAM_INPUT=2,MPAS_STREAM_INPUT_OUTPUT=3
end module
module reference_storage
 use mpas_kind_types
 use mpas_derived_types
 implicit none
 integer,target :: nc,nl,ns,nv,degree,np1,iqv=1
 integer :: write_mask=0,include_mask=0
 real,pointer :: zz(:,:),ww(:,:),uu(:,:),vv(:,:),th(:,:),qq(:,:,:),ex(:,:),pb(:,:),pp(:,:),vo(:,:),ki(:,:)
 real,pointer :: area(:),fields(:,:)
 integer,pointer :: cov(:,:)
 type(MPAS_pool_type),target :: pool_mesh,pool_state,pool_diag
 contains
 integer function field_index(name)
 character(*),intent(in) :: name
 select case(name)
 case('cape');field_index=1
 case('cin');field_index=2
 case('lcl');field_index=3
 case('lfc');field_index=4
 case('srh_0_1km');field_index=5
 case('srh_0_3km');field_index=6
 case('uzonal_surface');field_index=7
 case('uzonal_1km');field_index=8
 case('uzonal_6km');field_index=9
 case('umeridional_surface');field_index=10
 case('umeridional_1km');field_index=11
 case('umeridional_6km');field_index=12
 case('temperature_surface');field_index=13
 case('dewpoint_surface');field_index=14
 case('updraft_helicity_max');field_index=15
 case('w_velocity_max');field_index=16
 case('wind_speed_level1_max');field_index=17
 case default;field_index=0
 end select
 end function
end module
module mpas_constants
 use mpas_kind_types
 real(kind=RKIND),parameter :: rvord=461.6_RKIND/287.0_RKIND
end module
module mpas_log
 contains
 subroutine mpas_log_write(message,messageType,intArgs,realArgs,logicArgs)
 character(*) :: message
 integer,optional :: messageType,intArgs(:)
 real,optional :: realArgs(:)
 logical,optional :: logicArgs(:)
 end subroutine
end module
module mpas_atm_diagnostics_utils
 use reference_storage
 contains
 logical function MPAS_field_will_be_written(name)
 character(*) :: name
 MPAS_field_will_be_written=btest(write_mask,field_index(name)-1)
 end function
 integer function mpas_stream_inclusion_count(name,direction)
 character(*) :: name
 integer,optional :: direction
 mpas_stream_inclusion_count=0
 if(btest(include_mask,field_index(name)-1)) mpas_stream_inclusion_count=1
 end function
end module
module mpas_pool_routines
 use reference_storage
 implicit none
 interface mpas_pool_get_array
 module procedure get1,get2,get3,geti2
 end interface
 contains
 subroutine mpas_pool_get_subpool(parent,name,result)
 type(MPAS_pool_type),pointer :: parent,result
 character(*) :: name
 select case(name)
 case('mesh');result=>pool_mesh
 case('state');result=>pool_state
 case('diag');result=>pool_diag
 end select
 end subroutine
 subroutine mpas_pool_get_dimension(parent,name,result)
 type(MPAS_pool_type),pointer :: parent
 character(*) :: name
 integer,pointer :: result
 select case(name)
 case('nCells');result=>nc
 case('nCellsSolve');result=>ns
 case('nVertLevels');result=>nl
 case('nVertLevelsP1');result=>np1
 case('nVertices');result=>nv
 case('vertexDegree');result=>degree
 case('index_qv');result=>iqv
 case default;stop 77
 end select
 end subroutine
 subroutine get1(parent,name,result,time)
 type(MPAS_pool_type),pointer :: parent
 character(*) :: name
 real,pointer :: result(:)
 integer,optional :: time
 if(name=='areaCell') then
 result=>area
 else
 result=>fields(:,field_index(name))
 endif
 end subroutine
 subroutine get2(parent,name,result,time)
 type(MPAS_pool_type),pointer :: parent
 character(*) :: name
 real,pointer :: result(:,:)
 integer,optional :: time
 select case(name)
 case('zgrid');result=>zz
 case('w');result=>ww
 case('uReconstructZonal');result=>uu
 case('uReconstructMeridional');result=>vv
 case('theta_m');result=>th
 case('exner');result=>ex
 case('relhum');result=>ex ! dereferenced but unused by native
 case('pressure_base');result=>pb
 case('pressure_p');result=>pp
 case('vorticity');result=>vo
 case('kiteAreasOnVertex');result=>ki
 case default;stop 78
 end select
 end subroutine
 subroutine get3(parent,name,result,time)
 type(MPAS_pool_type),pointer :: parent
 character(*) :: name
 real,pointer :: result(:,:,:)
 integer,optional :: time
 result=>qq
 end subroutine
 subroutine geti2(parent,name,result,time)
 type(MPAS_pool_type),pointer :: parent
 character(*) :: name
 integer,pointer :: result(:,:)
 integer,optional :: time
 result=>cov
 end subroutine
end module
'''
BRIDGE = r'''
subroutine reference_run(nc_in,ns_in,nl_in,nv_in,deg_in,imask,wmask,action, &
                         z_in,w_in,u_in,v_in,th_in,q_in,ex_in,pb_in,pp_in, &
                         area_in,cov_in,ki_in,vo_in,out) bind(C)
 use iso_c_binding
 use reference_storage
 use mpas_convective_diagnostics
 implicit none
 integer(c_int),value :: nc_in,ns_in,nl_in,nv_in,deg_in,imask,wmask,action
 real(c_float),target :: z_in(nl_in+1,nc_in),w_in(nl_in+1,nc_in)
 real(c_float),target :: u_in(nl_in,nc_in),v_in(nl_in,nc_in),th_in(nl_in,nc_in)
 real(c_float),target :: q_in(1,nl_in,nc_in),ex_in(nl_in,nc_in),pb_in(nl_in,nc_in),pp_in(nl_in,nc_in)
 real(c_float),target :: area_in(nc_in),ki_in(deg_in,nv_in),vo_in(nl_in,nv_in),out(nc_in,17)
 integer(c_int),target :: cov_in(deg_in,nv_in)
 type(MPAS_pool_type),target :: allpool
 type(MPAS_clock_type),target :: simclock
 type(MPAS_pool_type),pointer :: ap
 type(MPAS_clock_type),pointer :: sc
 nc=nc_in;ns=ns_in;nl=nl_in;np1=nl+1;nv=nv_in;degree=deg_in
 include_mask=imask;write_mask=wmask
 zz=>z_in;ww=>w_in;uu=>u_in;vv=>v_in;th=>th_in;qq=>q_in
 ex=>ex_in;pb=>pb_in;pp=>pp_in;area=>area_in;ki=>ki_in;vo=>vo_in;cov=>cov_in;fields=>out
 ap=>allpool;sc=>simclock
 call convective_diagnostics_setup(ap,sc)
 select case(action)
 case(1);call convective_diagnostics_compute()
 case(2);call convective_diagnostics_update()
 case(3);call convective_diagnostics_reset()
 end select
end subroutine
'''
HOST_PRELUDE = r'''
#include <cmath>
#include <algorithm>
#define __device__
#define __global__
inline float __fadd_rn(float a,float b) {return a+b;}
inline float __fsub_rn(float a,float b) {return a-b;}
inline float __fmul_rn(float a,float b) {return a*b;}
inline float __fdiv_rn(float a,float b) {return a/b;}
inline float __fsqrt_rn(float a) {return sqrtf(a);}
inline float gfk_log(float a) {return logf(a);}
inline float gfk_exp_fma(float a) {return expf(a);}
inline float gfk_pow(float a,float b) {return powf(a,b);}
struct Index {int x;};
Index blockDim={1},blockIdx={0},threadIdx={0};
'''
HOST_BRIDGE = r'''
extern "C" void host_run(int nc,int ns,int nl,int nv,int degree,int imask,int wmask,int action,
 const float* z,const float* w,const float* u,const float* v,const float* theta,const float* qv,
 const float* ex,const float* pb,const float* pp,const float* area,const int* cov,
 const float* kite,const float* vort,float* out) {
 float* work=new float[ns*(17*nl+3)];float* uh=new float[(nc+1)*nl];
 if(action==1&&(wmask&16383)) for(int c=0;c<ns;c++) {
   blockIdx.x=c;compute_diag(nc,ns,nl,wmask,461.6f/287.0f,z,u,v,theta,qv,ex,pb,pp,out,work);
 }
 if(action==2) {
   if(imask&(1<<14)) for(int k=0;k<nl;k++) {
     blockIdx.x=k;vertex_helicity(nc,nl,nv,degree,cov,kite,vort,uh);
   }
   for(int c=0;c<ns;c++) {blockIdx.x=c;update_diag(nc,ns,nl,imask,z,w,u,v,area,uh,out);}
 }
 if(action==3) for(int c=0;c<nc;c++) {blockIdx.x=c;reset_diag(nc,wmask,out);}
 delete[] work;delete[] uh;
}
'''
DEVICE_HOST_INTRINSICS = r'''
#include <cstring>
inline double __dadd_rn(double a,double b) {return a+b;}
inline double __dsub_rn(double a,double b) {return a-b;}
inline double __dmul_rn(double a,double b) {return a*b;}
inline double __ddiv_rn(double a,double b) {return a/b;}
inline double __fma_rn(double a,double b,double c) {return fma(a,b,c);}
inline float __fmaf_rn(float a,float b,float c) {return fmaf(a,b,c);}
inline float __double2float_rn(double a) {return (float)a;}
inline unsigned int __float_as_uint(float x) {unsigned int y;memcpy(&y,&x,4);return y;}
inline float __uint_as_float(unsigned int x) {float y;memcpy(&y,&x,4);return y;}
inline float __int_as_float(int x) {float y;memcpy(&y,&x,4);return y;}
inline long long __double_as_longlong(double x) {long long y;memcpy(&y,&x,8);return y;}
inline double __longlong_as_double(long long x) {double y;memcpy(&y,&x,8);return y;}
'''


FLAGS = '-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations'


def prepare(directory: Path, source: Path, coverage=False):
    from hexcore.mpasdiag.convective_v842 import CUDA_SOURCE, device_source
    directory.mkdir(parents=True, exist_ok=True)
    original = source.read_bytes()
    (directory / 'mpas_convective_diagnostics.F').write_bytes(original)
    (directory / 'stubs.f90').write_text(STUBS)
    (directory / 'bridge.f90').write_text(BRIDGE)
    (directory / 'port.cpp').write_text(HOST_PRELUDE + CUDA_SOURCE + HOST_BRIDGE)
    import re
    prelude = re.sub(r'inline float gfk_.*?\n', '', HOST_PRELUDE)
    (directory / 'device_math_host.cpp').write_text(prelude+DEVICE_HOST_INTRINSICS+device_source()+HOST_BRIDGE)
    coverage_flags = ' --coverage' if coverage else ''
    (directory / 'Makefile').write_text(f'''all: native.so host.so host_math.so
native.so: stubs.f90 mpas_convective_diagnostics.F bridge.f90
\tgfortran {FLAGS}{coverage_flags} -fPIC -shared -ffree-form -ffree-line-length-none -cpp $^ -o $@
host.so: port.cpp
\tg++ {FLAGS} -fPIC -shared -std=c++17 $< -o $@
host_math.so: device_math_host.cpp
\tg++ {FLAGS} -fPIC -shared -std=c++17 $< -o $@
''')
    (directory / 'provenance.json').write_text(json.dumps({
        'source_path': str(source), 'source_sha256': hashlib.sha256(original).hexdigest(),
        'flags': FLAGS+coverage_flags, 'precision': 'RKIND=4,R8KIND=8', 'module_copy': 'pristine',
        'host_port_tier': 'debug only, live libm substituted for device libm',
    }, indent=2)+'\n')


class Reference:
    def __init__(self, directory: Path, host=False):
        libname = 'host_math.so' if host and os.environ.get('MPAS_CONVECTIVE_HOST_DEVICE_MATH')=='1' else ('host.so' if host else 'native.so')
        self.library = ctypes.CDLL(str(directory/libname))
        self.run = self.library.host_run if host else self.library.reference_run
        self.run.argtypes = [ctypes.c_int]*8+[ctypes.c_void_p]*14
        self.run.restype = None

    def __call__(self, arrays, out, ns, include, written, action):
        from hexcore.mpasdiag.convective_v842 import _mask
        nc, nl = arrays['theta_m'].shape
        nv, deg = arrays['cellsOnVertex'].shape
        names = ('zgrid','w','uReconstructZonal','uReconstructMeridional','theta_m','qv',
                 'exner','pressure_base','pressure_p','areaCell','cellsOnVertex',
                 'kiteAreasOnVertex','vorticity')
        inputs = [arrays[n] for n in names]+[out]
        assert all(a.flags.c_contiguous for a in inputs)
        self.run(nc,ns,nl,nv,deg,_mask(include),_mask(written),action,
                 *(a.ctypes.data for a in inputs))
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--source', type=Path, default=Path(os.environ.get('MPAS_REF', str(Path(__file__).resolve().parents[1] / 'vendor' / 'MPAS-Model-v8.4.2'))) / 'src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F')
    parser.add_argument('--coverage', action='store_true')
    args = parser.parse_args()
    prepare(args.directory, args.source, args.coverage)
