"""Build the pristine MPAS wrapper/preprocessor and its actual driver slice."""
from pathlib import Path
import json
import sys
from hexcore.mpasphys import sf_mynn_v842 as scheme
root=Path(sys.argv[1]);meta=json.loads((root/'metadata.json').read_text());fields=meta['fields']
idx={k:fields.index(k)+1 for k in fields}
(root/'mpas_kinds.F90').write_text('module mpas_kind_types\n integer,parameter :: RKIND=kind(1.0),StrKIND=256\nend module\n')
# The exact caller maps the shared atmospheric profiles before this slot.
args={k:'f(:,'+str(v)+')' for k,v in idx.items() if k not in ('qstar','wstar','rstoch1d','u1d','v1d','t1d','qv1d','p1d','rho1d','u1d2','v1d2','dz8w1d','dz2w1d')}
profiles=('u3d','v3d','t3d','qv3d','qc3d','p3d','rho3d','dz8w')
args.update({k:'profiles(:,:,:,'+str(i+1)+')' for i,k in enumerate(profiles)})
args['th3d']=args['t3d'];args['pi3d']=args['t3d'] # Neither is used in the pristine wrapper.
const=[k for k in meta['scalars'] if k not in ('isfflx','itimestep','spp_pbl','isftcflx','iz0tlnd')]
args.update({k:'constants('+str(i+1)+')' for i,k in enumerate(const)})
args.update(itimestep='step',isfflx='flux',spp_pbl='spp',isftcflx='water',iz0tlnd='land',errmsg='errmsg',errflg='errflg')
for k in ('ids','ims','its','jds','jms','jts','kds','kms','kts','jde','jme','jte'):args[k]='1'
for k in ('ide','ime','ite'):args[k]='n'
for k in ('kde','kme','kte'):args[k]='levels'
code='''subroutine native_wrapper(f,constants,profiles,step,flux,spp,water,land,n,levels) bind(c)
use iso_c_binding
use module_sf_mynn
implicit none
integer(c_int),value :: n,levels,step,flux,spp,water,land
real(c_float),intent(inout) :: f(n,'''+str(len(fields))+''')
real(c_float),intent(in) :: constants('''+str(len(const))+'''),profiles(n,levels,1,8)
character(len=256) :: errmsg
integer :: errflg
call sfclay_mynn( &
'''+', &\n'.join(k+'='+v for k,v in args.items())+''')
end subroutine
'''
# Copy arithmetic statements verbatim from MPAS's conversion loops.
# Pool lookups/storage copies are represented by the field-major C ABI.
source=(Path(sys.argv[2])/'src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F').read_text()
blend_start=source.index('          br(i)     = br_p(i,j)*xice(i)')
blend_end=source.index('\n       endif',blend_start)
blend=source[blend_start:blend_end]
for k in fields:
    blend=blend.replace(k+'_p(i,j)', 'f(i,'+str(idx[k])+')')
    blend=blend.replace(k+'_sea(i,j)', 'sea(i,'+str(idx[k])+')')
    blend=blend.replace(k+'(i)', 'f(i,'+str(idx[k])+')')
blend=blend.replace('t2m_p(i,j)','f(i,'+str(idx['t2'])+')').replace('t2m_sea(i,j)','sea(i,'+str(idx['t2'])+')').replace('t2m(i)','f(i,'+str(idx['t2'])+')')
blend=blend.replace('th2m_p(i,j)','f(i,'+str(idx['th2'])+')').replace('th2m_sea(i,j)','sea(i,'+str(idx['th2'])+')').replace('th2m(i)','f(i,'+str(idx['th2'])+')')
blend=blend.replace('regime_hold(i,j)','hold(i)').replace('xice(i)','ice(i)').replace('_RKIND','_c_float')
# MYNN-specific final conversion is separate from the common slice.
blend+='\n f(i,'+str(idx['ch'])+') = f(i,'+str(idx['ch'])+')*ice(i) + (1._c_float-ice(i))*sea(i,'+str(idx['ch'])+')\n'
code+='''subroutine native_from(f,sea,density,sst,ice,len_disp,threshold,fraction,n) bind(c)
use iso_c_binding
implicit none
integer(c_int),value :: n,fraction
real(c_float),value :: len_disp,threshold
real(c_float),intent(inout) :: f(n,'''+str(len(fields))+'''),sea(n,'''+str(len(fields))+''')
real(c_float),intent(in) :: density(n),sst(n),ice(n)
integer :: i
 do i=1,n
 f(i,'''+str(idx['dx'])+''')=len_disp/density(i)**0.25
 enddo
 if(fraction/=0) then
 sea=f
 sea(:,'''+str(idx['qsfc'])+''')=0._c_float
 do i=1,n
 if(ice(i)>=threshold .and. ice(i)<=1._c_float) then
 sea(i,'''+str(idx['xland'])+''')=2._c_float
 sea(i,'''+str(idx['mavail'])+''')=1._c_float
 sea(i,'''+str(idx['znt'])+''')=0.0001_c_float
 sea(i,'''+str(idx['tsk'])+''')=max(sst(i),271.4_c_float)
 endif
 enddo
 endif
end subroutine
subroutine native_to(f,sea,ice,hold,threshold,n) bind(c)
use iso_c_binding
implicit none
integer(c_int),value :: n
real(c_float),value :: threshold
real(c_float),intent(inout) :: f(n,'''+str(len(fields))+''')
real(c_float),intent(in) :: sea(n,'''+str(len(fields))+'''),ice(n),hold(n)
integer :: i
 do i=1,n
 if(ice(i)>=threshold .and. ice(i)<=1._c_float) then
'''+blend+'''
 endif
 enddo
end subroutine
'''
(root/'driver_bridge.F90').write_text(code)
(root/'Makefile').write_text((root/'Makefile').read_text().split('\nmpas_kinds.o:')[0])
with (root/'Makefile').open('a') as file:file.write('''
mpas_kinds.o: mpas_kinds.F90
\tgfortran $(FFLAGS) -c $<
sf_mynn_pre.o: sf_mynn_pre.F kinds.o
\tgfortran $(FFLAGS) -ffree-form -c $<
module_sf_mynn.o: module_sf_mynn.F sf_mynn_pre.o sf_mynn.o mpas_kinds.o
\tgfortran $(FFLAGS) -ffree-form -c $<
driver_bridge.o: driver_bridge.F90 module_sf_mynn.o
\tgfortran $(FFLAGS) -c $<
driver.so: kinds.o mynn_shared.o sf_mynn.o sf_mynn_pre.o mpas_kinds.o module_sf_mynn.o driver_bridge.o bridge.o
\tgfortran -shared -fprofile-arcs $^ -o $@
''')
cpp=(root/'witness.cpp').read_text()+'\n#define __fsqrt_rn sqrtf\n'+scheme._driver_source()
cpp+='''
extern "C" void witness_driver(float *f,float *sea,const float *profiles,
 const float *density,const float *sst,const float *ice,const float *constants,
 int step,int flux,int spp,int water,int land,int fraction,int n,int levels) {
 float *hold=new float[n];float *qc=new float[n];
 for(int i=0;i<n;++i) hold[i]=f[IDX_regime*n+i];
 blockDim.x=1;blockIdx.x=0;
 for(int i=0;i<n;++i) {
 threadIdx.x=i;
 sf_mynn_from_mpas(f,sea,density,sst,ice,480000.0f,0.01f,fraction,n);
 }
 for(int j=0;j<(fraction?2:1);++j) {
 float *a=j?sea:f;
 for(int i=0;i<n;++i) {
 threadIdx.x=i;
 sf_mynn_pre_columns(a,profiles,profiles+levels*n,profiles+2*levels*n,
 profiles+3*levels*n,profiles+4*levels*n,profiles+5*levels*n,
 profiles+6*levels*n,profiles+7*levels*n,a,qc,step==1,0,n);
 sf_mynn_columns(a,constants,step,flux,0,water,land,127,n);
 }
 }
 if(fraction) for(int i=0;i<n;++i) {
 threadIdx.x=i;
 sf_mynn_to_mpas(f,sea,ice,hold,0.01f,n);
 }
 delete[] hold;delete[] qc;
}
'''
(root/'driver-witness.cpp').write_text(cpp)
with (root/'Makefile').open('a') as file:file.write('''
driver-witness.so: driver-witness.cpp
\tg++ $(CFLAGS) -shared $< -o $@
''')
