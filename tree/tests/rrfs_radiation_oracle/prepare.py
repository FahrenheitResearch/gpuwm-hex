"""Copy pristine fork sources and source slices into owned standalone scratch."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ref, out = map(Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
physics = ref / 'src/core_atmosphere/physics'
here = Path(__file__).parent
files = ['src/framework/mpas_constants.F',
         'src/core_atmosphere/physics/mpas_atmphys_constants.F',
         'src/core_atmosphere/physics/mpas_atmphys_functions.F',
         'src/core_atmosphere/physics/mpas_atmphys_driver_cloudiness.F',
         'src/core_atmosphere/physics/physics_wrf/module_mp_thompson_cldfra3.F',
         'src/core_atmosphere/physics/physics_wrf/module_mp_thompson_aerosols.F',
         'src/core_atmosphere/physics/physics_wrf/module_ra_rrtmg_sw_aerosols.F']
pins = []
for name in files:
    source = ref / name
    shutil.copy2(source, out / source.name)
    pins.append({'path':name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
for name in ('framework.f90','wrappers.f90'):
    shutil.copy2(here/name,out/name)
# This slice is copied verbatim, including constants and expression order.
sw = physics / 'physics_wrf/module_ra_rrtmg_sw.F'
text = sw.read_text().splitlines()
start = next(i for i,l in enumerate(text) if 'da=6.2831853071795862' in l)
end = next(i for i in range(start,len(text)) if 'coszr(i,j) = coszrs' in text[i])+1
body = '\n'.join(text[start:end])
source = '''subroutine solar(n,julian,xtime,gmt,radt,degrad,declin,lat,lon,out,eq) bind(C)
 use iso_c_binding
 implicit none
 integer(c_int),value::n
 real(c_float),value::julian,xtime,gmt,radt,degrad,declin
 real(c_float)::lat(n),lon(n),out(n),eq(n)
 real::da,eot,xt24,tloctm,hrang,xxlat,coszrs,xlat(n,1),xlong(n,1),coszr(n,1)
 integer::i,j
 xlat(:,1)=lat;xlong(:,1)=lon;j=1
 do i=1,n
''' + body + '\n eq(i)=eot\n enddo\n out=coszr(:,1)\nend subroutine\n'
(out/'solar_slice.f90').write_text(source)
pins.append({'path':str(sw.relative_to(ref)),'first_line':start+1,'last_line':end,
             'window_sha256':hashlib.sha256(body.encode()).hexdigest()})
lw = physics/'mpas_atmphys_driver_radiation_lw.F'
lines=lw.read_text().splitlines()
windows=[]
for i,line in enumerate(lines):
 if 'case("mp_nssl2m","mp_tempo")' in line:
  end=next(k for k in range(i+1,len(lines)) if 'case default' in lines[k])
  body='\n'.join(l for l in lines[i+1:end] if 'call mpas_pool_get_array' not in l)
  windows.append(body)
  pins.append(dict(path=str(lw.relative_to(ref)),first_line=i+2,last_line=end,
      window_sha256=hashlib.sha256('\n'.join(lines[i+1:end]).encode()).hexdigest()))
source='''subroutine radius(ncol,nlev,enabled,input,diagnosed,out,initial,back) bind(C)
 use iso_c_binding
 implicit none
 integer,parameter::RKIND=4
 integer(c_int),value::ncol,nlev,enabled
 real(c_float)::input(nlev,ncol,3),diagnosed(nlev,ncol,3),out(nlev,ncol,3),initial(nlev,ncol,3),back(nlev,ncol,3)
 real::re_cloud(nlev,ncol),re_ice(nlev,ncol),re_snow(nlev,ncol)
 real::rre_cloud(nlev,ncol),rre_ice(nlev,ncol),rre_snow(nlev,ncol)
 real::recloud_p(ncol,nlev,1),reice_p(ncol,nlev,1),resnow_p(ncol,nlev,1)
 real::rrecloud_p(ncol,nlev,1),rreice_p(ncol,nlev,1),rresnow_p(ncol,nlev,1)
 integer::i,j,k,its,ite,jts,jte,kts,kte
 logical::config_microp_re
 its=1;ite=ncol;jts=1;jte=1;kts=1;kte=nlev;config_microp_re=enabled/=0
 re_cloud=input(:,:,1);re_ice=input(:,:,2);re_snow=input(:,:,3)
''' + windows[0] + '''
 out(:,:,1)=transpose(recloud_p(:,:,1));out(:,:,2)=transpose(reice_p(:,:,1));out(:,:,3)=transpose(resnow_p(:,:,1))
 initial(:,:,1)=transpose(rrecloud_p(:,:,1));initial(:,:,2)=transpose(rreice_p(:,:,1));initial(:,:,3)=transpose(rresnow_p(:,:,1))
 rrecloud_p(:,:,1)=transpose(diagnosed(:,:,1));rreice_p(:,:,1)=transpose(diagnosed(:,:,2));rresnow_p(:,:,1)=transpose(diagnosed(:,:,3))
''' + windows[1] + '''
 back(:,:,1)=rre_cloud;back(:,:,2)=rre_ice;back(:,:,3)=rre_snow
end subroutine
'''
(out/'radius_slice.f90').write_text(source)
(out/'source-pins.json').write_text(json.dumps(pins,indent=2)+'\n')
(out/'Makefile').write_text('''all: librrfs_oracle.so
librrfs_oracle.so: framework.f90 mpas_constants.F mpas_atmphys_constants.F mpas_atmphys_functions.F module_mp_thompson_cldfra3.F mpas_atmphys_driver_cloudiness.F module_mp_thompson_aerosols.F module_ra_rrtmg_sw_aerosols.F wrappers.f90 solar_slice.f90 radius_slice.f90
\tgfortran -O3 -cpp -DSINGLE_PRECISION -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -ffree-form -fconvert=big-endian -fPIC -shared --coverage $^ -o $@
''')
