"""Standalone compiled MPAS constants for pre-f3 binding diagnostics.

Copies complete pinned modules. No Python reevaluation of constant expressions.
This bridge does not repair or replace the published f3 runtime ABI.
"""
from pathlib import Path
import argparse, hashlib, json, shutil
from hexcore.mpasphys import kain_fritsch_v842 as kf
p=argparse.ArgumentParser()
p.add_argument('--mpas-source',type=Path,required=True)
p.add_argument('--build-dir',type=Path,required=True)
a=p.parse_args(); a.build_dir.mkdir(parents=True,exist_ok=True)
sources=['src/framework/mpas_kind_types.F','src/framework/mpas_constants.F',
         'src/core_atmosphere/physics/mpas_atmphys_constants.F']
pins={}
for name in sources:
    source=a.mpas_source/name
    shutil.copyfile(source,a.build_dir/source.name)
    pins[name]=hashlib.sha256(source.read_bytes()).hexdigest()
(a.build_dir/'copy.f90').write_text('''! MPAS-A v8.4.2 mpas_atmphys_constants.F:8-103; copy compiled values only.
subroutine kf_native_constants(values) bind(C)
 use iso_c_binding
 use mpas_atmphys_constants
 implicit none
 real(c_float), intent(out):: values(13)
 values = [xlv0,xlv1,xls0,xls1,cp,r_d,gravity,ep_1,ep_2,svp1,svp2,svp3,svpt0]
end subroutine
''')
(a.build_dir/'Makefile').write_text('''FC=gfortran
FFLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -fconvert=big-endian -fPIC -cpp -DSINGLE_PRECISION -Dmpas -ffree-form -ffree-line-length-none
all: libconstants.so
mpas_kind_types.o: mpas_kind_types.F
\t$(FC) $(FFLAGS) -c $< -o $@
mpas_constants.o: mpas_constants.F mpas_kind_types.o
\t$(FC) $(FFLAGS) -c $< -o $@
mpas_atmphys_constants.o: mpas_atmphys_constants.F mpas_constants.o
\t$(FC) $(FFLAGS) -c $< -o $@
copy.o: copy.f90 mpas_atmphys_constants.o
\t$(FC) $(FFLAGS) -c $< -o $@
libconstants.so: mpas_kind_types.o mpas_constants.o mpas_atmphys_constants.o copy.o
\t$(FC) -shared $^ -o $@
''')
(a.build_dir/'source-pins.json').write_text(json.dumps({'sources':pins,'precision':'RKIND single',
    'native_modules_modified':False,'native_runtime_abi_replaced':False},indent=2)+'\n')
