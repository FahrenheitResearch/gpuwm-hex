"""Build the pristine OML module and literal MPAS driver/init slices locally.

Run with --source MPAS_TREE --output LANE_SCRATCH as make -j4 (through the build queue on a shared host).
The caller must invoke the generated Makefile itself (through the build queue on a shared host), never rebuild
the shared reference. This file only writes inputs; it does not invoke a compiler.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import shutil

FLAGS = ("-O3 -ffree-line-length-none -fconvert=big-endian -ffree-form "
         "-ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations")
SOURCE_PINS = {
    "physics_wrf/module_sf_oml.F": "442ad1d13872c9edf3b39b6ccb6113ba06e7e84aeaffb77350bf132b445b5d87",
    "mpas_atmphys_driver_oml.F": "3bdbb195fe336d94d85bdda0deff7b3e25f44f9cb194f8db4e0736048d3e61d4",
    "mpas_atmphys_init.F": "fd7d2c61ddb650b8ebc9780a076c66af11c9b27a3b7a432004d3bc3488087355",
}


def prepare(source: Path, output: Path) -> None:
    from hexcore.mpasphys.oml_v842 import CUDA_SOURCE

    physics = source / "src/core_atmosphere/physics"
    for relative, digest in SOURCE_PINS.items():
        if hashlib.sha256((physics / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"OML oracle source differs from pinned MPAS-A v8.4.2: {relative}")
    output.mkdir(parents=True, exist_ok=True)
    module = physics / "physics_wrf/module_sf_oml.F"
    driver = physics / "mpas_atmphys_driver_oml.F"
    init = physics / "mpas_atmphys_init.F"
    shutil.copyfile(module, output / "module_sf_oml.F")
    driver_lines = driver.read_text().splitlines()
    init_lines = init.read_text().splitlines()
    driver_slice = "\n".join(driver_lines[122:146])
    init_slice = "\n".join(init_lines[346:381])
    # The sole removed line is framework logging, with no state effect.
    init_slice = "\n".join(line for line in init_slice.splitlines()
                           if "call mpas_log_write" not in line)
    aliases = ["t_oml", "t_oml_initial", "h_oml", "h_oml_initial", "hu_oml", "hv_oml",
               "skintemp", "hfx", "lh", "gsw", "glw", "t_oml_200m_initial", "ust",
               "sfc_emiss", "uReconstructZonal", "uReconstructMeridional", "latCell",
               "xland", "sst"]
    declarations = []
    associations = []
    for index, name in enumerate(aliases, 1):
        if name.startswith("uReconstruct"):
            declarations.append(f"real(c_float), pointer :: {name}(:,:)")
            associations.append(f"{name}(1:1,1:n) => a(:,{index})")
        else:
            declarations.append(f"real(c_float), pointer :: {name}(:)")
            associations.append(f"{name} => a(:,{index})")
    prefix = "\n".join(declarations + associations)
    wrapper = f"""module oml_oracle
use iso_c_binding
use module_sf_oml
implicit none
contains
subroutine replay(n, ns, mode, enabled, restart, a, stbolt, gravity, dt_pbl, &
                  oml_gamma, oml_relaxation_time, omega, config_oml_hml0) bind(c)
integer(c_int), value :: n, ns, mode, enabled, restart
real(c_float), target, intent(inout) :: a(n,19)
real(c_float), value :: stbolt, gravity, dt_pbl, oml_gamma, oml_relaxation_time, omega, config_oml_hml0
logical :: config_oml1d, config_do_restart
integer :: iCell, nCellsSolve
real(c_float) :: f_coriolis
{prefix}
config_oml1d = enabled /= 0
config_do_restart = restart /= 0
nCellsSolve = ns
if (mode == 0) then
do iCell = 1,nCellsSolve
call oml1d(t_oml(iCell),t_oml_initial(iCell),h_oml(iCell),h_oml_initial(iCell), &
hu_oml(iCell),hv_oml(iCell),skintemp(iCell),hfx(iCell),lh(iCell),gsw(iCell),glw(iCell), &
t_oml_200m_initial(iCell),uReconstructZonal(1,iCell),uReconstructMeridional(1,iCell), &
ust(iCell),latCell(iCell),sfc_emiss(iCell),stbolt,gravity,dt_pbl,oml_gamma,oml_relaxation_time)
enddo
else if (mode == 1) then
{driver_slice}
else
{init_slice}
endif
end subroutine
end module
"""
    (output / "oracle.f90").write_text(wrapper)
    # Compile the same arithmetic on the host to distinguish transcription
    # errors from device libm/instruction residue, without a Python data path.
    host_source = CUDA_SOURCE.replace("__device__", "").replace("__global__", "")
    host_source = host_source.replace("blockDim.x*blockIdx.x+threadIdx.x", "oml_host_cell")
    witness = """#include <cmath>
int oml_host_cell;
inline float glibc_sinf(float a) { return sinf(a); }
inline float __fmul_rn(float a,float b) { return a*b; }
inline float __fadd_rn(float a,float b) { return a+b; }
inline float __fsub_rn(float a,float b) { return a-b; }
inline float __fdiv_rn(float a,float b) { return a/b; }
inline float __fsqrt_rn(float a) { return sqrtf(a); }
""" + host_source + """
extern "C" void replay_cpp(int n, int ns, int mode, int enabled, int restart, float *a,
    float stbolt,float gravity,float dt,float gamma,float relax,float omega,float hml0) {
    if ((mode == 1 && !enabled) || (mode == 2 && (!enabled || restart))) return;
    for (int i=0;i<ns;i++) {
        oml_host_cell = i;
        if (mode == 2) {
            oml_init(ns,hml0,a,a+n,a+2*n,a+3*n,a+4*n,a+5*n,a+11*n,a+18*n);
        } else {
            oml_run(ns,mode==1,a,a+n,a+2*n,a+3*n,a+4*n,a+5*n,a+6*n,
                a+7*n,a+8*n,a+9*n,a+10*n,a+11*n,a+12*n,a+13*n,
                a+14*n,a+15*n,1,1,a+16*n,a+17*n,a+18*n,
                stbolt,gravity,dt,gamma,relax,omega);
        }
    }
}
"""
    (output / "witness.cpp").write_text(witness)
    # Compile the device math header unchanged apart from the CUDA qualifier.
    # Intrinsic compatibility maps to IEEE host operations and pinned glibc;
    # it contains no independent elementary-function approximation.
    header = files("gpuwm").joinpath("core/kernels/glibc_trig_flt32.cuh").read_text()
    compatibility = """#include <cstring>
inline float __double2float_rn(double a) { return (float)a; }
inline long long __double2ll_rn(double a) { return llrint(a); }
inline long long __double_as_longlong(double a) { long long r; std::memcpy(&r,&a,8); return r; }
inline double __longlong_as_double(long long a) { double r; std::memcpy(&r,&a,8); return r; }
inline unsigned int __float_as_uint(float a) { unsigned int r; std::memcpy(&r,&a,4); return r; }
inline float __uint_as_float(unsigned int a) { float r; std::memcpy(&r,&a,4); return r; }
inline double __dadd_rn(double a,double b) { return a+b; }
inline double __dsub_rn(double a,double b) { return a-b; }
inline double __dmul_rn(double a,double b) { return a*b; }
inline double __ddiv_rn(double a,double b) { return a/b; }
inline double __dsqrt_rn(double a) { return sqrt(a); }
inline double __fma_rn(double a,double b,double c) { return fma(a,b,c); }
inline float __fmaf_rn(float a,float b,float c) { return fmaf(a,b,c); }
inline unsigned long long __umul64hi(unsigned long long a,unsigned long long b) {
    return (unsigned long long)(((unsigned __int128)a*b)>>64);
}
"""
    shared_witness = witness.replace("inline float glibc_sinf(float a) { return sinf(a); }\n", "")
    at = shared_witness.index(host_source)
    shared_witness = (shared_witness[:at] + compatibility +
                      header.replace("__device__", "") + "\n" + shared_witness[at:])
    (output / "witness_shared_math.cpp").write_text(shared_witness)
    (output / "Makefile").write_text(f"""FC = gfortran
FFLAGS = {FLAGS} -fPIC --coverage
all: liboml_oracle.so liboml_witness.so liboml_shared_math_witness.so
module_sf_oml.o: module_sf_oml.F
\t$(FC) $(FFLAGS) -c $< -o $@
oracle.o: oracle.f90 module_sf_oml.o
\t$(FC) $(FFLAGS) -c $< -o $@
liboml_oracle.so: module_sf_oml.o oracle.o
\t$(FC) --coverage -shared $^ -o $@
liboml_witness.so: witness.cpp
\tg++ -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -fPIC -shared $< -o $@
liboml_shared_math_witness.so: witness_shared_math.cpp
\tg++ -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -fPIC -shared $< -o $@
""")
    (output / "source-pins.json").write_text(json.dumps({
        "flags": FLAGS, "coverage_flags": "-fPIC --coverage",
        "sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (module, driver, init)},
        "driver_slice_lines": [123,146], "init_slice_lines": [347,381],
        "removed": "mpas_log_write (logging only)",
        "cuda_source_sha256": hashlib.sha256(CUDA_SOURCE.encode()).hexdigest(),
        "shared_math_header_sha256": hashlib.sha256(header.encode()).hexdigest(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
