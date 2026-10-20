"""Generate complete CUDA text and the strict host-glibc witness from one AST.

This is offline build tooling. The backend does not parse Fortran or run any
field arithmetic in Python. All 31 source routines are translated together.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import translate

ROOT = Path(__file__).resolve().parent
TARGET = ROOT.parents[1] / "src/hexcore/mpasphys/rrfs/mp_nssl2m.py"
PINNED_SHA = "06bd80040c47a9613cb4a528f961d947404363b10043402040f8a1298e42edd4"

STATE = ("th", "qv", "qc", "qr", "qi", "qs", "qh", "qhl", "ccw", "crw", "cci", "csw", "chw", "chl", "cn", "vhw", "vhl")
THIRD_MOMENTS = ("zrw", "zhw", "zhl")
ENVIRONMENT = ("dn", "p", "pii", "dz", "w")
SURFACE = ("rainnc", "rainncv", "snownc", "snowncv", "grplnc", "grplncv", "sr")
DIAGNOSTICS = ("dbz", "re_cloud", "re_ice", "re_snow")


def load_target():
    spec = importlib.util.spec_from_file_location("nssl_port", TARGET)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entry_source(globals_, routines):
    # RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:498-503 and 750-776.
    init = routines["nssl_2mom_init"]
    init_values = {"ipctmp": "(moments==3 ? 8:5)", "mixphase": "0", "ihvol": "1", "myrank": "0", "mpiroot": "0"}
    values = []
    for name in init.arguments:
        if name in init_values:
            values.append(init_values[name])
        else:
            assert init.variables[name].optional
            values.append("{}")
    code = ["// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:498-503 at 1d6e370e.",
            'extern "C" __device__ void nssl_initialize_body(unsigned char* parameters,unsigned char* storage,unsigned long long* used,const double* overrides,const int* indices,int moments,int count,int* errors) {',
            "NsslArena arena{storage,0}; NsslColumn column{{},arena}; column.allocate_globals();",
            "for (int i=0;i<count;++i) { switch (indices[i]) {"]
    scalar_names = [v.name for v in globals_.values() if not v.dimensions and not v.parameter]
    for index, name in enumerate(scalar_names):
        code.append(f"case {index}: column.{name} = ({globals_[name].kind})overrides[i]; break;")
    code.extend(["} }", "column.f_nssl_2mom_init(" + ",".join(values) + ");",
                 "*reinterpret_cast<NsslState*>(parameters)=column; *used=arena.offset; *errors=column.native_error;", "}"])
    signature = "int nc,int nz,int nt,int nmicro,int diagnose,int dbz_due,int reqc,int reqi,int reqs,float dt,float* state,const float* environment,float* surface,float* diagnostics,float* refl1km,float* rainprod,float* evapprod,unsigned char* parameters,unsigned char* workspace,unsigned long long workspace_stride,int* errors"
    code.extend(["// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:728-776 at 1d6e370e.",
                 'extern "C" __device__ void nssl_step_body(int active,int workspace_column,' + signature + ") {",
                 "NsslArena arena{workspace+workspace_stride*workspace_column,0}; NsslColumn column{{},arena}; column.copy_globals(*reinterpret_cast<const NsslState*>(parameters));"])
    supplied = {}
    for buffer, names in (("state", STATE), ("environment", ENVIRONMENT), ("diagnostics", DIAGNOSTICS)):
        for index, name in enumerate(names):
            # Native input-only arrays are still described by a mutable view;
            # the original routine's INTENT contract determines their writes.
            ptr = f"const_cast<float*>({buffer})+active+{index}*nc*nz"
            code.append(f"NsslArray<float,3> a_{name}; a_{name}.data={ptr}; a_{name}.lo[0]=a_{name}.lo[1]=a_{name}.lo[2]=1; a_{name}.hi[0]=a_{name}.hi[2]=1; a_{name}.hi[1]=nz; a_{name}.stride[0]=1; a_{name}.stride[1]=nc; a_{name}.stride[2]=nc*nz;")
            supplied[name] = "a_" + name
    for index, name in enumerate(THIRD_MOMENTS, len(STATE)):
        code.append(f"NsslArray<float,3> a_{name}; if(column.ipconc==8) {{ a_{name}=a_th; a_{name}.data=state+active+{index}*nc*nz; }}")
        supplied[name] = "a_" + name
    for index, name in enumerate(SURFACE):
        code.append(f"NsslArray<float,2> a_{name}; a_{name}.data=surface+active+{index}*nc; a_{name}.lo[0]=a_{name}.lo[1]=a_{name}.hi[0]=a_{name}.hi[1]=1; a_{name}.stride[0]=1; a_{name}.stride[1]=nc;")
        supplied[name] = "a_" + name
    name = "refl_diagnostic"
    code.append(f"NsslArray<float,2> a_{name}; a_{name}.data=refl1km+active; a_{name}.lo[0]=a_{name}.lo[1]=a_{name}.hi[0]=a_{name}.hi[1]=1; a_{name}.stride[0]=1; a_{name}.stride[1]=nc;")
    supplied[name] = "a_" + name
    for name in ("rainprod", "evapprod"):
        code.append(f"NsslArray<float,3> a_{name}=a_th; a_{name}.data={name}+active;")
        supplied[name] = "a_" + name
    supplied.update(dtp="dt", itimestep="nt", ntmul="nmicro", ntcnt="1", lastloop="true", diagflag="(diagnose!=0)", diag_dbz="(dbz_due!=0)", has_reqc="reqc", has_reqi="reqi", has_reqs="reqs")
    for dim in ("ids", "ide", "ims", "ime", "its", "ite", "jds", "jde", "jms", "jme", "jts", "jte", "kds", "kms", "kts"):
        supplied[dim] = "1"
    for dim in ("kde", "kme", "kte"):
        supplied[dim] = "nz"
    driver = routines["nssl_2mom_driver"]
    values = []
    for name in driver.arguments:
        if name not in supplied:
            assert driver.variables[name].optional, name
        values.append(supplied.get(name, "{}"))
    code.extend(["column.f_nssl_2mom_driver(" + ",".join(values) + ");", "errors[active]=column.native_error;", "}"])
    code.extend([
        '// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_mp_nssl_2mom.F:174-1102 at 1d6e370e; module storage.',
        'extern "C" __global__ void nssl_sizes(unsigned long long* result) {result[0]=sizeof(NsslState);}',
        '// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:498-503 at 1d6e370e.',
        'extern "C" __global__ void nssl_initialize(unsigned char* parameters,unsigned char* storage,unsigned long long* used,const double* overrides,const int* indices,int moments,int count,int* errors) {nssl_initialize_body(parameters,storage,used,overrides,indices,moments,count,errors);}',
        '// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:728-770 at 1d6e370e.',
        'extern "C" __global__ void nssl_step(int first,int count,' + signature + ') {int thread=blockIdx.x*blockDim.x+threadIdx.x; int active=first+thread; if(thread<count) nssl_step_body(active,thread,nc,nz,nt,nmicro,diagnose,dbz_due,reqc,reqi,reqs,dt,state,environment,surface,diagnostics,refl1km,rainprod,evapprod,parameters,workspace,workspace_stride,errors);}',
    ])
    return "\n".join(code) + "\n", scalar_names


HOST_MATH = r'''
#include <cmath>
#include <cstring>
#define __device__
#define __noinline__
#define __global__
struct NsslCudaDim { int x=0; }; NsslCudaDim blockIdx,blockDim,threadIdx;
inline unsigned int __float_as_uint(float x){unsigned int u;std::memcpy(&u,&x,4);return u;}
inline float __uint_as_float(unsigned int u){float x;std::memcpy(&x,&u,4);return x;}
inline float __fadd_rn(float a,float b){return a+b;}
inline float __fsub_rn(float a,float b){return a-b;}
inline float __fmul_rn(float a,float b){return a*b;}
inline float __fdiv_rn(float a,float b){return a/b;}
inline double __dadd_rn(double a,double b){return a+b;}
inline double __dsub_rn(double a,double b){return a-b;}
inline double __dmul_rn(double a,double b){return a*b;}
inline double __ddiv_rn(double a,double b){return a/b;}
inline float __fsqrt_rn(float a){return std::sqrt(a);}
inline double __dsqrt_rn(double a){return std::sqrt(a);}
inline float nssl_exp(float a){return std::exp(a);}
inline double nssl_exp(double a){return std::exp(a);}
inline float nssl_log(float a){return std::log(a);}
inline double nssl_log(double a){return std::log(a);}
inline float nssl_log10(float a){return std::log10(a);}
inline double nssl_log10(double a){return std::log10(a);}
inline float nssl_pow(float a,float b){return std::pow(a,b);}
inline double nssl_pow(double a,double b){return std::pow(a,b);}
inline double nssl_pow(double a,float b){return std::pow(a,(double)b);}
inline double nssl_pow(float a,double b){return std::pow((double)a,b);}
inline float nssl_pow(float a,int b){return std::pow(a,(float)b);}
inline double nssl_pow(double a,int b){return std::pow(a,(double)b);}
inline float nssl_atan(float a){return std::atan(a);}
inline double nssl_atan(double a){return std::atan(a);}
inline float nssl_sin(float a){return std::sin(a);}
inline double nssl_sin(double a){return std::sin(a);}
inline float nssl_tanh(float a){return std::tanh(a);}
inline double nssl_tanh(double a){return std::tanh(a);}
'''


def prepare(source, out, receipt, host_out=None, constant_words=None):
    assert hashlib.sha256(source.read_bytes()).hexdigest() == PINNED_SHA
    words = None
    if constant_words:
        folded = json.loads(constant_words.read_text())
        assert folded["source_sha256"] == PINNED_SHA
        words = folded["words"]
        globals_, routines_ = translate.parse_modules([source])
        expected = {scope + "." + var.name
                    for scope, variables in [("module", globals_)] + [(r.name, r.variables) for r in routines_.values()]
                    for var in variables.values() if var.parameter and var.kind in ("float", "double")}
        assert set(words) == expected, "native constant observer must cover every floating parameter"
    text, routines = translate.generate([source], out, receipt, words)
    globals_, _ = translate.parse_modules([source])
    entry, scalar_names = entry_source(globals_, routines)
    namelist = next(stmt.split("/",2)[2] for line,stmt in translate.statements(source)
                    if stmt.startswith("namelist /nssl_mp_params/"))
    namelist_names = tuple(translate.split_top(namelist))
    assert all(not globals_[n].dimensions for n in namelist_names)
    module = load_target()
    marker = "# BEGIN GENERATED RRFS NSSL\n"
    prefix = TARGET.read_text().split(marker, 1)[0]
    TARGET.write_text(prefix + marker + "MODULE_SOURCE = r'''\n" + text + "'''\nENTRY_SOURCE = r'''\n" + entry + "'''\nGLOBAL_SCALARS = " + repr(tuple(scalar_names)) + "\nNSSL_NAMELIST_FIELDS = " + repr(namelist_names) + "\n# END GENERATED RRFS NSSL\n")
    if host_out:
        inspect = ['extern "C" const void* nssl_host_variable(unsigned char* p,int id,unsigned long long* count,int* width) {',
                   'const NsslState& s=*reinterpret_cast<const NsslState*>(p); switch(id) {']
        for index, var in enumerate(globals_.values()):
            if var.parameter or var.kind not in ("float", "double", "int"):
                continue
            pointer, count = (f"s.{var.name}.data", f"s.{var.name}.size()") if var.dimensions else (f"&s.{var.name}", "1")
            inspect.append(f"case {index}: *count={count}; *width=sizeof({var.kind}); return {pointer};")
        inspect += ['default: *count=0; *width=0; return nullptr;', '} }']
        powers = ['extern "C" void nssl_host_powers(int n,const float* x,const double* y,float* a,double* b) {',
                  'for(int i=0;i<n;++i) {']
        for exponent in range(2,10):
            powers += [f'a[({exponent}-2)*n+i]=nssl_ipow<{exponent}>(x[i]);',
                       f'b[({exponent}-2)*n+i]=nssl_ipow<{exponent}>(y[i]);']
        powers += ['} }']
        host_out.write_text(HOST_MATH + module.ARRAY_SUPPORT + module.OPERATOR_SUPPORT + text + entry + '\nextern "C" unsigned long long nssl_host_parameter_bytes(){return sizeof(NsslState);}\n' + '\n'.join(inspect+powers) + '\n')
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--host-out", type=Path)
    parser.add_argument("--constant-words", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.out, args.receipt, args.host_out, args.constant_words)
