// Apache-2.0. Build-only host binding of the already pinned device maths.
// The manylinux 2.28 libm differs from the strict reference. GNU ld wraps
// only the generator objects' external math calls, without changing MPAS.
#include "cuda_host_shim.h"
#include "glibc_flt32.cuh"
#include "glibc_flt64.cuh"
#include "thompson_aerosol_libm.cuh"
extern "C" {
float __wrap_expf(float x) { return thompson_aa_expf(x); }
float __wrap_logf(float x) { return thompson_aa_logf(x); }
float __wrap_powf(float x,float y) { return thompson_aa_powf(x,y); }
double __wrap_exp(double x) { return thompson_aa_exp(x); }
double __wrap_log(double x) { return thompson_aa_log(x); }
double __wrap_pow(double x,double y) { return thompson_aa_pow(x,y); }
double __wrap_log10(double x) { return thompson_aa_log10(x); }
}
