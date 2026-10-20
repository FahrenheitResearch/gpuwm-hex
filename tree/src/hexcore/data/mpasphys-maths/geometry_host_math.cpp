// Strict deformation geometry: host-independent GNU 2.43 float32 maths.
// SPDX-License-Identifier: Apache-2.0
// The included transcriptions and CORE-MATH file retain their MIT notices.
#include "cuda_host_shim.h"
static inline long long __double2ll_rn(double x) { return (long long)nearbyint(x); }
#include "glibc_trig_flt32.cuh"
extern "C" float cr_atan2f(float, float);
extern "C" float hex_sinf(float x) { return glibc_sinf(x); }
extern "C" float hex_cosf(float x) { return glibc_cosf(x); }
extern "C" float hex_asinf(float x) { return glibc_asinf(x); }
extern "C" float hex_atan2f(float y, float x) { return cr_atan2f(y, x); }
