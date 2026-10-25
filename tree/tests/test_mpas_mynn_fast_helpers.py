"""Protect strict helper tokens and execute fast finite arithmetic on the host.

These checks do not establish CUDA performance or the native bitwise gate.
"""
import ctypes
import hashlib
import shutil
import subprocess

import numpy as np
import pytest

from hexcore.mpasphys import bl_mynn_v842 as pbl


def test_strict_numerical_support_preprocesses_to_pinned_tokens():
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('g++ required for preprocessor witness')
    result = subprocess.run([compiler, '-E', '-P', '-x', 'c++',
                             '-D__CUDA_ARCH__=890', '-'],
                            input=pbl.MYNN_ARRAY_SUPPORT, text=True,
                            capture_output=True, check=True, timeout=30)
    # Preprocessed base 0d9a4943, with whitespace removed. The strict
    # workaround, NaN payload handling and descriptor routines all remain.
    digest = hashlib.sha256(''.join(result.stdout.split()).encode()).hexdigest()
    assert digest == '3457bf2178f9f928f73dfe8bdc2fcbc52b83c9cf7df87900f458dc69ed76f033'


@pytest.fixture(scope='module')
def arithmetic_libraries(tmp_path_factory):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('g++ required for arithmetic witness')
    folder = tmp_path_factory.mktemp('mynn-fast-arithmetic')
    preamble = r'''
#include <cstring>
#define __device__
#define __forceinline__ inline
unsigned int __float_as_uint(float a) {unsigned int r;std::memcpy(&r,&a,4);return r;}
float __uint_as_float(unsigned int a) {float r;std::memcpy(&r,&a,4);return r;}
long long __double_as_longlong(double a) {long long r;std::memcpy(&r,&a,8);return r;}
double __longlong_as_double(unsigned long long a) {double r;std::memcpy(&r,&a,8);return r;}
float mpas_add(float a,float b) {volatile float r=a+b;return r;}
float mpas_sub(float a,float b) {volatile float r=a-b;return r;}
float mpas_mul(float a,float b) {volatile float r=a*b;return r;}
float mpas_div(float a,float b) {volatile float r=a/b;return r;}
'''
    support = pbl.MYNN_ARRAY_SUPPORT
    conversions = support[support.index('__device__ __forceinline__ double mynn_real64'):]
    conversions = conversions.split('#ifdef __CUDA_ARCH__\n// MPAS physics_mmm', 1)[0]
    entry = r'''
extern "C" void arithmetic(int n,const float* a,const float* b,float* out,double* wide) {
    for(int i=0;i<n;++i) {
        out[5*i]=mpas_add(a[i],b[i]);out[5*i+1]=mpas_sub(a[i],b[i]);
        out[5*i+2]=mpas_mul(a[i],b[i]);out[5*i+3]=mpas_div(a[i],b[i]);
        wide[i]=mynn_real64(a[i]);out[5*i+4]=mynn_real32(wide[i]);
    }
}
'''
    libraries = []
    for fast in (False, True):
        src, binary = folder / f'{fast}.cpp', folder / f'{fast}.so'
        src.write_text(preamble + pbl.MYNN_IEEE_ARITHMETIC_SUPPORT + conversions + entry)
        command = [compiler, '-std=c++17', '-O2', '-shared', '-fPIC', '-ffp-contract=off',
                   '-D__CUDA_ARCH__=890']
        if fast:
            command.append('-DHEX_NUMERICS_FAST=1')
        subprocess.run([*command, str(src), '-o', str(binary)], check=True,
                       capture_output=True, text=True, timeout=30)
        library = ctypes.CDLL(str(binary))
        library.arithmetic.argtypes = [ctypes.c_int, *([ctypes.c_void_p] * 4)]
        libraries.append(library)
    return libraries


def test_fast_helpers_preserve_finite_words_and_signed_zero(arithmetic_libraries):
    rng = np.random.default_rng(842)
    words = rng.integers(0, 2**32, 4096, dtype=np.uint32)
    finite = words[(words & 0x7f800000) != 0x7f800000].view(np.float32)
    values = np.concatenate((np.array([0., -0., 1., -1., 2**-126, -(2**-126)], np.float32), finite))
    other = np.ascontiguousarray(values[::-1])
    outputs = []
    for library in arithmetic_libraries:
        result, wide = np.empty((values.size, 5), np.float32), np.empty(values.size, np.float64)
        library.arithmetic(values.size, values.ctypes.data, other.ctypes.data,
                           result.ctypes.data, wide.ctypes.data)
        outputs.append((result, wide))
    for strict, fast in zip(outputs[0], outputs[1]):
        # Invalid finite combinations may produce a NaN; only strict mode
        # promises its sign/payload. All other words, including zeros, match.
        np.testing.assert_array_equal(np.isnan(strict), np.isnan(fast))
        mask = ~np.isnan(strict)
        kind = np.uint32 if strict.dtype.itemsize == 4 else np.uint64
        np.testing.assert_array_equal(strict[mask].view(kind), fast[mask].view(kind))
