"""Native GNU positive powf evaluation for MPAS single precision.

Adapted from Arm optimized-routines v21.02 math/powf.c, powf_log2_data.c
and exp2f_data.c. Copyright (c) 2017-2019, Arm Limited. SPDX-License-Identifier: MIT.
Source: https://github.com/ARM-software/optimized-routines/tree/v21.02/math
The positive normal input path uses the same tables and operation order as
GNU libc 2.39 powf. Other inputs retain the existing RKIND fallback.
"""
import numpy as np
from .rkind_libm import _powf_correctly_rounded

_LOG = np.array([float.fromhex("0x1.661ec79f8f3bep+0"),float.fromhex("-0x1.efec65b963019p-2"),float.fromhex("0x1.571ed4aaf883dp+0"),float.fromhex("-0x1.b0b6832d4fca4p-2"),float.fromhex("0x1.49539f0f010bp+0"),float.fromhex("-0x1.7418b0a1fb77bp-2"),float.fromhex("0x1.3c995b0b80385p+0"),float.fromhex("-0x1.39de91a6dcf7bp-2"),float.fromhex("0x1.30d190c8864a5p+0"),float.fromhex("-0x1.01d9bf3f2b631p-2"),float.fromhex("0x1.25e227b0b8eap+0"),float.fromhex("-0x1.97c1d1b3b7afp-3"),float.fromhex("0x1.1bb4a4a1a343fp+0"),float.fromhex("-0x1.2f9e393af3c9fp-3"),float.fromhex("0x1.12358f08ae5bap+0"),float.fromhex("-0x1.960cbbf788d5cp-4"),float.fromhex("0x1.0953f419900a7p+0"),float.fromhex("-0x1.a6f9db6475fcep-5"),float.fromhex("0x1p+0"),float.fromhex("0x0p+0"),float.fromhex("0x1.e608cfd9a47acp-1"),float.fromhex("0x1.338ca9f24f53dp-4"),float.fromhex("0x1.ca4b31f026aap-1"),float.fromhex("0x1.476a9543891bap-3"),float.fromhex("0x1.b2036576afce6p-1"),float.fromhex("0x1.e840b4ac4e4d2p-3"),float.fromhex("0x1.9c2d163a1aa2dp-1"),float.fromhex("0x1.40645f0c6651cp-2"),float.fromhex("0x1.886e6037841edp-1"),float.fromhex("0x1.88e9c2c1b9ff8p-2"),float.fromhex("0x1.767dcf5534862p-1"),float.fromhex("0x1.ce0a44eb17bccp-2")], dtype=np.float64).reshape(16,2)

_POLY = np.array([float.fromhex("0x1.27616c9496e0bp-2"),float.fromhex("-0x1.71969a075c67ap-2"),float.fromhex("0x1.ec70a6ca7baddp-2"),float.fromhex("-0x1.7154748bef6c8p-1"),float.fromhex("0x1.71547652ab82bp0")], dtype=np.float64)

_EXP = np.array([0x3ff0000000000000,0x3fefd9b0d3158574,0x3fefb5586cf9890f,0x3fef9301d0125b51,0x3fef72b83c7d517b,0x3fef54873168b9aa,0x3fef387a6e756238,0x3fef1e9df51fdee1,0x3fef06fe0a31b715,0x3feef1a7373aa9cb,0x3feedea64c123422,0x3feece086061892d,0x3feebfdad5362a27,0x3feeb42b569d4f82,0x3feeab07dd485429,0x3feea47eb03a5585,0x3feea09e667f3bcd,0x3fee9f75e8ec5f74,0x3feea11473eb0187,0x3feea589994cce13,0x3feeace5422aa0db,0x3feeb737b0cdc5e5,0x3feec49182a3f090,0x3feed503b23e255d,0x3feee89f995ad3ad,0x3feeff76f2fb5e47,0x3fef199bdd85529c,0x3fef3720dcef9069,0x3fef5818dcfba487,0x3fef7c97337b9b5f,0x3fefa4afa2a490da,0x3fefd0765b6e4540], dtype=np.uint64)

def powf_native_gnu(base, exponent):
    array = np.asarray(base)
    if array.dtype != np.float32:
        return _powf_correctly_rounded(array, exponent)
    flat = np.ascontiguousarray(array).reshape(-1)
    ix = flat.view(np.uint32)
    valid = (ix >= 0x00800000) & (ix < 0x7f800000)
    if not np.all(valid):
        out = _powf_correctly_rounded(flat, exponent).copy()
        if np.any(valid):out[valid] = powf_native_gnu(flat[valid], exponent)
        return out.reshape(array.shape)
    tmp = ix - np.uint32(0x3f330000)
    index = (tmp >> 19) % 16
    top = tmp & np.uint32(0xff800000)
    z = (ix - top).view(np.float32).astype(np.float64)
    k = top.view(np.int32) >> 23
    r = z * _LOG[index,0] - 1.0
    y0 = _LOG[index,1] + k.astype(np.float64)
    r2 = r*r
    r4 = r2*r2
    y = _POLY[0]*r + _POLY[1]
    p = _POLY[2]*r + _POLY[3]
    q = _POLY[4]*r + y0
    q = p*r2 + q
    y = y*r4 + q
    xd = np.float64(np.float32(exponent))*y
    if np.any(np.abs(xd)>1000):
        return _powf_correctly_rounded(array,exponent)
    shift = float.fromhex('0x1.8p+47')
    kd = xd + shift
    ki = kd.view(np.uint64)
    r = xd - (kd-shift)
    t = _EXP[ki%32] + (ki << np.uint64(47))
    scale = t.view(np.float64)
    z = float.fromhex('0x1.c6af84b912394p-5')*r + float.fromhex('0x1.ebfce50fac4f3p-3')
    r2 = r*r
    y = float.fromhex('0x1.62e42ff0c52d6p-1')*r + 1.0
    y = z*r2 + y
    return (y*scale).astype(np.float32).reshape(array.shape)

CUDA_NATIVE_POWF = r"""
#define N 32
__constant__ double logtab[16][2] = {
  { 0x1.661ec79f8f3bep+0, -0x1.efec65b963019p-2 * 1 },
  { 0x1.571ed4aaf883dp+0, -0x1.b0b6832d4fca4p-2 * 1 },
  { 0x1.49539f0f010bp+0, -0x1.7418b0a1fb77bp-2 * 1 },
  { 0x1.3c995b0b80385p+0, -0x1.39de91a6dcf7bp-2 * 1 },
  { 0x1.30d190c8864a5p+0, -0x1.01d9bf3f2b631p-2 * 1 },
  { 0x1.25e227b0b8eap+0, -0x1.97c1d1b3b7afp-3 * 1 },
  { 0x1.1bb4a4a1a343fp+0, -0x1.2f9e393af3c9fp-3 * 1 },
  { 0x1.12358f08ae5bap+0, -0x1.960cbbf788d5cp-4 * 1 },
  { 0x1.0953f419900a7p+0, -0x1.a6f9db6475fcep-5 * 1 },
  { 0x1p+0, 0x0p+0 * 1 },
  { 0x1.e608cfd9a47acp-1, 0x1.338ca9f24f53dp-4 * 1 },
  { 0x1.ca4b31f026aap-1, 0x1.476a9543891bap-3 * 1 },
  { 0x1.b2036576afce6p-1, 0x1.e840b4ac4e4d2p-3 * 1 },
  { 0x1.9c2d163a1aa2dp-1, 0x1.40645f0c6651cp-2 * 1 },
  { 0x1.886e6037841edp-1, 0x1.88e9c2c1b9ff8p-2 * 1 },
  { 0x1.767dcf5534862p-1, 0x1.ce0a44eb17bccp-2 * 1 },
  };
__constant__ double logpoly[5] = {
  0x1.27616c9496e0bp-2 * 1, -0x1.71969a075c67ap-2 * 1,
  0x1.ec70a6ca7baddp-2 * 1, -0x1.7154748bef6c8p-1 * 1,
  0x1.71547652ab82bp0 * 1,
  };
__constant__ unsigned long long exptab[32] = {
#if N == 8
0x3ff0000000000000, 0x3fef72b83c7d517b, 0x3fef06fe0a31b715, 0x3feebfdad5362a27,
0x3feea09e667f3bcd, 0x3feeace5422aa0db, 0x3feee89f995ad3ad, 0x3fef5818dcfba487,
#elif N == 16
0x3ff0000000000000, 0x3fefb5586cf9890f, 0x3fef72b83c7d517b, 0x3fef387a6e756238,
0x3fef06fe0a31b715, 0x3feedea64c123422, 0x3feebfdad5362a27, 0x3feeab07dd485429,
0x3feea09e667f3bcd, 0x3feea11473eb0187, 0x3feeace5422aa0db, 0x3feec49182a3f090,
0x3feee89f995ad3ad, 0x3fef199bdd85529c, 0x3fef5818dcfba487, 0x3fefa4afa2a490da,
#elif N == 32
0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51,
0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069,
0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
#elif N == 64
0x3ff0000000000000, 0x3fefec9a3e778061, 0x3fefd9b0d3158574, 0x3fefc74518759bc8,
0x3fefb5586cf9890f, 0x3fefa3ec32d3d1a2, 0x3fef9301d0125b51, 0x3fef829aaea92de0,
0x3fef72b83c7d517b, 0x3fef635beb6fcb75, 0x3fef54873168b9aa, 0x3fef463b88628cd6,
0x3fef387a6e756238, 0x3fef2b4565e27cdd, 0x3fef1e9df51fdee1, 0x3fef1285a6e4030b,
0x3fef06fe0a31b715, 0x3feefc08b26416ff, 0x3feef1a7373aa9cb, 0x3feee7db34e59ff7,
0x3feedea64c123422, 0x3feed60a21f72e2a, 0x3feece086061892d, 0x3feec6a2b5c13cd0,
0x3feebfdad5362a27, 0x3feeb9b2769d2ca7, 0x3feeb42b569d4f82, 0x3feeaf4736b527da,
0x3feeab07dd485429, 0x3feea76f15ad2148, 0x3feea47eb03a5585, 0x3feea23882552225,
0x3feea09e667f3bcd, 0x3fee9fb23c651a2f, 0x3fee9f75e8ec5f74, 0x3fee9feb564267c9,
0x3feea11473eb0187, 0x3feea2f336cf4e62, 0x3feea589994cce13, 0x3feea8d99b4492ed,
0x3feeace5422aa0db, 0x3feeb1ae99157736, 0x3feeb737b0cdc5e5, 0x3feebd829fde4e50,
0x3feec49182a3f090, 0x3feecc667b5de565, 0x3feed503b23e255d, 0x3feede6b5579fdbf,
0x3feee89f995ad3ad, 0x3feef3a2b84f15fb, 0x3feeff76f2fb5e47, 0x3fef0c1e904bc1d2,
0x3fef199bdd85529c, 0x3fef27f12e57d14b, 0x3fef3720dcef9069, 0x3fef472d4a07897c,
0x3fef5818dcfba487, 0x3fef69e603db3285, 0x3fef7c97337b9b5f, 0x3fef902ee78b3ff6,
0x3fefa4afa2a490da, 0x3fefba1bee615a27, 0x3fefd0765b6e4540, 0x3fefe7c1819e90d8,
#endif
  };
#define MUL(a,b) __dmul_rn((a),(b))
#define ADD(a,b) __dadd_rn((a),(b))
#define SUB(a,b) __dsub_rn((a),(b))
#ifdef USE_FMA
#define MA(a,b,c) __fma_rn((a),(b),(c))
#else
#define MA(a,b,c) ADD(MUL((a),(b)),(c))
#endif
__device__ float mpas_native_powf_positive(float x,float power) {
 unsigned int ix=__float_as_uint(x);
 if(ix<0x00800000u || ix>=0x7f800000u) return __double2float_rn(pow((double)x,(double)power));
 unsigned int tmp=ix-0x3f330000u;
 unsigned int i=(tmp>>(23-4))%16,top=tmp&0xff800000u,iz=ix-top;
 int k=((int)top)>>23;
 double z=(double)__uint_as_float(iz);
 double r=MA(z,logtab[i][0],-1.0), y0=ADD(logtab[i][1],(double)k);
 double r2=MUL(r,r),r4=MUL(r2,r2);
 double y=MA(logpoly[0],r,logpoly[1]);
 double p=MA(logpoly[2],r,logpoly[3]);
 double q=MA(logpoly[4],r,y0);
 q=MA(p,r2,q); y=MA(y,r4,q);
 double xd=MUL((double)power,y),shift=0x1.8p+47;
 double kd=ADD(xd,shift);
 unsigned long long ki=__double_as_longlong(kd);
 kd=SUB(kd,shift);r=SUB(xd,kd);
 unsigned long long t=exptab[ki%32]+(ki<<47);
 double scale=__longlong_as_double(t);
 z=MA(0x1.c6af84b912394p-5,r,0x1.ebfce50fac4f3p-3);
 r2=MUL(r,r);y=MA(0x1.62e42ff0c52d6p-1,r,1.0);
 y=MA(z,r2,y);
 return __double2float_rn(MUL(y,scale));
}

#undef N
#undef MUL
#undef ADD
#undef SUB
#undef MA
"""
