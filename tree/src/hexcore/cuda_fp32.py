"""Shared guarded FP32 subnormal arithmetic for CUDA source strings.

The sm_120 NVRTC route is certified with ``-ftz=true`` and therefore flushes
FP32 subnormal operands and results.  These helpers keep the ordinary path in
FP32: FP64 decode/arithmetic is entered only for a bit-observed subnormal
operand or a flushed-zero result whose non-zero operands can have produced an
IEEE subnormal.  Defining ``MPAS_FTZ_FALLBACK_ENABLED=0`` compiles the exact
unprotected mutation control.
"""

from __future__ import annotations


CUDA_FTZ_HELPERS = r"""
#ifndef MPAS_FTZ_FALLBACK_ENABLED
#define MPAS_FTZ_FALLBACK_ENABLED 1
#endif

__device__ __forceinline__ unsigned int mpas_f32_magnitude_bits(
    const float value)
{
    return __float_as_uint(value) & 0x7fffffffu;
}

__device__ __forceinline__ bool mpas_is_subnormal_f32(const float value)
{
#if MPAS_FTZ_FALLBACK_ENABLED
    const unsigned int magnitude = mpas_f32_magnitude_bits(value);
    return magnitude != 0u && (magnitude & 0x7f800000u) == 0u;
#else
    return false;
#endif
}

// A plain float/double cast is one of the measured DAZ mechanisms, so decode
// a subnormal float from its integer mantissa instead.
__device__ __forceinline__ double mpas_f32_to_f64_ieee(const float value)
{
#if MPAS_FTZ_FALLBACK_ENABLED
    const unsigned int bits = __float_as_uint(value);
    const unsigned int magnitude = bits & 0x7fffffffu;
    if (magnitude != 0u && (magnitude & 0x7f800000u) == 0u) {
        const double decoded = ldexp(
            (double)(magnitude & 0x007fffffu), -149);
        return (bits & 0x80000000u) != 0u ? -decoded : decoded;
    }
#endif
    return (double)value;
}

// __double2float_rn flushes an FP32-subnormal result on the certified route,
// so pack that narrow result from the rounded integer mantissa.
__device__ __forceinline__ float mpas_f64_to_f32_ieee(const double value)
{
#if MPAS_FTZ_FALLBACK_ENABLED
    const double magnitude = fabs(value);
    if (magnitude > 0.0 && magnitude < 1.1754943508222875e-38) {
        const double scaled = rint(
            magnitude * 7.1362384635297994e+44);  // 2^149
        const unsigned int mantissa = (unsigned int)scaled;
        const unsigned int sign =
            (__double_as_longlong(value) < 0LL) ? 0x80000000u : 0u;
        return __uint_as_float(sign | mantissa);
    }
#endif
    return __double2float_rn(value);
}

__device__ __forceinline__ float mpas_add(
    const float left, const float right)
{
    const float ordinary = left + right;
#if MPAS_FTZ_FALLBACK_ENABLED
    const unsigned int result_magnitude = mpas_f32_magnitude_bits(ordinary);
    const unsigned int result_exponent = result_magnitude >> 23;
    // A result at exponent field >=25 cannot have been rounded differently
    // by an FP32 subnormal operand, which is then below half an ulp.
    if (result_exponent > 24u) return ordinary;
    const unsigned int left_magnitude = mpas_f32_magnitude_bits(left);
    const unsigned int right_magnitude = mpas_f32_magnitude_bits(right);
    const unsigned int left_exponent = left_magnitude >> 23;
    const unsigned int right_exponent = right_magnitude >> 23;
    if ((left_exponent == 0u && left_magnitude != 0u)
            || (right_exponent == 0u && right_magnitude != 0u)) {
        return mpas_f64_to_f32_ieee(
            mpas_f32_to_f64_ieee(left) + mpas_f32_to_f64_ieee(right));
    }
#endif
#if MPAS_FTZ_FALLBACK_ENABLED
    // Two normal binary32 values can cancel into the subnormal range only
    // while both exponent fields are <= 23; above that, one ulp is already
    // at least the smallest normal value.
    if (left_exponent <= 23u && right_exponent <= 23u
            && result_exponent == 0u
            && left_magnitude != right_magnitude
            && left_magnitude != 0u && right_magnitude != 0u) {
        return mpas_f64_to_f32_ieee(
            mpas_f32_to_f64_ieee(left) + mpas_f32_to_f64_ieee(right));
    }
#endif
    return ordinary;
}

__device__ __forceinline__ float mpas_sub(
    const float left, const float right)
{
    const float ordinary = left - right;
#if MPAS_FTZ_FALLBACK_ENABLED
    const unsigned int result_magnitude = mpas_f32_magnitude_bits(ordinary);
    const unsigned int result_exponent = result_magnitude >> 23;
    if (result_exponent > 24u) return ordinary;
    const unsigned int left_magnitude = mpas_f32_magnitude_bits(left);
    const unsigned int right_magnitude = mpas_f32_magnitude_bits(right);
    const unsigned int left_exponent = left_magnitude >> 23;
    const unsigned int right_exponent = right_magnitude >> 23;
    if ((left_exponent == 0u && left_magnitude != 0u)
            || (right_exponent == 0u && right_magnitude != 0u)) {
        return mpas_f64_to_f32_ieee(
            mpas_f32_to_f64_ieee(left) - mpas_f32_to_f64_ieee(right));
    }
#endif
#if MPAS_FTZ_FALLBACK_ENABLED
    if (left_exponent <= 23u && right_exponent <= 23u
            && result_exponent == 0u
            && left_magnitude != right_magnitude
            && left_magnitude != 0u && right_magnitude != 0u) {
        return mpas_f64_to_f32_ieee(
            mpas_f32_to_f64_ieee(left) - mpas_f32_to_f64_ieee(right));
    }
#endif
    return ordinary;
}

__device__ __forceinline__ float mpas_mul(
    const float left, const float right)
{
    const float ordinary = left * right;
#if MPAS_FTZ_FALLBACK_ENABLED
    // A normal finite/overflow result proves neither operand was lost to DAZ
    // and no FTZ result occurred.  Inspect operands only on the rare zero or
    // subnormal-result envelope.
    if ((mpas_f32_magnitude_bits(ordinary) & 0x7f800000u) == 0u) {
        const unsigned int left_magnitude = mpas_f32_magnitude_bits(left);
        const unsigned int right_magnitude = mpas_f32_magnitude_bits(right);
        if (left_magnitude != 0u && right_magnitude != 0u) {
            return mpas_f64_to_f32_ieee(
                mpas_f32_to_f64_ieee(left) * mpas_f32_to_f64_ieee(right));
        }
    }
#endif
    return ordinary;
}

__device__ __forceinline__ float mpas_div(
    const float left, const float right)
{
#if MPAS_FTZ_FALLBACK_ENABLED
    const unsigned int left_magnitude = mpas_f32_magnitude_bits(left);
    const unsigned int right_magnitude = mpas_f32_magnitude_bits(right);
    const unsigned int left_exponent = left_magnitude >> 23;
    const unsigned int right_exponent = right_magnitude >> 23;
    if ((left_exponent == 0u && left_magnitude != 0u)
            || (right_exponent == 0u && right_magnitude != 0u)) {
        return mpas_f64_to_f32_ieee(
            mpas_f32_to_f64_ieee(left) / mpas_f32_to_f64_ieee(right));
    }
#endif
    const float ordinary = left / right;
#if MPAS_FTZ_FALLBACK_ENABLED
    if (left_magnitude != 0u && right_magnitude != 0u
            && left_exponent + 126u <= right_exponent
            && (mpas_f32_magnitude_bits(ordinary) & 0x7f800000u) == 0u) {
        return mpas_f64_to_f32_ieee(
            mpas_f32_to_f64_ieee(left) / mpas_f32_to_f64_ieee(right));
    }
#endif
    return ordinary;
}

__device__ __forceinline__ float mpas_sqrt(const float value)
{
#if MPAS_FTZ_FALLBACK_ENABLED
    if (mpas_is_subnormal_f32(value)) {
        return mpas_f64_to_f32_ieee(sqrt(mpas_f32_to_f64_ieee(value)));
    }
#endif
    return sqrtf(value);
}

__device__ __forceinline__ float mpas_abs(const float value)
{
    return __uint_as_float(__float_as_uint(value) & 0x7fffffffu);
}

__device__ __forceinline__ float mpas_copysign(
    const float magnitude, const float sign)
{
    return __uint_as_float(
        (__float_as_uint(magnitude) & 0x7fffffffu)
        | (__float_as_uint(sign) & 0x80000000u));
}

__device__ __forceinline__ float mpas_min(
    const float left, const float right)
{
#if MPAS_FTZ_FALLBACK_ENABLED
    if (mpas_is_subnormal_f32(left) || mpas_is_subnormal_f32(right)) {
        return mpas_f32_to_f64_ieee(right) < mpas_f32_to_f64_ieee(left)
            ? right : left;
    }
#endif
    return fminf(left, right);
}

__device__ __forceinline__ float mpas_max(
    const float left, const float right)
{
#if MPAS_FTZ_FALLBACK_ENABLED
    if (mpas_is_subnormal_f32(left) || mpas_is_subnormal_f32(right)) {
        return mpas_f32_to_f64_ieee(right) > mpas_f32_to_f64_ieee(left)
            ? right : left;
    }
#endif
    return fmaxf(left, right);
}
"""


__all__ = ["CUDA_FTZ_HELPERS"]
