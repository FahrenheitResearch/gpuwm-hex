// Test-only host glibc witness. No CUDA fallback or scientific Python path.
#include <cmath>
#include <cstdint>
#include <cstring>
#define __device__
#define __global__
#define __constant__
#define __forceinline__ inline
struct Dim3 { int x; };
static Dim3 blockIdx, blockDim, threadIdx;
inline float __fadd_rn(float a,float b){return a+b;}
inline float __fsub_rn(float a,float b){return a-b;}
inline float __fmul_rn(float a,float b){return a*b;}
inline double __dadd_rn(double a,double b){return a+b;}
inline double __dsub_rn(double a,double b){return a-b;}
inline double __dmul_rn(double a,double b){return a*b;}
inline float __fdiv_rn(float a,float b){return a/b;}
inline double __ddiv_rn(double a,double b){return a/b;}
inline float __fsqrt_rn(float a){return sqrtf(a);}
inline double __dsqrt_rn(double a){return sqrt(a);}
inline uint32_t __float_as_uint(float a){uint32_t v;std::memcpy(&v,&a,4);return v;}
inline float __uint_as_float(uint32_t a){float v;std::memcpy(&v,&a,4);return v;}
inline long long __double_as_longlong(double a){long long v;std::memcpy(&v,&a,8);return v;}
inline double __longlong_as_double(long long a){double v;std::memcpy(&v,&a,8);return v;}
inline float mpas_expf(float a){return expf(a);}
inline double mpas_exp(double a){return exp(a);}
inline float mpas_logf(float a){return logf(a);}
inline double mpas_log(double a){return log(a);}
inline float mpas_log10f(float a){return log10f(a);}
inline double mpas_log10(double a){return log10(a);}
inline float mpas_powf(float a,float b){return powf(a,b);}
inline double mpas_pow(double a,double b){return pow(a,b);}
inline float mpas_atanf(float a){return atanf(a);}
inline double mpas_atan(double a){return atan(a);}
inline double mpas_atan2(double a,double b){return atan2(a,b);}
inline double mpas_hypot(double a,double b){return hypot(a,b);}
