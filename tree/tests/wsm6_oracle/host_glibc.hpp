// Test-only host execution of the actual CUDA bodies. Never a runtime backend.
#include <cmath>
#include <complex>
#define __device__
#define __global__
namespace thrust {template<class T> using complex=std::complex<T>;}
struct TestIndex {int x;};
static TestIndex blockIdx{0},blockDim{1},threadIdx{0};
inline float __fadd_rn(float a,float b){return a+b;}
inline float __fsub_rn(float a,float b){return a-b;}
inline float __fmul_rn(float a,float b){return a*b;}
inline float __fdiv_rn(float a,float b){return a/b;}
inline double __dadd_rn(double a,double b){return a+b;}
inline double __dsub_rn(double a,double b){return a-b;}
inline double __dmul_rn(double a,double b){return a*b;}
inline double __ddiv_rn(double a,double b){return a/b;}

#include <cstring>
inline unsigned int __float_as_uint(float x){unsigned int v;std::memcpy(&v,&x,4);return v;}
inline float __uint_as_float(unsigned int x){float v;std::memcpy(&v,&x,4);return v;}
inline long long __double_as_longlong(double x){long long v;std::memcpy(&v,&x,8);return v;}
inline double __longlong_as_double(unsigned long long x){double v;std::memcpy(&v,&x,8);return v;}
