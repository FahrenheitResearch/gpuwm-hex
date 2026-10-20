#include <cmath>
#include <cstring>
#include <cstdint>
using std::isnan;
#define SDDEV
#define SDKERNEL
#define SDGUARD
inline float __fadd_rn(float a,float b){return a+b;}inline double __dadd_rn(double a,double b){return a+b;}
inline float __fsub_rn(float a,float b){return a-b;}inline double __dsub_rn(double a,double b){return a-b;}
inline float __fmul_rn(float a,float b){return a*b;}inline double __dmul_rn(double a,double b){return a*b;}
inline float __fdiv_rn(float a,float b){return a/b;}inline double __ddiv_rn(double a,double b){return a/b;}
inline float __fsqrt_rn(float a){return sqrtf(a);}inline double __dsqrt_rn(double a){return sqrt(a);}
inline float gfk_exp_fma(float a){return expf(a);}inline float gfk_log(float a){return logf(a);}inline float gfk_pow(float a,float b){return powf(a,b);}
inline double glibc_exp(double a){return exp(a);}inline double glibc_log(double a){return log(a);}inline double glibc_pow(double a,double b){return pow(a,b);}
inline float glibc_sinf(float a){return sinf(a);}inline float glibc_cosf(float a){return cosf(a);}
inline float thompson_aa_log10f(float a){return log10f(a);}inline double thompson_aa_log10(double a){return log10(a);}
inline float smoke_shared_erf(float a){return erff(a);}
inline float __uint_as_float(uint32_t a){float f;memcpy(&f,&a,4);return f;}
inline uint32_t __float_as_uint(float a){uint32_t u;memcpy(&u,&a,4);return u;}
