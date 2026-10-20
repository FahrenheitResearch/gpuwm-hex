#include <cmath>
#include <cstring>
static unsigned int __float_as_uint(float a) {
unsigned int u; std::memcpy(&u,&a,sizeof u); return u;
}
static float __uint_as_float(unsigned int u) {
float a; std::memcpy(&a,&u,sizeof a); return a;
}
#define __device__
#define __global__
#define FADD(a,b) ((float)((float)(a)+(float)(b)))
#define FSUB(a,b) ((float)((float)(a)-(float)(b)))
#define FMUL(a,b) ((float)((float)(a)*(float)(b)))
#define FDIV(a,b) ((float)((float)(a)/(float)(b)))
#define FSQRT(a) sqrtf(a)
#define mm5_log logf
#define mm5_exp expf
#define mm5_pow powf
#define mm5_atan atanf
struct Dim { int x; };
static Dim blockIdx={0}, blockDim={1}, threadIdx={0};
