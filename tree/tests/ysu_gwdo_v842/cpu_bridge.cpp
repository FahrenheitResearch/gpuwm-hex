// Test-only host-glibc witness for production CUDA body. No physics translation.
// MPAS bl_gwdo.F90:55-643 and mpas_atmphys_driver_gwdo.F:525-550,668-689.
#include <cmath>
#define __device__
#define __global__
#define __forceinline__ inline
struct launch_dim { int x; };
static launch_dim blockDim,blockIdx,threadIdx;
static inline int atomicExch(int *p,int value) { int old=*p;*p=value;return old; }
static inline float mpas_add(float a,float b) { return a+b; }
static inline float mpas_sub(float a,float b) { return a-b; }
static inline float mpas_mul(float a,float b) { return a*b; }
static inline float mpas_div(float a,float b) { return a/b; }
#define mpas_sqrt sqrtf
#define mpas_abs fabsf
#define mpas_min fminf
#define mpas_max fmaxf
#define gfk_pow powf
#define td_atan2f atan2f
#define td_nint lroundf
using std::isfinite;
#include "kernel_body.inc"

extern "C" void ysu_gwdo_bridge(int nlev,int ncol,float dt,float len_disp,
 const float *u,const float *v,const float *t,const float *q,const float *p,const float *prsi,
 const float *exner,const float *z,const float *ru,const float *rv,const float *md,
 const float *var,const float *con,const float *oa1,const float *oa2,const float *oa3,const float *oa4,
 const float *ol1,const float *ol2,const float *ol3,const float *ol4,float *outputs,int *err)
{
 int words=nlev*ncol;
 *err=0;blockDim.x=ncol;blockIdx.x=0;
 for(int cell=0;cell<ncol;++cell) {
  threadIdx.x=cell;
  bl_ysu_gwdo_v842_f32(nlev,ncol,dt,len_disp,9.80616f,1004.5f,287.0f,461.6f/287.0f-1.0f,3.141592653589793f,
   u,v,t,q,p,prsi,exner,z,ru,rv,md,var,con,oa1,oa2,oa3,oa4,ol1,ol2,ol3,ol4,
   outputs,outputs+words,outputs+2*words,outputs+3*words,outputs+6*words,outputs+6*words+ncol,
   outputs+4*words,outputs+5*words,err);
 }
}

// MPAS module_bl_gwdo.F:177,205; native wrapper's resolved dx and constants.
extern "C" void ysu_gwdo_bridge_dx(int nlev,int ncol,float dt,float g,float cp,float rd,float fv,float pi,
 const float *u,const float *v,const float *t,const float *q,const float *p,const float *prsi,
 const float *exner,const float *z,const float *ru,const float *rv,const float *dx,
 const float *var,const float *con,const float *oa1,const float *oa2,const float *oa3,const float *oa4,
 const float *ol1,const float *ol2,const float *ol3,const float *ol4,float *outputs,int *err)
{
 int words=nlev*ncol;
 *err=0;blockDim.x=ncol;blockIdx.x=0;
 for(int cell=0;cell<ncol;++cell) {
  threadIdx.x=cell;
  bl_ysu_gwdo_v842_native_dx_f32(nlev,ncol,dt,1.0f,g,cp,rd,fv,pi,
   u,v,t,q,p,prsi,exner,z,ru,rv,dx,var,con,oa1,oa2,oa3,oa4,ol1,ol2,ol3,ol4,
   outputs,outputs+words,outputs+2*words,outputs+3*words,outputs+6*words,outputs+6*words+ncol,
   outputs+4*words,outputs+5*words,err);
 }
}
