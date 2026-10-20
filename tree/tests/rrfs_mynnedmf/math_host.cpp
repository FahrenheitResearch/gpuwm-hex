// Oracle calls only the current host glibc binary32 libm functions.
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <fenv.h>
#include <gnu/libc-version.h>
extern "C" const char* oracle_glibc_version() { return gnu_get_libc_version(); }
extern "C" void oracle_math(int op,uint32_t n,const uint32_t* input,const uint32_t* other,uint32_t* out) {
#pragma omp parallel
 {
  fesetround(FE_TONEAREST);
#pragma omp for
  for(uint32_t i=0;i<n;i++){
   float x,y,z;memcpy(&x,input+i,4);memcpy(&y,other+i,4);
   switch(op){case 0:z=expf(x);break;case 1:z=logf(x);break;case 2:z=log10f(x);break;case 3:z=atanf(x);break;case 4:z=tanhf(x);break;case 5:z=erff(x);break;default:z=powf(x,y);break;}
   memcpy(out+i,&z,4);
  }
 }
}
