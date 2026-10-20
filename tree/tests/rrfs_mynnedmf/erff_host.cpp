// Exhaustive oracle calls the running host's erff, never an approximation.
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <fenv.h>
#include <gnu/libc-version.h>
extern "C" const char* oracle_glibc_version() { return gnu_get_libc_version(); }
extern "C" void oracle_erff(uint64_t begin, uint32_t n, uint32_t* out) {
#pragma omp parallel
  {
    fesetround(FE_TONEAREST);
#pragma omp for
    for (uint32_t i=0; i<n; ++i) {
      uint32_t bits=uint32_t(begin+i); float x,y;
      memcpy(&x,&bits,4); y=erff(x); memcpy(out+i,&y,4);
    }
  }
}
