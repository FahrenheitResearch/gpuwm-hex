/* Which libm entry points round differently on this host?
 *
 * Native MPAS (gnu-strict) and hex's v841 geometry both call the HOST GNU libm,
 * so the identity claim is per host C library. This prints, for every libm
 * function the native atmosphere_model imports, a 64-bit FNV-1a digest of the
 * result bits over 2^22 deterministic inputs. Run it on two hosts and diff the
 * lines: a differing digest names a function whose rounding changed.
 * build: gcc -O0 -fno-builtin libm_host_probe.c -lm -o libm_host_probe
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <gnu/libc-version.h>
#define N (1u << 22)
static uint64_t state;
static double unit(void) { /* 53-bit LCG draw in [0,1) */
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(state >> 11) / 9007199254740992.0;
}
static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}
#define F1(name, lo, hi) do { state = 42; uint64_t h = 1469598103934665603ULL; \
    for (unsigned i = 0; i < N; i++) { volatile float x = (float)((lo) + ((hi) - (lo)) * unit()); \
        volatile float y = name(x); h = fnv(h, (void *)&y, 4); } \
    printf("%-8s %016llx\n", #name, (unsigned long long)h); } while (0)
#define F2(name, lo, hi, lo2, hi2) do { state = 42; uint64_t h = 1469598103934665603ULL; \
    for (unsigned i = 0; i < N; i++) { volatile float x = (float)((lo) + ((hi) - (lo)) * unit()); \
        volatile float z = (float)((lo2) + ((hi2) - (lo2)) * unit()); \
        volatile float y = name(x, z); h = fnv(h, (void *)&y, 4); } \
    printf("%-8s %016llx\n", #name, (unsigned long long)h); } while (0)
#define D1(name, lo, hi) do { state = 42; uint64_t h = 1469598103934665603ULL; \
    for (unsigned i = 0; i < N; i++) { volatile double x = (lo) + ((hi) - (lo)) * unit(); \
        volatile double y = name(x); h = fnv(h, (void *)&y, 8); } \
    printf("%-8s %016llx\n", #name, (unsigned long long)h); } while (0)
#define D2(name, lo, hi, lo2, hi2) do { state = 42; uint64_t h = 1469598103934665603ULL; \
    for (unsigned i = 0; i < N; i++) { volatile double x = (lo) + ((hi) - (lo)) * unit(); \
        volatile double z = (lo2) + ((hi2) - (lo2)) * unit(); \
        volatile double y = name(x, z); h = fnv(h, (void *)&y, 8); } \
    printf("%-8s %016llx\n", #name, (unsigned long long)h); } while (0)
static float sincos_s(float x) { float s, c; sincosf(x, &s, &c); return s; }
static float sincos_c(float x) { float s, c; sincosf(x, &s, &c); return c; }
int main(void) {
    printf("glibc    %s\n", gnu_get_libc_version());
    F1(sinf, -6.3, 6.3); F1(cosf, -6.3, 6.3); F1(tanf, -1.5, 1.5);
    F1(sincos_s, -6.3, 6.3); F1(sincos_c, -6.3, 6.3);
    F1(asinf, -1.0, 1.0); F1(acosf, -1.0, 1.0); F1(atanf, -50.0, 50.0);
    F2(atan2f, -1.0, 1.0, -1.0, 1.0);
    F1(expf, -20.0, 20.0); F1(logf, 1e-6, 1e6); F1(log10f, 1e-6, 1e6);
    F2(powf, 1e-3, 1e3, -3.0, 3.0); F1(erff, -4.0, 4.0); F1(tanhf, -10.0, 10.0);
    D1(cos, -6.3, 6.3); D1(atan, -50.0, 50.0); D1(exp, -20.0, 20.0); D1(log, 1e-6, 1e6);
    D1(log10, 1e-6, 1e6); D2(pow, 1e-3, 1e3, -3.0, 3.0);
    return 0;
}
