/* Proof harness only. Native MPAS random_id.c:21 seeds rand from time(NULL).
 * Control that input for a paired forecast without modifying native output.
 * CLOCK_MONOTONIC and clock_gettime, including Python time.time, stay real.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

time_t time(time_t *result) {
    const char *text = getenv("MPAS_PROOF_FILE_ID_SEED");
    time_t value;
    if (text) {
        char *end;
        errno = 0;
        unsigned long long seed = strtoull(text, &end, 10);
        if (!*text || *text == '-' || *end || errno || seed > UINT32_MAX) {
            fputs("invalid MPAS_PROOF_FILE_ID_SEED\n", stderr);
            abort();
        }
        value = (time_t)seed;
    } else {
        time_t (*original)(time_t *) = dlsym(RTLD_NEXT, "time");
        if (!original) abort();
        value = original(NULL);
    }
    if (result) *result = value;
    return value;
}
