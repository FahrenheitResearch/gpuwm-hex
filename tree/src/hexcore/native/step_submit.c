/* SPDX-License-Identifier: Apache-2.0
 * Plain-stream submission only. Driver symbols and storage belong to caller.
 * Build before forecast construction; no compiler, allocation, synchronization,
 * Python callback or CUDA graph is involved in hex_step_submit.
 */
#include <stddef.h>
#include <stdint.h>

typedef struct {
    const void *value;
    size_t offset, width;
    uint32_t from_frame;
} HexArgument;

typedef struct {
    uint32_t kind, grid[3], block[3], shared_bytes, argc;
    void *function;
    const HexArgument *args;
    uint64_t destination, source;
    size_t bytes;
    unsigned char fill;
} HexCommand;

typedef int (*Launch)(void *, unsigned, unsigned, unsigned,
                      unsigned, unsigned, unsigned, unsigned, void *, void **, void **);
typedef int (*Copy)(uint64_t, uint64_t, size_t, void *);
typedef int (*Clear)(uint64_t, unsigned char, size_t, void *);
typedef struct { Launch launch; Copy copy; Clear clear; } HexDriver;

enum { HEX_KERNEL = 1, HEX_COPY = 2, HEX_CLEAR = 3, HEX_MAX_ARGS = 256 };

uint32_t hex_step_abi_version(void) { return 1; }

/* One-time admission. The sealed descriptors stay immutable and alive. */
int hex_step_validate(const HexCommand *commands, size_t count, size_t frame_bytes) {
    if (!commands || !count) return -1;
    for (size_t i = 0; i < count; ++i) {
        const HexCommand *c = commands + i;
        if (c->kind == HEX_KERNEL) {
            if (!c->function || c->argc > HEX_MAX_ARGS || (c->argc && !c->args)) return -1;
            for (unsigned d = 0; d < 3; ++d)
                if (!c->grid[d] || !c->block[d]) return -1;
            for (uint32_t a = 0; a < c->argc; ++a) {
                const HexArgument *arg = c->args + a;
                if (!arg->width || arg->from_frame > 1) return -1;
                if (arg->from_frame) {
                    if (arg->offset > frame_bytes || arg->width > frame_bytes - arg->offset)
                        return -1;
                } else if (!arg->value) return -1;
            }
        } else if (c->kind == HEX_COPY) {
            if (!c->destination || !c->source) return -1;
        } else if (c->kind == HEX_CLEAR) {
            if (!c->destination) return -1;
        } else return -1;
    }
    return 0;
}

/* Return the first CUDA error without retrying a partially submitted step.
 * completed counts successfully issued commands, not completed GPU work.
 * The caller has already validated the immutable descriptors and frame size.
 */
int hex_step_submit(const HexDriver *driver, const HexCommand *commands,
                    size_t count, const void *frame, void *stream, size_t *completed) {
    *completed = 0;
    for (size_t i = 0; i < count; ++i) {
        const HexCommand *c = commands + i;
        int status;
        if (c->kind == HEX_KERNEL) {
            void *parameters[HEX_MAX_ARGS];
            for (uint32_t a = 0; a < c->argc; ++a) {
                const HexArgument *arg = c->args + a;
                parameters[a] = arg->from_frame
                    ? (void *)((const unsigned char *)frame + arg->offset)
                    : (void *)arg->value;
            }
            status = driver->launch(c->function, c->grid[0], c->grid[1], c->grid[2],
                c->block[0], c->block[1], c->block[2], c->shared_bytes, stream,
                c->argc ? parameters : NULL, NULL);
        } else if (c->kind == HEX_COPY) {
            status = driver->copy(c->destination, c->source, c->bytes, stream);
        } else {
            status = driver->clear(c->destination, c->fill, c->bytes, stream);
        }
        if (status) return status;
        *completed = i + 1;
    }
    return 0;
}
