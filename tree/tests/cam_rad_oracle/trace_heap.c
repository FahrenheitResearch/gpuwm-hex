/* Observe allocator reuse in the unchanged strict standalone CAM executable.
 * This diagnostic interposes glibc entry points without an extra allocation.
 * It records the first eight raw words, never interprets them as physics.
 * Compare native output hashes before treating an instrumented run as evidence.
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <link.h>
#include <malloc.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void __libc_free(void *);

static int trace_fd = -1;
static uintptr_t image_base;
static uintptr_t image_end;
static _Thread_local int logging;

struct record {
    uint64_t operation, pointer, bytes, caller;
    uint64_t words[8];
};

static int executable(struct dl_phdr_info *info, size_t bytes, void *unused)
{
    (void)bytes; (void)unused;
    if (info->dlpi_name && info->dlpi_name[0]) return 0;
    image_base = info->dlpi_addr;
    for (int i = 0; i < info->dlpi_phnum; ++i) {
        const ElfW(Phdr) *p = info->dlpi_phdr + i;
        if (p->p_type == PT_LOAD && image_end < image_base + p->p_vaddr + p->p_memsz)
            image_end = image_base + p->p_vaddr + p->p_memsz;
    }
    return 1;
}

__attribute__((constructor)) static void begin_trace(void)
{
    dl_iterate_phdr(executable, NULL);
    const char *path = getenv("CAM_NATIVE_HEAP_TRACE");
    if (path) trace_fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
}

static void record(uint64_t operation, void *pointer, size_t bytes, void *caller)
{
    uintptr_t pc = (uintptr_t)caller;
    if (logging || trace_fd < 0 || pc < image_base || pc >= image_end) return;
    logging = 1;
    struct record item = {operation, (uintptr_t)pointer, bytes, pc - image_base, {0}};
    if (pointer && bytes) memcpy(item.words, pointer, bytes < sizeof(item.words) ? bytes : sizeof(item.words));
    const char *data = (const char *)&item;
    size_t left = sizeof(item);
    while (left) {
        ssize_t written = syscall(SYS_write, trace_fd, data, left);
        if (written <= 0) break;
        data += written; left -= written;
    }
    logging = 0;
}

void *malloc(size_t bytes)
{
    void *pointer = __libc_malloc(bytes);
    record(1, pointer, bytes, __builtin_return_address(0));
    return pointer;
}

void *calloc(size_t number, size_t bytes)
{
    void *pointer = __libc_calloc(number, bytes);
    record(2, pointer, pointer ? malloc_usable_size(pointer) : 0, __builtin_return_address(0));
    return pointer;
}

void free(void *pointer)
{
    record(3, pointer, pointer ? malloc_usable_size(pointer) : 0, __builtin_return_address(0));
    __libc_free(pointer);
}

void *realloc(void *pointer, size_t bytes)
{
    void *caller = __builtin_return_address(0);
    record(4, pointer, pointer ? malloc_usable_size(pointer) : 0, caller);
    void *next = __libc_realloc(pointer, bytes);
    record(5, next, next ? malloc_usable_size(next) : 0, caller);
    return next;
}
