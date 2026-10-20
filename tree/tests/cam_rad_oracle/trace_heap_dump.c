/* Decode only the diagnostic's own raw allocator records, without Python. */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

struct record {
    uint64_t operation, pointer, bytes, caller;
    uint64_t words[8];
};

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    FILE *input = fopen(argv[1], "rb");
    if (!input) return 2;
    struct record item;
    uint64_t sequence = 0;
    while (fread(&item, sizeof(item), 1, input) == 1) {
        printf("{\"sequence\":%" PRIu64 ",\"operation\":%" PRIu64
               ",\"pointer\":\"0x%" PRIx64 "\",\"bytes\":%" PRIu64
               ",\"caller_offset\":\"0x%" PRIx64 "\",\"words\":[",
               ++sequence, item.operation, item.pointer, item.bytes, item.caller);
        for (int i = 0; i < 8; ++i)
            printf("%s\"0x%016" PRIx64 "\"", i ? "," : "", item.words[i]);
        puts("]}");
    }
    int failure = ferror(input) || !feof(input);
    fclose(input);
    return failure;
}
