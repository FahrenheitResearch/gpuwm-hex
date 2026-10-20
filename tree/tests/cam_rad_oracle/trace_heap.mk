CC = gcc
CFLAGS = -O2 -Wall -Wextra -Werror
SOURCES = ../../hex/tree/tests/cam_rad_oracle

all: trace_heap.so trace_heap_dump

trace_heap.so: $(SOURCES)/trace_heap.c
	$(CC) $(CFLAGS) -fPIC -shared $< -o $@

trace_heap_dump: $(SOURCES)/trace_heap_dump.c
	$(CC) $(CFLAGS) $< -o $@
