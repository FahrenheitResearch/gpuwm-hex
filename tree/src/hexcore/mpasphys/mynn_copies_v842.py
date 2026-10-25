"""Batch MYNN adapter copies without host staging or floating point arithmetic.

The MPAS pool and MMM arrays use different layouts. Each field occupies one
grid row; threads run along the destination's unit-stride axis (columns) and
each thread walks the other axis (levels), so loads and stores coalesce and no
thread divides an index. Strides preserve solve-cell slices, interface levels
and halos without an intermediate array.
"""
from functools import lru_cache

import numpy as np


@lru_cache(None)
def kernel_source(count):
    arguments, cases = [], []
    for j in range(count):
        arguments.extend((f'const unsigned int* s{j}', f'unsigned int* d{j}',
                          *(f'int {v}{j}' for v in ('nr', 'nc', 'sr', 'sc', 'dr', 'dc')),
                          f'unsigned int fill{j}'))
        cases.append(f'''case {j}: s=s{j}; d=d{j}; nr=nr{j}; nc=nc{j};
            sr=sr{j}; sc=sc{j}; dr=dr{j}; dc=dc{j}; fill=fill{j}; break;''')
    return '''extern "C" __global__ void mynn_copy_fields(''' + ','.join(arguments) + ''') {
        const unsigned int* s;
        unsigned int* d;
        unsigned int fill;
        long long nr,nc,sr,sc,dr,dc;
        switch(blockIdx.y) {
    ''' + '\n'.join(cases) + '''
        default: return;
        }
        long long r=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(r>=nr) return;
        unsigned int* dp=d+r*dr;
        if(s) {
            auto sp=s+r*sr;
            for(long long c=0;c<nc;++c) dp[c*dc]=sp[c*sc];
        } else {
            for(long long c=0;c<nc;++c) dp[c*dc]=fill;
        }
    }
    '''


class MynnCopies:
    def __init__(self, cp):
        self.cp = cp
        self.modules = {}

    def copy_many(self, pairs):
        """One launch for independent 32-bit copies, with no device waits.

        Pairs are (destination, source); None writes positive zero bits and
        an integer source writes that raw 32-bit word. Integer extents/strides
        match MPAS bounds; address products use 64 bits. Each field uses at
        most 48 parameter bytes, so 85 fields fit the 4 KiB launch ABI.
        All arrays belong to the current device and copy destinations must
        not overlap any other pair's source.
        """
        pairs = tuple(pairs)
        if not pairs:
            return
        if len(pairs) > 85:
            for first in range(0, len(pairs), 85):
                self.copy_many(pairs[first:first+85])
            return
        args, largest = [], 0
        for destination, source in pairs:
            fill = source is None or isinstance(source, (int, np.integer))
            if not fill and (source.shape != destination.shape or source.dtype != destination.dtype):
                raise ValueError('MYNN copy requires matching shape and dtype')
            if destination.ndim not in (1, 2) or destination.dtype.itemsize != 4:
                raise ValueError('MYNN copy requires rank one/two 32-bit fields')
            rows = destination.shape[0]
            cols = destination.shape[1] if destination.ndim == 2 else 1
            src = tuple(s // 4 for s in source.strides) if not fill else (0, 0)
            dst = tuple(s // 4 for s in destination.strides)
            src = (src[0], src[1] if destination.ndim == 2 else 0)
            dst = (dst[0], dst[1] if destination.ndim == 2 else 0)
            # Threads run along the destination's unit-stride axis (pool rows are
            # level-major, MMM arrays column-major): swap so that axis is "rows".
            if destination.ndim == 2 and abs(dst[1]) < abs(dst[0]):
                rows, cols, src, dst = cols, rows, src[::-1], dst[::-1]
            dimensions = (rows, cols, src[0], src[1], dst[0], dst[1])
            if any(v < -2147483648 or v > 2147483647 for v in dimensions):
                raise ValueError('MYNN extents and word strides require signed 32-bit values')
            args.extend((source if not fill else np.uint64(0), destination,
                         *(np.int32(v) for v in dimensions),
                         np.uint32((int(source) if source is not None and fill else 0) & 0xffffffff)))
            largest = max(largest, rows)
        if not largest:
            return
        device = self.cp.cuda.runtime.getDevice()
        key = (device, len(pairs))
        if key not in self.modules:
            from .bl_mynn_v842 import _compile_device_module
            self.modules[key] = _compile_device_module(kernel_source(len(pairs)))
        self.modules[key].get_function('mynn_copy_fields')(
            ((largest + 127) // 128, len(pairs)), (128,), tuple(args))
