"""Write cell/level views in HDF5 chunk order with bounded copy overlap.

The million-cell history writer copied a whole transposed field, then HDF5
assembled chunks that split both cells and levels. Preparing each chunk's
contiguous input avoids that whole-field temporary and strided assembly.
Only the caller touches NetCDF. Two pending copies bound temporary memory
and preserve the original cell-major chunk visitation and filter bytes.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .host_parallel import MIN_PARALLEL_ELEMENTS, threads


def write_variable(variable, payload):
    """Write the same values and chunk sequence as ``variable[:] = payload``."""
    if payload.ndim != 3 or payload.size < MIN_PARALLEL_ELEMENTS:
        variable[:] = np.ascontiguousarray(payload)
        return
    chunks = variable.chunking()
    if chunks == "contiguous" or len(chunks) != 3 or chunks[0] != 1 or payload.shape[0] != 1:
        variable[:] = np.ascontiguousarray(payload)
        return
    selections = iter((slice(None), slice(c, min(c + chunks[1], payload.shape[1])),
                       slice(l, min(l + chunks[2], payload.shape[2])))
                      for c in range(0, payload.shape[1], chunks[1])
                      for l in range(0, payload.shape[2], chunks[2]))
    with ThreadPoolExecutor(max_workers=min(2, threads()),
                            thread_name_prefix="hex-history-chunk") as pool:
        pending = deque()

        def enqueue():
            selection = next(selections, None)
            if selection is not None:
                pending.append((selection, pool.submit(np.ascontiguousarray, payload[selection])))

        enqueue()
        enqueue()
        while pending:
            selection, copied = pending.popleft()
            block = copied.result()
            variable[selection] = block
            del block, copied
            enqueue()
