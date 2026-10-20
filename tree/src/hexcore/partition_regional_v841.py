"""Partition-local views of the existing regional LBC file pool.

Only entity selection lives here. The shipped reader, stream admission and
CUDA derivation/interpolation remain the authorities for field values.
"""
from __future__ import annotations

import numpy as np

from .lbc import LbcFile, LbcPool


def project_lbc_file(admitted, layout):
    fields = {}
    for name, array in admitted.fields.items():
        indexes = layout.edge_l2g if name == "lbc_u" else layout.cell_l2g
        expected = layout._global_edges if name == "lbc_u" else layout._global_cells
        if array.shape[0] != expected:
            raise ValueError(f"{admitted.path}: {name} has {array.shape[0]} entities, expected {expected}")
        selected = np.ascontiguousarray(array[np.asarray(indexes, dtype=np.int64)])
        selected.setflags(write=False)
        fields[name] = selected
    return LbcFile(admitted.path, admitted.valid_time, fields)


class PartitionLbcPoolV841:
    """Delegate the native two-level admission, then gather each rank's rows."""

    def __init__(self, pool: LbcPool, layout):
        self.pool = pool
        self.layout = layout

    @property
    def interval_end(self):
        return self.pool.interval_end

    def start(self, when):
        return project_lbc_file(self.pool.start(when), self.layout)

    def advance(self, when=None):
        return project_lbc_file(self.pool.advance(when), self.layout)
