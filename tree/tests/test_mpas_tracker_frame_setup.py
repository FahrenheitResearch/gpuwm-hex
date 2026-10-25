"""The MPAS route's tracker frame carries the setup targets' fields (owner run run_51882d2a2a758db6: "qv not found")."""
import numpy as np

from hexcore.mpasphys.moving_v842 import scalar_column


class Native:
    def __init__(self, value):
        self._fields = {("state", "scalars"): None}
        self.value = value

    def get_field(self, pool, name):
        return self.value


class Layout:
    fields = [{"name": "scalars", "pool": "state", "pool_name": "scalars", "components": ["qv", "qc", "qr"],
               "dimensions": ["nCells", "nVertLevels"]}]


def test_qv_from_the_scalars_var_array_as_cells_by_levels():
    n, levels = 7, 4
    value = np.zeros((3, n + 1, levels), np.float32)   # [component, nCells + garbage, level]
    value[0, :n] = np.arange(n * levels, dtype=np.float32).reshape(n, levels)
    got = scalar_column(Native(value), Layout(), "qv", n, levels)
    assert got.shape == (n, levels)
    assert np.array_equal(got, value[0, :n])
    assert scalar_column(Native(value), Layout(), "qg", n, levels) is None


def test_levels_first_storage_is_transposed():
    n, levels = 9, 5
    value = np.random.default_rng(1).random((levels, n + 1, 3)).astype(np.float32)  # [level, cell, component]
    got = scalar_column(Native(value), Layout(), "qc", n, levels)
    assert np.array_equal(got, value[:, :n, 1].T)
