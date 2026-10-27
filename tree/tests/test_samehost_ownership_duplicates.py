"""Same-host ownership refuses a repeated local global index with a mark array, as the sort did (no GPU)."""
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.partition_executor_v841 import ExecutorError
from hexcore.partition_samehost_tables_v841 import _entity_ownership


def _layout(rank, l2g, owned, extent=6):
    return SimpleNamespace(partition=rank, _global_cells=extent, cell_l2g=np.asarray(l2g), n_owned_cells=owned)


def test_an_exact_cover_is_admitted():
    owner, owner_local, vectors, counts = _entity_ownership(
        (_layout(0, [0, 1, 2, 3], 3), _layout(1, [3, 4, 5, 2], 3)), "cell", "cells")
    assert owner.tolist() == [0, 0, 0, 1, 1, 1] and counts == (3, 3)


def test_a_repeated_local_global_index_is_refused():
    with pytest.raises(ExecutorError, match="rank 1 cells: duplicate local global index"):
        _entity_ownership((_layout(0, [0, 1, 2, 3], 3), _layout(1, [3, 4, 5, 5], 3)), "cell", "cells")
