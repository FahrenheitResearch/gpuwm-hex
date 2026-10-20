from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.lbc import LbcFile
from hexcore.partition_regional_v841 import PartitionLbcPoolV841, project_lbc_file


def test_lbc_projection_preserves_payload_and_entity_order():
    layout = SimpleNamespace(cell_l2g=np.array([3, 0, 2]), edge_l2g=np.array([1, 4]),
                             _global_cells=4, _global_edges=5)
    bits = np.array([0x80000000, 0x7fc12345, 0, 0x3f800000], dtype=np.uint32)
    cells = np.repeat(bits.view(np.float32)[:, None], 2, axis=1)
    edges = np.arange(10, dtype=np.float32).reshape(5, 2)
    source = LbcFile(Path("pool.nc"), datetime(2026, 1, 1), {"lbc_theta": cells, "lbc_u": edges})
    result = project_lbc_file(source, layout)
    assert result.path == source.path and result.valid_time == source.valid_time
    assert result.fields["lbc_theta"].tobytes() == cells[[3, 0, 2]].tobytes()
    assert result.fields["lbc_u"].tobytes() == edges[[1, 4]].tobytes()
    assert not result.fields["lbc_theta"].flags.writeable
    with pytest.raises(ValueError, match="expected 4"):
        project_lbc_file(LbcFile(source.path, source.valid_time, {"lbc_theta": cells[:3]}), layout)


def test_projected_pool_delegates_admission_and_clock():
    stamp = datetime(2026, 1, 1)
    source = LbcFile(Path("pool.nc"), stamp, {"lbc_theta": np.zeros((2, 1), np.float32)})
    class Pool:
        interval_end = stamp
        def start(self, when):
            assert when is stamp
            return source
        def advance(self, when):
            assert when is stamp
            return source
    layout = SimpleNamespace(cell_l2g=np.array([1]), _global_cells=2)
    view = PartitionLbcPoolV841(Pool(), layout)
    assert view.interval_end is stamp
    assert view.start(stamp).n_cells == 1
    assert view.advance(stamp).n_cells == 1
