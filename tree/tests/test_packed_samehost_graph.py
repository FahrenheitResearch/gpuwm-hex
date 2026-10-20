"""Packed production graph agrees with the retained text/tuple entry point."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.drivers import samehost_layouts_v841 as packed
from hexcore import host_parallel
from test_build_samehost_layouts_v841 import _grid, instrument


@pytest.mark.parametrize('n,width', [(0, 0), (3, 0), (1, 2), (12, 6), (103, 6)])
@pytest.mark.parametrize('workers', [1, 4])
def test_graph_bytes_components_and_cuts(n, width, workers, monkeypatch):
    monkeypatch.setenv(host_parallel.ENVIRONMENT, str(workers))
    monkeypatch.setattr(host_parallel, 'MIN_PARALLEL_ELEMENTS', 0)
    monkeypatch.setattr(host_parallel, 'MIN_BLOCK_COLUMNS', 1)
    rows = [tuple(sorted(set(((i-1) % n, (i+1) % n)) - {i})) if width else () for i in range(n)]
    adjacency = np.full((n, width), -1, dtype=np.int64)
    for i, row in enumerate(rows):
        adjacency[i, :len(row)] = row
        if len(row) and width > len(row):
            adjacency[i, -1] = row[0]  # Duplicate neighbors remain deduplicated.
    mesh = SimpleNamespace(dimensions={'nCells': n}, arrays={
        'cellsOnCell': adjacency, 'nEdgesOnCell': np.full(n, width)})
    graph = packed.PackedGraph(mesh)
    expected = (f'{n} {sum(map(len, rows))//2}\n' + ''.join(
        ' '.join(str(v+1) for v in row)+'\n' for row in rows)).encode('ascii')
    assert graph.metis_bytes() == expected
    assert graph.components() == packed.component_count(tuple(rows))
    for members in (np.arange(n), np.arange(0, n, 2), np.array([], dtype=int)):
        assert graph.components(members) == packed.component_count(tuple(rows), members)
    assignments = np.arange(n, dtype=np.int32) % 3
    assert graph.cut_edges(assignments) == sum(assignments[i] != assignments[j]
                                               for i, row in enumerate(rows) for j in row)//2


@pytest.mark.parametrize('ranks', [1, 2, 4])
@pytest.mark.parametrize('workers', [1, 4])
def test_production_assets_identical_without_text_reparse(tmp_path, ranks, workers, monkeypatch):
    monkeypatch.setenv(host_parallel.ENVIRONMENT, str(workers))
    grid = _grid(tmp_path)
    part = tmp_path / 'official.part'
    part.write_text(''.join(f'{i*ranks//12}\n' for i in range(12)))
    before = instrument.build_assets(grid, ranks=ranks, output_dir=tmp_path/'old', part=part)
    import hexcore.partition_assets_v841 as assets
    def refuse(*args, **kwargs):
        pytest.fail('packed production builder reparsed graph text')
    monkeypatch.setattr(assets, 'read_graph_info', refuse)
    after = packed.build_assets(grid, ranks=ranks, output_dir=tmp_path/'new', part=part)
    for key in ('graph', 'part'):
        assert Path(before[key]).read_bytes() == Path(after[key]).read_bytes()
    for key in ('cut_edges', 'owned_connected_components', 'exact_cover', 'replicated_cells'):
        assert before[key] == after[key]
    with np.load(before['layouts']) as a, np.load(after['layouts']) as b:
        assert a.files == b.files
        for name in a.files:
            assert a[name].dtype == b[name].dtype
            assert a[name].shape == b[name].shape
            assert a[name].tobytes() == b[name].tobytes()


def test_disconnected_partition_rejected(tmp_path):
    grid = _grid(tmp_path)
    part = tmp_path/'alternating.part'
    part.write_text('0\n1\n'*6)
    with pytest.raises(ValueError, match=r'partitions are not contiguous.*\[6, 6\]'):
        packed.build_assets(grid, ranks=2, output_dir=tmp_path/'bad', part=part)
