"""Actual-artifact synthetic MPI seam check, launched by MPICH with two ranks.

MPAS source: src/framework/mpas_dmpar.F:1600-2160,4788-6257.
Fixtures exercise transport only. They are not real-mesh forecast acceptance.
"""
import ctypes as ct
import json
import os
from pathlib import Path
import sys

import numpy as np

from hexcore.mpasphys.native_backend import NativePhysics


def stream_block(native, ids):
    native.block()
    for name, value in [('nCells', 4), ('nCellsSolve', 2), ('nEdges', 1), ('nEdgesSolve', 1),
                        ('nVertices', 1), ('nVerticesSolve', 1), ('maxEdges', 1), ('maxEdges2', 1), ('vertexDegree', 1)]:
        native.dimension('mesh', name, value)
    for name, values, dimension in [('indexToCellID', ids, 'nCells'), ('indexToEdgeID', [1], 'nEdges'),
                                    ('indexToVertexID', [1], 'nVertices'), ('nEdgesOnCell', [0] * 4, 'nCells'),
                                    ('nEdgesOnEdge', [0], 'nEdges')]:
        native.put_field('mesh', name, np.asarray(values, dtype=np.int32))
        native.register_field('mesh', name, dimensions=(dimension,))
    native.decomposition('nCells', global_ids=ids, owned=2)
    native.clock(start='2000-01-01_00:00:00', stop='2000-01-01_01:00:00', dt=30.)
    native.streams()


def main():
    library, output = sys.argv[1:]
    rank = int(os.environ['PMI_RANK'])
    assert int(os.environ['PMI_SIZE']) == 2
    ids = np.asarray([[1, 3, 4, 2], [2, 4, 3, 1]][rank], dtype=np.int32)
    checked = 0
    with NativePhysics(library, test_only=True) as native:
        native.block()
        native.decomposition('nCells', global_ids=ids, owned=2)
        for dtype, ranks in [(np.float32, range(1, 6)), (np.int32, range(1, 4))]:
            for ndim in ranks:
                shape = (2,) * (ndim - 1) + (4,)
                values = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
                if dtype == np.float32:
                    # Distinct finite and exceptional words, including -0 and a NaN payload.
                    words = np.asarray([0x80000000, 0x7fc01234, 0x3f800001, 0xff800000], dtype=np.uint32)
                    values = np.broadcast_to(words.view(np.float32), shape).copy()
                expected = values[..., ids - 1].copy()
                source = np.full(shape, -99, dtype=dtype)
                source[..., :2] = expected[..., :2]
                name = f'field_{np.dtype(dtype).name}_{ndim}'
                native.put_field('state', name, np.full(shape, -77, dtype=dtype), nlevels=2)
                native.put_field('state', name, source, nlevels=2, level=2)
                native.register_field('state', name,
                                      dimensions=('fixture',) * (ndim - 1) + ('nCells',), decomposed=True)
                native.exchange_halo(name, level=2)
                actual = native.get_field('state', name, level=2)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                np.testing.assert_array_equal(native.get_field('state', name, level=1), -77)
                checked += actual.size
    filename = str(Path(output, 'collective.nc').resolve())
    values = np.asarray([[1., -0., .5, -10.], [2., 3., 4., 5.]], dtype=np.float32)
    with NativePhysics(library, test_only=True) as native:
        stream_block(native, ids)
        native.put_field('state', 'theta', np.full((2, 4), -77., dtype=np.float32), nlevels=2)
        native.put_field('state', 'theta', values[:, ids - 1], nlevels=2, level=2)
        native.register_field('state', 'theta', dimensions=('nVertLevels', 'nCells'), decomposed=True)
        native.stream('write', filename=filename, direction='output')
        native.stream_field('write', 'theta')
        native.transfer_stream('write', read=False, level=2)
    # Different ownership and ordering on read. Native I/O fills owned cells,
    # and native halo exchange fills the remaining cells by their global IDs.
    read_ids = np.asarray([[4, 1, 2, 3], [3, 2, 1, 4]][rank], dtype=np.int32)
    with NativePhysics(library, test_only=True) as native:
        stream_block(native, read_ids)
        native.put_field('state', 'theta', np.full((2, 4), -77., dtype=np.float32), nlevels=2)
        native.put_field('state', 'theta', np.full((2, 4), -99., dtype=np.float32), nlevels=2, level=2)
        native.register_field('state', 'theta', dimensions=('nVertLevels', 'nCells'), decomposed=True)
        native.stream('read', filename=filename, direction='input')
        native.stream_field('read', 'theta')
        native.transfer_stream('read', read=True, level=2)
        np.testing.assert_array_equal(native.get_field('state', 'theta', level=2).view(np.uint32),
                                      values[:, read_ids - 1].view(np.uint32))
        np.testing.assert_array_equal(native.get_field('state', 'theta', level=1), -77.)
    Path(output, f'mpi-rank-{rank}.json').write_text(json.dumps({
        'schema': 'mpas-native-mpi-debug-v1', 'rank': rank, 'ranks': 2,
        'ids': ids.tolist(), 'owned': 2, 'checked_words': checked,
        'different_words': 0, 'field_ranks': {'real': [1, 2, 3, 4, 5], 'integer': [1, 2, 3]},
        'separate_time_levels': True, 'forecast_proof': False,
        'native_collective_stream': {'write_ids': ids.tolist(), 'read_ids': read_ids.tolist(),
                                     'words': 8, 'different_words': 0, 'native_input_halos': True},
    }, indent=2) + '\n')
    # This standalone probe owns process MPI. The backend never finalizes caller MPI.
    mpi = ct.CDLL(library)
    assert mpi.MPI_Finalize() == 0


if __name__ == '__main__':
    main()
