"""Complete native second-ring initialization against Rust, including halo rows.

MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:1481-1584 at 1d6e370e.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from test_initialization_rrfs_native import call


@pytest.mark.parametrize('cells,solved,width', [(5,5,2),(9,9,2),(12,4,4)])
@pytest.mark.parametrize('preserve', [False, True])
def test_complete_native_second_ring(cells,solved,width,preserve):
    from hexcore.core_rrfs import initialize_second_ring_rrfs
    path = os.environ.get('RRFS_CONNECTIVITY_ORACLE')
    bridge = os.environ.get('RW_MPAS_RRFS_CONNECTIVITY')
    if not path or not bridge:
        pytest.skip('whole native connectivity oracle and Rust bridge are required')
    shifts = (1,-1,2,-2)[:width]
    neighbors = np.array([[(cell+shift)%cells for shift in shifts] for cell in range(cells)], dtype='i4')
    counts = np.full(cells,width,'i4'); width2 = 2*width
    mesh = SimpleNamespace(nEdgesOnCell=counts,cellsOnCell=neighbors,maxEdges2=width2)
    entry_counts = np.arange(cells, dtype='i4') + 71 if preserve else np.zeros(cells,'i4')
    entry_neighbors = np.arange(cells*width2, dtype='i4').reshape(cells,width2) + 91 if preserve else np.full((cells,width2),-1,'i4')
    if preserve:
        mesh.nCellsOnCellsOnCell = entry_counts
        mesh.cellsOnCellsOnCell = entry_neighbors
    got_count, got = initialize_second_ring_rrfs(mesh,cell_solve_count=solved,bridge=bridge)
    values = dict(ncells=cells,ncellssolve=solved,maxedges=width,maxedges2=width2,
                  nedgesoncell=counts,ncellsoncellsoncell=entry_counts.copy(),
                  cellsoncell=np.asfortranarray((neighbors+1).T),
                  cellsoncellsoncell=np.asfortranarray((entry_neighbors+1).T),
                  xcell=np.zeros(cells,'f4'),ycell=np.zeros(cells,'f4'),zcell=np.zeros(cells,'f4'))
    call(path,'atm_compute_cellsOnCellsOnCell',values)
    checks=[]
    for name,actual,want in [('counts',got_count,values['ncellsoncellsoncell']),
                             ('neighbors',got+1,values['cellsoncellsoncell'].T)]:
        diff=int(np.count_nonzero(actual!=want))
        checks.append(dict(field=name,words=actual.size,differing_words=diff))
        assert diff==0
    output=os.environ.get('RRFS_RECEIPTS')
    if output:
        Path(output,f'connectivity-{cells}-{solved}-{width}-{preserve}.json').write_text(json.dumps(dict(
            tier='BITWISE SUBSYSTEM ONLY',native_ranks=1,comparisons=checks,
            branches=['native traversal order','duplicate removal','unowned rows and unused prefixes retain entry']),indent=2)+'\n')


def test_published_native_second_ring():
    """Whole fork core.F:1481-1584 on every explicitly supplied published grid."""
    from hexcore.core_rrfs import initialize_second_ring_rrfs
    from hexcore.mesh import Mesh
    import hashlib
    paths = os.environ.get('RRFS_CORE_GRIDS')
    oracle = os.environ.get('RRFS_CONNECTIVITY_ORACLE')
    bridge = os.environ.get('RW_MPAS_RRFS_CONNECTIVITY')
    if not paths or not oracle or not bridge:
        pytest.skip('published grids plus native oracle/Rust bridge required')
    rows = []
    for filename in json.loads(paths):
        path = Path(filename)
        mesh = Mesh.from_netcdf(path)
        cells, width = mesh.cellsOnCell.shape
        width2 = mesh.edgesOnEdge.shape[1]
        got_count, got = initialize_second_ring_rrfs(mesh, bridge=bridge)
        values = dict(ncells=cells,ncellssolve=cells,maxedges=width,maxedges2=width2,
                      nedgesoncell=np.asfortranarray(mesh.nEdgesOnCell.astype('i4')),
                      ncellsoncellsoncell=np.zeros(cells,'i4'),
                      cellsoncell=np.asfortranarray((mesh.cellsOnCell+1).T.astype('i4')),
                      cellsoncellsoncell=np.zeros((width2,cells),'i4',order='F'),
                      xcell=np.zeros(cells,'f4'),ycell=np.zeros(cells,'f4'),zcell=np.zeros(cells,'f4'))
        call(oracle,'atm_compute_cellsOnCellsOnCell',values)
        checks=[]
        for name,actual,want in [('counts',got_count,values['ncellsoncellsoncell']),
                                 ('neighbors',got+1,values['cellsoncellsoncell'].T)]:
            checks.append(dict(field=name,words=actual.size,
                               differing_words=int(np.count_nonzero(actual!=want))))
        rows.append(dict(input=str(path),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         cells=cells,comparisons=checks))
    output=os.environ.get('RRFS_RECEIPTS')
    if output:
        Path(output,'connectivity-published-grids.json').write_text(json.dumps(dict(
            tier='BITWISE SUBSYSTEM ONLY' if not any(c['differing_words'] for row in rows for c in row['comparisons']) else 'FAILED SUBSYSTEM COMPARISON',
            native_ranks=1,cases=rows),indent=2)+'\n')
    assert not any(c['differing_words'] for row in rows for c in row['comparisons'])
