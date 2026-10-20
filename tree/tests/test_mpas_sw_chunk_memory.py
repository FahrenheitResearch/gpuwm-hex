"""The wrapper must bound McICA before its already-chunked solver."""
import os
from pathlib import Path
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_default_wrapper_crosses_two_chunk_boundaries_against_native():
    root = os.environ.get('MPAS_RRTMG_SW_ORACLE')
    if not root:
        pytest.skip('MPAS_RRTMG_SW_ORACLE must name native tables.swd and columns.swd')
    import cupy as cp
    from mpas_rrtmg_sw_oracle.replay import read_swd
    from hexcore.mpasphys.rrtmg_sw_v842 import MPASShortwave, FIELD_NAMES, SURFACE_NAMES
    records = read_swd(Path(root) / 'columns.swd')
    tables = read_swd(Path(root) / 'tables.swd')
    engine = MPASShortwave(raw_coefficients={k: cp.asarray(v) for k, v in tables.items()
                                           if k.startswith('kg') and v.ndim})
    case = 'c00001'
    count = 513
    nz = records[f'{case}/out/rthratensw'].size
    def repeated(name):
        return cp.asarray(np.repeat(records[f'{case}/in/{name}'][None], count, axis=0))
    fields = {name: repeated(name)[:, :nz].copy() for name in FIELD_NAMES}
    surface = {name: cp.full(count, records[f'{case}/in/{name}'].item(), dtype=cp.float32)
               for name in SURFACE_NAMES}
    controls = ('xtime', 'gmt', 'julday', 'radt', 'degrad', 'declin', 'solcon', 'icloud',
                'cldovrlp', 'idcor', 'o3input', 'has_reqc', 'has_reqi', 'has_reqs', 'aer_opt')
    options = {name: records[f'{case}/in/{name}'].item() for name in controls}
    options.update(pin=cp.asarray(records[f'{case}/in/pin']), o3clim=repeated('o3clim'))
    if f'{case}/in/tauaer' in records:
        options['aerosol'] = {name: repeated(name).transpose(0, 2, 1).copy()
                              for name in ('tauaer', 'ssaaer', 'asyaer')}
    outputs = engine.rrtmg_swrad(fields, surface, repeated('p8w'), repeated('t8w'), **options)
    for name, output in outputs.items():
        expected = np.repeat(records[f'{case}/out/{name}'][None], count, axis=0)
        np.testing.assert_array_equal(output.get().view(np.uint32), expected.view(np.uint32))
