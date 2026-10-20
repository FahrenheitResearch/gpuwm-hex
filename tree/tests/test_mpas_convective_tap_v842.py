"""Whole module replay on f1's real native output-diagnostic pool captures.

The stated sample is the three published boundaries in each of two suites.
POST pool inputs have the reconstructed winds required by convective compute;
PRE diagnostic storage seeds unwritten words. Strict pristine Fortran supplies
the expected outputs of compute, maxima update and reset, independently of CUDA.
Rust owns ORCL0001 decoding; Python only maps resident driver argument arrays.
"""
from contextlib import ExitStack
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hexcore.mpasdiag.convective_v842 import ConvectiveDiagnostics, FIELDS, MAXIMA
from test_mpas_convective_v842 import references, receipt, gpu, compare


@pytest.mark.gpu
def test_f1_real_native_tap_cuda(references, receipt, gpu):
    index = os.environ.get('MPAS_CONVECTIVE_TAP_INDEX')
    if not index:
        pytest.skip('set MPAS_CONVECTIVE_TAP_INDEX to the authenticated f1 selection')
    from oracle_replay import Capture, _Metadata
    lib = ct.CDLL(os.environ['MPAS_CONVECTIVE_TAP_TRANSPORT'])
    lib.oracle_open.argtypes, lib.oracle_open.restype = [ct.c_char_p], ct.c_void_p
    lib.oracle_metadata.argtypes = [ct.c_void_p, ct.POINTER(_Metadata)]
    lib.oracle_close.argtypes = [ct.c_void_p]
    native, _ = references
    spec = json.loads(Path(index).read_text())
    assert spec['native_mpi_ranks'] == 1 and spec['candidate_processes'] == 1
    assert len(spec['cases']) == spec['expected_frames'] == 6
    root = Path(os.environ['MPAS_CONVECTIVE_TAP_ROOT'])
    for case in spec['cases']:
        with ExitStack() as stack:
            values = {}
            for name, record in case['files'].items():
                path = root / record['path']
                assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
                capture = stack.enter_context(Capture(lib, path))
                assert capture.header[15] == 0  # original MPI rank
                values[name] = capture.array()
            arrays = {name: np.ascontiguousarray(value.T)
                      for name, value in values.items()
                      if name not in ('scalars', 'index_qv', 'nCellsSolve')
                      and not name.startswith('entry_')}
            qindex = int(values['index_qv'].item()) - 1
            arrays['qv'] = np.ascontiguousarray(values['scalars'][qindex].T)
            nc, nl = arrays['theta_m'].shape
            ns = int(values['nCellsSolve'].item())
            expected = np.stack([values['entry_' + n] for n in FIELDS])
            actual = expected.copy()
            inputs = {name: gpu.asarray(a) for name, a in arrays.items()}
            cuda = ConvectiveDiagnostics(gpu.asarray(actual), ns, nl, MAXIMA)
            for action, written in ((1, FIELDS), (2, ()), (3, MAXIMA)):
                native(arrays, expected, ns, MAXIMA, written, action)
                if action == 1:
                    cuda.compute(inputs, written)
                elif action == 2:
                    cuda.update(inputs)
                else:
                    cuda.reset(MAXIMA)
                actual = gpu.asnumpy(cuda.outputs)
                compare(expected, actual, receipt,
                        case['suite'] + '/' + case['prefix'] + '/action' + str(action))
                assert actual.shape == (len(FIELDS), nc)
