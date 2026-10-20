"""Replay actual native restart/output columns decoded by rw-mpas Rust.

The bridge arrays preserve stored words. This verifies subsystem replay and
computed instantaneous native diagnostic fields; it is not a CUDA forecast.
"""
import json
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from hexcore.mpasdiag.convective_v842 import ConvectiveDiagnostics, FIELDS, MAXIMA
from test_mpas_convective_v842 import compare, gpu, receipt, references, signed_zero_replay, maxval_replay, subnormal_replay


ARRAYS = ('zgrid', 'w', 'uReconstructZonal', 'uReconstructMeridional',
          'theta_m', 'qv', 'exner', 'pressure_base', 'pressure_p', 'areaCell',
          'cellsOnVertex', 'kiteAreasOnVertex', 'vorticity')


def real_groups():
    root = os.environ.get('MPAS_CONVECTIVE_REAL_ARRAYS')
    if not root:
        pytest.skip('set MPAS_CONVECTIVE_REAL_ARRAYS to Rust-produced replay bridges')
    groups = json.loads((Path(root)/'groups.json').read_text())
    selected = os.environ.get('MPAS_CONVECTIVE_REAL_GROUP')
    if selected:
        if selected not in groups:
            raise ValueError('selected real replay group is absent from the sealed bridge')
        groups = {selected:groups[selected]}
    return [(name, [Path(root)/p for p in paths]) for name, paths in groups.items()]


def load(path):
    # Rust owns NetCDF decode, endian conversion and layout. np.load is only
    # the resident-array bridge; no scientific transform occurs in Python.
    return {name: np.load(path/(name+'.npy'), allow_pickle=False) for name in ARRAYS}


def compare_published(path, expected, rows, label, names=FIELDS[:14]):
    by_field = {}
    hashes = {}
    for i, name in enumerate(FIELDS):
        if name not in names or name in ('lcl','lfc'):
            continue  # Native never assigns these fields.
        stored = np.load(path/('native_'+name+'.npy'),allow_pickle=False)
        by_field[name] = int(np.count_nonzero(stored.view(np.uint32)!=expected[i].view(np.uint32)))
        hashes[name] = dict(published=hashlib.sha256(stored.tobytes()).hexdigest(),
                            replay=hashlib.sha256(expected[i].tobytes()).hexdigest())
    rows.append(dict(label=label,words=expected.shape[1]*len(by_field),
                     different=sum(by_field.values()),by_field=by_field,hashes=hashes))
    assert not sum(by_field.values()), by_field


def replay(native, port, rows, device=False):
    import cupy as cp
    for group, paths in real_groups():
        nc, nl = load(paths[0])['theta_m'].shape
        expected = np.zeros((len(FIELDS),nc),np.float32)
        actual = expected.copy()
        cuda = ConvectiveDiagnostics(cp.asarray(actual),nc,nl,MAXIMA) if device else None
        for path in paths:
            arrays = load(path)
            label = group+'/'+path.name
            native(arrays,expected,nc,MAXIMA,FIELDS,1)
            compare_published(path,expected,rows,label+' native instantaneous output')
            inputs = {n:cp.asarray(a) for n,a in arrays.items()} if device else arrays
            if device:
                cuda.compute(inputs,FIELDS)
                actual = cp.asnumpy(cuda.outputs)
            else:
                port(arrays,actual,nc,MAXIMA,FIELDS,1)
            compare(expected,actual,rows,label+' compute')
            native(arrays,expected,nc,MAXIMA,(),2)
            if device:
                cuda.update(inputs);actual=cp.asnumpy(cuda.outputs)
            else:
                port(arrays,actual,nc,MAXIMA,(),2)
            compare(expected,actual,rows,label+' sample maxima')
            # A sealed one-step native capture resets after its initial
            # output and updates exactly once before its final output.
            if all((path/('native_'+name+'.npy')).exists() for name in MAXIMA):
                compare_published(path,expected,rows,label+' native one-step maxima',MAXIMA)
            # This replay resets at each sampled boundary. Full timestep
            # maxima/reset cadence remains the whole-model gate's obligation.
            native(arrays,expected,nc,MAXIMA,MAXIMA,3)
            if device:
                cuda.reset(MAXIMA);actual=cp.asnumpy(cuda.outputs)
            else:
                port(arrays,actual,nc,MAXIMA,MAXIMA,3)
            compare(expected,actual,rows,label+' reset')


def test_real_native_host(references,receipt):
    replay(*references,receipt)


@pytest.mark.gpu
def test_real_native_cuda(references,receipt,gpu):
    replay(*references,receipt,device=True)


@pytest.mark.gpu
def test_signed_zero_cuda(references, receipt, gpu):
    signed_zero_replay(*references, receipt, device=True)


@pytest.mark.gpu
def test_maxval_cuda(references, receipt, gpu):
    maxval_replay(*references, receipt, device=True)


@pytest.mark.gpu
def test_subnormal_cuda(references, receipt, gpu):
    subnormal_replay(*references, receipt, device=True)
