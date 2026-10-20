"""Strict standalone LES CUDA/native replays; run on one GPU."""
import os
from pathlib import Path
import pytest


@pytest.mark.gpu
def test_real_cuda_matches_pristine_mpas():
    reference=os.environ.get('MPAS_LES_NATIVE_REPLAY')
    if not reference:
        pytest.skip('set MPAS_LES_NATIVE_REPLAY and run on one GPU')
    from replay_les_v842 import cuda
    receipt=os.environ['MPAS_LES_RECEIPT']
    assert Path(reference).is_file()
    assert cuda(reference,receipt)==0


@pytest.mark.gpu
def test_coupled_cuda_matches_native_rk_slice():
    reference=os.environ.get('MPAS_LES_NATIVE_BOUNDARY')
    if not reference:
        pytest.skip('set MPAS_LES_NATIVE_BOUNDARY and run on one GPU')
    from replay_les_boundary_v842 import cuda
    assert cuda(reference,os.environ['MPAS_LES_BOUNDARY_RECEIPT'])==0


@pytest.mark.gpu
def test_native_subnormal_input_survives_and_ftz_control_fails():
    reference=os.environ.get('MPAS_LES_NATIVE_REPLAY')
    if not reference:
        pytest.skip('set MPAS_LES_NATIVE_REPLAY and run on one GPU')
    from replay_les_subnormal_v842 import run
    assert run(reference,os.environ['MPAS_LES_SUBNORMAL_RECEIPT'])==0
