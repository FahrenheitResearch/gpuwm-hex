"""Whole real preparation boundary and an intentionally broken output."""
import json,os
from pathlib import Path
import pytest


@pytest.fixture
def packet():
    bank=os.environ.get('RRFS_DRIVER_INTERFACE_BANK')
    library=os.environ.get('RRFS_DRIVER_RECORD_LIBRARY')
    if bank is None or library is None:
        pytest.skip('requires sealed real native interface and Rust record bridge')
    bank=Path(bank)
    prefix,=json.loads((bank/'manifest.json').read_text())['captures']
    return bank,prefix,Path(library)


def run_packet(packet,target):
    from rrfs_driver_interface_replay import replay
    bank,prefix,library=packet
    return replay(bank/'dump',prefix,target,library,bank/'manifest.json')


def test_complete_real_preparation(packet,tmp_path):
    result=run_packet(packet,tmp_path)
    assert result['bitwise_equal']
    assert result['different_words']==0
    assert result['words']>0
    assert 'diag.surface_pressure.t1' in result['replaced_records']


def test_real_preparation_rejects_wrong_surface_pressure(packet,tmp_path,monkeypatch):
    import cupy as cp
    from hexcore.mpasphys.rrfs import driver_rrfs
    original=driver_rrfs.prepare_mpas_to_physics
    def broken(**kwargs):
        result=original(**kwargs)
        result['psfc_p']=result['psfc_p']+cp.float32(1)
        return result
    monkeypatch.setattr(driver_rrfs,'prepare_mpas_to_physics',broken)
    result=run_packet(packet,tmp_path)
    assert not result['bitwise_equal']
    assert result['different_words']>0
    assert {key for key,row in result['comparisons'].items() if not row['bitwise_equal']} == {'diag.surface_pressure.t1','vars.psfc_p'}
