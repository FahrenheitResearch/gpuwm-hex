"""Both complete real microphysics handoffs and a wrong-timestep witness."""
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def packet():
    bank = os.environ.get('RRFS_DRIVER_MICRO_INTERFACE_BANK')
    library = os.environ.get('RRFS_DRIVER_RECORD_LIBRARY')
    if not bank or not library:
        pytest.skip('requires sealed pristine-native handoff bank and Rust codec')
    bank = Path(bank)
    captures = json.loads((bank / 'manifest.json').read_text())['captures']
    return bank, captures, Path(library)


def run_packet(packet, target, entering):
    from rrfs_driver_micro_interface_replay import replay
    bank, captures, library = packet
    boundary = 'microphysics_from_mpas' if entering else 'microphysics_to_mpas'
    prefix, = [p for p in captures if p.startswith('boundary__' + boundary + '.')]
    return replay(bank / 'dump', prefix, target, library, bank / 'manifest.json')


@pytest.mark.parametrize('entering,records', [(True, 91), (False, 94)])
def test_complete_real_microphysics_handoff(packet, tmp_path, entering, records):
    result = run_packet(packet, tmp_path, entering)
    assert result['output_records'] == records
    assert result['bitwise_equal'] and result['different_words'] == 0
    assert result['words'] > 0
    # Allocated interface-level tails and untouched pool context are included.
    assert len(result['comparisons']) == records
    assert entering or any(key.startswith('state.scalars.t')
                           for key in result['replaced_records'])


def test_real_return_rejects_wrong_timestep(packet, tmp_path, monkeypatch):
    from hexcore.mpasphys.rrfs import driver_rrfs
    original = driver_rrfs.microphysics_to_mpas

    def wrong_timestep(**kwargs):
        kwargs['dt_dyn'] *= 2
        return original(**kwargs)

    monkeypatch.setattr(driver_rrfs, 'microphysics_to_mpas', wrong_timestep)
    result = run_packet(packet, tmp_path, False)
    failed = {key for key, row in result['comparisons'].items()
              if not row['bitwise_equal']}
    assert not result['bitwise_equal'] and result['different_words'] > 0
    assert 'tend.rt_diabatic_tend.t1' in failed
    assert 'diag.dtheta_dt_mp.t1' in failed
    assert all(key.startswith('tend_physics.') or key in {
        'tend.rt_diabatic_tend.t1', 'diag.dtheta_dt_mp.t1',
        'tend.tend_sfc_pressure.t1'} for key in failed)
