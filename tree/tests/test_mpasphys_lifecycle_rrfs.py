"""Real native lifecycle/schedule replay and an intentionally broken scheduler."""
import os
from pathlib import Path

import pytest

from rrfs_driver_lifecycle_replay import replay


@pytest.fixture
def real_bank():
    path = os.environ.get('RRFS_DRIVER_LIFECYCLE_BANK')
    if path is None:
        pytest.skip('requires sealed pristine-native real lifecycle bank')
    return Path(path)


def test_complete_real_lifecycle_and_schedule(real_bank):
    result = replay(Path(__file__).parents[2],real_bank)
    assert result['observed_calls'] > 0
    assert result['compared_schedule_words'] > 0
    assert result['bitwise_equal']
    assert all(stage['native'] == stage['hex'] for stage in result['order'])


def test_real_schedule_replay_rejects_wrong_microphysics_timestep(real_bank,monkeypatch):
    from hexcore.mpasphys.rrfs.driver_rrfs import PhysicsManager
    original = PhysicsManager.native_timestep_state
    def wrong_schedule(self):
        output = dict(original(self))
        output['dt_microp'] *= 2
        return output
    monkeypatch.setattr(PhysicsManager,'native_timestep_state',wrong_schedule)
    result = replay(Path(__file__).parents[2],real_bank)
    assert not result['bitwise_equal']
    assert result['different_words'] == 1
    assert [c['field'] for c in result['comparisons'] if not c['equal']] == ['dt_microp']
