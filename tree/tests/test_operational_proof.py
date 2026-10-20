"""Evidence must not turn a truncated or unobserved forecast into a pass."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('operational_proof', Path(__file__).parents[1] / 'tools' / 'operational_proof.py')
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


def artifact(tmp_path, *, health=True, final=True, completed=1, finite=True):
    fc = tmp_path / 'fc'
    fc.mkdir()
    row = dict(step=1, finite=finite, rho_min=1., theta_m_min=280., theta_m_max=310.,
               vertical_velocity_abs_max=1., qv_min=0., qv_max=.01, hydrometeor_min=0.)
    files = {'0': {'path': 'cuda-history.0.nc'}}
    if final:
        files['1'] = {'path': 'cuda-history.1.nc'}
    for descriptor in files.values():
        (fc / descriptor['path']).write_bytes(b'unit receipt artifact')
    (fc / 'cuda-v841-forecast-receipt.json').write_text(json.dumps(dict(
        status='passed', forecast=dict(walls={'steps': completed}, step_health=[row] if health else [],
                                      snapshot_files=files, refusal=None))))
    return dict(hours=1 / 180, dt_seconds=20., forecast_rc=0, forecast_wall_seconds=60.)


def test_empty_health_never_establishes_finite_state(tmp_path):
    result = proof.summarize(tmp_path, artifact(tmp_path, health=False))
    assert result['status'] == 'INCOMPLETE'
    assert 'per-step per-rank health coverage is incomplete' in result['errors']


def test_truncated_success_label_does_not_pass(tmp_path):
    result = proof.summarize(tmp_path, artifact(tmp_path, completed=0))
    assert result['status'] == 'INCOMPLETE'
    assert result['forecast_seconds_per_hour'] is None
    assert result['forecast_card_hours'] is None


def test_missing_final_frame_does_not_pass(tmp_path):
    assert proof.summarize(tmp_path, artifact(tmp_path, final=False))['status'] == 'INCOMPLETE'


def test_failed_health_does_not_pass(tmp_path):
    assert proof.summarize(tmp_path, artifact(tmp_path, finite=False))['status'] == 'INCOMPLETE'


def test_complete_receipt_hashes_actual_files_and_keeps_cost_unknown(tmp_path):
    result = proof.summarize(tmp_path, artifact(tmp_path))
    assert result['status'] == 'PASS'
    assert len(result['history_files']) == 2
    assert result['forecast_card_hours'] == 1 / 60
    assert result['forecast_whole_box_usd'] is None


def test_cost_uses_whole_box_price_without_dividing_by_cards(tmp_path):
    manifest = artifact(tmp_path)
    manifest['whole_box_usd_per_hour'] = 12.
    assert proof.summarize(tmp_path, manifest)['forecast_whole_box_usd'] == .2


def test_arm_keeps_case_and_schedule_identical_when_only_rank_count_changes(tmp_path):
    arm_spec = importlib.util.spec_from_file_location('operational_arm', Path(__file__).parents[1] / 'tools' / 'operational_arm.py')
    arm = importlib.util.module_from_spec(arm_spec)
    arm_spec.loader.exec_module(arm)
    case = dict(hex_tree='/hex/tree', engine='/engine', mesh='mesh', grid='/cache/grid',
                static='/cache/static', init='/cache/init', lbc='/cache/lbc', start='2026-10-03_12:00:00',
                init_source='declared source', dt_seconds=20, forecast_flags=['--convection', 'off'],
                layouts={'4': dict(layouts='/cache/layout4', part='/cache/part4')})
    one = arm.forecast_command(case, tmp_path, 1, 18)
    four = arm.forecast_command(case, tmp_path, 4, 18)
    assert four[:len(one)] == one
    assert four[len(one):] == ['--samehost-devices', '0,1,2,3', '--samehost-layouts', '/cache/layout4',
                              '--samehost-part', '/cache/part4', '--samehost-fingerprint-every', '3240']


def test_render_uses_native_window_and_shipped_door(tmp_path):
    post_spec = importlib.util.spec_from_file_location('operational_post', Path(__file__).parents[1] / 'tools' / 'operational_post.py')
    post = importlib.util.module_from_spec(post_spec)
    post_spec.loader.exec_module(post)
    command = post.render_command(dict(static='/cache/static', init='/cache/init', start='cycle'), tmp_path, [tmp_path / 'fc.nc'])
    assert command[1:5] == ['-m', 'hexcore.cli', 'render', '--history']
    assert command[command.index('--window') + 1] == 'mesh'
    assert command[command.index('--size') + 1] == '1200x900'
