"""Verify that long-run rendering keeps actual initial and final forecast files."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'operational_post', Path(__file__).parents[1] / 'tools/operational_post.py')
post = importlib.util.module_from_spec(spec)
spec.loader.exec_module(post)


def test_long_forecast_endpoint_render_uses_actual_terminal_frame():
    names = ['cuda-history.2026-10-11_00.00.00.nc',
             'cuda-history.2026-10-07_12.00.00.nc',
             'cuda-history.2026-10-08_06.00.00.nc']
    proof = {'history_files': dict.fromkeys(names)}
    run = Path('/forecast')
    frames = post.selected_histories(run, proof, True)
    assert [p.name for p in frames] == [names[1], names[0]]
    command = post.render_command({'static': 'static.nc', 'init': 'init.nc',
                                   'start': '2026-10-07_12:00:00',
                                   'products': '2m_temperature,total_qpf'}, run, frames)
    start = command.index('--history') + 1
    assert command[start:command.index('--mesh')] == list(map(str, frames))
    assert '--window' in command and command[command.index('--size') + 1] == '1200x900'


def test_default_retains_all_frames_and_single_endpoint_is_not_duplicated():
    run = Path('/forecast')
    proof = {'history_files': {'b.nc': {}, 'a.nc': {}, 'c.nc': {}}}
    assert [p.name for p in post.selected_histories(run, proof)] == ['a.nc', 'b.nc', 'c.nc']
    assert [p.name for p in post.selected_histories(run, {'history_files': {'a.nc': {}}}, True)] == ['a.nc']
