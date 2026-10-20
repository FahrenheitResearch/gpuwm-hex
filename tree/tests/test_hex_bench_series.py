"""Stub controller replay checks sequence and resumption, not forecast timing."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def test_series_resumes_without_repeating_completed_cold_runs(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'tools/hex_timeline/measure_series.py'
    spec = importlib.util.spec_from_file_location('hex_series', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'receipts').mkdir()
    calls = []

    def run(command, *, cwd, env):
        temperature = command[4]
        arm = 'A' if env['BENCH_REFERENCE'] == '1' else 'B'
        calls.append((arm, temperature))
        output = tmp_path / 'receipts' / f'fixture-{temperature}-{len(calls)}'
        output.mkdir()
        (output / 'receipt.json').write_text(json.dumps(dict(proxy_ff=.5, proxy_wall_seconds=2., stepping_wall_seconds=1.)))
        (output / 'case-summary.json').write_text(json.dumps(dict(door_wall_seconds=1., steps=2,
            final_state_sha256='test', history_sha256={})))
        assert command[-1] == 'hex-' + arm.lower()
        assert env['PYTHONPATH'] == str(cwd / 'src')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, 'run', run)
    argv = ['series', '--lane', str(tmp_path), '--case', 'fixture', '--controller', str(tmp_path / 'controller')]
    monkeypatch.setattr(sys, 'argv', argv + ['--stop-after', '2'])
    assert module.main() == 0
    assert calls == [('B', 'COLD'), ('A', 'COLD')]
    monkeypatch.setattr(sys, 'argv', argv)
    assert module.main() == 0
    assert calls == [('B', 'COLD'), ('A', 'COLD')] + [(arm, 'WARM') for arm in 'ABBA' * 3]
    assert module.main() == 0
    assert len(calls) == 14
