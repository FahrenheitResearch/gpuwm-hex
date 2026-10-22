"""The whole wheel copies native output from the configured scratch target."""
import importlib.util
from pathlib import Path
import sys


def test_scorer_install_uses_external_cargo_target_and_slot_budget(tmp_path, monkeypatch):
    recipe = Path(__file__).resolve().parents[1] / 'tools/build_obs_scorecard.py'
    spec = importlib.util.spec_from_file_location('scorer_recipe', recipe)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    # Keep the script's destination inside the test's private source tree.
    private = tmp_path / 'tree/tools/build_obs_scorecard.py'
    private.parent.mkdir(parents=True)
    monkeypatch.setattr(build, '__file__', str(private))
    project = tmp_path / 'project/Cargo.toml'
    monkeypatch.setattr(build, 'write_project', lambda *args: project)
    external = tmp_path / 'scratch/target'
    (external / 'release').mkdir(parents=True)
    binary = 'hex_obs_scorecard.exe' if sys.platform == 'win32' else 'hex_obs_scorecard'
    (external / 'release' / binary).write_bytes(b'compiled scorer in configured target')
    monkeypatch.setenv('CARGO_TARGET_DIR', str(external))
    monkeypatch.setenv('CARGO_BUILD_JOBS', '14')
    calls = []
    monkeypatch.setattr(build.subprocess, 'call', lambda command, **kwargs: calls.append(command) or 0)
    monkeypatch.setattr(sys, 'argv', [str(private), '--engine', str(tmp_path / 'engine'),
                                   '--out', str(tmp_path / 'project'), '--install'])
    assert build.main() == 0
    command = calls[0]
    assert command[command.index('--jobs') + 1] == '14'
    assert command[command.index('--target-dir') + 1] == str(external)
    installed = tmp_path / 'tree/src/hexcore/libexec' / binary
    assert installed.read_bytes() == (external / 'release' / binary).read_bytes()
