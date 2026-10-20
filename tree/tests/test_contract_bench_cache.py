"""A partial contract must not be labelled as a completed warm predecessor."""
import importlib.util
import json
from pathlib import Path
import pytest


def helper():
    path = Path(__file__).resolve().parents[1] / 'tools/bench_contract.py'
    spec = importlib.util.spec_from_file_location('contract_bench', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.complete_cache


def test_partial_contract_cannot_mark_warm(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    result = tmp_path / 'contract.json'
    result.write_text(json.dumps({'summary': {'decks': 7, 'decks_passed': 7,
        'kernels_covered': 21, 'kernels_declared': 22,
        'all_dual_run_identical': True, 'all_controls_have_teeth': True}}))
    with pytest.raises(ValueError, match='incomplete or failed'):
        helper()(str(cache), str(result), str(tmp_path))
    assert not (cache / 'complete.json').exists()


def test_cache_cannot_escape_lane(tmp_path):
    lane = tmp_path / 'lane'
    lane.mkdir()
    with pytest.raises(ValueError, match='below the current lane'):
        helper()(str(tmp_path / 'other'), 'unused.json', str(lane))
