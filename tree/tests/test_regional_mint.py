"""Runtime class admission requires intact, independent, finite forecast evidence."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from hexcore import regional_mint as mint
from hexcore.cuda_backend.regional_admission import (
    RegionalClassKey, admitted_class, admitted_class_for_key,
)
from hexcore.drivers import run_cuda_regional_mint as driver


def evidence(tmp_path):
    key = RegionalClassKey.build(boundary_zone_width=7, n_vert_levels=55,
                                finest_edge_m=1234.567, dt_seconds=5, kernel_set='a' * 64)
    run = dict(schema='gpuwm-hex.regional-mint-run/v1', instrument='run_cuda_regional_mint',
               pid=1, minted_without_anchor_gate=True, numerics='strict', key=key.as_dict(),
               steps=driver.STEPS, config_sha256='b' * 64, n_cells=100,
               bdy_mask_sha256='c' * 64, card='test-card',
               driver_sha256=hashlib.sha256(Path(driver.__file__).read_bytes()).hexdigest(),
               frames=[dict(step=step, finite={name: True for name in driver.FIELDS},
                            field_sha256={name: 'd' * 64 for name in driver.FIELDS})
                       for step in range(driver.STEPS + 1)])
    pair = [run, copy.deepcopy(run)]
    pair[1]['pid'] = 2
    document = dict(schema=mint.SCHEMA, key=key.as_dict(), class_id=mint.class_id(key),
                    parent='parent', mesh_row='cull', generated_utc='2026-10-20T00:00:00Z', runs=[])
    for index, item in enumerate(pair):
        path = tmp_path / f'run{index}.json'
        path.write_text(json.dumps(item))
        document['runs'].append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    path = tmp_path / 'class.mint.json'
    path.write_text(json.dumps(document))
    return key, pair, path


def test_intact_runtime_class_is_presented_to_forecast_admission(tmp_path, monkeypatch):
    key, pair, path = evidence(tmp_path)
    monkeypatch.setenv('GPUWM_HEX_REGIONAL_CONTRACT_DIR', str(tmp_path))
    assert admitted_class_for_key(key).class_id == mint.class_id(key)
    assert admitted_class(mint.class_id(key)).key == key
    # Editing a measured run lapses the class in both lookup paths.
    (tmp_path / 'run1.json').write_text('{}')
    assert admitted_class_for_key(key) is None
    assert admitted_class(mint.class_id(key)) is None


@pytest.mark.parametrize('defect', ['same_process', 'nonfinite', 'different_field', 'different_key', 'missing_step'])
def test_pair_rejects_unmeasured_or_different_runs(tmp_path, defect):
    key, pair, path = evidence(tmp_path)
    if defect == 'same_process':
        pair[1]['pid'] = pair[0]['pid']
    elif defect == 'nonfinite':
        pair[1]['frames'][3]['finite']['theta'] = False
    elif defect == 'different_field':
        pair[1]['frames'][3]['field_sha256']['theta'] = 'e' * 64
    elif defect == 'different_key':
        pair[1]['key']['kernel_set_sha256'] = 'e' * 64
    else:
        pair[1]['frames'].pop()
    with pytest.raises(ValueError):
        mint.checked_pair(pair, key)


def test_door_mints_fresh_key_and_passes_class_to_geometry_deck(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hexcore import regional_door as door
    from hexcore.cuda_backend import regional_admission as admission

    key, pair, path = evidence(tmp_path)
    generated = dict(cull=dict(grid='grid.nc', static='static.nc', lbc_dir='lbc', row='new-cull'),
                     vertical=dict(n_vert_levels=55), parent=dict(dt_seconds=5))
    value = dict(paths=dict(contract=str(tmp_path), init='init.nc'), start_time='start')
    monkeypatch.setattr(door, '_generated', lambda value: generated)
    monkeypatch.setattr(door, 'regional_class_key', lambda *a, **kw: key)
    monkeypatch.setattr(admission, 'admitted_class_for_key', lambda key: None)
    calls = []

    def earn(measured, mesh, plan, status):
        calls.append(measured)
        return mint.read_class(path)

    def deck(command, **kwargs):
        assert command[command.index('--class-id') + 1] == mint.class_id(key)
        (tmp_path / 'contract.json').write_text('{}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mint, 'mint', earn)
    monkeypatch.setattr(door.subprocess, 'run', deck)
    result = door._stage_contract(None, value, SimpleNamespace(note=lambda line: None))
    assert calls == [key]
    assert len(result['mint_receipts']) == 2
    assert result['class_id'] == mint.class_id(key)
