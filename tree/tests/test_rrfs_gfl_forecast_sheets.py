"""A bitwise receipt cannot authenticate stale, hidden or incomplete image evidence."""
import copy
import datetime
import importlib.util
import json
from pathlib import Path

import pytest

from hexcore.verify_mpas import digest


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    helpers = Path(__file__).parent / 'rrfs_gfl'
    spec = importlib.util.spec_from_file_location('gfl_forecast_reproducer', helpers / 'reproduce_forecasts.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    deck = tmp_path / 'deck.json'
    start = datetime.datetime(2023, 3, 31, 12)
    deck.write_text(json.dumps({'date': start.strftime('%Y-%m-%d_%H:%M:%S')}))
    native = tmp_path / 'PUBLICATION.json'
    table = tmp_path / 'difference-table.json'
    table.write_text('{}')
    monkeypatch.setattr(module, 'SHEET_TABLE_SHA256', digest(table))
    renderer = tmp_path / 'rw_wrfbatch'
    renderer.write_bytes(b'test renderer authority')
    files, compared, rows = {}, {}, []
    for lead in (6, 12, 24):
        name = 'output.' + (start + datetime.timedelta(hours=lead)).strftime('%Y-%m-%d_%H.%M.%S') + '.nc'
        files[name] = f'native-word-hash-{lead}'
        compared[name] = f'hex-word-hash-{lead}'
        panels = []
        for product, scale in module.SHEET_SCALES.items():
            image = tmp_path / f'f{lead:03d}' / (product + '.png')
            image.parent.mkdir(exist_ok=True)
            image.write_bytes(b'\x89PNG\r\n\x1a\nunit-test-placeholder')
            panels.append({'map': product, 'file': str(image.relative_to(tmp_path)),
                           'max_abs': 0., 'scale_half_range': scale, 'white_below': scale/5})
        rows.append({'lead': lead, 'hex_raw': f'/retired/forecast/{name}',
                     'hex_sha256': compared[name], 'native_raw': name,
                     'native_sha256': files[name], 'failures': [], 'skips': [], 'panels': panels})
    native.write_text(json.dumps({'files': files}))
    sheets = {'schema': 'mpas-sheets/v1', 'deck': str(deck), 'deck_sha256': digest(deck),
              'native_publication': str(native), 'difference_table': str(table),
              'renderer': str(renderer), 'renderer_sha256': digest(renderer), 'leads': rows}
    manifest = tmp_path / 'sheets.json'
    receipt = {'difference_sheets': {'path': str(manifest), 'failures': []},
               'provenance': {'hex_output_sha256': compared}}
    return module, receipt, sheets, deck, native, manifest


def seal(receipt, sheets, manifest):
    manifest.write_text(json.dumps(sheets))
    receipt['difference_sheets']['sha256'] = digest(manifest)


def test_complete_sheets_still_require_visual_review(evidence):
    module, receipt, sheets, deck, native, manifest = evidence
    seal(receipt, sheets, manifest)
    result = module.validate_difference_sheets(receipt, deck, native)
    assert result['visual_review_required'] is True
    assert len(result['images']) == 24
    assert all(image['sha256'] == digest(Path(image['path'])) for image in result['images'])


@pytest.mark.parametrize('fault', ['stale-deck', 'stale-hex', 'stale-native', 'missing-lead',
    'duplicate-map', 'missing-image', 'path-escape', 'hidden-scale', 'hidden-band',
    'missing-measurement', 'nan-measurement', 'renderer-changed', 'table-changed',
    'manifest-changed', 'skipped-product', 'failed-renderer', 'wrong-lead'])
def test_rejects_unbound_or_hidden_sheet_evidence(evidence, fault):
    module, receipt, sheets, deck, native, manifest = evidence
    receipt, sheets = copy.deepcopy(receipt), copy.deepcopy(sheets)
    row = sheets['leads'][0]
    panel = row['panels'][0]
    if fault == 'stale-deck': sheets['deck_sha256'] = 'another-deck'
    elif fault == 'stale-hex': row['hex_sha256'] = 'another-run'
    elif fault == 'stale-native': row['native_sha256'] = 'another-reference'
    elif fault == 'missing-lead': sheets['leads'].pop()
    elif fault == 'duplicate-map': row['panels'][1]['map'] = panel['map']
    elif fault == 'missing-image': (manifest.parent / panel['file']).unlink()
    elif fault == 'path-escape': panel['file'] = '../unowned.png'
    elif fault == 'hidden-scale': panel['scale_half_range'] *= 100
    elif fault == 'hidden-band': panel['white_below'] *= 100
    elif fault == 'missing-measurement': panel['max_abs'] = None
    elif fault == 'nan-measurement': panel['max_abs'] = float('nan')
    elif fault == 'renderer-changed': Path(sheets['renderer']).write_bytes(b'changed')
    elif fault == 'table-changed': Path(sheets['difference_table']).write_text('{"changed":true}')
    elif fault == 'skipped-product': row['skips'] = ['missing required field']
    elif fault == 'failed-renderer': receipt['difference_sheets']['failures'] = ['renderer exited 1']
    elif fault == 'wrong-lead': row['hex_raw'] = sheets['leads'][1]['hex_raw']
    seal(receipt, sheets, manifest)
    if fault == 'manifest-changed': manifest.write_text('{}')
    with pytest.raises((ValueError, OSError)):
        module.validate_difference_sheets(receipt, deck, native)
