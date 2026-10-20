"""Adversarial artifact metadata fixtures; no weather images or forecasts made."""
import datetime
import json
from pathlib import Path

import pytest

from rrfs_sheet_contract import MAPS, audit_sheets
from rrfs_proof_contract import digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'path': str(path), 'sha256': digest(path)}


@pytest.fixture
def evidence(tmp_path):
    deck = {'id': 'fixture', 'date': '2000-01-01_00:00:00'}
    baseline = {'provenance': {'deck_sha256': 'fixture-deck', 'hex_output_sha256': {}}}
    folder = tmp_path/'proof'
    sheetroot = folder/'sheets'
    renderer = folder/'renderer'
    renderer.parent.mkdir(parents=True)
    renderer.write_text('artifact fixture, not an executable')
    table = folder/'table.json'
    write(table, {})
    publication = {'files': {}}
    sheets = {'schema': 'mpas-sheets/v1', 'deck_sha256': 'fixture-deck',
              'renderer': str(renderer), 'renderer_sha256': digest(renderer),
              'difference_table': str(table), 'native_publication': str(folder/'PUBLICATION.json'),
              'leads': []}
    row = {'deck': 'fixture', 'difference_table_sha256': digest(table),
           'evolution': 'Contract fixture only; no scientific inspection claim.', 'images': []}
    for lead in (6, 12, 24):
        valid = datetime.datetime(2000, 1, 1) + datetime.timedelta(hours=lead)
        name = 'output.' + valid.strftime('%Y-%m-%d_%H.%M.%S') + '.nc'
        baseline['provenance']['hex_output_sha256'][name] = 'fixture-hex'
        publication['files'][name] = 'fixture-native'
        frame = dict(lead=lead, valid=valid.isoformat(), hex_raw=name, hex_sha256='fixture-hex',
                     native_raw=name, native_sha256='fixture-native', failures=[], skips=[], panels=[])
        for product in sorted(MAPS):
            rel = f'f{lead:03d}/{product}.png'
            path = sheetroot/rel
            path.parent.mkdir(parents=True, exist_ok=True)
            # Non-image bytes exercise binding/inventory checks, not rendering.
            path.write_text('artifact metadata fixture; this is not a PNG')
            row['images'].append(dict(path=rel, sha256=digest(path),
                                      observation='Metadata fixture; no weather shown.'))
            frame['panels'].append(dict(map=product, file=rel, max_abs=0,
                                       scale_half_range=1, white_below=0.2))
        sheets['leads'].append(frame)
    pubref = write(folder/'PUBLICATION.json', publication)
    baseline['provenance']['native_manifest_sha256'] = pubref['sha256']
    receiptpath = folder/'forecast.json'
    write(receiptpath, baseline)
    row['forecast_sha256'] = digest(receiptpath)
    row['sheets'] = write(sheetroot/'sheets.json', sheets)
    bundle = {'schema': 'rrfs-sheet-inspections-v1', 'source_commit': 'fixture-commit', 'decks': [row]}
    write(folder/'sheet-inspections.json', bundle)
    return folder, {'fixture': (deck, baseline, receiptpath)}, bundle, sheets


def test_complete_artifact_metadata_contract(evidence):
    folder, receipts, _, _ = evidence
    audit_sheets(folder, receipts, 'fixture-commit')


def test_rust_decimal_metadata(evidence):
    folder, receipts, bundle, sheets = evidence
    for frame in sheets['leads']:
        for panel in frame['panels']:
            panel.update(max_abs='0.0000', scale_half_range='5', white_below='0.2')
    row = bundle['decks'][0]
    row['sheets'] = write(Path(row['sheets']['path']), sheets)
    write(folder/'sheet-inspections.json', bundle)
    audit_sheets(folder, receipts, 'fixture-commit')


@pytest.mark.parametrize('value', (True, None, [], {}, '', 'bad', 'NaN', 'Infinity', '-1'))
def test_reject_invalid_rust_decimal_metadata(evidence, value):
    folder, receipts, bundle, sheets = evidence
    sheets['leads'][0]['panels'][0]['max_abs'] = value
    row = bundle['decks'][0]
    row['sheets'] = write(Path(row['sheets']['path']), sheets)
    write(folder/'sheet-inspections.json', bundle)
    with pytest.raises(ValueError):
        audit_sheets(folder, receipts, 'fixture-commit')


@pytest.mark.parametrize('defect', ('missing-bundle', 'wrong-commit', 'no-decks', 'wrong-deck',
    'duplicate-deck', 'changed-forecast', 'changed-image', 'missing-image-observation',
    'no-evolution', 'unreviewed-image', 'missing-lead', 'duplicate-lead', 'wrong-time',
    'incomplete-products', 'renderer-failure', 'skipped-product', 'wrong-hex', 'wrong-native',
    'wrong-publication', 'wrong-deck-hash', 'changed-scale', 'nonfinite-maximum'))
def test_reject_incomplete_or_unrelated_visual_evidence(evidence, defect):
    folder, receipts, bundle, sheets = evidence
    row, frame = bundle['decks'][0], sheets['leads'][0]
    sheetpath = Path(row['sheets']['path'])
    if defect == 'missing-bundle':
        (folder/'sheet-inspections.json').unlink()
        with pytest.raises(FileNotFoundError): audit_sheets(folder, receipts, 'fixture-commit')
        return
    if defect == 'wrong-commit': bundle['source_commit'] = 'old'
    elif defect == 'no-decks': bundle['decks'] = []
    elif defect == 'wrong-deck': row['deck'] = 'unrelated'
    elif defect == 'duplicate-deck': bundle['decks'].append(dict(row))
    elif defect == 'changed-forecast': row['forecast_sha256'] = 'unrelated'
    elif defect == 'changed-image': (sheetpath.parent/row['images'][0]['path']).write_text('changed')
    elif defect == 'missing-image-observation': row['images'][0]['observation'] = ''
    elif defect == 'no-evolution': row['evolution'] = ''
    elif defect == 'unreviewed-image': (sheetpath.parent/'extra.png').write_text('non-image fixture')
    elif defect == 'missing-lead': sheets['leads'].pop()
    elif defect == 'duplicate-lead': sheets['leads'].append(dict(frame))
    elif defect == 'wrong-time': frame['valid'] = 'unrelated'
    elif defect == 'incomplete-products': frame['panels'].pop()
    elif defect == 'renderer-failure': frame['failures'] = ['failure']
    elif defect == 'skipped-product': frame['skips'] = ['missing']
    elif defect == 'wrong-hex': frame['hex_sha256'] = 'unrelated'
    elif defect == 'wrong-native': frame['native_sha256'] = 'unrelated'
    elif defect == 'wrong-publication': receipts['fixture'][1]['provenance']['native_manifest_sha256'] = 'other'
    elif defect == 'wrong-deck-hash': sheets['deck_sha256'] = 'unrelated'
    elif defect == 'changed-scale': row['difference_table_sha256'] = 'other'
    elif defect == 'nonfinite-maximum': frame['panels'][0]['max_abs'] = float('nan')
    row['sheets'] = write(sheetpath, sheets)
    write(folder/'sheet-inspections.json', bundle)
    with pytest.raises(ValueError): audit_sheets(folder, receipts, 'fixture-commit')
