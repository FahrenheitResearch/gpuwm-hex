"""Validate custody and completeness of actual Rust difference sheets.

This reads receipts and hashes artifacts; weather processing stays in Rust.
A manifest with skipped products or absent panels cannot prove eight maps.
Human image review is recorded separately and is never inferred from zeroes.
"""
import hashlib
import json
from decimal import Decimal
from pathlib import Path

MAPS = frozenset(('2m-temperature', 'precip-total-since-start', 'mslp',
                 'mslp-isobars-10m-wind', '500hPa-height',
                 '500hPa-height-contours-wind', '10m-wind', 'composite-reflectivity'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def qualify(directory, expected, value=None):
    directory = Path(directory).resolve()
    if value is None:
        value = json.loads((directory / 'sheets.json').read_text())
    assert value['schema'] == 'mpas-sheets/v1'
    assert value['deck_sha256'] == expected['deck_sha256'], 'sheet deck differs'
    assert value['renderer_sha256'] == expected['renderer_sha256'], 'renderer differs'
    rows = value['leads']
    assert [r['lead'] for r in rows] == expected['leads'], 'requested leads differ'
    images = []
    for row in rows:
        assert not row['failures'] and not row['skips'], 'missing or failed weather products'
        assert len(row['panels']) == len(MAPS)
        assert {p['map'] for p in row['panels']} == MAPS, 'eight-map inventory differs'
        pins = expected['raw'][str(row['lead'])]
        assert row['hex_sha256'] == pins['hex'], 'actual Hex output differs'
        assert row['native_sha256'] == pins['native'], 'native publication differs'
        for panel in row['panels']:
            assert not isinstance(panel['max_abs'], bool) and Decimal(str(panel['max_abs'])) == 0, 'rendered cells differ'
            assert panel['rule'] == 'table', 'scale adapts to hide a difference'
            scale = expected['scales'][panel['map']]
            assert Decimal(str(panel['scale_half_range'])) == Decimal(str(scale['half_range'])), 'fixed scale differs'
            assert Decimal(str(panel['white_below'])) == Decimal(str(scale['step'])), 'white band differs'
            assert panel['file'] == f"f{row['lead']:03d}/{panel['map']}.png", 'image named for another map or lead'
            path = (directory / panel['file']).resolve()
            assert path.is_relative_to(directory) and path.is_file(), 'image absent or outside evidence'
            assert path.stat().st_size < 10_000_000, 'image exceeds receipt limit'
            assert path.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n', 'image is not PNG'
            images.append({'file': panel['file'], 'sha256': digest(path), 'map': panel['map'], 'lead': row['lead']})
    return {'schema': 'i4-sheet-custody/v1', 'sheets_sha256': digest(directory / 'sheets.json'),
            'complete_eight_products_three_leads': True, 'all_panel_max_abs_zero': True,
            'fixed_scales_authenticated': True, 'images': images,
            'human_inspection_performed': False}


def replay_faults(directory, expected):
    from copy import deepcopy
    original = json.loads((Path(directory) / 'sheets.json').read_text())
    controls = []
    def refuses(name, mutate):
        value = deepcopy(original)
        mutate(value)
        try:
            qualify(directory, expected, value)
        except (AssertionError, KeyError, ValueError):
            controls.append(name)
        else:
            raise AssertionError('corrupted visual evidence accepted: ' + name)
    refuses('wrong-deck', lambda v: v.__setitem__('deck_sha256', 'wrong'))
    refuses('wrong-renderer', lambda v: v.__setitem__('renderer_sha256', 'wrong'))
    refuses('missing-lead', lambda v: v['leads'].pop())
    refuses('wrong-lead', lambda v: v['leads'][0].__setitem__('lead', -1))
    refuses('render-failure', lambda v: v['leads'][0]['failures'].append('failure'))
    refuses('skipped-map', lambda v: v['leads'][0]['skips'].append('missing native product'))
    refuses('wrong-hex-bytes', lambda v: v['leads'][0].__setitem__('hex_sha256', 'wrong'))
    refuses('wrong-native-bytes', lambda v: v['leads'][0].__setitem__('native_sha256', 'wrong'))
    refuses('absent-map', lambda v: v['leads'][0]['panels'].pop())
    refuses('duplicate-map', lambda v: v['leads'][0]['panels'][0].__setitem__('map', v['leads'][0]['panels'][1]['map']))
    for key, bad in (('max_abs', '0.0001'), ('max_abs', True),
                     ('rule', 'percentile'), ('scale_half_range', '999'),
                     ('white_below', '999'), ('file', '../../outside.png')):
        refuses('corrupt-' + key + '-' + str(bad),
                lambda v, k=key, b=bad: v['leads'][0]['panels'][0].__setitem__(k, b))
    return controls
