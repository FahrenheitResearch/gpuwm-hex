"""Require all Rust-rendered panels before retiring a forecast's raw output.

These checks establish provenance and completeness. Looking at the images is
a separate delivery review; no numerical threshold substitutes for it.
"""
import hashlib
import json
import math
from pathlib import Path

MAPS = frozenset({
    '2m-temperature', 'precip-total-since-start', 'mslp',
    'mslp-isobars-10m-wind', '500hPa-height',
    '500hPa-height-contours-wind', '10m-wind', 'composite-reflectivity'
})
LEADS = (6, 12, 24)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_sheets(root, deck, native_publication, difference_table, compared_outputs):
    root = Path(root).resolve()
    manifest = root / 'sheets.json'
    data = json.loads(manifest.read_text())
    assert data['schema'] == 'mpas-sheets/v1'
    assert data['deck_sha256'] == sha(deck), 'sheets belong to a different deck'
    assert Path(data['native_publication']).resolve() == Path(native_publication).resolve()
    assert Path(data['difference_table']).resolve() == Path(difference_table).resolve()
    renderer = Path(data['renderer'])
    assert data['renderer_sha256'] == sha(renderer), 'renderer changed'
    assert sorted(row['lead'] for row in data['leads']) == list(LEADS), 'incomplete lead coverage'
    native = json.loads(Path(native_publication).read_text())
    images = {}
    for row in data['leads']:
        assert not row['failures'] and not row['skips'], 'incomplete rendered products'
        assert row['native_sha256'] == native['files'][row['native_raw']]
        assert row['hex_sha256'] == sha(row['hex_raw']), 'hex output changed after comparison'
        assert Path(row['hex_raw']).name == row['native_raw']
        assert row['hex_sha256'] == compared_outputs[row['native_raw']], 'sheets must use the word-compared hex bytes'
        assert len(row['panels']) == len(MAPS) and {p['map'] for p in row['panels']} == MAPS
        for panel in row['panels']:
            path = (root / panel['file']).resolve()
            assert path.is_relative_to(root), 'panel escapes its sheet directory'
            assert path.is_file() and path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
            assert path.stat().st_size < 10_000_000, 'panel exceeds receipt limit'
            assert path not in images, 'repeated image masquerades as separate panels'
            # The shared Rust renderer reports decimal strings. Accept its
            # actual metadata without changing the original manifest or scales.
            metrics = [panel[k] for k in ('max_abs', 'scale_half_range', 'white_below')]
            assert all(isinstance(v, (int, float, str)) and not isinstance(v, bool) for v in metrics)
            try:
                maximum, half_range, white_below = map(float, metrics)
            except (TypeError, ValueError) as error:
                raise AssertionError('renderer metrics must be numeric') from error
            assert math.isfinite(maximum) and maximum >= 0
            assert math.isfinite(half_range) and half_range > 0
            assert math.isfinite(white_below) and 0 < white_below <= half_range
            images[path] = sha(path)
    return dict(schema='r14-sheet-provenance-v1', sheets_sha256=sha(manifest),
                native_publication_sha256=sha(native_publication),
                difference_table_sha256=sha(difference_table),
                images={str(p.relative_to(root)): digest for p, digest in images.items()},
                visual_review='required separately: open every PNG before delivery')
