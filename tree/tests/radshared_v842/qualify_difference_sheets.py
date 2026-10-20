"""Authenticate whole-forecast image evidence without grading rounded scores.

Only metadata and file hashes are read here. Weather conversion/rendering stay
in the integrator's Rust tools. A successful renderer that skipped products
cannot satisfy the eight-product, three-lead delivery requirement.
"""
import argparse
import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path

# Published integrator difference-sensitive.json; a wider table needs an owner publication.
FINE_TABLE_SHA256 = 'a180ca0b0939f44f166182e585d30cdd87ba4aea4ca7b68768e0ff1a7e4cbdbe'

PRODUCTS = {
    '2m-temperature': '2m_temperature',
    'precip-total-since-start': 'total_qpf',
    'mslp-isobars-10m-wind': 'mslp_10m_winds',
    'mslp': 'var_mslp',
    '500hPa-height-contours-wind': '500mb_height_winds',
    '500hPa-height': 'var_geopotential_height_500hpa',
    '10m-wind': '10m_wind_speed_and_direction',
    'composite-reflectivity': 'composite_reflectivity',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def qualify(sheets_path, attempt_path, inspection_path, table_path):
    sheets_path = Path(sheets_path)
    attempt = read(attempt_path)
    original = attempt['original']
    assert original['process_count'] == attempt['process_count'] == 1
    assert attempt['render_exit'] == 0 and attempt['science_and_rendering_unchanged']
    assert sha(sheets_path) == attempt['sheets_json_sha256']
    for key in ('deck', 'original_receipt'):
        assert sha(original[key]) == original[key + '_sha256']
    publication = original['native_publication']
    assert sha(publication['path']) == publication['sha256']
    forecast = read(original['original_receipt'])
    assert forecast['evidence_kind'] == 'FORECAST' and forecast['model_execution']
    assert forecast['process_count'] == 1
    assert forecast['passed'] == original['original_passed']
    assert forecast.get('acceptance_tier') == original['original_acceptance_tier']
    deck = read(original['deck'])
    assert forecast['provenance']['deck_sha256'] == original['deck_sha256']
    assert forecast['deck'] == deck['id']
    assert all(forecast[key] == deck[key] for key in ('mesh', 'date', 'process_count'))
    assert deck['native']['sha256'] == publication['sha256']
    for name, binding in deck['inputs'].items():
        assert forecast['provenance']['inputs'][name]['sha256'] == binding['sha256']
    sheets = read(sheets_path)
    assert sheets['schema'] == 'mpas-sheets/v1'
    staged = attempt['staged_sha256']
    assert sha(sheets['deck']) == sheets['deck_sha256'] == staged['render-deck.json']
    assert sha(sheets['native_publication']) == publication['sha256']
    assert sha(sheets['renderer']) == sheets['renderer_sha256'] == staged['tools/rw_wrfbatch']
    assert sha(table_path) == staged['tools/difference-sensitive.json'] == FINE_TABLE_SHA256
    assert sha(sheets['difference_table']) == sha(table_path)
    rendered_deck = read(sheets['deck'])
    command = rendered_deck['hex_command']
    original_grid = deck['hex_command'][deck['hex_command'].index('--grid') + 1]
    grid_hash = forecast['provenance']['auxiliary_inputs'][Path(original_grid).name]['sha256']
    assert sha(command[command.index('--grid') + 1]) == staged['inputs/mesh.nc'] == grid_hash
    assert sha(rendered_deck['inputs']['init']['path']) == staged['inputs/init.nc'] == deck['inputs']['init']['sha256']
    command[command.index('--grid') + 1] = deck['hex_command'][deck['hex_command'].index('--grid') + 1]
    rendered_deck['inputs']['init']['path'] = deck['inputs']['init']['path']
    assert rendered_deck == deck, 'Rendering transport may relocate only mesh/init paths'
    inspection = read(inspection_path)
    assert inspection['sheets_json_sha256'] == sha(sheets_path)
    assert inspection['attempt_sha256'] == sha(attempt_path)
    reviewed = {row['file']: row for row in inspection['images']}
    assert len(reviewed) == len(inspection['images']), 'Duplicate image reviews'
    start = datetime.datetime.strptime(deck['date'], '%Y-%m-%d_%H:%M:%S')
    leads = sheets['leads']
    assert [row['lead'] for row in leads] == [6, 12, 24]
    table = read(table_path)
    rules = {}
    for name, product in PRODUCTS.items():
        ranges = {Decimal(str(row['half_range'])) for row in table['rows'] if row['product'] == product}
        assert len(ranges) == 1
        rules[name] = ranges.pop()
    missing = []
    images = []
    for row in leads:
        assert row['valid'] == (start + datetime.timedelta(hours=row['lead'])).isoformat()
        raw_name = 'output.' + datetime.datetime.fromisoformat(row['valid']).strftime('%Y-%m-%d_%H.%M.%S') + '.nc'
        assert row['native_raw'] == raw_name
        assert sha(row['hex_raw']) == row['hex_sha256'] == original['retained_sha256'][raw_name]
        assert row['hex_sha256'] == forecast['provenance']['hex_output_sha256'][raw_name]
        assert row['native_sha256'] == read(publication['path'])['files'][raw_name]
        assert sha(Path(sheets['native_publication']).parent / raw_name) == row['native_sha256']
        assert not row['failures']
        panels = row['panels']
        names = [panel['map'] for panel in panels]
        assert len(names) == len(set(names)) and set(names) <= set(PRODUCTS)
        missing.append(dict(lead=row['lead'], maps=sorted(set(PRODUCTS) - set(names)), skips=row['skips']))
        for panel in panels:
            assert panel['file'] == f"f{row['lead']:03d}/{panel['map']}.png"
            path = sheets_path.parent / panel['file']
            review = reviewed[panel['file']]
            assert review['opened'] is True and review['observation'].strip()
            assert sha(path) == review['sha256']
            assert panel['rule'] == 'table'
            half = rules[panel['map']]
            assert Decimal(panel['scale_half_range']) == half
            assert Decimal(panel['white_below']) == 2 * half / Decimal(str(table['bands']))
            assert panel['max_abs'] is not None  # Diagnostic only; never an acceptance threshold.
            images.append(dict(file=panel['file'], sha256=sha(path), observation=review['observation']))
    assert set(reviewed) == {row['file'] for row in images}
    complete = len(images) == 24 and all(not row['maps'] and not row['skips'] for row in missing)
    return dict(schema='p21-difference-sheets-qualification/v1', process_count=1,
        rendered_images=len(images), required_images=24, all_rendered_images_opened=True,
        complete_required_products=complete, missing_products=missing, images=images,
        fixed_fine_scales_authenticated=True, scores_are_diagnostics_only=True,
        original_forecast_passed=forecast['passed'], original_acceptance_unchanged=True,
        numerical_acceptance_changed=False, forecast_acceptance=False,
        sheets_json_sha256=sha(sheets_path), attempt_sha256=sha(attempt_path),
        inspection_sha256=sha(inspection_path), difference_table_sha256=sha(table_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sheets', 'attempt', 'inspection', 'table', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.receipt.exists(), 'Keep prior attempts immutable'
    value = qualify(args.sheets, args.attempt, args.inspection, args.table)
    args.receipt.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({key: value[key] for key in ('rendered_images', 'required_images', 'complete_required_products')}))
    return 0 if value['complete_required_products'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
