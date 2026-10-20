"""Image evidence faults; synthetic metadata is never forecast proof."""
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('p21_sheet_custody',
    Path(__file__).parent / 'radshared_v842/qualify_difference_sheets.py')
sheets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sheets)


@pytest.fixture
def complete_metadata(tmp_path, monkeypatch):
    def write(name, obj):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))
        return path

    date = '2000-01-01_00:00:00'
    init = tmp_path / 'relocated-init'
    grid = tmp_path / 'relocated-mesh'
    init.write_bytes(b'fixture initial input')
    grid.write_bytes(b'fixture mesh input')
    deck = dict(id='fixture-whole-forecast', mesh='fixture', date=date, process_count=1,
                inputs={'init': {'path': 'original-init', 'sha256': sheets.sha(init)}},
                hex_command=['run', '--grid', 'original-mesh'])
    original_deck = write('deck.json', deck)
    rendered = json.loads(json.dumps(deck))
    rendered['inputs']['init']['path'] = str(init)
    rendered['hex_command'][-1] = str(grid)
    render_deck = write('render-deck.json', rendered)
    renderer = tmp_path / 'renderer'
    renderer.write_bytes(b'fixture executable identity')
    table = write('table.json', dict(bands=10, rows=[
        dict(product=product, half_range=0.1) for product in sheets.PRODUCTS.values()]))
    monkeypatch.setattr(sheets, 'FINE_TABLE_SHA256', sheets.sha(table))
    leads, image_rows, files, hex_hashes = [], [], {}, {}
    for lead in (6, 12, 24):
        time = datetime.datetime.fromisoformat(date.replace('_', 'T')) + datetime.timedelta(hours=lead)
        name = 'output.' + time.strftime('%Y-%m-%d_%H.%M.%S') + '.nc'
        for run in ('hex', 'native'):
            p = tmp_path / run / name
            p.parent.mkdir(exist_ok=True)
            p.write_bytes(f'fixture metadata raw identity, {lead}'.encode())
        files[name] = sheets.sha(tmp_path / 'native' / name)
        hex_hashes[name] = sheets.sha(tmp_path / 'hex' / name)
        panels = []
        for product in sheets.PRODUCTS:
            name_png = f'f{lead:03d}/{product}.png'
            image = tmp_path / 'images' / name_png
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b'fixture image identity; no weather pixels')
            panels.append(dict(map=product, file=name_png, rule='table',
                               max_abs='0.0000', scale_half_range='0.1', white_below='0.02'))
            image_rows.append(dict(file=name_png, sha256=sheets.sha(image), opened=True,
                observation='Synthetic metadata fixture, no human weather inspection claimed.'))
        leads.append(dict(lead=lead, valid=time.isoformat(), native_raw=name,
            hex_raw=str(tmp_path / 'hex' / name), native_sha256=files[name],
            hex_sha256=hex_hashes[name], failures=[], skips=[], panels=panels))
    publication = write('native/PUBLICATION.json', dict(files=files))
    deck['native'] = {'path': str(publication), 'sha256': sheets.sha(publication)}
    rendered['native'] = deck['native']
    original_deck.write_text(json.dumps(deck))
    render_deck.write_text(json.dumps(rendered))
    forecast = write('forecast.json', dict(evidence_kind='FORECAST', model_execution=True,
        deck=deck['id'], mesh=deck['mesh'], date=date, process_count=1, passed=False, acceptance_tier=None,
        provenance={'hex_output_sha256': hex_hashes, 'deck_sha256': sheets.sha(original_deck),
                    'inputs': deck['inputs'], 'auxiliary_inputs': {'original-mesh': {'sha256': sheets.sha(grid)}}}))
    original = dict(process_count=1, deck=str(original_deck), deck_sha256=sheets.sha(original_deck),
        original_receipt=str(forecast), original_receipt_sha256=sheets.sha(forecast),
        native_publication={'path': str(publication), 'sha256': sheets.sha(publication)},
        retained_sha256=hex_hashes, original_passed=False, original_acceptance_tier=None)
    sheet_file = write('images/sheets.json', dict(schema='mpas-sheets/v1',
        deck=str(render_deck), deck_sha256=sheets.sha(render_deck), native_publication=str(publication),
        renderer=str(renderer), renderer_sha256=sheets.sha(renderer),
        difference_table=str(table), leads=leads))
    attempt = write('attempt.json', dict(original=original, process_count=1, render_exit=0,
        science_and_rendering_unchanged=True, sheets_json_sha256=sheets.sha(sheet_file),
        staged_sha256={'render-deck.json': sheets.sha(render_deck),
            'inputs/mesh.nc': sheets.sha(grid), 'inputs/init.nc': sheets.sha(init),
            'tools/rw_wrfbatch': sheets.sha(renderer), 'tools/difference-sensitive.json': sheets.sha(table)}))
    inspection = write('inspection.json', dict(sheets_json_sha256=sheets.sha(sheet_file),
        attempt_sha256=sheets.sha(attempt), images=image_rows))
    return sheet_file, attempt, inspection, table


def update_manifest(paths, mutation):
    sheet_file, attempt, inspection, _ = paths
    value = sheets.read(sheet_file)
    mutation(value)
    sheet_file.write_text(json.dumps(value))
    run = sheets.read(attempt)
    run['sheets_json_sha256'] = sheets.sha(sheet_file)
    attempt.write_text(json.dumps(run))
    reviews = sheets.read(inspection)
    reviews['sheets_json_sha256'] = sheets.sha(sheet_file)
    reviews['attempt_sha256'] = sheets.sha(attempt)
    inspection.write_text(json.dumps(reviews))


def test_complete_images_do_not_approve_failed_forecast(complete_metadata):
    value = sheets.qualify(*complete_metadata)
    assert value['complete_required_products'] and value['rendered_images'] == 24
    assert value['original_forecast_passed'] is False
    assert value['forecast_acceptance'] is False and value['numerical_acceptance_changed'] is False


def test_successful_renderer_with_skipped_products_is_incomplete(complete_metadata):
    def partial(value):
        for row in value['leads']:
            row['panels'] = [p for p in row['panels'] if p['map'] == 'mslp']
            row['skips'] = [['2m_temperature', 'fixture missing selector']]
    update_manifest(complete_metadata, partial)
    inspection = complete_metadata[2]
    reviews = sheets.read(inspection)
    reviews['images'] = [r for r in reviews['images'] if r['file'].endswith('/mslp.png')]
    inspection.write_text(json.dumps(reviews))
    value = sheets.qualify(*complete_metadata)
    assert value['complete_required_products'] is False and value['rendered_images'] == 3


@pytest.mark.parametrize('fault', ['lead', 'duplicate_product', 'widened_scale', 'auto_scale',
                                  'different_valid_time', 'render_failure'])
def test_renderer_manifest_faults_refuse(complete_metadata, fault):
    def corrupt(value):
        row = value['leads'][0]
        if fault == 'lead': row['lead'] = 5
        elif fault == 'duplicate_product': row['panels'].append(row['panels'][0])
        elif fault == 'widened_scale': row['panels'][0]['scale_half_range'] = '100'
        elif fault == 'auto_scale': row['panels'][0]['rule'] = 'percentile'
        elif fault == 'different_valid_time': row['valid'] = value['leads'][1]['valid']
        else: row['failures'] = ['fixture renderer failure']
    update_manifest(complete_metadata, corrupt)
    with pytest.raises(AssertionError): sheets.qualify(*complete_metadata)


@pytest.mark.parametrize('fault', ['image_bytes', 'unopened_image', 'duplicate_review',
                                  'hex_bytes', 'native_bytes', 'render_deck_science'])
def test_foreign_or_unreviewed_evidence_refuses(complete_metadata, fault):
    sheet_file, _, inspection, _ = complete_metadata
    value = sheets.read(sheet_file)
    if fault == 'image_bytes':
        (sheet_file.parent / value['leads'][0]['panels'][0]['file']).write_bytes(b'changed image')
    elif fault in ('unopened_image', 'duplicate_review'):
        reviews = sheets.read(inspection)
        if fault == 'unopened_image': reviews['images'][0]['opened'] = False
        else: reviews['images'].append(reviews['images'][0])
        inspection.write_text(json.dumps(reviews))
    elif fault == 'hex_bytes': Path(value['leads'][0]['hex_raw']).write_bytes(b'foreign hex')
    elif fault == 'native_bytes':
        (Path(value['native_publication']).parent / value['leads'][0]['native_raw']).write_bytes(b'foreign native')
    else:
        Path(value['deck']).write_text('{}')
    with pytest.raises(AssertionError): sheets.qualify(*complete_metadata)


def test_rounded_scores_never_change_forecast_acceptance(complete_metadata):
    update_manifest(complete_metadata, lambda value: value['leads'][0]['panels'][0].update(max_abs='1.2345'))
    value = sheets.qualify(*complete_metadata)
    assert value['complete_required_products'] and value['scores_are_diagnostics_only']
    assert value['forecast_acceptance'] is False


def test_self_resealed_wider_table_cannot_replace_owner_scale(complete_metadata):
    _, attempt, inspection, table = complete_metadata
    value = sheets.read(table)
    value['bands'] = 1000
    table.write_text(json.dumps(value))
    run = sheets.read(attempt)
    run['staged_sha256']['tools/difference-sensitive.json'] = sheets.sha(table)
    attempt.write_text(json.dumps(run))
    reviews = sheets.read(inspection)
    reviews['attempt_sha256'] = sheets.sha(attempt)
    inspection.write_text(json.dumps(reviews))
    with pytest.raises(AssertionError): sheets.qualify(*complete_metadata)


@pytest.mark.parametrize('fault', ['deck_hash', 'deck_id', 'date', 'mesh', 'native_publication',
                                  'init_binding', 'relocated_init', 'relocated_grid'])
def test_self_resealed_pairing_or_transport_fault_refuses(complete_metadata, fault):
    sheet_file, attempt, inspection, _ = complete_metadata
    run = sheets.read(attempt)
    forecast_path = Path(run['original']['original_receipt'])
    forecast = sheets.read(forecast_path)
    if fault == 'deck_hash': forecast['provenance']['deck_sha256'] = 'foreign-deck'
    elif fault == 'deck_id': forecast['deck'] = 'foreign-context'
    elif fault == 'date': forecast['date'] = '2000-01-02_00:00:00'
    elif fault == 'mesh': forecast['mesh'] = 'foreign-mesh'
    elif fault == 'init_binding': forecast['provenance']['inputs']['init']['sha256'] = 'foreign-init'
    elif fault == 'native_publication':
        run['original']['native_publication']['sha256'] = 'foreign-publication'
    else:
        rendered = sheets.read(sheets.read(sheet_file)['deck'])
        if fault == 'relocated_init':
            path = Path(rendered['inputs']['init']['path'])
            key = 'inputs/init.nc'
        else:
            path = Path(rendered['hex_command'][-1])
            key = 'inputs/mesh.nc'
        path.write_bytes(b'foreign transport asset')
        run['staged_sha256'][key] = sheets.sha(path)
    forecast_path.write_text(json.dumps(forecast))
    run['original']['original_receipt_sha256'] = sheets.sha(forecast_path)
    attempt.write_text(json.dumps(run))
    reviews = sheets.read(inspection)
    reviews['attempt_sha256'] = sheets.sha(attempt)
    inspection.write_text(json.dumps(reviews))
    with pytest.raises(AssertionError): sheets.qualify(*complete_metadata)
