"""Fail-closed full forecast reproducer; standalone receipts cannot pass it.

The integrator/r0 supplies immutable verification decks once native forecasts
are published. This does not synthesize native manifests or relax comparisons.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import subprocess


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


SHEET_SCALES = {'2m-temperature': 5., 'precip-total-since-start': .25, 'mslp': .1,
                'mslp-isobars-10m-wind': 5., '500hPa-height': .01,
                '500hPa-height-contours-wind': 5., '10m-wind': 5.,
                'composite-reflectivity': 10.}
# The integrator's 21:40Z fixed fine-scale table; units are the renderer's.
SHEET_TABLE_SHA256 = 'a180ca0b0939f44f166182e585d30cdd87ba4aea4ca7b68768e0ff1a7e4cbdbe'


def validate_difference_sheets(receipt, deck_path, native_publication):
    """Bind complete Rust sheets to this forecast's actual compared output.

    This authenticates evidence; opening every image remains a human review
    requirement. Numerical equality never substitutes for that review.
    """
    from hexcore.verify_mpas import digest
    binding = receipt.get('difference_sheets')
    require(binding and not binding.get('failures'),
            'forecast lacks complete Rust difference sheets')
    manifest = Path(binding['path']).resolve()
    require(digest(manifest) == binding['sha256'], 'difference sheet manifest changed')
    sheets = json.loads(manifest.read_text())
    require(sheets.get('schema') == 'mpas-sheets/v1', 'unsupported sheet schema')
    require(Path(sheets['deck']).resolve() == deck_path.resolve() and
            sheets['deck_sha256'] == digest(deck_path), 'sheets belong to another deck')
    require(Path(sheets['native_publication']).resolve() == native_publication.resolve(),
            'sheets use another native publication')
    require(digest(Path(sheets['difference_table'])) == SHEET_TABLE_SHA256,
            'sheets lack the integrator fixed fine-scale table')
    require(digest(Path(sheets['renderer'])) == sheets['renderer_sha256'],
            'Rust renderer artifact changed')
    publication = json.loads(native_publication.read_text())
    start = datetime.datetime.strptime(json.loads(deck_path.read_text())['date'], '%Y-%m-%d_%H:%M:%S')
    compared = receipt['provenance']['hex_output_sha256']
    rows = sheets.get('leads', [])
    require([row['lead'] for row in rows] == [6, 12, 24],
            'sheets must cover all three required forecast leads')
    image_refs = []
    for row in rows:
        require(not row['failures'] and not row['skips'], 'sheet products failed or skipped')
        expected = 'output.' + (start + datetime.timedelta(hours=row['lead'])).strftime('%Y-%m-%d_%H.%M.%S') + '.nc'
        require(Path(row['hex_raw']).name == expected and row['native_raw'] == expected,
                'sheet panel uses output from another forecast lead')
        require(compared.get(Path(row['hex_raw']).name) == row['hex_sha256'],
                'sheets did not render the compared hex output words')
        require(publication['files'].get(row['native_raw']) == row['native_sha256'],
                'sheets did not render the authenticated native output words')
        panels = row['panels']
        require(len(panels) == 8 and {p['map'] for p in panels} == set(SHEET_SCALES),
                'each lead must contain every required map exactly once')
        for panel in panels:
            image = (manifest.parent / panel['file']).resolve()
            require(image.is_relative_to(manifest.parent) and image.is_file(),
                    'sheet image missing or escapes its evidence directory')
            require(image.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n', 'sheet is not a PNG')
            for key in ('max_abs', 'scale_half_range', 'white_below'):
                value = panel.get(key)
                require(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and value >= 0,
                        'sheet lacks a finite difference measurement or scale')
            scale = SHEET_SCALES[panel['map']]
            require(panel['scale_half_range'] == scale and panel['white_below'] == scale/5,
                    'sheet uses a scale that can hide forecast differences')
            image_refs.append({'lead': row['lead'], 'map': panel['map'],
                               'path': str(image), 'sha256': digest(image)})
    return {'manifest': str(manifest), 'sha256': digest(manifest),
            'images': image_refs, 'visual_review_required': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--mpas-index', type=Path, required=True)
    parser.add_argument('--ratchet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from hexcore.verify_mpas import file_ref, gate_receipt_failures, read_json
    from hexcore.config import parse_fortran_namelist
    from hexcore.mpasphys.rrfs.driver_rrfs import resolve_namelist, specification
    checkout = Path(__file__).resolve().parents[3]
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=checkout, check=True)
    baseline = read_json(args.ratchet)
    spec = specification()
    require(spec['model_commit'] == '1d6e370e7b43a332074809f170532a216c2212fb',
            'reproducer requires the pinned native RRFS model authority')
    require(spec['dycore_path'] == 'v8.3.1-noaa',
            'RRFS comparison must select the fork dycore')
    require(spec['pin_set'] == 'rrfs', 'driver must select the native RRFS provider')
    require(not any(spec['matched_extensions'].values()),
            'matched forecast requires disabled hex extensions')
    index = json.loads(args.index.read_text())
    require(index['schema'] == 'mpas-gate-index-v1', 'unsupported forecast index')
    decks = []
    observed = set()
    for ref in index['gates']['G2']:
        path = file_ref(ref, args.index.parent)
        deck = json.loads(path.read_text())
        require(deck['pin_set'] == 'fd8df6511cbaf369fe818eaeb5a91acde341303dac86efd2d4f6aa6fea8cc346',
                'forecast deck uses another native pin-set hash')
        native = file_ref(deck['native'], path.parent)
        publication = json.loads(native.read_text())
        require(publication['schema'] == 'mpas.native-cache-publication/v1',
                'native reference must be an immutable r0 forecast publication')
        manifest = file_ref({'path':'MANIFEST.json', 'sha256':publication['files']['MANIFEST.json']},
                            native.parent)
        reference = json.loads(manifest.read_text())
        require(reference['binary_sha256'] == '0b822cc7e2d76455a056f8914609cdc740b10ec5722fe0b7a22a3955f8087739',
                'forecast reference must use the published primary strict fork binary')
        sheets = deck.get('difference_sheets')
        require(sheets and sheets.get('leads', '6,12,24') == '6,12,24',
                'forecast deck must render Rust difference sheets before output retirement')
        require(file_ref(sheets['native_publication'], path.parent) == native,
                'sheet deck must bind the same authenticated native forecast publication')
        namelist = file_ref(deck['inputs']['namelist'], path.parent)
        groups = parse_fortran_namelist(namelist.read_text())
        require(groups['nhyd_model']['config_run_duration'] == '1_00:00:00',
                'reproducer cannot accept a truncated native forecast deck')
        config = resolve_namelist(namelist.read_text(), backends={'convection':'hex-cuda'})
        require(all(str(value) == ('hex-cuda' if slot == 'convection' else 'native-cpu')
                    for slot, value in config.backends.items()),
                'each other physics slot must retain the fork native CPU backend')
        command = deck['hex_command']
        require(command[0] == 'gpuwm-hex' and command[1] == 'forecast',
                'forecast proof must execute the real gpuwm-hex door')
        require(command.count('--physics') == 1, 'only the convection backend may be overlaid')
        require(command[command.index('--physics')+1] == "config_convection_backend='hex-cuda'",
                'forecast overlay changes native options beyond this lane backend')
        require(command[command.index('--physics-backend')+1] == 'mpasphys',
                'forecast command must select the MPAS backend row')
        scheme = config.options['config_convection_scheme']
        suite = config.options['config_physics_suite']
        if suite == 'hrrrv5':
            label = 'hrrrv5'
        else:
            require(suite == 'none' and scheme == 'cu_grell_freitas_li',
                    'single-scheme forecast deck must select native GFL')
            label = 'sub3d-' + str(config.options['config_gfl_sub3d'])
            require(label in ('sub3d-0', 'sub3d-1'), 'unexpected sub3d proof option')
        observed.add((deck['mesh'], deck['date'], deck['process_count'], label))
        decks.append((path, deck))
    required = {(mesh, date, rank, label)
                for mesh in ('x1.2562', 'x1.10242', 'x1.10242.CONUS')
                for date in ('2023-03-31_12:00:00', '2022-12-22_12:00:00')
                for rank in (1, 2, 4) for label in ('sub3d-0', 'sub3d-1', 'hrrrv5')}
    require(required <= observed, 'forecast index lacks required mesh/date/sub3d/suite/rank proofs')
    tree = Path(__file__).resolve().parents[2]
    # The gate queues each entire forecast itself; do not nest a GPU allocation.
    for label, selection, path in (('rrfs', 'G2', args.index),
                                   ('mpas', 'G0,G1,G2', args.mpas_index)):
        subprocess.run(['bash', str(tree/'tools/mpas_gate.sh'), '--gates', selection,
                        '--index', str(path.resolve()), '--ratchet', str(args.ratchet.resolve()),
                        '--minutes', '120', '--out', str(args.output/label)], check=True)
    gate = json.loads((args.output/'rrfs/gate.json').read_text())
    require(gate['passed'] is True, 'RRFS whole forecast gate failed')
    results = gate['stages']['G2']['decks']
    require(len(results) == len(decks), 'gate omitted forecast decks')
    visual_evidence = []
    for result, (path, deck) in zip(results, decks):
        receipt = json.loads(Path(result['receipt']).read_text())
        require(not gate_receipt_failures(receipt, deck, 'G2', baseline), 'forecast or stream schedule incomplete')
        require(receipt.get('acceptance_tier') in ('BITWISE', 'BITWISE-WRITTEN-FIELDS'),
                'this reproducer accepts only the integrator bitwise forecast tiers')
        fields = receipt['comparison']['fields']
        written = [field for field in fields if field['status'] == 'compared']
        require(written and all(field['differing_words'] == 0 for field in written),
                'every written field must match; the gate authenticates unchanged missing inventory')
        require(not receipt.get('ratchet_regressions'), 'integrator ratchet regressed')
        visual_evidence.append({'deck': deck['id'], 'evidence': validate_difference_sheets(
            receipt, path, file_ref(deck['native'], path.parent))})
    require(json.loads((args.output/'mpas/gate.json').read_text())['passed'] is True,
            'MPAS regression gates failed')
    (args.output/'difference-sheets-index.json').write_text(
        json.dumps({'decks': visual_evidence, 'visual_review_required': True}, indent=2)+'\n')
    print('Every required GFL forecast and MPAS regression gate passed bitwise; '
          'all difference sheets await visual review')


if __name__ == '__main__':
    main()
