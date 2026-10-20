"""Audit forecast-bound sheets and recorded inspection under the owner's visual ruling.

Artifact validation cannot look at weather. The lane opens every PNG and records
its observations; the integrator opens the same files before landing. This audit
prevents unrelated forecasts, omitted panels, or unreviewed images from being
presented as that inspection. Weather decoding and rendering remain Rust.
"""
from __future__ import annotations

import datetime
import math
from pathlib import Path

from hexcore.verify_mpas import file_ref
from rrfs_proof_contract import digest, read, require

MAPS = frozenset(('2m-temperature', 'precip-total-since-start', 'mslp',
                 'mslp-isobars-10m-wind', '500hPa-height',
                 '500hPa-height-contours-wind', '10m-wind', 'composite-reflectivity'))


def audit_sheets(folder, receipts, commit):
    """Require at least one genuine proof deck and inspect every produced panel."""
    folder = Path(folder).resolve()
    bundle = read(folder/'sheet-inspections.json')
    require(bundle['schema'] == 'rrfs-sheet-inspections-v1'
            and bundle['source_commit'] == commit, 'unknown or stale sheet inspection')
    require(bundle['decks'], 'no inspected whole-forecast sheets')
    seen = set()
    for row in bundle['decks']:
        identity = row['deck']
        require(identity in receipts and identity not in seen,
                'sheets belong to an unknown or duplicate proof deck')
        seen.add(identity)
        deck, baseline, receipt_path = receipts[identity]
        require(row['forecast_sha256'] == digest(receipt_path), 'sheet forecast receipt changed')
        sheet_path = file_ref(row['sheets'], folder).resolve()
        require(sheet_path.is_relative_to(folder), 'sheets outside fresh proof directory')
        sheets = read(sheet_path)
        require(sheets['schema'] == 'mpas-sheets/v1'
                and sheets['deck_sha256'] == baseline['provenance']['deck_sha256'],
                'sheets do not use the forecast deck')
        publication_path = Path(sheets['native_publication'])
        require(digest(publication_path) == baseline['provenance']['native_manifest_sha256'],
                'sheets use another native publication')
        publication = read(publication_path)
        require(digest(sheets['renderer']) == sheets['renderer_sha256'], 'sheet renderer changed')
        require(digest(sheets['difference_table']) == row['difference_table_sha256'],
                'fixed difference scale changed after inspection')
        require(row['evolution'].strip(), 'forecast evolution was not inspected')
        inspected = {}
        for image in row['images']:
            path = file_ref({key: image[key] for key in ('path', 'sha256')},
                            sheet_path.parent).resolve()
            require(path.is_relative_to(sheet_path.parent) and path.suffix == '.png'
                    and path not in inspected, 'unknown or duplicate inspected image')
            require(image['observation'].strip(), 'image has no visual observation')
            inspected[path] = image
        generated, leads = set(), set()
        for frame in sheets['leads']:
            lead = frame['lead']
            require(type(lead) is int and lead >= 0 and lead not in leads,
                    'invalid or duplicate sheet lead')
            leads.add(lead)
            require(not frame['failures'] and not frame['skips'], 'sheet rendering incomplete')
            valid = datetime.datetime.strptime(deck['date'], '%Y-%m-%d_%H:%M:%S') + datetime.timedelta(hours=lead)
            name = 'output.' + valid.strftime('%Y-%m-%d_%H.%M.%S') + '.nc'
            require(Path(frame['hex_raw']).name == name and frame['valid'] == valid.isoformat(),
                    'sheet lead does not match its actual output time')
            require(frame['hex_sha256'] == baseline['provenance']['hex_output_sha256'].get(name),
                    'sheets do not use this forecast output')
            require(frame['native_raw'] == name and frame['native_sha256'] == publication['files'][name],
                    'sheet native output differs from the authenticated publication')
            panels = frame['panels']
            require(len(panels) == len(MAPS) and {p['map'] for p in panels} == MAPS,
                    'one or more required weather panels are absent')
            for panel in panels:
                path = (sheet_path.parent/panel['file']).resolve()
                require(path.is_relative_to(sheet_path.parent) and path in inspected
                        and path not in generated, 'weather panel was not opened and inspected')
                generated.add(path)
                # These are renderer metadata, never a replacement for looking.
                for key in ('max_abs', 'scale_half_range', 'white_below'):
                    raw = panel[key]
                    # mpas-sheets preserves Rust's decimal metadata as strings.
                    require(type(raw) in (int, float, str),
                            'weather panel has invalid difference-scale metadata')
                    try:
                        value = float(raw)
                    except ValueError:
                        raise ValueError('weather panel has invalid difference-scale metadata') from None
                    require(math.isfinite(value)
                            and (value >= 0 if key == 'max_abs' else value > 0),
                            'weather panel has invalid difference-scale metadata')
        require({6, 12, 24} <= leads, 'required forecast evolution leads absent')
        require(generated == set(inspected)
                == {p.resolve() for p in sheet_path.parent.rglob('*.png')},
                'produced or cited images were omitted from the visual inspection')
