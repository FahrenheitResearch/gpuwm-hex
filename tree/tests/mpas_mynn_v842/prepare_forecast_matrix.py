"""Prepare complete native MYNN forecasts from f0's authenticated deck matrix.

No reference source or binary is rebuilt. This controller changes only owned
namelist copies and invokes the existing native runner and Rust-backed verifier.
The required meshes, dates and option selections come from the proof matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from hexcore.config import parse_fortran_namelist
from hexcore.mpasphys.bl_mynn_v842 import MynnOptions


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(args):
    matrix = json.loads(args.matrix.read_text())
    publication = json.loads((args.decks/'PUBLICATION.json').read_text())
    baseline = {'config_mynn_'+k: v for k, v in vars(MynnOptions()).items()}
    # The option matrix exercises the DD/output/budget branches together;
    # single and original convection decks retain their published defaults.
    baseline.update(config_mynn_edmf_dd=1, config_mynn_edmf_output=1,
                    config_mynn_tkebudget=1)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for requirement in matrix['required']:
        label = requirement['id'].split('/', 1)[0]
        date = requirement['date'].replace(':', '.')
        parent = args.decks/'G2'/requirement['mesh']/date
        source = parent/('pbl_scheme-bl_mynn' if label == 'single'
                         else 'suite-convection_permitting')
        # f0's published suite directory uses this name in current deck sets.
        if not source.exists() and label != 'single':
            source = parent/'convection_permitting'
        native_deck = args.output/requirement['id'].replace('/', '__').replace(':', '.')/'deck'
        native_deck.mkdir(parents=True)
        for name in ['namelist.atmosphere', 'streams.atmosphere']:
            path = source/name
            relative = str(path.relative_to(args.decks))
            if publication['files'].get(relative) != digest(path):
                raise ValueError('f0 published deck changed: '+relative)
            shutil.copyfile(path, native_deck/name)
        if digest(parent/'INDEX.json') != publication['files'][str((parent/'INDEX.json').relative_to(args.decks))]:
            raise ValueError('f0 published runtime index changed')
        shutil.copyfile(parent/'INDEX.json', native_deck.parent/'INDEX.json')
        options = {}
        if requirement.get('options'):
            options = baseline | requirement['options']
            text = (native_deck/'namelist.atmosphere').read_text()
            marker = re.search(r'^\s*&physics\b', text, re.M|re.I)
            if marker is None:
                raise ValueError('Published namelist lacks physics group')
            for key in options:
                text = re.sub(r'^\s*'+re.escape(key)+r'\s*=.*$', '', text, flags=re.M|re.I)
            rendered = []
            for key, value in options.items():
                word = ('.true.' if value else '.false.') if isinstance(value, bool) else str(value)
                rendered.append(' '+key+' = '+word)
            marker = re.search(r'^\s*&physics\b', text, re.M|re.I)
            text = text[:marker.end()]+'\n'+'\n'.join(rendered)+text[marker.end():]
            (native_deck/'namelist.atmosphere').write_text(text)
        config = parse_fortran_namelist((native_deck/'namelist.atmosphere').read_text())
        if (config['physics']['config_physics_suite'] != requirement['suite']
                or config['nhyd_model']['config_run_duration'] != '1_00:00:00'
                or any(config['physics'][k] != v for k, v in options.items())):
            raise ValueError('Prepared native deck differs from its required configuration')
        rows.append(dict(requirement=requirement, original_deck=str(source),
                         native_deck=str(native_deck.resolve()),
                         namelist_sha256=digest(native_deck/'namelist.atmosphere'),
                         streams_sha256=digest(native_deck/'streams.atmosphere'),
                         selected_options=options))
    result = dict(schema='mpas-mynn-native-deck-preparation/v1',
                  scope='Complete 24h native deck matrix; no execution or parity claim',
                  process_count=1, matrix_sha256=digest(args.matrix),
                  original_publication_sha256=digest(args.decks/'PUBLICATION.json'),
                  rows=rows)
    (args.output/'prepared.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Prepared complete native forecasts:', len(rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('matrix', 'decks', 'output'):
        parser.add_argument('--'+name, type=lambda value: Path(value).resolve(), required=True)
    prepare(parser.parse_args())
