"""Capture both native MM5 driver branches using an existing strict executable.

Run on the assigned CPU cpuset. This only prepares private deck copies and
calls f0/f1 tools; it never builds native or decodes weather fields.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'decks', 'case', 'output', 'tables', 'forecast_tool',
                 'seal_tool', 'words', 'inventory', 'pins', 'receipt'):
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--mpi', required=True)
    p.add_argument('--duration', required=True, help='One source timestep, D_HH:MM:SS')
    p.add_argument('--pin-set-hash', required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    publication = a.case / 'PUBLICATION.json'
    rows = []
    for scheme in ('sf_monin_obukhov', 'sf_monin_obukhov_rev'):
        source = a.decks / ('sfclayer_scheme-' + scheme)
        deck = a.output / ('deck-' + scheme)
        shutil.copytree(source, deck)
        deck.chmod(0o755)
        namelist = deck / 'namelist.atmosphere'
        namelist.chmod(0o644)
        text, count = re.subn(r"(config_run_duration\s*=\s*)'[^']*'",
                             lambda m: m[1] + repr(a.duration), namelist.read_text())
        if count != 1:
            raise ValueError('deck must specify exactly one run duration')
        namelist.write_text(text)
        dumps = a.output / ('dumps-' + scheme)
        dumps.mkdir()
        env = dict(os.environ, MPAS_ORACLE_DIR=str(dumps.resolve()),
                   MPAS_ORACLE_STEPS='0,1', MPAS_ORACLE_KEYS=','.join((
                       'sfclay__driver_sfclayer__L892', 'sfclay__driver_sfclayer__L921',
                       'sfclayrev__driver_sfclayer__L953', 'sfclayrev__driver_sfclayer__L983')))
        run = a.output / ('run-' + scheme)
        with (a.output / (scheme + '.log')).open('w') as log:
            subprocess.run([sys.executable, str(a.forecast_tool), '--build', str(a.build),
                            '--deck', str(deck), '--case', str(a.case), '--out', str(run),
                            '--tables', str(a.tables), '--mpi', a.mpi, '--ranks', '1',
                            '--pin-set-hash', a.pin_set_hash], env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        recipe = json.dumps(dict(case=str(a.case), publication_sha256=sha(publication),
                                 binary_sha256=sha(a.build / 'atmosphere_model'),
                                 duration=a.duration, native_forecast_manifest_sha256=sha(run / 'MANIFEST.json')))
        subprocess.run([sys.executable, str(a.seal_tool), '--scheme', 'sfclay_mm5',
                        '--dump', str(dumps), '--words', str(a.words), '--seal',
                        '--inventory', str(a.inventory), '--pins', str(a.pins),
                        '--recipe', recipe], check=True)
        rows.append(dict(scheme=scheme, dumps=str(dumps),
                         manifest=json.loads((dumps / 'manifest.json').read_text()),
                         forecast_manifest=json.loads((run / 'MANIFEST.json').read_text())))
    a.receipt.write_text(json.dumps(dict(status='PASS', case=str(a.case),
                                       publication_sha256=sha(publication),
                                       forecast_tool_sha256=sha(a.forecast_tool),
                                       seal_tool_sha256=sha(a.seal_tool),
                                       captures=rows, hex_forecast_proven=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
