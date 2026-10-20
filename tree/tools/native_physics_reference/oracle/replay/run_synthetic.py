#!/usr/bin/env python3
"""Capture and replay every prepared native fixture, retaining exact receipts.

Python only orchestrates native executables and the Rust word comparator.
Run on the host authorized for the intended proof scope. Build preparation and
compilation are separate operations; this command never starts a compiler.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison_totals(data):
    """Count authenticated replay arrays, including restored receipt shards."""
    arrays = [a for c in data.get('captures', []) for a in c['arrays'].values()]
    def count(key):
        values = [a.get(key) for a in arrays]
        return sum(values) if values and all(type(n) is int for n in values) else None
    return {'arrays': len(arrays), 'compared_words': count('words'),
            'differing_words': count('different_words')}


def run(inventory, standalone, pins, words, scratch, receipts, reference, input_files=(), campaign=None, suite=None, owner=None):
    inventory, standalone, pins, words = [p.resolve() for p in
                                         (inventory, standalone, pins, words)]
    scratch, receipts = scratch.resolve(), receipts.resolve()
    if scratch.exists() or receipts.exists():
        raise ValueError('fixture scratch and receipt directories must be fresh')
    if any(v is not None for v in (campaign,suite,owner)) and any(v is None for v in (campaign,suite,owner)):
        raise ValueError('publication requires campaign root, suite and owner')
    manifests = sorted(standalone.glob('*/synthetic-manifest.json'))
    if not manifests:
        raise ValueError('no prepared native fixtures')
    inputs = {}
    for path in input_files:
        path = path.resolve()
        if path.name in inputs or not path.is_file():
            raise ValueError('native inputs must be files with unique basenames')
        inputs[path.name] = {'path': str(path), 'sha256': digest(path)}
    scratch.mkdir(parents=True)
    receipts.mkdir(parents=True)
    cli = Path(__file__).resolve().parents[3] / 'column_oracle_replay.py'
    spec=importlib.util.spec_from_file_location('column_oracle_replay',cli)
    harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
    summary = {'schema': 'mpas-native-synthetic-matrix-v1', 'reference': reference,
               'host': socket.gethostname(), 'process_count': 1,
               'inventory_sha256': digest(inventory), 'words_sha256': digest(words),
               'runner_sha256': digest(Path(__file__)),
               'runtime_inputs': inputs, 'rows': [], 'forecast_proven': False}
    failures = 0
    for manifest in manifests:
        fixtures = json.loads(manifest.read_text())
        build_path = manifest.parent / 'build-manifest.json'
        build = json.loads(build_path.read_text())
        if fixtures['inventory_sha256'] != digest(inventory):
            raise ValueError('fixture inventory differs')
        if fixtures['build_manifest_sha256'] != digest(build_path):
            raise ValueError('fixture build provenance changed')
        for recipe in build['recipes']:
            for name, profile in fixtures['profiles'].items():
                site = fixtures['site']
                label = site + '-' + recipe + '-' + name
                directory = manifest.parent / recipe
                source = directory / ('capture_' + name + '.F90')
                if digest(source) != profile['generated_source_sha256']:
                    raise ValueError('native fixture source changed')
                exe, replay = directory / ('capture_' + name), directory / 'replay'
                dump, results = scratch / label, scratch / (label + '-replay')
                dump.mkdir()
                work = scratch / (label + '-work')
                work.mkdir()
                for basename, entry in inputs.items():
                    source_input = Path(entry['path'])
                    if digest(source_input) != entry['sha256']:
                        raise ValueError('native runtime input changed')
                    shutil.copy2(source_input, work / basename)
                env = {k: v for k, v in os.environ.items()
                       if not k.startswith('MPAS_ORACLE_')}
                env.update(MPAS_ORACLE_DIR=str(dump), MPAS_ORACLE_KEYS=site)
                row = {'site': site, 'recipe': recipe, 'profile': name,
                       'capture_executable_sha256': digest(exe),
                       'replay_executable_sha256': digest(replay),
                       'synthetic_manifest_sha256': digest(manifest)}
                capture = subprocess.run([str(exe)], env=env, cwd=work,
                                         capture_output=True, text=True)
                (receipts / (label + '.native.log')).write_text(capture.stdout + capture.stderr)
                row['native_exit'] = capture.returncode
                common = ['--ref', reference, '--scheme', site, '--dump', str(dump),
                          '--words', str(words), '--recipe', recipe, '--process-count', '1']
                if capture.returncode == 0:
                    seal = subprocess.run([sys.executable, str(cli), *common, '--seal',
                        '--inventory', str(inventory), '--pins', str(pins)],
                        capture_output=True, text=True)
                    (receipts / (label + '.seal.log')).write_text(seal.stdout + seal.stderr)
                    row['seal_exit'] = seal.returncode
                    if seal.returncode == 0:
                        receipt = receipts / (label + '.json')
                        compare = subprocess.run([sys.executable, str(cli), *common,
                            '--executable', str(replay), '--results', str(results),
                            '--receipt', str(receipt)], env=env, cwd=work,
                            capture_output=True, text=True)
                        (receipts / (label + '.replay.log')).write_text(compare.stdout + compare.stderr)
                        row['replay_exit'] = compare.returncode
                        if receipt.exists():
                            data = harness.read_replay_receipt(receipt)
                            row.update(comparison_complete=data['comparison_complete'],
                                all_outputs_bitwise_equal=data['all_outputs_bitwise_equal'],
                                receipt_sha256=digest(receipt), **comparison_totals(data))
                passed = (row.get('replay_exit') == 0 and
                          row.get('comparison_complete') is True and
                          row.get('all_outputs_bitwise_equal') is True)
                if not passed:
                    failures += 1
                # Every attempted fixture has a receipt before owned run files
                # are retired; failures retain logs and exact word witnesses.
                if campaign is not None and row.get('seal_exit')==0:
                    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
                    from publish_dump import publish
                    published_receipt=receipts/(label+'.publication.json')
                    dump_manifest_sha256=digest(dump/'manifest.json')
                    try:
                        publication=publish(dump,campaign,suite,owner)
                    except (ValueError,OSError) as error:
                        # A full matrix still replays its remaining fixtures if
                        # publication is refused by the shared cap or filesystem.
                        # The failed publication prevents a successful matrix.
                        publication={'schema':'mpas-oracle-local-publication-failure-v1',
                                     'owner':owner,'suite':suite,'error':str(error),
                                     'dump_manifest_sha256':dump_manifest_sha256,
                                     'published':False}
                        failures+=1
                        row['publication_failed']=True
                    published_receipt.write_text(json.dumps(publication,indent=2)+'\n')
                    row['publication']=publication
                summary['rows'].append(row)
                summary['all_outputs_bitwise_equal'] = failures == 0
                (receipts / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
                if dump.exists():shutil.rmtree(dump)
                shutil.rmtree(work)
                if results.exists():
                    shutil.rmtree(results)
    return 1 if failures else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('inventory', 'standalone', 'pins', 'words', 'scratch', 'receipts'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--ref', choices=['mpas', 'rrfs'], required=True)
    p.add_argument('--input-file', type=Path, action='append', default=[],
                   help='native runtime input copied by basename into each fresh working directory')
    p.add_argument('--campaign-root',type=Path)
    p.add_argument('--suite',help='explicit publication table row; no suite is assumed')
    p.add_argument('--owner',help='lane owning the immutable local publication')
    a = p.parse_args()
    return run(a.inventory, a.standalone, a.pins, a.words,
               a.scratch, a.receipts, a.ref, a.input_file,a.campaign_root,a.suite,a.owner)


if __name__ == '__main__':
    raise SystemExit(main())
