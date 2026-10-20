#!/usr/bin/env python3
"""Run explicit native forecast rows as their published inputs become available.

Index: {"jobs": [{"build": path, "deck": path, "case": path, "out": path,
"tables": path, "mpi": path, "ranks": int, "pin_set_hash": str,
"publish": path, "description": str}]}.
Completed forecasts are published immediately, with every output stream hashed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET
from publish_cache import publish, sha
from matrix_support import published, free_memory_bytes, discard_copied_outputs


def validate_manifest(manifest, row):
    expected={'pin_set_hash':row['pin_set_hash'],'ranks':row['ranks'],
              'deck':Path(row['deck']).name,
              'binary_sha256':sha(Path(row['build'])/'atmosphere_model'),
              'namelist_sha256':sha(Path(row['deck'])/'namelist.atmosphere'),
              'streams_sha256':sha(Path(row['deck'])/'streams.atmosphere')}
    if bool(manifest.get('outputs_retired'))!=bool(row.get('digest_only')):
        raise ValueError('native output retention differs from requested reference contract')
    if row.get('digest_only') and manifest.get('output_digest_tool_sha256')!=sha(Path(row['comparator'])):
        raise ValueError('native field digest reader differs')
    if any(manifest.get(key)!=value for key,value in expected.items()):
        raise ValueError('completed forecast contract differs from requested job')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=Path)
    parser.add_argument('--ready', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--join-existing', action='store_true',
                        help='join inspected live native forecasts without restarting them')
    args = parser.parse_args()
    jobs = json.loads(args.index.read_text())['jobs']
    if not jobs:
        raise ValueError('empty native forecast matrix')
    results = []
    pending = list(jobs)
    while pending:
        progress = False
        for row in list(pending):
            out = Path(row['out'])
            destination = Path(row['publish'])
            publication = published(destination, out)
            if publication:
                validate_manifest(json.loads((destination/'MANIFEST.json').read_text()),row)
                results.append({'job': row, 'publication': publication})
                pending.remove(row)
                progress = True
                args.receipt.write_text(json.dumps({'complete': not pending,
                    'jobs': results}, indent=2)+'\n')
                continue
            case = Path(row['case'])
            deck = Path(row['deck'])
            if not (case/'PUBLICATION.json').is_file():
                continue
            input_stream = next(s for s in ET.parse(deck/'streams.atmosphere').getroot()
                                if s.get('name') == 'input')
            if not (case/input_stream.get('filename_template')).is_file():
                raise ValueError('published case lacks deck input')
            argv = ['python', str(Path(__file__).with_name('run_forecast.py'))]
            for key in ('build', 'deck', 'case', 'out', 'tables', 'mpi', 'ranks', 'pin_set_hash'):
                argv.extend(['--'+key.replace('_', '-'), str(row[key])])
            if row.get('digest_only'):
                argv.extend(['--digest-only','--comparator',row['comparator']])
                for package in row.get('inactive_packages',[]):argv.extend(['--inactive-package',package])
            if not (out/'MANIFEST.json').is_file():
                if out.exists():
                    if args.join_existing:
                        continue
                    raise ValueError(f'unfinished forecast needs inspection: {out}')
                if free_memory_bytes() < 16 * 1024**3:
                    continue
                print(json.dumps({'start': row}), flush=True)
                subprocess.run(argv, check=True)
            manifest = json.loads((out/'MANIFEST.json').read_text())
            validate_manifest(manifest,row)
            publication = publish(out, Path(row['publish']),
                [*([] if manifest.get('outputs_retired') else manifest['outputs']), 'CLOCKS.json', 'MANIFEST.json', '*.log', 'log.atmosphere.*.out',
                 'namelist.atmosphere', 'streams.atmosphere'], args.ready, row['description'])
            result = {'job': row, 'publication': publication}
            results.append(result)
            args.receipt.write_text(json.dumps({'complete': len(results)==len(jobs),
                'jobs': results}, indent=2)+'\n')
            print(json.dumps({'completed': result}), flush=True)
            discard_copied_outputs(out, destination, manifest['outputs'])
            pending.remove(row)
            progress = True
            break
        if not progress:
            time.sleep(20)


if __name__ == '__main__':
    main()
