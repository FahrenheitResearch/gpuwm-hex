#!/usr/bin/env python3
"""Run the supplied checkout's actual forecast door at fixed process count.

The index pins verification decks and lists incomplete unit requirements. A
partial delivery can reproduce a forecast attempt but cannot return success.
Numeric NetCDF comparisons stay in the pinned Rust verify-mpas artifact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from hexcore import site_runner


def run(checkout, index_path, work_root):
    index = json.loads(index_path.read_text())
    work = work_root / uuid.uuid4().hex
    work.mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(checkout/'tree/src')+':'+str(checkout/'tree/tools')+
               ':'+(os.environ.get('GPUWM_ENGINE_SRC') or os.environ.get('MPAS_ENGINE') or '.'))
    summary = {'schema':'mpas-f1-whole-forecast-reproduction-v1', 'process_count':1,
               'checkout':str(checkout), 'hex_commit':subprocess.check_output(
                   ['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip(),
               'cases':[], 'passed':False,
               'unimplemented_requirements':index['unimplemented_requirements']}
    for entry in index['cases']:
        deck = (index_path.parent/entry['deck']).resolve()
        if hashlib.sha256(deck.read_bytes()).hexdigest() != entry['deck_sha256']:
            raise ValueError('reproduction deck hash changed')
        data = json.loads(deck.read_text())
        if data.get('ranks') != 1:
            raise ValueError('reproduction fixes native/hex process count to one')
        receipt = work/(entry['id']+'.json')
        command = site_runner.gpu_command(['taskset','-c','8-23',
                   sys.executable,'-m','hexcore.cli','verify-mpas', '--deck',str(deck),
                   '--receipt',str(receipt),'--work-root',str(work/entry['id']),
                   '--comparator',index['comparator']], 10)
        with (work/(entry['id']+'.log')).open('w') as stream:
            result = subprocess.run(command,cwd=checkout/'tree',env=env,
                                    stdout=stream,stderr=subprocess.STDOUT)
        proof = json.loads(receipt.read_text()) if receipt.is_file() else {}
        after_commit = subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
        checkout_unchanged = after_commit == summary['hex_commit']
        accepted = (result.returncode == 0 and proof.get('passed') is True and
                    proof.get('tier') == 'BITWISE' and proof.get('model_execution') is True and
                    proof.get('evidence_kind') == 'FORECAST' and checkout_unchanged)
        summary['cases'].append({'id':entry['id'],'process_count':1,'accepted':accepted,
                                 'returncode':result.returncode,'receipt':str(receipt),
                                 'checkout_commit_after_execution':after_commit,
                                 'checkout_unchanged':checkout_unchanged,
                                 'deck_sha256':entry['deck_sha256']})
        if not accepted:
            break
    summary['passed'] = bool(summary['cases']) and not summary['unimplemented_requirements'] and (
        len(summary['cases']) == len(index['cases']) and all(x['accepted'] for x in summary['cases']))
    path = work/'summary.json'
    path.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'passed':summary['passed'],'summary':str(path)}),flush=True)
    return 0 if summary['passed'] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['checkout','index','work-root']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    return run(a.checkout.resolve(),a.index.resolve(),a.work_root.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
