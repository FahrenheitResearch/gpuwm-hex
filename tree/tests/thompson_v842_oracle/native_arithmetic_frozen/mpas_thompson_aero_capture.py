"""Capture a native aerosol call using f0's published runner and f1's tap.

This is test orchestration only: no field decoding, reference rebuild, backend
shim or forecast acceptance. The caller supplies an unchanged native artifact
and explicit deck and timestep selection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('build','deck','case','out','tables','dump','runner','receipt',
                'build_provenance','publication'):
        p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    p.add_argument('--mpi',required=True)
    p.add_argument('--pin-set-hash',required=True)
    p.add_argument('--steps',default='1',help='native tap timestep selection')
    a=p.parse_args()
    native=sha(a.build/'atmosphere_model')
    provenance=json.loads(a.build_provenance.read_text())
    if native!=provenance['native_binary_sha256']:
        raise ValueError('tap binary differs from the recorded native artifact')
    publication=json.loads(a.publication.read_text())
    # Publication hashes name untouched inputs, including the initialized case.
    for name,expected in publication['files'].items():
        path=a.case/name
        if name.endswith('.nc') and sha(path)!=expected:
            raise ValueError(f'published native input changed: {name}')
    a.dump.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    env.update(MPAS_ORACLE_DIR=str(a.dump.resolve()),
        MPAS_ORACLE_KEYS='mp_gt_driver__driver_microphysics__L415',
        MPAS_ORACLE_STEPS=a.steps)
    command=[sys.executable,str(a.runner),'--build',str(a.build),
        '--deck',str(a.deck),'--case',str(a.case),'--out',str(a.out),
        '--tables',str(a.tables),'--mpi',str(Path(a.mpi).resolve()),'--ranks','1',
        '--pin-set-hash',a.pin_set_hash]
    result=subprocess.run(command,env=env)
    bins=list(a.dump.glob('*.bin'))
    receipt=dict(acceptance=False,forecast_proven=False,host=os.uname().nodename,
        returncode=result.returncode,native_binary_sha256=native,
        publication_sha256=sha(a.publication),build_provenance_sha256=sha(a.build_provenance),
        runner_sha256=sha(a.runner),namelist_sha256=sha(a.deck/'namelist.atmosphere'),
        streams_sha256=sha(a.deck/'streams.atmosphere'),
        oracle_selection={k:env[k] for k in ('MPAS_ORACLE_KEYS','MPAS_ORACLE_STEPS')},
        dump_files=len(bins),dump_bytes=sum(f.stat().st_size for f in bins))
    a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    if result.returncode:raise SystemExit(result.returncode)
    if not bins:raise ValueError('native forecast produced no aerosol tap records')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
