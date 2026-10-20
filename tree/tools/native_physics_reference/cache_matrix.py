#!/usr/bin/env python3
"""Initialize and publish explicit native case rows as their GRIB files arrive.

Run under the native CPU affinity. An optional existing first row joins a job
already launched with cache_cases.py, preserving its in-progress output.
"""
import argparse
import json
from pathlib import Path
import time
from cache_cases import prepare, validate, contract
from publish_cache import publish
from verify_acceptance import validate as validate_acceptance
from matrix_support import published, free_memory_bytes, discard_copied_outputs

def dependencies(row):
    paths=[row['grid'],row['graph']]
    if row.get('shared_ungrib'):
        paths.append(str(Path(row['shared_ungrib'])/'COMPLETE.json'))
    else:paths.extend(row['gribs'])
    if row.get('stages')==[8]:paths.append(row['initial'])
    if row.get('ranks',1)>1:paths.append(row['graph']+f'.part.{row["ranks"]}')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=Path)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--acceptance-index', type=Path, required=True)
    parser.add_argument('--ready', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--join-first', action='store_true')
    parser.add_argument('--join-existing', action='store_true')
    for name in ('init','ungrib','vtable','geog','mpi'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    checks = json.loads(args.acceptance_index.read_text())['checks']
    if not checks:
        raise ValueError('no acceptance receipts')
    for check in checks:
        errors = validate_acceptance(json.loads(Path(check['receipt']).read_text()),
            check['hours'], check['cells'], check['levels'], check['init_sha256'])
        if errors:
            raise ValueError(errors)
    rows = json.loads(args.index.read_text())['cases']
    if not rows:
        raise ValueError('empty native case matrix')
    pending = list(enumerate(rows))
    results = []
    while pending:
        progress = False
        for index, row in list(pending):
            out = args.work.resolve()/row['mesh']/row['start'].replace(':','.')
            destination=Path(row['publish'])
            publication = published(destination, out)
            if publication:
                if contract(json.loads((destination/'MANIFEST.json').read_text())['case'])!=contract(row):
                    raise ValueError('published native case contract differs')
                results.append({'case':row,'publication':publication})
                pending.remove((index,row));progress=True
                args.receipt.write_text(json.dumps({'complete': not pending,
                    'cases': results}, indent=2)+'\n')
                continue
            if any(not Path(path).is_file() for path in dependencies(row)):
                continue
            validate(row,args.init,args.ungrib,args.vtable,args.geog)
            if (args.join_first and index == 0) or (args.join_existing and out.exists()):
                if not (out/'MANIFEST.json').is_file():
                    continue
                if contract(json.loads((out/'MANIFEST.json').read_text())['case'])!=contract(row):
                    raise ValueError('completed native case contract differs')
            else:
                if free_memory_bytes() < 16 * 1024**3:
                    continue
                print(json.dumps({'start': row}), flush=True)
                out = prepare(row,args.work.resolve(),args.init,args.ungrib,args.vtable,args.geog,args.mpi)
            # Forecasts need the native graph partitions, not scratch dependencies.
            for partition in Path(row['graph']).parent.glob(Path(row['graph']).name+'.part.*'):
                target = out/partition.name
                if not target.exists():
                    target.symlink_to(partition.resolve(strict=True))
            publication = publish(out,Path(row['publish']),
                ['*.nc','*.graph.info*','MANIFEST.json','case-*','namelist.init_atmosphere',
                 'streams.init_atmosphere','ungrib.log'],args.ready,row['description'])
            result = {'case': row, 'publication': publication}
            results.append(result)
            args.receipt.write_text(json.dumps({'complete': len(results)==len(rows),
                'cases': results},indent=2)+'\n')
            print(json.dumps({'completed': result}),flush=True)
            manifest = json.loads((out/'MANIFEST.json').read_text())
            discard_copied_outputs(out, destination, manifest['files'])
            # Ungrib intermediates are disposable, with every byte already hashed.
            for name in manifest['intermediates']:
                intermediate = out/name
                if intermediate.is_file() and not intermediate.is_symlink():
                    intermediate.unlink()
            pending.remove((index,row))
            progress = True
            break
        if not progress:
            time.sleep(20)


if __name__ == '__main__':
    main()
