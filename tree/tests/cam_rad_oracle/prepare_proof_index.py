"""Bind the complete committed CAM matrix to retained strict native forecasts.

No execution or equivalence is implied by this prepared index. The model door,
strict-provider verifier and whole-stream comparator establish that separately.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reference(path):
    return dict(path=str(path.resolve()),sha256=digest(path))


def prepare(args):
    from hexcore.mpasphys.driver_v842 import resolve_namelist
    from hexcore.physics_suites import load_table
    plan=json.loads(args.plan.read_text())
    foundation=json.loads(args.foundation_index.read_text())
    provider=foundation['native_provider']
    library=provider['artifacts']['library']
    for item in provider['artifacts'].values():
        if digest(item['path'])!=item['sha256']:
            raise ValueError('Foundation native provider dependency differs')
    prepared=[]
    entries=[]
    table=load_table()['mpasphys_v842']['slots']
    pool_reader=foundation['dependencies']['pool_reader']
    comparator=foundation['dependencies']['official_comparator']
    selected=list(enumerate(plan['required_cases']))
    if args.only_case is not None:
        if not 0 <= args.only_case < len(selected):
            raise ValueError('Selected whole forecast is outside the required matrix')
        selected=[selected[args.only_case]]
    for number,required in selected:
        root=args.native_root/('case-'+str(number))
        summary=json.loads((root/'summary.json').read_text())
        if (not summary['complete'] or summary['process_count']!=1
                or summary['required_plan_sha256']!=digest(args.plan)
                or len(summary['cases'])!=1):
            raise ValueError('Native whole-deck capture is incomplete or mismatched')
        captured=summary['cases'][0]
        if captured['id']!=required['id'] or captured['host']!=args.proof_host:
            raise ValueError('Native reference lacks the matching deck/host proof scope')
        native=root/str(number)/'native'
        publication=native/'PUBLICATION.json'
        if digest(publication)!=captured['publication_sha256']:
            raise ValueError('Native publication differs after transfer')
        manifest=json.loads((native/'MANIFEST.json').read_text())
        namelist=native/'namelist.atmosphere'
        streams=native/'streams.atmosphere'
        for name,path in (('namelist',namelist),('streams',streams)):
            if digest(path)!=required[name+'_sha256']:
                raise ValueError('Required CAM deck inputs changed')
        config=resolve_namelist(namelist.read_text())
        cuda=[slot for slot,scheme in config.selectors.items()
              if (slot,scheme) in (('radiation_lw','cam_lw'),('radiation_sw','cam_sw'))]
        if not cuda:raise ValueError('Required deck does not select CAM')
        parts=required['source'].split('/')
        init=native/captured['runtime']['init']
        grid=native/(parts[1]+'.grid.nc')
        auxiliary={name:reference(native/name) for name in manifest['inputs']
                   if name not in ('namelist.atmosphere','streams.atmosphere',
                                   'atmosphere_model',init.name)}
        if any(value['sha256']!=manifest['inputs'][name] for name,value in auxiliary.items()):
            raise ValueError('Native auxiliary input differs')
        duration=captured['runtime']['duration']
        if duration!='1_00:00:00':raise ValueError('Required native forecast did not run24h')
        dt=captured['runtime']['dt']
        if 86400%dt:raise ValueError('Native24h clock is off its fixed timestep')
        command=['gpuwm-hex','forecast','--mesh',parts[1], '--grid',str(grid.resolve()),
                 '--init','{init}','--physics-namelist','{namelist}','--streams','{streams}',
                 '--physics-backend','mpasphys','--native-physics-library',library['path'],
                 '--native-pool-reader',pool_reader['path'],
                 '--mpas-source',str(args.mpas_source.resolve()),
                 '--stream-list-directory',str(args.stream_lists.resolve()),
                 '--native-physics-tables',str(args.tables.resolve()),'--hours','24',
                 '--physics',','.join(table[slot]['backend_selector']+"='hex-cuda'" for slot in cuda),
                 '--out','{out}']
        deck=dict(schema='mpas-verification-deck-v1', pin_set=manifest['pin_set_hash'],
                  mesh=parts[1],date=captured['runtime']['start'],id=required['id'],
                  role='G2',suite=config.suite,physics='on',process_count=1,ranks=1,
                  inputs=dict(namelist=reference(namelist),streams=reference(streams),init=reference(init)),
                  auxiliary_inputs=auxiliary,native=reference(publication),hex_command=command,
                  duration_steps=86400//dt,duration_hours=24,
                  hex_support_inputs={'native-library':library})
        path=args.output/(str(number)+'.json')
        encoded=json.dumps(deck,indent=2)+'\n'
        prepared.append((path,encoded))
        entries.append(dict(id=required['id'],deck=str(path.resolve()),
            deck_sha256=hashlib.sha256(encoded.encode()).hexdigest(),cuda_slots=cuda))
    index=dict(schema='mpas-cam-whole-proof-v1',process_count=1,proof_host=args.proof_host,
               slot_minutes=120,required_cases=[row['id'] for row in plan['required_cases']],
               required_plan_sha256=digest(args.plan),native_provider=provider,entries=entries,
               forecast_proven=False,complete_preparation=len(selected)==len(plan['required_cases']),
               dependencies=dict(provider_verifier=reference(args.provider_verifier),
                 comparator=comparator,pool_reader=pool_reader,mpi_library=reference(args.mpi_library)))
    args.output.mkdir(parents=True,exist_ok=False)
    for path,encoded in prepared:path.write_text(encoded)
    args.index.parent.mkdir(parents=True,exist_ok=True)
    args.index.write_text(json.dumps(index,indent=2)+'\n')
    print(len(entries),'native-bound complete CAM proof decks prepared; no forecast proof claimed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','foundation-index','native-root','output','index','provider-verifier',
                 'mpi-library','mpas-source','stream-lists','tables'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--proof-host',required=True)
    parser.add_argument('--only-case',type=int,help='prepare one whole deck for immediate paired proof; incomplete index cannot qualify the lane')
    prepare(parser.parse_args())
