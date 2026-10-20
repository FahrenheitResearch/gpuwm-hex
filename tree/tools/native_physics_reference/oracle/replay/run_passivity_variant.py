#!/usr/bin/env python3
"""Compare complete pristine and diagnostic native forecasts with Rust.

Inputs and exact namelists are authenticated opaque files. Python never reads
weather values. This isolates compiled instrumentation on a stated host/count;
it is not a Hex forecast gate or a complete native-defect-policy witness.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def authenticate_variant(binary, seal_path):
    seal = json.loads(seal_path.read_text())
    if sha(binary.parent / 'variant.json') != seal['variant_manifest_sha256']:
        raise ValueError('variant manifest changed after compilation')
    for name, digest in seal['artifacts'].items():
        if Path(name).name != name or sha(binary.parent / name) != digest:
            raise ValueError('compiled variant artifact changed: ' + name)
    if seal['artifacts'].get(binary.name) != sha(binary):
        raise ValueError('variant executable is absent from compiled artifact seal')
    return sha(seal_path)


def compare_runs(args, out, roots, runs, initial, timestep, name):
    """Compare every stored word of two complete forecasts, using Rust."""
    same_inventory=runs[0]['outputs'].keys()==runs[1]['outputs'].keys()
    fields=[];comparator_exit=None
    if same_inventory and all(r['exit']==0 and r['outputs'] and not r['changed_inputs'] for r in runs):
        samples=[]
        for member in sorted(runs[0]['outputs']):
            match=re.fullmatch(r'(.+)\.(\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2})\.nc',member)
            if not match:raise ValueError('output has no explicit native time: '+member)
            time=datetime.datetime.strptime(match[2],'%Y-%m-%d_%H.%M.%S')
            step=(time-initial).total_seconds()/timestep
            if step<0 or not step.is_integer():
                raise ValueError('native output time is outside the timestep schedule')
            samples.append(dict(stream=match[1],steps=[int(step)],times=[time.strftime('%Y-%m-%d_%H:%M:%S')],
                strict=str((roots[0]/member).resolve()),hex=str((roots[1]/member).resolve())))
        request=out/(name+'request.json')
        write(request,dict(schema='mpas-stream-compare-v1',samples=samples))
        receipt=out/(name+'comparison.json')
        command=[str(args.comparator.resolve()),'--request',str(request.resolve()),'--out',str(receipt.resolve())]
        process=subprocess.run(command,capture_output=True,text=True)
        comparator_exit=process.returncode
        (out/(name+'comparator.log')).write_text(process.stdout+process.stderr)
        if process.returncode==0:fields=json.loads(receipt.read_text())['fields']
    return dict(same_output_inventory=same_inventory,
        comparison_complete=bool(fields) and all(r['status']=='compared' for r in fields),
        comparator_exit=comparator_exit,compared_words=sum(r.get('compared_words',0) for r in fields),
        differing_words=sum(r.get('differing_words',0) for r in fields),
        literal_files_equal=bool(runs[0]['outputs']) and same_inventory and runs[0]['outputs']==runs[1]['outputs'])


def execute(args):
    runner_hash = sha(Path(__file__))
    manifest_digest = sha(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    if sha(args.pristine) != manifest['native_binary_sha256']:
        raise ValueError('pristine executable does not match the input publication')
    variant_seal = (authenticate_variant(args.variant, args.variant_seal)
                    if args.variant_seal else None)
    tools = {str(args.mpiexec.resolve()): sha(args.mpiexec),
             str(args.comparator.resolve()): sha(args.comparator)}
    for path, member in ((args.mpiexec, 'bin/mpiexec'), (args.comparator, 'bin/rw_mpas_verify')):
        if sha(path) != manifest['files'][member]:
            raise ValueError('published native tool changed: ' + member)
    observer=getattr(args,'observer',None)
    entry_replay=getattr(args,'observer_entry_replay_root',None)
    if entry_replay and not observer:
        raise ValueError('recorded entry replay requires an external observer')
    if observer:
        tools[str(observer.resolve())]=sha(observer)
        if not args.observer_source:
            raise ValueError('native observer requires its source provenance')
        tools[str(args.observer_source.resolve())]=sha(args.observer_source)
    for member, digest in manifest['files'].items():
        if member.startswith('lib/'):
            path = args.library_dir / Path(member).name
            if sha(path) != digest:
                raise ValueError('published native library changed: ' + member)
            tools[str(path.resolve())] = digest
    for name, digest in manifest['inputs'].items():
        member = Path(name)
        if member.is_absolute() or '..' in member.parts:
            raise ValueError('unsafe input member: ' + name)
        if sha(args.inputs_root / member) != digest:
            raise ValueError('input publication changed: ' + name)
    args.out.mkdir(parents=True, exist_ok=False)
    args.work.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith('MPAS_ORACLE_')}
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               LD_LIBRARY_PATH=str(args.library_dir.resolve()) + ':' + env.get('LD_LIBRARY_PATH', ''))
    rows = []
    for number, namelist in enumerate(args.namelist):
        out = args.out / str(number)
        out.mkdir()
        text = namelist.read_text()
        namelist_digest = sha(namelist)
        start = re.search(r"config_start_time\s*=\s*['\"]([^'\"]+)['\"]", text, re.I)
        dt = re.search(r'config_dt\s*=\s*([\d.eEdD+-]+)', text, re.I)
        initial = datetime.datetime.strptime(start[1], '%Y-%m-%d_%H:%M:%S')
        timestep = float(dt[1].replace('D', 'e').replace('d', 'e'))
        replay_source=None
        if entry_replay:
            replay_source=entry_replay/str(number)/'pristine-observed-entries'
            replay_inputs=[p for p in replay_source.iterdir() if p.is_file()]
            if not replay_inputs or not (replay_source/'entry-context.tsv').is_file():
                raise ValueError('recorded native entry context absent')
            for p in replay_inputs:tools[str(p.resolve())]=sha(p)
        runs, roots = [], []
        executions=[('pristine',args.pristine),('variant',args.variant)]
        if observer:executions += [('pristine-observed',args.pristine),('variant-observed',args.variant)]
        for label, binary in executions:
            binary_digest = sha(binary)
            root = args.work / f'{number}-{label}'
            root.mkdir()
            for name in manifest['inputs']:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(args.inputs_root / name, target)
                target.chmod(target.stat().st_mode | 0o200)
            shutil.copyfile(namelist, root / 'namelist.atmosphere')
            inputs = {name: sha(root / name) for name in manifest['inputs']}
            expected_inputs = dict(manifest['inputs'])
            expected_inputs['namelist.atmosphere'] = namelist_digest
            if inputs != expected_inputs:
                raise ValueError('staged native inputs differ from authenticated publication')
            argv = [str(args.mpiexec.resolve()), '-np', '1', str(binary.resolve())]
            if label.endswith('-observed'):
                argv=[str(args.mpiexec.resolve()),'-np','1',str(observer.resolve()),str(binary.resolve()),
                      str((out/(label+'-entries')).resolve())]
                if replay_source and label=='variant-observed':
                    argv += ['--replay-entries',str(replay_source.resolve())]
            with (out / (label + '.log')).open('w') as log:
                result = subprocess.run(argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            if sha(binary) != binary_digest:
                raise ValueError('native executable changed while running: ' + label)
            outputs = {str(p.relative_to(root)): sha(p) for p in root.rglob('*.nc')
                       if str(p.relative_to(root)) not in inputs}
            runs.append(dict(label=label, argv=argv, binary_sha256=binary_digest,
                             exit=result.returncode, inputs=inputs, outputs=outputs,
                             changed_inputs=[n for n, h in inputs.items() if sha(root / n) != h]))
            if label.endswith('-observed'):
                capture_root=out/(label+'-entries')
                runs[-1]['observer_outputs']={str(p.relative_to(capture_root)):sha(p)
                    for p in capture_root.rglob('*') if p.is_file()}
            for log in root.glob('log.atmosphere.*'):
                shutil.copyfile(log, out / (label + '-' + log.name))
            roots.append(root)
        comparison=compare_runs(args,out,roots[:2],runs[:2],initial,timestep,'')
        complete=comparison['comparison_complete']
        if sha(namelist)!=namelist_digest:
            raise ValueError('native deck changed while running')
        row=dict(namelist=str(namelist),namelist_sha256=namelist_digest,runs=runs,**comparison)
        if observer:
            observed={}
            for name,indices in [('pristine-observer-',(0,2)),('variant-observer-',(1,3)),('observed-pair-',(2,3))]:
                result=compare_runs(args,out,[roots[i] for i in indices],[runs[i] for i in indices],initial,timestep,name)
                observed[name.removesuffix('-')]=result
            row['observer_comparisons']=observed
            row['observer_passive']=all(observed[name]['comparison_complete'] and observed[name]['differing_words']==0
                for name in ('pristine-observer','variant-observer'))
            if replay_source:
                row['entry_replay_intervention']=True
                row['entry_replay_source']=str(replay_source.resolve())
                row['observer_passive']=False
            complete=complete and all(c['comparison_complete'] for c in observed.values())
        write(out / 'summary.json', row)
        rows.append(row)
        # Retire only this tool's outputs after the full Rust receipt is sealed.
        if complete:
            for root in roots:
                run = runs[roots.index(root)]
                for name, digest in {**run['inputs'], **run['outputs']}.items():
                    if sha(root / name) != digest:
                        raise ValueError('native runtime file changed before retirement: ' + name)
                shutil.rmtree(root)
    passed = all(r['comparison_complete'] and r['differing_words'] == 0 and r.get('observer_passive',True) for r in rows)
    if args.variant_seal and authenticate_variant(args.variant, args.variant_seal) != variant_seal:
        raise ValueError('compiled artifact seal changed during native forecasts')
    if any(sha(Path(path)) != digest for path, digest in tools.items()):
        raise ValueError('published tool or library changed during native forecasts')
    if sha(args.manifest) != manifest_digest or sha(args.pristine) != manifest['native_binary_sha256']:
        raise ValueError('native publication or pristine executable changed during forecasts')
    write(args.out / 'summary.json', dict(schema='mpas-compiled-hook-passivity-forecast-v1',
        runner_sha256=runner_hash,
        host=socket.gethostname(), process_count=1, input_manifest_sha256=manifest_digest,
        variant_manifest_sha256=(sha(args.variant.parent / 'variant.json')
                                 if (args.variant.parent / 'variant.json').is_file() else None),
        comparator_sha256=sha(args.comparator), rows=rows, all_scientific_words_equal=passed,
        variant_artifact_seal_sha256=variant_seal, native_tools=tools,
        scope='compiled-hook diagnosis; not Hex gate or native-defect-policy closure',
        forecast_proven=False))
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs-root', 'manifest', 'pristine', 'variant', 'mpiexec',
                 'comparator', 'library-dir', 'out', 'work'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--namelist', type=Path, action='append', required=True)
    parser.add_argument('--variant-seal', type=Path)
    parser.add_argument('--observer',type=Path,help='External native entry observer; both binaries also run unwatched')
    parser.add_argument('--observer-source',type=Path,help='Source provenance of the external observer')
    parser.add_argument('--observer-entry-replay-root',type=Path,
        help='Recorded pristine entry inputs; diagnostic intervention, never a passive capture claim')
    raise SystemExit(execute(parser.parse_args()))
