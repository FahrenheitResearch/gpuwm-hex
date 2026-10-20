#!/usr/bin/env python3
"""Replay every invocation while keeping disposable working data bounded.

Python selects and authenticates opaque files. GNU tar/gzip stage archives;
Fortran replays the physics and oracle_words decodes and compares its fields.
The canonical archive and its manifest stay on the source host.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess

TOOLS=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('column_oracle_replay',TOOLS/'column_oracle_replay.py')
harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def member_prefix(name):
    # Native derived-state labels retain Fortran's percent component selector.
    # It is an opaque filename character, never a shell expression.
    if not re.fullmatch(r'[A-Za-z0-9_.%+-]+\.bin',name):raise ValueError('invalid native capture filename')
    if '.pre.' not in name and '.post.' not in name:raise ValueError('native record has no phase')
    prefix=name.split('.pre.')[0].split('.post.')[0]
    if not re.fullmatch(r'[A-Za-z0-9_.+-]+',prefix):raise ValueError('invalid native capture prefix')
    return prefix


def host_command(host,command):
    return ['bash','-c',command] if host=='local' else ['ssh',host,command]


def remote(host,command,**kwargs):
    return subprocess.run(host_command(host,command),check=True,**kwargs)


def copy_to_host(source,host,target):
    if host=='local':shutil.copy2(source,target)
    else:subprocess.run(['rsync',str(source),host+':'+target],check=True)


def diagnose(receipt,stage,results,words,target):
    """Keep first differing native/input/output words before scratch retirement.

    Rust alone decodes values. The full comparison receipt binds every output
    hash and every differing word count; these samples explain those failures.
    """
    report=harness.read_replay_receipt(receipt)
    captures=[]
    for capture in report.get('captures',[]):
        arrays={}
        for name,comparison in capture['arrays'].items():
            location=comparison.get('first_difference')
            if not location:continue
            indices=','.join(map(str,location['indices']))
            prefix=capture['capture']
            row={'comparison':comparison,'samples':{}}
            for label,root,phase in [('reference',stage,'post'),('replay',results,'post'),('input',stage,'pre')]:
                path=root/(prefix+'.'+phase+'.'+name+'.bin')
                if not path.is_file():continue
                sample=subprocess.run([str(words),'sample',str(path),indices],capture_output=True,text=True)
                row['samples'][label]={'record_sha256':harness.hash_file(path),
                    'word':json.loads(sample.stdout) if sample.returncode==0 else None,
                    'sample_error':sample.stderr if sample.returncode else None}
            arrays[name]=row
        if arrays:captures.append({'capture':capture['capture'],'site':capture['site'],'arrays':arrays})
    if captures:
        harness.write_replay_receipt(target,{'schema':'mpas-native-replay-differing-word-samples-v1',
            'replay_receipt_sha256':harness.hash_file(receipt),'word_helper_sha256':harness.hash_file(words),
            'captures':captures,'forecast_proven':False})


def unpack(producer,target,members=None,compressed=False,selection=None):
    """Pipe an opaque archive through the system archiver, without field I/O."""
    target.mkdir()
    with members.open('rb') if members else open('/dev/null','rb') as inputs:
        source=subprocess.Popen(producer,stdin=inputs,stdout=subprocess.PIPE)
        flags=['--use-compress-program=zstd -d --long=31','--ignore-zeros','--wildcards','./'+selection+'*.bin'] if selection else []
        result=subprocess.run(['tar','-xz' if compressed else '-x','-C',str(target),*flags],stdin=source.stdout)
        source.stdout.close();status=source.wait()
    if status or result.returncode:raise ValueError(f'capture archive transport failed ({status},{result.returncode})')


def pack(root,host,target):
    """Compress on this build host and stream directly to canonical storage."""
    partial=target+'.partial'
    remote(host,'test ! -e '+shlex.quote(target)+' && test ! -e '+shlex.quote(partial))
    tar=subprocess.Popen(['tar','-C',str(root),'-cf','-','.'],stdout=subprocess.PIPE)
    compressor=subprocess.Popen(['gzip','-1'],stdin=tar.stdout,stdout=subprocess.PIPE)
    tar.stdout.close()
    sink=subprocess.run(host_command(host,'cat > '+shlex.quote(partial)),stdin=compressor.stdout)
    compressor.stdout.close();statuses=[tar.wait(),compressor.wait(),sink.returncode]
    if any(statuses):raise ValueError(f'capture archive compression failed {statuses}')
    remote(host,'mv '+shlex.quote(partial)+' '+shlex.quote(target))
    digest=remote(host,'sha256sum '+shlex.quote(target),capture_output=True,text=True).stdout.split()[0]
    if not re.fullmatch('[0-9a-f]{64}',digest):raise ValueError('invalid canonical archive SHA256')
    return digest


def archive_origin(a):
    """Authenticate a shared catalogue while retaining original receipt paths."""
    path=getattr(a,'archive_publication',None)
    if path is None:return a.archive_root
    if not a.from_archives or not a.consolidated_archive or not a.archive_index:
        raise ValueError('archive publication requires complete consolidated archived replay')
    publication=json.loads(path.read_text())
    if (publication['schema']!='mpas-oracle-archived-publication-v1' or
        publication['path']!=a.archive_root or publication['forecast_proven']):
        raise ValueError('shared archive publication provenance differs')
    members=publication['members']
    for name,local in [('capture-index.json',a.archive_index),('capture.json',a.capture_receipt),
                       ('inventory.json',a.inventory),('consolidated.archive.json',a.consolidated_archive)]:
        if harness.hash_file(local)!=members[name]['sha256']:
            raise ValueError('shared archive metadata differs')
    if any(Path(name).name!=name or not re.fullmatch(r'[A-Za-z0-9_.+-]+',name) for name in members):
        raise ValueError('invalid shared archive member')
    # One source-host pass authenticates opaque published files, including the
    # compressed payload. Rust continues to decode every staged native record.
    command='sha256sum -- '+' '.join(shlex.quote(a.archive_root+'/'+name) for name in members)
    checks=remote(a.host,command,capture_output=True,text=True).stdout.splitlines()
    actual={Path(line[66:]).name:line[:64] for line in checks}
    if actual!={name:row['sha256'] for name,row in members.items()}:
        raise ValueError('shared archive member hash differs')
    return publication['source_archive_root']


def replay_build(a, capture_inventory):
    """Bind each replay recipe to its source interface and authenticated binaries.

    Owned Makefile flags alone change instrumentation_sha256 across recipes;
    every other inventory key must agree, including all generated source bytes.
    """
    recipe=getattr(a,'replay_recipe','gnu-strict')
    paths=[getattr(a,n,None) for n in ('replay_inventory','replay_artifacts','replay_audit')]
    if not any(paths):
        if recipe!='gnu-strict':raise ValueError('alternate replay recipe requires inventory, artifacts and audit')
        return {'recipe':recipe,'explicit_build_binding':False}
    if not all(paths):raise ValueError('replay build binding requires inventory, artifacts and audit')
    inventory,artifacts,audit=paths
    other=json.loads(inventory.read_text())
    keys=set(capture_inventory)|set(other)
    changed=[k for k in keys if k!='instrumentation_sha256' and capture_inventory.get(k)!=other.get(k)]
    if changed:raise ValueError('replay source interface differs: '+','.join(sorted(changed)))
    manifest_path=a.executable_dir/'build-manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if (manifest['schema']!='mpas-native-framework-replays-v1' or manifest['recipe']!=recipe or
        manifest['inventory_sha256']!=harness.hash_file(inventory) or manifest['framework_stubs']):
        raise ValueError('replay build recipe or inventory differs')
    flag_audit=json.loads(audit.read_text())
    if flag_audit['recipe']!=recipe or not flag_audit['all_ok']:
        raise ValueError('replay compiler audit does not pass for the recipe')
    for row in flag_audit['objects']:
        object_path=inventory.parent/row['object']
        if not object_path.resolve().is_relative_to(inventory.parent.resolve()):
            raise ValueError('invalid audited native object path')
        if harness.hash_file(object_path)!=row['object_sha256']:
            raise ValueError('replay audited native object differs: '+row['object'])
    authenticated={str(Path(row['path']).resolve()):row['sha256'] for row in json.loads(artifacts.read_text())}
    if authenticated.get(str(manifest_path.resolve()))!=harness.hash_file(manifest_path):
        raise ValueError('replay build manifest artifact differs')
    for name,digest in manifest['replay_source_sha256'].items():
        if Path(name).name!=name or not name.startswith('replay_'):raise ValueError('invalid replay build target')
        source=a.executable_dir/'sources'/(name+'.F90')
        executable=a.executable_dir/name
        if harness.hash_file(source)!=digest:raise ValueError('replay generated source differs: '+name)
        if authenticated.get(str(executable.resolve()))!=harness.hash_file(executable):
            raise ValueError('replay executable artifact differs: '+name)
    return {'recipe':recipe,'explicit_build_binding':True,
            'inventory_sha256':harness.hash_file(inventory),'artifacts_sha256':harness.hash_file(artifacts),
            'audit_sha256':harness.hash_file(audit),'build_manifest_sha256':harness.hash_file(manifest_path),
            'recipe_specific_instrumentation_sha256':other['instrumentation_sha256']}


def run(a):
    for name in ('working','receipts'):
        path=getattr(a,name)
        if path.exists():raise ValueError(f'{name} must be fresh')
        path.mkdir(parents=True)
    inventory=json.loads(a.inventory.read_text())
    build_binding=replay_build(a,inventory)
    capture=json.loads(a.capture_receipt.read_text())
    if capture['schema']!='mpas-oracle-passivity-v1' or capture['exit']!=0 or capture['changed_inputs']:
        raise ValueError('capture must come from a completed native forecast with unchanged inputs')
    if capture['inventory_sha256']!=harness.hash_file(a.inventory):raise ValueError('capture inventory hash differs')
    if capture['capture']['dump']!=a.source_dump:raise ValueError('capture source directory differs')
    # Source has finished. Group every record, including every nested entry,
    # using filenames; all typed metadata is checked by the Rust reader.
    groups={};archives={};consolidated=None
    origin=archive_origin(a)
    if a.consolidated_archive:
        if not a.from_archives:raise ValueError('consolidated archive requires archived replay')
        consolidated=json.loads(a.consolidated_archive.read_text())
        if (consolidated['schema']!='mpas-native-consolidated-archive-v1' or
            not consolidated['all_decoded_tar_bytes_equal'] or
            consolidated['archive']!=origin+'/complete.tar.zst' or
            consolidated['capture_receipt_sha256']!=harness.hash_file(a.capture_receipt) or
            consolidated['inventory_sha256']!=harness.hash_file(a.inventory)):
            raise ValueError('consolidated native provenance differs')
        actual=remote(a.host,'sha256sum '+shlex.quote(a.archive_root+'/complete.tar.zst'),capture_output=True,text=True).stdout.split()[0]
        if actual!=consolidated['archive_sha256']:raise ValueError('consolidated archive hash differs')
    if a.sites and not set(a.sites).issubset({site['site'] for site in inventory['sites']}):
        raise ValueError('selected native replay site is absent from the inventory')
    if a.from_archives:
        if a.retire_raw:raise ValueError('archived replay cannot retire raw records')
        if a.archive_index is None and not a.sites:
            raise ValueError('whole archived replay requires the completed capture index')
        index=json.loads(a.archive_index.read_text()) if a.archive_index else None
        if index and (index['schema']!='mpas-native-capture-replay-matrix-v1' or not index['completed'] or
                      index['capture_receipt_sha256']!=harness.hash_file(a.capture_receipt) or
                      index['inventory_sha256']!=harness.hash_file(a.inventory)):
            raise ValueError('archived capture index is incomplete or differs')
        if consolidated and index and consolidated['capture_index_sha256']!=harness.hash_file(a.archive_index):
            raise ValueError('consolidated capture index hash differs')
        names=remote(a.host,'find -H '+shlex.quote(a.archive_root)+" -maxdepth 1 -type f -name '*.archive.json' -printf '%f\\n'",capture_output=True,text=True).stdout.splitlines()
        for name in names:
            if name=='consolidated.archive.json':continue
            prefix=name.removesuffix('.archive.json')
            member_prefix(prefix+'.pre.check.bin')
            if a.sites and not any(prefix.startswith(site+'.s') for site in a.sites):continue
            row=json.loads(remote(a.host,'cat '+shlex.quote(a.archive_root+'/'+name),capture_output=True,text=True).stdout)
            if (row['schema']!='mpas-native-invocation-archive-v1' or row['capture']!=prefix or
                row['capture_receipt_sha256']!=harness.hash_file(a.capture_receipt) or
                row['archive']!=origin+'/'+prefix+'.tar.gz' or
                not row['all_restored_member_sha256_equal']):
                raise ValueError('canonical archive provenance differs')
            archives[prefix]=row;groups[prefix]=range(row['records'])
        expected={row['capture']:row['records'] for row in index['captures'] if not a.sites or row['site'] in a.sites} if index else None
        if expected is not None and expected!={prefix:len(files) for prefix,files in groups.items()}:
            raise ValueError('canonical archives do not match the complete capture index')
    else:
        names=remote(a.host,'find '+shlex.quote(a.source_dump)+" -maxdepth 1 -type f -name '*.bin' -printf '%f\\n'",capture_output=True,text=True).stdout.splitlines()
        for name in names:
            prefix=member_prefix(name)
            if a.sites and not any(prefix.startswith(site+'.s') for site in a.sites):continue
            groups.setdefault(prefix,[]).append(name)
    if not groups:raise ValueError('capture source has no records')
    remote(a.host,'mkdir -p '+shlex.quote(a.archive_root))
    runtime=a.working/'runtime'
    members=a.working/'runtime-members.txt'
    members.write_text('\n'.join(n for n in capture['inputs'] if not n.endswith('.nc'))+'\n')
    if any(not Path(n).parts or Path(n).is_absolute() or '..' in Path(n).parts
           for n in members.read_text().splitlines()):raise ValueError('invalid native runtime member')
    # Native decks link tables into their working directory. Materialize their
    # bytes here; absolute links may point at a source-host-only data mount.
    unpack(host_command(a.host,'tar -h -C '+shlex.quote(a.runtime_root)+' -cf - --verbatim-files-from -T -'),runtime,members)
    # Authenticate runtime inputs independently of archive transport.
    for name,row in capture['inputs'].items():
        if not name.endswith('.nc') and harness.hash_file(runtime/name)!=row['sha256']:
            raise ValueError('native replay runtime input hash differs')
    rows=[]
    for prefix,files in sorted(groups.items(),key=lambda item:(len(item[1]),item[0])):
        stage=a.working/'capture';results=a.working/'outputs';restored=a.working/'restore'
        if a.from_archives:
            archived=archives[prefix]
            if consolidated:
                original=next(row for row in consolidated['invocations'] if row['capture']==prefix)
                if any(original[key]!=archived[key] for key in archived):raise ValueError('consolidated invocation provenance differs')
                target=a.archive_root+'/complete.tar.zst';digest=consolidated['archive_sha256']
                unpack(host_command(a.host,'cat '+shlex.quote(target)),stage,selection=prefix)
                with (stage/'manifest.json').open('wb') as manifest_output:
                    remote(a.host,'cat '+shlex.quote(a.archive_root+'/'+prefix+'.manifest.json'),stdout=manifest_output)
                if harness.hash_file(stage/'manifest.json')!=original['manifest_sha256']:
                    raise ValueError('consolidated invocation manifest hash differs')
            else:
                target=archived['archive']
                digest=remote(a.host,'sha256sum '+shlex.quote(target),capture_output=True,text=True).stdout.split()[0]
                if digest!=archived['archive_sha256']:raise ValueError('canonical archive hash differs')
                unpack(host_command(a.host,'cat '+shlex.quote(target)),stage,compressed=True)
            sealed=harness.verify(stage)
            if (sealed['dump_root_sha256']!=archived['dump_root_sha256'] or
                len(sealed['files'])!=archived['records'] or
                sealed['inventory_sha256']!=harness.hash_file(a.inventory)):
                raise ValueError('canonical invocation manifest differs')
        else:
            members.write_text('\n'.join(sorted(files))+'\n')
            unpack(host_command(a.host,'tar -C '+shlex.quote(a.source_dump)+' -cf - --verbatim-files-from -T -'),stage,members)
            sealed=harness.seal(stage,a.inventory,a.pins,a.words,'gnu-strict',capture['process_count'],'rrfs')
        if set(sealed['captures'])!={prefix}:raise ValueError('staged invocation contains another native capture')
        site=sealed['captures'][prefix]['site']
        receipt=a.receipts/(prefix+'.replay.json')
        executable=a.executable_dir/('replay_'+site)
        options=argparse.Namespace(dump=stage,scheme=site,ref='rrfs',backend='native-replay',
            recipe=build_binding['recipe'],results=results,receipt=receipt,executable=executable,
            words=a.words,coverage=None,gpu_slot_minutes=None,process_count=capture['process_count'])
        # Runtime tables retain their native basenames; no numerical initializer is replaced.
        import os
        previous=Path.cwd()
        try:
            os.chdir(runtime)
            try:status=harness.replay(options,sealed)
            except (OSError,ValueError,subprocess.CalledProcessError) as error:
                harness.failure_receipt(options,error,sealed);status=2
        finally:os.chdir(previous)
        if status==1:diagnose(receipt,stage,results,a.words,a.receipts/(prefix+'.samples.json'))
        for log in results.glob('*.log'):
            shutil.copy2(log,a.receipts/(prefix+'.native.log'))
        if not a.from_archives:
            target=a.archive_root+'/'+prefix+'.tar.gz'
            digest=pack(stage,a.host,target)
            unpack(host_command(a.host,'cat '+shlex.quote(target)),restored,compressed=True)
            returned=harness.verify(restored)
            if returned['dump_root_sha256']!=sealed['dump_root_sha256']:
                raise ValueError('restored invocation root differs')
        archive_receipt=a.receipts/(prefix+'.archive.json')
        write(archive_receipt,{'schema':'mpas-native-invocation-archive-v1','canonical_host':a.host,
            'archive':target,'archive_sha256':digest,'dump_root_sha256':sealed['dump_root_sha256'],
            'capture':prefix,'records':len(sealed['files']),'raw_bytes':sum(r['bytes'] for r in sealed['files'].values()),
            'capture_receipt_sha256':harness.hash_file(a.capture_receipt),'all_restored_member_sha256_equal':True,
            'replay_receipt_sha256':harness.hash_file(receipt),'replay_status':status,'forecast_proven':False})
        # Preserve the manifest separately on the canonical host for inspection.
        if not a.from_archives:
            copy_to_host(stage/'manifest.json',a.host,a.archive_root+'/'+prefix+'.manifest.json')
            copy_to_host(archive_receipt,a.host,a.archive_root+'/'+prefix+'.archive.json')
        if a.retire_raw:
            delete="import os,sys; root=sys.argv[1]; " \
                   "[(os.unlink(os.path.join(root,n.rstrip('\\n')))) for n in sys.stdin]"
            with members.open('rb') as inputs:
                remote(a.host,'python3 -c '+shlex.quote(delete)+' '+shlex.quote(a.source_dump),stdin=inputs)
        rows.append({'capture':prefix,'site':site,'status':status,'receipt':receipt.name,
                     'archive_receipt':archive_receipt.name,'records':len(files)})
        write(a.receipts/'progress.json',{'captures':rows,'completed':False,'forecast_proven':False})
        # Both native success and failure are retained; outputs can be regenerated from the exact archive.
        for path in (stage,results,restored):
            if path.exists():shutil.rmtree(path)
    summary={'schema':'mpas-native-capture-replay-matrix-v1','capture_receipt_sha256':harness.hash_file(a.capture_receipt),
        'inventory_sha256':harness.hash_file(a.inventory),'runner_sha256':harness.hash_file(Path(__file__)),
        'process_count':capture['process_count'],'replay_build':build_binding,'captures':rows,'records':sum(r['records'] for r in rows),
        'all_outputs_bitwise_equal':all(r['status']==0 for r in rows),'completed':True,'forecast_proven':False,
        'scope':'selected-native-sites' if a.sites else 'complete-capture-index',
        'selected_sites':a.sites,'complete_source_index_authenticated':bool(a.archive_index) if a.from_archives else not bool(a.sites)}
    write(a.receipts/'summary.json',summary)
    return 0 if summary['all_outputs_bitwise_equal'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inventory','pins','words','capture-receipt','working','receipts','executable-dir'):
        p.add_argument('--'+name,type=lambda value:Path(value).resolve(),required=True)
    for name in ('host','source-dump','archive-root','runtime-root'):p.add_argument('--'+name,required=True)
    p.add_argument('--replay-recipe',choices=['gnu-strict','gnu-o3','coverage'],default='gnu-strict')
    for name in ('replay-inventory','replay-artifacts','replay-audit'):
        p.add_argument('--'+name,type=lambda value:Path(value).resolve())
    p.add_argument('--retire-raw',action='store_true',help='retire owned raw records only after archive hash authentication and replay receipt')
    p.add_argument('--from-archives',action='store_true',help='authenticate and replay every canonical invocation archive without rewriting it')
    p.add_argument('--archive-index',type=lambda value:Path(value).resolve(),help='completed capture matrix summary required for archived replay')
    p.add_argument('--sites',nargs='+',help='explicit native site subset; without a complete index, scope is every available archive for these sites')
    p.add_argument('--consolidated-archive',type=lambda value:Path(value).resolve(),help='authenticated canonical long-window archive receipt')
    p.add_argument('--archive-publication',type=lambda value:Path(value).resolve(),help='immutable shared archive catalogue; original producer paths remain authenticated provenance')
    a=p.parse_args()
    try:return run(a)
    except (OSError,ValueError,subprocess.CalledProcessError) as error:
        print('native capture matrix: '+str(error));return 2

if __name__=='__main__':raise SystemExit(main())
