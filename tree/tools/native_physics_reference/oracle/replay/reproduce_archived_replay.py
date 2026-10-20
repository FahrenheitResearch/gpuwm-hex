#!/usr/bin/env python3
"""Regenerate complete native-framework replays of a stated real archive set.

The checkout supplies the replay generator and Rust word transport. Authenticated
owned native archives supply complete numerical modules and framework libraries;
they are not rebuilt or changed. Canonical input packets stay read-only, while
GNU tar stages each invocation and Rust compares every output word.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from hexcore import site_runner


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def authority(index, dependency_root=None):
    data=json.loads(index.read_text())
    if data['schema']!='real-native-archived-replay-inputs-v1':
        raise ValueError('unsupported archived replay authority')
    if data.get('recipe','gnu-strict') not in ('gnu-strict','gnu-o3','coverage'):
        raise ValueError('unsupported native replay recipe')
    if not data['cases'] or not data['sites'] or len(set(data['sites']))!=len(data['sites']):
        raise ValueError('real replay case/site selection is empty or duplicated')
    if len({c['id'] for c in data['cases']})!=len(data['cases']):
        raise ValueError('duplicate real native case id')
    for case in data['cases']:
        if Path(case['id']).name!=case['id'] or case['id'] in ('.','..'):
            raise ValueError('invalid real native case id')
    dependencies={row['path']:row['sha256'] for row in data['dependencies']}
    def local(path):
        # Portable collection transports metadata by content hash. Recorded
        # canonical paths and the original index bytes remain unchanged.
        return Path(path) if dependency_root is None else dependency_root/dependencies[str(path)]
    for row in data['dependencies']:
        if sha(local(row['path']))!=row['sha256']:
            raise ValueError('real native dependency differs: '+row['path'])
    bound={str(Path(row['path']).resolve()) for row in data['dependencies']}
    required=[data[k] for k in ('audit','link_manifest','pins')]
    if 'producer_audit' in data:required.append(data['producer_audit'])
    required.append(str(Path(data['build'])/'oracle-inventory.json'))
    for case in data['cases']:
        required.extend(case[k] for k in ('capture','inventory','capture_index'))
        published='publication' in case
        if published!=('consolidated' in case):
            raise ValueError('published replay needs both publication and consolidation')
        if published:
            required.extend(case[k] for k in ('publication','consolidated'))
        elif not Path(case.get('archive_root','')).is_absolute():
            raise ValueError('individual invocation replay needs an absolute archive root')
    if not {str(Path(p).resolve()) for p in required}.issubset(bound):
        raise ValueError('native replay metadata dependency is not hash-bound')
    for case in data['cases']:
        producer=json.loads(local(case['capture']).read_text())
        # Scientific-word passivity is a measured scope; literal whole-file
        # equality remains separately recorded and is never inferred here.
        if (producer['schema']!='mpas-oracle-passivity-v1' or producer['exit']!=0
                or producer['changed_inputs'] or producer['process_count']!=1
                or not producer['comparison_complete'] or not producer['all_field_words_equal']
                or producer['differing_words']!=0 or producer['missing_or_incompatible_samples']):
            raise ValueError('native producer scientific comparison is incomplete or differs')
        if producer['object_audit_sha256']!=sha(local(data.get('producer_audit',data['audit']))):
            raise ValueError('native producer object audit differs')
    return data


def execute(a):
    data=authority(a.index)
    recipe=data.get('recipe','gnu-strict')
    a.out.mkdir(parents=True,exist_ok=False)
    oracle=a.checkout.resolve()/'tree/tools/native_physics_reference/oracle'
    scripts=oracle/'replay'
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    if recipe=='coverage':
        env.update(GCOV_PREFIX=str((a.out/'profiles').absolute()),GCOV_PREFIX_STRIP='0')
    native=a.out/'native';helpers=a.out/'tools';helpers.mkdir()
    links=a.out/'link-args.json'
    links.write_text(json.dumps(json.loads(Path(data['link_manifest']).read_text())['external_link_args'])+'\n')
    with (a.out/'generate.log').open('w') as log:
        subprocess.run([sys.executable,str(scripts/'prepare_framework.py'),
            '--build',data['build'],'--out',str(native.absolute()),'--recipe',recipe,
            '--compiler',data['compiler'],'--external-link-args',str(links.resolve()),
            '--sites',*data['sites']],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    (helpers/'oracle_words.rs').write_bytes((oracle/'oracle_words.rs').read_bytes())
    (helpers/'Makefile').write_text('all: oracle_words\noracle_words: oracle_words.rs\n\trustc --edition=2021 -O $< -o $@\n')
    for name,directory in [('native',native),('words',helpers)]:
        command=([str(a.build_launcher.resolve()),str(directory.resolve())] if a.build_launcher else
                 site_runner.build_command(['make','-j4','-C',str(directory.resolve())]))
        with (a.out/(name+'-build.log')).open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    build=json.loads((native/'build-manifest.json').read_text())
    artifacts=a.out/'artifacts.json'
    paths=[native/'build-manifest.json',*[native/name for name in build['replay_source_sha256']]]
    artifacts.write_text(json.dumps([{'path':str(p.resolve()),'sha256':sha(p)} for p in paths],indent=2)+'\n')
    result=dict(schema='real-native-archived-replay-reproduction-v1',checkout=str(a.checkout.resolve()),
        process_count=1,native_replay_host=os.uname().nodename,recipe=recipe,
        input_index_sha256=sha(a.index),runner_sha256=sha(Path(__file__)),
        build_manifest_sha256=sha(native/'build-manifest.json'),artifacts_sha256=sha(artifacts),
        tools_sha256={str(p.relative_to(oracle)):sha(p) for p in oracle.rglob('*')
                      if p.is_file() and p.suffix in {'.py','.rs','.json','.F','.F90'}},
        build_launcher_sha256=sha(a.build_launcher) if a.build_launcher else None,
        selected_sites=data['sites'],cases=[],comparison_complete=False,
        arrays=0,compared_words=0,differing_words=0,forecast_proven=False)
    # Use the exact checkout's matrix and shard reader; numerical payloads are
    # never decoded in Python and no expected POST word seeds a replay.
    sys.path.insert(0,str(scripts))
    import run_capture_matrix as matrix
    for case in data['cases']:
        producer=json.loads(Path(case['capture']).read_text())
        archive_root=(json.loads(Path(case['publication']).read_text())['path']
                      if 'publication' in case else case['archive_root'])
        publication_args=(['--consolidated-archive',case['consolidated'],
                           '--archive-publication',case['publication']]
                          if 'publication' in case else [])
        target=a.out/case['id']
        with (a.out/(case['id']+'.log')).open('w') as log:
            run=subprocess.run([sys.executable,str(scripts/'run_capture_matrix.py'),
                '--inventory',case['inventory'],'--pins',data['pins'],
                '--words',str((helpers/'oracle_words').resolve()),'--capture-receipt',case['capture'],
                '--working',str(target.resolve()),'--receipts',str((a.out/(case['id']+'-receipts')).resolve()),
                '--executable-dir',str(native.resolve()),'--host','local',
                '--source-dump',producer['capture']['dump'],'--archive-root',archive_root,
                '--runtime-root',case['runtime'],'--from-archives','--archive-index',case['capture_index'],
                '--sites',*data['sites'],'--replay-inventory',str(Path(data['build'])/'oracle-inventory.json'),
                '--replay-artifacts',str(artifacts.resolve()),'--replay-audit',data['audit'],
                '--replay-recipe',recipe,*publication_args],
                # The native input authority remains strict for all recipes.
                # Recipe source/interface and object audit are independently
                # authenticated by the matrix, including changed flag hashes.
                env=env,stdout=log,stderr=subprocess.STDOUT)
        summary=a.out/(case['id']+'-receipts')/'summary.json'
        row=dict(id=case['id'],exit=run.returncode,comparison_complete=False,arrays=0,
            compared_words=0,differing_words=0,producer_scientific_passive=True,
            producer_literal_byte_passive=producer['passivity_byte_equal'])
        if summary.is_file():
            measured=json.loads(summary.read_text())
            row.update(comparison_complete=measured['completed'] and measured['all_outputs_bitwise_equal'],
                       summary_sha256=sha(summary),calls=len(measured['captures']))
            for call in measured['captures']:
                receipt=matrix.harness.read_replay_receipt(summary.parent/call['receipt'])
                for capture in receipt['captures']:
                    for array in capture['arrays'].values():
                        row['arrays']+=1
                        row['compared_words']+=array['words']
                        row['differing_words']+=array['different_words']
        result['cases'].append(row)
        for key in ('arrays','compared_words','differing_words'):result[key]+=row[key]
        (a.out/'reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
        # This fresh owned directory contains only staged runtime tables and
        # disposable records; original hash-bound runtime inputs stay intact.
        if target.exists():shutil.rmtree(target)
    if recipe=='coverage':
        result['coverage_counters_sha256']={str(p.relative_to(a.out)):sha(p)
                                            for p in (a.out/'profiles').rglob('*.gcda')}
    result['comparison_complete']=all(c['exit']==0 and c['comparison_complete']
        and c['compared_words']>0 and c['differing_words']==0 for c in result['cases'])
    (a.out/'reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['comparison_complete'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkout','index','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--build-launcher',type=Path,help='Host build adapter; must return built executables to the generated directory')
    a=p.parse_args()
    try:raise SystemExit(execute(a))
    except (OSError,ValueError,KeyError,subprocess.CalledProcessError) as error:
        print('archived native replay: '+str(error),file=sys.stderr)
        raise SystemExit(2)
