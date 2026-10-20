#!/usr/bin/env python3
"""Collect actual GNU counters for a complete native synthetic matrix.

Counters redirected with GCOV_PREFIX are matched to their original absolute
object paths. Source locations are mapped back to the pristine pinned tree;
generated transport/harness branches remain outside scheme coverage totals.
This does not infer branch closure or invent inputs for uncovered branches.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from coverage import summarize


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def compilation_directory(obj):
    result=subprocess.run(['gcov-dump',str(obj)],check=True,text=True,capture_output=True)
    matches=re.findall(r'^.*:cwd: (.+)$',result.stdout,re.M)
    if len(matches)!=1 or not Path(matches[0]).is_absolute():
        raise ValueError(f'coverage object has no unique absolute compile directory: {obj}')
    return Path(matches[0])


def collect_framework(standalone,prefix,source,receipts,scratch,build,audit):
    """Collect inventoried native modules linked into real-framework fixtures."""
    standalone,prefix,source,receipts,scratch,build,audit=[p.resolve() for p in
        (standalone,prefix,source,receipts,scratch,build,audit)]
    if scratch.exists():raise ValueError('coverage scratch must be fresh')
    inventory=build/'oracle-inventory.json'
    native=json.loads(inventory.read_text())
    flags=json.loads(audit.read_text())
    if not flags['all_ok'] or flags['recipe']!='coverage':
        raise ValueError('native coverage object audit must pass')
    for relative,expected in native['generated_sha256'].items():
        if digest(build/relative)!=expected:raise ValueError('native coverage source changed')
    matrix=json.loads((receipts/'summary.json').read_text())
    if matrix['inventory_sha256']!=digest(inventory):
        raise ValueError('coverage matrix and native inventories differ')
    scratch.mkdir(parents=True)
    totals=[]
    for manifest in sorted(standalone.glob('*/synthetic-manifest.json')):
        fixture=json.loads(manifest.read_text())
        binding_path=manifest.parent/'build-manifest.json'
        binding=json.loads(binding_path.read_text())
        if (fixture['build_manifest_sha256']!=digest(binding_path) or
            binding['object_audit_sha256']!=digest(audit) or
            binding['inventory_sha256']!=digest(inventory) or binding['framework_stubs']):
            raise ValueError('native framework coverage binding differs')
        for library,expected in binding['native_library_sha256'].items():
            if digest(Path(library))!=expected:raise ValueError('native coverage library changed')
        site=next(s for s in native['sites'] if s['site']==fixture['site'])
        rows=[r for r in matrix['rows'] if r['site']==site['site'] and r['recipe']=='coverage']
        if {r['profile'] for r in rows}!=set(fixture['profiles']):
            raise ValueError('coverage fixtures do not match attempted native matrix')
        sources={m['path'] for m in native['module_state'] if m['name'] in site['state_modules']}
        objects=set()
        for relative in sorted(sources):
            candidates=list((build/'src').rglob(Path(relative).stem+'.gcno'))
            if len(candidates)!=1:raise ValueError('native module has no unique coverage object: '+relative)
            objects.add(candidates[0])
        files=[];excluded=[];provenance=[];reports=[];missing=[];exits=[]
        for ordinal,obj in enumerate(sorted(objects)):
            stage=scratch/(site['site']+'-'+str(ordinal));stage.mkdir()
            compiled=compilation_directory(obj)
            counter=prefix/str(compiled/obj.with_suffix('.gcda').name).lstrip('/')
            row={'object':str(obj),'object_sha256':digest(obj),'counter':str(counter),
                 'compilation_directory':str(compiled)}
            if not counter.is_file():
                # A module that never ran is not a zero-hit runtime witness.
                missing.append(str(counter));provenance.append(row);shutil.rmtree(stage);continue
            row['counter_sha256']=digest(counter);provenance.append(row)
            shutil.copy2(obj,stage/obj.name);shutil.copy2(counter,stage/counter.name)
            run=subprocess.run(['gcov','--json-format','-b','-c',obj.name],
                               cwd=stage,text=True,capture_output=True)
            exits.append(run.returncode)
            (receipts/(site['site']+'-'+obj.stem+'-gcov.log')).write_text(run.stdout+run.stderr)
            for compressed in sorted(stage.glob('*.gcov.json.gz')):
                data=json.loads(gzip.decompress(compressed.read_bytes()))
                report=summarize(data,source,build,compile_root=build,
                                 generated_source_sha256=native['generated_sha256'])
                if any(f['native_source'] not in sources for f in report['files']):
                    raise ValueError('coverage object does not match inventoried module sources')
                files.extend(report['files']);excluded.extend(report['excluded_generated_sources'])
                reports.append({'object':str(obj),'path':compressed.name,'sha256':digest(compressed)})
            shutil.rmtree(stage)
        result={'schema':'rrfs-native-framework-gcov-v1','site':site['site'],'native_recipe':'coverage',
                'host':matrix['host'],'process_count':matrix['process_count'],
                'profiles':list(fixture['profiles']),'native_matrix_sha256':digest(receipts/'summary.json'),
                'inventory_sha256':digest(inventory),'object_audit_sha256':digest(audit),
                'fixture_manifest_sha256':digest(manifest),'collector_sha256':digest(Path(__file__)),
                'gcov_exit':max(exits,default=1),'counter_provenance':provenance,
                'missing_runtime_counters':missing,'gcov_reports':reports,'files':files,
                'excluded_generated_sources':excluded,'branch_closure_proven':False,'forecast_proven':False}
        target=receipts/(site['site']+'-coverage.json')
        payload=(json.dumps(result,indent=2)+'\n').encode()
        if len(payload)>=10_000_000:raise ValueError('native coverage receipt exceeds delivery cap')
        target.write_bytes(payload)
        totals.append({'site':site['site'],'branches':sum(f['branches_total'] for f in files),
                       'hit':sum(f['branches_taken'] for f in files),'missing_runtime_counters':len(missing),
                       'gcov_exit':result['gcov_exit'],'receipt_sha256':digest(target)})
    (receipts/'coverage-summary.json').write_text(json.dumps(totals,indent=2)+'\n')
    return 0 if totals and all(r['gcov_exit']==0 and not r['missing_runtime_counters'] for r in totals) else 1


def collect(standalone,prefix,source,receipts,scratch,build=None,audit=None):
    if (build is None)!=(audit is None):raise ValueError('framework coverage requires build and audit')
    if build is not None:return collect_framework(standalone,prefix,source,receipts,scratch,build,audit)
    standalone,prefix,source,receipts,scratch=[p.resolve() for p in
        (standalone,prefix,source,receipts,scratch)]
    if scratch.exists():raise ValueError('coverage scratch must be fresh')
    matrix=json.loads((receipts/'summary.json').read_text())
    if not matrix['rows']:raise ValueError('native matrix contains no attempts')
    scratch.mkdir(parents=True)
    totals=[]
    for manifest in sorted(standalone.glob('*/synthetic-manifest.json')):
        site=manifest.parent.name;directory=manifest.parent/'coverage'
        if not directory.is_dir():continue
        rows=[r for r in matrix['rows'] if r['site']==site and r['recipe']=='coverage']
        fixture=json.loads(manifest.read_text())
        if {r['profile'] for r in rows}!=set(fixture['profiles']):
            raise ValueError('coverage fixtures do not match attempted native matrix')
        objects=sorted(directory.glob('*.gcno'))
        if not objects:raise ValueError('native coverage objects are absent')
        stage=scratch/site;stage.mkdir()
        provenance=[];missing=[]
        for obj in objects:
            compiled=compilation_directory(obj)
            counter=prefix/str(compiled/obj.with_suffix('.gcda').name).lstrip('/')
            shutil.copy2(obj,stage/obj.name)
            row={'object':str(obj),'object_sha256':digest(obj),'counter':str(counter),
                 'compilation_directory':str(compiled)}
            if counter.is_file():
                shutil.copy2(counter,stage/counter.name)
                row['counter_sha256']=digest(counter)
            else:
                # Never turn an absent runtime witness into zero-hit coverage.
                missing.append(str(counter))
            provenance.append(row)
        run=subprocess.run(['gcov','--json-format','-b','-c',*[p.name for p in objects]],
                           cwd=stage,text=True,capture_output=True)
        (receipts/(site+'-gcov.log')).write_text(run.stdout+run.stderr)
        files=[];excluded=[];reports=[]
        for compressed in sorted(stage.glob('*.gcov.json.gz')):
            data=json.loads(gzip.decompress(compressed.read_bytes()))
            if not data.get('current_working_directory'):
                raise ValueError('gcov report lacks its compilation directory')
            report=summarize(data,source,directory,compile_root=data['current_working_directory'])
            files.extend(report['files']);excluded.extend(report['excluded_generated_sources'])
            reports.append({'path':compressed.name,'sha256':digest(compressed)})
        result={'schema':'rrfs-native-gcov-v1','site':site,'native_recipe':'coverage',
                'profiles':list(fixture['profiles']),'native_matrix_sha256':digest(receipts/'summary.json'),
                'collector_sha256':digest(Path(__file__)),'gcov_exit':run.returncode,
                'counter_provenance':provenance,'missing_runtime_counters':missing,
                'gcov_reports':reports,'files':files,'excluded_generated_sources':excluded,
                'branch_closure_proven':False,'forecast_proven':False}
        target=receipts/(site+'-coverage.json')
        target.write_text(json.dumps(result,indent=2)+'\n')
        totals.append({'site':site,'branches':sum(f['branches_total'] for f in files),
                       'hit':sum(f['branches_taken'] for f in files),
                       'missing_runtime_counters':len(missing),'gcov_exit':run.returncode,
                       'receipt_sha256':digest(target)})
        shutil.rmtree(stage)
    (receipts/'coverage-summary.json').write_text(json.dumps(totals,indent=2)+'\n')
    return 0 if totals and all(r['gcov_exit']==0 and not r['missing_runtime_counters'] for r in totals) else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['standalone','prefix','source','receipts','scratch']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--build',type=Path,help='audited full native framework build')
    parser.add_argument('--audit',type=Path,help='passing coverage object flag audit')
    return collect(**vars(parser.parse_args()))


if __name__=='__main__':raise SystemExit(main())
