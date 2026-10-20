"""Observe unchanged original native CAM allocations and replay complete captures.

MPAS module_ra_cam.F:195-232,364-365,678-679. Test-only PRE observations
never replace original failure receipts or qualify whole forecasts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def offsets(binary):
    symbols=subprocess.check_output(['nm','-n',str(binary)],text=True)
    entries=[int(line.split()[0],16) for line in symbols.splitlines()
             if line.endswith(' __module_ra_cam_MOD_camrad')]
    assert len(entries)==1
    header=subprocess.check_output(['readelf','-h',str(binary)],text=True)
    assert re.search(r'Type:\s+DYN',header),'Tracer requires native PIE'
    assembly=subprocess.check_output(['objdump','--disassemble=__module_ra_cam_MOD_camrad',
                                      '--line-numbers',str(binary)],text=True)
    source=None;found={364:[],365:[]}
    for line in assembly.splitlines():
        match=re.search(r'module_ra_cam\.F:(\d+)',line)
        if match:source=int(match[1])
        instruction=re.match(r'\s*([0-9a-f]+):\s+((?:[0-9a-f]{2}\s+)+)\s*(.*)',line)
        if instruction and source in found and re.search(r'call\s+.*<malloc@plt>',instruction[3]):
            count=len(instruction[2].split())
            found[source].append(dict(source_line=source,instruction=line.strip(),
                                      return_offset=int(instruction[1],16)+count))
    assert all(len(rows)==1 for rows in found.values()),found
    return entries[0],found[365][0],found[364][0]


def observe(plan,root,tracer,all_fixtures=False,reuse_root=None):
    from compact_compare import boundary,slices
    selected=json.loads(plan.read_text())
    if not all_fixtures:selected=[row for row in selected if row['executable']=='cam_real_capture']
    assert len(selected)==(14 if all_fixtures else 6)
    root.mkdir(parents=True,exist_ok=False)
    results=[]
    for row in selected:
        destination=root/row['label'];destination.mkdir()
        baseline=Path(row['directory']).resolve()
        source=Path(row['native_source_directory'])
        binary=source/row['executable']
        assert digest(binary)==row['native_executable_sha256']
        for name,expected in row['files'].items():
            assert (baseline/name).stat().st_size==expected['bytes']
            assert digest(baseline/name)==expected['sha256']
        previous=Path(reuse_root)/row['label'] if reuse_root else None
        if previous and (previous/'observation.json').exists():
            result=json.loads((previous/'observation.json').read_text())
            assert result['original_native_outputs_unchanged']
            assert result['native_executable_sha256']==row['native_executable_sha256']
            assert result['expected_records']==row['expected_records']
            for name,expected in result['native_files'].items():
                assert digest(previous/name)==expected==row['files'][name]['sha256']
            for name,expected in result['entry_files'].items():
                assert digest(previous/('entry-'+name+'.bin'))==expected
            for path in previous.iterdir():
                if path.is_file():shutil.copy2(path,destination/path.name)
            results.append(result)
            print(row['label'],'reused authenticated complete observation',flush=True)
            continue
        schema=json.loads(Path(row['inventory']).read_text())['schema']
        assert [(index,schema['arguments'][index]['name']) for index in (97,98,101,102)]==[
            (97,'its'),(98,'ite'),(101,'kts'),(102,'kte')]
        # run_columns.F90:105-108 opens argument 1 with STATUS='replace':
        # synthetic native-columns.bin is an output, never a fixture input.
        # The real-profile reader consumes its existing native column input.
        input_names=('native-columns.bin',) if row['executable']=='cam_real_capture' else ()
        for name in (row['executable'],*input_names,
                     'CAM_ABS_DATA.DBL','CAM_AEROPT_DATA.DBL','OZONE_DAT.TBL','OZONE_LAT.TBL','OZONE_PLEV.TBL'):
            shutil.copy2(source/name,destination/name)
        for name in ('stored-columns.bin','stored-columns.json'):
            if (source/name).exists():shutil.copy2(source/name,destination/name)
        copied=destination/row['executable']
        assert digest(copied)==row['native_executable_sha256']
        entry,ci,cl=offsets(copied)
        command=[str(tracer.resolve()),str(copied.resolve()),hex(entry),hex(ci['return_offset']),
                 hex(cl['return_offset']),str(destination.resolve()),*row['arguments']]
        with (destination/'native-trace.log').open('w') as log:
            run=subprocess.run(command,cwd=destination,stdout=log,stderr=subprocess.STDOUT)
        assert run.returncode==0,(row['label'],run.returncode)
        text=(destination/'native-trace.log').read_text()
        assert f"CAM_NATIVE_ENTRY_COMPLETE calls={row['expected_records']} exit=0" in text
        comparisons={}
        for name in ('cam-before.bin','cam-after.bin'):
            receipt=destination/(name+'.comparison.json')
            comparisons[name]=boundary(baseline/name,destination/name,Path(row['inventory']),receipt)
            assert len(json.loads(receipt.read_text())['records'])==row['expected_records']
        for name in ('cam-state.bin','cam-state-after.bin'):
            comparisons[name]=slices(baseline/name,destination/name,
                [('Complete native module state',schema['globals'],1)],
                destination/(name+'.comparison.json'),'unchanged executable traced versus original capture')
        result=dict(label=row['label'],native_executable_sha256=digest(copied),
                    baseline=str(baseline),inventory=row['inventory'],expected_records=row['expected_records'],
                    entry_offset=entry,allocation_sites=[ci,cl],process_count=1,
                    host=__import__('socket').gethostname(),comparison_differences=comparisons,
                    original_native_outputs_unchanged=not any(comparisons.values()),
                    native_files={name:digest(destination/name) for name in row['files']},
                    entry_files={name:digest(destination/('entry-'+name+'.bin')) for name in ('tauxci','tauxcl')},
                    no_injected_library=True,no_child_allocation=True,no_post_seed=True,
                    excluded_words=0,whole_model_acceptance=False)
        write(destination/'observation.json',result);results.append(result)
        write(root/'summary.json',dict(complete=len(results)==len(selected),records=sum(r['expected_records'] for r in results),
            fixtures=results,original_failures_superseded=False,whole_model_acceptance=False,
            original_native_outputs_unchanged=all(r['original_native_outputs_unchanged'] for r in results)))
        print(row['label'],comparisons,flush=True)
    write(root/'summary.json',dict(complete=True,records=sum(r['expected_records'] for r in results),
        fixtures=results,original_failures_superseded=False,whole_model_acceptance=False,
        original_native_outputs_unchanged=all(r['original_native_outputs_unchanged'] for r in results)))
    return int(not all(r['original_native_outputs_unchanged'] for r in results))


def replay(root,result_root):
    from replay_cuda import replay as replay_capture
    result_root.mkdir(parents=True,exist_ok=False)
    observation=json.loads((root/'summary.json').read_text())
    assert observation['complete']
    results=[]
    for row in observation['fixtures']:
        native=root/row['label'];destination=result_root/row['label'];destination.mkdir()
        # Trace mismatch remains a failure, never substitute it for old captures.
        assert row['original_native_outputs_unchanged'],row['label']
        for name,expected in row['native_files'].items():
            assert digest(native/name)==expected
            shutil.copy2(Path(row['baseline'])/name,destination/name)
        for name,expected in row['entry_files'].items():
            assert digest(native/('entry-'+name+'.bin'))==expected
        receipt=destination/'boundary.json'
        rc=replay_capture(destination,Path(row['inventory']),receipt,rust_transport=True,native_entry=native)
        boundary=json.loads(receipt.read_text());state=json.loads((destination/'boundary-state.json').read_text())
        assert len(boundary['records'])==row['expected_records']
        results.append(dict(label=row['label'],records=len(boundary['records']),returncode=rc,
            boundary_words=boundary['words'],boundary_differing_words=boundary['total_differing_words'],
            state_words=state['words'],state_differing_words=state['differing_words'],
            native_executable_sha256=row['native_executable_sha256']))
        write(result_root/'summary.json',dict(complete=len(results)==len(observation['fixtures']),
            fixtures=results,records=sum(r['records'] for r in results),
            differing_words=sum(r['boundary_differing_words']+r['state_differing_words'] for r in results),
            words=sum(r['boundary_words']+r['state_words'] for r in results),
            original_unseeded_failures_superseded=False,whole_model_acceptance=False,
            observed_automatic_pre_test_only=True,excluded_words=0,process_count=1,
            observation_summary_sha256=digest(root/'summary.json')))
    return int(any(r['returncode'] for r in results))


def replay_batch(root,result_root,limit):
    """Checkpoint complete records; only the full fixed matrix can pass."""
    from replay_cuda import replay as replay_capture
    observation=json.loads((root/'summary.json').read_text())
    assert observation['complete'] and observation['original_native_outputs_unchanged']
    result_root.mkdir(parents=True,exist_ok=True)
    results=[];budget=limit;failed=False
    for row in observation['fixtures']:
        native=root/row['label'];destination=result_root/row['label']
        receipt=destination/'boundary.json';checkpoint=destination/'checkpoint.json'
        for name,expected in row['native_files'].items():assert digest(native/name)==expected
        for name,expected in row['entry_files'].items():assert digest(native/('entry-'+name+'.bin'))==expected
        if not destination.exists():
            destination.mkdir()
            for name in row['native_files']:shutil.copy2(Path(row['baseline'])/name,destination/name)
        existing=json.loads(receipt.read_text()) if receipt.exists() else {}
        if not existing.get('complete') and budget:
            before=json.loads(checkpoint.read_text())['records'] if checkpoint.exists() else 0
            rc=replay_capture(destination,Path(row['inventory']),receipt,rust_transport=True,
                native_entry=native,checkpoint=checkpoint,record_limit=budget,recover_tail=True)
            existing=json.loads(receipt.read_text())
            after=len(existing['records']);budget-=after-before
            assert budget>=0
            if rc==1:failed=True
        if existing:
            complete=existing.get('complete',False)
            state=json.loads((destination/'boundary-state.json').read_text()) if complete else {}
            if complete:assert len(existing['records'])==row['expected_records']
            results.append(dict(label=row['label'],records=len(existing['records']),complete=complete,
                boundary_words=existing['words'],boundary_differing_words=existing['total_differing_words'],
                state_words=state.get('words',0),state_differing_words=state.get('differing_words',0)))
            if existing['total_differing_words'] or state.get('differing_words',0):failed=True
        if not budget:break
    complete=len(results)==len(observation['fixtures']) and all(r['complete'] for r in results)
    summary=dict(complete=complete,records=sum(r['records'] for r in results),fixtures=results,
        words=sum(r['boundary_words']+r['state_words'] for r in results),
        differing_words=sum(r['boundary_differing_words']+r['state_differing_words'] for r in results),
        required_records=observation['records'],observed_automatic_pre_test_only=True,
        original_unseeded_failures_superseded=False,whole_model_acceptance=False,excluded_words=0,process_count=1,
        observation_summary_sha256=digest(root/'summary.json'))
    write(result_root/'summary.json',summary)
    if failed:return 1
    return 0 if complete else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['observe','replay','replay-batch'])
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--tracer',type=Path)
    parser.add_argument('--result-root',type=Path)
    parser.add_argument('--all-fixtures',action='store_true')
    parser.add_argument('--reuse-root',type=Path)
    parser.add_argument('--record-limit',type=int,default=32)
    args=parser.parse_args()
    raise SystemExit(observe(args.plan,args.root,args.tracer,args.all_fixtures,args.reuse_root) if args.mode=='observe'
                     else replay_batch(args.root,args.result_root,args.record_limit) if args.mode=='replay-batch'
                     else replay(args.root,args.result_root))
