"""Whole shipping restart forecasts; mpas_atm_core.F:109-150,589-705.

Run inside the proof host's GPU queue. Every original stream and output time
is graded by Rust. Straight/half-day/hex restart/native restart forecasts and
both native consumption directions use the supplied committed checkout.
Native physics continuity differences are recorded separately from native
consuming hex versus native consuming native. No exclusions are permitted.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET

from hexcore.config import parse_fortran_namelist
from replay_native_physics_restart_v842 import digest


def native_launcher(executable):
    """Launch the native process control; MPAS mpas_atm_core.F:109-150.

    A system launcher may use an incompatible PMI protocol. Prefer the launcher
    belonging to the MPI library actually linked by the strict executable.
    """
    explicit=os.environ.get('MPAS_NATIVE_MPIEXEC')
    if explicit:return explicit
    linked=subprocess.check_output(['ldd',str(executable)],text=True)
    match=re.search(r'\s*libmpi\.so\S*\s+=>\s+(\S+)',linked)
    if match:
        candidate=Path(match[1]).parent.parent/'bin/mpiexec'
        if os.access(candidate,os.X_OK):return str(candidate)
    return shutil.which('mpiexec')


def compare_headers(strict, candidate, *, scratch):
    """Field metadata identity; MPAS mpas_io_streams.F:4315-4363.

    Rust owns NetCDF header decoding. Python compares typed metadata only.
    Global build/run provenance differs between forecasts; geometry must match.
    """
    from hexcore.restart_v842 import inspect_stream
    from hashlib import sha256
    result=dict(passed=False,field_count=0,failures=[])
    try:
        expected=inspect_stream(strict,scratch=scratch)
        actual=inspect_stream(candidate,scratch=scratch)
        for label,header in [('strict',expected),('candidate',actual)]:
            result[label+'_header_sha256']=sha256(json.dumps(header,sort_keys=True,
                separators=(',',':')).encode()).hexdigest()
        result['field_count']=len(expected['variables'])
        for key in ('dimensions','unlimited'):
            if expected[key]!=actual[key]:
                result['failures'].append(dict(contract=key,strict=expected[key],candidate=actual[key]))
        for name in sorted(expected['variables'].keys()|actual['variables'].keys()):
            a=expected['variables'].get(name);b=actual['variables'].get(name)
            if a is None or b is None:
                result['failures'].append(dict(field=name,contract='membership',
                    strict_present=a is not None,candidate_present=b is not None))
                continue
            for key in ('dtype','dimensions','shape','attrs'):
                if a[key]!=b[key]:
                    result['failures'].append(dict(field=name,contract=key,strict=a[key],candidate=b[key]))
        for name in ('on_a_sphere','sphere_radius','is_periodic','x_period','y_period'):
            a=expected['global_attrs'].get(name);b=actual['global_attrs'].get(name)
            if a!=b:result['failures'].append(dict(global_attribute=name,contract='geometry',strict=a,candidate=b))
        result['global_provenance_differences']=sorted(name for name in
            expected['global_attrs'].keys()|actual['global_attrs'].keys()
            if expected['global_attrs'].get(name)!=actual['global_attrs'].get(name))
        result['passed']=not result['failures']
    except Exception as error:
        result['failure']=str(error)
    return result


def compare_series(args, deck, label, candidate, start, first, last, dt, *, strict=None):
    """Every native stream/time/word; mpas_io_streams.F:2430-3251.

    Native resets restart/output/diagnostics initial alarms on warm starts;
    mpas_atm_core.F:663-705. Surface/UGWP still publish the initial state.
    """
    publication=json.loads((args.reference/'PUBLICATION.json').read_text())
    templates={s.attrib['name']:s.attrib['filename_template']
               for s in ET.parse(args.reference/'streams.atmosphere').getroot()
               if 'filename_template' in s.attrib}
    samples=[];seen=set();inventory={}
    for name,sha in sorted(publication['files'].items()):
        if not name.endswith('.nc'):
            continue
        stream=name.split('.')[0]
        assert stream in templates,'unexpected original output stream: '+name
        stamp=datetime.strptime(name[len(stream)+1:-3],'%Y-%m-%d_%H.%M.%S')
        if not first<=stamp<=last:
            continue
        if first>start and stamp==first and stream in ('restart','output','diagnostics'):
            continue
        original=args.reference/name
        assert digest(original)==sha,'original native authority changed: '+name
        elapsed=(stamp-start).total_seconds()
        assert elapsed/dt==int(elapsed/dt),'original output not on model step'
        step=int(elapsed/dt);seen.add((stream,step))
        authority=original if strict is None else strict/name
        sample=dict(stream=stream,strict=str(authority),hex=str(candidate/name),
                    steps=[step],times=[stamp.strftime('%Y-%m-%d_%H:%M:%S')])
        if stream=='surface':
            sample['clock_witness']=dict(filename_template=templates[stream])
        samples.append(sample)
        for side,path in [('strict',authority),('candidate',candidate/name)]:
            if path.is_file():inventory[side+'/'+name]=digest(path)
    expected={(stream,step) for stream,steps in deck['expected_samples'].items()
              for step in steps if first<=start+timedelta(seconds=step*dt)<=last
              and not (first>start and start+timedelta(seconds=step*dt)==first
                       and stream in ('restart','output','diagnostics'))}
    assert seen==expected and samples,'full original stream/time authority required'
    request=args.receipts/(label+'-request.json')
    request.write_text(json.dumps(dict(schema='mpas-stream-compare-v1',samples=samples),indent=2)+'\n')
    output=args.receipts/(label+'-words.json')
    subprocess.run([str(args.comparator),'--request',str(request),'--out',str(output)],check=True)
    rows=json.loads(output.read_text())['fields']
    bad=[r for r in rows if r['status']!='compared' or r['differing_words']]
    streams={}
    for row in rows:
        count=streams.setdefault(row['stream'],dict(field_samples=0,compared_words=0,
                         differing_words=0,incompatible_samples=0))
        count['field_samples']+=1
        if row['status']=='compared':
            count['compared_words']+=row['compared_words']
            count['differing_words']+=row['differing_words']
        else:count['incompatible_samples']+=1
    headers=[]
    for sample in samples:
        header=compare_headers(Path(sample['strict']),Path(sample['hex']),
                               scratch=args.scratch/'header-audit')
        headers.append(dict(stream=sample['stream'],step=sample['steps'][0],
                            time=sample['times'][0],**header))
        streams.setdefault(sample['stream'],dict(field_samples=0,compared_words=0,
            differing_words=0,incompatible_samples=0)).setdefault('metadata_incompatible_files',0)
        streams[sample['stream']]['metadata_incompatible_files']+=int(not header['passed'])
    metadata=args.receipts/(label+'-metadata.json')
    metadata.write_text(json.dumps(dict(schema='o1-native-stream-header-grade-v1',
        process_count=1,field_contract='Exact field membership, stored primitive types, dimensions, shapes, typed attributes; global geometry',
        samples=headers),indent=2)+'\n')
    failed_headers=[h for h in headers if not h['passed']]
    result=dict(label=label,process_count=1,samples=len(samples),field_samples=len(rows),
                compared_words=sum(r['compared_words'] for r in rows if r['status']=='compared'),
                differing_words=sum(r['differing_words'] for r in rows if r['status']=='compared'),
                incompatible_samples=sum(r['status']!='compared' for r in rows),
                passed=not bad and not failed_headers,
                metadata_incompatible_files=len(failed_headers),
                metadata_receipt_sha256=digest(metadata),
                first_metadata_failure=({k:v for k,v in failed_headers[0].items() if k!='failures'}
                    | {'first_failure':failed_headers[0]['failures'][:1]}) if failed_headers else None,
                first_difference=bad[0] if bad else None,by_stream=streams,
                receipt_sha256=digest(output))
    (args.receipts/(label+'-outputs-sha256.json')).write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result


def main(argv=None):
    """Six complete forecast arms; mpas_atm_core.F:129-150,589-705."""
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkout','deck','reference','comparator','scratch','receipts'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stream-test-binding',action='store_true',
        help='Supporting six-arm experiment with the owned output callback; not production qualification')
    parser.add_argument('--resident-momentum-binding',action='store_true',
        help='Supporting computed CUDA ru/rw handoff; requires stream-test-binding')
    parser.add_argument('--proposed-driver-binding',action='store_true',
        help='Supporting exact owner proposals in private complete module copies')
    args=parser.parse_args(argv)
    assert not args.resident_momentum_binding or args.stream_test_binding
    assert not args.proposed_driver_binding or args.stream_test_binding
    assert not (args.proposed_driver_binding and args.resident_momentum_binding)
    assert os.environ.get('MPAS_IN_GPU_SLOT')=='1','actual GPU proof admission required'
    assert os.access(args.comparator,os.X_OK),'word comparator must be executable'
    subprocess.run(['git','-C',str(args.checkout),'diff','--quiet','HEAD'],check=True)
    assert not subprocess.check_output(['git','-C',str(args.checkout),'ls-files','--others',
             '--exclude-standard','tree/src','tree/tools','tree/tests'],text=True).strip()
    tip=subprocess.check_output(['git','-C',str(args.checkout),'rev-parse','HEAD'],text=True).strip()
    args.scratch=args.scratch.resolve();args.receipts=args.receipts.resolve()
    assert not args.scratch.exists(),'forecast scratch must be fresh'
    args.scratch.mkdir(parents=True);args.receipts.mkdir(parents=True,exist_ok=True)
    deck=json.loads(args.deck.read_text())
    assert deck['process_count']==1
    for item in (*deck['inputs'].values(),*deck.get('hex_support_inputs',{}).values()):
        assert digest(item['path'])==item['sha256'],'original deck dependency hash changed'
    assert digest(args.reference/'PUBLICATION.json')==deck['native']['sha256']
    text=Path(deck['inputs']['namelist']['path']).read_text()
    configs={k:v for group in parse_fortran_namelist(text).values() for k,v in group.items()}
    start=datetime.strptime(configs['config_start_time'],'%Y-%m-%d_%H:%M:%S')
    assert configs['config_run_duration']=='1_00:00:00'
    assert not configs.get('config_do_restart',False)
    half=start+timedelta(hours=12);end=start+timedelta(hours=24);dt=configs['config_dt']
    half_text,count=re.subn(r"(config_run_duration\s*=\s*)'[^']*'",r"\1'0_12:00:00'",text)
    assert count==1
    resumed,count=re.subn(r"(config_start_time\s*=\s*)'[^']*'",
                         lambda m:m.group(1)+repr(half.strftime('%Y-%m-%d_%H:%M:%S')),half_text)
    assert count==1
    resumed+='\n&restart\nconfig_do_restart=.true.\n/\n'
    restart_name='restart.'+half.strftime('%Y-%m-%d_%H.%M.%S')+'.nc'
    saved=args.scratch/'hex-restart.nc';straight=args.scratch/'straight-output'
    native_continuation=args.scratch/'native-continuation'
    summary=dict(schema='o1-actual-hex-restart-attempt-v2',process_count=1,host=socket.gethostname(),
                 hex_tip=tip,deck_sha256=digest(args.deck),orchestration_sha256=digest(__file__),
                 publication_sha256=digest(args.reference/'PUBLICATION.json'),
                 suite=configs['config_physics_suite'],cases=[],relationships=[],passed=False,
                 production_dispatch=not args.stream_test_binding,
                 resident_momentum_binding=args.resident_momentum_binding,
                 proposed_driver_binding=args.proposed_driver_binding,
                 scope='Every native stream/time; computed hex straight and both restart ingress paths; native consuming computed hex versus native consuming native')
    if args.stream_test_binding:
        summary['scope']='Supporting output-callback binding experiment; '+summary['scope']
    if args.proposed_driver_binding:
        summary['scope']='Private complete owner-proposal experiment; '+summary['scope']
    def grade(label, directory, first, last, *, authority=None):
        """Retain a failed whole attempt too; mpas_io_streams.F:2430-3251."""
        try:return compare_series(args,deck,label,directory,start,first,last,dt,strict=authority)
        except Exception as error:return dict(label=label,passed=False,comparison_failure=str(error))
    try:
        for label in ('straight','half','resumed-hex','resumed-native'):
            run=args.scratch/label;run.mkdir()
            namelist=run/'namelist.atmosphere'
            namelist.write_text(text if label=='straight' else half_text if label=='half' else resumed)
            source=(Path(deck['inputs']['init']['path']) if label in ('straight','half') else
                    saved if label=='resumed-hex' else args.reference/restart_name)
            if not source.is_file():
                summary['cases'].append(dict(label=label,passed=False,failure='previous hex half-day did not produce restart authority'))
                shutil.rmtree(run);continue
            substitutions=dict(namelist=str(namelist),streams=deck['inputs']['streams']['path'],
                               init=str(source),out=str(run/'output'))
            command=[part.format(**substitutions) for part in deck['hex_command']]
            command=[sys.executable,'-m','hexcore.cli',*command[1:]]
            if args.stream_test_binding:
                command=[sys.executable,str(args.checkout/'tree/tests/replay_owned_stream_forecast_v842.py'),
                         '--cli-resident-momenta' if args.resident_momentum_binding else '--cli',
                         *command[3:]]
            if args.proposed_driver_binding:
                command[1]=str(args.checkout/'tree/tests/replay_proposed_stream_driver_v842.py')
            hours=24 if label=='straight' else 12
            if '--hours' in command:
                index=command.index('--hours');command[index+1]=str(hours)
            if label.startswith('resumed-'):command+=['--stream-input','restart='+str(source)]
            env=dict(os.environ,PYTHONPATH=str(args.checkout/'tree/src')+':'+
                     str(args.checkout/'tree/tools')+':'+(os.environ.get('GPUWM_ENGINE_SRC') or os.environ.get('MPAS_ENGINE') or '.'))
            (args.receipts/(label+'-execution.json')).write_text(json.dumps(dict(
                command=command,process_count=1,hex_tip=tip,source_sha256=digest(source),
                namelist_sha256=digest(namelist)),indent=2)+'\n')
            with (args.receipts/(label+'-run.log')).open('w') as log:
                result=subprocess.run(command,cwd=args.checkout/'tree',env=env,stdout=log,stderr=subprocess.STDOUT)
            proposal=run/'private-owner-proposal/proposal.json'
            if args.proposed_driver_binding and proposal.exists():
                shutil.copyfile(proposal,args.receipts/(label+'-owner-proposal.json'))
            completed=False;owned_writer=False;model=run/'output/native-physics-receipt.json'
            if model.exists():
                shutil.copyfile(model,args.receipts/(label+'-model.json'))
                progress=json.loads(model.read_text());expected_steps=int(hours*3600/dt)
                completed=(progress.get('completed',False)
                           and progress.get('steps_requested')==expected_steps
                           and progress.get('steps_completed')==expected_steps)
                # A native Fortran writer passing its own comparison does not
                # exercise this lane's port. Require actual publication through
                # NativePoolStreams, with the complete selected field authority.
                records=progress.get('streams',[])
                owned_writer=(bool(records) and all(
                    row.get('writer')=='rw_mpas_stream' for row in records)
                    and not any(progress.get('missing_fields',{}).values()))
            first=half if label.startswith('resumed-') else start
            last=half if label=='half' else end
            result_grade=grade(label,run/'output',first,last)
            result_grade.update(exit_code=result.returncode,forecast_completed=completed,
                               owned_writer_complete=owned_writer,
                               log_sha256=digest(args.receipts/(label+'-run.log')))
            result_grade['passed']=bool(result_grade['passed'] and completed
                                       and owned_writer and result.returncode==0)
            summary['cases'].append(result_grade)
            if label!='straight' and straight.exists():
                summary['relationships'].append(grade(label+'-versus-hex-straight',run/'output',first,last,authority=straight))
            if label=='half' and (run/'output'/restart_name).is_file():
                shutil.copyfile(run/'output'/restart_name,saved)
                summary['hex_restart_sha256']=digest(saved)
            if label=='straight' and (run/'output').exists():
                (run/'output').rename(straight)
            shutil.rmtree(run)
        # Native consumes both kinds of actual restart. Preserve its native
        # continuation as the reference for its computed-hex consumption arm.
        for label in ('native-resumed-native','native-resumed-hex'):
            source=args.reference/restart_name if label=='native-resumed-native' else saved
            if not source.is_file():
                summary['cases'].append(dict(label=label,passed=False,
                    failure='hex half-day did not produce a computed restart for native consumption'))
                continue
            run=args.scratch/label;run.mkdir()
            (run/'namelist.atmosphere').write_text(resumed)
            shutil.copyfile(args.reference/'streams.atmosphere',run/'streams.atmosphere')
            manifest=json.loads((args.reference/'MANIFEST.json').read_text())
            native_root=Path(deck['hex_command'][deck['hex_command'].index('--stream-list-directory')+1])
            executable=native_root/'atmosphere_model'
            assert digest(executable)==manifest['binary_sha256'],'strict native executable changed'
            for name,item in deck.get('auxiliary_inputs',{}).items():
                asset=Path(item['path'])
                assert digest(asset)==item['sha256'],'original native runtime asset changed: '+name
                (run/name).symlink_to(asset.resolve())
            for key,item in deck['inputs'].items():
                if key in ('namelist','streams'):continue
                asset=Path(item['path'])
                if asset.name in manifest['inputs']:
                    assert digest(asset)==manifest['inputs'][asset.name]
                    (run/asset.name).symlink_to(asset.resolve())
            (run/restart_name).symlink_to(source)
            (run/'restart_timestamp').write_text(half.strftime('%Y-%m-%d_%H:%M:%S')+'\n')
            mpi=native_launcher(executable)
            assert mpi,'native process-count launcher unavailable'
            command=[mpi,'-n','1',str(executable)]
            execution=dict(command=command,process_count=1,native_binary_sha256=digest(executable),
                           hex_tip=tip,source_sha256=digest(source),
                           namelist_sha256=digest(run/'namelist.atmosphere'))
            (args.receipts/(label+'-execution.json')).write_text(json.dumps(execution,indent=2)+'\n')
            with (args.receipts/(label+'-run.log')).open('w') as log:
                result=subprocess.run(command,cwd=run,stdout=log,stderr=subprocess.STDOUT)
            for log in run.glob('log.atmosphere.*'):
                shutil.copyfile(log,args.receipts/(label+'-'+log.name))
            native_grade=grade(label,run,half,end)
            native_grade.update(exit_code=result.returncode,log_sha256=digest(args.receipts/(label+'-run.log')),
                                comparison_scope='native continuation versus original native straight (diagnostic control)')
            summary['cases'].append(native_grade)
            if label=='native-resumed-native':
                native_continuation.mkdir()
                for path in run.glob('*.nc'):
                    if not path.is_symlink():shutil.move(path,native_continuation/path.name)
            elif native_continuation.exists():
                summary['relationships'].append(grade('native-consumption-hex-versus-native',run,half,end,authority=native_continuation))
            shutil.rmtree(run)
        # The independent native straight/restart control was proved separately.
        # Its physics continuity grades remain visible above, never substituted
        # for native consuming computed-hex versus actual native restart state.
        summary['passed']=(len(summary['cases'])==6
            and all(c['passed'] for c in summary['cases'][:4])
            and all(c.get('exit_code')==0 for c in summary['cases'][4:])
            and len(summary['relationships'])==4
            and all(c['passed'] for c in summary['relationships']))
    except Exception as error:
        summary['failure']=str(error)
    finally:
        # Preserve required Rust-rendered whole-forecast evidence before retiring
        # the computed output. MPAS mpas_atm_core.F:675-705.
        sheet_out=args.receipts.parent/'sheets'/args.receipts.name
        try:
            renderer=shutil.which('mpas-sheets')
            assert renderer,'required Rust difference-sheet launcher unavailable'
            command=[renderer,'--deck',str(args.deck),'--hex-dir',str(straight),
                     '--native-publication',str(args.reference/'PUBLICATION.json'),
                     '--out',str(sheet_out),'--leads','6,12,24',
                     '--label','o1-streams-restart '+summary['suite']+
                         (' supporting owned writer' if args.stream_test_binding else ' production')]
            env=dict(os.environ,TMPDIR=str(args.scratch/'sheets-scratch'))
            with (args.receipts/'sheets-run.log').open('w') as log:
                rendered=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            assert rendered.returncode==0,'Rust difference-sheet generation failed'
            manifest=sheet_out/'sheets.json'
            sheets=json.loads(manifest.read_text())
            assert [row['lead'] for row in sheets['leads']]==[6,12,24]
            products={'2m_temperature':'2m-temperature',
                'total_qpf':'precip-total-since-start',
                'mslp_10m_winds':'mslp-isobars-10m-wind',
                'var_mslp':'mslp',
                '500mb_height_winds':'500hPa-height-contours-wind',
                'var_geopotential_height_500hpa':'500hPa-height',
                '10m_wind_speed_and_direction':'10m-wind',
                'composite_reflectivity':'composite-reflectivity'}
            for row in sheets['leads']:
                assert not row['failures'],'failed difference sheets'
                rendered_maps=[panel['map'] for panel in row['panels']]
                absent_native=[]
                for product,reason in row['skips']:
                    assert reason.startswith('run B: ') and ': missing stored selector(s): ' in reason,\
                        'difference sheet skipped without native field absence: '+reason
                    absent_native.append(products[product.replace(':','_')])
                accounted=rendered_maps+absent_native
                assert len(accounted)==len(set(accounted))==8 and set(accounted)==set(products.values()),\
                    'incomplete difference sheet inventory'
                for panel in row['panels']:
                    png=sheet_out/panel['file']
                    assert png.is_file() and png.stat().st_size<10_000_000,'missing or oversized sheet'
            summary['sheets']=dict(produced=True,manifest=str(manifest),
                manifest_sha256=digest(manifest),command=command,process_count=1,
                hex_tip=tip,production_dispatch=not args.stream_test_binding,
                visual_review_required=True,visual_review_completed=False)
        except Exception as error:
            summary['sheets']=dict(produced=False,failure=str(error),process_count=1)
        summary['passed']=bool(summary['passed'] and summary['sheets']['produced'])
        (args.receipts/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        shutil.rmtree(args.scratch)
    print(json.dumps(summary),flush=True)
    return 0 if summary['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
