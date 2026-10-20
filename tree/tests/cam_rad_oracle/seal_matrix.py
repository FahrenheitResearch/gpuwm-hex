"""Seal complete CAM matrix receipts while counting every Rust-compared word.

Incomplete prefixes and mismatched native/artifact pins are rejected. A sealed
failure stays a failure; this supporting evidence never proves a forecast.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(fixtures, results, output):
    native = json.loads(fixtures.read_text())
    summary = json.loads((results/'summary.json').read_text())
    if (len(native) != 14 or sum(r['expected_records'] for r in native) != 1760
            or not summary.get('complete') or len(summary['fixtures']) != 14):
        raise ValueError('The complete 1760-call matrix is required')
    rows = []
    for index, (expected, actual) in enumerate(zip(native, summary['fixtures'])):
        if (actual['records'] != expected['expected_records']
                or Path(actual['fixture']).resolve() != Path(expected['directory']).resolve()
                or actual['result'] not in ('PASS', 'FAIL')):
            raise ValueError('Fixture completion/order differs')
        stem = f"{index:02d}-{Path(actual['fixture']).name}-boundary"
        boundary_path = results/(stem+'.json')
        state_path = results/(stem+'-state.json')
        boundary = json.loads(boundary_path.read_text())
        state = json.loads(state_path.read_text())
        if (not boundary.get('complete') or boundary['excluded_words'] != 0
                or state['excluded_words'] != 0 or boundary['process_count'] != 1
                or state['process_count'] != 1
                or boundary['reference_sha256'] != expected['files']['cam-after.bin']['sha256']
                or state['native_sha256'] != expected['files']['cam-state-after.bin']['sha256']
                or boundary['cuda_source_sha256'] != summary['cuda_source_sha256']
                or boundary['transport_library_sha256'] != summary['transport_library_sha256']
                or state['transport_library_sha256'] != summary['transport_library_sha256']):
            raise ValueError('Full-word comparison identity differs')
        rows.append(dict(label=expected['label'], calls=actual['records'],
            boundary_words=boundary['words'], state_words=state['words'],
            boundary_differing_words=boundary['total_differing_words'],
            state_differing_words=state['differing_words'],
            boundary_receipt_sha256=sha(boundary_path), state_receipt_sha256=sha(state_path)))
    complete = dict(scope='complete whole CAM CUDA matrix; supporting evidence only',
        process_count=1, excluded_words=0, complete=True,
        run_host=summary.get('run_host'),
        whole_model_acceptance=False, fixtures=rows,
        calls=sum(r['calls'] for r in rows),
        boundary_words=sum(r['boundary_words'] for r in rows),
        state_words=sum(r['state_words'] for r in rows),
        differing_words=sum(r['boundary_differing_words']+r['state_differing_words'] for r in rows),
        fixtures_sha256=sha(fixtures), replay_summary_sha256=sha(results/'summary.json'),
        cuda_source_sha256=summary['cuda_source_sha256'],
        artifact_sha256=summary['artifact_sha256'], module_sha256=summary['module_sha256'],
        transport_library_sha256=summary['transport_library_sha256'])
    complete['passed'] = complete['differing_words'] == 0
    if complete['passed'] != summary['passed']:
        raise ValueError('Replay pass status differs from total word comparison')
    output.write_text(json.dumps(complete, indent=2)+'\n')
    return int(not complete['passed'])


def seal_observed(fixtures, observations, results, output, module, artifact):
    """Bind all original calls with PRE at module_ra_cam.F:364-365.

    This qualification counts every output/state word and remains distinct
    from the original unseeded failures and whole-model acceptance.
    """
    expected=json.loads(fixtures.read_text())
    native=json.loads((observations/'summary.json').read_text())
    cuda=json.loads((results/'summary.json').read_text())
    if (len(expected)!=14 or sum(r['expected_records'] for r in expected)!=1760
            or not native['complete'] or not native['original_native_outputs_unchanged']
            or not cuda['complete'] or cuda['records']!=1760
            or len(native['fixtures'])!=14 or len(cuda['fixtures'])!=14
            or cuda['observation_summary_sha256']!=sha(observations/'summary.json')):
        raise ValueError('Complete unchanged original 1760-call observation/replay required')
    artifact_record=json.loads(artifact.with_suffix('.json').read_text())
    if sha(artifact)!=artifact_record['artifact_sha256']:
        raise ValueError('Fixed CUDA artifact bytes changed')
    copies=output.parent/(output.stem+'-comparisons');copies.mkdir(exist_ok=False)
    rows=[];receipt_hashes={};native_words=0
    for plan,observed,replayed in zip(expected,native['fixtures'],cuda['fixtures']):
        label=plan['label'];n=observations/label;c=results/label
        if (observed['label']!=label or replayed['label']!=label
                or not observed['original_native_outputs_unchanged'] or not replayed['complete']
                or observed['native_executable_sha256']!=plan['native_executable_sha256']
                or replayed['records']!=plan['expected_records']):
            raise ValueError('Original fixture completion/order/binary differs')
        for name,pin in plan['files'].items():
            if sha(n/name)!=pin['sha256'] or (n/name).stat().st_size!=pin['bytes']:
                raise ValueError('Observed native capture changed: '+label+'/'+name)
        for name,pin in observed['entry_files'].items():
            if sha(n/('entry-'+name+'.bin'))!=pin:
                raise ValueError('Observed native PRE changed: '+label+'/'+name)
        boundary=json.loads((c/'boundary.json').read_text())
        state=json.loads((c/'boundary-state.json').read_text())
        if (not boundary['complete'] or len(boundary['records'])!=plan['expected_records']
                or boundary['process_count']!=1 or state['process_count']!=1
                or boundary['excluded_words']!=0 or state['excluded_words']!=0
                or boundary['reference_sha256']!=plan['files']['cam-after.bin']['sha256']
                or state['native_sha256']!=plan['files']['cam-state-after.bin']['sha256']
                or boundary['observed_entry_sha256']!=observed['entry_files']
                or boundary['cuda_source_sha256']!=artifact_record['source_sha256']
                or boundary['original_unseeded_failure_superseded']
                or not boundary['observed_native_automatic_pre_test_only']):
            raise ValueError('Complete PRE replay identity differs: '+label)
        for name in plan['files']:
            p=n/(name+'.comparison.json');comparison=json.loads(p.read_text())
            if comparison.get('total_differing_words',comparison.get('differing_words'))!=0:
                raise ValueError('Tracing changed original native outputs: '+label)
            native_words+=comparison['words']
            copied=copies/(label+'-'+name+'.json');shutil.copy2(p,copied)
            receipt_hashes[copied.name]=sha(copied)
        for name in ('boundary.json','boundary-state.json'):
            copied=copies/(label+'-'+name);shutil.copy2(c/name,copied)
            receipt_hashes[copied.name]=sha(copied)
        rows.append(dict(label=label,calls=replayed['records'],
            boundary_words=boundary['words'],state_words=state['words'],
            boundary_differing_words=boundary['total_differing_words'],
            state_differing_words=state['differing_words'],
            native_executable_sha256=observed['native_executable_sha256'],
            observed_entry_sha256=observed['entry_files']))
    differences=sum(r['boundary_differing_words']+r['state_differing_words'] for r in rows)
    words=sum(r['boundary_words']+r['state_words'] for r in rows)
    if words!=cuda['words'] or differences!=cuda['differing_words']:
        raise ValueError('Full-word replay totals differ')
    sealed=dict(scope='complete unchanged-executable CAM PRE observation and CUDA replay',
        complete=True,calls=1760,process_count=1,excluded_words=0,
        native_compared_words=native_words,native_differing_words=0,
        words=words,differing_words=differences,passed=differences==0,fixtures=rows,
        whole_model_acceptance=False,observed_automatic_pre_test_only=True,
        original_unseeded_failures_superseded=False,
        fixtures_sha256=sha(fixtures),observation_summary_sha256=sha(observations/'summary.json'),
        replay_summary_sha256=sha(results/'summary.json'),module_sha256=sha(module),
        artifact_sha256=sha(artifact),artifact_record_sha256=sha(artifact.with_suffix('.json')),
        cuda_source_sha256=artifact_record['source_sha256'],receipt_sha256=receipt_hashes)
    output.write_text(json.dumps(sealed,indent=2)+'\n')
    return int(not sealed['passed'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fixtures', 'results', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('observations','module','artifact'):
        parser.add_argument('--'+name,type=Path)
    args = parser.parse_args()
    if args.observations and (not args.module or not args.artifact):
        parser.error('Observed matrix sealing requires module and fixed artifact')
    raise SystemExit(seal_observed(args.fixtures,args.observations,args.results,args.output,
                                  args.module,args.artifact) if args.observations
                     else seal(args.fixtures, args.results, args.output))
