#!/usr/bin/env python3
"""Capture a short pristine NSSL run with r1's external GNU ABI observer.

The observer and ordinary execution use the same immutable binary and inputs.
All emitted NetCDF words are compared by the existing Rust comparator. Python
only stages authenticated files, edits the run duration and schedules processes.
This is capture/passivity evidence, never CUDA replay or forecast acceptance.
Fork boundary: src/core_atmosphere/physics/physics_wrf/module_mp_nssl_2mom.F:
1995-3031 at 1d6e370e7b43a332074809f170532a216c2212fb.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def execute(a):
    publication = json.loads((a.publication/'PUBLICATION.json').read_text())
    manifest = json.loads((a.publication/'MANIFEST.json').read_text())
    assert sha(a.publication/'MANIFEST.json') == publication['files']['MANIFEST.json']
    binary = a.build/'atmosphere_model'
    assert sha(binary) == manifest['binary_sha256']
    assert manifest['recipe'] == 'gnu-strict' and manifest['ranks'] == 1
    assert manifest['resolved_configuration']['resolved_settings']['config_microp_scheme'] == 'mp_nssl2m'
    spec = json.loads(a.observer_spec.read_text())
    assert spec['callee']['name'] == 'nssl_2mom_driver'
    for name, digest in spec['generated_sha256'].items():
        assert sha(a.observer_spec.parent/name) == digest
    inputs = {}
    # Explicit roots can include the published case products and pinned tables.
    # Match both basename and content; never take an arbitrary same-name table.
    needed = manifest['inputs']
    for root in a.input_root:
        for path in root.rglob('*'):
            if path.name in needed and path.is_file() and sha(path) == needed[path.name]:
                inputs[path.name] = path.resolve()
    assert set(inputs) == set(needed), 'Missing authenticated inputs: '+str(sorted(set(needed)-set(inputs)))
    a.out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    runs = []
    for label in ('ordinary', 'observed'):
        run = a.out/label
        run.mkdir()
        for name, digest in manifest['resolved_configuration']['files'].items():
            assert sha(a.publication/name) == digest == publication['files'][name]
            target = run/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(a.publication/name, target)
            # Publications are immutable. Their private run copies must allow
            # duration/output edits without changing the source permissions.
            target.chmod(0o600)
        namelist = run/'namelist.atmosphere'
        text, changed = re.subn(r"(config_run_duration\s*=\s*)'[^']*'", r"\g<1>'"+a.duration+"'", namelist.read_text())
        assert changed == 1
        namelist.write_text(text)
        streams = run/'streams.atmosphere'
        xml = ET.parse(streams)
        for stream in xml.getroot():
            if 'output' in stream.get('type', '').split(';') and stream.get('output_interval'):
                stream.set('output_interval', a.duration)
        xml.write(streams)
        for name, source in inputs.items():
            # All case ingress is private: native DA streams may write to it.
            if name in manifest['case']['files']:
                shutil.copyfile(source, run/name)
                (run/name).chmod(0o600)
            else:
                (run/name).symlink_to(source)
        command = [str(binary.resolve())]
        if label == 'observed':
            command = [str(a.observer.resolve()), *command, str((a.out/'captures').resolve())]
        with (a.out/(label+'.log')).open('w') as log:
            result = subprocess.run(command, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = dict(label=label, argv=command, exit=result.returncode,
                   deck_sha256={p.name:sha(p) for p in (namelist,streams)},
                   emitted_netcdf={p.name:sha(p) for p in run.glob('*.nc') if not p.is_symlink()},
                   log_sha256=sha(a.out/(label+'.log')))
        runs.append(row)
        write(a.out/'runs.json', runs)
        if result.returncode:
            raise RuntimeError(label+' native execution failed: '+str(result.returncode))
    compare_path = Path(__file__).parents[2]/'tools/native_physics_reference/compare_runs.py'
    module_spec = importlib.util.spec_from_file_location('native_stream_compare', compare_path)
    compare = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(compare)
    comparison = compare.compare(a.out/'ordinary', a.out/'observed', a.comparator, '*.nc')
    write(a.out/'passivity.json', comparison)
    calls = [json.loads(line) for line in (a.out/'captures/rank0-calls.jsonl').read_text().splitlines()]
    pre, post = ([c for c in calls if c['phase']==phase] for phase in ('pre','post'))
    complete = bool(pre) and [(c['call'],c['step']) for c in pre] == [(c['call'],c['step']) for c in post]
    members = {p.name:sha(p) for p in sorted((a.out/'captures').iterdir()) if p.is_file()}
    write(a.out/'capture-members.json', members)
    unchanged = {name:sha(path)==needed[name] for name,path in inputs.items()}
    result = dict(schema='r16-nssl2m.pristine-real-capture/v1', host=socket.gethostname(),
        process_count=1, duration=a.duration, observed_calls=len(pre),
        captured_argument_arrays=sum(name.endswith('.bin') for name in members),
        complete_entry_return_pairs=complete,
        passivity_all_output_words_equal=comparison['all_identical'],
        source_inputs_unchanged=unchanged, capture_member_manifest_sha256=sha(a.out/'capture-members.json'),
        publication=str(a.publication.resolve()), publication_sha256=sha(a.publication/'PUBLICATION.json'),
        binary_sha256=sha(binary), observer_sha256=sha(a.observer),
        observer_spec_sha256=sha(a.observer_spec), runner_sha256=sha(Path(__file__)),
        inputs={name:dict(path=str(path),sha256=needed[name]) for name,path in inputs.items()},
        runs=runs, forecast_acceptance=False, cuda_replay_proven=False,
        passed=complete and comparison['all_identical'] and all(unchanged.values()))
    write(a.out/'summary.json', result)
    print(json.dumps({k:result[k] for k in ('passed','observed_calls','captured_argument_arrays','passivity_all_output_words_equal')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('publication','build','observer','observer-spec','comparator','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--input-root',type=Path,action='append',required=True)
    parser.add_argument('--duration',required=True)
    raise SystemExit(execute(parser.parse_args()))
