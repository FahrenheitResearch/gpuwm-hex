"""Complete original native failure attempts, preserving inputs and signals.

CPU-only supporting witness; neither a Hex forecast nor an exclusion approval.
The supplied tracer observes only its own child. This runner does not decode
weather data, repair packages, alter a namelist or replace the native binary.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from hexcore.verify_mpas import digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--tables', type=Path, required=True)
    p.add_argument('--table-manifest', type=Path, required=True)
    p.add_argument('--partition-root', type=Path, required=True)
    p.add_argument('--ranks', nargs='+', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    a = p.parse_args()
    binding_path = a.original / 'binding.json'
    binding = json.loads(binding_path.read_text())
    native = (a.original / 'atmosphere_model').resolve()
    assert digest(native) == binding['binary_sha256']
    partition_binding = json.loads((a.partition_root / 'partition-binding.json').read_text())
    partition = a.partition_root / 'x1.10242.CONUS.graph.info.part.2'
    assert digest(partition) == partition_binding['partition_sha256']
    assert set(a.ranks) <= {1, 2} and a.ranks and len(set(a.ranks)) == len(a.ranks)
    tables = {p.name: digest(p) for p in a.tables.iterdir() if p.is_file()}
    expected_tables = json.loads(a.table_manifest.read_text())['table_sha256']
    assert tables == expected_tables, 'complete original native table inventory differs'
    tracer = (a.partition_root / 'fault_trace.py').resolve()
    launcher = (a.partition_root / 'mpi-bin/mpiexec.hydra').resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    a.work.mkdir(parents=True, exist_ok=False)
    # Avoid large signal-generated core files. The read-only tracer preserves
    # the native handler's actual signal; this merely sets owned process limits.
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    results = []
    for ranks in a.ranks:
        for repeat in (1, 2):
            name = f'ranks-{ranks}-repeat-{repeat}'
            work = a.work / name
            work.mkdir()
            for filename, expected in binding['deck_inputs'].items():
                source = a.original / 'deck' / filename
                assert digest(source) == expected
                target = work / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            for row in binding['inputs']:
                source = (a.original / row['filename']).resolve()
                assert digest(source) == row['sha256']
                (work / row['filename']).symlink_to(source)
            for filename, expected in tables.items():
                source = (a.tables / filename).resolve()
                assert digest(source) == expected
                (work / filename).symlink_to(source)
            if ranks == 2:
                (work / partition.name).symlink_to(partition.resolve())
            trace_out = a.out.resolve() / (name + '-trace.json')
            command = [str(launcher), '-launcher', 'fork', '-np', str(ranks),
                       sys.executable, str(tracer), '--binary', str(native),
                       '--out', str(trace_out)]
            with (a.out / (name + '.stdout')).open('w') as log:
                run = subprocess.run(command, cwd=work, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=600,
                                     env={**os.environ, 'OMP_NUM_THREADS': '1'})
            for log in work.glob('log.*'):
                shutil.copyfile(log, a.out / (name + '-' + log.name))
            traces = {p.name: json.loads(p.read_text())
                      for p in a.out.glob(name + '-trace-rank*.json')}
            emitted = {p.name: digest(p) for p in work.glob('*.nc') if not p.is_symlink()}
            unchanged = all(digest(a.original / row['filename']) == row['sha256']
                            for row in binding['inputs'])
            unchanged = unchanged and all(digest(a.tables / n) == h for n, h in tables.items())
            unchanged = unchanged and all(digest(work / n) == h
                                           for n, h in binding['deck_inputs'].items())
            assert unchanged
            result = {'process_count': ranks, 'repeat': repeat,
                      'host': os.uname().nodename, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
                      'hours_requested': 24, 'command': command, 'exit_code': run.returncode,
                      'rank_fault_traces': traces, 'weather_outputs': emitted,
                      'native_and_staged_inputs_unchanged': unchanged}
            results.append(result)
            write_json(a.out / 'summary.json', {
                'schema': 'rrfs-native-fatal-crosshost-attempt-v1', 'binding': binding,
                'binding_sha256': digest(binding_path), 'native_binary_sha256': digest(native),
                'table_sha256': tables, 'partition_binding': partition_binding,
                'tracer_sha256': digest(tracer), 'launcher_sha256': digest(launcher),
                'proxy_sha256': digest(launcher.parent / 'hydra_pmi_proxy'),
                'runner_sha256': digest(Path(__file__)), 'results': results,
                'exclusions': [], 'hex_forecast_identity': False,
                'bounds_checked_witness': False, 'policy_exclusion_approved': False})
            print(name, 'exit', run.returncode,
                  [(t.get('rank'), [(f['binary_offset'], f['registers']['rdx'])
                                   for f in t['faults']]) for t in traces.values()], flush=True)


if __name__ == '__main__':
    main()
