"""Measure strict executable repeats; never authorize a native-word exclusion.

The unchanged published CI science deck is deliberately outside the required
small-mesh/date acceptance matrix. Only its output field list gains tsnav and
snow state. Native/Rust own all field decoding and comparison.
"""
import argparse
import hashlib
import json
import shutil
import socket
import subprocess
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--input-receipt', type=Path, required=True)
    parser.add_argument('--comparator', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--partition', type=Path, required=True,
                        help='Explicit fixed two-rank cell partition fixture; no mesh decoding in Python')
    parser.add_argument('--ranks',nargs='+',type=int,default=[1,2])
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--keep-outputs',action='store_true',help='Retain owned outputs until cross-host and checked-word comparisons are sealed')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    provenance = json.loads(args.input_receipt.read_text())
    executable = Path(provenance['argv'][-1])
    if sha(executable) != provenance['native_binary_sha256']:
        raise ValueError('Published strict executable hash differs')
    for member in provenance['inputs'].values():
        if sha(Path(member['source'])) != member['sha256']:
            raise ValueError('Published native input hash differs: ' + member['source'])
    result = dict(schema='ruc-strict-executable-repeat-v1', host=socket.gethostname(),
                  scope='original 1h GSL CI science deck, additional observed snow fields; not lane acceptance or a complete defect-policy witness',
                  source_receipt_sha256=sha(args.input_receipt),
                  native_binary_sha256=sha(executable), comparator_sha256=sha(args.comparator),
                  inputs=provenance['inputs'], runs=[], comparisons=[],
                  partition_sha256=sha(args.partition),
                  partition_scope='explicit two-rank test configuration; native mesh/topology unchanged; not a published forecast decomposition',
                  native_defect_policy_complete=False, exclusions=[])
    output_names = None
    run_directories = []
    for ranks in args.ranks:
        for repeat in range(args.repeats):
            work = args.root / f'rank{ranks}-repeat{repeat}'
            work.mkdir()
            if ranks == 2:
                (work/'graph.info.part.2').symlink_to(args.partition.resolve())
            for name, member in provenance['inputs'].items():
                source = Path(member['source'])
                if name == 'stream_list.atmosphere.diagnostics':
                    original = source.read_text().splitlines()
                    additions = ('tsnav', 'soilt1', 'snow', 'snowh', 'snowc')
                    (work / name).write_text('\n'.join(original + [f for f in additions if f not in original]) + '\n')
                else:
                    (work / name).symlink_to(source)
            command = ['taskset', '-c', '8-23', provenance['argv'][0], '-np', str(ranks), str(executable)]
            with (work / 'stdout.log').open('w') as stream:
                run = subprocess.run(command, cwd=work, stdout=stream, stderr=subprocess.STDOUT)
            logs = {p.name: sha(p) for p in work.glob('log.atmosphere.*')}
            outputs = sorted(p.name for p in work.glob('*.nc') if not p.is_symlink())
            result['runs'].append(dict(process_count=ranks, repeat=repeat, command=command,
                                       exit=run.returncode, logs=logs, outputs={p: sha(work/p) for p in outputs},
                                       observation_list_sha256=sha(work/'stream_list.atmosphere.diagnostics')))
            if run.returncode == 0 and outputs:
                run_directories.append((ranks, repeat, work))
                output_names = outputs if output_names is None else output_names
            else:
                result['execution_complete'] = False
    # Compare every stored field, keeping differing words and missing outputs
    # separate. Rust checks stored types and actual xtime against the request.
    if len(run_directories) >= 2:
        pairs = [(run_directories[0], other) for other in run_directories[1:]]
        if len(run_directories) == 4:
            pairs.append((run_directories[2], run_directories[3]))
        for (base_ranks, base_repeat, baseline), (ranks, repeat, work) in pairs:
            samples = []
            missing = []
            for step, name in enumerate(output_names):
                if not (work/name).exists():
                    missing.append(name)
                    continue
                prefix, clock = name.split('.', 1)
                clock = clock.removesuffix('.nc')
                date, time = clock.split('_')
                samples.append(dict(stream=prefix, steps=[step], times=[date+'_'+time.replace('.', ':')],
                                    strict=str(baseline/name), hex=str(work/name)))
            tag = f'-base-rank{base_ranks}-repeat{base_repeat}-rank{ranks}-repeat{repeat}'
            request = args.root / ('compare'+tag+'.json')
            compared = args.receipt.with_name(args.receipt.stem + tag+'.json')
            request.write_text(json.dumps(dict(schema='mpas-stream-compare-v1', samples=samples), indent=2)+'\n')
            comparison = subprocess.run([str(args.comparator), '--request', str(request), '--out', str(compared)],
                                        capture_output=True, text=True)
            item = dict(baseline_process_count=base_ranks, baseline_repeat=base_repeat,
                        process_count=ranks, repeat=repeat, comparator_exit=comparison.returncode,
                        missing_outputs=missing, receipt=str(compared), stderr=comparison.stderr)
            if compared.exists():
                fields = json.loads(compared.read_text())['fields']
                item.update(differing_words=sum(f['differing_words'] for f in fields if f['status']=='compared'),
                            missing_or_incompatible_samples=sum(f['status']!='compared' for f in fields),
                            tsnav=[f for f in fields if f['field']=='tsnav'])
            result['comparisons'].append(item)
        result['execution_complete'] = len(run_directories) == len(args.ranks)*args.repeats
    args.receipt.write_text(json.dumps(result, indent=2)+'\n')
    # Preserve logs as small receipts before retiring only our own run payloads.
    for work in args.root.glob('rank*-repeat*'):
        for log in work.glob('log.atmosphere.*'):
            shutil.copyfile(log, args.receipt.with_name(args.receipt.stem+'-'+work.name+'-'+log.name))
        for payload in work.glob('*.nc'):
            if not args.keep_outputs and not payload.is_symlink():
                payload.unlink()
    print(json.dumps({k: result[k] for k in ('host', 'execution_complete', 'comparisons')}, indent=2))
    raise SystemExit(not result.get('execution_complete', False))


if __name__ == '__main__':
    main()
