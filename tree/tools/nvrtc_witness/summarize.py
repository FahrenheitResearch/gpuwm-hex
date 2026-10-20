"""Reconcile untimed CuPy NVRTC events against strace -ff execve receipts."""
import argparse
import json
from pathlib import Path
import re


def summarize(root, exec_prefix=None, expected_pids=()):
    root = Path(root)
    issues = []
    sessions = {}
    for path in sorted(root.glob('[0-9]*.jsonl')):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                issues.append('invalid JSON: ' + str(path))
                continue
            sessions.setdefault((row['pid'], row['session']), []).append(row)
    expected = set(map(int, expected_pids))
    exec_files = []
    if exec_prefix:
        prefix = Path(exec_prefix)
        for path in sorted(prefix.parent.glob(prefix.name + '.*')):
            if not path.name.rsplit('.', 1)[-1].isdigit():
                continue
            exec_files.append(str(path))
            for line in path.read_text().splitlines():
                if not re.search(r'= 0\s*$', line):
                    continue
                match = re.search(r'execve\("([^"]+)"', line)
                if match and re.fullmatch(r'python(?:\d+(?:\.\d+)*)?', Path(match[1]).name):
                    expected.add(int(path.name.rsplit('.', 1)[-1]))
        if not exec_files:
            issues.append('no exec census files')
    elif not expected:
        issues.append('no independent expected process census')
    seen = {key[0] for key in sessions}
    if not sessions:
        issues.append('no instrumented sessions')
    if not expected:
        issues.append('no expected Python process')
    if expected - seen:
        issues.append('missing Python PIDs: ' + repr(sorted(expected - seen)))
    processes = []
    for (pid, session), rows in sorted(sessions.items()):
        ready = [r for r in rows if r['event'] == 'ready']
        final = [r for r in rows if r['event'] == 'finished']
        starts = [r for r in rows if r['event'] == 'compile_start']
        ends = [r for r in rows if r['event'] == 'compile_end']
        if len(ready) != 1 or len(final) != 1:
            issues.append(f'{pid}/{session}: missing or duplicate ready/finished')
        if any(r['event'] == 'hook_error' for r in rows):
            issues.append(f'{pid}/{session}: hook import failed')
        if any(not r.get('hook_intact', False) for r in ready + final):
            issues.append(f'{pid}/{session}: hook replaced')
        if sorted(r['call'] for r in starts) != sorted(r['call'] for r in ends):
            issues.append(f'{pid}/{session}: unfinished compiler call')
        if final and final[0]['calls'] != len(starts):
            issues.append(f'{pid}/{session}: compiler count mismatch')
        processes.append(dict(pid=pid, session=session, calls=len(starts),
            failed_calls=sum(not r['ok'] for r in ends),
            empty_source_calls=sum(r.get('source_bytes') == 0 for r in starts)))
    total = sum(p['calls'] for p in processes)
    return dict(schema='hex.nvrtc-witness/v1', coverage_valid=not issues,
        zero_nvrtc_observed=not issues and total == 0, nvrtc_calls=total,
        expected_python_pids=sorted(expected), processes=processes, issues=issues,
        exec_census_files=exec_files,
        scope='CuPy low-level nvrtc.compileProgram across inherited Python processes; untimed. Native code calling NVRTC outside CuPy is not covered.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--exec-prefix')
    parser.add_argument('--expected-pid', action='append', type=int, default=[])
    parser.add_argument('--require-zero', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.exec_prefix, args.expected_pid)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['coverage_valid'] and (not args.require_zero or result['zero_nvrtc_observed']) else 1)
