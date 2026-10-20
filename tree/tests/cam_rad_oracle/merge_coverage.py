"""Union native gcov evidence only when source and branch graphs agree."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json


def merge(reference, directories, receipt):
    graphs = {}
    inputs = []
    pins = {}
    for directory in directories:
        for path in sorted(directory.glob('*.gcov.json.gz')):
            inputs.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            data = json.loads(gzip.decompress(path.read_bytes()))
            for file in data['files']:
                name = Path(file['file']).name
                sources = list((reference / 'src').rglob(name))
                if len(sources) != 1 or not (directory / name).exists():
                    raise ValueError(f'cannot establish pristine source {name}')
                raw = sources[0].read_bytes()
                if raw != (directory / name).read_bytes():
                    raise ValueError(f'changed native numeric source {name}')
                pins[name] = hashlib.sha256(raw).hexdigest()
                lines = {line['line_number']: line for line in file['lines']}
                functions = {f['demangled_name']: f['execution_count'] for f in file['functions']}
                if name not in graphs:
                    graphs[name] = (lines, functions)
                    continue
                previous, counts = graphs[name]
                if previous.keys() != lines.keys():
                    raise ValueError(f'executable line graph changed for {name}')
                for n, line in lines.items():
                    existing = previous[n]
                    a, b = existing.get('branches', []), line.get('branches', [])
                    signature = lambda items: [(v['fallthrough'], v['throw']) for v in items]
                    if signature(a) != signature(b):
                        raise ValueError(f'branch graph changed for {name}:{n}')
                    existing['count'] += line['count']
                    for x, y in zip(a, b, strict=True):
                        x['count'] += y['count']
                if counts.keys() != functions.keys():
                    raise ValueError(f'function graph changed for {name}')
                for name_, count in functions.items():
                    counts[name_] += count
    coverage = []
    for name, (lines, functions) in sorted(graphs.items()):
        branches = [(n, i, b) for n, line in lines.items()
                    for i, b in enumerate(line.get('branches', []))]
        coverage.append({'source': name, 'executable_lines': len(lines),
                         'executed_lines': sum(l['count'] > 0 for l in lines.values()),
                         'branches': len(branches),
                         'taken_branches': sum(b['count'] > 0 for _, _, b in branches),
                         'unreached_branches': [dict(line=n, branch=i, fallthrough=b['fallthrough'], throw=b['throw'])
                                               for n, i, b in branches if b['count'] == 0],
                         'uncalled_routines': [n for n, c in functions.items() if c == 0]})
    receipt.write_text(json.dumps(coverage, indent=2) + '\n')
    receipt.with_name(receipt.stem + '-inputs.json').write_text(json.dumps({
        'scope': 'union of pristine native synthetic scheme and supplemental helper coverage; no CUDA identity claim',
        'sources': pins, 'inputs': inputs,
        'coverage_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest()}, indent=2) + '\n')
    print(json.dumps([{k: c[k] for k in ('source', 'branches', 'taken_branches')} for c in coverage]))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--directory', type=Path, action='append', required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    merge(a.reference, a.directory, a.receipt)
