"""Combine exact gcov edge counts from whole-module replay matrices."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def read(path):
    branches, lines = {}, {}
    line, source = 0, ''
    for row in path.read_text().splitlines():
        match = re.match(r'\s*([^:]+):\s*(\d+):(.*)', row)
        if match:
            line, source = int(match[2]), match[3].strip()
            count = match[1].strip().removesuffix('*')
            if line and count != '-':
                lines[line] = count.isdigit() and int(count)>0
        elif row.startswith('branch '):
            ordinal = int(row.split()[1])
            count = re.search(r'\btaken\s+(\d+)(%)?', row)
            if count and count[2]:
                raise ValueError('gcov -c absolute counts are required')
            branches[line, ordinal] = dict(line=line, branch=ordinal,
                                           executed='never executed' not in row,
                                           count=int(count[1]) if count else 0,
                                           source=source)
    return branches, lines


def main(args):
    files = {}
    gaps = []
    prior = json.loads(args.disposition.read_text())['rows']
    dispositions = {(r['file'], r['line'], r['branch']): r['disposition'] for r in prior}
    for name in ('cu_ntiedtke.F90.gcov', 'cu_ntiedtke_pre.F.gcov',
                 'cu_ntiedtke_post.F.gcov', 'module_cu_ntiedtke.F.gcov'):
        paths = [directory/name for directory in args.gcov]
        combined, executed = {}, {}
        keyset = None
        for path in paths:
            branches, lines = read(path)
            if keyset is not None and keyset != set(branches):
                raise ValueError(f'{path}: native edge layout differs')
            keyset = set(branches)
            for key, row in branches.items():
                if key not in combined:
                    combined[key] = dict(row)
                else:
                    combined[key]['count'] += row['count']
                    combined[key]['executed'] |= row['executed']
            for line, reached in lines.items():
                executed[line] = executed.get(line, False) or reached
        for key, row in sorted(combined.items()):
            if row['count'] == 0:
                gaps.append(dict(file=name, **row, disposition=dispositions.get(
                    (name, *key), 'Untaken in these complete matrices; no unreachable claim.')))
        files[name] = dict(lines=len(executed), lines_executed=sum(executed.values()),
                           branches=len(combined),
                           executed=sum(r['executed'] for r in combined.values()),
                           taken_at_least_once=sum(r['count']>0 for r in combined.values()),
                           inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                   for p in paths])
    result = dict(schema='mpas-ntiedtke-native-coverage-union/v1',
                  diagnostic_branch_inventory=True, every_branch_covered=not gaps,
                  native_process_count=1, files=files, uncovered=gaps)
    args.receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gcov', type=Path, action='append', required=True)
    parser.add_argument('--disposition', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    main(parser.parse_args())
