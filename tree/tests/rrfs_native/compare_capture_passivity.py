"""Compare every stored native word with Rust; observer passivity is required."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--comparator', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scope', required=True)
    parser.add_argument('--authority', type=Path,
                        help='input manifest binding a separately labelled source authority')
    args = parser.parse_args()
    comparisons = []
    passed = True
    for baseline, other in [('strict', 'disabled'), ('strict', 'active'), ('disabled', 'active')]:
        left, right = args.runs/baseline, args.runs/other
        names = sorted(p.name for p in left.glob('*.nc') if not p.is_symlink())
        actual = {p.name for p in right.glob('*.nc') if not p.is_symlink()}
        equal = bool(names) and actual == set(names)
        passed &= equal
        fields = []
        for name in names:
            if name not in actual:
                fields.append(dict(name=name, missing=True))
                continue
            run = subprocess.run([str(args.comparator.resolve()), str(left/name), str(right/name)], capture_output=True, text=True)
            value = json.loads(run.stdout) if run.stdout else dict(error=run.stderr)
            fields.append(dict(name=name, exit_code=run.returncode, comparison=value))
            passed &= run.returncode == 0 and value.get('all_identical', False)
        comparisons.append(dict(baseline=baseline, other=other, inventory_equal=equal, samples=fields))
    receipt = dict(scope=args.scope, comparisons=comparisons, passed=passed)
    if args.authority:
        receipt['source_authority'] = json.loads(args.authority.read_text())['source_authority']
    args.out.write_text(json.dumps(receipt, indent=2)+'\n')
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
