"""Retire explicitly named duplicate receipts after authenticating retained bytes.

Canonical references and the retained receipt directory are never removed.
This only removes redundant lane-owned copies, including failed proof receipts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main(args):
    lane = Path.cwd().resolve()
    retired = []
    for pair in args.pair:
        original, retained = [Path(value).resolve() for value in pair.split('=', 1)]
        if not original.is_relative_to(lane / 'n1out'):
            raise ValueError('Only explicitly named owned n1out copies may be retired')
        if not retained.is_relative_to(lane / 'receipts') or original == retained:
            raise ValueError('Retained copy must be an independent owned receipt')
        manifest = json.loads((retained / 'SEAL.json').read_text())
        expected = manifest['files']
        actual = {str(p.relative_to(original)) for p in original.rglob('*') if p.is_file()}
        if actual != set(expected):
            raise ValueError('Duplicate receipt has changed or has unsealed files')
        total = 0
        for name, row in expected.items():
            source = original / name
            if source.stat().st_size != row['bytes'] or digest(source) != row['sha256']:
                raise ValueError('Original receipt differs from its seal: ' + name)
            if 'ordered_chunks' in row:
                whole = hashlib.sha256()
                for part in row['ordered_chunks']:
                    chunk = retained / part['file']
                    if chunk.stat().st_size != part['bytes'] or digest(chunk) != part['sha256']:
                        raise ValueError('Retained receipt chunk changed')
                    whole.update(chunk.read_bytes())
                authenticated = whole.hexdigest()
            else:
                target = retained / name
                if target.stat().st_size != row['bytes']:
                    raise ValueError('Retained receipt size changed')
                authenticated = digest(target)
            if authenticated != row['sha256']:
                raise ValueError('Retained complete receipt differs: ' + name)
            total += row['bytes']
        shutil.rmtree(original)
        retired.append(dict(original=str(original), retained=str(retained),
                            seal_sha256=digest(retained / 'SEAL.json'),
                            bytes_retired=total, omitted_receipt_bytes=0))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(dict(scope=__doc__, retired=retired,
        bytes_retired=sum(row['bytes_retired'] for row in retired)), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', action='append', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    main(parser.parse_args())
