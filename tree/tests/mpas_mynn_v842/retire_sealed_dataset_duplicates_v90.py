"""Retire only named, completed sealed datasets identical to canonical receipts.

Two seals may record different original source paths. Their complete scientific
file inventories must be identical, and every retained chunk and file is hashed.
No active source, shared reference, or unsealed file can be retired here.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def authenticate(root, seal):
    expected = {'SEAL.json': None}
    for name, row in seal['files'].items():
        if 'ordered_chunks' in row:
            whole = hashlib.sha256()
            for part in row['ordered_chunks']:
                assert part['file'] not in expected
                expected[part['file']] = part
                p = root / part['file']
                assert p.stat().st_size == part['bytes'] and sha(p) == part['sha256']
                with p.open('rb') as f:
                    while data := f.read(1024 * 1024):
                        whole.update(data)
            assert whole.hexdigest() == row['sha256']
        else:
            assert name not in expected
            expected[name] = row
            p = root / name
            assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256']
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    assert actual == set(expected), 'unsealed or missing source file'


def main(a):
    lane = Path.cwd().resolve()
    a.output.mkdir(exist_ok=False, parents=True)
    retired = []
    for index, pair in enumerate(a.pair):
        source, canonical = [Path(x).resolve() for x in pair.split('=', 1)]
        assert source.is_relative_to(lane / 'n1out')
        assert canonical.is_relative_to(lane / 'receipts')
        left = json.loads((source / 'SEAL.json').read_text())
        right = json.loads((canonical / 'SEAL.json').read_text())
        assert left['files'] == right['files']
        authenticate(source, left)
        authenticate(canonical, right)
        shutil.copy2(source / 'SEAL.json', a.output / f'{index}-original-seal.json')
        retired.append(dict(source=str(source), canonical=str(canonical),
                            original_seal_sha256=sha(source / 'SEAL.json'),
                            canonical_seal_sha256=sha(canonical / 'SEAL.json'),
                            bytes_retired=sum(p.stat().st_size for p in source.rglob('*') if p.is_file()),
                            omitted_receipt_bytes=0))
        shutil.rmtree(source)
        (a.output / 'summary.json').write_text(json.dumps(dict(scope=__doc__, retired=retired), indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair', action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
