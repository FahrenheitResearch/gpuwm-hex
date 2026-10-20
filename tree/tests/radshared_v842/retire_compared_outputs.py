"""Retire explicitly listed disposable outputs after their receipts are sealed.

This is storage orchestration only. Native references and replay inputs must
not appear in the manifest. Every selected byte identity is checked before
any file is removed; comparison receipts keep their original conclusions.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'lane', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert manifest['disposable_outputs_only'] is True
    assert manifest['native_references_included'] is False
    lane = args.lane.resolve(strict=True)
    roots = [Path(x).resolve(strict=True) for x in manifest['owned_output_roots']]
    assert roots and all(x.is_relative_to(lane) for x in roots)
    for pin in manifest['sealed_receipts']:
        assert sha(Path(pin['path'])) == pin['sha256']
    checked = []
    paths = set()
    for row in manifest['files']:
        path = Path(row['path'])
        assert not path.is_symlink() and path.is_file()
        resolved = path.resolve(strict=True)
        assert resolved not in paths
        paths.add(resolved)
        assert any(resolved.is_relative_to(root) for root in roots)
        assert sha(path) == row['sha256']
        before = path.stat()
        assert before.st_size == row['bytes']
        checked.append((path, before))
    receipt = dict(schema='p21-compared-output-retirement/v1',
                   manifest_sha256=sha(args.manifest), complete=False,
                   scientific_receipts_unchanged=True, native_references_deleted=False,
                   files=manifest['files'], bytes=sum(x['bytes'] for x in manifest['files']))
    assert not args.receipt.exists()
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    for path, before in checked:
        current = path.stat()
        assert (current.st_ino, current.st_size, current.st_mtime_ns) == (
            before.st_ino, before.st_size, before.st_mtime_ns)
        path.unlink()
    receipt['complete'] = True
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(files=len(checked), bytes=receipt['bytes'], complete=True)))


if __name__ == '__main__':
    main()
