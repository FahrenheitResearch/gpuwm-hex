"""Restore exact diagnostic bytes from a canonical chunked receipt seal.

No scientific field is decoded or changed. Every chunk and complete file must
match its admitted hash before an existing executable or trace is reused.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check(path, item):
    if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
        raise ValueError(f'Canonical receipt hash mismatch: {path}')


def main(a):
    seal_path = a.source / 'SEAL.json'
    seal = json.loads(seal_path.read_text())
    rows = []
    for name in a.file:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Restored receipt must stay inside its directory')
        item = seal['files'][name]
        target = a.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            check(target, item)
        else:
            parts = item.get('ordered_chunks', [dict(file=name, **item)])
            for part in parts:
                check(a.source / part['file'], part)
            temporary = target.with_name(target.name + '.restoring')
            with temporary.open('xb') as output:
                for part in parts:
                    with (a.source / part['file']).open('rb') as source:
                        shutil.copyfileobj(source, output)
            check(temporary, item)
            temporary.rename(target)
        if name == 'p12-source-trace':
            target.chmod(0o755)
        rows.append(dict(file=name, bytes=item['bytes'], sha256=item['sha256']))
    a.receipt.write_text(json.dumps(dict(scope=__doc__,
        canonical_seal_sha256=sha(seal_path), files=rows,
        compiler_invoked=False, omitted_bytes=0, excluded_words=[]), indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'receipt'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--file', action='append', required=True)
    main(p.parse_args())
