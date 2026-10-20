"""Transport authenticated records once per SHA; restore ordinary dump files.

This performs only file hashing, copying and linking. The frozen f1 door still
checks the original manifest and Rust still decodes and compares every record.
No candidate or reference payload is modified by this transport.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export(specification, output):
    output.mkdir(parents=True, exist_ok=True)
    objects = output / 'objects'
    objects.mkdir(exist_ok=True)
    rows = []
    checked = set()
    for entry in json.loads(specification.read_text()):
        source = Path(entry['source'])
        manifest = source / 'manifest.json'
        original = json.loads(manifest.read_text())
        prefixes = [name for name, capture in original['captures'].items()
                    if (not entry.get('site') or capture['site'] == entry['site'])
                    and (not entry.get('steps') or
                         any(f'.s{step:010d}.' in name for step in entry['steps']))]
        files = {name: row for name, row in original['files'].items()
                 if any(name.startswith(prefix + '.pre.') or
                        name.startswith(prefix + '.post.') for prefix in prefixes)}
        metadata = output / entry['label']
        metadata.mkdir(exist_ok=True)
        if (metadata / 'manifest.json').exists():
            assert digest(metadata / 'manifest.json') == digest(manifest)
        else:
            shutil.copyfile(manifest, metadata / 'manifest.json')
        for name, row in files.items():
            target = objects / row['sha256']
            if target.exists() and row['sha256'] not in checked:
                assert target.stat().st_size == row['bytes']
                assert digest(target) == row['sha256']
            elif not target.exists():
                roots = [source, *map(Path, entry.get('members', []))]
                candidate = next((root / name for root in roots
                                  if (root / name).is_file()), None)
                if candidate is None:
                    raise ValueError(f'original native record unavailable: {name}')
                if candidate.stat().st_size != row['bytes'] or digest(candidate) != row['sha256']:
                    raise ValueError(f'original native record authentication failed: {name}')
                os.link(candidate, target)
            checked.add(row['sha256'])
        rows.append(dict(label=entry['label'], files=files,
                         original_manifest_sha256=digest(manifest)))
    (output / 'transport.json').write_text(json.dumps(rows) + '\n')
    print(json.dumps(dict(captures=len(rows), unique_files=len(list(objects.iterdir())),
                          unique_bytes=sum(p.stat().st_size for p in objects.iterdir()),
                          unchanged_payloads=True)), flush=True)


def restore(transport, output, resume=False):
    output.mkdir(parents=True, exist_ok=resume)
    checked = set()
    for row in json.loads((transport / 'transport.json').read_text()):
        target = output / row['label']
        target.mkdir(exist_ok=resume)
        manifest = transport / row['label'] / 'manifest.json'
        assert digest(manifest) == row['original_manifest_sha256']
        if (target / 'manifest.json').exists():
            assert digest(target / 'manifest.json') == row['original_manifest_sha256']
        else:
            shutil.copyfile(manifest, target / 'manifest.json')
        for name, member in row['files'].items():
            assert Path(name).name == name
            source = transport / 'objects' / member['sha256']
            if member['sha256'] not in checked:
                assert source.stat().st_size == member['bytes']
                assert digest(source) == member['sha256']
                checked.add(member['sha256'])
            candidate=target / name
            if candidate.exists():
                assert resume
                assert candidate.stat().st_size==member['bytes']
                assert candidate.samefile(source) or digest(candidate)==member['sha256']
            else:
                os.link(source, candidate)
    print(json.dumps(dict(captures=len(list(output.iterdir())),
                          unique_files_verified=len(checked), unchanged_payloads=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path)
    parser.add_argument('--restore', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-restore', action='store_true')
    args = parser.parse_args()
    if bool(args.export) == bool(args.restore):
        parser.error('choose --export or --restore')
    if args.export:
        export(args.export, args.output)
    else:
        restore(args.restore, args.output, args.resume_restore)
