#!/usr/bin/env python3
"""Download explicit public input URLs with curl; retain byte hashes and headers.

Input JSON: {"files": [{"url": "https://...", "name": "..."}]}.
This copies bytes only. GRIB interpretation stays in native ungrib.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess


def download(row, directory):
    name = row['name']
    if Path(name).name != name or name in ('.', '..', 'MANIFEST.json'):
        raise ValueError('input names must be simple filenames')
    if not row['url'].startswith('https://'):
        raise ValueError('HTTPS input URL required')
    path = directory / name
    partial = directory / (name + '.partial')
    if not path.exists():
        subprocess.run(['curl', '--silent', '--show-error', '--fail', '--location', '--retry', '5',
                        '--connect-timeout', '30', '--max-time', '1800',
                        '--continue-at', '-', '--dump-header', str(directory / (name + '.headers')),
                        '--output', str(partial), row['url']], check=True)
        partial.rename(path)
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f'empty input {path}')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    result = dict(row, sha256=digest, bytes=path.stat().st_size)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=Path)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    rows = json.loads(args.index.read_text())['files']
    if not rows or len({r['name'] for r in rows}) != len(rows):
        raise ValueError('nonempty unique input names required')
    if not 1 <= args.jobs <= 4:
        raise ValueError('one to four download jobs allowed')
    args.directory.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda row: download(row, args.directory), rows))
    manifest = {'schema': 'mpas.public-input-download/v1', 'files': results}
    temporary = args.directory / 'MANIFEST.json.partial'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(args.directory / 'MANIFEST.json')


if __name__ == '__main__':
    main()
