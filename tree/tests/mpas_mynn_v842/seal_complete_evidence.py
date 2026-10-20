"""Preserve a complete receipt directory, splitting large files without loss."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}
    for path in sorted(args.source.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(args.source)
        size = path.stat().st_size
        row = dict(bytes=size, sha256=sha(path))
        if size < 10000000:
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            if sha(destination) != row['sha256']:
                raise ValueError('Receipt changed during sealing')
        else:
            chunks = []
            whole = hashlib.sha256()
            with path.open('rb') as source:
                while data := source.read(8000000):
                    destination = args.output / (str(relative)+f'.part-{len(chunks):05}')
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                    whole.update(data)
                    chunks.append(dict(file=str(destination.relative_to(args.output)),
                        bytes=len(data), sha256=sha(destination)))
            if whole.hexdigest() != row['sha256']:
                raise ValueError('Large receipt changed during sealing')
            row['ordered_chunks'] = chunks
        files[str(relative)] = row
    (args.output / 'SEAL.json').write_text(json.dumps(dict(scope=__doc__,
        source=str(args.source), process_count=1, files=files,
        omitted_receipt_bytes=0, excluded_words=[]), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
