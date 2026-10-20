"""Retire only authenticated copies created by a completed sheet job."""
import argparse
import hashlib
import json
from pathlib import Path


def retire(job, receipt):
    authority = json.loads((job / 'authority.json').read_text())
    rows = []
    for name, expected in authority['render_input_files'].items():
        path = job / name
        assert path.is_file() and not path.is_symlink()
        assert path.resolve().is_relative_to(job.resolve())
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
        rows.append(dict(path=name, sha256=expected, bytes=path.stat().st_size))
    result = dict(schema='p3-completed-sheet-copy-retirement/v1', acceptance=False,
        authority_sha256=hashlib.sha256((job / 'authority.json').read_bytes()).hexdigest(),
        job=str(job), files=rows, retired_bytes=sum(row['bytes'] for row in rows),
        scope='Only the copied rendering inputs. Original native/proof dependencies are untouched.')
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    for row in rows:
        (job / row['path']).unlink()
    assert not any((job / row['path']).exists() for row in rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    parser.add_argument('receipt', type=Path)
    retire(**vars(parser.parse_args()))
