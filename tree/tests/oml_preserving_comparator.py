#!/usr/bin/env python
"""Delegate to the pinned Rust comparator and retain original candidate bytes.

The real CLI/verifier still executes its full comparison and writes its own
unaltered receipt. Retained files allow a separate Rust check of character and
integer words after the verifier retires its disposable forecast directory.
Python copies files and records provenance; it never decodes weather fields.
"""
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def retain_completed_file(candidate, target):
    """Keep completed output across verifier unlinking without duplicating it.

    This runs only after the real forecast and pinned Rust comparison finish.
    Same-filesystem hard links preserve the original inode when f2 removes its
    disposable directory. An actual cross-device error requires a byte copy.
    The caller authenticates both paths before recording the capture.
    """
    if target.exists() or target.is_symlink():
        raise ValueError('Preserve an existing candidate capture')
    try:
        os.link(candidate, target)
        return 'hardlink'
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
    shutil.copyfile(candidate, target)
    return 'cross-device-copy'


def main():
    original = Path(os.environ['MPAS_OML_ORIGINAL_COMPARATOR']).resolve()
    if (original == Path(__file__).resolve()
            or sha(original) != os.environ['MPAS_OML_ORIGINAL_COMPARATOR_SHA256']):
        raise ValueError('Pinned original Rust comparator changed')
    args = sys.argv[1:]
    status = subprocess.run([str(original), *args]).returncode
    if status or '--request' not in args:
        return status
    request_path = Path(args[args.index('--request') + 1]).resolve()
    output_path = Path(args[args.index('--out') + 1]).resolve()
    request = json.loads(request_path.read_text())
    retained = Path(os.environ['MPAS_OML_RETAINED_WORD_OUTPUT']).resolve()
    if retained.exists():
        raise ValueError('Retained forecast bytes already exist; never overwrite')
    retained.mkdir(parents=True)
    files = []
    seen = set()
    for sample in request['samples']:
        candidate = Path(sample['hex']).resolve()
        if not candidate.is_file() or candidate in seen:
            continue
        seen.add(candidate)
        target = retained / candidate.name
        if target.exists():
            raise ValueError('Distinct candidate files share a filename')
        preservation = retain_completed_file(candidate, target)
        digest = sha(candidate)
        if sha(target) != digest:
            raise ValueError('Original candidate bytes changed during preservation')
        strict = Path(sample['strict']).resolve()
        files.append(dict(stream=sample['stream'], candidate=str(target),
                          candidate_sha256=digest, candidate_bytes=target.stat().st_size,
                          preservation_method=preservation,
                          strict=str(strict), strict_sha256=sha(strict)))
    if not files:
        raise ValueError('No original written forecast files to retain')
    receipt = dict(schema='p19-original-written-forecast-files/v1',
                   forecast_acceptance=False, process_count=1, files=files,
                   comparator=dict(path=str(original), sha256=sha(original)),
                   orchestration=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
                   numeric_comparison=dict(path=str(output_path), sha256=sha(output_path)),
                   request=dict(path=str(request_path), sha256=sha(request_path)))
    (retained / 'ORIGINALS.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return status


if __name__ == '__main__':
    raise SystemExit(main())
