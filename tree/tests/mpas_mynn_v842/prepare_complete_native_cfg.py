"""Prepare original full-module CFG evidence for uncovered driver edges.

The native scientific files and full bank harness are copied unchanged. GCC
diagnostic dumps identify generated descriptor and optional-argument paths;
the complete original coverage union remains the reachability evidence.
"""
from pathlib import Path
import hashlib
import json
import shutil


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    source = Path('cpuwork/finite-edge-bank-v1/native')
    output = Path('cpuwork/complete-native-cfg-v69')
    output.mkdir(exist_ok=False)
    files = {}
    for path in source.iterdir():
        if path.suffix not in ('.F', '.F90') and path.name != 'Makefile':
            continue
        shutil.copy2(path, output/path.name)
        files[path.name] = sha(path)
    make = output/'Makefile'
    original = make.read_text()
    assert original.count('-O3 -cpp') == 1
    make.write_text(original.replace('-O3 -cpp',
        '-O3 -fdump-tree-cfg-lineno -fdump-tree-original -cpp'))
    receipt = Path('n1out/complete-native-cfg-v69')
    receipt.mkdir(exist_ok=False)
    (receipt/'admission.json').write_text(json.dumps(dict(scope=__doc__,
        process_count=1, files=files, Makefile=make.read_text(),
        original_Makefile_sha256=files['Makefile'], excluded_words=[],
        acceptance_proven=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
