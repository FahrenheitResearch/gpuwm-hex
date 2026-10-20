"""Collect every existing native coverage graph without rebuilding science.

The original GNU gcno graph, detailed arc/line dump and annotated gcov output
bind branch numbers to compiler-generated conditions. Source hashes must
match the admitted complete bank. No uncovered branch is reclassified here.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    root = Path('cpuwork/finite-edge-bank-v1/coverage').resolve()
    out = Path('n1out/complete-native-gcov-graph-v81').resolve()
    out.mkdir(exist_ok=False)
    graphs = sorted(root.glob('*.gcno'))
    assert graphs, 'Original complete bank coverage build is not present'
    admission = json.loads(Path('receipts/finite-edge-source-trace-v1/original-source-admission.json').read_text())
    sources = {}
    for path in sorted(root.glob('*')):
        if path.suffix in ('.F', '.F90'):
            expected = admission['files']['coverage/' + path.name]
            assert sha(path) == expected
            sources[path.name] = expected
            shutil.copy2(path, out / path.name)
    rows = []
    for graph in graphs:
        shutil.copy2(graph, out / graph.name)
        dump = out / (graph.name + '.dump')
        with dump.open('w') as f:
            subprocess.run(['gcov-dump', '-l', str(graph)], stdout=f,
                           stderr=subprocess.STDOUT, check=True)
        rows.append(dict(graph=graph.name, sha256=sha(graph),
                         dump_sha256=sha(dump)))
    with (out / 'gcov.log').open('w') as f:
        result = subprocess.run(['gcov', '-b', '-c', '-a', *map(str, graphs)],
                                cwd=out, stdout=f, stderr=subprocess.STDOUT)
    (out / 'admission.json').write_text(json.dumps(dict(
        scope=__doc__, graphs=rows, native_sources=sources,
        gcov_returncode=result.returncode, process_count=1,
        compiler_invoked=False, gcov_branches_classified=[],
        excluded_words=[], acceptance_proven=False), indent=2) + '\n')
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
