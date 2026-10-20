"""Compare every full radiative-cloud affected-word set; no exclusions."""
from pathlib import Path
import hashlib
import json
import subprocess
import socket


def sha(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    bank = Path('n1out/radiative-cloud-bank-v3')
    cuda = Path('n1out/radiative-cloud-cuda-v12/cuda.json')
    out = Path('n1out/radiative-cloud-complete-sets-v12')
    out.mkdir(exist_ok=False)
    tool = Path('cpuwork/tools/p12-native-mask-sets')
    reference = json.loads((bank / 'native.json').read_text())['reference_sha256']
    records = []
    pairs = [('repeat-cuda', bank / 'native.json', cuda),
             ('coverage-cuda', bank / 'coverage.json', cuda),
             ('snan-cuda', bank / 'snan.json', cuda),
             ('host-cuda', bank / 'host.json', cuda),
             ('coverage-snan', bank / 'coverage.json', bank / 'snan.json'),
             ('host-snan', bank / 'host.json', bank / 'snan.json')]
    for label, left, right in pairs:
        for path in [left, right]:
            grade = json.loads(path.read_text())
            if (grade['words'] != 3473152 or grade['excluded_words'] != [] or
                grade.get('reference_sha256', grade.get('fixture_sha256')) != reference):
                raise ValueError('Complete original bank reference differs')
            if any(sha(Path(c['path'])) != c['sha256'] for c in grade['grade_chunks']):
                raise ValueError('Complete bank grade changed')
        target = out / (label + '.json')
        with (out / (label + '.log')).open('w') as log:
            rc = subprocess.run([str(tool), str(left), str(right), str(target)],
                                stdout=log, stderr=subprocess.STDOUT).returncode
        if rc not in [0, 1]:
            raise ValueError('Rust did not complete the full affected-word inventory')
        result = json.loads(target.read_text())
        records.append(dict(label=label, words=result['words'],
            measured=result['measured_affected_words'], diagnostic=result['diagnostic_affected_words'],
            intersection=result['intersection_words'], symmetric_difference=result['symmetric_difference_words'],
            same_nonempty_set=result['same_nonempty_affected_word_set'],
            receipt=target.name, sha256=sha(target)))
    (out / 'summary.json').write_text(json.dumps(dict(scope=__doc__, process_count=1,
        host=socket.gethostname(), tool_sha256=sha(tool), comparisons=records,
        exclusions_proposed=False, excluded_words=[]), indent=2) + '\n')


if __name__ == '__main__':
    main()
