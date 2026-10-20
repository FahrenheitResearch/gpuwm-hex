"""Compare complete native-bank affected sets against one unchanged reference.

All masks are graded by Rust. No intersection is used to exclude any word.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import socket
import subprocess


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main(args):
    args.output.mkdir(exist_ok=False)
    if args.bank:
        grades = {name: args.bank / (name + '.json')
                  for name in ['native', 'coverage', 'snan', 'host']}
    else:
        grades = {}
        for value in args.grades:
            name, path = value.split('=', 1)
            if not name or name in grades:
                raise ValueError('Each original grade must have a unique label')
            grades[name] = Path(path)
    if len(grades) < 2:
        raise ValueError('At least two complete affected sets are required')
    if args.cuda:
        grades['cuda'] = args.cuda
    admitted = {}
    references = set()
    for name, path in grades.items():
        grade = json.loads(path.read_text())
        if grade['words'] != args.words or grade['excluded_words'] != []:
            raise ValueError('Complete native-bank inventory differs')
        references.add(grade.get('reference_sha256', grade.get('fixture_sha256')))
        for chunk in grade['grade_chunks']:
            if sha(Path(chunk['path'])) != chunk['sha256']:
                raise ValueError('Complete bank grade changed')
        admitted[name] = dict(path=str(path), sha256=sha(path),
                              differing_words=grade['differing_words'])
    if len(references) != 1 or None in references:
        raise ValueError('The unchanged native reference differs')
    results = []
    for left, right in itertools.combinations(grades, 2):
        label = left + '-' + right
        target = args.output / (label + '.json')
        with (args.output / (label + '.log')).open('w') as log:
            rc = subprocess.run([str(args.tool), str(grades[left]),
                                 str(grades[right]), str(target)],
                                stdout=log, stderr=subprocess.STDOUT).returncode
        if rc not in [0, 1]:
            raise ValueError('Rust did not complete the full affected set')
        result = json.loads(target.read_text())
        results.append(dict(label=label, words=result['words'],
            measured=result['measured_affected_words'],
            diagnostic=result['diagnostic_affected_words'],
            intersection=result['intersection_words'],
            symmetric_difference=result['symmetric_difference_words'],
            same_nonempty_set=result['same_nonempty_affected_word_set'],
            receipt=target.name, sha256=sha(target)))
    (args.output / 'summary.json').write_text(json.dumps(dict(scope=__doc__,
        process_count=1, host=socket.gethostname(), tool_sha256=sha(args.tool),
        reference_sha256=next(iter(references)), admitted_grades=admitted,
        comparisons=results, exclusions_proposed=False, excluded_words=[]),
        indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['output', 'tool']:
        parser.add_argument('--' + name, type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--bank', type=Path)
    inputs.add_argument('--grades', nargs='+')
    parser.add_argument('--cuda', type=Path)
    parser.add_argument('--words', type=int, required=True)
    main(parser.parse_args())
