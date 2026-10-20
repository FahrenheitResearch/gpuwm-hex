"""Compare complete unchanged native forecasts across measured hosts/ranks.

Every numeric and fixed-width stored word is read and compared by Rust. Native
outputs stay canonical on the host that wrote them. This diagnostic never authorizes exclusions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def admit(path):
    record = json.loads((path/'WITNESS.json').read_text())
    for name, expected in record['files'].items():
        if sha(path/name) != expected:
            raise ValueError('Canonical native bytes changed: '+name)
    if not (path/'MANIFEST.json').exists():
        if record['completed']:
            raise ValueError('A completed native witness lacks its manifest')
        return record
    manifest = json.loads((path/'MANIFEST.json').read_text())
    for name in manifest['outputs']:
        if manifest['outputs'][name] != record['files'].get(name):
            raise ValueError('Native output inventory changed: '+name)
    if manifest['ranks'] != record['process_count']:
        raise ValueError('Native rank metadata disagrees')
    for name, expected in manifest['inputs'].items():
        if name.endswith('.nc') and sha(path/name) != expected:
            raise ValueError('Static native input changed: '+name)
    return record


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    reference = admit(a.reference)
    if not reference['completed'] or reference['runner_exit'] != 0:
        raise ValueError('The reference must be a complete unchanged native run')
    results = []
    left = {p.name for p in a.reference.glob('*.nc') if p.is_file()}
    for candidate in a.candidates:
        record = admit(candidate)
        for key in ['original_binary_sha256','native_namelist_sha256','native_streams_sha256']:
            if record[key] != reference[key]:
                raise ValueError('Native witness changes an admitted input: '+key)
        label = record['host']+'-rank'+str(record['process_count'])+'-'+candidate.name
        root = a.output/label
        root.mkdir(exist_ok=False)
        (root/'reference-admission.json').write_text(json.dumps(reference, indent=2)+'\n')
        (root/'candidate-admission.json').write_text(json.dumps(record, indent=2)+'\n')
        right = {p.name for p in candidate.glob('*.nc') if p.is_file()}
        inventory = []
        groups = []
        words = differences = missing_words = 0
        for name in sorted(left):
            receipt = root/(name+'.json')
            masks = root/(name+'.masks')
            target = str(candidate/name) if name in right else '-'
            command = [str(a.tool),str(a.reference/name),target,'ALL_REFERENCE',str(receipt),str(masks)]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode not in [0,1]:
                inventory.append(dict(file=name,comparison_complete=False,
                    returncode=completed.returncode,error=completed.stderr))
                continue
            grade = json.loads(receipt.read_text())
            file_words = sum(field['words'] for field in grade['fields'])
            words += file_words
            differences += grade['differing_words']
            file_missing = sum(field['missing_words'] for field in grade['fields'])
            missing_words += file_missing
            groups.extend(dict(file=name,field=field['name'],words=field['words'],
                               differing_words=field['differing_words'],
                               mask=field['differing_word_mask']) for field in grade['fields'])
            inventory.append(dict(file=name,comparison_complete=True,
                compared_words=file_words,differing_words=grade['differing_words'],
                source_reference_sha256=sha(a.reference/name),
                missing_words=file_missing,
                source_candidate_sha256=sha(candidate/name) if name in right else None,receipt_sha256=sha(receipt)))
        result = dict(scope=__doc__,reference_host=reference['host'],candidate_host=record['host'],
            comparison_host=socket.gethostname(),reference_process_count=reference['process_count'],
            candidate_process_count=record['process_count'],words=words,differing_words=differences,
            missing_words=missing_words, reference_completed=reference['completed'],
            candidate_completed=record['completed'], candidate_runner_exit=record['runner_exit'],
            common_files=len(left & right),missing_files=sorted(left-right),extra_files=sorted(right-left),
            word_comparison_complete=all(r['comparison_complete'] for r in inventory),
            comparison_complete=record['completed'] and left==right and all(r['comparison_complete'] for r in inventory),
            files=inventory,groups=groups,tool_sha256=sha(a.tool),excluded_words=[],exclusions_proposed=False)
        (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        results.append({k:v for k,v in result.items() if k not in ['files','groups']})
        (a.output/'summary.json').write_text(json.dumps(dict(scope=__doc__,comparisons=results,
            excluded_words=[],exclusions_proposed=False),indent=2)+'\n')
    return int(any(not r['comparison_complete'] or r['differing_words'] for r in results))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['reference','tool','output']:
        parser.add_argument('--'+name,type=lambda s:Path(s).resolve(),required=True)
    parser.add_argument('--candidates',type=lambda s:Path(s).resolve(),nargs='+',required=True)
    raise SystemExit(run(parser.parse_args()))
