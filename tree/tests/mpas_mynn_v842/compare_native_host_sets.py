"""Authenticate complete native host/rank masks against the same reference.

Partial native runs remain partial. Comparing their affected positions never
authorizes an exclusion or replaces comparison of the actual field values.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for root in a.inputs:
        summary=json.loads((root/'summary.json').read_text())
        reference=json.loads((root/'reference-admission.json').read_text())
        if not summary['word_comparison_complete']:
            raise ValueError('Every reference word must have been graded')
        groups=[]
        for row in summary['files']:
            path=root/(row['file']+'.json')
            if sha(path)!=row['receipt_sha256']:
                raise ValueError('Native word grade changed')
            grade=json.loads(path.read_text())
            groups.extend(dict(file=row['file'],field=field['name'],words=field['words'],
                differing_words=field['differing_words'],mask=field['differing_word_mask'])
                for field in grade['fields'])
        target=a.output/(root.name+'.json')
        target.write_text(json.dumps(dict(scope=__doc__,source_summary_sha256=sha(root/'summary.json'),
            reference=reference,groups=groups,excluded_words=[]),indent=2)+'\n')
        records.append((target,reference,summary))
    measured,ref,left=records[0]
    results=[]
    for diagnostic,right_ref,right in records[1:]:
        if right_ref!=ref:
            raise ValueError('Affected sets were measured against different native references')
        target=a.output/(diagnostic.stem+'-against-measured.json')
        command=[str(a.tool),str(measured),str(diagnostic),str(target)]
        completed=subprocess.run(command,capture_output=True,text=True)
        if completed.returncode not in [0,1]:
            raise ValueError(completed.stderr)
        grade=json.loads(target.read_text())
        results.append(dict(measured_host=left['candidate_host'],measured_process_count=left['candidate_process_count'],
            diagnostic_host=right['candidate_host'],diagnostic_process_count=right['candidate_process_count'],
            measured_forecast_complete=left['candidate_completed'],diagnostic_forecast_complete=right['candidate_completed'],
            words=grade['words'],intersection_words=grade['intersection_words'],
            symmetric_difference_words=grade['symmetric_difference_words'],
            same_nonempty_affected_word_set=grade['same_nonempty_affected_word_set'],
            receipt=target.name,sha256=sha(target),returncode=completed.returncode))
    (a.output/'summary.json').write_text(json.dumps(dict(scope=__doc__,comparisons=results,
        tool_sha256=sha(a.tool),excluded_words=[],exclusions_proposed=False),indent=2)+'\n')
    return int(any(not r['same_nonempty_affected_word_set'] for r in results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=lambda s:Path(s).resolve(),nargs='+',required=True)
    for name in ['output','tool']:
        p.add_argument('--'+name,type=lambda s:Path(s).resolve(),required=True)
    raise SystemExit(run(p.parse_args()))
