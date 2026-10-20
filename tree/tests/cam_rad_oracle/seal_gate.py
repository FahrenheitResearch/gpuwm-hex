"""Seal completed product/base G0 attempts without waiving any failure."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def attempt(directory,checkout):
    if subprocess.check_output(['git','-C',str(checkout),'status','--porcelain'],text=True):
        raise ValueError('Gate checkout differs from committed HEAD')
    gate=directory/'G0/gate.json'
    result=json.loads(gate.read_text())
    text=(directory/'G0/G0-0.log').read_text()
    match=re.search(r'^(\d+) failed, (\d+) passed, (\d+) skipped, (\d+) deselected',text,re.M)
    if match is None:raise ValueError('Full G0 test summary is absent')
    failures=re.findall(r'^FAILED (\S+)',text,re.M)
    if int(match[1])!=len(failures):raise ValueError('Failure inventory differs from full test summary')
    citations=(directory/'G0/G0-1.log').read_text()
    audit=re.search(r'(\d+) citation occurrences, (\d+) unique ranges, (\d+) failures',citations)
    if audit is None:raise ValueError('Complete citation audit is absent')
    return dict(commit=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip(),
        passed=result['passed'],exit_codes=result['stages']['G0']['exit_codes'],
        failed=int(match[1]),passed_tests=int(match[2]),skipped=int(match[3]),deselected=int(match[4]),
        failure_ids=failures,citations=dict(occurrences=int(audit[1]),ranges=int(audit[2]),failures=int(audit[3])),
        files={str(p.relative_to(directory)):digest(p) for p in (gate,directory/'G0/G0-0.log',directory/'G0/G0-1.log')})


def main(args):
    product=attempt(args.product,args.checkout)
    baseline=attempt(args.baseline,args.base_checkout)
    result=dict(scope='complete product and matching-base G0 attempts; original failure grades preserved',
        passed=product['passed'],whole_model_acceptance=False,product=product,baseline=baseline,
        identical_failure_ids=set(product['failure_ids'])==set(baseline['failure_ids']),
        added_failure_ids=sorted(set(product['failure_ids'])-set(baseline['failure_ids'])),
        retired_failure_ids=sorted(set(baseline['failure_ids'])-set(product['failure_ids'])),
        hostprep_sha256=digest(args.hostprep),orchestration_sha256=digest(Path(__file__)),
        waivers=0,ratchet_changed=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('product','baseline','checkout','base-checkout','hostprep','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
