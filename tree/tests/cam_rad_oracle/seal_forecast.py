"""Seal full CAM forecast/comparison metadata without changing its proof grade."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def seal(args):
    execution=json.loads((args.forecast/'execution.json').read_text())
    summary=json.loads((args.comparison/'summary.json').read_text())
    verification=json.loads((args.comparison/'verify.json').read_text())
    if summary['execution_sha256']!=digest(args.forecast/'execution.json'):
        raise ValueError('Comparison execution changed')
    if summary['verify_sha256']!=digest(args.comparison/'verify.json'):
        raise ValueError('Field comparison changed')
    streams=defaultdict(lambda:dict(words=0,differing_words=0,samples=0))
    for row in verification['comparison']['fields']:
        if row['status']=='compared':
            stream=streams[row['stream']]
            stream['words']+=row['compared_words'];stream['differing_words']+=row['differing_words'];stream['samples']+=1
    if sum(r['words'] for r in streams.values())!=summary['written_compared_words']:
        raise ValueError('Summary word count differs from full comparison')
    if sum(r['differing_words'] for r in streams.values())!=summary['written_differing_words']:
        raise ValueError('Summary differences differ from full comparison')
    control=None
    if args.control is not None:
        source=args.control/'summary.json';baseline=json.loads(source.read_text())
        control=dict(summary_sha256=digest(source),inventory_identical=summary['missing_samples']==baseline['missing_samples'])
    policy=verification.get('recorded_missing_policy',{})
    result=dict(scope='sealed complete CAM door execution and written-field comparison; original grade retained',
        whole_model_acceptance=False,process_count=summary['process_count'],summary=summary,
        summary_sha256=digest(args.comparison/'summary.json'),streams=dict(streams),
        module_sha256=execution['module_sha256'],artifact_sha256=execution['artifact_sha256'],
        missing_samples=len(summary['missing_samples']),missing_inventory_control=control,
        recorded_missing_policy_applied=policy.get('applied'),recorded_missing_policy_failures=len(policy.get('failures',[])),
        acceptance_tier=verification.get('acceptance_tier'),all_native_fields_bitwise=verification.get('all_native_fields_bitwise'),
        sealer_sha256=digest(Path(__file__)),excluded_words=summary['excluded_words'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(summary['written_differing_words'],summary['written_compared_words'],dict(streams),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('forecast','comparison','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--control',type=Path)
    seal(parser.parse_args())
