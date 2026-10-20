"""Seal whole CAM oracle replays, including every declared POST member.

Rust's per-array word receipts decide the result. Missing captures, missing
arrays, incomplete comparisons and process-count changes cannot become passes.
This supporting comparison never establishes whole-forecast acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(pairs, output):
    rows=[]
    for manifest_path,receipt_path in pairs:
        manifest=json.loads(manifest_path.read_text())
        result=json.loads(receipt_path.read_text())
        if (not result['comparison_complete']
                or result['execution_mode']!='whole-capture'
                or result['backend']!='hex-cuda'
                or result['capture_process_count']!=1
                or result['replay_process_count']!=1
                or manifest['process_count']!=1
                or manifest['dump_root_sha256']!=result['dump_root_sha256']):
            raise ValueError('Whole-capture provenance/completion differs')
        captures=result['captures']
        if (len(captures)!=len(manifest['captures'])
                or {r['capture'] for r in captures}!=set(manifest['captures'])):
            raise ValueError('Complete capture set differs')
        arrays=words=differences=0
        for capture in captures:
            key=capture['capture']
            prefix=key+'.post.'
            expected={name[len(prefix):-4] for name in manifest['files']
                      if name.startswith(prefix) and name.endswith('.bin')}
            if (capture['site']!=manifest['captures'][key]['site']
                    or set(capture['arrays'])!=expected or not expected):
                raise ValueError('Every native POST array must be compared')
            for array in capture['arrays'].values():
                if (array['different_words'] is None
                        or array['bitwise_equal']!=(array['different_words']==0)):
                    raise ValueError('Array comparison is incomplete/inconsistent')
                arrays+=1;words+=array['words'];differences+=array['different_words']
        if result['all_outputs_bitwise_equal']!=(differences==0):
            raise ValueError('Word total differs from replay status')
        rows.append(dict(scheme=result['scheme'],captures=len(captures),arrays=arrays,
                         words=words,different_words=differences,
                         manifest_sha256=digest(manifest_path),receipt_sha256=digest(receipt_path),
                         hidden_state_complete=result['hidden_state_complete']))
    if not rows:
        raise ValueError('No whole captures supplied')
    total=sum(r['different_words'] for r in rows)
    record=dict(scope='whole CAM captured-output comparison; supporting evidence only',
                complete=True,process_count=1,excluded_words=0,
                whole_model_acceptance=False,passed=total==0,
                arrays=sum(r['arrays'] for r in rows),words=sum(r['words'] for r in rows),
                differing_words=total,replays=rows)
    output.write_text(json.dumps(record,indent=2)+'\n')
    return int(not record['passed'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',nargs=2,type=Path,action='append',required=True,
                        metavar=('MANIFEST','RECEIPT'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(seal(args.replay,args.output))
