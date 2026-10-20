"""Run every native TEMPO flag tuple in a separate process; retain crashes honestly."""
import argparse,itertools,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
rows=[]
parser=argparse.ArgumentParser();parser.add_argument('--all-records',action='store_true');parser.add_argument('--write-blobs',action='store_true');parser.add_argument('--levels',type=int,default=24);parser.add_argument('--oracle-prefix',default='oracle-');parser.add_argument('--receipt',type=Path,default=ROOT/'n1out/host-flag-matrix.json')
parser.add_argument('--native-library',type=Path)
parser.add_argument('--stress-corpus',action='store_true')
args=parser.parse_args()
for flags in itertools.product((0,1),repeat=4):
    tag=''.join(map(str,flags));log=ROOT/'n1out'/(args.oracle_prefix+'matrix-'+tag+'.log')
    with log.open('w') as output:
        options=[] if args.write_blobs and flags[0]==0 and flags[2]==0 else ['--no-blob']
        if args.all_records:options.append('--all-records')
        if args.stress_corpus:options.append('--stress-corpus')
        options+=['--levels',str(args.levels),'--oracle-prefix',args.oracle_prefix]
        if args.native_library:options+=['--native-library',str(args.native_library.resolve())]
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('replay.py')),json.dumps(flags),*options],stdout=output,stderr=subprocess.STDOUT)
    receipt=ROOT/'n1out'/(args.oracle_prefix+tag)/'host-replay.json'
    row={'flags':flags,'exit_code':result.returncode,'log':str(log.relative_to(ROOT))}
    if result.returncode==0 and receipt.exists():
        data=json.loads(receipt.read_text());row.update(passing_steps=sum(not r['state'] and not r['diagnostics'] for r in data['cases']),steps=len(data['cases']),receipt=str(receipt.relative_to(ROOT)))
    rows.append(row)
    args.receipt.write_text(json.dumps({'forecast_proven':False,'levels':args.levels,'rows':rows},indent=2)+'\n')
    print(json.dumps(row),flush=True)
