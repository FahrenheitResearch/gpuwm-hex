"""Compare every standalone diagnostic record; never exclude any word."""
import argparse,hashlib,json
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path


def compare(reference,candidate,receipt,scope):
    result={'scope':scope,'reference':str(reference),'candidate':str(candidate),
            'different_words':0,'words':0,'record_mismatches':0,'decks':{},'fields':{}}
    hashes=[hashlib.sha256(),hashlib.sha256()]
    ranges=[]
    with reference.open('rb') as ref,candidate.open('rb') as test:
        for offset,pair in enumerate(zip_longest(ref,test)):
            left,right=pair
            for digest,line in zip(hashes,pair):
                if line is not None:digest.update(line)
            if left is None or right is None:
                result['record_mismatches']+=1
                continue
            a,b=left.split(),right.split()
            if a[:-1]!=b[:-1]:result['record_mismatches']+=1
            field=a[0].decode();deck=a[1].decode()
            f=result['fields'].setdefault(field,{'words':0,'different_words':0,'examples':[]})
            d=result['decks'].setdefault(deck,{'words':0,'different_words':0,'fields':{}})
            result['words']+=1;f['words']+=1;d['words']+=1
            if a[-1]!=b[-1]:
                result['different_words']+=1;f['different_words']+=1;d['different_words']+=1
                d['fields'][field]=d['fields'].get(field,0)+1
                if len(f['examples'])<8:
                    f['examples'].append({'indices':list(map(int,a[1:-1])),
                                          'reference':a[-1].decode(),'candidate':b[-1].decode()})
                if ranges and ranges[-1][1]==offset-1:ranges[-1][1]=offset
                else:ranges.append([offset,offset])
    result['reference_sha256'],result['candidate_sha256']=[h.hexdigest() for h in hashes]
    result['different_record_ranges_zero_based_inclusive']=ranges
    result['equal']=result['different_words']==result['record_mismatches']==0
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('scope','words','different_words','record_mismatches','equal')}))
    return int(not result['equal'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True);p.add_argument('--scope',required=True)
    a=p.parse_args();raise SystemExit(compare(a.reference,a.candidate,a.receipt,a.scope))
