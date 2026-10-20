#!/usr/bin/env python3
"""Compare every pair of real native replay outputs with the Rust word reader.

These differences describe only the sealed inputs and named executables.
They never determine a proof result or supply a tolerance.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from column_oracle_replay import read_replay_receipt


def sha(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def process_counts(rows):
    counts=[r.get('replay_process_count') for r in rows]
    if not counts or any(type(c) is not int or c<1 for c in counts):
        raise ValueError('every compiler replay needs its recorded positive process count')
    if len(set(counts))!=1:
        raise ValueError('compiler debugging comparison requires the same process count')
    return counts


def compare_authenticated_array(left, right, files, words):
    """Reuse exact computed comparisons; distinct changed outputs need bytes.

    The caller binds these original array rows to authenticated replay receipts.
    Hash equality proves byte identity, and equality to the common native
    reference makes its already measured comparison exactly the requested pair.
    Differences between two other hashes are measured by Rust, never inferred
    from their separate distances to the reference.
    """
    keys=('words','different_words','max_ulp','different_nonfinite_words',
          'first_difference','bitwise_equal')
    for row in (left,right):
        if row.get('comparison_complete',True) is not True or row.get('schema_mismatch'):
            raise ValueError('incomplete original native array comparison')
        for key in ('reference_sha256','replay_sha256'):
            if not isinstance(row.get(key),str) or not re.fullmatch('[0-9a-f]{64}',row[key]):
                raise ValueError('native array comparison needs record hashes')
        if any(key not in row for key in keys):
            raise ValueError('missing original native array statistics')
        n,d=row['words'],row['different_words']
        nf,ulp=row['different_nonfinite_words'],row['max_ulp']
        if (type(n) is not int or type(d) is not int or not 0<=d<=n
                or type(nf) is not int or not 0<=nf<=d
                or not (ulp is None or type(ulp) is int and ulp>=0)
                or (row['first_difference'] is None)!=(d==0)
                or type(row['bitwise_equal']) is not bool or row['bitwise_equal']!=(d==0)
                or (row['replay_sha256']==row['reference_sha256'] and d!=0)):
            raise ValueError('inconsistent original native array comparison')
    if left['reference_sha256']!=right['reference_sha256'] or left['words']!=right['words']:
        raise ValueError('native array references disagree')
    if left['replay_sha256']==right['replay_sha256']:
        if any(left[k]!=right[k] for k in keys):
            raise ValueError('identical output hashes have inconsistent original statistics')
        return dict(words=left['words'],different_words=0,
                    max_ulp=None if left['max_ulp'] is None else 0,
                    different_nonfinite_words=0,first_difference=None,bitwise_equal=True,
                    evidence='identical computed output SHA256')
    for baseline,other in ((left,right),(right,left)):
        if baseline['replay_sha256']==baseline['reference_sha256']:
            return dict({key:other[key] for key in keys},
                        evidence='computed output equals common native reference SHA256')
    for row,path in zip((left,right),files):
        if not path.is_file() or sha(path)!=row['replay_sha256']:
            raise ValueError(f'distinct changed native outputs require authenticated bytes: {path}')
    run=subprocess.run([str(words.resolve()),'compare',*map(str,files)],
                       check=True,capture_output=True,text=True)
    result=json.loads(run.stdout)
    if not result.get('comparison_complete',True) or result.get('schema_mismatch'):
        raise ValueError('incomparable changed native output schemas')
    return dict(result,evidence='Rust comparison of authenticated computed outputs')


def write_receipt(path,result,limit=9_000_000):
    """Keep complete per-array evidence in receipts below the delivery cap."""
    counts=process_counts([{'replay_process_count':c} for c in result.get('replay_process_counts',[])])
    result=dict(result,purpose='debugging-only',proof_bar=False,process_count=counts[0])
    encode=lambda value:(json.dumps(value,separators=(',',':'))+'\n').encode()
    payload=encode(result)
    if len(payload)<=limit:
        path.write_bytes(payload)
        return
    header={'schema':'mpas-native-replay-spread-arrays-v1','arrays':{},
            'purpose':'debugging-only','proof_bar':False,'replay_process_counts':counts}
    overhead=len(encode(header))
    arrays=result['arrays'];chunks=[];chunk={};size=overhead
    for name,row in arrays.items():
        member_size=len(encode({name:row}))
        if member_size+overhead>limit:raise ValueError('one native array spread exceeds receipt cap')
        if chunk and size+member_size>limit:
            chunks.append(chunk);chunk={};size=overhead
        chunk[name]=row;size+=member_size
    if chunk:chunks.append(chunk)
    shards=[]
    for ordinal,chunk in enumerate(chunks,1):
        shard=path.with_name(f'{path.stem}.arrays-{ordinal:04d}.json')
        payload=encode({'schema':'mpas-native-replay-spread-arrays-v1','arrays':chunk,
                        'purpose':'debugging-only','proof_bar':False,
                        'replay_process_counts':result['replay_process_counts']})
        if len(payload)>limit:raise ValueError('native spread shard exceeds receipt cap')
        with shard.open('xb') as stream:stream.write(payload)
        shards.append({'file':shard.name,'sha256':sha(shard),'bytes':len(payload),'arrays':len(chunk)})
    index=dict(result,arrays={},schema='mpas-native-replay-spread-v2',array_shards=shards,
               array_count=len(arrays))
    path.write_bytes(encode(index))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',type=Path,action='append',required=True)
    p.add_argument('--results',type=Path,action='append',required=True)
    p.add_argument('--words',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reuse-authenticated-comparisons',action='store_true',
                   help='reuse exact original comparisons for retired equal outputs; changed pairs still require bytes')
    a=p.parse_args()
    if len(a.receipt)!=len(a.results):p.error('one result directory per receipt is required')
    rows=[read_replay_receipt(r) for r in a.receipt]
    counts=process_counts(rows)
    recipes=[r['replay_recipe'] for r in rows]
    required={'gnu-strict','gnu-o3','ifx-o3','ifx-valuesafe'}
    if set(recipes)!=required or len(recipes)!=4:raise ValueError('exactly the four native recipes are required')
    for key in ['dump_root_sha256','pin_set_sha256','source_sha256','scheme']:
        if len({json.dumps(r[key],sort_keys=True) for r in rows})!=1:raise ValueError(f'receipts disagree: {key}')
    sets=[{(c['capture'],n) for c in r['captures'] for n in c['arrays']} for r in rows]
    if any(s!=sets[0] for s in sets):raise ValueError('replay capture/array sets disagree')
    arrays={}
    for capture,name in sorted(sets[0]):
        pairs=[]
        for i,j in itertools.combinations(range(4),2):
            files=[a.results[k]/f'{capture}.post.{name}.bin' for k in [i,j]]
            records=[next(c for c in rows[k]['captures'] if c['capture']==capture)['arrays'][name]
                     for k in (i,j)]
            if a.reuse_authenticated_comparisons:
                comparison=compare_authenticated_array(*records,files,a.words)
                pairs.append(dict(recipes=[recipes[i],recipes[j]],
                                  output_sha256=[r['replay_sha256'] for r in records],**comparison))
                continue
            for k,f in zip([i,j],files):
                expected=next(c for c in rows[k]['captures'] if c['capture']==capture)['arrays'][name]['replay_sha256']
                if sha(f)!=expected:raise ValueError(f'replay output hash changed: {f}')
            result=subprocess.run([str(a.words.resolve()),'compare',*map(str,files)],check=True,capture_output=True,text=True)
            comparison=json.loads(result.stdout)
            if not comparison.get('comparison_complete',True):
                raise ValueError(f'incomparable native schemas for {capture}:{name}: {comparison["schema_mismatch"]}')
            pairs.append(dict(recipes=[recipes[i],recipes[j]],**comparison))
        arrays[f'{capture}:{name}']={'largest_pairwise_differing_words':max(r['different_words'] for r in pairs),
            'largest_pairwise_max_ulp':max(r['max_ulp'] or 0 for r in pairs),
            'pairs':pairs}
    result={k:rows[0][k] for k in ['dump_root_sha256','pin_set_sha256','source_sha256','scheme']}
    result.update(schema='mpas-native-replay-spread-v1',arrays=arrays,forecast_proven=False,
                  purpose='debugging-only',proof_bar=False,
                  process_count=counts[0],replay_process_counts=counts,
                  receipts=[{'recipe':r['replay_recipe'],'sha256':sha(f),'executable_sha256':r['executable_sha256']} for r,f in zip(rows,a.receipt)])
    write_receipt(a.out,result)


if __name__=='__main__':main()
