"""Replay stated immutable native calls on a candidate oracle checkout.

MPAS mpas_atmphys_driver_cloudiness.F:210 and radiation_lw/sw.F:865/919.
Tar/zstd extract records; native Fortran reads them; Rust compares every word.
No forecast, expected-output seed, spread allowance or hidden-state closure.
"""
from hexcore import site_runner
import argparse
import hashlib
import importlib.util
import json
import socket
from pathlib import Path
import subprocess
import sys
from replay.generate_replay import generate
from replay.prepare_native import prepare


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def check(checkout,profile,out,reuse_inputs=None):
    p=json.loads(profile.read_text());assert p['process_count']==1
    if not p['captures'] or len(set(p['captures']))!=len(p['captures']):
        raise ValueError('real replay must select distinct observed native captures')
    checkout=checkout.resolve();out=out.resolve();out.mkdir()
    dirty=subprocess.check_output(['git','-C',str(checkout),'status','--porcelain'],text=True)
    if dirty:raise ValueError('real-input receipt requires a clean candidate checkout')
    for filename,digest in p['input_sha256'].items():
        if sha(Path(filename))!=digest:raise ValueError('real replay dependency differs: '+filename)
    build=Path(p['build']);inventory=json.loads((build/'oracle-inventory.json').read_text())
    here=checkout/'tree/tools/native_physics_reference/oracle'
    # A historical dump is replayed with its original compiled schema. A newer
    # hidden-state helper cannot read data which that capture never recorded.
    # Bind the candidate's ABI/decoder/replay generator separately; retain the
    # original native generator bindings rather than relabeling its artifact.
    compatible=('replay/generate_replay.py','replay/prepare_native.py',
                'mpas_oracle_dump.F','oracle_words.rs')
    for name in compatible:
        digest=inventory['generator_sha256'][name]
        if sha(here/name)!=digest:raise ValueError('candidate oracle generator differs from compiled native source: '+name)
    module=checkout/'tree/tools/column_oracle_replay.py'
    spec=importlib.util.spec_from_file_location('candidate_replay',module)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    parts=[Path(x)for x in p['archive_parts']]
    # Authenticate the concatenated immutable container without a second copy.
    digest=hashlib.sha256()
    for part in parts:
        with part.open('rb') as f:
            while block:=f.read(1048576):digest.update(block)
    if digest.hexdigest()!=p['archive_sha256']:raise ValueError('native input archive differs')
    if reuse_inputs is not None:
        raw=reuse_inputs.resolve();subset=h.verify(raw)
        if (subset['dump_root_sha256']!=p['reuse_subset_dump_root_sha256'] or
            subset.get('parent_dump_root_sha256')!=p['parent_dump_root_sha256'] or
            subset.get('parent_archive_sha256')!=p['archive_sha256'] or
            set(subset['captures'])!=set(p['captures'])):
            raise ValueError('authenticated reused native subset differs')
        parent=subset;files=subset['files']
    else:
        raw=out/'inputs';raw.mkdir()
        subset,parent,files=extract_inputs(p,parts,raw,out,h)
    sites={s['site']:s for s in parent['sites']};sources=out/'sources';sources.mkdir()
    for prefix in p['captures']:
        site=sites[parent['captures'][prefix]['site']]
        (sources/('replay_'+site['site']+'.F90')).write_text(generate(site))
    native=out/'native';metadata=prepare(build,Path(p['build_log']),native,sources)
    with(out/'build.log').open('w')as log:subprocess.run(site_runner.build_command(['make','-j4','-C',str(native)]),stdout=log,stderr=subprocess.STDOUT,check=True)
    rows=[]
    for number,prefix in enumerate(p['captures']):
        site=parent['captures'][prefix]['site'];target=out/(str(number)+'.json');results=out/(str(number)+'-results')
        args=argparse.Namespace(dump=raw,scheme=site,backend='native-replay',recipe='gnu-strict',results=results,
             receipt=target,executable=native/('replay_'+site),words=Path(p['words']),coverage=None,gpu_slot_minutes=None,
             process_count=1,cell_id=None,capture=prefix,whole_capture=False)
        rc=h.replay(args,subset);receipt=json.loads(target.read_text());arrays=receipt['captures'][0]['arrays']
        row=dict(capture=prefix,process_count=1,returncode=rc,receipt_sha256=sha(target),
             words=sum(a['words']for a in arrays.values()),different_words=sum(a['different_words']for a in arrays.values()))
        rows.append(row)
        if rc==0:
            for name,entry in arrays.items():
                f=results/(prefix+'.post.'+name+'.bin')
                if sha(f)!=entry['replay_sha256']:raise ValueError('computed native record changed before retirement')
                f.unlink()
        else:raise ValueError('native real-input replay differs')
    receipt=dict(process_count=1,execution_host=socket.gethostname(),profile_sha256=sha(profile),candidate_commit=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip(),
        parent_dump_root_sha256=p['parent_dump_root_sha256'],subset_dump_root_sha256=subset['dump_root_sha256'],
        candidate_replay_source_sha256=sha(module),real_input_checker_source_sha256=sha(Path(__file__)),
        pin_set_sha256=parent['pin_set_sha256'],native_build=metadata,captures=rows,all_bitwise_equal=True,
        native_artifact_generator_sha256=inventory['generator_sha256'],
        candidate_compatible_generator_sha256={name:sha(here/name)for name in compatible},
        scope='Candidate replay orchestration and unchanged transport ABI on original captured-schema native libraries; new timer/stream helpers require their separately bound native checks and fresh captures',
        global_hidden_state_complete=False,compiler_spread_proof_bar=False,forecast_attempted=False)
    (out/'summary.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if reuse_inputs is None:
        for name,entry in files.items():
            f=raw/name
            if sha(f)!=entry['sha256']:raise ValueError('input changed before retirement')
            f.unlink()
    print(json.dumps(dict(process_count=1,words=sum(r['words']for r in rows),different_words=0)))


def extract_inputs(p,parts,raw,out,h):
    patterns=['./manifest.json']+[f'./{prefix}.*.bin'for prefix in p['captures']]
    with (out/'extract.log').open('w')as log:
        cat=subprocess.Popen(['cat',*map(str,parts)],stdout=subprocess.PIPE)
        decode=subprocess.Popen(['zstd','-dc','--long=31'],stdin=cat.stdout,stdout=subprocess.PIPE,stderr=log);cat.stdout.close()
        extract=subprocess.run(['tar','-xf','-','-C',str(raw),'--wildcards',*patterns],stdin=decode.stdout,stderr=log)
        decode.stdout.close()
        if extract.returncode or decode.wait() or cat.wait():raise ValueError('native archive extraction failed')
    original=raw/'manifest.json';parent=json.loads(original.read_text())
    if parent.get('process_count')!=1:raise ValueError('native capture must use one process')
    if parent['dump_root_sha256']!=p['parent_dump_root_sha256']:raise ValueError('native parent root differs')
    body={k:v for k,v in parent.items()if k!='dump_root_sha256'}
    if hashlib.sha256(h.canonical(body)).hexdigest()!=parent['dump_root_sha256']:raise ValueError('native parent manifest differs')
    files={f.name:parent['files'][f.name]for f in raw.glob('*.bin')}
    for name,info in files.items():
        if sha(raw/name)!=info['sha256']or (raw/name).stat().st_size!=info['bytes']:raise ValueError('native record differs')
    original.unlink()
    subset=dict(body,files=files,captures={k:parent['captures'][k]for k in p['captures']},
                parent_dump_root_sha256=parent['dump_root_sha256'],parent_archive_sha256=p['archive_sha256'])
    subset['dump_root_sha256']=hashlib.sha256(h.canonical(subset)).hexdigest()
    (raw/'manifest.json').write_text(json.dumps(subset,indent=2)+'\n');h.verify(raw)
    return subset,parent,files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkout','profile','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reuse-inputs',type=Path,help='authenticate and reuse an already extracted immutable subset')
    a=parser.parse_args();check(a.checkout,a.profile,a.out,a.reuse_inputs)


if __name__=='__main__':main()
