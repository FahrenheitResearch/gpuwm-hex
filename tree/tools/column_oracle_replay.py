#!/usr/bin/env python3
"""Execute real scheme replay and emit exact-word debugging receipts.

Weather-field decoding, comparison and extraction run in oracle_words (Rust).
A passing receipt does not establish a forecast or scheme hidden-state proof.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

from hexcore import site_runner


def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


def write_replay_receipt(path,receipt,limit=9_000_000):
    """Preserve every comparison while keeping each delivery file below its cap."""
    encode=lambda value:(json.dumps(value,separators=(',',':'))+'\n').encode()
    captures=receipt['captures']
    base=dict(receipt,captures=[])
    size=len(encode(base))+sum(len(encode(row)) for row in captures)
    if size<=limit:
        path.write_bytes(encode(receipt))
        return
    shards=[];pending=[];current=None;used=128
    def flush():
        nonlocal pending,current,used
        if not pending:return
        payload=encode({'schema':'mpas-column-replay-captures-v1','captures':pending})
        if len(payload)>limit:raise ValueError('native replay comparison shard exceeds receipt cap')
        part=path.with_name(f'{path.stem}.captures-{len(shards)+1:04d}.json')
        with part.open('xb') as stream:stream.write(payload)
        shards.append({'file':part.name,'sha256':hash_file(part),'bytes':len(payload),
                       'arrays':sum(len(row['arrays']) for row in pending)})
        pending=[];current=None;used=128
    for row in captures:
        overhead=len(encode(dict(row,arrays={})))+32
        if not row['arrays']:
            if used+overhead>limit:flush()
            if overhead+128>limit:raise ValueError('native replay capture exceeds receipt cap')
            current=dict(row,arrays={});pending.append(current);used+=overhead
            continue
        for name,value in row['arrays'].items():
            member=len(encode({name:value}))+8
            if overhead+member+128>limit:
                raise ValueError('one native replay comparison exceeds receipt cap')
            if used+overhead+member>limit:flush()
            if current is None or current['capture']!=row['capture']:
                current=dict(row,arrays={});pending.append(current);used+=overhead
            current['arrays'][name]=value;used+=member
    flush()
    index=dict(base,schema='mpas-column-replay-v2',capture_count=len(captures),
               array_count=sum(len(row['arrays']) for row in captures),capture_shards=shards)
    payload=encode(index)
    if len(payload)>limit:raise ValueError('native replay comparison index exceeds receipt cap')
    path.write_bytes(payload)


def read_replay_receipt(path):
    """Authenticate and reconstruct bounded receipts for downstream evidence."""
    receipt=json.loads(path.read_text())
    if receipt.get('schema')!='mpas-column-replay-v2':return receipt
    captures={};parts=set()
    for part in receipt['capture_shards']:
        name=part['file']
        if Path(name).name!=name or name in parts:
            raise ValueError('invalid native replay comparison shard path')
        parts.add(name);source=safe_file(path.parent,name)
        if source.stat().st_size!=part['bytes'] or hash_file(source)!=part['sha256']:
            raise ValueError('native replay comparison shard hash mismatch')
        rows=json.loads(source.read_text())
        if rows['schema']!='mpas-column-replay-captures-v1':
            raise ValueError('invalid native replay comparison shard schema')
        if sum(len(row['arrays']) for row in rows['captures'])!=part['arrays']:
            raise ValueError('native replay comparison shard count differs')
        for row in rows['captures']:
            target=captures.setdefault(row['capture'],dict(row,arrays={}))
            if target['site']!=row['site'] or target['arrays'].keys() & row['arrays'].keys():
                raise ValueError('duplicate or incompatible native replay comparisons')
            target['arrays'].update(row['arrays'])
    if len(captures)!=receipt['capture_count'] or sum(len(r['arrays']) for r in captures.values())!=receipt['array_count']:
        raise ValueError('native replay comparison index count differs')
    receipt['captures']=list(captures.values())
    return receipt


def words(executable,*args):
    p=subprocess.run([str(executable),*map(str,args)],capture_output=True,text=True,check=True)
    return json.loads(p.stdout) if p.stdout else None


def native_resource_limits():
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    return {'parent_stack_soft_bytes': soft, 'parent_stack_hard_bytes': hard,
            'child_stack_soft_bytes': hard, 'child_stack_hard_bytes': hard}


def run_native_replay(command):
    # Intel's generated reshape/transfer temporary in read_r8_4 exhausted the
    # 8 MiB process stack when restoring Thompson tables. The unchanged binary
    # completed all 13 real calls with the hard stack limit. Change only the
    # replay child: native source, compiler flags and the parent are untouched.
    hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
    return subprocess.run(command, capture_output=True, text=True,
                          preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_STACK, (hard, hard)))


def safe_file(root,name):
    # Oracle records are flat members. Reject traversal lexically before
    # inspecting the member; resolving every ancestor per record makes large
    # driver-pool manifests spend most of their time repeating directory I/O.
    if not isinstance(name,str) or not name or name in {'.','..'} or Path(name).name!=name:
        raise ValueError(f'invalid dump member: {name}')
    p=root/name
    if p.is_symlink() or not p.is_file():
        raise ValueError(f'invalid dump member: {name}')
    return p


def seal(root,inventory,pins,helper,recipe,process_count=None,reference=None):
    if (root/'manifest.json').exists():raise ValueError('refusing to replace sealed manifest')
    sources=json.loads(inventory.read_text())
    validate_reference(sources,reference)
    bins=sorted(root.glob('*.bin'))
    if not bins:raise ValueError('dump root has no records')
    if sum(p.stat().st_size for p in bins)>100*1000**3:raise ValueError('dump exceeds campaign 100 GB cap')
    files={p.name:{'sha256':hash_file(p),'bytes':p.stat().st_size} for p in bins}
    captures={}
    for p in bins:
        if '.pre.' not in p.name and '.post.' not in p.name:continue
        prefix=p.name.split('.pre.')[0].split('.post.')[0]
        if prefix not in captures:
            meta=words(helper,'header',p,'-')
            captures[prefix]={'site':meta['key']}
    pintext=pins.read_text()
    pins_hash=hashlib.sha256(pintext.encode()).hexdigest()
    # A fork inventory authenticates its model and every submodule. The path
    # selecting its dumps must use that source identity, rather than the hash
    # of a separately formatted explanatory pins document.
    source_pins=sources.get('pin_set_sha256') if sources.get('reference')=='rrfs' else None
    if source_pins is not None and (not isinstance(source_pins,str) or len(source_pins)!=64
                                  or any(c not in '0123456789abcdef' for c in source_pins)):
        raise ValueError('invalid authenticated source pin-set hash')
    manifest={'schema':'mpas-column-dump-v1','reference':sources.get('reference','mpas'),'files':files,'captures':captures,
              'sites':sources['sites'],'source_sha256':sources['source_sha256'],
              'scheme_aliases':sources.get('scheme_aliases',{}),
              'scheme_groups':sources.get('scheme_groups',{}),
              'pin_set_sha256':source_pins or pins_hash,'pins':pintext,
              'pins_document_sha256':pins_hash,
              'recipe':recipe,'inventory_sha256':hash_file(inventory),
              'argument_complete':sources['argument_complete'],
              'hidden_state_complete':sources['hidden_state_complete'],'forecast_proven':False}
    if process_count is not None:
        if process_count<1:raise ValueError('capture process count must be positive')
        manifest['process_count']=process_count
    manifest['build_provenance']={k:sources[k] for k in ('generated_sha256','generator_sha256',
                                'instrumentation_sha256','strict_reference_manifest_sha256',
                                'strict_reference_binary_sha256') if k in sources}
    if 'input_provenance' in sources:manifest['input_provenance']=sources['input_provenance']
    if 'boundaries' in sources:manifest['boundaries']=sources['boundaries']
    manifest['dump_root_sha256']=hashlib.sha256(canonical(manifest)).hexdigest()
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def verify(root):
    manifest=json.loads((root/'manifest.json').read_text())
    expected=manifest.pop('dump_root_sha256')
    actual=hashlib.sha256(canonical(manifest)).hexdigest()
    if actual!=expected:raise ValueError('dump manifest root hash mismatch')
    manifest['dump_root_sha256']=expected
    actual_members={p.name for p in root.glob('*.bin')}
    if actual_members!=set(manifest['files']):raise ValueError('dump member set differs from sealed manifest')
    for name,info in manifest['files'].items():
        p=safe_file(root,name)
        if p.stat().st_size!=info['bytes'] or hash_file(p)!=info['sha256']:
            raise ValueError(f'dump member hash mismatch: {name}')
    return manifest


def validate_reference(manifest,requested):
    actual=manifest.get('reference','mpas')
    if actual not in {'mpas','rrfs'}:raise ValueError(f'unknown sealed reference: {actual}')
    if requested is not None and actual!=requested:
        raise ValueError(f'reference mismatch: requested {requested}, captured {actual}')
    return actual


ALIAS_CATALOG = Path(__file__).parent/'native_physics_reference/oracle/scheme_aliases.json'


def alias_contract(manifest):
    """Bind public lookup names separately from immutable native captures."""
    sealed=manifest.get('scheme_aliases',{})
    catalog=json.loads(ALIAS_CATALOG.read_text())
    for table in (sealed,catalog):
        if not isinstance(table,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in table.items()):
            raise ValueError('invalid scheme alias table')
    # Existing sealed definitions retain their original meaning. A newer public
    # lookup name changes selection only; no captured member or root is resealed.
    return {**catalog,**sealed}, {'catalog_sha256':hash_file(ALIAS_CATALOG),
        'catalog_aliases':catalog,'sealed_aliases_sha256':hashlib.sha256(canonical(sealed)).hexdigest(),
        'precedence':'sealed definitions before current public lookup catalog',
        'capture_manifest_changed':False}


def resolve_scheme(manifest,requested,alias_context=None):
    """Resolve source-bound public lookup names without changing call-site keys."""
    aliases,_=alias_contract(manifest) if alias_context is None else alias_context
    if not isinstance(aliases,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in aliases.items()):
        raise ValueError('invalid sealed scheme alias table')
    seen=set();resolved=requested
    while resolved in aliases:
        if resolved in seen:raise ValueError('cyclic sealed scheme alias table')
        seen.add(resolved);resolved=aliases[resolved]
    return resolved


def selected_schemes(manifest,requested,alias_context=None):
    """Expand a sealed public group into native scheme names, without special paths."""
    alias_context=alias_contract(manifest) if alias_context is None else alias_context
    resolved=resolve_scheme(manifest,requested,alias_context)
    groups=manifest.get('scheme_groups',{})
    if not isinstance(groups,dict) or any(not isinstance(k,str) or not isinstance(v,list)
            or not v or any(not isinstance(member,str) for member in v) for k,v in groups.items()):
        raise ValueError('invalid sealed scheme group table')
    result=set()
    def expand(name,parents):
        name=resolve_scheme(manifest,name,alias_context)
        if name in parents:raise ValueError('cyclic sealed scheme group table')
        if name in groups:
            for member in groups[name]:expand(member,parents|{name})
        else:result.add(name)
    expand(resolved,set())
    return result


def compare_record(a,prefix,name):
    filename=prefix+'.post.'+name+'.bin'
    src=safe_file(getattr(a,'reference_dump',a.dump),filename);candidate=safe_file(a.results,filename)
    result=words(a.words,'compare',src,candidate)
    result.update(reference_sha256=hash_file(src),replay_sha256=hash_file(candidate))
    location=result['first_difference']
    if location and location['cellID'] is not None:
        # Each capture/cell has the same inputs for every differing output.
        # Reuse its extract rather than copying whole hidden tables per array.
        repro=a.results/(prefix+'.repro.cell'+str(location['cellID']))
        cache=vars(a).setdefault('_column_diagnostics',{})
        key=(prefix,location['cellID'])
        if key not in cache:
            try:
                words(a.words,'column-capture',a.dump,prefix,location['cellID'],repro,'pre')
                (repro/'location.json').write_text(json.dumps(location,indent=2)+'\n')
                cache[key]={'column_inputs':str(repro)}
            except (OSError,ValueError,subprocess.CalledProcessError) as error:
                # RUC hidden state can have a different cell axis from its
                # argument records. A refused diagnostic extraction must not
                # discard the actual Rust word comparison or later arrays.
                detail={'reason':str(error),'complete':False}
                if isinstance(error,subprocess.CalledProcessError):
                    detail.update(exit=error.returncode,stderr=(error.stderr or '')[:16000])
                cache[key]={'column_inputs_error':detail}
        result.update(cache[key])
    return result


def extract_column(a,manifest,prefix,cell_id):
    """Keep native inputs separate from expected outputs; Rust crops the words."""
    inputs=a.results/'column-inputs'/prefix
    reference=a.results/'column-reference'/prefix
    inputs.parent.mkdir(parents=True,exist_ok=True);reference.parent.mkdir(parents=True,exist_ok=True)
    words(a.words,'column-capture',a.dump,prefix,cell_id,inputs,'pre')
    words(a.words,'column-capture',a.dump,prefix,cell_id,reference,'post')
    members={}
    for name,info in manifest['files'].items():
        if not name.startswith((prefix+'.pre.',prefix+'.post.')):continue
        destination=(inputs if name.startswith(prefix+'.pre.') else reference)/name
        members[name]={'source_sha256':info['sha256'],
                       'column_sha256':hash_file(destination)}
    return inputs,reference,members


def post_index(names):
    """Index record filenames once; payload decoding stays in Rust."""
    indexed={}
    for name in names:
        if '.post.' not in name or not name.endswith('.bin'):continue
        prefix,leaf=name.split('.post.',1)
        indexed.setdefault(prefix,set()).add(leaf[:-4])
    return indexed


def replay(a,verified_manifest=None,scheme_alias_context=None):
    # Generated standalone native replays are invoked directly, without MPI.
    # Reject misleading metadata before executing or creating evidence.
    if a.backend=='native-replay' and getattr(a,'process_count',1)!=1:
        raise ValueError('standalone native replay invokes exactly one process; --process-count must be 1')
    manifest=verify(a.dump) if verified_manifest is None else verified_manifest
    reference=validate_reference(manifest,getattr(a,'ref',None))
    sites={s['site']:s for s in manifest['sites']}
    alias_context=alias_contract(manifest) if scheme_alias_context is None else scheme_alias_context
    schemes=selected_schemes(manifest,a.scheme,alias_context)
    selected=[(p,sites[c['site']]) for p,c in manifest['captures'].items()
              if c['site'] in sites and (sites[c['site']]['scheme'] in schemes or c['site'] in schemes)]
    capture=getattr(a,'capture',None)
    if capture is not None:selected=[item for item in selected if item[0]==capture]
    if not selected:raise ValueError(f'no captures for scheme {a.scheme}')
    if a.backend=='hex-cuda':
        if not a.gpu_slot_minutes:raise ValueError('hex-cuda replay requires --gpu-slot-minutes (1 through 10)')
    if a.results.exists():raise ValueError('results must be a fresh directory; prevents stale replay evidence')
    whole_capture=getattr(a,'whole_capture',False)
    cell_id=getattr(a,'cell_id',None)
    if cell_id is not None and (a.backend!='native-replay' or whole_capture):
        raise ValueError('source-bound column mode requires a per-capture native replay executable')
    if whole_capture:
        if len({site['site'] for _,site in selected})!=1 or len(selected)!=len(manifest['captures']):
            raise ValueError('whole-capture executable requires a dump containing exactly the selected call site')
        command=[str(a.executable.resolve()),str(a.dump.resolve()),str(a.results.resolve())]
        if a.backend=='hex-cuda':command=site_runner.gpu_command(command,a.gpu_slot_minutes)
        run=(run_native_replay(command) if a.backend=='native-replay'
             else subprocess.run(command,capture_output=True,text=True))
        a.results.mkdir(parents=True,exist_ok=True)
        (a.results/'whole-capture.log').write_text(run.stdout+run.stderr)
        if run.returncode:raise ValueError(f'replay failed ({run.returncode}): {a.results/"whole-capture.log"}')
    else:
        a.results.mkdir(parents=True)
    reference_posts=post_index(manifest['files'])
    candidate_posts=post_index(p.name for p in a.results.glob('*.post.*.bin')) if whole_capture else None
    rows=[]
    for prefix,site in selected:
        comparison=a
        column_members=None
        if not whole_capture:
            inputs=a.dump
            if cell_id is not None:
                inputs,column_reference,column_members=extract_column(a,manifest,prefix,cell_id)
                comparison=argparse.Namespace(**vars(a));comparison.reference_dump=column_reference
            command=[str(a.executable.resolve()),str((inputs/prefix).resolve()),str((a.results/prefix).resolve())]
            if cell_id is not None:command.append('column')
            if a.backend=='hex-cuda':command=site_runner.gpu_command(command,a.gpu_slot_minutes)
            run=(run_native_replay(command) if a.backend=='native-replay'
                 else subprocess.run(command,capture_output=True,text=True))
            (a.results/(prefix+'.log')).write_text(run.stdout+run.stderr)
            if run.returncode:raise ValueError(f'replay failed ({run.returncode}): {a.results/(prefix+".log")}')
        row={'capture':prefix,'site':site['site'],'arrays':{}}
        if column_members is not None:
            row['column']={'cell_id':cell_id,'inputs':str(inputs),'members':column_members}
        expected=set()
        arguments=[leaf for parent in site['arguments'] for leaf in parent.get('components',[parent])]
        pool_prefixes=tuple(('argument.' if arg.get('native_pool') else '')+arg['name']+'.'
                            for arg in arguments if arg.get('native_pool') or arg.get('pool_argument'))
        hidden_names={name for name in reference_posts.get(prefix,set()) if name.startswith(('state.','local.',*pool_prefixes))}
        for arg in arguments:
            if arg['intent']=='in' or arg.get('native_pool') or arg.get('pool_argument'):continue
            name=prefix+'.post.'+arg['name']+'.bin'
            if arg['optional'] or arg.get('actual_optional',False):
                flag=prefix+'.post.'+arg['name']+'.present.bin'
                expected.add(flag)
                srcflag=safe_file(getattr(comparison,'reference_dump',a.dump),flag);outflag=safe_file(a.results,flag)
                flagresult=words(a.words,'compare',srcflag,outflag)
                flagresult.update(reference_sha256=hash_file(srcflag),replay_sha256=hash_file(outflag))
                row['arrays'][arg['name']+'.present']=flagresult
                src_present=words(a.words,'present',srcflag,'-');out_present=words(a.words,'present',outflag,'-')
                if out_present:expected.add(name)
                if not src_present or not out_present:continue
            expected.add(name)
            row['arrays'][arg['name']]=compare_record(comparison,prefix,arg['name'])
        # Container pointers have presence/alias records and primitive children,
        # but no raw derived-object value. Compare the exact captured hidden
        # record set, including presence-only containers and absent arrays.
        for hidden in sorted(hidden_names):
            expected.add(prefix+'.post.'+hidden+'.bin')
            row['arrays'][hidden]=compare_record(comparison,prefix,hidden)
        actual=({prefix+'.post.'+name+'.bin' for name in candidate_posts.get(prefix,set())}
                if whole_capture else {p.name for p in a.results.glob(prefix+'.post.*.bin')})
        if actual!=expected:raise ValueError(f'replay output set differs: missing={expected-actual},extra={actual-expected}')
        rows.append(row)
    receipt={'schema':'mpas-column-replay-v1','reference':reference,'scheme':a.scheme,
             'resolved_scheme':next(iter(schemes)) if len(schemes)==1 else sorted(schemes),
             'resolved_schemes':sorted(schemes),'scheme_alias_contract':alias_context[1],'backend':a.backend,
             'dump_root_sha256':manifest['dump_root_sha256'],'pin_set_sha256':manifest['pin_set_sha256'],
             'source_sha256':manifest['source_sha256'],'capture_recipe':manifest['recipe'],
             'replay_recipe':a.recipe,
             'replay_process_count':getattr(a,'process_count',1),
             'capture_process_count':manifest.get('process_count'),
             'execution_mode':'column' if cell_id is not None else ('whole-capture' if whole_capture else 'per-capture'),
             'column_cell_id':cell_id,
             'executable_sha256':hash_file(a.executable),'word_helper_sha256':hash_file(a.words),
             'captures':rows,'all_outputs_bitwise_equal':all(v['bitwise_equal'] for r in rows for v in r['arrays'].values()),
             'comparison_complete':all(v.get('comparison_complete',True) for r in rows for v in r['arrays'].values()),
             'hidden_state_complete':manifest['hidden_state_complete'],'forecast_proven':False,
             'branch_coverage':None}
    if a.backend=='native-replay':receipt['replay_resource_limits']=native_resource_limits()
    receipt['build_provenance']=manifest.get('build_provenance',{})
    if 'input_provenance' in manifest:receipt['input_provenance']=manifest['input_provenance']
    if a.coverage:
        receipt['branch_coverage']={'sha256':hash_file(a.coverage),'report':json.loads(a.coverage.read_text())}
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    write_replay_receipt(a.receipt,receipt)
    print(json.dumps({'all_outputs_bitwise_equal':receipt['all_outputs_bitwise_equal'],'receipt':str(a.receipt)}))
    return 0 if receipt['all_outputs_bitwise_equal'] else 1


def failure_receipt(a,error,manifest=None):
    if not a.receipt or a.receipt.exists():return
    receipt={'schema':'mpas-column-replay-failure-v1','scheme':a.scheme,'backend':a.backend,
             'replay_process_count':getattr(a,'process_count',1),
             'all_outputs_bitwise_equal':False,'comparison_complete':False,'forecast_proven':False,
             'reason':str(error)}
    if manifest:
        receipt.update(dump_root_sha256=manifest['dump_root_sha256'],pin_set_sha256=manifest['pin_set_sha256'])
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')


def batch(a):
    """Authenticate an immutable capture once, then run its actual call sites."""
    manifest=verify(a.dump)
    reference=validate_reference(manifest,getattr(a,'ref',None))
    captured={c['site'] for c in manifest['captures'].values()}
    alias_context=alias_contract(manifest)
    schemes=selected_schemes(manifest,a.scheme,alias_context)
    sites=[s for s in manifest['sites'] if s['site'] in captured and
           ('all' in schemes or schemes.intersection({s['site'],s['scheme']}))]
    if not sites:raise ValueError(f'no captures for scheme {a.scheme}')
    if a.results.exists() or a.receipt.exists():raise ValueError('batch result/receipt directories must be fresh')
    a.results.mkdir(parents=True);a.receipt.mkdir(parents=True)
    root=a.executable_dir.resolve();rows=[]
    for site in sites:
        item=argparse.Namespace(**vars(a));item.scheme=site['site']
        item.results=a.results/site['site'];item.receipt=a.receipt/(site['site']+'.json')
        try:
            item.executable=(root/a.executable_pattern.format(site=site['site'],recipe=a.recipe)).resolve()
            if root not in item.executable.parents:raise ValueError('executable pattern escapes its root')
            if not item.executable.is_file():raise ValueError(f'missing replay executable: {item.executable}')
            status=replay(item,manifest,alias_context)
        except (OSError,ValueError,subprocess.CalledProcessError) as error:
            failure_receipt(item,error,manifest);status=2
        rows.append({'site':site['site'],'status':status,'receipt':str(item.receipt)})
    summary={'schema':'mpas-column-replay-batch-v1','reference':reference,'dump_root_sha256':manifest['dump_root_sha256'],
             'pin_set_sha256':manifest['pin_set_sha256'],'replay_recipe':a.recipe,
             'requested_scheme':a.scheme,'resolved_schemes':sorted(schemes),
             'scheme_alias_contract':alias_context[1],
             'replay_process_count':getattr(a,'process_count',1),'capture_process_count':manifest.get('process_count'),
             'all_outputs_bitwise_equal':all(r['status']==0 for r in rows),'sites':rows,'forecast_proven':False}
    (a.receipt/'batch.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'batch':str(a.receipt/'batch.json'),'all_outputs_bitwise_equal':summary['all_outputs_bitwise_equal']}))
    return 0 if summary['all_outputs_bitwise_equal'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scheme',required=True)
    p.add_argument('--ref',choices=['mpas','rrfs'],default='mpas',help='authenticate the sealed native source family')
    p.add_argument('--dump',type=Path,required=True)
    p.add_argument('--backend',choices=['native-replay','hex-cuda'],default='native-replay')
    p.add_argument('--executable',type=Path)
    p.add_argument('--whole-capture',action='store_true',help='execute a single-site dump root once, then independently compare every capture')
    p.add_argument('--capture',help='replay exactly this sealed capture prefix')
    p.add_argument('--cell-id',type=int,help='extract and replay one global cell through source-bound native column limits')
    p.add_argument('--batch',action='store_true',help='run all selected captured call sites after one dump-root authentication')
    p.add_argument('--executable-dir',type=Path)
    p.add_argument('--executable-pattern',default='replay_{site}',help='path below executable-dir, with {site} and {recipe}')
    p.add_argument('--words',type=Path,required=True)
    p.add_argument('--results',type=Path)
    p.add_argument('--receipt',type=Path)
    p.add_argument('--coverage',type=Path)
    p.add_argument('--gpu-slot-minutes',type=int,choices=range(1,11))
    p.add_argument('--seal',action='store_true')
    p.add_argument('--inventory',type=Path)
    p.add_argument('--pins',type=Path)
    p.add_argument('--recipe')
    p.add_argument('--process-count',type=int,default=1,help='fixed process count of the invoked replay executable')
    a=p.parse_args()
    if a.process_count<1:p.error('--process-count must be positive')
    if a.cell_id is not None and a.cell_id<1:p.error('--cell-id must be positive')
    try:
        if a.seal:
            if not all([a.inventory,a.pins,a.recipe]):p.error('--seal requires --inventory, --pins and --recipe')
            m=seal(a.dump,a.inventory,a.pins,a.words,a.recipe,a.process_count,a.ref)
            print(json.dumps({'dump_root_sha256':m['dump_root_sha256']}))
            return 0
        if not a.recipe:p.error('replay requires --recipe identifying the executable build')
        if a.batch:
            if not all([a.executable_dir,a.results,a.receipt]):p.error('--batch requires --executable-dir, --results and --receipt directories')
            return batch(a)
        if not all([a.executable,a.results,a.receipt]):p.error('replay requires --executable, --results and --receipt')
        return replay(a)
    except (OSError,ValueError,subprocess.CalledProcessError) as e:
        if not a.batch:failure_receipt(a,e)
        print(f'oracle: {e}',file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
