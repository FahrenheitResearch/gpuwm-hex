#!/usr/bin/env python3
"""Summarize GNU gcov branch counters, retaining uncovered source locations."""
import argparse
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess


def allocation_control_mapping(old,new,mapping):
    """Map source-verified single-line conditional ALLOCATE controls.

    allocation_state.patch_allocations expands native IF/ALLOCATE into a block
    to observe allocator entry storage. Only the original condition and original
    ALLOCATE map back; the observation hook and generated ENDIF stay excluded.
    """
    candidates={}
    existing_native=set(mapping.values())
    normalize=lambda value:re.sub(r'\s+','',value).lower()
    for index,line in enumerate(new):
        tag=re.fullmatch(r"\s*call\s+oracle_allocation\('([a-z0-9_.]+\.L(\d+))',\s*&\s*",line,re.I)
        if not tag or index<2:continue
        native=int(tag[2])
        if native<1 or native>len(old) or native in existing_native:continue
        statement=old[native-1].split('!',1)[0]
        source=re.fullmatch(r'\s*(if\s*\(\s*\.not\.\s*allocated\s*\(\s*(\w+)\s*\)\s*\))\s*(allocate\s*\(.*\))\s*',statement,re.I)
        if not source or tag[1].split('.')[-2].lower()!=source[2].lower():continue
        # Bounds in this conservative native form contain no character literal.
        if any(c in source[3] for c in "'\""):continue
        if normalize(new[index-2])!=normalize(source[1]+' then'):continue
        if normalize(new[index-1])!=normalize(source[3]):continue
        candidates.setdefault(native,[]).append(dict(native_line=native,
            generated_condition_line=index-1,generated_allocate_line=index,tag=tag[1]))
    verified=[]
    for native,rows in sorted(candidates.items()):
        if len(rows)!=1:continue
        row=rows[0]
        if row['generated_condition_line'] in mapping or row['generated_allocate_line'] in mapping:continue
        mapping[row['generated_condition_line']]=native
        mapping[row['generated_allocate_line']]=native
        verified.append(row)
    return verified


def summarize(data, source_root=None, build_root=None, compile_root=None, generated_source_sha256=None, *, compiled_build_root=None, include_counters=False, map_allocation_controls=False, retain_unmapped_sources=False):
    if compiled_build_root is not None:
        if compile_root is not None and Path(compile_root).resolve()!=Path(compiled_build_root).resolve():
            raise ValueError('conflicting compilation roots')
        compile_root=compiled_build_root
    if bool(source_root) != bool(build_root):
        raise ValueError('source-root and build-root must be supplied together')
    if build_root:
        build_root = Path(build_root).resolve()
        source_root = Path(source_root).resolve()
        working = Path(data.get('current_working_directory', build_root)).resolve()
    if compile_root is not None and not build_root:
        raise ValueError('compile-root requires source-root and build-root')
    files=[];excluded=[];unmapped=[]
    for f in data['files']:
        lines=f['lines']
        extra={}
        if source_root:
            generated=Path(f['file'])
            if compile_root is not None:
                # GNU records the original compilation path, which may be on
                # another host or precede an owned build-directory relocation.
                # Map its source suffix onto the authenticated returned tree.
                original_working=Path(data.get('current_working_directory',compile_root))
                if not generated.is_absolute():generated=original_working/generated
                try:relative=generated.relative_to(Path(compile_root))
                except ValueError:
                    if not retain_unmapped_sources:raise
                    unmapped.append({'source':f['file'],'reason':'source outside owned build root'})
                    continue
                if '..' in relative.parts:raise ValueError('gcov source escapes compile root')
                generated=(build_root/relative).resolve()
                generated.relative_to(build_root)
            else:
                if not generated.is_absolute():generated=working/generated
                generated=generated.resolve()
                relative=generated.relative_to(build_root)
            original=source_root/relative
            generated_only=(relative.name in {'mpas_oracle_dump.F','mpas_oracle_pool.F',
                'mpas_oracle_pool_restore.F','framework_log_stub.F90','replay.F90'} or
                relative.name.startswith(('capture_','replay-')))
            if not original.is_file() and generated_source_sha256 is not None and not generated_only:
                # Native package Makefiles copy wrappers before compiling.
                # Admit only a unique byte-identical copy authenticated by the
                # source inventory; basename similarity alone is insufficient.
                digest=hashlib.sha256(generated.read_bytes()).hexdigest()
                candidates=[]
                for name,expected in generated_source_sha256.items():
                    candidate=Path(name)
                    if candidate.is_absolute() or '..' in candidate.parts:
                        raise ValueError('invalid inventoried coverage source path')
                    if candidate.name!=relative.name or expected!=digest:continue
                    if not (source_root/candidate).is_file():continue
                    if hashlib.sha256((build_root/candidate).read_bytes()).hexdigest()!=expected:
                        raise ValueError('inventoried coverage source bytes differ')
                    candidates.append(candidate)
                if len(candidates)!=1:
                    raise ValueError('compiled source has no unique authenticated native copy')
                extra['compiled_source']=str(relative)
                extra['compiled_copy_sha256']=digest
                relative=candidates[0];original=source_root/relative
            if not original.is_file():
                if retain_unmapped_sources:
                    if not generated.is_file():raise ValueError(f'unmapped generated coverage source missing: {generated}')
                    unmapped.append({'source':f['file'],'reason':'source absent from pristine tree','instrumented_source_sha256':hashlib.sha256(generated.read_bytes()).hexdigest(),'generated_source':str(relative),'native_branches_counted':0})
                    continue
                if generated_only:
                    excluded.append({'source':str(generated),'reason':'generated oracle/harness source',
                                     'sha256':hashlib.sha256(generated.read_bytes()).hexdigest()})
                    continue
                raise ValueError(f'no pristine native source for {generated}')
            old=original.read_text().splitlines();new=generated.read_text().splitlines()
            mapping={j+k+1:i+k+1 for i,j,n in difflib.SequenceMatcher(None,old,new,autojunk=False).get_matching_blocks() for k in range(n)}
            allocations=allocation_control_mapping(old,new,mapping) if map_allocation_controls else []
            mapped_original=set(mapping.values())
            mapped=[dict(l,line_number=mapping[l['line_number']]) for l in lines if l['line_number'] in mapping]
            extra={**extra,'source_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
                   'instrumented_source_sha256':hashlib.sha256(generated.read_bytes()).hexdigest(),
                   'instrumentation_lines_excluded':len(lines)-len(mapped),
                   'native_source':str(relative),
                   'unmapped_native_lines':[i+1 for i,line in enumerate(old)
                                            if line.strip() and i+1 not in mapped_original]}
            if map_allocation_controls:
                extra.update(allocation_control_mappings=allocations,source_mapping='exact-lines-plus-native-allocation-controls-v1')
            lines=mapped
        branches=[(l['line_number'],b) for l in lines for b in l.get('branches',[])]
        if include_counters:
            extra['line_counters']=[dict(line=l['line_number'],function=l.get('function_name',''),count=l['count'],branches=l.get('branches',[])) for l in lines]
        files.append({'source':f['file'],'lines_total':len(lines),'lines_executed':sum(l['count']>0 for l in lines),
                      'branches_total':len(branches),'branches_taken':sum(b['count']>0 for _,b in branches),
                      'uncovered_lines':[l['line_number'] for l in lines if not l['count']],
                      'uncovered_branches':[{'line':n,'branch':b} for n,b in branches if not b['count']],**extra})
    if source_root and not files and not (retain_unmapped_sources and unmapped):
        raise ValueError('no pristine native source was mapped from gcov')
    result={'schema':'mpas-oracle-gcov-v1','gcc_version':data['gcc_version'],'files':files,'excluded_generated_sources':excluded,'unmapped_files':unmapped}
    if retain_unmapped_sources:
        result.update(retain_unmapped_sources=True,native_branch_coverage_proven=bool(files),scope='Owned generated sources are authenticated separately and contribute no native branches.')
    return result



def validate_object_sources(report, primary, source_root, build_root, source_hashes):
    """Bind extra gcov files to unchanged native INCLUDEs of this object.

    GNU emits separate rows for preprocessor/Fortran include files. They are
    native branches, so retain their counters instead of dropping the rows or
    accepting arbitrary extra sources merely because a gcov object names them.
    """
    source_root, build_root = Path(source_root).resolve(), Path(build_root).resolve()
    primary = Path(primary)
    if primary.is_absolute() or '..' in primary.parts:
        raise ValueError('native coverage primary source escapes root')
    expected_primary = source_hashes.get(str(primary))
    if not expected_primary or hashlib.sha256((source_root / primary).read_bytes()).hexdigest() != expected_primary:
        raise ValueError('native coverage primary source changed')
    included = {}
    visited = set()
    pattern = re.compile(r'^\s*(?:#\s*include|include)\s*["\']([^"\']+)["\']', re.I | re.M)

    def visit(relative):
        if relative in visited:
            return
        visited.add(relative)
        original = source_root / relative
        for name in pattern.findall(original.read_text()):
            candidate = relative.parent / name
            # Only the actual local include relationship can authorize an extra
            # report row. External/compiler headers remain outside this graph.
            if '..' in candidate.parts or candidate.is_absolute():
                raise ValueError('native coverage include escapes source directory')
            native = source_root / candidate
            if not native.is_file():
                continue
            generated = build_root / candidate
            digest = hashlib.sha256(native.read_bytes()).hexdigest()
            expected = source_hashes.get(str(candidate))
            if expected is None:
                top = Path(subprocess.check_output(
                    ['git', '-C', str(native.parent), 'rev-parse', '--show-toplevel'],
                    text=True).strip())
                tracked = subprocess.check_output(
                    ['git', '-C', str(top), 'show', 'HEAD:' + str(native.relative_to(top))])
                expected = hashlib.sha256(tracked).hexdigest()
            if digest != expected or not generated.is_file() or hashlib.sha256(generated.read_bytes()).hexdigest() != expected:
                raise ValueError('native coverage included source changed: ' + str(candidate))
            included[str(candidate)] = dict(source_sha256=expected,
                                             compiled_source_sha256=expected)
            visit(candidate)

    visit(primary)
    names = [row['native_source'] for row in report['files']]
    if str(primary) not in names or len(names) != len(set(names)):
        raise ValueError('coverage object primary source is absent or duplicated')
    for row in report['files']:
        name = row['native_source']
        if name == str(primary):
            continue
        if name not in included:
            raise ValueError('coverage notes map to a different native source')
        row.update(included_by_native_object=str(primary),
                   include_authority=included[name])
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gcov',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-root',type=Path,help='pinned pristine MPAS root; excludes generated instrumentation branches')
    p.add_argument('--build-root',type=Path,help='standalone compilation directory, for source line mapping')
    p.add_argument('--compiled-build-root',type=Path,help='original compiler directory when build products were relocated')
    p.add_argument('--inventory',type=Path,help='bind native Makefile wrapper copies to generated source hashes')
    a=p.parse_args()
    data=json.loads(gzip.decompress(a.gcov.read_bytes()))
    inventory=json.loads(a.inventory.read_text()) if a.inventory else None
    report=summarize(data,a.source_root,a.build_root,a.compiled_build_root,
                     inventory['generated_sha256'] if inventory else None)
    report['gcov_sha256']=hashlib.sha256(a.gcov.read_bytes()).hexdigest()
    a.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
