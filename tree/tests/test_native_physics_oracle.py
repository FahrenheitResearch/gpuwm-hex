"""Executable transport/replay checks; compilation always uses the shared slot."""
from hexcore import site_runner
import site_support
from pathlib import Path
import importlib.util
import json
import shutil
import os
import subprocess
import sys
import pytest

ORACLE=Path(__file__).resolve().parents[1]/'tools/native_physics_reference/oracle'
sys.path.insert(0,str(ORACLE))
from fortran_source import statements, routines, bind, active_source
from fortran_source import declarations
from state import modules
from generate_transport import generate as transport_source
from generate_pool import generate as pool_source
from replay.generate_replay import generate as replay_source


def _build_command(args):
    """site_runner.build_command, after checking the Fortran build toolchain is on PATH."""
    site_support.tools_or_skip("make", "gfortran", what="compiling this native-oracle fixture")
    return site_runner.build_command(args)


def test_coverage_matrix_preserves_function_identity_and_build_scope():
    from copy import deepcopy
    from replay.merge_coverage import merge
    branch = dict(count=0,source_block_id=2,destination_block_id=3,
                  fallthrough=True,throw=False)
    # The same source line and block IDs occur in two emitted functions.
    rows = [dict(line=8,function=name,count=0,branches=[dict(branch)])
            for name in ('first','second')]
    source = dict(native_source='native.F',source_sha256='a',
                  instrumented_source_sha256='b',line_counters=rows,
                  unmapped_native_lines=[])
    first = dict(process_count=1,gcc_version='15.2.0',notes_sha256='c',
                 files=[source],witness={'capture':'first'})
    second = deepcopy(first)
    first['files'][0]['line_counters'][0]['branches'][0]['count'] = 1
    second['files'][0]['line_counters'][1]['branches'][0]['count'] = 1
    first['files'][0]['line_counters'][0]['count'] = 1
    second['files'][0]['line_counters'][1]['count'] = 1
    combined = merge([first,second])
    assert combined['files'][0]['branches_total']==2
    assert combined['files'][0]['branches_taken']==2
    assert combined['coverage_complete'] and not combined['forecast_proven']
    assert not combined['proof_bar']
    changed = deepcopy(second)
    changed['files'][0]['line_counters'][1]['branches'][0]['destination_block_id']=7
    with pytest.raises(ValueError,match='topology'):
        merge([first,changed])
    changed['notes_sha256']='different-build'
    assert len(merge([first,changed])['files'])==2
    changed['process_count']=24
    with pytest.raises(ValueError,match='process counts'):
        merge([first,changed])
    second['files'][0]['unmapped_native_lines']=[9]
    first['files'][0]['unmapped_native_lines']=[9]
    assert not merge([first,second])['coverage_complete']

def test_coverage_merge_refuses_a_report_without_branch_block_ids():
    # gcov from GCC 13 and older writes branches with no block ids (JSON format 1).
    from replay.merge_coverage import merge
    branch = dict(count=1, fallthrough=True, throw=False)
    source = dict(native_source='native.F', source_sha256='a', instrumented_source_sha256='b',
                  line_counters=[dict(line=8, function='first', count=1, branches=[branch])],
                  unmapped_native_lines=[])
    report = dict(process_count=1, gcc_version='13.3.0', notes_sha256='c', files=[source],
                  witness={'capture': 'first'})
    with pytest.raises(ValueError, match='branch block ids, which GCC 14'):
        merge([report])

def test_trial_forecast_matrix_preserves_all_explicit_cases_and_rejects_bad_overrides():
    from reproduce_dump_identity import forecast_cases
    profile = dict(deck='plain-deck', case='native-case', steps='0,1', keys='selected',
                   host='native-host', process_count=1)
    assert forecast_cases(profile) == [profile]
    matrix = dict(profile, cases=[dict(name='first', deck='first-deck'),
                                 dict(name='second', deck='second-deck', keys='other')])
    cases = forecast_cases(matrix)
    assert [p['deck'] for p in cases] == ['first-deck', 'second-deck']
    assert [p['keys'] for p in cases] == ['selected', 'other']
    assert all(p['process_count'] == 1 and p['host'] == 'native-host' for p in cases)
    assert profile['deck'] == 'plain-deck' and matrix['cases'][0]['deck'] == 'first-deck'
    for items in ([], [dict(name='../escape')], [dict(name='x'), dict(name='x')],
                  [dict(name='x', process_count=24)]):
        with pytest.raises(ValueError, match='case|matrix'):
            forecast_cases(dict(profile, cases=items))

def test_native_replay_large_table_stack_limit_is_child_only():
    import resource
    spec = importlib.util.spec_from_file_location('stack_replay', ORACLE.parent.parent / 'column_oracle_replay.py')
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    original = resource.getrlimit(resource.RLIMIT_STACK)
    result = harness.run_native_replay([sys.executable, '-c',
        'import json,resource; print(json.dumps(resource.getrlimit(resource.RLIMIT_STACK)))'])
    assert result.returncode == 0
    assert json.loads(result.stdout) == [original[1], original[1]]
    assert resource.getrlimit(resource.RLIMIT_STACK) == original
    assert harness.native_resource_limits()['child_stack_soft_bytes'] == original[1]


def test_module_state_audit_discovers_silent_complex_omission(tmp_path):
    from replay.audit_state import compare_declarations
    physics = tmp_path / 'src/core_atmosphere/physics'
    physics.mkdir(parents=True)
    # The two complex SAVE fields at native mp_radar.F90:43 were absent from
    # the old inventory's roots, leaves AND issues. The source audit must still
    # discover them; an inventory-only completeness check could not do so.
    (physics / 'radar.F90').write_text('''module mp_radar
integer, parameter :: R8KIND = 8
complex(kind=R8KIND) :: m_w_0, m_i_0
real(kind=R8KIND) :: radar_value
end module mp_radar
''')
    native = modules(tmp_path)['mp_radar']
    assert compare_declarations(native, native) == []
    legacy = dict(native, issues=[])
    for field in ('roots', 'variables'):
        legacy[field] = [v for v in native[field] if not v['name'].startswith('m_')]
    gaps = compare_declarations(native, legacy)
    assert {(g['field'], g['name']) for g in gaps} == {
        (field, name) for field in ('roots', 'variables') for name in ('m_w_0', 'm_i_0')}


def test_calendar_source_components_keep_target_bounds_and_pointer_state(tmp_path):
    from framework_calendar import components
    directory=tmp_path/'src/external/esmf_time_f90';directory.mkdir(parents=True)
    (directory/'ESMF_BaseTimeMod.F90').write_text('''module ESMF_BaseTimeMod
 type ESMF_BaseTime
 integer(ESMF_KIND_I8) :: S,Sn,Sd
 end type
 end module
''')
    calendar=directory/'ESMF_CalendarMod.F90'
    calendar.write_text('''module ESMF_CalendarMod
 type ESMF_CalKind_Flag
 integer :: caltype
 end type
 type ESMF_DaysPerYear
 integer :: D=0,Dn=0,Dd=1
 end type
 type ESMF_Calendar
 type(ESMF_CalKind_Flag) :: Type
 logical :: Set=.false.
 integer :: DaysPerMonth(MONTHS_PER_YEAR)=0
 integer :: SecondsPerDay=0,SecondsPerYear=0
 type(ESMF_DaysPerYear) :: DaysPerYear
 end type
 end module
''')
    target=declarations('type(ESMF_BaseTime), target :: monthbdys(0:MONTHS_PER_YEAR)')[0]
    fields=components(tmp_path,target)
    assert len(fields)==3
    assert all(f['type']=='integer(esmf_kind_i8)' and f['dims']==['0:MONTHS_PER_YEAR']
               and f['lower_expr']=='lbound(monthbdys)' for f in fields)
    pointer=declarations('type(ESMF_Calendar), pointer :: defaultCal')[0]
    fields=components(tmp_path,pointer)
    assert len(fields)==9 and fields[0]['container_pointer']
    assert all(f['conditions']==['associated(defaultcal)'] for f in fields[1:])
    assert next(f for f in fields if f['name'].endswith('%dayspermonth'))['dims']==['MONTHS_PER_YEAR']
    # A changed native layout must fail explicitly rather than dropping storage.
    calendar.write_text(calendar.read_text().replace('logical :: Set=.false.',
                                                  'logical, pointer :: Set'))
    with pytest.raises(ValueError,match='component storage changed'):
        components(tmp_path,pointer)


def test_module_state_audit_checks_absent_allocations_and_missing_post():
    from replay.audit_state import membership
    variables = [('radar', dict(name='table', attrs=', allocatable', dims=[':'])),
                 ('radar', dict(name='m_w_0', attrs='', dims=[]))]
    members = {'call.post.state.radar.table.present.bin', 'call.post.state.radar.m_w_0.bin'}
    # A captured false descriptor authorizes an absent value, but never an
    # omitted descriptor or a missing scalar, including complex hidden state.
    assert membership(variables, 'call', 'post', members, lambda _: False)['complete']
    present_table = membership(variables, 'call', 'post', members, lambda _: True)
    assert present_table['missing'] == ['call.post.state.radar.table.bin']
    members.remove('call.post.state.radar.m_w_0.bin')
    absent_scalar = membership(variables, 'call', 'post', members, lambda _: False)
    assert absent_scalar['missing'] == ['call.post.state.radar.m_w_0.bin']


def test_module_state_audit_requires_native_conditional_controls():
    from replay.audit_state import membership
    variables = [('clock', dict(name='has_start', type='logical', attrs='', dims=[])),
                 ('clock', dict(name='start', type='integer', attrs='', dims=[],
                                conditions=['has_start'])),
                 ('clock', dict(name='unsupported', type='real', attrs='', dims=[],
                                conditions=['has_start .and. another_flag']))]
    members = {'call.pre.state.clock.has_start.bin'}
    inactive = membership(variables[:2], 'call', 'pre', members, lambda _: False)
    assert inactive['complete'] and len(inactive['inactive_source_guarded_variables']) == 1
    active = membership(variables[:2], 'call', 'pre', members, lambda _: True)
    assert active['missing'] == ['call.pre.state.clock.start.bin']
    unknown = membership(variables, 'call', 'pre', members, lambda _: False)
    assert not unknown['complete'] and len(unknown['conditional_unverified']) == 1
    missing = membership(variables[:2], 'call', 'pre', set(), lambda _: False)
    assert not missing['complete'] and missing['conditional_unverified']
    with pytest.raises(ValueError, match='decoded logical'):
        membership(variables[:2], 'call', 'pre', members, lambda _: 1)


@pytest.mark.parametrize('defect', ['omitted-array', 'repeated-pair', 'changed-member', 'unwitnessed-rank'])
def test_native_matrix_rejects_incomplete_or_changed_evidence(tmp_path, defect):
    from replay.audit_matrix import audit_site, sha, RECIPES
    import itertools
    paths={}; base=dict(scheme='native_scheme',dump_root_sha256='root',pin_set_sha256='pins',
        source_sha256={'native.F':'source'},backend='native-replay',comparison_complete=True,
        replay_process_count=1,executable_sha256='exe',all_outputs_bitwise_equal=True,
        captures=[dict(capture='call',arrays={'q':dict(words=4,different_words=0,
            bitwise_equal=True,max_ulp=0)})])
    for recipe in RECIPES | {'coverage'}:
        path=tmp_path/(recipe+'.json');path.write_text(json.dumps(dict(base,replay_recipe=recipe)));paths[recipe]=path
    pairs=[dict(recipes=list(pair)) for pair in itertools.combinations(sorted(RECIPES),2)]
    spread=dict(scheme='native_scheme',dump_root_sha256='root',pin_set_sha256='pins',
        source_sha256={'native.F':'source'},arrays={'call:q':dict(pairs=pairs,
            largest_pairwise_differing_words=0,largest_pairwise_max_ulp=0)},
        receipts=[dict(recipe=r,sha256=sha(paths[r]),executable_sha256='exe')for r in RECIPES])
    target=tmp_path/'spread.json';target.write_text(json.dumps(spread))
    assert audit_site(paths,target)['output_words']==4
    # Compiler differences, however large, cannot change the strict result.
    intel=json.loads(paths['ifx-o3'].read_text());intel['all_outputs_bitwise_equal']=False
    intel['captures'][0]['arrays']['q'].update(different_words=4,bitwise_equal=False,max_ulp=2**32-1)
    paths['ifx-o3'].write_text(json.dumps(intel))
    next(m for m in spread['receipts']if m['recipe']=='ifx-o3')['sha256']=sha(paths['ifx-o3'])
    spread['arrays']['call:q'].update(largest_pairwise_differing_words=4,largest_pairwise_max_ulp=2**32-1)
    target.write_text(json.dumps(spread))
    assert audit_site(paths,target)['strict_differing_words']==0
    if defect=='omitted-array':spread['arrays']={}
    elif defect=='repeated-pair':spread['arrays']['call:q']['pairs'][-1]=pairs[0]
    else:
        recipe='ifx-o3';receipt=json.loads(paths[recipe].read_text())
        if defect=='changed-member':receipt['executable_sha256']='other-executable'
        else:receipt.pop('replay_process_count')
        paths[recipe].write_text(json.dumps(receipt))
    target.write_text(json.dumps(spread))
    with pytest.raises(ValueError):audit_site(paths,target)


def test_compiler_differences_are_debugging_only_and_process_count_is_real(tmp_path):
    from replay.spread import process_counts,write_receipt
    assert process_counts([{'replay_process_count':1}]*4)==[1]*4
    for counts in ([1,None,1,1],[1,2,1,1],[1,True,1,1],[1,0,1,1]):
        with pytest.raises(ValueError):process_counts([{'replay_process_count':c}for c in counts])
    # Even an input asking for a proof bar cannot turn compiler differences
    # into acceptance. Preserve all measured differences without a threshold.
    path=tmp_path/'debug.json'
    write_receipt(path,{'arrays':{'q':{'largest_pairwise_differing_words':10**9,
         'largest_pairwise_max_ulp':2**32-1}},'proof_bar':True,'replay_process_counts':[1]*4})
    result=json.loads(path.read_text())
    assert result['purpose']=='debugging-only' and not result['proof_bar']
    assert result['process_count']==1 and result['arrays']['q']['largest_pairwise_differing_words']==10**9
    spec=importlib.util.spec_from_file_location('process_replay',ORACLE.parent.parent/'column_oracle_replay.py')
    harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
    import argparse
    with pytest.raises(ValueError,match='exactly one process'):
        harness.replay(argparse.Namespace(backend='native-replay',process_count=2))


def test_lexer_and_mpas_preprocessing(tmp_path):
    source='''module example
contains
#if defined(mpas)
subroutine scheme(x, y, &
                     optional_flag)
#else
subroutine other(x)
#endif
real, intent(in), dimension(-1:2,1:3) :: x
real, intent(out) :: y
logical, optional, intent(in) :: optional_flag
call callee(x(:, f(1,2)), message='quoted ! , '' quote') ! actual comment
end subroutine scheme
end module
'''
    path=tmp_path/'example.F';path.write_text(source)
    r=list(routines(path,tmp_path))
    assert len(r)==1 and r[0].name=='scheme'
    assert r[0].variables['x'].dims==['-1:2','1:3']
    assert r[0].variables['optional_flag'].optional
    assert bind(r[0],['x=a','y=b'])=={'x':'a','y':'b'}
    assert r[0].body[-1].text=="call callee(x(:, f(1,2)), message='quoted ! , '' quote')"
    with pytest.raises(ValueError,match='missing required'):bind(r[0],['x=a'])
    with pytest.raises(ValueError,match='invalid/duplicate'):bind(r[0],['x=a','x=b'])
    assert len(active_source(source).splitlines())==len(source.splitlines())


def test_generated_transport_is_current():
    assert (ORACLE/'mpas_oracle_dump.F').read_text()==transport_source()
    assert (ORACLE/'mpas_oracle_pool.F').read_text()==pool_source()


def test_native_spread_receipt_shards_preserve_every_array(tmp_path):
    from replay.spread import write_receipt,sha
    arrays={f'capture:array{i}':{'pairs':[{'different_words':i,'max_ulp':2*i}]*6}
            for i in range(8)}
    result={'schema':'mpas-native-replay-spread-v1','arrays':arrays,'proof_bar':False,
            'replay_process_counts':[1]*4}
    path=tmp_path/'spread.json';write_receipt(path,result,limit=1200)
    index=json.loads(path.read_text());restored={}
    assert index['schema']=='mpas-native-replay-spread-v2' and index['array_count']==8
    for shard in index['array_shards']:
        p=tmp_path/shard['file']
        assert p.stat().st_size==shard['bytes']<=1200 and sha(p)==shard['sha256']
        part=json.loads(p.read_text())
        assert part['replay_process_counts']==[1]*4 and not part['proof_bar']
        restored.update(part['arrays'])
    assert restored==arrays and not index['proof_bar']
    with pytest.raises(ValueError,match='one native array'):
        write_receipt(tmp_path/'too-small.json',result,limit=200)


def test_retired_native_spread_requires_bytes_for_distinct_changed_outputs(transport,tmp_path):
    from copy import deepcopy
    from replay.spread import compare_authenticated_array,sha
    # Three genuine Fortran outputs share one native array schema. Distances
    # from a common reference cannot determine the changed/changed pair.
    (tmp_path/'produce.F90').write_text('''program produce
 use iso_fortran_env, only: int32,real32
 use mpas_oracle_dump
 implicit none
 real(real32) :: q(-1:2)
 integer :: event,i
 character(len=8) :: key
 do i=0,2
  write(key,'(a,i1)') 'variant',i
  call oracle_set_context([205_int32,309_int32,413_int32,517_int32],7,0)
  call oracle_begin(trim(key),event)
  q=1.0_real32
  q(2)=transfer(int(z'3f800000',int32)+i,0.0_real32)
  call oracle_write(trim(key),'post','q',q,event,lower=lbound(q),cell_axis=1)
 enddo
end program
''')
    (tmp_path/'Makefile').write_text(f'''all: produce
produce: produce.F90
\tgfortran -I{transport} -ffree-line-length-none -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations $< {transport}/transport.o -o $@
''')
    subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,check=True,capture_output=True)
    raw=tmp_path/'raw';raw.mkdir()
    env={k:v for k,v in os.environ.items() if not k.startswith('MPAS_ORACLE_')}
    env['MPAS_ORACLE_DIR']=str(raw)
    subprocess.run([str(tmp_path/'produce')],env=env,check=True,capture_output=True)
    files=[next(raw.glob(f'variant{i}*.post.q.bin')) for i in range(3)]
    rows=[]
    for file in files:
        row=json.loads(compare(transport,files[0],file).stdout)
        row.update(reference_sha256=sha(files[0]),replay_sha256=sha(file));rows.append(row)
    unavailable=[tmp_path/'retired-left.bin',tmp_path/'retired-right.bin']
    helper=transport/'oracle_words'
    equal=compare_authenticated_array(rows[1],rows[1],unavailable,helper)
    assert equal['different_words']==0 and equal['max_ulp']==0
    baseline=compare_authenticated_array(rows[2],rows[0],unavailable,helper)
    assert baseline['different_words']==1 and baseline['max_ulp']==2
    with pytest.raises(ValueError,match='require authenticated bytes'):
        compare_authenticated_array(rows[1],rows[2],unavailable,helper)
    changed=compare_authenticated_array(rows[1],rows[2],files[1:],helper)
    assert changed['different_words']==1 and changed['max_ulp']==1
    assert changed['first_difference']['cellID']==309
    assert changed['first_difference']['indices']==[2]
    assert changed['evidence'].startswith('Rust comparison')
    for defect in ('different_words','reference_sha256','max_ulp','comparison_complete'):
        bad=deepcopy(rows[1])
        bad[defect]={'different_words':0,'reference_sha256':'a'*64,
                     'max_ulp':-1,'comparison_complete':False}[defect]
        with pytest.raises(ValueError):
            compare_authenticated_array(bad,rows[2],files[1:],helper)
    with pytest.raises(ValueError,match='authenticated bytes'):
        compare_authenticated_array(rows[1],rows[2],files[:2],helper)


def test_native_coverage_resolves_compile_directory_and_excludes_hooks(tmp_path):
    from replay.coverage import summarize
    source=tmp_path/'source';source.mkdir();build=tmp_path/'build';build.mkdir()
    alias=tmp_path/'build-alias';alias.symlink_to(build,target_is_directory=True)
    for p in [source,build]:(p/'physics').mkdir()
    (source/'physics/native.F').write_text('subroutine scheme\nx=x+1\nend subroutine\n')
    (build/'physics/native.F').write_text('subroutine scheme\ncall generated_hook()\nx=x+1\nend subroutine\n')
    data={'gcc_version':'fixture','current_working_directory':str(alias/'physics'),
          'files':[{'file':'native.F','lines':[
              {'line_number':2,'count':1,'branches':[{'count':1},{'count':0}]},
              {'line_number':3,'count':1,'branches':[{'count':2},{'count':0}]}]}]}
    rows=summarize(data,source,alias)['files'];assert len(rows)==1
    assert rows[0]['branches_total']==2 and rows[0]['branches_taken']==1
    assert rows[0]['instrumentation_lines_excluded']==1
    assert rows[0]['uncovered_branches'][0]['line']==2
    data['files'][0]['file']='missing.F'
    with pytest.raises(ValueError,match='no pristine native source'):summarize(data,source,build)


def test_native_coverage_collection_preserves_duplicate_source_names(tmp_path):
    from replay.collect_coverage import collect,sha
    source=tmp_path/'source';build=tmp_path/'build'
    targets=[]
    for name in ['left','right']:
        native=source/name;generated=build/name
        native.mkdir(parents=True);generated.mkdir(parents=True)
        text='''program branch
 implicit none
 integer :: n
 character(len=16) :: argument
 call get_command_argument(1,argument)
 read(argument,*) n
 if(n>0) then
   print *, n
 else
   print *, -n
 endif
end program
'''
        (native/'branch.f90').write_text(text)
        (generated/'branch.f90').write_text(text)
        targets.append(f'{name}/branch')
    (build/'Makefile').write_text('all: '+' '.join(targets)+'\n'+''.join(
        f'{name}/branch: {name}/branch.f90\n\tcd {name} && gfortran -O0 --coverage branch.f90 -o branch\n'
        for name in ['left','right']))
    subprocess.run(_build_command(['make','-j4']),cwd=build,
                   env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','f1-native-tap')),check=True,capture_output=True)
    rows=[]
    for name,argument in [('left','1'),('right','-1')]:
        subprocess.run([str(build/name/'branch'),argument],check=True,capture_output=True)
        counter=build/name/'branch.gcda';before=sha(counter)
        row=collect(counter,source,build,tmp_path/'reports'/name,replay_process_counts=[1])
        assert sha(counter)==before
        report=json.loads(Path(row['receipt']).read_text())
        assert report['counters_cumulative'] and not report['coverage_complete']
        assert report['process_count']==1 and report['replay_process_counts']==[1]
        assert report['files'][0]['native_source']==f'{name}/branch.f90'
        assert report['files'][0]['branches_taken']>0
        assert report['files'][0]['uncovered_branches']
        rows.append(row)
    assert rows[0]['receipt']!=rows[1]['receipt']
    # A second-host collector must map paths embedded by the first host's compiler.
    relocated=tmp_path/'relocated'
    shutil.copytree(build,relocated)
    for name in ['left','right']:
        row=collect(relocated/name/'branch.gcda',source,relocated,
                    tmp_path/'relocated-reports'/name,build,[1])
        report=json.loads(Path(row['receipt']).read_text())
        original=json.loads(Path(rows[['left','right'].index(name)]['receipt']).read_text())
        assert report['counter_sha256']==original['counter_sha256']
        assert report['files']==original['files']
        assert report['compiled_build_root']==str(build.resolve())
        assert report['collected_build_root']==str(relocated.resolve())


def test_coverage_refuses_unknown_or_mixed_process_counts(tmp_path):
    from replay.collect_coverage import collect,execution_counts
    for counts in [None,[],[None],[0],[True],['1'],[1,24]]:
        with pytest.raises(ValueError,match='process counts'):
            # Reject the evidence before touching counters or creating output.
            collect(tmp_path/'missing.gcda',tmp_path,tmp_path,tmp_path/'out',
                    replay_process_counts=counts)
        assert not (tmp_path/'out').exists()
    assert execution_counts([24,24])==[24,24]


def test_whole_native_coverage_retains_real_generated_objects(tmp_path):
    from replay.collect_coverage import collect,sha
    from replay.coverage import summarize
    from replay.merge_coverage import merge
    source=tmp_path/'source';source.mkdir()
    build=tmp_path/'build';build.mkdir()
    # The linked native routine and the generated entry both execute. Only the
    # first has a pristine MPAS-equivalent source and counts as native coverage.
    native='''subroutine native(n)
 implicit none
 integer,intent(in)::n
 if(n>0) then
  print *,n
 else
  print *,-n
 endif
end subroutine
'''
    generated='''program replay
 implicit none
 call native(1)
 call native(-1)
end program
'''
    (source/'native.f90').write_text(native)
    (build/'native.f90').write_text(native)
    (build/'generated.f90').write_text(generated)
    (build/'Makefile').write_text('all: replay\nreplay: native.o generated.o\n\tgfortran --coverage $^ -o $@\n%.o: %.f90\n\tgfortran -O0 --coverage -c $< -o $@\n')
    subprocess.run(_build_command(['make','-j4']),cwd=build,
                   env=dict(os.environ,MPAS_UNIT='f1-native-tap'),check=True,capture_output=True)
    subprocess.run([str(build/'replay')],cwd=build,check=True,capture_output=True)
    reports=[]
    for name in ['native','generated']:
        counter=build/(name+'.gcda');before=sha(counter)
        row=collect(counter,source,build,tmp_path/'reports'/name,replay_process_counts=[1])
        assert sha(counter)==before
        report=json.loads(Path(row['receipt']).read_text())
        assert report['process_count']==1 and not report['coverage_complete']
        report['witness']=dict(counter_sha256=before,receipt_sha256=row['sha256'])
        # Native topology still carries actual function/block identities.
        import gzip
        raw=json.loads(gzip.decompress((tmp_path/'reports'/name/(name+'.gcov.json.gz')).read_bytes()))
        branches=[b for f in raw['files'] for l in f['lines'] for b in l.get('branches',[])]
        if branches and not all('source_block_id' in b for b in branches):
            pytest.skip(f"gcov from gcc {raw['gcc_version']} writes no branch block ids (GCC 14 and "
                        "later do); the coverage merge keys every branch on them and refuses without")
        mapped=summarize(raw,source,build,include_counters=True,retain_unmapped_sources=True)
        report.update(files=mapped['files'],unmapped_files=mapped['unmapped_files'])
        reports.append(report)
    assert reports[0]['files'][0]['branches_taken']>0
    assert reports[1]['files']==[]
    row=reports[1]['unmapped_files'][0]
    assert row['native_branches_counted']==0 and row['instrumented_source_sha256']==sha(build/'generated.f90')
    assert row['generated_source']=='generated.f90'
    assert not merge([reports[1]])['coverage_complete']
    assert not merge(reports)['coverage_complete']
    assert merge(reports)['files']==merge([reports[0]])['files']
    relocated=tmp_path/'relocated';shutil.copytree(build,relocated)
    row=collect(relocated/'generated.gcda',source,relocated,tmp_path/'relocated-report',build,[1])
    assert json.loads(Path(row['receipt']).read_text())['unmapped_files']==reports[1]['unmapped_files']
    (relocated/'generated.f90').unlink()
    with pytest.raises(ValueError,match='generated coverage source missing'):
        collect(relocated/'generated.gcda',source,relocated,tmp_path/'missing-source-report',build,[1])


def test_object_audit_uses_latest_command_and_verifies_copies(tmp_path):
    from audit_build import audit
    source=tmp_path/'physics';source.mkdir()
    copies=tmp_path/'libphys';copies.mkdir()
    (source/'scheme.F').write_text('end\n')
    (source/'scheme.o').write_bytes(b'compiled object')
    (copies/'scheme.o').write_bytes(b'compiled object')
    log=tmp_path/'build.log'
    strict='gfortran -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -c scheme.F\n'
    log.write_text(strict)
    result=audit(tmp_path,log)
    assert result['all_ok'] and len(result['objects'])==2
    assert result['objects'][0]['identical_copy_of']=='physics/scheme.o'
    log.write_text(strict+'gfortran -Ofast -c scheme.F\n')
    assert not audit(tmp_path,log)['all_ok']
    log.write_text(strict)
    (copies/'scheme.o').write_bytes(b'stale object')
    assert audit(tmp_path,log)['errors']==['libphys/scheme.o']


def test_hidden_function_state_and_separate_parameters(tmp_path):
    root=tmp_path/'src/core_atmosphere/physics';root.mkdir(parents=True)
    (root/'hidden.F').write_text('''module hidden
 implicit none
 real :: constant,table(2)
 real(kind=double) :: spectral_table(2)
 integer(kind=im) :: indices(2)
 parameter(constant=2.0)
 type old_style
  real :: component
 end type old_style
 contains
 real function sample(x) result(y)
 real :: x
 integer, save :: counter=0
 y=x
 end function sample
 end module hidden
''')
    info=modules(tmp_path)['hidden']
    assert info['contains'] is not None
    assert [v['name'] for v in info['variables']]==['table','spectral_table','indices']
    assert any(i.get('routine')=='sample' and i.get('name')=='counter' for i in info['issues'])
    assert not declarations('real_time=2.0')


@pytest.fixture(scope='module')
def transport(tmp_path_factory):
    directory=tmp_path_factory.mktemp('transport')
    (directory/'transport.F').write_text(transport_source())
    # Field construction runs in Fortran, including NaN payload and signed zero.
    source='''program roundtrip
 use iso_fortran_env, only: int8,int32,int64,real32,real64
 use mpas_oracle_dump
 implicit none
 integer :: event,i
 real(real32) :: a(-1:2,4:6),b(-1:2,4:6),v4,w4
 real(real64) :: v8,w8
 complex(real32) :: z4,wz4,za4(-1:2,4:6)
 complex(real64) :: z8,wz8,za8(-1:2,4:6)
 integer(int32) :: vi,wi
 logical :: vl,wl
 character(len=7) :: vc,wc
 character(len=4096) :: directory,path
'''
    ranks=[]
    for tag,typ in [('r4','real(real32)'),('r8','real(real64)'),('z4','complex(real32)'),('z8','complex(real64)'),('i','integer(int32)'),('i8','integer(int64)'),('l','logical'),('c','character(len=7)')]:
        for rank in range(1,6):
            name=f'{tag}_{rank}';ranks.append((name,rank,tag))
            source+=f" {typ} :: {name}({','.join('2' for _ in range(rank))}),copy_{name}({','.join('2' for _ in range(rank))})\n"
    source+=''' call get_command_argument(1,directory)
 a=reshape([(real(i,real32),i=1,12)],shape(a))
 a(0,5)=transfer(int(z'80000000',int32),v4)
 a(1,5)=transfer(int(z'7fc01234',int32),v4)
 v4=-1.25_real32;v8=-2.5_real64;vi=-18;vl=.true.;vc='a!b,c d'
 z4=transfer([int(z'80000000',int32),int(z'7fc01234',int32)],z4)
 z8=transfer([int(z'8000000000000000',int64),int(z'7ff8000000001234',int64)],z8)
 za4=cmplx(1.0_real32,2.0_real32,kind=real32);za4(0,5)=z4
 za8=cmplx(1.0_real64,2.0_real64,kind=real64);za8(0,5)=z8
 call oracle_set_context([205_int32,309_int32,413_int32,517_int32],7,2)
 call oracle_begin('typed',event)
 call oracle_write('typed','pre','a',a,event,lower=lbound(a),cell_axis=1,level_axis=2)
 call oracle_write('typed','pre','r4',v4,event)
 call oracle_write('typed','pre','r8',v8,event)
 call oracle_write('typed','pre','i',vi,event)
 call oracle_write('typed','pre','l',vl,event)
 call oracle_write('typed','pre','c',vc,event)
 call oracle_write('typed','pre','z4',z4,event)
 call oracle_write('typed','pre','z8',z8,event)
 call oracle_write('typed','pre','za4',za4,event,lower=lbound(za4),cell_axis=1,level_axis=2)
 call oracle_write('typed','pre','za8',za8,event,lower=lbound(za8),cell_axis=1,level_axis=2)
'''
    for name,rank,tag in ranks:
        val={'r4':'-1.25_real32','r8':'-2.5_real64','z4':'z4','z8':'z8','i':'-18','i8':'-9223372036854775807_int64','l':'.true.','c':"'a!b,c d'"}[tag]
        source+=f" {name}={val}\n call oracle_write('typed','pre','{name}',{name},event,lower=lbound({name}))\n"
    source+=''' path=trim(directory)//'/typed.s0000000007.e0000000001.r000002.pre.'
 if(oracle_enabled('typed')) then
  call oracle_read(trim(path)//'a.bin',b)
  if(any(transfer(a,[0_int8])/=transfer(b,[0_int8]))) error stop 'array word roundtrip'
  call oracle_read(trim(path)//'r4.bin',w4)
  call oracle_read(trim(path)//'r8.bin',w8)
  call oracle_read(trim(path)//'i.bin',wi)
  call oracle_read(trim(path)//'l.bin',wl)
  call oracle_read(trim(path)//'c.bin',wc)
  call oracle_read(trim(path)//'z4.bin',wz4)
  call oracle_read(trim(path)//'z8.bin',wz8)
  if(any(transfer(z4,[0_int8])/=transfer(wz4,[0_int8]))) error stop 'complex32 scalar bits'
  if(any(transfer(z8,[0_int8])/=transfer(wz8,[0_int8]))) error stop 'complex64 scalar bits'
  if(v4/=w4.or.v8/=w8.or.vi/=wi.or.(vl.neqv.wl).or.vc/=wc) error stop 'scalar roundtrip'
'''
    for name,rank,tag in ranks:
        source+=f"  call oracle_read(trim(path)//'{name}.bin',copy_{name})\n  if(any(transfer({name},[0_int8])/=transfer(copy_{name},[0_int8]))) error stop 'rank roundtrip'\n"
    source+=''' endif
 call oracle_begin('mismatch',event)
 a(2,6)=transfer(transfer(a(2,6),vi)+1_int32,v4)
 call oracle_write('mismatch','pre','a',a,event,lower=lbound(a),cell_axis=1,level_axis=2)
 za4(2,6)=cmplx(1.0_real32,transfer(transfer(2.0_real32,0_int32)+1_int32,0.0_real32),kind=real32)
 za8(2,6)=cmplx(1.0_real64,transfer(transfer(2.0_real64,0_int64)+1_int64,0.0_real64),kind=real64)
 call oracle_write('mismatch','pre','za4',za4,event,lower=lbound(za4),cell_axis=1,level_axis=2)
 call oracle_write('mismatch','pre','za8',za8,event,lower=lbound(za8),cell_axis=1,level_axis=2)
end program
'''
    (directory/'roundtrip.F90').write_text(source)
    site_support.tools_or_skip("make", "gfortran", "rustc", what="the transport fixture")
    (directory/'oracle_words.rs').write_bytes((ORACLE/'oracle_words.rs').read_bytes())
    (directory/'Makefile').write_text('''FC=gfortran
FLAGS=-std=f2008 -fimplicit-none -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -ffree-form -fconvert=big-endian
all: roundtrip oracle_words
transport.o: transport.F
	$(FC) $(FLAGS) -c $< -o $@
roundtrip: roundtrip.F90 transport.o
	$(FC) $(FLAGS) $^ -o $@
oracle_words: oracle_words.rs
	rustc --edition=2021 -O $< -o $@
''')
    env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','f1-native-tap'))
    result=subprocess.run(_build_command(['make','-j4']),cwd=directory,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    return directory


def run_transport(transport,directory,**selection):
    directory.mkdir()
    env={k:v for k,v in os.environ.items() if not k.startswith('MPAS_ORACLE_')}
    env.update(selection)
    subprocess.run([str(transport/'roundtrip'),str(directory)],env=env,check=True,capture_output=True)


def compare(transport,a,b):
    return subprocess.run([str(transport/'oracle_words'),'compare',str(a),str(b)],capture_output=True,text=True)


def test_framework_globals_bridge_preserves_module_order_and_bits(transport,tmp_path):
    from framework_state import interface_source
    # Framework compiles first, without the physics module search path. A caller
    # then imports both modules: private bridge names must not cause ambiguity.
    declarations = '''module framework_globals
 use iso_fortran_env, only: int32,int64,real64
 implicit none
 integer :: TheCalendar=2,yearWidth=4
 integer(int64) :: counters(-1:1)=[1_int64,2_int64,3_int64]
 real(real64) :: table(2)=[-0.0_real64,1.25_real64]
 complex(real64) :: radar=(-0.0_real64,2.5_real64)
 logical :: initialized=.true.
 character(len=7) :: label='calendar'
'''
    fields = ('TheCalendar','yearWidth','counters','table','radar','initialized','label')
    source = declarations + interface_source() + '''contains
 subroutine snapshot(phase,event)
 character(len=*),intent(in) :: phase
 integer,intent(in) :: event
 if(.not.oracle_enabled('framework')) return
'''
    for name in fields:
        lower = f',lower=lbound({name})' if name in ('counters','table') else ''
        source += f" call oracle_write('framework',phase,'{name}',{name},event{lower})\n"
    source += ''' end subroutine
 subroutine restore(prefix)
 character(len=*),intent(in) :: prefix
 integer :: code,width,rank_n,dims(5),lower(5),unit
 call oracle_header(trim(prefix)//'TheCalendar.bin',code,width,rank_n,dims,lower,unit)
 if(code/=3.or.width/=4.or.rank_n/=0) error stop 'framework header'
 close(unit)
'''
    for name in fields:
        source += f" call oracle_read(trim(prefix)//'{name}.bin',{name})\n"
    source += ''' end subroutine
 subroutine advance(result)
 integer,intent(out) :: result
 integer,save :: counter=0
 integer :: invocation
 call oracle_local_begin('framework_globals.advance',invocation)
 call oracle_hidden('local.framework_globals.advance.counter',counter,invocation,'pre')
 counter=counter+1
 result=counter
 call oracle_hidden('local.framework_globals.advance.counter',counter,invocation,'post')
 end subroutine
 end module
'''
    (tmp_path/'framework.F90').write_text(source)
    (tmp_path/'main.F90').write_text('''program framework_roundtrip
 use framework_globals
 use mpas_oracle_dump
 implicit none
 integer :: event,result
 character(len=4096) :: directory
 call get_command_argument(1,directory)
 call oracle_set_context([1],1,0)
 call oracle_begin('framework',event)
 call snapshot('pre',event)
 TheCalendar=-9;yearWidth=-9;counters=-9;table=-9;radar=-9
 initialized=.false.;label='changed'
 call restore(trim(directory)//'/framework.s0000000001.e0000000001.r000000.pre.')
 call snapshot('post',event)
 call oracle_begin('framework_local',event)
 call oracle_scope_begin('framework_local',event)
 call advance(result)
 call oracle_scope_end()
 if(result/=1) error stop 'framework first local call'
 call oracle_begin('framework_local',event)
 call oracle_replay_begin(trim(directory)//'/framework_local.s0000000001.e0000000001.r000000','framework_local',event)
 call advance(result)
 call oracle_scope_end()
 if(result/=1) error stop 'framework local SAVE input restore'
end program
''')
    (tmp_path/'Makefile').write_text(f'''FC=gfortran
FLAGS=-ffree-form -ffree-line-length-none -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations
all: framework_roundtrip
framework.o: framework.F90
\t$(FC) $(FLAGS) -c $< -o $@
framework_roundtrip: framework.o main.F90
\t$(FC) $(FLAGS) -I{transport} main.F90 framework.o {transport}/transport.o -o $@
''')
    build=subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,
                         env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','f1-native-tap')),capture_output=True,text=True)
    assert build.returncode==0,build.stdout+build.stderr
    root=tmp_path/'dump';root.mkdir()
    run=subprocess.run([str(tmp_path/'framework_roundtrip'),str(root)],
                       env=dict(os.environ,MPAS_ORACLE_DIR=str(root),MPAS_ORACLE_STEPS='1',
                                MPAS_ORACLE_KEYS='framework,framework_local'),capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    assert len(list(root.glob('*.bin')))==2*len(fields)+3
    for name in fields:
        pre=next(root.glob(f'*.pre.{name}.bin'));post=next(root.glob(f'*.post.{name}.bin'))
        result=compare(transport,pre,post)
        assert result.returncode==0,result.stdout+result.stderr
        assert json.loads(result.stdout)['different_words']==0
    native=next(root.glob('framework_local*.e0000000001*.post.*.bin'))
    replay=next(root.glob('framework_local*.e0000000002*.post.*.bin'))
    assert json.loads(compare(transport,native,replay).stdout)['different_words']==0


@pytest.mark.parametrize("nested", [False, True])
def test_framework_allocations_restore_bounds_and_saved_words(transport,tmp_path,nested):
    from framework_state import interface_source
    source='''module framework_allocations
 use iso_fortran_env, only: int64,real64
 implicit none
 real(real64),allocatable,save :: weights(:)
'''+interface_source()+'''contains
 subroutine advance()
 integer :: invocation
 real(real64),allocatable :: work(:)
 call oracle_local_begin('framework_allocations.advance',invocation)
 call oracle_hidden_alloc('local.framework.weights',weights,invocation,'pre')
 if(.not.allocated(weights)) error stop 'missing saved allocation'
 if(any(lbound(weights)/=[-2]).or.any(ubound(weights)/=[0])) error stop 'saved bounds'
 allocate(work(-2:0))
 work=-9.0_real64
 call oracle_allocation('framework.work',work,lower=lbound(work))
 ! The automatic entry hook must restore bytes before the native read.
 if(any(work/=-9.0_real64)) error stop 'automatic allocation input'
 work=weights
 deallocate(weights)
 allocate(weights(-3:0))
 weights(-3)=-0.0_real64
 weights(-2:0)=work
 call oracle_hidden_alloc('local.framework.weights',weights,invocation,'post')
 end subroutine
 end module
'''
    (tmp_path/'framework.F90').write_text(source)
    program='''program check
 use framework_allocations
 use mpas_oracle_dump
 implicit none
 integer :: event
 character(len=4096) :: directory
 call get_command_argument(1,directory)
 call oracle_set_context([1],1,0)
 allocate(weights(-2:0))
 weights=[-0.0_real64,1.25_real64,transfer(int(z'7ff8000000000042',int64),0.0_real64)]
 call oracle_begin('framework_alloc',event)
 call oracle_scope_begin('framework_alloc',event)
{advance}
 call oracle_scope_end()
 deallocate(weights)
 allocate(weights(8:8));weights=-100.0_real64
 call oracle_begin('framework_alloc',event)
 call oracle_replay_begin(trim(directory)//'/framework_alloc.s0000000001.e0000000001.r000000','framework_alloc',event)
 call oracle_set_output(trim(directory)//'/framework_alloc.s0000000001.e0000000002.r000000')
{advance}
 call oracle_scope_end()
end program
'''
    advance=" call oracle_scope_begin('child',1)\n call advance()\n call oracle_scope_end()" if nested else ' call advance()'
    (tmp_path/'main.F90').write_text(program.replace('{advance}',advance))
    (tmp_path/'Makefile').write_text(f'''FC=gfortran
FLAGS=-O3 -ffree-form -ffree-line-length-none -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations
all: check
framework.o: framework.F90
\t$(FC) $(FLAGS) -c $< -o $@
check: framework.o main.F90
\t$(FC) $(FLAGS) -I{transport} main.F90 framework.o {transport}/transport.o -o $@
''')
    build=subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,
                         env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','r1-rrfs-native-tap')),capture_output=True,text=True)
    assert build.returncode==0,build.stdout+build.stderr
    root=tmp_path/'dump';root.mkdir()
    run=subprocess.run([str(tmp_path/'check'),str(root)],
                       env=dict(os.environ,MPAS_ORACLE_DIR=str(root),MPAS_ORACLE_STEPS='1',
                                MPAS_ORACLE_KEYS='framework_alloc,child'),capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    assert list(root.glob('*.pre.allocation.framework.work.*.bin'))
    originals=list(root.glob('framework_alloc*.e0000000001*.post.*.bin'))
    assert len(originals)==2
    for native in originals:
        replay=root/native.name.replace('.e0000000001.','.e0000000002.')
        result=compare(transport,native,replay)
        assert result.returncode==0,result.stdout+result.stderr
        assert json.loads(result.stdout)['different_words']==0


def test_typed_roundtrip_and_gating(transport,tmp_path):
    off=tmp_path/'off';run_transport(transport,off)
    assert not list(off.iterdir())
    excluded=tmp_path/'excluded';run_transport(transport,excluded,MPAS_ORACLE_DIR=str(excluded),MPAS_ORACLE_STEPS='1,2')
    assert not list(excluded.iterdir())
    selected=tmp_path/'selected';run_transport(transport,selected,MPAS_ORACLE_DIR=str(selected),MPAS_ORACLE_STEPS='7',MPAS_ORACLE_KEYS='typed')
    assert len(list(selected.glob('*.bin')))==50
    assert not list(selected.glob('mismatch*'))
    for file in selected.glob('*.bin'):
        receipt=json.loads(compare(transport,file,file).stdout)
        assert receipt['bitwise_equal']
    validation=json.loads(subprocess.run([str(transport/'oracle_words'),'validate',str(selected)],check=True,capture_output=True,text=True).stdout)
    assert validation['all_records_valid'] and validation['records']==50
    flag=next(selected.glob('*.pre.l.bin'))
    result=subprocess.run([str(transport/'oracle_words'),'present',str(flag),'-'],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout) is True



def test_complex_component_ulp_and_column(transport,tmp_path):
    root=tmp_path/'dump';run_transport(transport,root,MPAS_ORACLE_DIR=str(root))
    for name in ('za4','za8'):
        a=next(root.glob('typed*.pre.'+name+'.bin'))
        b=next(root.glob('mismatch*.pre.'+name+'.bin'))
        report=json.loads(compare(transport,a,b).stdout)
        assert report['words']==24 and report['different_words']==1
        assert report['max_ulp']==1 and report['different_nonfinite_words']==0
        assert report['first_difference']==dict(word=23,cellID=309,level=6,indices=[2,6],component='imaginary')
        output=tmp_path/(name+'-column.bin')
        subprocess.run([str(transport/'oracle_words'),'column',str(a),'309',str(output)],check=True)
        assert json.loads(compare(transport,output,output).stdout)['words']==6
        schema=json.loads(compare(transport,a,output).stdout)
        assert schema['words']==24 and schema['other_words']==6
    from replay.generate_replay import replay_type
    for typ,wanted in [('complex','complex(real32)'),('complex*8','complex(real32)'),
                       ('complex*16','complex(real64)'),('double complex','complex(real64)'),
                       ('complex(kind=r8kind)','complex(real64)')]:
        assert declarations(typ+' :: dielectric')[0].supported
        assert replay_type(typ)==wanted
    with pytest.raises(ValueError,match='unresolved complex kind'):
        replay_type('complex(kind=16)')


def test_sample_preserves_words_and_fortran_locations(transport,tmp_path):
    root=tmp_path/'dump';run_transport(transport,root,MPAS_ORACLE_DIR=str(root))
    array=next(root.glob('typed*.pre.a.bin'))
    def sample(record,indices):
        return subprocess.run([str(transport/'oracle_words'),'sample',str(record),indices],
                              capture_output=True,text=True)
    row=json.loads(sample(array,'2,6').stdout)
    assert row['indices']==[2,6] and row['cellID']==309 and row['level']==6
    assert len(row['bits'])==8
    column=tmp_path/'column.bin'
    subprocess.run([str(transport/'oracle_words'),'column',str(array),'309',str(column)],check=True)
    assert json.loads(sample(column,'2,6').stdout)==row
    assert sample(column,'1,6').returncode!=0
    assert sample(array,'2').returncode!=0
    assert json.loads(sample(array,'0,5').stdout)['bits']=='80000000'
    assert json.loads(sample(array,'1,5').stdout)['bits']=='7fc01234'

def test_global_id_level_ulp_and_repro(transport,tmp_path):
    root=tmp_path/'dump';run_transport(transport,root,MPAS_ORACLE_DIR=str(root))
    a=next(root.glob('typed.*.a.bin'));b=next(root.glob('mismatch.*.a.bin'))
    report=json.loads(compare(transport,a,b).stdout)
    assert report['different_words']==1 and report['max_ulp']==1
    assert report['first_difference']=={'word':11,'cellID':309,'level':6,'indices':[2,6]}
    # With lower=-1, changed coordinate 2 maps to global_ids(2), ID 309.
    # Extract a mapped column at local index 1 (205), preserving raw signed-zero/NaN bytes.
    output=tmp_path/'column.bin'
    subprocess.run([str(transport/'oracle_words'),'column',str(a),'205',str(output)],check=True)
    assert json.loads(compare(transport,output,output).stdout)['words']==3
    unmapped=next(root.glob('typed.*.r4_1.bin'))
    shared=tmp_path/'shared-state.bin'
    subprocess.run([str(transport/'oracle_words'),'column',str(unmapped),'205',str(shared)],check=True)
    assert shared.stat().st_ino==unmapped.stat().st_ino
    assert json.loads(compare(transport,unmapped,shared).stdout)['bitwise_equal']
    schema=json.loads(compare(transport,a,output).stdout)
    assert not schema['bitwise_equal'] and not schema['comparison_complete']
    assert schema['different_words'] is None
    assert set(schema['schema_mismatch'])=={'dimensions','global_ids','lower_bounds'}
    truncated=tmp_path/'truncated.bin';truncated.write_bytes(a.read_bytes()[:-1])
    assert compare(transport,a,truncated).returncode!=0
    invalid=tmp_path/'invalid';invalid.mkdir()
    (invalid/'record.bin').write_bytes(truncated.read_bytes())
    assert subprocess.run([str(transport/'oracle_words'),'validate',str(invalid)],capture_output=True).returncode!=0
    trailing=tmp_path/'trailing.bin';trailing.write_bytes(a.read_bytes()+b'x')
    assert compare(transport,a,trailing).returncode!=0


def test_rust_capture_column_preserves_all_input_types(transport,tmp_path):
    root=tmp_path/'dump';run_transport(transport,root,MPAS_ORACLE_DIR=str(root))
    prefix=next(root.glob('typed*.pre.a.bin')).name.removesuffix('.pre.a.bin')
    target=tmp_path/'column'
    command=[str(transport/'oracle_words'),'column-capture',str(root),prefix,'205',str(target),'pre']
    result=json.loads(subprocess.run(command,check=True,capture_output=True,text=True).stdout)
    inputs=sorted(root.glob(prefix+'.pre.*.bin'))
    assert result['records']==len(inputs) and result['cell_mapped_records']>0
    assert {p.name for p in target.iterdir()}=={p.name for p in inputs}
    single=tmp_path/'single';single.mkdir()
    for record in inputs:
        destination=single/record.name
        subprocess.run([str(transport/'oracle_words'),'column',str(record),'205',str(destination)],check=True,capture_output=True)
        assert destination.read_bytes()==(target/record.name).read_bytes()
    # Capture membership and immutable destination checks happen in Rust.
    assert subprocess.run(command,capture_output=True).returncode!=0
    command[3]='../escape';command[5]=str(tmp_path/'escape')
    assert subprocess.run(command,capture_output=True).returncode!=0
    assert not (tmp_path/'escape').exists()
    command[3]='uncaptured'
    assert subprocess.run(command,capture_output=True).returncode!=0
    linked=root/(prefix+'.pre.external.bin');linked.symlink_to(inputs[0])
    command[3]=prefix;command[5]=str(tmp_path/'symlink-record')
    assert subprocess.run(command,capture_output=True).returncode!=0


def test_refused_column_diagnostic_preserves_native_word_difference(transport,tmp_path):
    import argparse
    root=tmp_path/'dump';run_transport(transport,root,MPAS_ORACLE_DIR=str(root))
    source=next(root.glob('typed*.pre.a.bin'))
    changed=next(root.glob('mismatch*.pre.a.bin'))
    prefix=source.name.removesuffix('.pre.a.bin')
    inputs=tmp_path/'inputs';inputs.mkdir()
    results=tmp_path/'results';results.mkdir()
    # A hidden input captured on a different cell subset cannot supply the
    # differing output's cell. Exercise the real Rust extraction refusal.
    subprocess.run([str(transport/'oracle_words'),'column',str(source),'205',str(inputs/source.name)],
                   check=True,capture_output=True)
    for name in ('a','later'):
        shutil.copy2(source,inputs/(prefix+'.post.'+name+'.bin'))
        shutil.copy2(changed if name=='a' else source,results/(prefix+'.post.'+name+'.bin'))
    spec=importlib.util.spec_from_file_location('diagnostic_replay',ORACLE.parents[1]/'column_oracle_replay.py')
    harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
    options=argparse.Namespace(dump=inputs,results=results,words=transport/'oracle_words')
    report=harness.compare_record(options,prefix,'a')
    assert report.get('comparison_complete',True) and report['different_words']==1
    assert report['first_difference']['cellID']==309
    assert 'requested global cell ID absent' in report['column_inputs_error']['stderr']
    assert 'column_inputs' not in report
    assert harness.compare_record(options,prefix,'later')['bitwise_equal']


def test_derived_component_replay_and_continued_access():
    from derived import array_arguments
    from fortran_source import statements
    import re
    spec='private\npublic :: allocate, &\n deallocate, driver\n'
    assert any(re.match(r'public\s*(?:::)?',st.text,re.I) and re.search(r'\bdeallocate\b',st.text,re.I) for st in statements(spec))
    leaf={'name':'io.temperature','replay_expr':'io%temperature','type':'real',
          'dims':[':'], 'attrs':['allocatable'],'optional':True,'intent':'inout',
          'cell_axis':1,'level_axis':0}
    site={'site':'example','scheme':'scheme','path':'driver.F','line':1,'end_line':2,
          'issues':[],'callee':{'module':'native','path':'native.F','line':1,'end_line':5},
          'arguments':[{'name':'io','type':'type(example_type)','optional':False,
                        'derived_module':'example_types','components':[leaf]}]}
    assert array_arguments(site)==[leaf]
    source=replay_source(site)
    assert 'use example_types, only: example_type' in source
    assert 'type(example_type) :: io' in source
    assert 'allocate(io%temperature(' in source
    assert 'io_temperature_present=allocated(io%temperature)' in source
    assert 'io=io &' in source
    assert 'if(.not.context_set.and.io_temperature_present)' in source
    assert 'real(real32), allocatable :: io.temperature' not in source


def test_local_save_data_and_common_hooks(tmp_path):
    from local_state import local_variables,patch_routines
    source='''module hidden
contains
subroutine calculate(x)
real :: x,a,b,constant
integer,save :: calls=0
parameter(constant=2.0)
common /shared/ a /other/ b
real :: table(2)
data table /1.0,2.0/
if(x<0.0) return
calls=calls+1
x=x+table(1)+a+b
end subroutine calculate
end module hidden
'''
    original=tmp_path/'hidden.F';original.write_text(source)
    routine=next(routines(original,tmp_path))
    variables,issues=local_variables(routine)
    assert {v['name'] for v in variables}=={'calls','a','b','table'}
    assert not issues
    patched,records=patch_routines(original,source,{'hidden'})
    assert "call oracle_local_begin('hidden.calculate',oracle_local_event)" in patched
    assert "local.hidden.calculate.calls',calls,oracle_local_event,'pre'" in patched
    assert patched.count("local.hidden.calculate.calls',calls,oracle_local_event,'post'")==2
    assert 'if (x<0.0) then' in patched
    assert records[0]['line']==3


def test_native_clock_components():
    from derived import time_interval_components
    root=site_support.mpas_source()
    if not root.exists():pytest.skip('pinned MPAS source is not installed')
    fields=time_interval_components(root,'interval')
    by_name={v['name']:v for v in fields}
    assert by_name['interval%ti%basetime%s']['type']=='integer(esmf_kind_i8)'
    calendar='interval%ti%starttime%calendar'
    assert by_name[calendar]['container_pointer']
    assert by_name[calendar]['conditions']==['interval%ti%starttime_set']
    assert by_name[calendar+'%daysperyear%d']['conditions']==[
        'interval%ti%starttime_set',f'associated({calendar})']


def test_local_state_executable_restore(transport,tmp_path):
    from local_state import patch_routines
    original=tmp_path/'hidden.F'
    original.write_text('''module hidden
contains
subroutine calculate(x)
real :: x
integer,save :: calls=0
real :: table(2)
data table /1.0,2.0/
calls=calls+1
x=x+table(1)+real(calls)
table(1)=table(1)+0.5
if(x<0.0) return
end subroutine calculate
end module hidden
''')
    patched,_=patch_routines(original,original.read_text(),{'hidden'})
    (tmp_path/'hidden-patched.F90').write_text(patched)
    (tmp_path/'check.F90').write_text('''program check
use hidden
use mpas_oracle_dump
implicit none
real :: x
integer :: event
character(len=4096) :: first,second
call get_command_argument(1,first)
call get_command_argument(2,second)
x=4.0
call oracle_begin('hidden',event)
call oracle_set_output(trim(first))
call oracle_scope_begin('hidden',event)
call calculate(x)
call calculate(x)
call oracle_write('hidden','post','x',x,event)
call oracle_scope_end()
call oracle_clear_output()
x=4.0
call oracle_replay_begin(trim(first),'hidden',event)
call oracle_set_output(trim(second))
call calculate(x)
call calculate(x)
call oracle_write('hidden','post','x',x,event)
call oracle_scope_end()
call oracle_clear_output()
end program
''')
    (tmp_path/'Makefile').write_text(f'''FC=gfortran
FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-form -ffree-line-length-none -fconvert=big-endian -I{transport}
all: check
hidden.o: hidden-patched.F90
\t$(FC) $(FLAGS) -c $< -o $@
check: check.F90 hidden.o
\t$(FC) $(FLAGS) $^ {transport}/transport.o -o $@
''')
    subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','f1-native-tap')),check=True,capture_output=True)
    first=tmp_path/'capture';second=tmp_path/'replay'
    subprocess.run([str(tmp_path/'check'),str(first),str(second)],check=True,capture_output=True)
    originals=list(tmp_path.glob('capture.post.*.bin'))
    assert len(originals)==3
    assert len(list(tmp_path.glob('capture.pre.*.bin')))==2
    for source in originals:
        candidate=tmp_path/source.name.replace('capture.','replay.',1)
        assert json.loads(compare(transport,source,candidate).stdout)['bitwise_equal']


def test_static_save_accessor_preserves_native_storage_and_skipped_paths(transport,tmp_path):
    from local_state import patch_routines
    from static_accessors import patch
    original=tmp_path/'native.F90'
    original.write_text('''module module_ra_rrtmg_lw
contains
subroutine o3data(o3prof,plev,kts,kte)
implicit none
integer,intent(in)::kts,kte
real,intent(inout)::o3prof(kts:kte+1)
real,intent(in)::plev(kts:kte+2)
real::prlevh(kts:kte+2),o3sum(31),o3win(31),ppsum(31),ppwin(31)
data o3sum/31*1.0/,o3win/31*2.0/,ppsum/31*3.0/,ppwin/31*4.0/
prlevh=plev
o3prof=o3sum(1)+prlevh(kts)
o3sum=o3sum+1.0
end subroutine o3data
subroutine rrtmg_lwrad(kts,kte,q,run)
implicit none
integer,intent(in)::kts,kte
real,intent(inout)::q(kts:kte+1)
logical,intent(in)::run
real::p(kts:kte+2)
p=0.0
if(run) call o3data(q,p,kts,kte)
end subroutine rrtmg_lwrad
end module module_ra_rrtmg_lw
''')
    hooked,_=patch_routines(original,original.read_text(),{'module_ra_rrtmg_lw'})
    source,records=patch(hooked,'native.F90',tmp_path)
    assert len(records)==1 and not records[0]['numerical_body_executed']
    with pytest.raises(ValueError,match='automatic bounds changed'):
        patch(hooked,'native.F90',tmp_path,requests=({'module':'module_ra_rrtmg_lw',
            'routine':'o3data','variables':{'o3sum','o3win','ppsum','ppwin'},
            'bounds':(),'callers':()},))
    (tmp_path/'native-hooked.F90').write_text(source)
    (tmp_path/'check.F90').write_text('''program check
use module_ra_rrtmg_lw
use mpas_oracle_dump
implicit none
real::q(2:5),p(2:6),saved(31)
integer::event
character(len=4096)::prefix,out
call get_command_argument(1,prefix)
call get_command_argument(2,out)
q=-7.0;p=0.0
call oracle_set_context([101],1,0)
call oracle_set_output(trim(prefix)//'.day')
call oracle_begin('day',event)
call oracle_scope_begin('day',event)
call rrtmg_lwrad(2,4,q,.true.)
if(any(q/=1.0)) stop 1
call oracle_write('day','post','q',q,event,lower=[2])
call oracle_scope_end()
call oracle_clear_output()
call o3data(q,p,2,4)
if(any(q/=2.0)) stop 2
q=-7.0
call oracle_set_output(trim(prefix)//'.night')
call oracle_begin('night',event)
call oracle_scope_begin('night',event)
call rrtmg_lwrad(2,4,q,.false.)
if(any(q/=-7.0)) stop 3
call oracle_write('night','post','q',q,event,lower=[2])
call oracle_scope_end()
call oracle_clear_output()
call oracle_read(trim(prefix)//'.night.pre.local.module_ra_rrtmg_lw.o3data.o3sum.bin',saved)
if(any(saved/=3.0)) stop 4
call o3data(q,p,2,4)
q=-7.0
call oracle_replay_begin(trim(prefix)//'.night','night',1)
call oracle_set_output(trim(out)//'.night')
call rrtmg_lwrad(2,4,q,.false.)
call oracle_write('night','post','q',q,1,lower=[2])
call oracle_scope_end()
call oracle_clear_output()
call o3data(q,p,2,4)
if(any(q/=3.0)) stop 5
q=-7.0
call oracle_replay_begin(trim(prefix)//'.day','day',1)
call oracle_set_output(trim(out)//'.day')
call rrtmg_lwrad(2,4,q,.true.)
call oracle_write('day','post','q',q,1,lower=[2])
call oracle_scope_end()
call oracle_clear_output()
end program
''')
    (tmp_path/'Makefile').write_text(f'''FC=gfortran
FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -fcheck=bounds -ffree-form -ffree-line-length-none -fconvert=big-endian -I{transport}
all: check check_coverage
native.o: native-hooked.F90
\t$(FC) $(FLAGS) -c $< -o $@
check: check.F90 native.o
\t$(FC) $(FLAGS) $^ {transport}/transport.o -o $@
native_cov.o: native-hooked.F90 native.o
\t$(FC) $(FLAGS) --coverage -Wno-error=coverage-invalid-line-number -c $< -o $@
check_coverage: check.F90 native_cov.o
\t$(FC) $(FLAGS) --coverage -Wno-error=coverage-invalid-line-number $^ {transport}/transport.o -o $@
''')
    result=subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,
                          env=dict(os.environ,MPAS_UNIT='f1-native-tap'),capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    coverage_warning='coverage-invalid-line-number' in result.stdout+result.stderr
    result=subprocess.run([str(tmp_path/'check'),str(tmp_path/'original'),str(tmp_path/'replay')],
                          capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    words=0
    for phase in ['day','night']:
        originals=list(tmp_path.glob('original.'+phase+'.post.*.bin'))
        assert len(originals)==5
        for original in originals:
            replay=tmp_path/original.name.replace('original.','replay.',1)
            receipt=json.loads(compare(transport,original,replay).stdout)
            assert receipt['bitwise_equal'],original.name
            words+=receipt['words']
    assert words==256
    covered=subprocess.run([str(tmp_path/'check_coverage'),str(tmp_path/'cov-original'),
                            str(tmp_path/'cov-replay')],capture_output=True,text=True)
    assert covered.returncode==0,covered.stdout+covered.stderr
    coverage_words=0
    for phase in ['day','night']:
        originals=list(tmp_path.glob('cov-original.'+phase+'.post.*.bin'))
        assert len(originals)==5
        for original in originals:
            replay=tmp_path/original.name.replace('cov-original.','cov-replay.',1)
            receipt=json.loads(compare(transport,original,replay).stdout)
            assert receipt['bitwise_equal'],original.name
            coverage_words+=receipt['words']
    assert coverage_words==256
    assert (tmp_path/'native_cov.gcda').is_file()
    counters=subprocess.run(['gcov','-j','-b','-c','native_cov.gcda'],cwd=tmp_path,
                            capture_output=True,text=True)
    assert counters.returncode==0,counters.stdout+counters.stderr
    assert (tmp_path/'native_cov.gcov.json.gz').is_file()
    (tmp_path/'static-save-proof.json').write_text(json.dumps(dict(
        schema='f1-static-save-native-storage-lifecycle-v53',process_count=1,
        words=words,different_words=0,day_and_skipped_night=True,
        original_save_storage_restored=True,automatic_bounds_checked=True,
        coverage_words=coverage_words,coverage_different_words=0,
        coverage_metadata_warning=coverage_warning,
        numerical_body_executed_by_accessor=False,global_hidden_state_complete=False),indent=2)+'\n')


@pytest.mark.parametrize('selection',['outer','outer,inner'])
def test_nested_scopes_retain_outer_local_state_and_allocation_inputs(transport,tmp_path,selection):
    from local_state import patch_routines
    original=tmp_path/'nested.F90'
    original.write_text("""module nested
contains
subroutine calculate(x)
real :: x
integer,save :: calls=0
real :: table(2)
data table /1.0,2.0/
real,allocatable :: buffer(:)
calls=calls+1
allocate(buffer(2))
buffer=-7.0
call oracle_allocation('nested.calculate.buffer',buffer,lower=lbound(buffer))
x=x+table(1)+real(calls)+buffer(1)
table(1)=table(1)+0.5
deallocate(buffer)
end subroutine
end module
""".replace('real :: x','use mpas_oracle_dump,only:oracle_allocation\nreal :: x'))
    patched,_=patch_routines(original,original.read_text(),{'nested'})
    (tmp_path/'nested-patched.F90').write_text(patched)
    (tmp_path/'check.F90').write_text("""program check
use nested
use mpas_oracle_dump
use iso_fortran_env,only:int32
implicit none
integer :: event,inner_event
real :: x
character(len=4096) :: input,output
call get_command_argument(1,input)
call get_command_argument(2,output)
call oracle_set_context([101_int32],1,0)
x=4.0
if(len_trim(input)>0) then
 call oracle_replay_begin(trim(input),'outer',1)
 call oracle_set_output(trim(output))
else
 call oracle_begin('outer',event)
 call oracle_scope_begin('outer',event)
endif
call oracle_begin('inner',inner_event)
call oracle_scope_begin('inner',inner_event)
call calculate(x)
call calculate(x)
call oracle_scope_end()
call calculate(x)
call oracle_begin('inner',inner_event)
call oracle_scope_begin('inner',inner_event)
call calculate(x)
call oracle_scope_end()
call oracle_write('outer','post','x',x,1)
call oracle_scope_end()
end program
""")
    (tmp_path/'Makefile').write_text(f"""FC=gfortran
FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-form -ffree-line-length-none -I{transport}
all: check
nested.o: nested-patched.F90
\t$(FC) $(FLAGS) -c $< -o $@
check: check.F90 nested.o
\t$(FC) $(FLAGS) $^ {transport}/transport.o -o $@
""")
    build=subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','f1-native-tap')),capture_output=True,text=True)
    assert build.returncode==0,build.stdout+build.stderr
    dump=tmp_path/'dump';dump.mkdir()
    env=dict(os.environ,MPAS_ORACLE_DIR=str(dump),MPAS_ORACLE_KEYS=selection,MPAS_ORACLE_STEPS='1')
    capture=subprocess.run([str(tmp_path/'check')],env=env,capture_output=True,text=True)
    assert capture.returncode==0,capture.stderr
    outer=next(dump.glob('outer*.post.x.bin'));prefix=str(outer).removesuffix('.post.x.bin')
    assert len(list(dump.glob('outer*.pre.allocation.*.bin')))==4
    assert len(list(dump.glob('outer*.pre.local.*.bin')))==2
    assert bool(list(dump.glob('inner*.pre.local.*.bin')))==('inner' in selection)
    output=tmp_path/'replay'
    replay=subprocess.run([str(tmp_path/'check'),prefix,str(output)],env=env,capture_output=True,text=True)
    assert replay.returncode==0,replay.stderr
    for source in dump.glob('outer*.post.*.bin'):
        candidate=Path(str(output)+source.name[len(Path(prefix).name):])
        assert json.loads(compare(transport,source,candidate).stdout)['bitwise_equal']


def test_partial_out_entry_storage_replay(transport,tmp_path):
    # RRTMG leaves option-dependent OUT diagnostics and halo words untouched.
    # Restore pre-call storage, never the expected post-call output.
    site={'site':'partial','scheme':'calculate','driver':'driver','path':'native.F90',
          'line':1,'end_line':8,'issues':[],
          'callee':{'module':'native','path':'native.F90','line':1,'end_line':8},
          'arguments':[{'name':'x','actual':'x','type':'real','attrs':'intent(out)',
                        'dims':['1:4'],'intent':'out','optional':False,
                        'entry_storage':True,'cell_axis':1,'level_axis':0}]}
    (tmp_path/'native.F90').write_text('''module native
contains
subroutine calculate(x)
real,intent(out) :: x(4)
x(1)=2.0
end subroutine
end module
''')
    (tmp_path/'replay.F90').write_text(replay_source(site))
    (tmp_path/'capture.F90').write_text('''program capture
use iso_fortran_env,only:int32
use native
use mpas_oracle_dump
implicit none
real :: x(4)
character(len=4096) :: prefix
call get_command_argument(1,prefix)
x=[5.0,7.0,11.0,13.0]
call oracle_set_context([101_int32,205_int32,309_int32,413_int32],1,0)
call oracle_set_output(trim(prefix))
call oracle_write('partial','pre','x',x,1,lower=lbound(x),cell_axis=1)
call calculate(x)
call oracle_write('partial','post','x',x,1,lower=lbound(x),cell_axis=1)
end program
''')
    (tmp_path/'Makefile').write_text(f'''FC=gfortran
FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -ffree-form -fconvert=big-endian -I{transport}
all: capture replay
native.o: native.F90
\t$(FC) $(FLAGS) -c $< -o $@
capture: capture.F90 native.o
\t$(FC) $(FLAGS) $^ {transport}/transport.o -o $@
replay: replay.F90 native.o
\t$(FC) $(FLAGS) $^ {transport}/transport.o -o $@
''')
    subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','f1-native-tap')),check=True,capture_output=True)
    first=tmp_path/'captured';second=tmp_path/'replayed'
    subprocess.run([str(tmp_path/'capture'),str(first)],check=True,capture_output=True)
    subprocess.run([str(tmp_path/'replay'),str(first),str(second)],check=True,capture_output=True)
    assert json.loads(compare(transport,tmp_path/'captured.pre.x.bin',tmp_path/'captured.post.x.bin').stdout)['different_words']==1
    assert json.loads(compare(transport,tmp_path/'captured.post.x.bin',tmp_path/'replayed.post.x.bin').stdout)['bitwise_equal']


def test_imported_optional_storage_inventory(tmp_path,authenticate=False):
    from generate import inventory,overlay
    root=tmp_path/'src/core_atmosphere/physics';root.mkdir(parents=True)
    (root/'native.F90').write_text("""module native
real,allocatable :: shared_array(:)
contains
subroutine calculate(x,y,omitted)
real,optional,intent(in) :: x(:)
real,intent(out) :: y
logical,optional,intent(in) :: omitted
y=42.0
end subroutine
end module
""")
    (root/'driver.F90').write_text("""module example
use native
contains
subroutine driver_example()
real :: result
call calculate(shared_array,result)
end subroutine
end module
""")
    info=inventory(tmp_path,authenticate=False)
    site=info['sites'][0]
    assert not site['issues']
    assert site['arguments'][0]['presence_guard']=='allocated(shared_array)'
    assert site['arguments'][1]['entry_storage']
    assert [a['name'] for a in site['omitted_optional_arguments']]==['omitted']
    out=tmp_path/'overlay';overlay(tmp_path,out,info)
    source=(out/site['path']).read_text()
    assert "'x.present',allocated(shared_array)" in source
    assert "'pre', 'y'" in source


def test_embedded_interface_uses_native_callee_and_owned_axes(tmp_path):
    from generate import inventory, overlay
    physics=tmp_path/'src/core_atmosphere/physics';physics.mkdir(parents=True)
    (physics/'ngw.F90').write_text('''module ngw
contains
subroutine ugwpv1_ngw_run(ugrs,xlatd,rublten)
real,intent(in) :: ugrs(:,:),xlatd(:)
real,intent(inout) :: rublten(:,:)
rublten=rublten+ugrs
end subroutine
subroutine unrelated(x)
real,intent(inout) :: x(:,:)
x=x+1
end subroutine
subroutine gwdo_ugwp(u,lat,du)
real,intent(in) :: u(:,:),lat(:)
real,intent(inout) :: du(:,:)
call unrelated(du)
call ugwpv1_ngw_run(ugrs=u,xlatd=lat,rublten=du)
end subroutine
end module
''')
    info=inventory(tmp_path,authenticate=False)
    assert len(info['sites'])==1 and not info['unresolved']
    site=info['sites'][0]
    assert site['scheme']=='ugwp_ngw' and site['callee']['name']=='ugwpv1_ngw_run'
    assert not site['issues']
    assert [(a['cell_axis'],a['level_axis']) for a in site['arguments']]==[(1,2),(1,0),(1,2)]
    out=tmp_path/'overlay';overlay(tmp_path,out,info)
    patched=(out/site['path']).read_text()
    assert 'cell_axis=1, level_axis=2' in patched
    replay=replay_source(site)
    assert 'only: ugwpv1_ngw_run' in replay and 'call ugwpv1_ngw_run(' in replay
    assert 'call ugwp_ngw(' not in replay


def test_physics_init_gate_and_distinct_module_tables(transport,tmp_path):
    from state import generate_state
    from local_state import generate_locals
    root=tmp_path/'source';physics=root/'src/core_atmosphere/physics'
    physics.mkdir(parents=True)
    for module,value in [('mpas_atmphys_vars','[3.0,5.0]'),('mpas_atmphys_rrtmg_lwinit','[7.0,11.0]')]:
        (physics/(module+'.F90')).write_text(f'module {module}\nreal :: table(2)={value}\nend module {module}\n')
    (physics/'mpas_atmphys_init.F').write_text("""module mpas_atmphys_init
contains
subroutine physics_init(mesh,route)
integer,intent(in) :: mesh,route
integer,save :: counter=3
counter=counter+1
call unrelated()
if(route==1)then
 return
endif
if(route==2)return
end subroutine physics_init
subroutine unrelated()
integer,save :: counter=7
counter=counter+1
end subroutine unrelated
end module mpas_atmphys_init
""")
    out=tmp_path/'generated';info={'sites':[]}
    generate_state(root,out,info)
    generate_locals(root,out,info)
    init=info["initialization_scope"]
    assert init["key"]=="state__physics_init" and not init["local_state_complete"]
    assert {r["routine"] for r in init["local_state_inventory"]}=={"physics_init","unrelated"}
    sources=out/'src/core_atmosphere/physics'
    (tmp_path/'pool.F90').write_text("""module mpas_oracle_pool
use mpas_oracle_dump
contains
subroutine oracle_mesh_context(mesh,step)
integer,intent(in) :: mesh,step
call oracle_set_context([mesh],step,0)
end subroutine
end module
""")
    (tmp_path/'main.F90').write_text("""program check
use mpas_atmphys_init
use mpas_oracle_dump,only:oracle_begin,oracle_write
integer :: route,event,counter
 do route=1,3
 call physics_init(101,route)
 call unrelated()
 call oracle_begin('expected',event)
 counter=route+2
 call oracle_write('expected','pre','initcounter',counter,event)
 counter=route+3
 call oracle_write('expected','post','initcounter',counter,event)
 counter=2*route+5
 call oracle_write('expected','pre','nestedcounter',counter,event)
 counter=2*route+6
 call oracle_write('expected','post','nestedcounter',counter,event)
 enddo
end program
""")
    (tmp_path/'Makefile').write_text(f"""FC=gfortran
FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-form -ffree-line-length-none -I{transport}
all: check
pool.o: pool.F90
\t$(FC) $(FLAGS) -c $< -o $@
vars.o: {sources}/mpas_atmphys_vars.F90
\t$(FC) $(FLAGS) -c $< -o $@
lw.o: {sources}/mpas_atmphys_rrtmg_lwinit.F90
\t$(FC) $(FLAGS) -c $< -o $@
init.o: {sources}/mpas_atmphys_init.F pool.o vars.o lw.o
\t$(FC) $(FLAGS) -c $< -o $@
check: main.F90 init.o pool.o vars.o lw.o
\t$(FC) $(FLAGS) $^ {transport}/transport.o -o $@
""")
    subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','f1-native-tap')),check=True,capture_output=True)
    dump=tmp_path/'dump';dump.mkdir()
    env=dict(os.environ,MPAS_ORACLE_DIR=str(dump),MPAS_ORACLE_KEYS='state__physics_init,expected',MPAS_ORACLE_STEPS='0')
    subprocess.run([str(tmp_path/'check')],env=env,check=True,capture_output=True)
    records=sorted(dump.glob('*.bin'))
    assert len(records)==30
    names=[json.loads(subprocess.run([str(transport/'oracle_words'),'header',str(p),'-'],check=True,capture_output=True,text=True).stdout)['name'] for p in records]
    assert names.count('state.mpas_atmphys_rrtmg_lwinit.table')==3
    assert names.count('state.mpas_atmphys_vars.table')==3
    rows=[]
    for routine,expected in [('physics_init','initcounter'),('unrelated','nestedcounter')]:
        for phase in ['pre','post']:
            native=sorted(dump.glob('state__physics_init*.'+phase+'.local.mpas_atmphys_init.'+routine+'.counter.bin'))
            references=sorted(dump.glob('expected*.'+phase+'.'+expected+'.bin'))
            assert len(native)==len(references)==3
            for captured,reference in zip(native,references):
                value=json.loads(compare(transport,reference,captured).stdout)
                assert value['bitwise_equal']
                rows.append(value)
    (tmp_path/'init-lifecycle-proof.json').write_text(json.dumps(dict(
        schema='f1-whole-init-scope-local-lifecycle-v51',process_count=1,
        paths=['unconditional return','conditional return','normal exit'],
        words=sum(x['words'] for x in rows),different_words=0,
        scope_leak=False,global_hidden_state_complete=False),indent=2)+'\n')


@pytest.mark.parametrize('optional_output', [False, True])
def test_fixture_postcondition_runs_at_native_call_boundary(transport,tmp_path,optional_output):
    from replay.generate_fixture import fixture
    # An absent optional first output must not govern the scenario assertion.
    # An input-only interface still needs the same post-call assertion boundary.
    argument=lambda name,intent,optional=False:dict(name=name,type='real',dims=[],
        attrs=[],intent=intent,optional=optional,entry_storage=True)
    arguments=([argument('x','out',True)] if optional_output else [])+[argument('y','in')]
    signature='x,y' if optional_output else 'y'
    optional_declaration='real,optional,intent(out) :: x\nif(present(x)) x=y+1.0' if optional_output else ''
    native=f'''module native
contains
subroutine calculate({signature})
real,intent(in) :: y
{optional_declaration}
end subroutine
end module
'''
    site=dict(site='calculate__driver_example__L5',scheme='calculate',issues=[],
        arguments=arguments,path='driver.F90',line=5,end_line=5,
        callee=dict(module='native',path='native.F90',line=3,end_line=7))
    profile=dict(precision='real32',assignments={'y':'y+11.0_real32'},
        post_assertions=['y==14.0_real32'])
    passed=fixture(site,profile)
    failed=fixture(site,dict(profile,post_assertions=['y<0.0_real32']))
    boundary=passed.index('! oracle native call completed')
    assertion=passed.index("error stop 'native fixture postcondition failed'")
    assert boundary<assertion
    if optional_output:
        assert assertion<passed.index("'post','x.present'")
    pool_site=dict(site,arguments=[dict(name='fields',type='type(mpas_pool_type)',
        attrs=[],intent='inout',optional=False,pool_argument=True),argument('y','in')])
    pool_fixture=fixture(pool_site,profile)
    assert "'pre','fields'" in pool_fixture
    assert pool_fixture.index("error stop 'native fixture postcondition failed'")<pool_fixture.index("'post','fields'")
    (tmp_path/'native.F90').write_text(native)
    (tmp_path/'passed.F90').write_text(passed);(tmp_path/'failed.F90').write_text(failed)
    optional_seed="call oracle_write('calculate__driver_example__L5','pre','x.present',.false.,event)" if optional_output else ''
    (tmp_path/'seed.F90').write_text(f'''program seed
use iso_fortran_env,only:int32
use mpas_oracle_dump
implicit none
integer :: event
character(len=4096) :: prefix
call get_command_argument(1,prefix)
call oracle_set_context([101_int32],1,0)
call oracle_begin('calculate__driver_example__L5',event)
call oracle_set_output(trim(prefix))
{optional_seed}
call oracle_write('calculate__driver_example__L5','pre','y',3.0,event)
end program
''')
    (tmp_path/'Makefile').write_text(f'''FC=gfortran
FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -I{transport}
all: seed passed failed
native.o: native.F90
\t$(FC) $(FLAGS) -c $< -o $@
seed: seed.F90
\t$(FC) $(FLAGS) $< {transport}/transport.o -o $@
passed: passed.F90 native.o
\t$(FC) $(FLAGS) $^ {transport}/transport.o -o $@
failed: failed.F90 native.o
\t$(FC) $(FLAGS) $^ {transport}/transport.o -o $@
''')
    subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','f1-native-tap')),check=True,capture_output=True)
    inputs=tmp_path/'inputs';inputs.mkdir();prefix='calculate__driver_example__L5.s0000000001.e0000000001.r000000'
    subprocess.run([str(tmp_path/'seed'),str(inputs/prefix)],check=True,capture_output=True)
    subprocess.run([str(tmp_path/'passed'),str(inputs/prefix),str(tmp_path/'passed-result')],check=True,capture_output=True)
    result=subprocess.run([str(tmp_path/'failed'),str(inputs/prefix),str(tmp_path/'failed-result')],capture_output=True,text=True)
    assert result.returncode!=0 and 'native fixture postcondition failed' in result.stderr
    assert not list(tmp_path.glob('failed-result.post.*.bin'))


def test_fixture_optional_inputs_require_presence_before_any_perturbation(transport,tmp_path):
    from replay.generate_fixture import fixture
    argument=lambda name,intent,optional=False:dict(name=name,type='real',dims=[],
        attrs=[],intent=intent,optional=optional,entry_storage=True)
    site=dict(site='calculate__driver_example__L5',scheme='calculate',issues=[],
        arguments=[argument('humidity','in',True),argument('temperature','in'),
                   argument('result','out')],path='driver.F90',line=5,end_line=5,
        callee=dict(module='native',path='native.F90',line=3,end_line=9))
    profile=dict(precision='real32',required_present=['humidity'],
        assignments={'humidity':'max(humidity,0.025_real32)','temperature':'270.0_real32'},
        pre_assertions=['humidity>=0.025_real32'],post_assertions=['result==270.025_real32'])
    source=fixture(site,profile)
    assert source.index('required optional input absent')<source.index('humidity=max(')
    for invalid in [None,'humidity',['humidity','humidity'],['temperature'],['missing'],[1]]:
        with pytest.raises(ValueError,match='required_present'):
            fixture(site,dict(profile,required_present=invalid))
    with pytest.raises(ValueError,match='needs required_present'):
        fixture(site,dict(profile,required_present=[]))
    with pytest.raises(ValueError,match='expression optional input'):
        fixture(site,dict(precision='real32',assignments={'temperature':'temperature+humidity'}))
    with pytest.raises(ValueError,match='undefined OUT'):
        fixture(site,dict(precision='real32',assignments={'temperature':'result'}))
    with pytest.raises(ValueError,match='assertion optional input'):
        fixture(site,dict(precision='real32',assignments={'temperature':'270.0_real32'},
                         post_assertions=['humidity>0.0_real32']))
    (tmp_path/'fixture.F90').write_text(source)
    (tmp_path/'native.F90').write_text('''module native
contains
subroutine calculate(humidity,temperature,result)
real,optional,intent(in) :: humidity
real,intent(in) :: temperature
real,intent(out) :: result
result=temperature
if(present(humidity)) result=result+humidity
end subroutine
end module
''')
    (tmp_path/'seed.F90').write_text('''program seed
use iso_fortran_env,only:int32
use mpas_oracle_dump
implicit none
integer :: event
logical :: available
character(len=4096) :: prefix,mode
call get_command_argument(1,prefix)
call get_command_argument(2,mode)
available=mode=='present'
call oracle_set_context([101_int32],1,0)
call oracle_begin('calculate__driver_example__L5',event)
call oracle_set_output(trim(prefix))
call oracle_write('calculate__driver_example__L5','pre','humidity.present',available,event)
if(available) call oracle_write('calculate__driver_example__L5','pre','humidity',0.005,event)
call oracle_write('calculate__driver_example__L5','pre','temperature',290.0,event)
call oracle_write('calculate__driver_example__L5','pre','result',0.0,event)
end program
''')
    (tmp_path/'Makefile').write_text(f'''FC=gfortran
FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -I{transport}
all: seed fixture
native.o: native.F90
\t$(FC) $(FLAGS) -c $< -o $@
seed: seed.F90
\t$(FC) $(FLAGS) $< {transport}/transport.o -o $@
fixture: fixture.F90 native.o
\t$(FC) $(FLAGS) $^ {transport}/transport.o -o $@
''')
    subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,
        env=dict(os.environ,MPAS_UNIT='f1-native-tap'),check=True,capture_output=True)
    prefix='calculate__driver_example__L5.s0000000001.e0000000001.r000000'
    for mode in ['present','absent']:
        inputs=tmp_path/mode;inputs.mkdir()
        subprocess.run([str(tmp_path/'seed'),str(inputs/prefix),mode],check=True,capture_output=True)
        run=subprocess.run([str(tmp_path/'fixture'),str(inputs/prefix),str(inputs/'computed')],
                           capture_output=True,text=True)
        if mode=='present':
            assert run.returncode==0,run.stderr
            assert (inputs/'computed.post.result.bin').is_file()
        else:
            assert run.returncode!=0 and 'required optional input absent' in run.stderr
            assert not list(inputs.glob('computed.pre.*.bin'))
            assert not list(inputs.glob('computed.post.*.bin'))


def test_module_actual_alias_replay(transport,tmp_path,monkeypatch):
    from generate import inventory,overlay
    from state import generate_state
    root=tmp_path/'source';physics=root/'src/core_atmosphere/physics';physics.mkdir(parents=True)
    (physics/'native.F90').write_text('''module native
use iso_fortran_env,only:real64
real :: total=0.0
real,allocatable :: samples(:)
complex(kind=real64) :: dielectric=(0.0_real64,0.0_real64)
contains
subroutine calculate(x,y,warm)
real,intent(out) :: x
real,intent(inout) :: y(:)
logical,intent(in) :: warm
x=real(dielectric,kind(x))+real(aimag(dielectric),kind(x))
dielectric=dielectric+cmplx(3.0_real64,-4.0_real64,kind=real64)
y=y+3.0
if(warm) y=y+5.0
end subroutine calculate
end module native
''')
    (physics/'driver.F90').write_text('''module example
use native,only: calculate,renamed_total=>total,samples
contains
subroutine driver_example()
call calculate(renamed_total,samples,.false.)
end subroutine driver_example
end module example
''')
    (physics/'mpas_atmphys_init.F').write_text('''module init
contains
subroutine physics_init(mesh)
integer :: mesh
mesh=mesh
end subroutine physics_init
end module init
''')
    m=inventory(root,authenticate=False);site=m['sites'][0]
    assert not site['issues']
    assert site['arguments'][0]['module_actual']['name']=='total'
    assert site['arguments'][1]['module_actual']['name']=='samples'
    out=tmp_path/'generated';overlay(root,out,m);generate_state(root,out,m)
    (tmp_path/'replay.F90').write_text(replay_source(site))
    generated=out/'src/core_atmosphere/physics'
    (tmp_path/'capture.F90').write_text('''program capture
use iso_fortran_env,only:int32
use native
use example
use mpas_oracle_dump
implicit none
character(len=4096) :: directory
call get_command_argument(1,directory)
call oracle_set_context([101_int32,205_int32],1,0)
allocate(samples(-2:-1));samples=[5.0,7.0]
total=11.0
dielectric=cmplx(7.0_real64,13.0_real64,kind=real64)
call driver_example()
end program capture
''')
    (tmp_path/'Makefile').write_text(f'''FC=gfortran
FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -ffree-form -fconvert=big-endian -I{transport}
all: capture replay
native.o: {generated}/native.F90
\t$(FC) $(FLAGS) -c $< -o $@
driver.o: {generated}/driver.F90 native.o
\t$(FC) $(FLAGS) -c $< -o $@
capture: capture.F90 driver.o native.o
\t$(FC) $(FLAGS) $^ {transport}/transport.o -o $@
replay: replay.F90 native.o
\t$(FC) $(FLAGS) $^ {transport}/transport.o -o $@
''')
    subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','f1-native-tap')),check=True,capture_output=True)
    dump=tmp_path/'dump';dump.mkdir()
    env=dict(os.environ,MPAS_ORACLE_DIR=str(dump),MPAS_ORACLE_STEPS='1',MPAS_ORACLE_KEYS=site['site'])
    subprocess.run([str(tmp_path/'capture'),str(dump)],env=env,check=True,capture_output=True)
    prefix=next(dump.glob('*.pre.x.bin')).name.removesuffix('.pre.x.bin')
    output=tmp_path/'result'
    subprocess.run([str(tmp_path/'replay'),str(dump/prefix),str(output)],check=True,capture_output=True)
    for reference in dump.glob(prefix+'.post.*.bin'):
        name=reference.name[len(prefix):]
        assert json.loads(compare(transport,reference,tmp_path/('result'+name)).stdout)['bitwise_equal'],name
    assert list(dump.glob('*.post.state.native.total.bin'))
    assert list(dump.glob('*.pre.state.native.dielectric.bin'))
    # A module parameter has no mutable storage to alias, even when declared
    # using a separate PARAMETER statement. ONLY also prevents false imports.
    assert 'oracle_actual_x => total' in (tmp_path/'replay.F90').read_text()
    spec=importlib.util.spec_from_file_location('column_oracle_replay',ORACLE.parents[1]/'column_oracle_replay.py')
    harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
    with pytest.raises(ValueError,match='invalid dump member'):
        harness.safe_file(dump,'../capture.F90')
    link=dump/'external-link';link.symlink_to(tmp_path/'capture.F90')
    with pytest.raises(ValueError,match='invalid dump member'):
        harness.safe_file(dump,link.name)
    link.unlink()
    m['scheme_aliases']={'public_calculate':site['scheme']}
    m['scheme_groups']={'public_group':['public_calculate','uncaptured_native_scheme']}
    inventory_path=tmp_path/'inventory.json';inventory_path.write_text(json.dumps(m))
    pins=tmp_path/'PINS.txt';pins.write_text('native alias regression source\n')
    harness.seal(dump,inventory_path,pins,transport/'oracle_words','gnu-strict')
    import argparse
    options=argparse.Namespace(dump=dump,scheme='public_calculate',backend='native-replay',recipe='gnu-strict',
        results=tmp_path/'batch-results',receipt=tmp_path/'batch-receipts',executable_dir=tmp_path,
        executable_pattern='replay',words=transport/'oracle_words',coverage=None,gpu_slot_minutes=None)
    assert harness.batch(options)==0
    summary=json.loads((options.receipt/'batch.json').read_text())
    assert summary['all_outputs_bitwise_equal'] and len(summary['sites'])==1
    direct=argparse.Namespace(**vars(options))
    direct.executable=tmp_path/'replay';direct.results=tmp_path/'single-results'
    direct.receipt=tmp_path/'single-receipt.json'
    assert harness.replay(direct)==0
    result=json.loads(direct.receipt.read_text())
    assert result['resolved_scheme']==site['scheme'] and result['all_outputs_bitwise_equal']
    assert harness.selected_schemes({'scheme_groups':{'sfclay_mm5':['sfclay','sfclayrev']}},'sfclay_mm5')=={'sfclay','sfclayrev'}
    with pytest.raises(ValueError,match='cyclic'):
        harness.selected_schemes({'scheme_groups':{'a':['b'],'b':['a']}},'a')
    options.scheme='public_group';options.results=tmp_path/'group-results';options.receipt=tmp_path/'group-receipts'
    assert harness.batch(options)==0
    grouped=json.loads((options.receipt/'batch.json').read_text())
    assert grouped['all_outputs_bitwise_equal'] and len(grouped['sites'])==1
    options.scheme='public_calculate'
    assert harness.resolve_scheme({'scheme_aliases':{'oml':'oml1d'}},'oml')=='oml1d'
    public=json.loads((ORACLE/'scheme_aliases.json').read_text())
    assert harness.resolve_scheme({'scheme_aliases':public},'thompson')=='mp_gt_driver'
    assert harness.resolve_scheme({'scheme_aliases':{}},'ntiedtke')=='cu_ntiedtke_driver'
    assert harness.resolve_scheme({'scheme_aliases':{'ntiedtke':'older_native'}},'ntiedtke')=='older_native'
    # A new public name selects an old sealed capture without mutating its root.
    sealed_sha=harness.hash_file(dump/'manifest.json')
    catalog=tmp_path/'lookup-aliases.json'
    catalog.write_text(json.dumps({'new_public_calculate':site['scheme']}))
    monkeypatch.setattr(harness,'ALIAS_CATALOG',catalog)
    older=argparse.Namespace(**vars(direct));older.scheme='new_public_calculate'
    older.results=tmp_path/'new-name-results';older.receipt=tmp_path/'new-name-receipt.json'
    assert harness.replay(older)==0
    lookup=json.loads(older.receipt.read_text())
    assert lookup['all_outputs_bitwise_equal'] and lookup['resolved_scheme']==site['scheme']
    assert lookup['scheme_alias_contract']['catalog_sha256']==harness.hash_file(catalog)
    assert not lookup['scheme_alias_contract']['capture_manifest_changed']
    assert harness.hash_file(dump/'manifest.json')==sealed_sha
    with pytest.raises(ValueError,match='cyclic'):
        harness.resolve_scheme({'scheme_aliases':{'a':'b','b':'a'}},'a')
    options.results=tmp_path/'missing-results';options.receipt=tmp_path/'missing-receipts'
    options.executable_pattern='absent_replay'
    assert harness.batch(options)==1
    failure=json.loads((options.receipt/(site['site']+'.json')).read_text())
    assert not failure['comparison_complete'] and 'missing replay executable' in failure['reason']
    from replay.generate_fixture import fixture
    profile={'precision':'real32','assignments':{'y':'y+11.0_real32','warm':'.true.'},
             'pre_assertions':['all(y(:)>10.0_real32)'],
             'post_assertions':['all(y>=24.0_real32)']}
    perturbed=fixture(site,profile)
    (tmp_path/'fixture.F90').write_text(perturbed)
    failed_profile=dict(profile,post_assertions=['all(y<0.0_real32)'])
    (tmp_path/'failed_fixture.F90').write_text(fixture(site,failed_profile))
    with (tmp_path/'Makefile').open('a') as f:
        f.write(f'fixture: fixture.F90 native.o\n\t$(FC) $(FLAGS) $^ {transport}/transport.o -o $@\n')
        f.write(f'failed_fixture: failed_fixture.F90 native.o\n\t$(FC) $(FLAGS) $^ {transport}/transport.o -o $@\n')
    subprocess.run(_build_command(['make','-j4','fixture','failed_fixture']),cwd=tmp_path,env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','f1-native-tap')),check=True,capture_output=True)
    synthetic=tmp_path/'synthetic';synthetic.mkdir();synthetic_prefix=synthetic/prefix
    subprocess.run([str(tmp_path/'fixture'),str(dump/prefix),str(synthetic_prefix)],check=True,capture_output=True)
    subprocess.run([str(tmp_path/'fixture'),str(dump/prefix),str(synthetic/prefix)+'-second'],check=True,capture_output=True)
    failed=tmp_path/'failed-assertion';failed.mkdir()
    attempt=subprocess.run([str(tmp_path/'failed_fixture'),str(dump/prefix),str(failed/prefix)],capture_output=True,text=True)
    assert attempt.returncode!=0 and 'native fixture postcondition failed' in attempt.stderr
    assert not list(failed.glob('*.post.*.bin'))
    with pytest.raises(ValueError,match='undefined OUT'):
        fixture(site,dict(profile,pre_assertions=['x>0.0_real32']))
    with pytest.raises(ValueError,match='logical expression'):
        fixture(site,dict(profile,post_assertions=['all(y>0); stop']))
    changed=json.loads(compare(transport,dump/(prefix+'.pre.y.bin'),synthetic/(prefix+'.pre.y.bin')).stdout)
    assert changed['different_words']==2
    harness.seal(synthetic,inventory_path,pins,transport/'oracle_words','gnu-strict')
    options.dump=synthetic;options.results=tmp_path/'synthetic-results';options.receipt=tmp_path/'synthetic-receipts'
    options.executable_pattern='replay'
    assert harness.batch(options)==0
    # The directory ABI executes once, while every native call is independently
    # compared. All numeric work still runs in the compiled native executable.
    adapter=tmp_path/'whole-replay.py'
    adapter.write_text(f'''#!/usr/bin/env python3
import pathlib,subprocess,sys
source=pathlib.Path(sys.argv[1]);target=pathlib.Path(sys.argv[2]);target.mkdir(parents=True)
print('one root invocation')
prefixes=sorted({{p.name.split('.pre.')[0] for p in source.glob('*.pre.x.bin')}})
for prefix in prefixes:
 subprocess.run([{str(tmp_path/'replay')!r},str(source/prefix),str(target/prefix)],check=True)
''')
    adapter.chmod(0o755)
    options.scheme='public_group';options.executable=adapter;options.whole_capture=True
    options.results=tmp_path/'whole-results';options.receipt=tmp_path/'whole-receipt.json'
    scans=[];original_glob=Path.glob
    def track_glob(path,pattern):
        if path in {synthetic,options.results}:scans.append((path,pattern))
        return original_glob(path,pattern)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'glob',track_glob)
        assert harness.replay(options)==0
    whole=json.loads(options.receipt.read_text())
    assert whole['execution_mode']=='whole-capture' and len(whole['captures'])==2
    assert whole['replay_process_count']==1
    assert [pattern for path,pattern in scans if path==options.results]==['*.post.*.bin']
    assert all(not pattern.startswith(prefix) for _,pattern in scans)
    assert (options.results/'whole-capture.log').read_text().count('one root invocation')==1
    # Indexing must still reject an executable's extra output for either call.
    adapter.write_text(adapter.read_text()+f"\nimport shutil\nshutil.copyfile(target/(prefixes[-1]+'.post.x.bin'),target/(prefixes[-1]+'.post.unexpected.bin'))\n")
    options.results=tmp_path/'extra-whole-results';options.receipt=tmp_path/'extra-whole.json'
    with pytest.raises(ValueError,match='output set differs'):
        harness.replay(options)
    # Refuse mixed-site roots before executing the directory ABI.
    rejected=json.loads((synthetic/'manifest.json').read_text())
    rejected['captures']['unselected']={'site':'other_native_site'}
    rejected['sites'].append(dict(site,site='other_native_site',scheme='other_scheme'))
    options.results=tmp_path/'rejected-whole-results'
    with pytest.raises(ValueError,match='exactly the selected call site'):
        harness.replay(options,rejected)
    assert not options.results.exists()
    with pytest.raises(ValueError,match='precision mismatch'):
        fixture(site,{'precision':'real64','assignments':{'y':'y+11.0_real64'}})
    # Publish a real sealed native capture, preserving its authenticated bytes.
    import shutil
    from publish_dump import publish
    owned=tmp_path/'publication-source';shutil.copytree(dump,owned)
    publication=publish(owned,tmp_path/'campaign','native-group','f1-native-tap')
    final=Path(publication['path'])
    assert not owned.exists() and (final/'PUBLICATION.json').exists()
    assert harness.verify(final)['dump_root_sha256']==publication['dump_root_sha256']
    assert not list(final.parent.glob('.partial-*'))
    retained=tmp_path/'retained-publication-source';shutil.copytree(dump,retained)
    linked=publish(retained,tmp_path/'campaign','native-linked','f1-native-tap',keep_source=True)
    linked_root=Path(linked['path'])
    assert retained.is_dir() and linked['source_retained']
    assert harness.verify(retained)['dump_root_sha256']==harness.verify(linked_root)['dump_root_sha256']
    member=next(retained.glob('*.bin'))
    assert member.samefile(linked_root/member.name)
    assert not (retained/'manifest.json').samefile(linked_root/'manifest.json')
    (linked_root/member.name).unlink()
    assert harness.verify(retained)['dump_root_sha256']==linked['dump_root_sha256']
    with pytest.raises(ValueError,match='member set differs'):
        harness.verify(linked_root)


def test_calendar_pointer_alias_restore(transport,tmp_path):
    from generate import inventory,overlay
    from state import generate_state
    root=tmp_path/'source';physics=root/'src/core_atmosphere/physics';physics.mkdir(parents=True)
    definitions={
        'src/framework/mpas_timekeeping_types.inc':'''type mpas_timeinterval_type
 type(esmf_timeinterval) :: ti
end type
''',
        'src/external/esmf_time_f90/ESMF_TimeIntervalMod.F90':'''type esmf_timeinterval
 type(esmf_time) :: starttime
 logical :: starttime_set=.true.
end type
''',
        'src/external/esmf_time_f90/ESMF_ShrTimeMod.F90':'''type esmf_time
 type(esmf_calendar),pointer :: calendar=>null()
end type
''',
        'src/external/esmf_time_f90/ESMF_BaseTimeMod.F90':'',
        'src/external/esmf_time_f90/ESMF_CalendarMod.F90':'''type esmf_calendar
 integer :: day
end type
'''}
    for name,text in definitions.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    # Calendar pointers must use the provider's actual type, just as native
    # MPAS does. An inline lookalike cannot associate with its global targets.
    (tmp_path/'calendar.F90').write_text('''module ESMF_CalendarMod
type esmf_calendar
 integer :: day
end type
type(esmf_calendar),target :: default_target=esmf_calendar(31)
type(esmf_calendar),pointer :: defaultcal=>default_target
type(esmf_calendar),pointer :: gregoriancal=>null(),noleapcal=>null()
end module
''')
    (physics/'native.F90').write_text('module native\nuse ESMF_CalendarMod,only:esmf_calendar\n'+''.join(
        definitions[n] for n in reversed(list(definitions)) if 'CalendarMod' not in n)+'''type(mpas_timeinterval_type) :: first,second,third,absent,global_first,global_second
contains
subroutine calculate(x)
integer,intent(out) :: x
first%ti%starttime%calendar%day=first%ti%starttime%calendar%day+1
global_first%ti%starttime%calendar%day=global_first%ti%starttime%calendar%day+2
x=second%ti%starttime%calendar%day+third%ti%starttime%calendar%day+global_second%ti%starttime%calendar%day
end subroutine
end module native
''')
    (physics/'driver.F90').write_text('''module example
use native
contains
subroutine driver_example()
integer :: x
call calculate(x)
end subroutine
end module example
''')
    (physics/'mpas_atmphys_init.F').write_text('''module init
contains
subroutine physics_init(mesh)
integer :: mesh
mesh=mesh
end subroutine physics_init
end module init
''')
    m=inventory(root,authenticate=False);site=m['sites'][0];out=tmp_path/'generated';overlay(root,out,m);generate_state(root,out,m)
    info=next(i for i in m['module_state'] if i['name']=='native')
    assert not any('calendar pointer alias' in i['reason'] for i in info['issues'])
    generated=out/'src/core_atmosphere/physics'
    (tmp_path/'replay.F90').write_text(replay_source(site))
    replay=(tmp_path/'replay.F90').read_text().replace(
        '  call calculate(',
        "  if(.not.associated(global_first%ti%starttime%calendar,defaultcal)) error stop 'lost external target'\n"
        "  if(.not.associated(global_second%ti%starttime%calendar,defaultcal)) error stop 'lost shared external target'\n"
        '  call calculate(')
    # The replay imports routines with ONLY; expose the pointer assertions.
    replay=replay.replace('  use native, only: calculate',
        '  use ESMF_CalendarMod,only:defaultcal\n  use native, only: calculate,global_first,global_second')
    (tmp_path/'replay.F90').write_text(replay)
    (tmp_path/'capture.F90').write_text('''program capture
use iso_fortran_env,only:int32
use ESMF_CalendarMod,only:defaultcal
use native
use mpas_oracle_dump
implicit none
character(len=4096) :: prefix
integer :: x
call get_command_argument(1,prefix)
allocate(first%ti%starttime%calendar)
first%ti%starttime%calendar%day=17
second%ti%starttime%calendar=>first%ti%starttime%calendar
! Equal values in distinct targets must retain distinct alias classes.
allocate(third%ti%starttime%calendar)
third%ti%starttime%calendar%day=17
nullify(absent%ti%starttime%calendar)
global_first%ti%starttime%calendar=>defaultcal
global_second%ti%starttime%calendar=>defaultcal
call oracle_set_context([101_int32],1,0)
call oracle_set_output(trim(prefix))
x=0
call oracle_write('calculate__driver_example__L6','pre','x',x,1)
call oracle_state_native('calculate__driver_example__L6','pre',1)
call calculate(x)
call oracle_write('calculate__driver_example__L6','post','x',x,1)
call oracle_state_native('calculate__driver_example__L6','post',1)
end program
'''.replace('calculate__driver_example__L6',site['site']))
    (tmp_path/'Makefile').write_text(f'''FC=gfortran
FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-form -ffree-line-length-none -fconvert=big-endian -I{transport}
all: capture replay
calendar.o: calendar.F90
	$(FC) $(FLAGS) -c $< -o $@
native.o: {generated}/native.F90 calendar.o
	$(FC) $(FLAGS) -c $< -o $@
capture: capture.F90 native.o calendar.o
	$(FC) $(FLAGS) $^ {transport}/transport.o -o $@
replay: replay.F90 native.o calendar.o
	$(FC) $(FLAGS) $^ {transport}/transport.o -o $@
''')
    subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','f1-native-tap')),check=True,capture_output=True)
    subprocess.run([str(tmp_path/'capture'),str(tmp_path/'captured')],check=True,capture_output=True)
    subprocess.run([str(tmp_path/'replay'),str(tmp_path/'captured'),str(tmp_path/'replayed')],check=True,capture_output=True)
    for reference in tmp_path.glob('captured.post.*.bin'):
        candidate=tmp_path/reference.name.replace('captured.','replayed.',1)
        assert json.loads(compare(transport,reference,candidate).stdout)['bitwise_equal'],reference.name
    assert len(list(tmp_path.glob('captured.pre.*.alias.bin')))==6
    # Presence-only derived containers must pass the actual sealed replay door.
    spec=importlib.util.spec_from_file_location('column_oracle_replay',ORACLE.parents[1]/'column_oracle_replay.py')
    harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
    dump=tmp_path/'dump';dump.mkdir()
    for source in tmp_path.glob('captured.*.bin'):
        (dump/source.name).write_bytes(source.read_bytes())
    inventory_path=tmp_path/'inventory.json';inventory_path.write_text(json.dumps(m))
    pins=tmp_path/'PINS.txt';pins.write_text('calendar alias native regression\n')
    harness.seal(dump,inventory_path,pins,transport/'oracle_words','gnu-strict')
    import argparse
    options=argparse.Namespace(dump=dump,scheme=site['site'],backend='native-replay',recipe='gnu-strict',
        results=tmp_path/'door-results',receipt=tmp_path/'door-receipt.json',executable=tmp_path/'replay',
        words=transport/'oracle_words',coverage=None,gpu_slot_minutes=None)
    assert harness.replay(options)==0


def test_clock_graph_with_actual_native_framework(transport,tmp_path):
    """Native graph: cycles, shared targets, exact bounds and inactive input."""
    from clock_graph import generate
    import re
    native=os.environ.get('MPAS_ORACLE_NATIVE_BUILD')
    if not native:
        pytest.skip('set MPAS_ORACLE_NATIVE_BUILD to an owned strict MPAS build')
    native=Path(native)
    source=site_support.mpas_source(preferred=('MPAS_ORACLE_SOURCE',))
    generated,info=generate(source)
    (tmp_path/'clock.F90').write_text(generated)
    includes=sorted({p.parent for p in native.rglob('*.mod')})
    def initialize(typ,value):
        text=''
        for field in info['types'][typ]['fields']:
            expr=value+'%'+field['name']
            if 'pointer' in field['attrs']:continue
            nested=re.fullmatch(r'type\s*\(\s*(\w+)\s*\)',field['type'],re.I)
            if nested:text+=initialize(nested[1].lower(),expr)
            else:
                if field['type'].startswith('character'):literal="'native clock state'"
                elif field['type'].startswith('logical'):literal='.true.'
                elif 'esmf_kind_i8' in field['type'].lower():literal='1234567890123456_8'
                else:literal='123'
                text+=f' {expr}={literal}\n'
        return text
    fixture='''program fixture
 use mpas_timekeeping,only:MPAS_Clock_type,MPAS_Alarm_type
 use mpas_derived_types,only:domain_type
 use ESMF_ClockMod,only:ESMF_ClockInt
 use ESMF_AlarmMod,only:ESMF_AlarmInt
 use ESMF_CalendarMod,only:defaultcal,ESMF_Calendar
 use mpas_oracle_clock
 use mpas_oracle_dump
 implicit none
 type(MPAS_Clock_type),pointer::clock=>null(),restored=>null()
 type(domain_type)::domain
 type(ESMF_ClockInt),pointer::held_clockint=>null()
 type(ESMF_AlarmInt),pointer::held_alarmint=>null()
 type(MPAS_Alarm_type),pointer::held_head=>null()
 type(ESMF_Calendar),pointer::localcal=>null()
 character(len=4096)::prefix,mode
 character(len=32)::ordinal
 integer::iteration
 call get_command_argument(1,prefix)
 call get_command_argument(2,mode)
 allocate(defaultcal,localcal,clock)
 allocate(clock%c%clockint)
 allocate(clock%c%clockint%alarmlist(-2:0))
 allocate(clock%c%clockint%alarmlist(-2)%alarmint)
 clock%c%clockint%alarmlist(-1)%alarmint=>clock%c%clockint%alarmlist(-2)%alarmint
 nullify(clock%c%clockint%alarmlist(0)%alarmint)
 allocate(clock%alarmListHead)
 allocate(clock%alarmListHead%next)
 clock%alarmListHead%next%next=>clock%alarmListHead
'''
    for typ,expr in [('esmf_calendar','defaultcal'),('esmf_calendar','localcal'),
        ('mpas_clock_type','clock'),('esmf_clockint','clock%c%clockint'),
        ('esmf_alarmint','clock%c%clockint%alarmlist(-2)%alarmint'),
        ('mpas_alarm_type','clock%alarmListHead'),('mpas_alarm_type','clock%alarmListHead%next')]:
        fixture+=initialize(typ,expr)
    fixture+=''' clock%c%clockint%starttime%calendar=>defaultcal
 clock%c%clockint%stoptime%calendar=>localcal
 clock%c%clockint%currtime%calendar=>localcal
 clock%c%clockint%timestep%starttime%calendar=>defaultcal
 clock%c%clockint%alarmlist(-2)%alarmint%ringtime%calendar=>localcal
 clock%alarmListHead%ringtime%t%calendar=>defaultcal
 clock%alarmListHead%next%ringtime%t%calendar=>localcal
 clock%alarmListHead%ringTimeInterval%ti%starttime_set=.false.
 call oracle_set_context([101],1,0)
 if(trim(mode)/='disabled')call oracle_set_output(trim(prefix))
 if(trim(mode)=='absent')nullify(clock)
 if(trim(mode)=='bad')then
  call oracle_write('clock','pre','state.clock.present',.true.,1)
  call oracle_write('clock','pre','state.clock.target',999,1)
 elseif(trim(mode)/='load')then
  call oracle_clock_capture('clock','pre','state.clock',clock,1)
 endif
 if(trim(mode)=='capture'.or.trim(mode)=='disabled')stop
 restored=>clock
 domain%clock=>clock
 if(associated(clock))then
 held_clockint=>clock%c%clockint
 held_alarmint=>clock%c%clockint%alarmlist(-2)%alarmint
 held_head=>clock%alarmListHead
 endif
 do iteration=1,2
  call oracle_clock_restore(trim(prefix),'state.clock',restored)
  if(trim(mode)=='absent')then
   if(associated(restored))error stop 'lost absent root'
   stop
  endif
  if(.not.associated(restored,domain%clock))error stop 'detached native domain clock root'
  if(.not.associated(restored%c%clockint,held_clockint))error stop 'detached compatible native clockint'
  if(.not.associated(restored%c%clockint%alarmlist(-2)%alarmint,held_alarmint))error stop 'detached compatible native alarmint'
  if(.not.associated(restored%alarmListHead,held_head))error stop 'detached compatible native alarm head'
  if(.not.associated(restored%alarmListHead%next%next,restored%alarmListHead))error stop 'lost cycle'
  if(.not.associated(restored%c%clockint%alarmlist(-2)%alarmint,restored%c%clockint%alarmlist(-1)%alarmint))error stop 'lost shared alarm'
  if(associated(restored%c%clockint%alarmlist(0)%alarmint))error stop 'lost absent alarm'
  if(lbound(restored%c%clockint%alarmlist,1)/=-2.or.ubound(restored%c%clockint%alarmlist,1)/=0)error stop 'lost native lower bounds'
  if(.not.associated(restored%c%clockint%starttime%calendar,defaultcal))error stop 'lost global calendar'
  if(.not.associated(restored%c%clockint%stoptime%calendar,restored%c%clockint%currtime%calendar))error stop 'lost calendar alias'
  if(associated(restored%c%clockint%stoptime%calendar,defaultcal))error stop 'merged equal calendars'
  if(restored%c%clockint%advancecount/=1234567890123456_8)error stop 'lost native int64'
  write(ordinal,'(i0)')iteration
  call oracle_set_output(trim(prefix)//'.restored'//trim(ordinal))
  call oracle_clock_capture('clock','post','state.clock',restored,iteration)
  restored%c%clockint%advancecount=-1_8
  if(iteration==1)then
   ! Restore distinct nodes after existing pointers were merged into a cycle
   ! or a provider calendar. This must not overwrite the first node/provider.
   restored%alarmListHead%next=>restored%alarmListHead
   restored%c%clockint%stoptime%calendar=>defaultcal
   nullify(restored%c%clockint%currtime%calendar)
  endif
 enddo
end program
'''
    (tmp_path/'fixture.F90').write_text(fixture)
    flags='-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-form -ffree-line-length-none -fconvert=big-endian -fcheck=all'
    flags+=' '+ ' '.join(f'-I{p}' for p in [transport,*includes])
    (tmp_path/'Makefile').write_text(f'''FC=mpif90
FLAGS={flags}
all: fixture
clock.o: clock.F90
\t$(FC) $(FLAGS) -c $< -o $@
fixture: fixture.F90 clock.o
\t$(FC) $(FLAGS) $^ {transport}/transport.o -Wl,--start-group {native}/src/framework/libframework.a {native}/src/external/esmf_time_f90/libesmf_time.a -Wl,--end-group -lnetcdff -lnetcdf -o $@
''')
    result=subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,
        env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','f1-native-tap')),capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    result=subprocess.run([str(tmp_path/'fixture'),str(tmp_path/'roundtrip'),'roundtrip'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    pre=list(tmp_path.glob('roundtrip.pre.*.bin'))
    assert len(pre)>100
    for original in pre:
        for iteration in (1,2):
            other=tmp_path/original.name.replace('.pre.',f'.restored{iteration}.post.',1)
            assert json.loads(compare(transport,original,other).stdout)['bitwise_equal'],original.name
    from clock_graph import audit_membership
    from functools import lru_cache
    graph_rows=[]
    @lru_cache(maxsize=None)
    def read_record(operation,filename,indices):
        return json.loads(subprocess.run([str(transport/'oracle_words'),operation,
            str(tmp_path/filename),indices],check=True,capture_output=True,text=True).stdout)
    for prefix,phase in [('roundtrip','pre'),('roundtrip.restored1','post'),
                         ('roundtrip.restored2','post')]:
        filenames={p.name for p in tmp_path.glob(prefix+'.'+phase+'.*.bin')}
        graph=audit_membership(info,prefix,phase,'state.clock',filenames,read_record)
        assert graph['complete'],json.dumps(graph,indent=2)
        assert graph['typed_node_counts']['mpas_alarm_type']==2
        assert graph['typed_node_counts']['esmf_alarmint']==1
        assert graph['calendar_provider_references']==['defaultcal']
        assert not graph['external_aliases_complete']
        rootname='state.clock'
        graph_rows.append(dict(prefix=prefix,phase=phase,audit=graph))
        removed=prefix+'.'+phase+'.state.clock.node.esmf_clockint.1.numalarms.bin'
        incomplete=audit_membership(info,prefix,phase,rootname,filenames-{removed},read_record)
        assert not incomplete['complete']and removed in incomplete['missing']
        orphan=prefix+'.'+phase+'.state.clock.node.esmf_clockint.2.numalarms.bin'
        incomplete=audit_membership(info,prefix,phase,rootname,filenames|{orphan},read_record)
        assert not incomplete['complete']and orphan in incomplete['unexpected']
    result=subprocess.run([str(tmp_path/'fixture'),str(tmp_path/'invalid'),'bad'],capture_output=True,text=True)
    assert result.returncode!=0 and 'forward reference' in result.stderr
    invalid=audit_membership(info,'invalid','pre','state.clock',
        {p.name for p in tmp_path.glob('invalid.pre.*.bin')},read_record)
    assert not invalid['complete']and any('forward reference'in x['reason']for x in invalid['topology_issues'])
    disabled=tmp_path/'disabled';disabled.mkdir()
    subprocess.run([str(tmp_path/'fixture'),str(disabled/'capture'),'disabled'],
        env=dict(os.environ,MPAS_ORACLE_DIR=str(disabled),MPAS_ORACLE_STEPS='1',
                 MPAS_ORACLE_KEYS='another_key'),check=True,capture_output=True)
    assert not list(disabled.glob('*.bin'))
    subprocess.run([str(tmp_path/'fixture'),str(tmp_path/'absent'),'absent'],check=True,capture_output=True)
    assert len(list(tmp_path.glob('absent.pre.*.bin')))==2
    absent=audit_membership(info,'absent','pre','state.clock',
        {p.name for p in tmp_path.glob('absent.pre.*.bin')},read_record)
    assert absent['complete']and sum(absent['typed_node_counts'].values())==0
    graph_rows.append(dict(prefix='absent',phase='pre',audit=absent))
    (tmp_path/'clock-membership-proof.json').write_text(json.dumps(dict(process_count=1,
        graphs=graph_rows,native_source_sha256=info['source_sha256'],
        generated_source_sha256=__import__('hashlib').sha256((tmp_path/'clock.F90').read_bytes()).hexdigest(),
        executable_sha256=__import__('hashlib').sha256((tmp_path/'fixture').read_bytes()).hexdigest(),
        invalid_forward_reference_refused=True,missing_leaf_refused=True,
        orphan_node_refused=True,global_hidden_state_complete=False),indent=2)+'\n')
    missing=tmp_path/'roundtrip.pre.state.clock.node.esmf_clockint.1.numalarms.bin'
    missing.unlink()
    result=subprocess.run([str(tmp_path/'fixture'),str(tmp_path/'roundtrip'),'load'],capture_output=True,text=True)
    assert result.returncode!=0 and 'missing input record' in result.stderr


def test_timer_graph_with_actual_native_framework(transport,tmp_path):
    """Native timer types: cycles, shared R4/R8 arrays, bounds, MPI metadata."""
    from timer_graph import generate,audit_membership
    from functools import lru_cache
    native=os.environ.get('MPAS_ORACLE_NATIVE_BUILD')
    if not native:pytest.skip('set MPAS_ORACLE_NATIVE_BUILD to an owned strict MPAS build')
    native=Path(native)
    source=site_support.mpas_source(preferred=('MPAS_ORACLE_SOURCE',))
    generated,info=generate(source)
    assert set(info['types'])=={'mpas_timer_root','mpas_timer_node','dm_info'}
    (tmp_path/'timer.F90').write_text(generated)
    fixture='''program fixture
 use mpas_kind_types
 use mpas_derived_types,only:mpas_timer_root,mpas_timer_node,dm_info,domain_type
 use mpas_timer,only:mpas_timer_set_context
 use mpas_oracle_timer
 use mpas_oracle_dump
 implicit none
 type(mpas_timer_root),pointer::timer=>null(),restored=>null()
 type(domain_type)::domain
 type(mpas_timer_node),pointer::held=>null()
 type(dm_info),pointer::held_dm=>null()
 character(len=4096)::prefix,mode
 character(len=32)::ordinal
 integer::iteration,timer_test_unit
 call get_command_argument(1,prefix)
 call get_command_argument(2,mode)
 allocate(timer)
 allocate(timer%root_timer,timer%dminfo)
 allocate(timer%root_timer%child)
 timer%root_timer%child%parent=>timer%root_timer
 timer%current_timer=>timer%root_timer%child
 timer%root_timer%next=>timer%root_timer%child
 timer%root_timer%child%next=>timer%root_timer
 timer%root_timer%timer_name='whole native timer root'
 timer%root_timer%child%timer_name='whole native timer child'
'''
    for expr in ('timer%root_timer','timer%root_timer%child'):
        fixture+=f''' {expr}%printed=.false.
 {expr}%nlen=17
 {expr}%calls=123
 allocate({expr}%running(-2:0))
 {expr}%running=[.false.,.true.,.false.]
 allocate({expr}%start_time(-2:0),{expr}%total_time(-2:0))
 {expr}%start_time=[1.234567890123456_r8kind,huge(1.0_r8kind),-tiny(1.0_r8kind)]
 {expr}%end_time(4:)=>{expr}%start_time
 {expr}%total_time=-{expr}%start_time
 allocate({expr}%max_time(-1:1),{expr}%avg_time(-1:1))
 {expr}%max_time=[tiny(1.0_rkind),-1.25_rkind,huge(1.0_rkind)]
 {expr}%min_time(7:)=>{expr}%max_time
 {expr}%avg_time=-{expr}%max_time
'''
    fixture+=''' timer%dminfo%comm=transfer(27,timer%dminfo%comm)
 timer%dminfo%info=transfer(28,timer%dminfo%info)
 timer%dminfo%nprocs=1
 timer%dminfo%my_proc_id=0
 timer%dminfo%initialized_mpi=.true.
 timer%dminfo%total_blocks=3
 timer%dminfo%explicitDecomp=.true.
 allocate(timer%dminfo%block_proc_list(-3:-1))
 timer%dminfo%block_proc_list=[31,32,33]
 domain%timer_root=>timer
 domain%dminfo=>timer%dminfo
 call mpas_timer_set_context(domain)
 held=>timer%root_timer
 held_dm=>timer%dminfo
 restored=>timer
 call oracle_set_context([101],1,0)
 if(trim(mode)/='disabled')call oracle_set_output(trim(prefix))
 if(trim(mode)=='absent')nullify(timer,restored)
 if(trim(mode)=='bad')then
  call oracle_write('timer','pre','state.timer.target',999,1)
 elseif(trim(mode)/='load')then
  call oracle_timer_capture('timer','pre','state.timer',timer,1)
 endif
 if(trim(mode)=='badalias')then
  open(newunit=timer_test_unit,file=trim(prefix)//'.pre.state.timer.node.mpas_timer_node.1.end_time.bounds.bin',status='old')
  close(timer_test_unit,status='delete')
  call oracle_write('timer','pre','state.timer.node.mpas_timer_node.1.end_time.bounds',[4,2],1)
 endif
 if(trim(mode)=='capture'.or.trim(mode)=='disabled')stop
 if(trim(mode)=='load')nullify(restored)
 do iteration=1,2
  call oracle_timer_restore(trim(prefix),'state.timer',restored)
  if(trim(mode)=='absent')then
   if(associated(restored))error stop 'lost absent timer root'
   stop
  endif
  if(trim(mode)/='load')then
   if(.not.associated(restored,domain%timer_root))error stop 'detached domain timer root'
   if(.not.associated(restored%root_timer,held))error stop 'detached compatible timer node'
   if(.not.associated(restored%dminfo,held_dm))error stop 'detached compatible MPI metadata'
  endif
  if(.not.associated(restored%root_timer%child%parent,restored%root_timer))error stop 'lost parent cycle'
  if(.not.associated(restored%root_timer%child%next,restored%root_timer))error stop 'lost next cycle'
  if(.not.associated(restored%current_timer,restored%root_timer%child))error stop 'lost current timer alias'
  if(.not.associated(restored%root_timer%start_time,restored%root_timer%end_time))error stop 'lost R8 target alias'
  if(lbound(restored%root_timer%end_time,1)/=4)error stop 'lost R8 view bounds'
  if(lbound(restored%root_timer%start_time,1)/=-2)error stop 'lost R8 target bounds'
  if(.not.associated(restored%root_timer%max_time,restored%root_timer%min_time))error stop 'lost R4 target alias'
  if(lbound(restored%root_timer%min_time,1)/=7)error stop 'lost R4 view bounds'
  if(lbound(restored%dminfo%block_proc_list,1)/=-3)error stop 'lost allocated integer bounds'
  if(allocated(restored%dminfo%block_local_id_list))error stop 'lost absent integer allocation'
  write(ordinal,'(i0)')iteration
  call oracle_set_output(trim(prefix)//'.restored'//trim(ordinal))
  call oracle_timer_capture('timer','post','state.timer',restored,iteration)
  restored%root_timer%start_time=-1.0_r8kind
  restored%dminfo%comm=transfer(-1,restored%dminfo%comm)
  if(iteration==1)then
   restored%root_timer%child=>restored%root_timer
   nullify(restored%current_timer)
   restored%root_timer%end_time=>restored%root_timer%total_time
   allocate(restored%dminfo%block_local_id_list(2:3))
  endif
 enddo
end program
'''
    (tmp_path/'fixture.F90').write_text(fixture)
    includes=sorted({p.parent for p in native.rglob('*.mod')})
    flags='-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-form -ffree-line-length-none -fconvert=big-endian -fcheck=all'
    flags+=' '+' '.join(f'-I{p}'for p in [transport,*includes])
    (tmp_path/'Makefile').write_text(f'''FC=mpif90
FLAGS={flags}
all: fixture
timer.o: timer.F90
\t$(FC) $(FLAGS) -c $< -o $@
fixture: fixture.F90 timer.o
\t$(FC) $(FLAGS) $^ {transport}/transport.o -Wl,--start-group {native}/src/framework/libframework.a {native}/src/external/esmf_time_f90/libesmf_time.a -Wl,--end-group -lnetcdff -lnetcdf -o $@
''')
    r=subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,env=dict(os.environ,MPAS_UNIT='f1-native-tap'),capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    for mode in ('roundtrip','capture','load','absent'):
        prefix=tmp_path/('capture'if mode=='load'else mode)
        r=subprocess.run([str(tmp_path/'fixture'),str(prefix),mode],capture_output=True,text=True)
        assert r.returncode==0,r.stdout+r.stderr
    rows=[]
    @lru_cache(maxsize=None)
    def read(operation,filename,indices):
        return json.loads(subprocess.run([str(transport/'oracle_words'),operation,str(tmp_path/filename),indices],check=True,capture_output=True,text=True).stdout)
    words=0
    for prefix in ('roundtrip','capture'):
        originals=list(tmp_path.glob(prefix+'.pre.*.bin'))
        assert len(originals)>60
        for original in originals:
            for iteration in (1,2):
                other=tmp_path/original.name.replace('.pre.',f'.restored{iteration}.post.',1)
                stats=json.loads(compare(transport,original,other).stdout)
                assert stats['bitwise_equal'],original.name
                words+=stats['words']
        for suffix,phase in [('', 'pre'),('.restored1','post'),('.restored2','post')]:
            context=prefix+suffix
            filenames={p.name for p in tmp_path.glob(context+'.'+phase+'.*.bin')}
            audit=audit_membership(info,context,phase,'state.timer',filenames,read)
            assert audit['complete'],json.dumps(audit,indent=2)
            assert audit['typed_node_counts']['mpas_timer_node']==2
            assert not audit['mpi_resources_restored'] and not audit['external_aliases_complete']
            rows.append(dict(prefix=context,phase=phase,audit=audit))
            removed=context+'.'+phase+'.state.timer.node.dm_info.1.comm.bin'
            bad=audit_membership(info,context,phase,'state.timer',filenames-{removed},read)
            assert not bad['complete']and removed in bad['missing']
            orphan=context+'.'+phase+'.state.timer.node.dm_info.2.comm.bin'
            bad=audit_membership(info,context,phase,'state.timer',filenames|{orphan},read)
            assert not bad['complete']and orphan in bad['unexpected']
    r=subprocess.run([str(tmp_path/'fixture'),str(tmp_path/'invalid'),'bad'],capture_output=True,text=True)
    assert r.returncode!=0 and 'forward reference'in r.stderr
    audit=audit_membership(info,'invalid','pre','state.timer',{p.name for p in tmp_path.glob('invalid.pre.*.bin')},read)
    assert not audit['complete'] and any('forward reference'in x['reason']for x in audit['topology_issues'])
    r=subprocess.run([str(tmp_path/'fixture'),str(tmp_path/'alias-extent'),'badalias'],capture_output=True,text=True)
    assert r.returncode!=0 and 'alias extent differs'in r.stderr
    audit=audit_membership(info,'alias-extent','pre','state.timer',{p.name for p in tmp_path.glob('alias-extent.pre.*.bin')},read)
    assert not audit['complete']and any('alias extent differs'in x['reason']for x in audit['topology_issues'])
    audit=audit_membership(info,'absent','pre','state.timer',{p.name for p in tmp_path.glob('absent.pre.*.bin')},read)
    assert audit['complete']and sum(audit['typed_node_counts'].values())==0
    disabled=tmp_path/'disabled';disabled.mkdir()
    subprocess.run([str(tmp_path/'fixture'),str(disabled/'capture'),'disabled'],env=dict(os.environ,MPAS_ORACLE_DIR=str(disabled),MPAS_ORACLE_STEPS='1',MPAS_ORACLE_KEYS='another_key'),check=True,capture_output=True)
    assert not list(disabled.glob('*.bin'))
    proof=dict(process_count=1,graphs=rows,words=words,different_words=0,
        native_source_sha256=info['source_sha256'],generated_source_sha256=__import__('hashlib').sha256(generated.encode()).hexdigest(),
        executable_sha256=__import__('hashlib').sha256((tmp_path/'fixture').read_bytes()).hexdigest(),
        cyclic_graph_restored=True,primitive_alias_views_restored=True,
        external_aliases_complete=False,mpi_resources_restored=False,wall_clock_reproducibility_proven=False,
        missing_leaf_refused=True,orphan_node_refused=True,forward_reference_refused=True)
    (tmp_path/'timer-membership-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    (tmp_path/'roundtrip.pre.state.timer.node.dm_info.1.comm.bin').unlink()
    r=subprocess.run([str(tmp_path/'fixture'),str(tmp_path/'roundtrip'),'load'],capture_output=True,text=True)
    assert r.returncode!=0 and 'missing input record'in r.stderr


def test_stream_graph_with_actual_native_framework(transport,tmp_path):
    """All eight native connectivity roots, metadata, cycles and array aliases."""
    from stream_graph import generate,audit_membership,instrument_stream_writes
    from state import modules
    from functools import lru_cache
    native=os.environ.get('MPAS_ORACLE_NATIVE_BUILD')
    if not native:pytest.skip('set MPAS_ORACLE_NATIVE_BUILD to an owned strict MPAS build')
    native=Path(native);source=site_support.mpas_source(preferred=('MPAS_ORACLE_SOURCE',))
    generated,info=generate(source);roots=info['root_names'];assert len(roots)==8
    observed=instrument_stream_writes((source/'src/framework/mpas_stream_manager.F').read_text())
    pre=observed.index('call prewrite_reindex(');post=observed.index('call postwrite_reindex(')
    assert pre<observed.index("'stream_reindex','pre'")<observed.index('call MPAS_writeStream(stream % stream')
    assert observed.index("'stream_reindex','post'")<post
    inventory=modules(source)['mpas_stream_manager']
    assert len(inventory['variables'])==1 and inventory['variables'][0]['graph_stream']==info
    assert {r['name']for r in inventory['roots']}==set(roots)
    (tmp_path/'stream.F90').write_text(generated)
    rootvars=','.join(roots);a,b=roots[:2]
    initialize=''
    for f in info['fields']:
        if 'pointer'in f['attrs']:continue
        value="'observed native metadata'"if f['type'].lower().startswith('character')else'.true.'if f['type'].lower().startswith('logical')else'17'
        initialize+=f" p%{f['name']}={value}\n"
    fixture=f'''program fixture
 use mpas_kind_types
 use mpas_derived_types,only:field2DInteger
 use mpas_oracle_stream
 use mpas_oracle_dump
 implicit none
 type(field2DInteger),pointer::{rootvars}
 type(field2DInteger),pointer::held
 character(len=4096)::prefix,mode
 character(len=32)::ordinal
 integer::iteration,unit
 call get_command_argument(1,prefix)
 call get_command_argument(2,mode)
 nullify({rootvars})
 allocate({a},{b})
 call initialize({a})
 call initialize({b})
 {a}%next=>{b}
 {b}%prev=>{a}
 {b}%next=>{a}
 {roots[2]}=>{a}
 allocate({a}%array(-2:0,4:5))
 {a}%array=reshape([1,2,3,4,5,6],[3,2])
 {b}%array(-9:,7:)=>{a}%array
 allocate({a}%constituentNames(-5:-4))
 {a}%constituentNames=['first ','second']
 {b}%constituentNames(9:)=>{a}%constituentNames
 held=>{a}
 call oracle_set_context([101],1,0)
 if(mode/='disabled')call oracle_set_output(trim(prefix))
 if(mode=='absent')nullify({rootvars})
 if(mode=='external')allocate({a}%block)
 if(mode/='load')call oracle_stream_capture('stream','pre','state.stream',{rootvars},1)
 if(mode=='capture'.or.mode=='disabled')stop
 if(mode=='load')nullify({rootvars})
 if(mode=='bad')then
  open(newunit=unit,file=trim(prefix)//'.pre.state.stream.{a}.target.bin',status='old')
  close(unit,status='delete')
  call oracle_write('stream','pre','state.stream.{a}.target',999,1)
 endif
 do iteration=1,2
  call oracle_stream_restore(trim(prefix),'state.stream',{rootvars})
  if(mode=='absent')then
   if(associated({a}))error stop 'lost absent root'
   stop
  endif
  if(mode/='load')then
   if(.not.associated({a},held))error stop 'lost compatible live node'
  endif
  if(.not.associated({roots[2]},{a}))error stop 'lost shared root'
  if(.not.associated({b},{a}%next))error stop 'lost child root alias'
  if(.not.associated({b}%next,{a}))error stop 'lost next cycle'
  if(.not.associated({b}%prev,{a}))error stop 'lost previous cycle'
  if(.not.associated({a}%array,{b}%array))error stop 'lost array alias'
  if(any(lbound({b}%array)/=[-9,7]))error stop 'lost rank2 view bounds'
  if(.not.associated({a}%constituentNames,{b}%constituentNames))error stop 'lost character alias'
  if(lbound({b}%constituentNames,1)/=9)error stop 'lost character view bounds'
  write(ordinal,'(i0)')iteration
  call oracle_set_output(trim(prefix)//'.restored'//trim(ordinal))
  call oracle_stream_capture('stream','post','state.stream',{rootvars},iteration)
  {a}%array=-1
  {a}%fieldName='mutated'
  {a}%isActive=.false.
  {b}=>{a}
  {a}%next=>{a}
 enddo
 contains
 subroutine initialize(p)
 type(field2DInteger),intent(inout)::p
 {initialize}
 end subroutine
end program
'''
    (tmp_path/'fixture.F90').write_text(fixture)
    includes=sorted({p.parent for p in native.rglob('*.mod')})
    flags='-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-form -ffree-line-length-none -fconvert=big-endian -fcheck=all'
    flags+=' '+' '.join(f'-I{p}'for p in [transport,*includes])
    (tmp_path/'Makefile').write_text(f'''FC=mpif90
FLAGS={flags}
all: fixture
stream.o: stream.F90
\t$(FC) $(FLAGS) -c $< -o $@
fixture: fixture.F90 stream.o
\t$(FC) $(FLAGS) $^ {transport}/transport.o -Wl,--start-group {native}/src/framework/libframework.a {native}/src/external/esmf_time_f90/libesmf_time.a -Wl,--end-group -lnetcdff -lnetcdf -o $@
''')
    r=subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,env=dict(os.environ,MPAS_UNIT='f1-native-tap'),capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    for mode in ('roundtrip','capture','load','absent'):
        prefix=tmp_path/('capture'if mode=='load'else mode)
        r=subprocess.run([str(tmp_path/'fixture'),str(prefix),mode],capture_output=True,text=True)
        assert r.returncode==0,r.stdout+r.stderr
    @lru_cache(maxsize=None)
    def read(op,filename,indices):
        return json.loads(subprocess.run([str(transport/'oracle_words'),op,str(tmp_path/filename),indices],check=True,capture_output=True,text=True).stdout)
    words=0;rows=[]
    for prefix in ('roundtrip','capture'):
        originals=list(tmp_path.glob(prefix+'.pre.*.bin'));assert len(originals)>40
        for original in originals:
            for iteration in (1,2):
                other=tmp_path/original.name.replace('.pre.',f'.restored{iteration}.post.',1)
                stats=json.loads(compare(transport,original,other).stdout);assert stats['bitwise_equal'],original.name
                words+=stats['words']
        for suffix,phase in [('', 'pre'),('.restored1','post'),('.restored2','post')]:
            context=prefix+suffix;files={p.name for p in tmp_path.glob(context+'.'+phase+'.*.bin')}
            audit=audit_membership(info,context,phase,'state.stream',files,read)
            assert audit['complete'],json.dumps(audit,indent=2)
            assert audit['typed_node_counts']['node']==2
            rows.append(dict(prefix=context,phase=phase,audit=audit))
            missing=context+'.'+phase+'.state.stream.node.node.1.fieldname.bin'
            bad=audit_membership(info,context,phase,'state.stream',files-{missing},read)
            assert not bad['complete']and missing in bad['missing']
            orphan=context+'.'+phase+'.state.stream.node.node.3.fieldname.bin'
            bad=audit_membership(info,context,phase,'state.stream',files|{orphan},read)
            assert not bad['complete']and orphan in bad['unexpected']
    for mode,message in [('bad','forward reference'),('external','external pointee')]:
        r=subprocess.run([str(tmp_path/'fixture'),str(tmp_path/mode),mode],capture_output=True,text=True)
        assert r.returncode!=0 and message in r.stderr
        files={p.name for p in tmp_path.glob(mode+'.pre.*.bin')}
        assert not audit_membership(info,mode,'pre','state.stream',files,read)['complete']
    disabled=tmp_path/'disabled';disabled.mkdir()
    subprocess.run([str(tmp_path/'fixture'),str(disabled/'capture'),'disabled'],env=dict(os.environ,MPAS_ORACLE_DIR=str(disabled),MPAS_ORACLE_STEPS='1',MPAS_ORACLE_KEYS='other'),check=True,capture_output=True)
    assert not list(disabled.glob('*.bin'))
    proof=dict(process_count=1,graphs=rows,words=words,different_words=0,
        native_source_sha256=info['source_sha256'],generated_source_sha256=__import__('hashlib').sha256(generated.encode()).hexdigest(),
        executable_sha256=__import__('hashlib').sha256((tmp_path/'fixture').read_bytes()).hexdigest(),
        external_aliases_complete=False,external_pointees_complete=False,uninitialized_native_metadata_witness_complete=False,
        populated_native_stream_forecast_capture_proven=False,all_eight_roots_observed=True)
    (tmp_path/'stream-membership-proof.json').write_text(json.dumps(proof,indent=2)+'\n')


def test_pool_restore_with_actual_native_framework(transport, tmp_path):
    """Exercise native MPAS pools rather than substituting framework types/APIs."""
    import shutil
    from generate_pool_restore import generate as restore_source
    native = os.environ.get('MPAS_ORACLE_NATIVE_BUILD')
    if not native:
        pytest.skip('set MPAS_ORACLE_NATIVE_BUILD to an owned strict MPAS build')
    native = Path(native)
    for path in (native/'src/framework').glob('*.mod'):
        shutil.copy2(path, tmp_path/path.name)
    shutil.copy2(native/'src/framework/libframework.a', tmp_path/'libframework.a')
    shutil.copy2(native/'src/external/esmf_time_f90/libesmf_time.a', tmp_path/'libesmf_time.a')
    (tmp_path/'mpas_oracle_dump.F').write_text(transport_source())
    (tmp_path/'mpas_oracle_pool.F').write_text(pool_source())
    (tmp_path/'mpas_oracle_pool_restore.F').write_text(restore_source())
    shutil.copy2(ORACLE/'replay/pool_fixture.F90', tmp_path/'pool_fixture.F90')
    (tmp_path/'Makefile').write_text('''FC=mpif90
FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-form -ffree-line-length-none -fconvert=big-endian
all: pool_fixture
mpas_oracle_dump.o: mpas_oracle_dump.F
\t$(FC) $(FLAGS) -c $< -o $@
mpas_oracle_pool.o: mpas_oracle_pool.F mpas_oracle_dump.o
\t$(FC) $(FLAGS) -c $< -o $@
mpas_oracle_pool_restore.o: mpas_oracle_pool_restore.F mpas_oracle_dump.o
\t$(FC) $(FLAGS) -c $< -o $@
pool_fixture: pool_fixture.F90 mpas_oracle_pool.o mpas_oracle_pool_restore.o
\t$(FC) $(FLAGS) $^ mpas_oracle_dump.o -Wl,--start-group libframework.a libesmf_time.a -Wl,--end-group -lnetcdff -lnetcdf -o $@
''')
    result = subprocess.run(_build_command(['make','-j4']), cwd=tmp_path,
                            env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','f1-native-tap')), capture_output=True,text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    subprocess.run([str(tmp_path/'pool_fixture'), str(tmp_path/'original'), str(tmp_path/'restored')],
                   check=True,capture_output=True,text=True)
    originals = {p.name.removeprefix('original.post.'):p for p in tmp_path.glob('original.post.*.bin')}
    restored = {p.name.removeprefix('restored.post.'):p for p in tmp_path.glob('restored.post.*.bin')}
    assert originals.keys() == restored.keys()
    assert len(originals) > 90
    for name, path in originals.items():
        assert json.loads(compare(transport,path,restored[name]).stdout)['bitwise_equal'], name



def test_actual_framework_ledger_and_pool_error_state(transport, tmp_path):
    """Whole pinned mpas_io_units.F:29 and mpas_pool_routines.F:108 modules."""
    from state import generate_state
    native_build = os.environ.get('MPAS_ORACLE_NATIVE_BUILD')
    if not native_build:
        pytest.skip('set MPAS_ORACLE_NATIVE_BUILD to an owned strict MPAS build')
    native_build = Path(native_build)
    pinned = site_support.mpas_source(preferred=('MPAS_ORACLE_NATIVE_SOURCE',))
    root = tmp_path/'source'
    framework = root/'src/framework'; framework.mkdir(parents=True)
    physics = root/'src/core_atmosphere/physics'; physics.mkdir(parents=True)
    # Inventory the whole actual logger too, including its source-derived
    # pointer targets. External aliases/file positions remain explicit gaps.
    for name in ('mpas_io_units.F','mpas_pool_routines.F','mpas_log.F','mpas_log_types.inc'):
        shutil.copy2(pinned/'src/framework'/name, framework/name)
        (framework/name).chmod(0o644)
    (physics/'mpas_atmphys_init.F').write_text("""module mpas_atmphys_vars
use mpas_io_units
use mpas_pool_routines
contains
subroutine physics_init(mesh)
integer :: mesh
mesh=mesh
end subroutine physics_init
end module mpas_atmphys_vars
""")
    manifest = {'sites': [], 'boundaries': []}
    generated = tmp_path/'generated'
    shutil.copytree(root, generated)
    generate_state(root, generated, manifest)
    infos = {m['name']: m for m in manifest['module_state']}
    assert infos['mpas_io_units']['variables'][0]['name'] == 'unitsinuse'
    assert infos['mpas_io_units']['variables'][0]['dims'] == ['0:maxUnits']
    assert infos['mpas_pool_routines']['variables'][0]['name'] == 'currenterrorlevel'
    assert {v['name'] for v in infos['mpas_log']['roots']} == {'mpas_log_info'}
    assert any(i['name'] == 'mpas_log_info' for i in infos['mpas_log']['issues'])
    assert not manifest['hidden_state_complete']
    for directory in ('src/framework','src/external/esmf_time_f90'):
        for path in (native_build/directory).glob('*.mod'):
            shutil.copy2(path, tmp_path/path.name)
    for directory, library in (('src/framework','libframework.a'),
                               ('src/external/esmf_time_f90','libesmf_time.a')):
        shutil.copy2(native_build/directory/library, tmp_path/library)
    (tmp_path/'ledger.F90').write_text("""program ledger_roundtrip
use mpas_io_units, only: mpas_new_unit,mpas_release_unit, &
 oracle_state_mpas_io_units,oracle_restore_mpas_io_units
use mpas_pool_routines, only: mpas_pool_set_error_level,mpas_pool_get_error_level, &
 oracle_state_mpas_pool_routines,oracle_restore_mpas_pool_routines
use mpas_oracle_dump, only: oracle_set_context,oracle_begin
implicit none
integer :: first,next_unit,restored_next,event,iteration,i
character(len=4096) :: directory,prefix
call get_command_argument(1,directory)
call oracle_set_context([1],1,0)
do i=0,200
 call mpas_release_unit(i)
enddo
call mpas_new_unit(first)
call mpas_pool_set_error_level(1)
call oracle_begin('framework',event)
call oracle_state_mpas_io_units('framework','pre',event)
call oracle_state_mpas_pool_routines('framework','pre',event)
prefix=trim(directory)//'/framework.s0000000001.e0000000001.r000000'
call mpas_new_unit(next_unit)
if(next_unit==first.or.next_unit<0) error stop 'native unit reservation failed'
do iteration=1,2
 call mpas_release_unit(first)
 call mpas_pool_set_error_level(2)
 call oracle_restore_mpas_io_units(trim(prefix))
 call oracle_restore_mpas_pool_routines(trim(prefix))
 if(mpas_pool_get_error_level()/=1) error stop 'native error level not restored'
 if(iteration>1) call oracle_begin('framework',event)
 call oracle_state_mpas_io_units('framework','post',event)
 call oracle_state_mpas_pool_routines('framework','post',event)
 call mpas_new_unit(restored_next)
 if(restored_next/=next_unit) error stop 'native unit ledger not restored'
enddo
end program ledger_roundtrip
""")
    source = generated/'src/framework'
    (tmp_path/'Makefile').write_text(f"""FC=mpif90
FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-form -ffree-line-length-none -fconvert=big-endian
all: ledger_roundtrip
mpas_io_units.o: {source}/mpas_io_units.F
\t$(FC) $(FLAGS) -c $< -o $@
mpas_pool_routines.o: {source}/mpas_pool_routines.F
\t$(FC) $(FLAGS) -c $< -o $@
ledger_roundtrip: ledger.F90 mpas_io_units.o mpas_pool_routines.o
\t$(FC) $(FLAGS) -I{transport} $^ {transport}/transport.o -Wl,--start-group libframework.a libesmf_time.a -Wl,--end-group -lnetcdff -lnetcdf -o $@
""")
    result = subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,
                            env=dict(os.environ,MPAS_UNIT='f1-native-tap'),capture_output=True,text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    dump=tmp_path/'dump';dump.mkdir()
    result = subprocess.run([str(tmp_path/'ledger_roundtrip'),str(dump)],
        env=dict(os.environ,MPAS_ORACLE_DIR=str(dump),MPAS_ORACLE_STEPS='1',
                 MPAS_ORACLE_KEYS='framework'),capture_output=True,text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert len(list(dump.glob('*.bin'))) == 6
    for name in ('state.mpas_io_units.unitsinuse','state.mpas_pool_routines.currenterrorlevel'):
        original = next(dump.glob('*.pre.'+name+'.bin'))
        restored = sorted(dump.glob('*.post.'+name+'.bin'))
        assert len(restored) == 2
        for record in restored:
            assert json.loads(compare(transport, original, record).stdout)['bitwise_equal']
    ledger = next(dump.glob('*.pre.state.mpas_io_units.unitsinuse.bin'))
    header = json.loads(subprocess.check_output([str(transport/'oracle_words'),'header',str(ledger),'-'],text=True))
    assert header['lower'] == [0] and header['shape'] == [201]


def test_actual_framework_allocation_bridge_preserves_build_order(transport,tmp_path):
    """Whole native mpas_string_utils.F:34-69, allocator PRE before assignment."""
    from state import generate_state
    from allocation_state import generate_allocations
    if not os.environ.get('MPAS_ORACLE_NATIVE_BUILD'):
        pytest.skip('set MPAS_ORACLE_NATIVE_BUILD to enable pinned native framework fixtures')
    pinned=site_support.mpas_source(preferred=('MPAS_ORACLE_NATIVE_SOURCE',))
    root=tmp_path/'source';framework=root/'src/framework';framework.mkdir(parents=True)
    physics=root/'src/core_atmosphere/physics';physics.mkdir(parents=True)
    shutil.copy2(pinned/'src/framework/mpas_string_utils.F',framework/'mpas_string_utils.F')
    (framework/'mpas_string_utils.F').chmod(0o644)
    (physics/'mpas_atmphys_init.F').write_text("""module mpas_atmphys_vars
use mpas_string_utils
contains
subroutine physics_init(mesh)
integer :: mesh
mesh=mesh
end subroutine physics_init
end module mpas_atmphys_vars
""")
    manifest={'sites':[],'boundaries':[]};generated=tmp_path/'generated'
    shutil.copytree(root,generated);generate_state(root,generated,manifest)
    generate_allocations(root,generated,manifest)
    native=generated/'src/framework/mpas_string_utils.F'
    assert 'use mpas_oracle_dump' not in native.read_text().lower()
    record=next(r for r in manifest['allocation_state'] if r['routine']=='mpas_split_string')
    assert not record['issues'] and len(record['variables'])==1
    (tmp_path/'strings.F90').write_text("""program strings_roundtrip
use mpas_string_utils,only:mpas_split_string
use mpas_oracle_dump,only:oracle_set_context,oracle_begin,oracle_scope_begin, &
 oracle_scope_end,oracle_replay_begin,oracle_set_output,oracle_write
implicit none
character(len=12),pointer :: parts(:)
character(len=4096) :: prefix,output
integer :: event
nullify(parts)
call oracle_set_context([1],1,0)
call get_command_argument(1,prefix)
call get_command_argument(2,output)
if(len_trim(prefix)>0)then
 event=1
 call oracle_replay_begin(trim(prefix),'framework_strings',event)
 call oracle_set_output(trim(output))
else
 call oracle_begin('framework_strings',event)
 call oracle_scope_begin('framework_strings',event)
endif
call mpas_split_string('wind,cloud,ice',',',parts)
if(size(parts)/=3)error stop 'native string count changed'
if(any(parts/=['wind        ','cloud       ','ice         '])) error stop 'native string behavior changed'
call oracle_write('framework_strings','post','parts',parts,event,lower=lbound(parts))
call oracle_scope_end()
deallocate(parts)
end program strings_roundtrip
""")
    (tmp_path/'Makefile').write_text(f"""FC=gfortran
FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -ffree-form -fconvert=big-endian
all: strings_roundtrip
mpas_string_utils.o: {native}
\t$(FC) $(FLAGS) -c $< -o $@
strings_roundtrip: strings.F90 mpas_string_utils.o
\t$(FC) $(FLAGS) -I{transport} $^ {transport}/transport.o -o $@
""")
    result=subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,
                         env=dict(os.environ,MPAS_UNIT='f1-native-tap'),capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    dump=tmp_path/'dump';dump.mkdir()
    env=dict(os.environ,MPAS_ORACLE_DIR=str(dump),MPAS_ORACLE_STEPS='1',MPAS_ORACLE_KEYS='framework_strings')
    subprocess.run([str(tmp_path/'strings_roundtrip')],env=env,check=True,capture_output=True)
    original=next(dump.glob('*.post.parts.bin'))
    prefix=original.name.split('.post.')[0]
    pre=next(dump.glob('*.pre.allocation.*.bin'))
    subprocess.run([str(tmp_path/'strings_roundtrip'),str(dump/prefix),str(tmp_path/'replayed')],check=True,capture_output=True)
    assert json.loads(compare(transport,original,tmp_path/'replayed.post.parts.bin').stdout)['bitwise_equal']
    pre.unlink()
    result=subprocess.run([str(tmp_path/'strings_roundtrip'),str(dump/prefix),str(tmp_path/'missing')],capture_output=True,text=True)
    assert result.returncode!=0 and 'oracle missing input record' in result.stderr

def test_allocator_entry_native_replay(transport,tmp_path):
    from allocation_state import patch_allocations
    from state import modules
    root=tmp_path/'source';physics=root/'src/core_atmosphere/physics';physics.mkdir(parents=True)
    source=physics/'allocator.F90'
    source.write_text('''module allocator_native
real,allocatable :: work(:)
contains
subroutine calculate(observed)
real,intent(out) :: observed(4,2)
integer :: iteration
 do iteration=1,2
 if(.not.allocated(work)) allocate(work(-1:2))
 work(-1)=42.0+iteration
 work(0)=17.0
 observed(:,iteration)=work
 deallocate(work)
 enddo
end subroutine calculate
end module allocator_native
''')
    text,records=patch_allocations(source,source.read_text(),modules(root),{'allocator_native'})
    assert len(records)==1 and not records[0]['issues']
    assert records[0]['line']==8
    assert "if(.not.allocated(work)) then" in text
    (tmp_path/'native.F90').write_text(text)
    (tmp_path/'door.F90').write_text('''program door
use iso_fortran_env,only:int32
use allocator_native
use mpas_oracle_dump
implicit none
real :: observed(4,2)
character(len=4096) :: prefix,output
integer :: event
call get_command_argument(1,prefix)
call get_command_argument(2,output)
call oracle_set_context([17_int32],1,0)
if(len_trim(output)>0) then
 call oracle_set_output(output)
 call oracle_replay_begin(prefix,'allocator',1)
else
 call oracle_begin('allocator',event)
 call oracle_scope_begin('allocator',event)
endif
call calculate(observed)
call oracle_write('allocator','post','observed',observed,1)
call oracle_scope_end()
end program door
''')
    (tmp_path/'Makefile').write_text(f'''all: door
door: native.F90 door.F90
\tgfortran -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -I{transport} $^ {transport}/transport.o -o $@
''')
    subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','f1-native-tap')),check=True,capture_output=True)
    dump=tmp_path/'dump';dump.mkdir()
    env=dict(os.environ,MPAS_ORACLE_DIR=str(dump),MPAS_ORACLE_STEPS='1',MPAS_ORACLE_KEYS='allocator')
    subprocess.run([str(tmp_path/'door')],env=env,check=True,capture_output=True)
    entries=sorted(dump.glob('*.pre.allocation.*.bin'))
    assert len(entries)==2 and entries[0].name.endswith('.n0000000001.bin') and entries[1].name.endswith('.n0000000002.bin')
    reference=next(dump.glob('*.post.observed.bin'));prefix=reference.name.split('.post.')[0]
    result=tmp_path/'replayed'
    subprocess.run([str(tmp_path/'door'),str(dump/prefix),str(result)],check=True,capture_output=True)
    assert json.loads(compare(transport,reference,tmp_path/'replayed.post.observed.bin').stdout)['bitwise_equal']


def test_automatic_entry_storage_replays_unwritten_words_and_call_shapes(transport,tmp_path):
    from allocation_state import patch_entry_storage
    source=tmp_path/'automatic.F90'
    source.write_text('''module automatic_native
use iso_fortran_env,only:real64
contains
subroutine calculate(n,observed)
integer,intent(in) :: n
real(real64),intent(out) :: observed(2,0:n)
real(real64) :: liquid(0:n),ice(0:n)
liquid(0)=42.0_real64
ice(0)=17.0_real64
observed(1,:)=liquid
observed(2,:)=ice
end subroutine
end module
''')
    request=dict(module='automatic_native',routine='calculate',variables=['liquid','ice'])
    text,records=patch_entry_storage(source,source.read_text(),[request])
    assert len(records)==2 and all(not row['issues'] for row in records)
    (tmp_path/'native.F90').write_text(text)
    (tmp_path/'door.F90').write_text('''program door
use iso_fortran_env,only:real64
use automatic_native
use mpas_oracle_dump
implicit none
real(real64) :: first(2,0:7),second(2,0:3)
character(len=4096) :: prefix,output
integer :: event
call get_command_argument(1,prefix)
call get_command_argument(2,output)
call oracle_set_context([17],1,0)
if(len_trim(output)>0) then
 call oracle_set_output(output)
 call oracle_replay_begin(prefix,'automatic',1)
else
 call oracle_begin('automatic',event)
 call oracle_scope_begin('automatic',event)
endif
call calculate(7,first)
call calculate(3,second)
call oracle_write('automatic','post','first',first,1,lower=[1,0])
call oracle_write('automatic','post','second',second,1,lower=[1,0])
call oracle_scope_end()
end program
''')
    (tmp_path/'Makefile').write_text(f'''all: door
door: native.F90 door.F90
\tgfortran -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -I{transport} $^ {transport}/transport.o -o $@
''')
    subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','f1-native-tap')),check=True,capture_output=True)
    dump=tmp_path/'dump';dump.mkdir()
    subprocess.run([str(tmp_path/'door')],env=dict(os.environ,MPAS_ORACLE_DIR=str(dump),
        MPAS_ORACLE_STEPS='1',MPAS_ORACLE_KEYS='automatic'),check=True,capture_output=True)
    entries=sorted(dump.glob('*.pre.allocation.*.bin'))
    assert len(entries)==4
    for record in entries:
        header=json.loads(subprocess.check_output([str(transport/'oracle_words'),'header',str(record),'-'],text=True))
        assert header['lower']==[0]
    prefix=next(dump.glob('*.post.first.bin')).name.split('.post.')[0]
    subprocess.run([str(tmp_path/'door'),str(dump/prefix),str(tmp_path/'replayed')],check=True,capture_output=True)
    for name in ('first','second'):
        original=dump/(prefix+'.post.'+name+'.bin')
        assert json.loads(compare(transport,original,tmp_path/('replayed.post.'+name+'.bin')).stdout)['bitwise_equal']
    # A missing native entry input must fail; POST outputs cannot seed it.
    entries[0].unlink()
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([str(tmp_path/'door'),str(dump/prefix),str(tmp_path/'incomplete')],check=True,capture_output=True)
    with pytest.raises(ValueError,match='declaration changed'):
        patch_entry_storage(source,source.read_text(),[dict(request,variables=['absent'])])


def test_driver_boundary_module_state(transport,tmp_path):
    from state import generate_state
    from allocation_state import generate_allocations
    from local_state import generate_locals
    import shutil
    root=tmp_path/'source';physics=root/'src/core_atmosphere/physics';physics.mkdir(parents=True)
    dynamics=root/'src/core_atmosphere/dynamics';dynamics.mkdir()
    native=dynamics/'mpas_atm_iau.F'
    native.write_text('''module boundary_native
use iso_fortran_env,only:real64
complex(real64) :: dielectric=(0.0_real64,0.0_real64)
contains
subroutine driver_payload(x)
real,intent(inout) :: x
real,allocatable :: delta(:)
integer,save :: invocations=7
invocations=invocations+1
allocate(delta(1))
delta=real(dielectric,kind(x))
x=x+delta(1)
deallocate(delta)
dielectric=dielectric+cmplx(2.0_real64,-1.0_real64,kind=real64)
end subroutine
end module boundary_native
''')
    caller=physics/'caller.F90';caller.write_text('''module boundary_caller
use boundary_native,only:driver_payload
contains
subroutine driver_outer(x)
use mpas_oracle_dump,only:oracle_begin
real,intent(inout) :: x
integer :: oracle_boundary_event
call oracle_begin('boundary__driver_payload',oracle_boundary_event)
call driver_payload(x)
end subroutine
end module boundary_caller
''')
    (physics/'mpas_atmphys_init.F').write_text('''module mpas_atmphys_vars
integer :: initialized_count=0
contains
subroutine physics_init(mesh)
integer :: mesh
mesh=mesh
end subroutine physics_init
end module mpas_atmphys_vars
''')
    st=next(st for st in statements(caller.read_text()) if st.text=='call driver_payload(x)')
    manifest={'sites':[],'boundaries':[{'key':'boundary__driver_payload',
              'path':str(caller.relative_to(root)),'line':st.start+1,'end_line':st.end,'driver':'driver_outer'}]}
    out=tmp_path/'generated';shutil.copytree(root,out);generate_state(root,out,manifest)
    generate_locals(root,out,manifest)
    generate_allocations(root,out,manifest)
    b=manifest['boundaries'][0]
    assert b['state_modules']==['boundary_native']
    assert not b['module_state_complete'] and not b['local_state_complete']
    assert 'iso_fortran_env' in b['uninventoried_state_modules']
    assert 'boundary_native' not in manifest['init_state_modules']
    assert not any(i.get('reason')=='local SAVE state requires capture' for i in b['state_issues'])
    local=next(r for r in b['local_state_inventory'] if r['routine']=='driver_payload')
    assert local['variables'][0]['name']=='invocations' and not local['issues']
    init=(out/'src/core_atmosphere/physics/mpas_atmphys_init.F').read_text()
    assert 'subroutine physics_init(mesh)\n      use mpas_oracle_dump' in init
    assert 'use mpas_atmphys_vars, only: oracle_state_mpas_atmphys_vars' not in init
    allocation=next(r for r in manifest['allocation_state'] if r['routine']=='driver_payload')
    assert allocation['variables'][0]['name']=='delta' and not allocation['issues']
    generated=out/'src/core_atmosphere/physics'
    (tmp_path/'capture.F90').write_text('''program capture
use iso_fortran_env,only:real64,int32
use boundary_native
use boundary_caller
use mpas_oracle_dump
implicit none
real :: x
integer :: event,expected_invocations
call oracle_set_context([101_int32],1,0)
dielectric=cmplx(5.0_real64,-3.0_real64,kind=real64);x=1.0
call oracle_begin('expected',event)
call oracle_write('expected','pre','dielectric',dielectric,event)
expected_invocations=7
call oracle_write('expected','pre','invocations',expected_invocations,event)
call driver_outer(x)
if(x/=6.0) error stop 'boundary tap changed native arithmetic'
call oracle_write('expected','post','dielectric',dielectric,event)
expected_invocations=8
call oracle_write('expected','post','invocations',expected_invocations,event)
end program
''')
    (tmp_path/'Makefile').write_text(f'''FC=gfortran
FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-form -ffree-line-length-none -I{transport}
all: capture
native.o: {out}/src/core_atmosphere/dynamics/mpas_atm_iau.F
\t$(FC) $(FLAGS) -c $< -o $@
caller.o: {generated}/caller.F90 native.o
\t$(FC) $(FLAGS) -c $< -o $@
capture: capture.F90 caller.o native.o
\t$(FC) $(FLAGS) $^ {transport}/transport.o -o $@
''')
    result=subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','f1-native-tap')),capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    dump=tmp_path/'dump';dump.mkdir()
    env=dict(os.environ,MPAS_ORACLE_DIR=str(dump),MPAS_ORACLE_KEYS='boundary__driver_payload,expected',MPAS_ORACLE_STEPS='1')
    subprocess.run([str(tmp_path/'capture')],env=env,check=True,capture_output=True)
    for phase in ('pre','post'):
        expected=next(dump.glob('expected*.'+phase+'.dielectric.bin'))
        captured=next(dump.glob('boundary__driver_payload*.'+phase+'.state.boundary_native.dielectric.bin'))
        assert json.loads(compare(transport,expected,captured).stdout)['bitwise_equal']
        expected=next(dump.glob('expected*.'+phase+'.invocations.bin'))
        captured=next(dump.glob('boundary__driver_payload*.'+phase+'.local.boundary_native.driver_payload.invocations.bin'))
        assert json.loads(compare(transport,expected,captured).stdout)['bitwise_equal']


def test_extracted_wrf_column_replays_with_native_loop_bounds(transport,tmp_path):
    from generate import inventory,overlay
    import shutil
    root=tmp_path/'source';physics=root/'src/core_atmosphere/physics';physics.mkdir(parents=True)
    (physics/'column.F90').write_text('''module column_native
contains
subroutine advance(q,ims,ime,its,ite)
integer,intent(in) :: ims,ime,its,ite
real,intent(inout) :: q(ims:ime,1:4)
integer :: i,k
do k=1,4
 do i=its,ite
  q(i,k)=q(i,k)*2.0+real(k)
 enddo
enddo
end subroutine
subroutine driver_column(q)
real,intent(inout) :: q(1:3,1:4)
call advance(q,1,3,1,3)
end subroutine
end module
''')
    m=inventory(root,authenticate=False);site=m['sites'][0];generated=tmp_path/'generated';overlay(root,generated,m)
    shutil.copy2(generated/site['path'],tmp_path/'column.F90')
    (tmp_path/'replay.F90').write_text(replay_source(site))
    (tmp_path/'capture.F90').write_text('''program capture
use column_native
use mpas_oracle_dump
implicit none
real :: q(3,4)
integer :: i,k
call oracle_set_context([17,23,31],1,0)
do k=1,4
 do i=1,3
  q(i,k)=real(i*k)
 enddo
enddo
call driver_column(q)
end program
''')
    (tmp_path/'Makefile').write_text(f'''FC=gfortran
FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -I{transport}
all: capture replay
column.o: column.F90
\t$(FC) $(FLAGS) -c $< -o $@
capture: column.o capture.F90
\t$(FC) $(FLAGS) $^ {transport}/transport.o -o $@
replay: column.o replay.F90
\t$(FC) $(FLAGS) $^ {transport}/transport.o -o $@
''')
    subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','f1-native-tap')),check=True,capture_output=True)
    dump=tmp_path/'capture-dump';dump.mkdir()
    subprocess.run([str(tmp_path/'capture')],env=dict(os.environ,MPAS_ORACLE_DIR=str(dump)),check=True,capture_output=True)
    prefix=next(dump.glob('*.pre.q.bin')).name.removesuffix('.pre.q.bin')
    cropped=tmp_path/'column-input';cropped.mkdir()
    for record in dump.glob('*.bin'):
        subprocess.run([str(transport/'oracle_words'),'column',str(record),'23',str(cropped/record.name)],check=True,capture_output=True)
    output=tmp_path/'column-output';output.mkdir()
    subprocess.run([str(tmp_path/'replay'),str(cropped/prefix),str(output/prefix),'column'],check=True,capture_output=True)
    assert json.loads(compare(transport,cropped/(prefix+'.post.q.bin'),output/(prefix+'.post.q.bin')).stdout)['bitwise_equal']
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([str(tmp_path/'replay'),str(dump/prefix),str(output/'invalid-full-record'),'column'],check=True,capture_output=True)
    # Exercise the public door, including input/expected-output separation and
    # authentication of the unchanged whole-mesh source capture.
    import argparse
    spec=importlib.util.spec_from_file_location('column_oracle_replay',ORACLE.parents[1]/'column_oracle_replay.py')
    harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
    inventory_path=tmp_path/'inventory.json';inventory_path.write_text(json.dumps(m))
    pins=tmp_path/'PINS.txt';pins.write_text('source-bound native column regression\n')
    sealed=harness.seal(dump,inventory_path,pins,transport/'oracle_words','gnu-strict',1)
    options=argparse.Namespace(dump=dump,scheme=site['site'],backend='native-replay',recipe='gnu-strict',
        executable=tmp_path/'replay',results=tmp_path/'door-column',receipt=tmp_path/'door-column.json',
        words=transport/'oracle_words',coverage=None,gpu_slot_minutes=None,process_count=1,
        cell_id=23,capture=prefix,whole_capture=False)
    assert harness.replay(options)==0
    receipt=json.loads(options.receipt.read_text())
    assert receipt['execution_mode']=='column' and receipt['column_cell_id']==23
    assert receipt['dump_root_sha256']==sealed['dump_root_sha256']
    assert receipt['all_outputs_bitwise_equal'] and not receipt['forecast_proven']
    row=receipt['captures'][0]
    assert row['arrays']['q']['words']==4 and row['arrays']['q']['different_words']==0
    assert all('.pre.' in p.name for p in Path(row['column']['inputs']).iterdir())
    assert harness.verify(dump)==sealed
    options.results=tmp_path/'missing-cell';options.cell_id=999
    with pytest.raises(subprocess.CalledProcessError):harness.replay(options,sealed)
    options.results=tmp_path/'missing-capture';options.capture='uncaptured'
    with pytest.raises(ValueError,match='no captures'):harness.replay(options,sealed)


def test_native_input_read_projection_trace_authority(tmp_path):
    """An input projection may remove only witnessed, unused allocation PREs."""
    path=ORACLE/'replay/project_input_reads.py'
    spec=importlib.util.spec_from_file_location('project_input_reads',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=tmp_path/'inputs';root.mkdir();trace=tmp_path/'native.trace'
    member='scheme.s0000000001.e0000000001.r000000.pre.q.bin'
    good=f'42 openat(AT_FDCWD, "{root/member}", O_RDONLY|O_CLOEXEC) = 3\n'
    trace.write_text(good + '42 openat(AT_FDCWD, "/usr/lib/library.so", O_RDONLY) = 4\n')
    assert module.read_set(trace,root,{member})=={member}
    failures=[
        (good.replace(' = 3',' = -1 ENOENT'),'native input open failed'),
        (good.replace('O_RDONLY','O_RDWR'),'opened for writing'),
        (good.replace(member,'unsealed.pre.q.bin'),'unsealed input'),
        (good.replace(' = 3',' = ?'),'unparsed native input open'),
        ('42 openat(AT_FDCWD, "/usr/lib/library.so", O_RDONLY) = 4\n','no successful native input reads'),
    ]
    for text,reason in failures:
        trace.write_text(text)
        with pytest.raises(ValueError,match=reason):module.read_set(trace,root,{member})


def test_local_source_audit_detects_omitted_hook_and_incomplete_pairs(tmp_path):
    path=ORACLE/'replay/audit_state.py'
    spec=importlib.util.spec_from_file_location('local_source_audit',path)
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    native=tmp_path/'native.F90'
    native.write_text('''module retained
contains
subroutine advance
integer, save :: counter=0
real :: table(2)
save table
counter=counter+1
end subroutine
end module
''')
    hooks=audit.local_inventory(tmp_path,{'retained'},
        {'retained':{'path':'native.F90'}},[])
    assert hooks[0]['differences']==['source or recorded local hook is missing']
    record=hooks[0]['source']
    matched=audit.local_inventory(tmp_path,{'retained'},
        {'retained':{'path':'native.F90'}},[record])
    assert not matched[0]['differences']
    observations=[dict(routine='advance',name='counter',reason='local SAVE state requires capture'),
                  dict(routine='advance',reason='COMMON/DATA local state requires capture')]
    unsupported=dict(name='external_target',reason='pointer target alias identity requires restoration')
    unresolved, represented=audit.reconcile_local_issues('retained',observations+[unsupported],matched)
    assert unresolved==[unsupported] and represented==observations
    assert audit.reconcile_local_issues('retained',observations,hooks)==(observations,[])
    changed=[dict(matched[0],differences=['native DATA declarations changed'])]
    assert audit.reconcile_local_issues('retained',observations,changed)==(observations,[])
    descriptor=[dict(matched[0],source=dict(record,issues=['hidden storage requires descriptor restoration']))]
    assert audit.reconcile_local_issues('retained',observations,descriptor)==(observations,[])
    unknown=dict(routine='advance',name='uninventoried',reason='local SAVE state requires capture')
    assert audit.reconcile_local_issues('retained',[unknown],matched)==([unknown],[])
    prefix='advance.s0000000001.e0000000001.r000000'
    unknown=audit.local_membership(matched,prefix,set())
    assert unknown['observed_pairs_complete']
    assert not unknown['execution_coverage_complete']
    names={f'{prefix}.{phase}.local.retained.advance.{field}.bin'
           for phase in ['pre','post']for field in ['counter','table']}
    assert audit.local_membership(matched,prefix,names)['execution_coverage_complete']
    primed=audit.local_membership(matched,prefix,names,
        [dict(module='retained',routine='advance')])
    assert primed['observed_pairs_complete']
    assert not primed['execution_coverage_complete']
    assert primed['routines'][0]['static_accessor_possible']
    names.remove(prefix+'.pre.local.retained.advance.table.bin')
    broken=audit.local_membership(matched,prefix,names)
    assert not broken['observed_pairs_complete']
    assert broken['routines'][0]['missing']==[prefix+'.pre.local.retained.advance.table.bin']
    names.add(prefix+'.post.local.retained.advance.uninventoried.bin')
    assert audit.local_membership(matched,prefix,names)['unexpected']==[
        prefix+'.post.local.retained.advance.uninventoried.bin']


def test_automatic_entry_audit_rejects_missing_pre_and_source_storage_changes(tmp_path):
    path=ORACLE/'replay/audit_state.py'
    spec=importlib.util.spec_from_file_location('automatic_source_audit',path)
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    native=tmp_path/'native.F90'
    native.write_text('''module retained
contains
subroutine advance(n,out)
integer,intent(in)::n
real,intent(out)::out(n)
real :: storage(n)
out=storage
end subroutine
end module
''')
    requests=[dict(module='retained',routine='advance',variables=['storage'],path='native.F90')]
    missing=audit.entry_inventory(tmp_path,{'retained'},[],requests)
    assert missing[0]['differences']==['source or recorded automatic entry input is missing']
    record=missing[0]['source']
    hooks=audit.entry_inventory(tmp_path,{'retained'},[record],requests)
    assert not hooks[0]['differences']
    altered=dict(record,dims=['n+1'])
    changed=audit.entry_inventory(tmp_path,{'retained'},[altered],requests)
    assert changed[0]['differences'][0]['field']=='dims'
    prefix='advance.s0000000001.e0000000001.r000000'
    tag=record['record'];callee=dict(module='retained',name='advance')
    post={f'{prefix}.post.allocation.{tag}.n0000000001.bin'}
    assert not audit.entry_membership(hooks,prefix,post,callee)['observed_inputs_complete']
    pre={f'{prefix}.pre.allocation.{tag}.n0000000001.bin'}
    assert audit.entry_membership(hooks,prefix,pre,callee)['observed_inputs_complete']
    pre.add(f'{prefix}.pre.allocation.{tag}.n0000000003.bin')
    assert not audit.entry_membership(hooks,prefix,pre,callee)['observed_inputs_complete']
    unknown=audit.entry_membership(hooks,prefix,set(),dict(module='retained',name='other'))
    assert unknown['observed_inputs_complete'] and not unknown['execution_coverage_complete']
    native.write_text(native.read_text().replace('real :: storage(n)','real, save :: storage(n)'))
    with pytest.raises(ValueError,match='declaration changed'):
        audit.entry_inventory(tmp_path,{'retained'},[record],requests)


def test_callback_interface_arguments_are_not_module_state(tmp_path):
    physics=tmp_path/'src/core_atmosphere/physics';physics.mkdir(parents=True)
    (physics/'callback.F90').write_text('''module callback
implicit none
integer :: live_state=3
abstract interface
 subroutine halo_exchange(domain,halo_group,ierr)
 use callback_types,only: domain_type
 type(domain_type),intent(inout) :: domain
 character(len=*),intent(in) :: halo_group
 integer,optional,intent(out) :: ierr
 end subroutine
end interface
contains
subroutine run()
 live_state=live_state+1
end subroutine
end module
''')
    info=modules(tmp_path)['callback']
    assert [v['name'] for v in info['variables']]==['live_state']
    assert info['uses']==[]


def test_native_mpi_f08_handle_transport(tmp_path):
    import shlex
    # A wrapper can be installed without its Fortran-2008 modules (the G0
    # OpenMPI wrapper names a missing gfortran-mod-16 directory). This is an
    # external toolchain dependency, not a failure of the generated transport.
    # A supplied compiler is authoritative: never skip its installation errors.
    configured = os.environ.get('MPAS_ORACLE_MPIFC')
    compiler = shlex.split(configured or 'mpif90')
    if not compiler or shutil.which(compiler[0]) is None:
        if configured is not None:
            pytest.fail('MPAS_ORACLE_MPIFC must name an installed MPI Fortran compiler')
        pytest.skip('needs an MPI Fortran compiler; install GNU MPICH/OpenMPI with mpi_f08.mod or set MPAS_ORACLE_MPIFC')
    shown = subprocess.run([*compiler, '-show'], capture_output=True, text=True)
    assert shown.returncode == 0, shown.stdout + shown.stderr
    flags = iter(shlex.split(shown.stdout))
    includes = [tmp_path]
    for flag in flags:
        if flag == '-I':
            includes.append(Path(next(flags)))
        elif flag.startswith('-I'):
            includes.append(Path(flag[2:]))
    if not any((directory / 'mpi_f08.mod').is_file() for directory in includes):
        reason = ('needs MPI Fortran-2008 module mpi_f08.mod in the wrapper include directories; '
                  'install GNU MPICH/OpenMPI development modules or set MPAS_ORACLE_MPIFC; '
                  f'wrapper reports: {shown.stdout.strip()}')
        if configured is not None:
            pytest.fail(reason)
        pytest.skip(reason)
    (tmp_path/'mpas_oracle_dump.F').write_text(transport_source())
    (tmp_path/'handles.F90').write_text('''program handles
use mpi_f08, only: MPI_Comm,MPI_Info,MPI_COMM_WORLD,MPI_INFO_NULL
use mpas_oracle_dump
implicit none
type(MPI_Comm) :: comm
type(MPI_Info) :: info
call oracle_set_output('handles')
call oracle_write('mpi','pre','comm',MPI_COMM_WORLD,1,cell_axis=0,level_axis=0)
call oracle_write('mpi','pre','info',MPI_INFO_NULL,1,cell_axis=0,level_axis=0)
call oracle_read('handles.pre.comm.bin',comm)
call oracle_read('handles.pre.info.bin',info)
if(comm%MPI_VAL/=MPI_COMM_WORLD%MPI_VAL) error stop 'communicator changed'
if(info%MPI_VAL/=MPI_INFO_NULL%MPI_VAL) error stop 'info changed'
end program
''')
    (tmp_path/'Makefile').write_text('''all: handles
handles: mpas_oracle_dump.F handles.F90
\t$(MPIFC) -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-form -ffree-line-length-none -DMPAS_USE_MPI_F08 $^ -o $@
''')
    subprocess.run(_build_command(['make','-j4',f'MPIFC={shlex.join(compiler)}']),cwd=tmp_path,
                   env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','r1-rrfs-native-tap')),check=True,capture_output=True)
    subprocess.run([str(tmp_path/'handles')],cwd=tmp_path,check=True,capture_output=True)


def test_object_audit_resolves_native_out_of_directory_sources(tmp_path):
    from audit_build import audit
    source=tmp_path/'physics/TEMPO/src';source.mkdir(parents=True)
    package=source.parent
    tools=tmp_path/'utils';tools.mkdir()
    copies=tmp_path/'libphys';copies.mkdir()
    (source/'scheme.F90').write_text('end\n')
    (source/'tables.F90').write_text('end\n')
    (package/'scheme.o').write_bytes(b'scheme object')
    (copies/'scheme.o').write_bytes(b'scheme object')
    (tools/'tempo_tables.o').write_bytes(b'tables object')
    flags='mpif90 -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations'
    log=tmp_path/'build.log'
    log.write_text(flags+' -c src/scheme.F90\n'+flags+' -c ../physics/TEMPO/src/tables.F90 -o tempo_tables.o\n')
    result=audit(tmp_path,log)
    assert result['all_ok'] and len(result['objects'])==3
    table=next(x for x in result['objects'] if x['object']=='utils/tempo_tables.o')
    assert table['compiled_object']=='utils/tempo_tables.o'
    duplicate=tmp_path/'other/src';duplicate.mkdir(parents=True)
    (duplicate/'scheme.F90').write_text('end\n')
    (duplicate.parent/'scheme.o').write_bytes(b'different object')
    assert not audit(tmp_path,log)['all_ok']


def test_physics_init_inside_instrumented_state_module(transport,tmp_path):
    from state import generate_state
    root=tmp_path/'source';physics=root/'src/core_atmosphere/physics'
    physics.mkdir(parents=True)
    (physics/'mpas_atmphys_init.F').write_text('''module mpas_atmphys_vars
real :: table(2)=[3.0,5.0]
contains
subroutine physics_init(mesh)
integer,intent(in) :: mesh
integer,parameter :: native_parameter(2)=[1,2]
call unrelated()
end subroutine physics_init
subroutine unrelated()
end subroutine
end module mpas_atmphys_vars
''')
    out=tmp_path/'generated';generate_state(root,out,{'sites':[]})
    (tmp_path/'pool.F90').write_text('''module mpas_oracle_pool
contains
subroutine oracle_mesh_context(mesh,step)
integer :: mesh,step
end subroutine
end module
''')
    (tmp_path/'main.F90').write_text('''program check
use mpas_atmphys_vars
call physics_init(101)
if(any(table/=[3.0,5.0])) error stop 'native table changed'
end program
''')
    source=out/'src/core_atmosphere/physics/mpas_atmphys_init.F'
    (tmp_path/'Makefile').write_text(f'''all: check
check: pool.F90 {source} main.F90
\tgfortran -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-form -ffree-line-length-none -I{transport} $^ {transport}/transport.o -o $@
''')
    subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,
                   env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','r1-rrfs-native-tap')),check=True,capture_output=True)
    subprocess.run([str(tmp_path/'check')],cwd=tmp_path,check=True,capture_output=True)


def test_native_coverage_maps_returned_sources_from_absent_build_host(tmp_path):
    from replay.coverage import summarize
    source=tmp_path/'pristine';build=tmp_path/'returned'
    for root in (source,build):(root/'src').mkdir(parents=True)
    (source/'src/scheme.F').write_text('subroutine scheme\nx=x+1\nend subroutine\n')
    (build/'src/scheme.F').write_text('subroutine scheme\ncall oracle_hook()\nx=x+1\nend subroutine\n')
    compiled=tmp_path/'absent-host-build'
    data={'gcc_version':'fixture','current_working_directory':str(compiled),
          'files':[{'file':str(compiled/'src/scheme.F'),'lines':[
              {'line_number':2,'count':1,'branches':[{'count':1},{'count':0}]},
              {'line_number':3,'count':1,'branches':[{'count':2},{'count':0}]}]}]}
    report=summarize(data,source,build,compile_root=compiled)
    row=report['files'][0]
    assert row['branches_total']==2 and row['branches_taken']==1
    assert row['uncovered_branches'][0]['line']==2
    assert row['instrumentation_lines_excluded']==1
    data['files'][0]['file']='../foreign.F'
    with pytest.raises(ValueError,match='escapes compile root'):
        summarize(data,source,build,compile_root=compiled)


def test_automatic_temporary_entry_replays_unset_native_words(transport,tmp_path):
    from allocation_state import generate_automatic_entries
    root=tmp_path/'source';physics=root/'src/core_atmosphere/physics';physics.mkdir(parents=True)
    native=physics/'native.F90'
    native.write_text('''module native
abstract interface
 subroutine exchange_halo(ierr)
 integer,optional,intent(out) :: ierr
 end subroutine
end interface
contains
subroutine scheme(n,out,flag,optional_flag)
integer,intent(in) :: n
real,intent(out) :: out(n)
integer,intent(out) :: flag
integer,optional,intent(out) :: optional_flag
real :: temporary(n*1)
temporary(1:n-1)=3.0
out=temporary
! Like RUC snowtemp's thin-snow path, this branch leaves OUT flags unset.
call leave_unset(flag)
out(1)=real(flag)
if(present(optional_flag)) out(2)=real(optional_flag)
call lbound()
end subroutine
subroutine leave_unset(flag)
integer,intent(out) :: flag
end subroutine
subroutine vector_copy(n,x,y)
integer,intent(in) :: n
real,intent(in) :: x(*)
real,intent(out) :: y(*)
y(1:n)=x(1:n)
end subroutine
subroutine lbound()
end subroutine
end module
''')
    out=tmp_path/'generated';target=out/native.relative_to(root);target.parent.mkdir(parents=True)
    target.write_text(native.read_text())
    m={'pin_set':'mpas-v842','module_state':[{'name':'native','path':str(native.relative_to(root))}]}
    generate_automatic_entries(root,out,m)
    assert [v['name'] for r in m['automatic_entry_state'] for v in r['variables']]==['out','flag','optional_flag','temporary','flag']
    unresolved=next(r for r in m['automatic_entry_state'] if r['routine']=='vector_copy')
    assert unresolved['issues']==['y: assumed-size entry extent requires caller descriptor']
    assert not unresolved['variables']
    assert not m['hidden_state_complete']
    (tmp_path/'capture.F90').write_text('''program capture
use iso_fortran_env,only:int32
use native
use mpas_oracle_dump
implicit none
real :: out(55)
integer :: event,i,flag,optional_flag
character(len=4096) :: prefix,result
call get_command_argument(1,prefix)
call get_command_argument(2,result)
call oracle_set_context([101_int32],1,0)
if(len_trim(result)>0) then
 call oracle_replay_begin(trim(prefix),'entry',1)
 call oracle_set_output(trim(result))
else
 call oracle_begin('entry',event)
 call oracle_scope_begin('entry',event)
 call oracle_set_output(trim(prefix))
endif
do i=1,2
 flag=7;optional_flag=9
 if(len_trim(result)>0) then
  flag=-77;optional_flag=-99
 endif
 if(i==1) then
  call scheme(55,out,flag,optional_flag)
 else
  call scheme(55,out,flag)
 endif
 if(i==1) then
  call oracle_write('entry','post','first',out,1)
 else
  call oracle_write('entry','post','second',out,1)
 endif
enddo
end program
''')
    (tmp_path/'Makefile').write_text(f'''FC=gfortran
FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-form -ffree-line-length-none -I{transport}
all: capture
native.o: {target}
	$(FC) $(FLAGS) -c $< -o $@
capture: capture.F90 native.o
	$(FC) $(FLAGS) $^ {transport}/transport.o -o $@
''')
    result=subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,
                          env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','r1-rrfs-native-tap')),capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    subprocess.run([str(tmp_path/'capture'),str(tmp_path/'original')],check=True,capture_output=True)
    assert len(list(tmp_path.glob('original.pre.allocation.*.bin')))==9
    subprocess.run([str(tmp_path/'capture'),str(tmp_path/'original'),str(tmp_path/'restored')],check=True,capture_output=True)
    for name in ('first','second'):
        row=json.loads(compare(transport,tmp_path/f'original.post.{name}.bin',tmp_path/f'restored.post.{name}.bin').stdout)
        assert row['bitwise_equal'] and row['words']==55


def test_column_replay_receipt_parts_preserve_all_comparisons(tmp_path):
    spec=importlib.util.spec_from_file_location('column_receipts',ORACLE.parents[1]/'column_oracle_replay.py')
    harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
    result={'schema':'mpas-column-replay-v1','all_outputs_bitwise_equal':False,
            'captures':[{'capture':'capture-a','site':'native-a','arrays':{
                f'state.{i}':{'words':55,'different_words':int(i==17),'bitwise_equal':i!=17,
                             'reference_sha256':'a'*64,'replay_sha256':'b'*64}
                for i in range(30)}},
                {'capture':'capture-b','site':'native-b','arrays':{}},
                {'capture':'capture-c','site':'native-c','arrays':{'flag':{'words':1,'bitwise_equal':True}}}]}
    receipt=tmp_path/'replay.json'
    harness.write_replay_receipt(receipt,result,limit=2400)
    index=json.loads(receipt.read_text());assert index['schema']=='mpas-column-replay-v2'
    assert index['array_count']==31 and index['capture_count']==3
    assert all(p.stat().st_size<=2400 for p in tmp_path.glob('*.json'))
    restored=harness.read_replay_receipt(receipt)
    assert restored['captures']==result['captures']
    assert not restored['all_outputs_bitwise_equal']
    part=tmp_path/index['capture_shards'][0]['file'];part.write_bytes(part.read_bytes()+b' ')
    with pytest.raises(ValueError,match='shard hash mismatch'):harness.read_replay_receipt(receipt)



def test_fork_seal_uses_authenticated_source_pins(transport,tmp_path):
    spec=importlib.util.spec_from_file_location('fork_seal',ORACLE.parents[1]/'column_oracle_replay.py')
    harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
    dump=tmp_path/'dump'
    run_transport(transport,dump,MPAS_ORACLE_DIR=str(dump),MPAS_ORACLE_KEYS='typed',MPAS_ORACLE_STEPS='7')
    sources={'reference':'rrfs','pin_set_sha256':'a'*64,'sites':[],'source_sha256':{},
             'argument_complete':True,'hidden_state_complete':False}
    inventory=tmp_path/'inventory.json';inventory.write_text(json.dumps(sources))
    pins=tmp_path/'PINS.txt';pins.write_text('An explanatory document with its own formatting.\n')
    sealed=harness.seal(dump,inventory,pins,transport/'oracle_words','gnu-strict',1,'rrfs')
    import hashlib
    assert sealed['pin_set_sha256']==sources['pin_set_sha256']
    assert sealed['pins_document_sha256']==hashlib.sha256(pins.read_bytes()).hexdigest()
    assert harness.verify(dump)==sealed



def test_native_caller_sequence_association_replays_captured_rank(transport,tmp_path):
    from dataclasses import asdict
    sys.path.insert(0,str(ORACLE/'replay'))
    from prepare_framework import caller_storage
    from run_capture_matrix import member_prefix
    prefix='native.s0000000007.e0000000001.r000000'
    assert member_prefix(prefix+'.post.state.object%component.bin')==prefix
    with pytest.raises(ValueError,match='invalid native capture filename'):
        member_prefix('../'+prefix+'.post.x.bin')
    native='''module native
contains
subroutine scheme(x,ims,ime,jms,jme)
integer,intent(in) :: ims,ime,jms,jme
real,intent(inout) :: x(ims:ime,jms:jme)
x=x+2.0
end subroutine
end module
'''
    (tmp_path/'native.F90').write_text(native)
    (tmp_path/'driver.F90').write_text('''subroutine driver_example()
use native
real,pointer :: x(:)
allocate(x(-2:-1))
call scheme(x,-2,-1,1,1)
end subroutine
''')
    routine=next(routines(tmp_path/'native.F90',tmp_path))
    site={'site':'native','scheme':'scheme','driver':'driver_example','path':'driver.F90',
          'line':5,'end_line':5,'callee':{'module':'native','path':'native.F90','line':3,'end_line':7},
          'issues':[],'arguments':[dict(asdict(routine.variables[n]),actual=n) for n in routine.args]}
    prepared=caller_storage(site,tmp_path)
    assert prepared['arguments'][0]['capture_rank']==1
    assert 'capture_rank' not in site['arguments'][0]
    (tmp_path/'replay.F90').write_text(replay_source(prepared))
    (tmp_path/'capture.F90').write_text('''program capture
use native
use mpas_oracle_dump
implicit none
real :: x(-2:-1)
integer :: event,ims=-2,ime=-1,jms=1,jme=1
call oracle_set_context([101,205],7,0)
call oracle_begin('native',event)
x=[3.0,7.0]
call oracle_write('native','pre','x',x,event,lower=lbound(x),cell_axis=1)
call oracle_write('native','pre','ims',ims,event)
call oracle_write('native','pre','ime',ime,event)
call oracle_write('native','pre','jms',jms,event)
call oracle_write('native','pre','jme',jme,event)
call scheme(x,ims,ime,jms,jme)
call oracle_write('native','post','x',x,event,lower=lbound(x),cell_axis=1)
end program
''')
    (tmp_path/'Makefile').write_text(f'''FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -I{transport}
all: capture replay
native.o: native.F90
\tgfortran $(FLAGS) -c $< -o $@
capture: capture.F90 native.o
\tgfortran $(FLAGS) $^ {transport}/transport.o -o $@
replay: replay.F90 native.o
\tgfortran $(FLAGS) $^ {transport}/transport.o -o $@
''')
    subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,check=True,capture_output=True,
                   env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','r1-rrfs-native-tap')))
    dump=tmp_path/'dump';dump.mkdir()
    subprocess.run([str(tmp_path/'capture')],cwd=tmp_path,check=True,capture_output=True,
                   env=dict(os.environ,MPAS_ORACLE_DIR=str(dump)))
    subprocess.run([str(tmp_path/'replay'),str(dump/prefix),str(tmp_path/'result')],
                   cwd=tmp_path,check=True,capture_output=True)
    native_output=dump/(prefix+'.post.x.bin');replayed=tmp_path/'result.post.x.bin'
    assert json.loads(compare(transport,native_output,replayed).stdout)['bitwise_equal']



def test_framework_replays_legacy_derived_out_descriptors(transport,tmp_path):
    from dataclasses import asdict
    from prepare_framework import caller_storage
    (tmp_path/'native.F90').write_text('''module native
implicit none
type config
 integer :: mode
 real,allocatable :: data(:)
end type
contains
subroutine scheme(cfg)
type(config),intent(out) :: cfg
cfg%mode=7
allocate(cfg%data(-1:0))
cfg%data=[3.0,9.0]
end subroutine
end module
''')
    (tmp_path/'driver.F90').write_text('''subroutine driver_example()
use native
type(config) :: cfg
call scheme(cfg)
end subroutine
''')
    from derived import primitive_components
    routine=next(routines(tmp_path/'native.F90',tmp_path))
    arg=dict(asdict(routine.variables['cfg']),actual='cfg',derived_module='native',
             components=primitive_components(tmp_path,{'path':'native.F90'},'config','cfg','cfg','out'))
    site={'site':'native','scheme':'scheme','driver':'driver_example','path':'driver.F90',
          'line':4,'end_line':4,'callee':{'module':'native','path':'native.F90','line':9,'end_line':14},
          'issues':[],'arguments':[arg]}
    prepared=caller_storage(site,tmp_path)
    assert all(not component['entry_storage'] for component in prepared['arguments'][0]['components'])
    assert all(component['entry_storage'] for component in site['arguments'][0]['components'])
    source=replay_source(prepared)
    assert '.pre.cfg.' not in source
    (tmp_path/'replay.F90').write_text(source)
    (tmp_path/'capture.F90').write_text('''program capture
use native
use mpas_oracle_dump
implicit none
type(config) :: cfg
integer :: event
call oracle_set_context([101],7,0)
call oracle_begin('native',event)
call scheme(cfg)
call oracle_write('native','post','cfg.mode',cfg%mode,event)
call oracle_write('native','post','cfg.data.present',allocated(cfg%data),event)
call oracle_write('native','post','cfg.data',cfg%data,event,lower=lbound(cfg%data))
end program
''')
    (tmp_path/'Makefile').write_text(f'''FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -I{transport}
all: capture replay
native.o: native.F90
\tgfortran $(FLAGS) -c $< -o $@
capture: capture.F90 native.o
\tgfortran $(FLAGS) $^ {transport}/transport.o -o $@
replay: replay.F90 native.o
\tgfortran $(FLAGS) $^ {transport}/transport.o -o $@
''')
    subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,check=True,capture_output=True,
                   env=dict(os.environ,MPAS_UNIT=os.environ.get('MPAS_UNIT','r1-rrfs-native-tap')))
    dump=tmp_path/'dump';dump.mkdir()
    subprocess.run([str(tmp_path/'capture')],cwd=tmp_path,check=True,capture_output=True,
                   env=dict(os.environ,MPAS_ORACLE_DIR=str(dump)))
    prefix=dump/'native.s0000000007.e0000000001.r000000'
    subprocess.run([str(tmp_path/'replay'),str(prefix),str(tmp_path/'result')],
                   cwd=tmp_path,check=True,capture_output=True)
    for name in ('cfg.mode','cfg.data','cfg.data.present'):
        assert json.loads(compare(transport,Path(str(prefix)+'.post.'+name+'.bin'),
                                  tmp_path/('result.post.'+name+'.bin')).stdout)['bitwise_equal']



def test_native_coverage_authenticates_makefile_source_copies(tmp_path):
    from replay.coverage import summarize
    import hashlib
    native=tmp_path/'native';build=tmp_path/'build'
    wrapper=Path('package/MPAS/module.F90');compiled=Path('package/module.F90')
    (native/wrapper).parent.mkdir(parents=True);(build/wrapper).parent.mkdir(parents=True)
    (native/wrapper).write_text('subroutine work\nif(enabled) call native_operation()\nend subroutine\n')
    instrumented='! owned instrumentation\n'+(native/wrapper).read_text()
    (build/wrapper).write_text(instrumented);(build/compiled).write_text(instrumented)
    digest=hashlib.sha256(instrumented.encode()).hexdigest()
    data={'gcc_version':'15','current_working_directory':str(build/'package'),
          'files':[{'file':'module.F90','lines':[{'line_number':3,'count':1,
                    'branches':[{'count':1},{'count':0}]}]}]}
    result=summarize(data,native,build,generated_source_sha256={str(wrapper):digest})['files'][0]
    assert result['native_source']==str(wrapper) and result['compiled_source']==str(compiled)
    assert result['compiled_copy_sha256']==digest
    assert result['branches_taken']==1 and result['branches_total']==2
    assert result['uncovered_branches'][0]['line']==2
    # A basename match cannot authorize a changed compiler input.
    (build/compiled).write_text(instrumented+'! changed compiler input\n')
    with pytest.raises(ValueError,match='unique authenticated native copy'):
        summarize(data,native,build,generated_source_sha256={str(wrapper):digest})
    (build/compiled).write_text(instrumented)
    (build/wrapper).write_text('changed inventoried source')
    with pytest.raises(ValueError,match='inventoried coverage source bytes differ'):
        summarize(data,native,build,generated_source_sha256={str(wrapper):digest})
def test_whole_native_logger_target_state(transport, tmp_path):
    """Pinned mpas_log.F:54 and complete mpas_log_types.inc:11-43 targets."""
    import hashlib
    from state import generate_state
    native_build = os.environ.get('MPAS_ORACLE_NATIVE_BUILD')
    if not native_build:
        pytest.skip('set MPAS_ORACLE_NATIVE_BUILD for whole native logger fixture')
    native_build = Path(native_build)
    pinned = site_support.mpas_source(preferred=('MPAS_ORACLE_NATIVE_SOURCE',))
    root = tmp_path/'source'; framework=root/'src/framework'; framework.mkdir(parents=True)
    physics=root/'src/core_atmosphere/physics'; physics.mkdir(parents=True)
    for name in ('mpas_log.F','mpas_log_types.inc'):
        shutil.copy2(pinned/'src/framework'/name,framework/name)
        (framework/name).chmod(0o644)
    (physics/'mpas_atmphys_init.F').write_text('''module mpas_atmphys_vars
use mpas_log
contains
subroutine physics_init(mesh)
integer :: mesh
mesh=mesh
end subroutine physics_init
end module mpas_atmphys_vars
''')
    generated=tmp_path/'generated';shutil.copytree(root,generated)
    manifest={'sites':[],'boundaries':[]};generate_state(root,generated,manifest)
    logger=next(m for m in manifest['module_state']if m['name']=='mpas_log')
    assert len(logger['variables'])==21
    assert sum(v.get('container_pointer',False)for v in logger['variables'])==3
    assert any('file-descriptor' in issue['reason']for issue in logger['issues'])
    assert not manifest['hidden_state_complete']
    for directory in ('src/framework','src/external/esmf_time_f90'):
        for path in (native_build/directory).glob('*.mod'):
            shutil.copy2(path,tmp_path/path.name)
    for directory,library in [('src/framework','libframework.a'),
                              ('src/external/esmf_time_f90','libesmf_time.a')]:
        shutil.copy2(native_build/directory/library,tmp_path/library)
    (tmp_path/'logger.F90').write_text('''program logger_roundtrip
use mpas_derived_types,only:mpas_log_type,mpas_log_file_type,MPAS_LOG_OUT,MPAS_LOG_WARN,MPAS_LOG_ERR
use mpas_log,only:mpas_log_info,mpas_log_write,oracle_state_mpas_log,oracle_restore_mpas_log
use mpas_oracle_dump,only:oracle_set_context,oracle_begin,oracle_set_output
implicit none
type(mpas_log_type),target :: provider
type(mpas_log_file_type),target :: output_file,error_file
character(len=4096) :: mode,kind,prefix,result
integer :: first_unit,second_unit,event,iteration,count_before
logical :: cold
call get_command_argument(1,mode)
call get_command_argument(2,kind)
call get_command_argument(3,prefix)
call get_command_argument(4,result)
open(newunit=first_unit,file='fixture-output',status='replace')
open(newunit=second_unit,file='fixture-error',status='replace')
provider%outputLog=>output_file
provider%errorLog=>error_file
provider%taskID=0;provider%nTasks=1;provider%coreName='whole-native-logger';provider%domainID=7
provider%outputMessageCount=0;provider%warningMessageCount=0
provider%errorMessageCount=0;provider%criticalErrorMessageCount=13
output_file%unitNum=first_unit;output_file%fileName='fixture-output'
output_file%isOpen=.true.;output_file%isActive=.true.;output_file%openedByLogModule=.false.
error_file%unitNum=second_unit;error_file%fileName='fixture-error'
error_file%isOpen=.true.;error_file%isActive=.true.;error_file%openedByLogModule=.false.
mpas_log_info=>provider
call oracle_set_context([1],1,0)
if(trim(mode)=='capture') then
 if(trim(kind)=='shared')provider%errorLog=>provider%outputLog
 if(trim(kind)=='absent')nullify(mpas_log_info)
 if(associated(mpas_log_info))then
  call mpas_log_write('native output',messageType=MPAS_LOG_OUT)
  call mpas_log_write('native warning',messageType=MPAS_LOG_WARN)
  call mpas_log_write('native error',messageType=MPAS_LOG_ERR)
 endif
 call oracle_begin('logger',event)
 call oracle_set_output(trim(prefix))
 call oracle_state_mpas_log('logger','pre',event)
else
 cold=trim(mode)=='cold'
 if(cold)nullify(mpas_log_info)
 do iteration=1,2
  if(associated(mpas_log_info))then
   mpas_log_info%taskID=-1;mpas_log_info%nTasks=-1;mpas_log_info%coreName='mutated'
   mpas_log_info%domainID=-1;mpas_log_info%outputMessageCount=-1
   mpas_log_info%warningMessageCount=-1;mpas_log_info%errorMessageCount=-1
   mpas_log_info%criticalErrorMessageCount=-1
   if(trim(kind)=='independent')mpas_log_info%errorLog=>mpas_log_info%outputLog
   if(trim(kind)=='shared')mpas_log_info%errorLog=>error_file
   if(associated(mpas_log_info%outputLog))then
    mpas_log_info%outputLog%unitNum=-99;mpas_log_info%outputLog%fileName='mutated'
    mpas_log_info%outputLog%isOpen=.false.;mpas_log_info%outputLog%isActive=.false.
    mpas_log_info%outputLog%openedByLogModule=.true.
   endif
   if(associated(mpas_log_info%errorLog))then
    mpas_log_info%errorLog%unitNum=-99;mpas_log_info%errorLog%fileName='mutated'
    mpas_log_info%errorLog%isOpen=.false.;mpas_log_info%errorLog%isActive=.false.
    mpas_log_info%errorLog%openedByLogModule=.true.
   endif
  endif
  call oracle_restore_mpas_log(trim(prefix))
  if(trim(kind)=='absent')then
   if(associated(mpas_log_info))error stop 'absent root not restored'
  else
   if(.not.associated(mpas_log_info))error stop 'logger root not restored'
   if(.not.cold)then
    if(.not.associated(mpas_log_info,provider))error stop 'native domain alias detached'
   endif
   if(trim(kind)=='shared')then
    if(.not.associated(mpas_log_info%outputLog,mpas_log_info%errorLog))error stop 'shared files split'
   else
    if(associated(mpas_log_info%outputLog,mpas_log_info%errorLog))error stop 'independent files aliased'
   endif
  endif
  call oracle_begin('logger',event)
  call oracle_set_output(trim(result)//achar(48+iteration))
  call oracle_state_mpas_log('logger','post',event)
  if(associated(mpas_log_info))then
   count_before=mpas_log_info%outputMessageCount
   call mpas_log_write('actual native restored logger behavior')
   if(mpas_log_info%outputMessageCount/=count_before+1)error stop 'native logger behavior differs'
  endif
 enddo
endif
close(first_unit);close(second_unit)
end program
''')
    source=generated/'src/framework/mpas_log.F'
    (tmp_path/'Makefile').write_text(f'''FC=mpif90
FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-form -ffree-line-length-none -fconvert=big-endian
all: logger_roundtrip
mpas_log.o: {source}
\t$(FC) $(FLAGS) -D_MPI -c $< -o $@
logger_roundtrip: logger.F90 mpas_log.o
\t$(FC) $(FLAGS) -I{transport} $^ {transport}/transport.o -Wl,--start-group libframework.a libesmf_time.a -Wl,--end-group -lnetcdff -lnetcdf -o $@
''')
    result=subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,
        env=dict(os.environ,MPAS_UNIT='f1-native-tap'),capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    records=[]
    membership_receipts=[]
    from replay.audit_state import membership
    logger_variables=[('mpas_log', v) for v in logger['variables']]

    def audit_members(folder, label, phase):
        names={p.name for p in folder.glob(f'{label}.{phase}.*.bin')}
        result=membership(logger_variables, label, phase, names,
            lambda member: json.loads(subprocess.check_output(
                [str(transport/'oracle_words'),'present',str(folder/member),'-'],text=True)))
        assert result['complete'], result
        assert not result['alias_semantics_complete']
        membership_receipts.append(dict(kind=folder.name, label=label, phase=phase, **result))
        # A missing root descriptor or alias never authorizes omitting leaves.
        for suffix in ['.present.bin','.alias.bin']:
            root_record=f'{label}.{phase}.state.mpas_log.mpas_log_info{suffix}'
            broken=membership(logger_variables, label, phase, names-{root_record},
                lambda member: json.loads(subprocess.check_output(
                    [str(transport/'oracle_words'),'present',str(folder/member),'-'],text=True)))
            assert not broken['complete'] and root_record in broken['missing']

    for kind in ['independent','shared','absent']:
        folder=tmp_path/kind;folder.mkdir();prefix=folder/'capture'
        env=dict(os.environ,MPAS_ORACLE_DIR=str(folder),MPAS_ORACLE_STEPS='1',MPAS_ORACLE_KEYS='logger')
        subprocess.run([str(tmp_path/'logger_roundtrip'),'capture',kind,str(prefix),'unused'],
                       cwd=folder,env=env,check=True,capture_output=True)
        audit_members(folder, 'capture', 'pre')
        for mode in ['warm','cold']:
            target=folder/mode
            run=subprocess.run([str(tmp_path/'logger_roundtrip'),mode,kind,str(prefix),str(target)],
                               cwd=folder,env=env,capture_output=True,text=True)
            assert run.returncode==0,run.stdout+run.stderr
            for iteration in [1,2]:
                audit_members(folder, mode+str(iteration), 'post')
            for original in folder.glob('capture.pre.*.bin'):
                suffix=original.name.removeprefix('capture.pre.')
                for iteration in [1,2]:
                    replayed=folder/(mode+str(iteration)+'.post.'+suffix)
                    comparison=json.loads(compare(transport,original,replayed).stdout)
                    assert comparison['bitwise_equal'],(kind,mode,suffix)
                    records.append(dict(kind=kind,mode=mode,iteration=iteration,
                        pre_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                        post_sha256=hashlib.sha256(replayed.read_bytes()).hexdigest(),comparison=comparison))
    # Unsupported changed native layouts cannot silently shed hidden fields.
    declaration=framework/'mpas_log_types.inc';text=declaration.read_text()
    declaration.write_text(text.replace('integer :: domainID','type(unknown_native_type), pointer :: domainID'))
    from state import modules
    with pytest.raises(ValueError,match='topology changed'):
        modules(root)
    declaration.write_text(text)
    missing=tmp_path/'missing';missing.mkdir()
    for original in (tmp_path/'independent').glob('capture.pre.*.bin'):
        shutil.copy2(original,missing/original.name)
    # Native derived-component selectors remain '%' in record names.
    absent=missing/'capture.pre.state.mpas_log.mpas_log_info%errormessagecount.bin'
    assert absent.is_file()
    absent.unlink()
    refused=subprocess.run([str(tmp_path/'logger_roundtrip'),'cold','independent',
                            str(missing/'capture'),str(missing/'result')],cwd=missing,
        env=dict(os.environ,MPAS_ORACLE_DIR=str(missing),MPAS_ORACLE_STEPS='1',MPAS_ORACLE_KEYS='logger'),
        capture_output=True,text=True)
    assert refused.returncode!=0 and 'oracle missing input record' in refused.stdout+refused.stderr
    assert not list(missing.glob('result*.post.*.bin'))
    (tmp_path/'logger-proof.json').write_text(json.dumps(dict(
        schema='f1-whole-native-logger-target-restore-v49',process_count=1,
        native_source_sha256=hashlib.sha256((framework/'mpas_log.F').read_bytes()).hexdigest(),
        native_types_sha256=hashlib.sha256(text.encode()).hexdigest(),
        generated_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        executable_sha256=hashlib.sha256((tmp_path/'logger_roundtrip').read_bytes()).hexdigest(),
        comparisons=records,all_bitwise_equal=True,global_hidden_state_complete=False,
        conditional_membership=membership_receipts,
        external_domain_root_preserved=True,file_descriptors_restored=False,
        missing_pre_refused=True,missing_pre_exit=refused.returncode),indent=2)+'\n')


@pytest.mark.parametrize('recipe', ['gnu-strict', 'gnu-o3', 'coverage'])
def test_forwarded_absent_actual_keeps_required_native_dummy(transport, tmp_path, recipe):
    """Real GNU calls: NULL forwarded to an unused required array/scalar dummy.

    RRFS SMOKE uses this ABI when ebb_dcycle=1 forwards absent fire_end_hr.
    Observing or replaying it must preserve the native required declaration.
    """
    from generate import inventory, overlay
    from replay.generate_fixture import fixture
    physics=tmp_path/'src/core_atmosphere/physics';physics.mkdir(parents=True)
    native=physics/'native.F90'
    native.write_text('''module native_scheme
contains
subroutine calculate(selector,x,y,result)
 integer,intent(in) :: selector
 real,intent(inout) :: x(-2:2),y
 real,intent(inout) :: result
 if(selector==1) then
  x=x+y
  y=y+1.0
  result=sum(x)+y
 else
  result=42.0
 endif
end subroutine calculate
end module native_scheme
''')
    driver=physics/'driver.F90'
    driver.write_text('''module native_driver
use native_scheme
contains
subroutine driver_example(selector,result,x,y)
 integer,intent(in) :: selector
 real,intent(inout) :: result
 real,optional,intent(inout) :: x(-2:2),y
 call calculate(selector,x,y,result)
end subroutine driver_example
end module native_driver
''')
    info=inventory(tmp_path,authenticate=False);site=info['sites'][0]
    assert not site['issues']
    forwarded=[a for a in site['arguments'] if a['name'] in ['x','y']]
    assert all(a['actual_optional'] and not a['optional'] for a in forwarded)
    assert [a['actual_presence_guard'] for a in forwarded]==['present(x)','present(y)']
    # Perturbations may not dereference an absent required dummy either.
    with pytest.raises(ValueError,match='present mandatory input'):
        fixture(site,dict(precision='real32',assignments={'x':'x+1.0'}))
    instrumented=tmp_path/'observed';overlay(tmp_path,instrumented,info)
    replay=tmp_path/'replay.F90';replay.write_text(replay_source(site))
    assert 'real(real32), allocatable :: y' in replay.read_text()
    assert 'x=x' in replay.read_text() and 'y=y' in replay.read_text()
    main=tmp_path/'main.F90'
    main.write_text('''program exercise
 use native_driver
 implicit none
 real :: x(-2:2),y,result
 result=-99.0
 call driver_example(0,result)
 if(result/=42.0) error stop 'absent native result changed'
 print *,result
 x=[-0.0,1.25,-2.5,4.0,8.0];y=-0.75;result=-77.0
 call driver_example(1,result,x,y)
 print *,x,y,result
end program exercise
''')
    (tmp_path/'ordinary').mkdir();(tmp_path/'tapped').mkdir()
    flags='-O3' if recipe=='gnu-o3' else '-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations'
    if recipe=='coverage':flags+=' --coverage'
    flags+=' -ffree-form -ffree-line-length-none -fconvert=big-endian'
    observed_driver=instrumented/driver.relative_to(tmp_path)
    (tmp_path/'Makefile').write_text(f'''FC=gfortran
FLAGS={flags}
all: ordinary/model tapped/model replay
ordinary/model: {native} {driver} {main}
\tcd ordinary && $(FC) $(FLAGS) -c {native} -o native.o && $(FC) $(FLAGS) -c {driver} -o driver.o && $(FC) $(FLAGS) {main} native.o driver.o -o model
tapped/model: {native} {observed_driver} {main}
\tcd tapped && $(FC) $(FLAGS) -I{transport} -c {native} -o native.o && $(FC) $(FLAGS) -I{transport} -c {observed_driver} -o driver.o && $(FC) $(FLAGS) -I{transport} {main} native.o driver.o {transport}/transport.o -o model
replay: replay.F90 ordinary/model
\t$(FC) $(FLAGS) -Iordinary -I{transport} replay.F90 ordinary/native.o {transport}/transport.o -o replay
''')
    result=subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,
        env=dict(os.environ,MPAS_UNIT='r1-rrfs-native-tap'),capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    clean_env={k:v for k,v in os.environ.items() if not k.startswith('MPAS_ORACLE_')}
    ordinary=subprocess.run([str(tmp_path/'ordinary/model')],cwd=tmp_path,env=clean_env,
                            check=True,capture_output=True)
    packets=tmp_path/'packets';packets.mkdir()
    env=dict(clean_env,MPAS_ORACLE_DIR=str(packets),MPAS_ORACLE_KEYS=site['site'],MPAS_ORACLE_STEPS='0')
    observed=subprocess.run([str(tmp_path/'tapped/model')],cwd=tmp_path,env=env,
                            check=True,capture_output=True)
    assert ordinary.stdout==observed.stdout
    prefixes=sorted(p.name.removesuffix('.pre.selector.bin') for p in packets.glob('*.pre.selector.bin'))
    assert len(prefixes)==2
    for n,prefix in enumerate(prefixes):
        outputs=tmp_path/('output-'+str(n));outputs.mkdir();target=outputs/'result'
        result=subprocess.run([str(tmp_path/'replay'),str(packets/prefix),str(target)],
                              cwd=tmp_path,env=clean_env,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        expected={p.name.removeprefix(prefix+'.post.'):p for p in packets.glob(prefix+'.post.*.bin')}
        assert set(expected)=={p.name.removeprefix('result.post.') for p in outputs.glob('result.post.*.bin')}
        for name,original in expected.items():
            actual=outputs/('result.post.'+name)
            comparison=json.loads(compare(transport,original,actual).stdout)
            assert comparison['bitwise_equal'],(recipe,n,name,comparison)
        for name in ['x','y']:
            flag=json.loads(subprocess.run([str(transport/'oracle_words'),'present',
                str(packets/(prefix+'.pre.'+name+'.present.bin')),'-'],
                check=True,capture_output=True,text=True).stdout)
            assert flag==(n==1)
            assert (packets/(prefix+'.pre.'+name+'.bin')).exists()==(n==1)


def test_native_coverage_preserves_authenticated_include_branches(tmp_path):
    from replay.coverage import summarize, validate_object_sources
    import hashlib
    source=tmp_path/'native';build=tmp_path/'build'
    primary=Path('framework/fields.F');first=Path('framework/duplicate.inc');nested=Path('framework/tails.inc')
    contents={primary:'#include "duplicate.inc"\nsubroutine fields\nend\n',
              first:'include "tails.inc"\nif (enabled) call work()\n',
              nested:'if (associated(next)) call advance()\n'}
    hashes={}
    for relative,text in contents.items():
        for root in (source,build):
            (root/relative).parent.mkdir(parents=True,exist_ok=True)
            (root/relative).write_text(text)
        hashes[str(relative)]=hashlib.sha256(text.encode()).hexdigest()
    data={'gcc_version':'15','current_working_directory':str(build/'framework'),
          'files':[{'file':p.name,'lines':[{'line_number':1,'count':1,
                    'branches':[{'count':1},{'count':0}]}]} for p in contents]}
    report=summarize(data,source,build)
    validate_object_sources(report,primary,source,build,hashes)
    assert len(report['files'])==3
    assert sum(r['branches_total'] for r in report['files'])==6
    assert sum(r['branches_taken'] for r in report['files'])==3
    assert all(r['included_by_native_object']==str(primary) for r in report['files'][1:])
    assert all(r['uncovered_branches'] for r in report['files'])
    (build/nested).write_text('changed compiler header')
    with pytest.raises(ValueError,match='included source changed'):
        validate_object_sources(report,primary,source,build,hashes)
    (build/nested).write_text(contents[nested])
    # An unchanged but unrelated source cannot join this object's coverage.
    rogue=dict(report['files'][0],native_source='framework/other.F')
    with pytest.raises(ValueError,match='different native source'):
        validate_object_sources(dict(report,files=report['files']+[rogue]),primary,source,build,hashes)
    with pytest.raises(ValueError,match='absent or duplicated'):
        validate_object_sources(dict(report,files=report['files'][1:]),primary,source,build,hashes)
    (source/nested).write_text('changed native header')
    with pytest.raises(ValueError,match='included source changed'):
        validate_object_sources(report,primary,source,build,hashes)

    (source/nested).write_text(contents[nested])
    (source/primary).write_text('changed primary native source')
    with pytest.raises(ValueError,match='primary source changed'):
        validate_object_sources(report,primary,source,build,hashes)
    with pytest.raises(ValueError,match='escapes root'):
        validate_object_sources(report,Path('../fields.F'),source,build,hashes)


def test_coverage_maps_only_verified_native_conditional_allocate(tmp_path):
    import gzip
    from replay.coverage import summarize, allocation_control_mapping
    source_root=tmp_path/'original';build=tmp_path/'build'
    source_root.mkdir();build.mkdir()
    original='''program native
 implicit none
 real,allocatable :: a(:)
 integer :: i,n
 character(len=20) :: argument
 call get_command_argument(1,argument)
 read(argument,*) n
 do i=1,n
  if(.not.allocated(a)) allocate(a(2))
  a=real(i)
 enddo
 print *,a
end program
'''
    line=next(i+1 for i,s in enumerate(original.splitlines())if 'if(.not.allocated' in s)
    tag=f'native.native.a.L{line}'
    block=f'''  if(.not.allocated(a)) then
  allocate(a(2))
  call oracle_allocation('{tag}', &
       a,lower=lbound(a))
  endif'''
    generated=original.replace('  if(.not.allocated(a)) allocate(a(2))',block)
    generated=generated.replace(' implicit none',' interface\n subroutine oracle_allocation(tag,a,lower)\n character(len=*),intent(in)::tag\n real,intent(in)::a(:)\n integer,intent(in)::lower(:)\n end subroutine\n end interface\n implicit none',1)
    # IMPLICIT NONE must precede an interface block in native Fortran.
    generated=generated.replace(' interface\n',' implicit none\n interface\n',1).replace(' end interface\n implicit none',' end interface',1)
    generated+='''subroutine oracle_allocation(tag,a,lower)
 implicit none
 character(len=*),intent(in)::tag
 real,intent(in)::a(:)
 integer,intent(in)::lower(:)
end subroutine
'''
    (source_root/'native.F90').write_text(original)
    path=build/'native.F90';path.write_text(generated)
    (build/'Makefile').write_text('all: native\nnative: native.F90\n\tgfortran -O0 --coverage -ffree-form -c $< -o native.o\n\tgfortran --coverage native.o -o $@\n')
    compiled=subprocess.run(_build_command(['make','-j4']),cwd=build,
        env=dict(os.environ,MPAS_UNIT='f1-native-tap'),capture_output=True,text=True)
    assert compiled.returncode==0,compiled.stdout+compiled.stderr
    subprocess.run([str(build/'native'),'2'],cwd=build,check=True,capture_output=True)
    collected=subprocess.run(['gcov','-j','-b','-c','native.gcda'],cwd=build,capture_output=True,text=True)
    assert collected.returncode==0,collected.stdout+collected.stderr
    data=json.loads(gzip.decompress((build/'native.gcov.json.gz').read_bytes()))
    legacy=summarize(data,source_root,build,include_counters=True)
    result=summarize(data,source_root,build,include_counters=True,map_allocation_controls=True)
    assert line in legacy['files'][0]['unmapped_native_lines']
    assert line not in result['files'][0]['unmapped_native_lines']
    mapped=result['files'][0]['allocation_control_mappings']
    assert len(mapped)==1 and mapped[0]['native_line']==line and mapped[0]['tag']==tag
    conditional=[r for r in result['files'][0]['line_counters']if r['line']==line and r['branches']]
    # Allocation runtime checks are native too; retain them beside the IF.
    raw_rows=[r for r in data['files'][0]['lines']if r['line_number'] in
              [mapped[0]['generated_condition_line'],mapped[0]['generated_allocate_line']]]
    assert len(conditional)>=1
    assert result['files'][0]['branches_total']==legacy['files'][0]['branches_total']+sum(len(r['branches'])for r in raw_rows)
    # Counter block IDs/counts retain actual compiler observations, not guesses.
    raw_row=next(r for r in raw_rows if r['line_number']==mapped[0]['generated_condition_line'])
    assert len(raw_row['branches'])==2 and all(b['count']>0 for b in raw_row['branches'])
    assert any(r['branches']==raw_row['branches']for r in conditional)
    for changed in [generated.replace('if(.not.allocated(a)) then','if(allocated(a)) then'),
                    generated.replace('allocate(a(2))','allocate(a(3))'),
                    generated.replace(tag,f'native.native.a.L{line+1}')]:
        assert not allocation_control_mapping(original.splitlines(),changed.splitlines(),{})
    duplicate=generated.replace(block,block+'\n'+block)
    assert not allocation_control_mapping(original.splitlines(),duplicate.splitlines(),{})


# r1-rrfs-native-tap: original external scalar-character transport and complete boundaries.
def test_external_scalar_character_preserves_every_byte_and_native_read(transport, tmp_path):
    import struct
    # Include NULs, high bytes, spaces and all byte values: no text decode/re-encode.
    payload = bytes(range(256)) * 2
    header = [5, 512, 0] + [1] * 5 + [1] * 5 + [7, 2, 0, 0, 0, 0, 256, 0]
    tags = b''.join(x.ljust(256, b' ') for x in (b'boundary', b'pre', b'config_label'))
    original = tmp_path / 'original.bin'
    original.write_bytes(b'ORCL0001' + struct.pack('<21i', *header) + tags + payload)
    before = original.read_bytes()
    canonical = tmp_path / 'canonical.bin'
    result = subprocess.run([str(transport / 'oracle_words'), 'normalize-character',
                             str(original), str(canonical)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == dict(character_bytes=512, payload_unchanged=True,
                                           ids_unchanged=True, tags_unchanged=True)
    assert original.read_bytes() == before
    assert canonical.read_bytes()[92:] == before[92:]
    metadata = json.loads(subprocess.check_output([str(transport / 'oracle_words'), 'header',
                                                  str(canonical)], text=True))
    assert metadata['character_length'] == 512 and metadata['width'] == 1
    (tmp_path / 'main.F90').write_text("""program character_read
 use mpas_oracle_dump
 implicit none
 character(len=512) :: text
 character(len=4096) :: input,output
 integer :: event,i
 call get_command_argument(1,input)
 call get_command_argument(2,output)
 call oracle_read(trim(input),text)
 do i=1,len(text)
   if(iachar(text(i:i))/=mod(i-1,256)) error stop 'character byte changed'
 enddo
 call oracle_restore_context(trim(input),event)
 call oracle_set_output(trim(output))
 call oracle_write('boundary','pre','config_label',text,event)
end program
""")
    (tmp_path / 'Makefile').write_text(f"""all: character_read
character_read: main.F90
\tgfortran -ffree-line-length-none -I{transport} main.F90 {transport}/transport.o -o $@
""")
    build = subprocess.run(_build_command(['make', '-j4']), cwd=tmp_path,
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    prefix = tmp_path / 'roundtrip'
    subprocess.run([str(tmp_path / 'character_read'), str(canonical), str(prefix)], check=True)
    actual = next(tmp_path.glob('roundtrip*.bin'))
    assert json.loads(compare(transport, canonical, actual).stdout)['different_words'] == 0
    duplicate = subprocess.run([str(transport / 'oracle_words'), 'normalize-character',
                                str(original), str(canonical)], capture_output=True)
    assert duplicate.returncode != 0 and original.read_bytes() == before


@pytest.mark.parametrize('code,width,rank,dims', [
    (3, 4, 0, [1]*5), (5, 4, 1, [1]*5),
    (5, 4, 0, [2,1,1,1,1]), (5, 1, 0, [2,2,1,1,1]),
])
def test_external_character_refuses_ambiguous_or_noncharacter_records(
        transport, tmp_path, code, width, rank, dims):
    import struct
    import math
    header = [code,width,rank] + dims + [1]*5 + [0,0,0,0,0,0,256,0]
    source = tmp_path / 'source.bin'; target = tmp_path / 'target.bin'
    source.write_bytes(b'ORCL0001' + struct.pack('<21i',*header) + b' '*768 +
                       b'x'*(width*math.prod(dims)))
    result = subprocess.run([str(transport/'oracle_words'),'normalize-character',
                             str(source),str(target)],capture_output=True)
    assert result.returncode != 0 and not target.exists()


def observed_test_bank(tmp_path):
    from replay.replay_observed_boundaries import sha
    source=tmp_path/'source'; source.mkdir()
    path=source/'native.F'
    path.write_text("""module original
 private
contains
 subroutine before(configs,state)
 integer :: configs,state
 state=state+configs
 end subroutine
 subroutine after(configs,state)
 integer :: configs,state
 state=state-configs
 end subroutine
end module
""")
    bank=tmp_path/'bank';bank.mkdir();(bank/'dump').mkdir()
    producer=dict(process_count=1,host='native-host',native_sha256='native',
                  observer_sha256='observer',input_request_sha256='',runs={'active':{'exit':0}})
    passive=dict(bitwise_equal=True,different_words=0,host='native-host',words=17)
    (bank/'input-request.json').write_text('{}')
    producer['input_request_sha256']=sha(bank/'input-request.json')
    (bank/'producer.json').write_text(json.dumps(producer))
    (bank/'passivity.json').write_text(json.dumps(passive))
    captures={}
    for key in ('before','after'):
        capture=key+'.s0000000000.e0000000000.r000000'
        members={}
        for phase in ('pre','post'):
            member=capture+'.'+phase+'.argument.state.bin'
            (bank/'dump'/member).write_bytes(b'fixture-authentication-only-'+phase.encode())
            members[member]=sha(bank/'dump'/member)
        captures[capture]={'files':members}
    native=dict(schema='test-original-bank',model_commit='test-pin',input_origin='native-tap',
                process_count=1,observer_passivity_proven=True,native_binary_sha256='native',
                observer_sha256='observer',input_request_sha256=sha(bank/'input-request.json'),
                producer_attempt_sha256=sha(bank/'producer.json'),passivity_receipt_sha256=sha(bank/'passivity.json'),
                source_sha256={'native.F':sha(path)},captures=captures)
    (bank/'manifest.json').write_text(json.dumps(native))
    spec=dict(metadata={name:sha(bank/name) for name in ('manifest.json','producer.json','passivity.json','input-request.json')},
              native_bank_schema='test-original-bank',model_commit='test-pin',source='native.F',source_sha256=sha(path),
              module='original',routines={key:dict(routine=key,arguments=['configs','state']) for key in ('before','after')})
    return bank,source,spec


@pytest.mark.parametrize('mutation',['packet','extra-packet','metadata','source','missing-post','missing-call','failed-producer'])
def test_complete_boundary_authentication_rejects_mutated_or_incomplete_banks(tmp_path,mutation):
    from replay.replay_observed_boundaries import authenticate,sha
    bank,source,spec=observed_test_bank(tmp_path)
    native=json.loads((bank/'manifest.json').read_text())
    _,_,_,records=authenticate(bank,source,spec)
    assert len(records)==2
    capture=next(iter(native['captures']));members=native['captures'][capture]['files']
    if mutation=='packet':
        (bank/'dump'/next(iter(members))).write_bytes(b'changed')
    elif mutation=='extra-packet':
        (bank/'dump'/(capture+'.post.hidden.bin')).write_bytes(b'extra')
    elif mutation=='metadata':
        (bank/'passivity.json').write_text('{}')
    elif mutation=='source':
        (source/'native.F').write_text('changed')
    elif mutation=='missing-post':
        member=next(name for name in members if '.post.' in name)
        del members[member];(bank/'dump'/member).unlink()
        (bank/'manifest.json').write_text(json.dumps(native));spec['metadata']['manifest.json']=sha(bank/'manifest.json')
    elif mutation=='missing-call':
        del native['captures'][capture]
        (bank/'manifest.json').write_text(json.dumps(native));spec['metadata']['manifest.json']=sha(bank/'manifest.json')
    else:
        producer=json.loads((bank/'producer.json').read_text());producer['runs']['active']['exit']=1
        (bank/'producer.json').write_text(json.dumps(producer));spec['metadata']['producer.json']=sha(bank/'producer.json')
        native['producer_attempt_sha256']=sha(bank/'producer.json')
        (bank/'manifest.json').write_text(json.dumps(native));spec['metadata']['manifest.json']=sha(bank/'manifest.json')
    with pytest.raises(ValueError):authenticate(bank,source,spec)


def test_complete_boundary_private_export_preserves_entire_native_module(tmp_path):
    from replay.replay_observed_boundaries import export_source
    bank,source,spec=observed_test_bank(tmp_path)
    original=(source/'native.F').read_text()
    changed,ranges=export_source(source,spec)
    assert changed.replace('\n public :: before,after\n','',1)==original
    assert len(ranges)==2 and ranges[0]['line']<ranges[1]['line']
    spec['routines']['after']['arguments']=['configs']
    with pytest.raises(ValueError,match='interface'):export_source(source,spec)
