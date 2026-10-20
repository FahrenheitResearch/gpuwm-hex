"""Failure and concurrency contracts; real-builder identity is a separate proof."""
import json
import os
from pathlib import Path
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from hexcore import static_cache as sc


@pytest.fixture
def case(tmp_path):
    engine, grid, geog = tmp_path/'builder', tmp_path/'grid', tmp_path/'geog'
    engine.write_bytes(b'builder-v1'); grid.write_bytes(b'mesh-v1'); geog.mkdir()
    (geog/'tile').write_bytes(b'geography-v1')
    return dict(engine=engine, grid=grid, geog=geog, cache=tmp_path/'cache',
                out=tmp_path/'out.nc', receipt=tmp_path/'out.json')


def fake_builder(command, **kwargs):
    def value(flag): return Path(command[command.index(flag)+1])
    value('--out').write_bytes(value('--grid').read_bytes()+b'|'+(value('--geog')/'tile').read_bytes())
    value('--receipt').write_text(json.dumps(dict(schema='builder-test',output_path=str(value('--out')))))


def test_reuse_rebuild_and_receipt(case):
    with patch.object(sc.subprocess, 'run', side_effect=fake_builder) as run:
        a=sc.build_static(**case)
        b=sc.build_static(**case)
        proof=sc.build_static(**case, verify_rebuild=True)
    assert not a['hit'] and b['hit'] and proof['rebuild_comparison']['equal']
    assert run.call_count==2
    assert not (Path(proof['rebuild_comparison']['rebuild'])/'static.nc').exists()
    assert proof['deleted'][0]['bytes'] > 0
    assert Path(case['out']).read_bytes()==b'mesh-v1|geography-v1'
    assert json.loads(case['receipt'].read_text())['output_path']==str(case['out'])


@pytest.mark.parametrize('changed', ['grid','engine','geog'])
def test_content_change_invalidates_even_with_restored_mtime(case, changed):
    with patch.object(sc.subprocess, 'run', side_effect=fake_builder):
        a=sc.build_static(**case)
        p=case[changed] if changed!='geog' else case['geog']/'tile'
        before=p.stat(); p.write_bytes(p.read_bytes().replace(b'v1', b'v2'))
        os.utime(p, ns=(before.st_atime_ns,before.st_mtime_ns))
        case['out']=case['out'].with_name('second.nc')
        b=sc.build_static(**case)
    assert a['key']!=b['key'] and not b['hit']


def test_new_geography_file_changes_key(case):
    with patch.object(sc.subprocess, 'run', side_effect=fake_builder):
        a=sc.build_static(**case)
        (case['geog']/'new-tile').write_bytes(b'new')
        b=sc.build_static(**case)
    assert a['key']!=b['key']


def test_corrupt_cache_refused(case):
    with patch.object(sc.subprocess, 'run', side_effect=fake_builder):
        a=sc.build_static(**case)
        (Path(a['cache'])/'static.nc').write_bytes(b'broken')
        with pytest.raises(RuntimeError, match='integrity'):
            sc.build_static(**case)


def test_foreign_output_not_replaced(case):
    case['out'].write_bytes(b'belongs to another case')
    with patch.object(sc.subprocess, 'run', side_effect=fake_builder):
        with pytest.raises(FileExistsError): sc.build_static(**case)
    assert case['out'].read_bytes()==b'belongs to another case'


def test_explicit_clobber_points_to_shared_artifact(case):
    case['out'].write_bytes(b'old output')
    with patch.object(sc.subprocess, 'run', side_effect=fake_builder):
        result = sc.build_static(**case, clobber=True)
    assert result['deleted'][0]['bytes'] == 10
    assert case['out'].is_symlink()
    assert case['out'].resolve()==(Path(result['cache'])/'static.nc').resolve()
    assert sc.digest(case['out'])==result['sha256']


def test_failed_build_not_published(case):
    with patch.object(sc.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['builder'])):
        with pytest.raises(subprocess.CalledProcessError): sc.build_static(**case)
    assert not case['out'].exists()
    assert not list(case['cache'].glob('*/complete.json'))


def test_concurrent_members_build_once(case):
    with patch.object(sc.subprocess, 'run', side_effect=fake_builder) as run:
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs=[dict(case,out=case['out'].with_name(f'member{i}.nc'),
                       receipt=case['receipt'].with_name(f'member{i}.json')) for i in range(4)]
            results=list(pool.map(lambda kw:sc.build_static(**kw), jobs))
    assert run.call_count==1 and sum(not r['hit'] for r in results)==1
    assert len({job['out'].resolve() for job in jobs})==1


def test_rebuild_difference_never_promoted(case):
    with patch.object(sc.subprocess, 'run', side_effect=fake_builder): a=sc.build_static(**case)
    def wrong(command, **kwargs):
        fake_builder(command, **kwargs)
        Path(command[command.index('--out')+1]).write_bytes(b'wrong rebuild')
    with patch.object(sc.subprocess, 'run', side_effect=wrong):
        with pytest.raises(RuntimeError, match='not byte-identical'): sc.build_static(**case,verify_rebuild=True)
    assert sc.digest(Path(a['cache'])/'static.nc')==a['sha256']


def test_input_mutation_during_build_not_published(case):
    def mutating(command, **kwargs):
        fake_builder(command, **kwargs)
        case['grid'].write_bytes(b'new mesh during build')
    with patch.object(sc.subprocess,'run',side_effect=mutating):
        with pytest.raises(RuntimeError,match='changed during build'): sc.build_static(**case)
    assert not case['out'].exists()


def test_options_are_part_of_key_and_paths_cannot_be_injected(case):
    with patch.object(sc.subprocess,'run',side_effect=fake_builder):
        a=sc.build_static(**case)
        b=sc.build_static(**case, options=['--valid-time','2022-06-13_21:00:00'])
        assert a['key']!=b['key']
        with pytest.raises(ValueError): sc.build_static(**case,options=['--out','elsewhere'])


def test_default_cache_location_is_the_hex_cache_never_a_host_path(tmp_path, monkeypatch):
    """No explicit cache=: GPUWM_HEX_STATIC_CACHE, else $GPUWM_HEX_CACHE_ROOT/static, else ~/.cache/gpuwm-hex/static.

    THE BREAKAGE THIS PREVENTS: the cache used to default to one rented host's shared-disk path, so the same call on
    any other machine tried to create that folder at the filesystem root and failed with a permission error."""
    assert sc.cache_dir({'GPUWM_HEX_STATIC_CACHE': str(tmp_path / 'a')}) == (tmp_path / 'a').absolute()
    assert sc.cache_dir({'GPUWM_HEX_CACHE_ROOT': str(tmp_path / 'root')}) == tmp_path / 'root' / 'static'
    monkeypatch.setattr(sc.Path, 'home', classmethod(lambda cls: tmp_path / 'home'))
    assert sc.cache_dir({}) == tmp_path / 'home' / '.cache' / 'gpuwm-hex' / 'static'
    with pytest.raises(ValueError, match='turns the static cache off'):
        sc.cache_dir({'GPUWM_HEX_STATIC_CACHE': 'off'})


def test_default_cache_is_used_when_none_is_passed(case, monkeypatch):
    monkeypatch.setenv('GPUWM_HEX_STATIC_CACHE', str(case['cache']))
    del case['cache']
    with patch.object(sc.subprocess, 'run', side_effect=fake_builder):
        result = sc.build_static(**case)
    assert Path(result['cache']).parent == case['out'].parent / 'cache'
    assert case['out'].resolve() == (Path(result['cache']) / 'static.nc').resolve()


def test_a_caller_runner_builds_the_static(case):
    """runner(command, log) replaces the in-process build (the mesh-point door passes its own engine runner)."""
    calls = []

    def runner(command, log):
        calls.append((list(map(str, command)), Path(log)))
        fake_builder(command)

    with patch.object(sc.subprocess, 'run', side_effect=AssertionError('the runner builds, not subprocess.run')):
        first = sc.build_static(**case, runner=runner)
        second = sc.build_static(**case, runner=runner)
    assert len(calls) == 1 and not first['hit'] and second['hit']
    command, log = calls[0]
    assert command[0] == str(case['engine'].resolve()) and '--receipt' in command
    assert log.name == 'build.log'
    assert case['out'].read_bytes() == b'mesh-v1|geography-v1'


def test_a_failing_runner_publishes_nothing(case):
    def runner(command, log):
        raise RuntimeError('builder refused')

    with pytest.raises(RuntimeError, match='builder refused'):
        sc.build_static(**case, runner=runner)
    assert not case['out'].exists()
    assert not list(case['cache'].glob('*/complete.json'))
