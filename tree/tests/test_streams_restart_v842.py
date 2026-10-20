"""v8.4.2 stream membership and real Rust native-file round trips.

These are contract and bridge checks, not the required whole-forecast proof.
"""
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from netCDF4 import Dataset

from hexcore.output import HistoryField
from hexcore.restart_v842 import (RestartControl,read_stream,read_restart,write_stream,write_restart)
from hexcore.streams_v842 import (STREAM_NAMES,StreamAlarms,_REGISTRY,_all_packages,field_definition,
                                 package_active,parse_streams,registry_stream)


def test_v842_output_packages_and_persistent_restart_contract():
    dry=registry_stream('output',physics=False).fields(physics=False)
    wet=registry_stream('output').fields({'mp_wsm6_in','sfclayer','lsm_noah'})
    assert {'qv','u','w','theta'} <= set(dry)
    assert 'rainnc' not in dry and 'qc' not in dry
    assert {'qc','qr','qi','qs','qg'} <= set(wet)
    assert 'ni' not in wet
    assert {'i_rainnc','i_rainc','kpbl','hpbl','cd','cda','ck','cka','pin','ozmixm'} <= set(registry_stream('output').fields(_all_packages()))
    restart=registry_stream('restart').fields(_all_packages())
    assert {'rthratensw','rthratenlw','smois','tslb','rainnc','rainc'} <= set(restart)
    assert 'rho_zz' in restart and 'theta_m' in restart
    assert {'surface','iau','invariant','da_state','diag_ugwp','ugwp_oro_data_in','ugwp_ngw_in'} <= set(STREAM_NAMES)
    assert package_active('one;two',{'two'})
    assert not package_active('one;two',{'three'})


def test_native_deck_files_mutable_members_and_interval_aliases(tmp_path):
    (tmp_path/'fields').write_text('theta\n u\tqv\n')
    path=tmp_path/'streams.atmosphere'
    path.write_text('''<streams>
      <immutable_stream name="input" type="input" filename_template="init.nc" input_interval="initial_only"/>
      <immutable_stream name="restart" type="input;output" filename_template="restart.$h.nc" input_interval="initial_only" output_interval="12:00:00"/>
      <stream name="output" type="output" filename_template="history.$h.nc" output_interval="stream:restart:output_interval">
        <file name="fields"/><var name="w"/><var name="qc" packages="mp_wsm6_in"/>
      </stream></streams>''')
    streams=parse_streams(path)
    assert streams['output'].options['output_interval']=='12:00:00'
    assert streams['output'].fields(streams=streams)==('theta','u','qv','w')
    assert streams['output'].fields({'mp_wsm6_in'},streams=streams)==('theta','u','qv','w','qc')
    assert 'rho_zz' in streams['restart'].fields()
    t=datetime(2001,1,1)
    assert streams['output'].filename(t+timedelta(hours=13),t)=='history.12.nc'
    alarms=StreamAlarms(streams,t,t,t+timedelta(days=1),do_restart=True)
    assert not alarms.due('output',t)
    assert not alarms.due('restart',t)
    assert not alarms.due('output',t+timedelta(hours=1))
    assert alarms.due('output',t+timedelta(hours=12,minutes=1))
    alarms.reset('output',t+timedelta(hours=12,minutes=1))
    assert not alarms.due('output',t+timedelta(hours=13))
    assert alarms.due('output',t+timedelta(days=1))


def test_deck_immutable_members_and_cycles_are_rejected(tmp_path):
    path=tmp_path/'streams.atmosphere'
    path.write_text('<streams><immutable_stream name="restart" type="output" filename_template="r.nc" output_interval="none"><var name="u"/></immutable_stream></streams>')
    with pytest.raises(ValueError,match='membership'):parse_streams(path)
    path.write_text('<streams><stream name="output" type="output" filename_template="r.nc" output_interval="1:00:00"><stream name="output"/></stream></streams>')
    with pytest.raises(ValueError,match='cyclic'):parse_streams(path)


@pytest.fixture
def bridge():
    if not os.environ.get('GPUWM_HEX_RW_MPAS_STREAM'):
        pytest.skip('real rw_mpas_stream artifact not supplied')


def test_staged_bridge_writes_without_environment_or_path(tmp_path, monkeypatch, bridge):
    """Clean G0 consumers find the installed Rust artifact, not a lane override."""
    from hexcore import engines, restart_v842
    from hexcore.output import write_history
    import shutil
    directory=tmp_path/'bridges';directory.mkdir()
    executable=directory/engines.executable_name('rw_mpas_stream')
    shutil.copy2(os.environ['GPUWM_HEX_RW_MPAS_STREAM'],executable)
    monkeypatch.delenv('GPUWM_HEX_RW_MPAS_STREAM')
    monkeypatch.delenv('GPUWM_RW_MPAS_STREAM',raising=False)
    monkeypatch.setenv('PATH','')
    monkeypatch.setattr(engines,'USER_BRIDGE_DIR',directory)
    assert Path(restart_v842._bridge())==executable.resolve()
    words=np.array([0x00000001,0x80000000,0x3f800000],dtype=np.uint32)
    target=tmp_path/'history.nc'
    write_history(target,SimpleNamespace(arrays={},dimensions={'nCells':3}),
        {'t2m':HistoryField(words.view(np.float32),('nCells',))},datetime(2001,1,1))
    with Dataset(target) as dataset:
        np.testing.assert_array_equal(dataset['t2m'][:].data.view(np.uint32).ravel(),words)


def test_explicit_missing_stream_bridge_does_not_fall_through(tmp_path,monkeypatch):
    from hexcore import engines, restart_v842
    monkeypatch.setenv('GPUWM_HEX_RW_MPAS_STREAM',str(tmp_path/'missing'))
    with pytest.raises(engines.EngineRefusal,match='explicit setting'):
        restart_v842._bridge()


def _fixture(stream,packages,physics=True):
    fields={}; dimensions={'Time':1,'StrLen':64,'nCells':5,'nEdges':7,'nVertices':9,
        'nVertLevels':3,'nVertLevelsP1':4,'nVertLevelsP2':5,'nSoilLevels':4,'TWO':2,'maxEdges':6,
        'maxEdges2':12,'vertexDegree':3,'nMonths':12,'nOro':4}
    for name in stream.fields(packages,physics=physics):
        definition=field_definition(name,physics=physics)
        dims=tuple(d for d in definition['dimensions'] if d!='Time')
        for d in dims: dimensions.setdefault(d,2)
        if name in ('initial_time','xtime','Time'):continue
        if definition['type']=='text':fields[name]='TEST'
        else:
            shape=tuple(dimensions[d] for d in dims)
            data=np.arange(np.prod(shape,dtype=int),dtype=np.float32).reshape(shape)
            if definition['type'] in ('integer','logical'):data=data.astype(np.int32)
            elif data.size:
                # Include words that numeric equality would fail to distinguish.
                data.flat[0]=np.float32(-0.0)
                if data.size>1:data.view(np.uint32).flat[1]=0x7fc01234
                if data.size>2:data.view(np.uint32).flat[2]=0x00000001
                if data.size>3:data.view(np.uint32).flat[3]=0x007fffff
                if data.size>4:data.view(np.uint32).flat[4]=0x80000001
            fields[name]=HistoryField(data,dims)
    mesh=SimpleNamespace(arrays={},dimensions=dimensions,attrs={},converted_connectivity=())
    return mesh,fields


@pytest.mark.parametrize('name',STREAM_NAMES)
@pytest.mark.parametrize('physics',[False,True])
def test_all_native_stream_layouts_and_primitive_words(tmp_path,bridge,name,physics):
    if name not in _REGISTRY['streams' if physics else 'dry_streams']:pytest.skip('not compiled without DO_PHYSICS')
    packages=_all_packages()
    stream=registry_stream(name,physics=physics)
    if not stream.fields(packages,physics=physics):pytest.skip('empty stream without DO_PHYSICS')
    # Input-only stream fixture emission exercises its native contract as data.
    stream=replace(stream,options={**stream.options,'type':'output'})
    mesh,fields=_fixture(stream,packages,physics)
    path=tmp_path/f'{name}.nc'
    write_stream(path,stream,mesh,fields,'2001-01-02_00:00:00',initial_time='2001-01-01_00:00:00',active_packages=packages,physics=physics)
    with Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        assert set(ds.variables)==set(stream.fields(packages,physics=physics))
        assert ds.dimensions['Time'].isunlimited() if 'Time' in ds.dimensions else True
        for variable in ds.variables.values():
            definition=field_definition(variable.name,physics=physics)
            assert variable.dimensions==tuple(reversed(definition['dimensions']))
            for attr,value in definition['attrs'].items():
                if variable.name=='Time' and attr=='units':continue
                assert variable.getncattr(attr)==value
            if variable.name in fields and isinstance(fields[variable.name],HistoryField):
                original=fields[variable.name]
                data=variable[0] if 'Time' in variable.dimensions else variable[:]
                order=tuple(reversed(original.dimensions))
                expected=original.values.transpose(tuple(original.dimensions.index(d) for d in order))
                assert np.asarray(data).tobytes()==expected.tobytes(),variable.name
    with read_stream(path,scratch=tmp_path) as native:
        assert set(native.variables)==set(stream.fields(packages,physics=physics))
        if 'xtime' in native.variables:assert native.variables['xtime'].text==('2001-01-02_00:00:00',)


def test_native_restart_read_write_and_persistent_words(tmp_path,bridge):
    packages=_all_packages();stream=registry_stream('restart')
    mesh,fields=_fixture(stream,packages)
    path=tmp_path/'restart.nc';timestamp=tmp_path/'restart_timestamp'
    write_restart(path,mesh,fields,'2001-01-01_12:00:00',initial_time='2001-01-01_00:00:00',active_packages=packages,timestamp_path=timestamp)
    assert timestamp.read_text().strip()=='2001-01-01_12:00:00'
    with read_restart(path,valid_time='2001-01-01_12:00:00',scratch=tmp_path,active_packages=packages) as native:
        assert set(native.variables)==set(stream.fields(packages))
        for name in ('rthratensw','rthratenlw','smois','tslb','rainnc'):
            original=fields[name]
            expected=original.values.transpose(tuple(reversed(range(original.values.ndim))))
            assert native.variables[name].values[0].tobytes()==expected.tobytes()
    with pytest.raises(ValueError,match='no exact_time record'):
        with read_restart(path,valid_time='2001-01-01_13:00:00',scratch=tmp_path,active_packages=packages):pass
    fields.pop('rthratensw')
    with pytest.raises(ValueError,match='rthratensw'):write_restart(tmp_path/'missing.nc',mesh,fields,'2001-01-01_12:00:00',initial_time='2001-01-01_00:00:00',active_packages=packages)
    assert not (tmp_path/'missing.nc').exists()


def test_native_append_connectivity_and_no_clobber(tmp_path,bridge):
    stream=replace(registry_stream('output'),members=(('var','u',''),('var','xtime',''),('var','cellsOnEdge','')))
    mesh=SimpleNamespace(arrays={'cellsOnEdge':np.array([[0,-1],[1,2],[2,3]],dtype=np.int32)},dimensions={'nEdges':3,'TWO':2,'nVertLevels':2},
        variable_dimensions={'cellsOnEdge':('nEdges','TWO')},attrs={},converted_connectivity={'cellsOnEdge'})
    data=np.arange(6,dtype=np.float32).reshape(2,3)
    fields={'u':HistoryField(data,('nVertLevels','nEdges'))}
    path=tmp_path/'history.nc'
    write_stream(path,stream,mesh,fields,'2001-01-01_00:00:00',initial_time='2001-01-01_00:00:00')
    with pytest.raises(FileExistsError):write_stream(path,stream,mesh,fields,'2001-01-01_00:00:00',initial_time='2001-01-01_00:00:00')
    stream=replace(stream,options={**stream.options,'clobber_mode':'append'})
    write_stream(path,stream,mesh,fields,'2001-01-01_06:00:00',initial_time='2001-01-01_00:00:00',record_index=1)
    with Dataset(path) as ds:
        assert ds.variables['u'].shape==(2,3,2)
        assert np.array_equal(ds.variables['cellsOnEdge'][:],mesh.arrays['cellsOnEdge']+1)
    with pytest.raises(ValueError,match='existing record'):write_stream(path,stream,mesh,fields,'2001-01-01_06:00:00',initial_time='2001-01-01_00:00:00',record_index=1)


@pytest.mark.parametrize('restart,da',[(False,False),(False,True),(True,False),(True,True)])
def test_restart_dacycling_control_truth_table(restart,da):
    control=RestartControl.from_namelist('&run\n config_do_restart='+('.true.' if restart else '.false.')+', config_do_DAcycling='+('.true.' if da else '.false.')+'\n/')
    assert control.initial_stream==('restart' if restart else 'input')
    assert control.initialize_coupled_diagnostics==(not restart or da)
    assert ('output' in control.suppressed_initial_outputs)==restart


@pytest.mark.parametrize('format,model',[('netcdf','NETCDF3_64BIT_OFFSET'),('pnetcdf','NETCDF3_64BIT_OFFSET'),('pnetcdf,cdf5','NETCDF3_64BIT_DATA'),('netcdf4','NETCDF4')])
@pytest.mark.parametrize('precision',['single','double'])
def test_native_formats_real_precision_and_typed_attrs(tmp_path,bridge,format,model,precision):
    stream=replace(registry_stream('output'),options={'type':'output','io_type':format,'precision':precision},
        members=(('var','u',''),('var','xtime','')))
    mesh=SimpleNamespace(arrays={},dimensions={'nEdges':3,'nVertLevels':2},attrs={})
    data=np.arange(6,dtype=np.float32).reshape(2,3)
    path=tmp_path/'native.nc'
    write_stream(path,stream,mesh,{'u':HistoryField(data,('nVertLevels','nEdges'),{'test_integer':{'dtype':'i4','values':[3,4]}})},
        '2001-01-01_00:00:00',initial_time='2001-01-01_00:00:00')
    with Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        assert ds.data_model==model
        assert ds.variables['u'].dtype==np.dtype('f8' if precision=='double' else 'f4')
        assert ds.variables['u'][0].tobytes()==data.T.astype(ds.variables['u'].dtype).tobytes()
    with read_stream(path,scratch=tmp_path) as native:
        assert native.variables['u'].attrs['test_integer']=={'dtype':'i4','values':[3,4]}


@pytest.mark.parametrize('format',['netcdf','pnetcdf','pnetcdf,cdf5','netcdf4'])
@pytest.mark.parametrize('precision',['single','double'])
def test_native_stream_subnormals_survive_permutation(tmp_path,bridge,format,precision):
    stream=replace(registry_stream('output'),options={'type':'output','io_type':format,'precision':precision},
                   members=(('var','u',''),))
    if precision=='single':
        bits=np.array([1,0x007fffff,0x80000001,0x807fffff,0,0x80000000],dtype=np.uint32)
        data=bits.view(np.float32).reshape(2,3)
    else:
        bits=np.array([1,0x000fffffffffffff,0x8000000000000001,0x800fffffffffffff,0,0x8000000000000000],dtype=np.uint64)
        data=bits.view(np.float64).reshape(2,3)
    mesh=SimpleNamespace(arrays={},dimensions={'nEdges':3,'nVertLevels':2},attrs={})
    path=tmp_path/'subnormal.nc'
    write_stream(path,stream,mesh,{'u':HistoryField(data,('nVertLevels','nEdges'))},
                 '2001-01-01_00:00:00',initial_time='2001-01-01_00:00:00')
    with read_stream(path,scratch=tmp_path) as native:
        variable=native.variables['u']
        assert variable.dimensions==('Time','nEdges','nVertLevels')
        assert variable.values.dtype==data.dtype
        assert variable.values[0].tobytes()==data.T.tobytes()


def test_append_rejects_changed_extent_and_leaves_original(tmp_path,bridge):
    stream=replace(registry_stream('output'),members=(('var','u',''),('var','xtime','')))
    mesh=SimpleNamespace(arrays={},dimensions={'nEdges':3,'nVertLevels':2},attrs={})
    path=tmp_path/'original.nc'
    fields={'u':HistoryField(np.zeros((2,3),dtype=np.float32),('nVertLevels','nEdges'))}
    write_stream(path,stream,mesh,fields,'2001-01-01_00:00:00',initial_time='2001-01-01_00:00:00')
    original=path.read_bytes()
    stream=replace(stream,options={**stream.options,'clobber_mode':'append'})
    mesh.dimensions['nEdges']=2
    fields['u']=HistoryField(np.zeros((2,2),dtype=np.float32),('nVertLevels','nEdges'))
    with pytest.raises(RuntimeError,match='extent mismatch'):
        write_stream(path,stream,mesh,fields,'2001-01-01_06:00:00',initial_time='2001-01-01_00:00:00',record_index=1)
    assert path.read_bytes()==original
    assert not list(tmp_path.glob('.original.nc.*'))


def test_independently_written_native_restart_input(tmp_path,bridge):
    # Use an independent NetCDF implementation to create a native-layout
    # input, including double-precision real data supported by native streams.
    stream=replace(registry_stream('restart'),members=(('var','u',''),('var','xtime','')))
    path=tmp_path/'native-input.nc'
    words=np.array([-0.0,1.25,2.5,3.75,5.0,6.25],dtype=np.float64).reshape(3,2)
    with Dataset(path,'w',format='NETCDF3_64BIT_OFFSET') as ds:
        for name,size in [('Time',None),('StrLen',64),('nEdges',3),('nVertLevels',2)]:ds.createDimension(name,size)
        ds.createVariable('u','f8',('Time','nEdges','nVertLevels'))[0]=words
        stamp=ds.createVariable('xtime','S1',('Time','StrLen'))
        stamp[0]=np.frombuffer(b'2001-01-01_12:00:00'.ljust(64,b' '),dtype='S1')
    with read_restart(path,valid_time='2001-01-01_12:00:00',scratch=tmp_path,stream=stream) as native:
        assert native.variables['u'].values[0].tobytes()==words.tobytes()
    with read_restart(path,valid_time='2001-01-01_12:00:00',scratch=tmp_path,stream=stream,authority_layout=True) as native:
        assert native.variables['u'].dimensions==('nVertLevels','nEdges')
        assert native.variables['u'].values.tobytes()==words.T.astype(np.float32).tobytes()
    malformed=tmp_path/'wrong-primitive.nc'
    with Dataset(malformed,'w',format='NETCDF3_64BIT_OFFSET') as ds:
        for name,size in [('Time',None),('StrLen',64),('nEdges',3),('nVertLevels',2)]:ds.createDimension(name,size)
        ds.createVariable('u','i4',('Time','nEdges','nVertLevels'))[0]=np.zeros((3,2),dtype=np.int32)
        ds.createVariable('xtime','S1',('Time','StrLen'))[0]=np.frombuffer(b'2001-01-01_12:00:00'.ljust(64,b' '),dtype='S1')
    with pytest.raises(ValueError,match='primitive'):
        with read_restart(malformed,valid_time='2001-01-01_12:00:00',scratch=tmp_path,stream=stream):pass


@pytest.mark.parametrize('text,seconds',[('86400',86400),('90:00',5400),('00:90:00',5400),('0_0:0:1.25',1.25)])
def test_native_fixed_interval_spellings(text,seconds):
    from hexcore.streams_v842 import interval_seconds
    assert interval_seconds(text)==seconds


def test_native_restart_connectivity_restored_in_rust(tmp_path,bridge):
    stream=replace(registry_stream('restart'),members=(('var','xtime',''),('var','cellsOnEdge','')))
    raw=np.array([[0,-1],[1,2],[2,3]],dtype=np.int32)
    mesh=SimpleNamespace(arrays={'cellsOnEdge':raw},dimensions={'nEdges':3,'TWO':2},
        variable_dimensions={'cellsOnEdge':('nEdges','TWO')},attrs={},converted_connectivity={'cellsOnEdge'})
    path=tmp_path/'restart.nc'
    write_stream(path,stream,mesh,{},'2001-01-01_12:00:00',initial_time='2001-01-01_00:00:00')
    with read_restart(path,valid_time='2001-01-01_12:00:00',scratch=tmp_path,stream=stream,
                      authority_layout=True,zero_based_connectivity={'cellsOnEdge'}) as native:
        assert native.variables['cellsOnEdge'].dimensions==('TWO','nEdges')
        assert native.variables['cellsOnEdge'].values.tobytes()==raw.T.tobytes()
    with pytest.raises(ValueError,match='authority_layout'):
        with read_restart(path,valid_time='2001-01-01_12:00:00',scratch=tmp_path,stream=stream,
                          zero_based_connectivity={'cellsOnEdge'}):pass


def test_replaced_legacy_snapshot_refreshes_durability_receipt(tmp_path,bridge):
    import json
    from hexcore.output import write_history
    from hexcore.history_writer import durable_marker
    mesh=SimpleNamespace(arrays={},dimensions={'nEdges':3,'nVertLevels':2},attrs={})
    path=tmp_path/'history.nc'
    fields={'u':HistoryField(np.ones((2,3),dtype=np.float32),('nVertLevels','nEdges'))}
    write_history(path,mesh,fields,'2001-01-01_00:00:00',include_mesh=False)
    old=json.loads(durable_marker(path).read_text())
    fields['u']=HistoryField(np.full((2,3),2,dtype=np.float32),('nVertLevels','nEdges'))
    write_history(path,mesh,fields,'2001-01-01_01:00:00',include_mesh=False,stream_options={'clobber_mode':'replace_files'})
    receipt=json.loads(durable_marker(path).read_text())
    assert receipt['mtime_ns']==path.stat().st_mtime_ns and receipt['bytes']==path.stat().st_size
    assert receipt['mtime_ns']!=old['mtime_ns']
    with Dataset(path) as ds:assert np.all(ds.variables['u'][:]==2)


def test_native_field_file_accepts_var_array_tokens(tmp_path):
    (tmp_path/'fields').write_text('scalars\n theta\n')
    path=tmp_path/'streams.atmosphere'
    path.write_text('<streams><stream name="output" type="output" filename_template="out.nc" output_interval="86400"><file name="fields"/></stream></streams>')
    deck=parse_streams(path)
    fields=deck['output'].fields({'mp_wsm6_in'},streams=deck)
    assert fields==('qv','qc','qr','qi','qs','qg','theta')
    assert deck['output'].fields(streams=deck)==('qv','theta')


def test_closed_native_reader_descriptor_is_safe_to_represent(tmp_path,bridge):
    native=registry_stream('surface')
    stream=replace(native,options={**native.options,'type':'input;output'})
    mesh,fields=_fixture(stream,())
    path=tmp_path/'surface.nc'
    write_stream(path,stream,mesh,fields,'2001-01-01_00:00:00',initial_time='2001-01-01_00:00:00')
    with read_stream(path,scratch=tmp_path) as native:
        variable=native.variables['sst']
    # Pytest and normal traceback renderers represent locals after unwinding.
    # Representing memmap values here used to dereference unmapped memory.
    assert 'dimensions=' in repr(variable)
    assert 'values=' not in repr(variable)
    assert 'sst' in repr(native)


@pytest.mark.parametrize('defect',['none','attributes','geometry','membership','missing-file'])
def test_whole_forecast_header_identity_uses_rust(tmp_path,bridge,defect):
    from replay_hex_restart_forecast_v842 import compare_headers
    from hexcore.restart_v842 import inspect_stream
    native=registry_stream('surface')
    native=replace(native,options={**native.options,'type':'input;output'})
    mesh,fields=_fixture(native,())
    strict=tmp_path/'strict.nc';candidate=tmp_path/'candidate.nc'
    write_stream(strict,native,mesh,fields,'2001-01-01_00:00:00',
                 initial_time='2001-01-01_00:00:00',global_attrs={'sphere_radius':{'dtype':'f8','values':[6371220.0]}})
    if defect!='missing-file':
        if defect=='membership':
            selected=replace(native,members=(('var','sst',''),))
        else:selected=native
        attrs={'sst':{'units':'incorrect'}} if defect=='attributes' else None
        radius=1.0 if defect=='geometry' else 6371220.0
        write_stream(candidate,selected,mesh,fields,'2001-01-01_00:00:00',
            initial_time='2001-01-01_00:00:00',field_attrs=attrs,
            global_attrs={'sphere_radius':{'dtype':'f8','values':[radius]}})
    grade=compare_headers(strict,candidate,scratch=tmp_path/'metadata')
    assert grade['passed']==(defect=='none')
    header=inspect_stream(strict,scratch=tmp_path/'metadata')
    with read_stream(strict,scratch=tmp_path/'decode') as decoded:
        assert header['dimensions']==decoded.dimensions
        assert header['global_attrs']==decoded.global_attrs
        assert set(header['variables'])==set(decoded.variables)
        for name,var in decoded.variables.items():
            expected=dict(dtype=var.values.dtype.str.lstrip('<>='),
                dimensions=list(var.dimensions),shape=list(var.values.shape),attrs=var.attrs)
            assert {k:header['variables'][name][k] for k in expected}==expected
            assert set(header['variables'][name]['attribute_order'])==set(var.attrs)
            assert set(header['variables'][name]['attribute_words'])==set(var.attrs)
    assert not list((tmp_path/'metadata').iterdir())


@pytest.mark.parametrize('wrong_units',[False,True])
def test_whole_series_cannot_pass_zero_words_with_wrong_attributes(tmp_path,bridge,wrong_units):
    import json
    from hashlib import sha256
    from replay_hex_restart_forecast_v842 import compare_series
    comparator=os.environ.get('MPAS_STREAM_WORD_COMPARATOR')
    if not comparator:
        pytest.skip('actual Rust word comparator not supplied')
    stream=registry_stream('surface')
    stream=replace(stream,options={**stream.options,'type':'input;output'})
    mesh,fields=_fixture(stream,())
    source=tmp_path/'source';candidate=tmp_path/'candidate'
    source.mkdir();candidate.mkdir()
    stamp=datetime(2001,1,1)
    name='surface.2001-01-01_00.00.00.nc'
    for directory in (source,candidate):
        attrs={'sst':{'units':'wrong'}} if directory==candidate and wrong_units else None
        write_stream(directory/name,stream,mesh,fields,stamp,initial_time=stamp,field_attrs=attrs)
    (source/'PUBLICATION.json').write_text(json.dumps({'files':{
        name:sha256((source/name).read_bytes()).hexdigest()}}))
    (source/'streams.atmosphere').write_text('<streams><stream name="surface" filename_template="surface.$Y-$M-$D_$h.$m.$s.nc"/></streams>')
    receipts=tmp_path/'receipts';receipts.mkdir()
    args=SimpleNamespace(reference=source,comparator=Path(comparator),receipts=receipts,scratch=tmp_path/'scratch')
    result=compare_series(args,{'expected_samples':{'surface':[0]}},'series',candidate,stamp,stamp,stamp,1200)
    assert result['differing_words']==0
    assert result['incompatible_samples']==0
    assert result['metadata_incompatible_files']==int(wrong_units)
    assert result['passed']==(not wrong_units)
    assert (receipts/'series-metadata.json').is_file()


@pytest.mark.parametrize('mode', ['never_modify', 'replace_files'])
@pytest.mark.parametrize('cross_device', [False, True])
def test_stream_publication_is_atomic_without_same_device_copy(tmp_path, monkeypatch, mode, cross_device):
    import errno
    from hexcore import restart_v842
    staged = tmp_path/'native.nc'; staged.write_bytes(b'complete frame')
    target = tmp_path/'output.nc'
    source_inode = staged.stat().st_ino
    if mode == 'replace_files':
        target.write_bytes(b'old frame')
    operation = 'link' if mode == 'never_modify' else 'replace'
    real_operation = getattr(os, operation)
    def publish(source, destination):
        if cross_device and Path(source) == staged:
            raise OSError(errno.EXDEV, 'cross-device link')
        return real_operation(source, destination)
    monkeypatch.setattr(os, operation, publish)
    if not cross_device:
        def unexpected_copy(*args):
            raise AssertionError('same-filesystem publication must not copy the frame')
        monkeypatch.setattr(restart_v842.shutil, 'copyfileobj', unexpected_copy)
    restart_v842._publish_stream(staged, target, mode)
    assert target.read_bytes() == b'complete frame'
    assert (target.stat().st_ino == source_inode) is not cross_device
    assert not list(tmp_path.glob('.output.nc.*'))


def test_stream_publication_copy_failure_preserves_previous_file(tmp_path, monkeypatch):
    import errno
    from hexcore import restart_v842
    staged = tmp_path/'native.nc'; staged.write_bytes(b'complete frame')
    target = tmp_path/'output.nc'; target.write_bytes(b'old frame')
    def cross_device(*args):
        raise OSError(errno.EXDEV, 'cross-device link')
    def disk_full(source, destination):
        destination.write(b'partial')
        raise OSError(errno.ENOSPC, 'disk full')
    monkeypatch.setattr(os, 'replace', cross_device)
    monkeypatch.setattr(restart_v842.shutil, 'copyfileobj', disk_full)
    with pytest.raises(OSError, match='disk full'):
        restart_v842._publish_stream(staged, target, 'replace_files')
    assert target.read_bytes() == b'old frame'
    assert staged.read_bytes() == b'complete frame'
    assert not list(tmp_path.glob('.output.nc.*'))


def test_stream_publication_never_modify_preserves_existing_destination(tmp_path):
    from hexcore import restart_v842
    staged = tmp_path/'native.nc'; staged.write_bytes(b'new frame')
    target = tmp_path/'output.nc'; target.write_bytes(b'old frame')
    with pytest.raises(FileExistsError):
        restart_v842._publish_stream(staged, target, 'never_modify')
    assert target.read_bytes() == b'old frame'
    assert staged.read_bytes() == b'new frame'
