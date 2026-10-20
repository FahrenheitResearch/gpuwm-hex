"""Complete stream/restart boundary on actual strict native pools, process 1.

Fixture values only: these checks do not prove a computed forecast. No native
field/pool calls are mocked. MPAS Registry.xml:650-925,1360-4020.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hexcore.mpasphys import resolve_namelist
from hexcore.mpasphys.native_backend import NativePhysics
from hexcore.mpasphys.pool_v842 import RegistryLayout
from hexcore.restart_v842 import NativePoolStreams, read_stream, write_restart


@pytest.mark.parametrize('partial_authority',[False,True])
@pytest.mark.parametrize('suite',['none','mesoscale_reference','convection_permitting'])
@pytest.mark.parametrize('stop',['none','2000-03-01_12:00:00'])
@pytest.mark.parametrize('ugwp',[False,True])
def test_whole_native_pool_streams_and_restart(suite,stop,ugwp,partial_authority,tmp_path):
    library=os.environ.get('MPAS_NATIVE_LIBRARY')
    if not library:
        pytest.skip('actual strict native library not supplied')
    source=Path(os.environ['MPAS_SOURCE'])
    deck=Path(os.environ['MPAS_NATIVE_POOL_STREAM_DECK'])
    assets=Path(os.environ['MPAS_NATIVE_STREAM_DEFAULTS'])
    text=f'''&physics
config_physics_suite='{suite}'
config_gwdo_scheme='{'bl_ugwp_gwdo' if ugwp else 'suite'}'
config_ugwp_diags={'.true.' if ugwp else '.false.'}
/
&nhyd_model
config_start_time='2000-02-29_12:00:00'
config_stop_time='{stop}'
config_run_duration='1_00:00:00'
/
'''
    layout=RegistryLayout(source,resolve_namelist(text),text,
            dict(nCells=2,nEdges=3,nVertices=2,maxEdges=6,maxEdges2=12,
                 vertexDegree=3,nVertLevels=4))
    # Use the real campaign deck, not a rewritten field list. All sizes and
    # component aliases come from the pristine Registry allocation contract.
    with NativePhysics(library,test_only=True) as native:
        for ordinal,row in enumerate(layout.fields):
            for level in range(1,row['nlevels']+1):
                if row['kind']=='character':
                    native.put_text_field(row['pool'],row['pool_name'],
                        layout.configs['config_start_time'],level=level,nlevels=row['nlevels'])
                else:
                    size=int(np.prod(row['shape']))
                    words=np.arange(size,dtype=np.uint32)+ordinal+1
                    data=(words.view(np.float32) if row['kind']=='real'
                          else words.view(np.int32)).reshape(row['shape'])
                    native.put_field(row['pool'],row['pool_name'],data,level=level,
                                     nlevels=row['nlevels'])
        unavailable={'output':['pressure']} if partial_authority else {}
        writer=NativePoolStreams(native,layout,deck,tmp_path/'out',list_directory=assets,
                                 unavailable_fields=unavailable)
        receipt={}
        writer.write(step=0,dt=1200,receipt=receipt)
        writer.write(step=3,dt=1200,receipt=receipt)
        assert not any(r['name']=='restart' and r['step']==0 for r in receipt['streams'])
        expected={name for name,s in writer.streams.items()
                  if 'output' in s.options['type'].split(';') and
                  s.fields(layout.packages,streams=writer.streams)}
        assert {r['name'] for r in receipt['streams']}==expected
        assert ('diag_ugwp' in expected)==ugwp
        captured={}
        for record in receipt['streams']:
            stream=writer.streams[record['name']]
            fields=writer.fields(stream)
            with read_stream(record['path'],scratch=tmp_path/'read') as actual:
                expected_fields=set(stream.fields(layout.packages,streams=writer.streams))
                expected_fields-=set(unavailable.get(stream.name,()))
                assert set(actual.variables)==expected_fields
                assert writer.streams[stream.name]==stream  # Native contract remains complete.
                assert receipt['missing_fields'].get('output',[])==(['pressure'] if partial_authority else [])
                for name,field in fields.items():
                    if isinstance(field,str):
                        assert actual.variables[name].text==(field,)
                        continue
                    variable=actual.variables[name]
                    raw=variable.values
                    if 'Time' in variable.dimensions:
                        raw=raw[0]
                    axes=tuple(reversed(field.dimensions))
                    reference=field.values.transpose(tuple(field.dimensions.index(d) for d in axes))
                    assert raw.shape==reference.shape
                    assert raw.tobytes()==reference.tobytes(),name
            if record['name']=='restart':
                captured={name:field.values.copy() for name,field in fields.items()
                          if not isinstance(field,str)}
                restart=record['path']
        # Mutate actual native buffers, then restore every persistent field and
        # every constituent through Rust decode into both allocated time levels.
        for row in layout.fields:
            if row['kind']=='character':
                continue
            buffer=native.get_field(row['pool'],row['pool_name'])
            buffer[...] = 0
            for level in range(1,row['nlevels']+1):
                native.put_field(row['pool'],row['pool_name'],buffer,level=level,
                                 nlevels=row['nlevels'])
        layout.configs['config_do_restart']=True
        layout.configs['config_start_time']='2000-02-29_13:00:00'
        reader=NativePoolStreams(native,layout,deck,tmp_path/'resumed',list_directory=assets)
        restored=reader.restore(restart,scratch=tmp_path/'restore')
        assert set(restored)==set(reader.streams['restart'].fields(layout.packages,streams=reader.streams))
        assert reader.initial==writer.initial
        for name,field in reader.fields(reader.streams['restart']).items():
            if name in captured:
                assert field.values.tobytes()==captured[name].tobytes(),name
                row,component=reader.rows[name]
                for level in range(1,row['nlevels']+1):
                    buffer=native.get_field(row['pool'],row['pool_name'],level=level)
                    if component is not None:buffer=buffer[component]
                    value=buffer[tuple(slice(0,n) for n in captured[name].shape)]
                    assert value.tobytes()==captured[name].tobytes(),(name,level)
        resumed_receipt={}
        reader.write(step=0,dt=1200,receipt=resumed_receipt)
        assert not {'output','diagnostics','restart'} & {
            r['name'] for r in resumed_receipt.get('streams',[])}
        target=os.environ.get('MPAS_NATIVE_POOL_STREAM_RECEIPT')
        if target:
            with Path(target).open('a') as log:
                log.write(json.dumps(dict(suite=suite,process_count=1,serial_transport_count=1,
                    stop_time_option=stop,
                    ugwp_diagnostics=ugwp,
                    declared_unavailable_fields=unavailable,
                    stream_names=sorted(expected),files=len(receipt['streams']),
                    restart_startup_streams=sorted(r['name'] for r in resumed_receipt.get('streams',[])),
                    persistent_fields=len(restored),different_words=0,
                    scope='Actual strict native pool fixture transport; no computed forecast claim'))+'\n')


@pytest.mark.parametrize('explicit_attrs',[None,False,True])
def test_published_restart_through_actual_native_pool(explicit_attrs,tmp_path):
    """MPAS mpas_atm_core.F:129-150,894-902: real persistent authority."""
    root=os.environ.get('MPAS_NATIVE_POOL_REAL_PUBLICATION')
    library=os.environ.get('MPAS_NATIVE_LIBRARY')
    if not root or not library:
        pytest.skip('actual published restart and strict native library not supplied')
    import hashlib
    root=Path(root)
    publication=json.loads((root/'PUBLICATION.json').read_text())
    manifest=json.loads((root/'MANIFEST.json').read_text())
    assert manifest['ranks']==1
    candidates=sorted(root.glob('restart.*.nc'))
    assert candidates
    path=candidates[len(candidates)//2-1]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==publication['files'][path.name]
    text=(root/'namelist.atmosphere').read_text()
    assert hashlib.sha256(text.encode()).hexdigest()==publication['files']['namelist.atmosphere']
    with read_stream(path,scratch=tmp_path/'metadata',select=('xtime',)) as metadata:
        valid=metadata.variables['xtime'].text[0]
        dimensions=dict(metadata.dimensions)
        attrs=dict(metadata.global_attrs)
    layout=RegistryLayout(Path(os.environ['MPAS_SOURCE']),resolve_namelist(text),text,dimensions)
    layout.configs.update(config_start_time=valid,config_do_restart=True)
    with NativePhysics(library,test_only=True) as native:
        for row in layout.fields:
            for level in range(1,row['nlevels']+1):
                if row['kind']=='character':
                    native.put_text_field(row['pool'],row['pool_name'],'',level=level,nlevels=row['nlevels'])
                else:
                    native.put_field(row['pool'],row['pool_name'],
                        np.zeros(row['shape'],dtype=np.float32 if row['kind']=='real' else np.int32),
                        level=level,nlevels=row['nlevels'])
        writer=NativePoolStreams(native,layout,root/'streams.atmosphere',tmp_path/'out',
            list_directory=Path(os.environ['MPAS_NATIVE_STREAM_DEFAULTS']),
            global_attrs=attrs if explicit_attrs else None)
        if explicit_attrs is not None:
            writer.read_input_attributes(path,scratch=tmp_path/'attributes')
        restored=writer.restore(path,scratch=tmp_path/'restore')
        assert writer.attrs==attrs
        stream=writer.streams['restart']
        output=tmp_path/'pool-restart.nc'
        write_restart(output,writer.mesh,writer.fields(stream),valid,initial_time=writer.initial,
                      stream=stream,streams=writer.streams,active_packages=layout.packages,
                      global_attrs=writer.attrs,field_attrs=writer.field_attrs,scratch=tmp_path/'write')
        from replay_hex_restart_forecast_v842 import compare_headers
        header_grade=compare_headers(path,output,scratch=tmp_path/'header-audit')
        assert header_grade['passed'],header_grade
        assert header_grade['field_count']==len(restored)
        words=0
        with read_stream(path,scratch=tmp_path/'reference') as reference:
            with read_stream(output,scratch=tmp_path/'candidate') as candidate:
                assert reference.dimensions==candidate.dimensions
                assert reference.global_attrs==candidate.global_attrs
                assert set(reference.variables)==set(candidate.variables)
                for name,field in reference.variables.items():
                    actual=candidate.variables[name]
                    assert field.dimensions==actual.dimensions,name
                    assert field.attrs==actual.attrs,name
                    assert field.values.dtype==actual.values.dtype,name
                    assert field.values.tobytes()==actual.values.tobytes(),name
                    words+=field.values.size
        target=os.environ.get('MPAS_NATIVE_POOL_STREAM_RECEIPT')
        if target:
            with Path(target).open('a') as log:
                log.write(json.dumps(dict(native_process_count=1,serial_transport_count=1,
                    explicit_global_attributes=explicit_attrs,
                    attribute_preread=explicit_attrs is not None,
                    source=path.name,source_sha256=publication['files'][path.name],
                    persistent_fields=len(restored),primitive_elements=words,different_words=0,
                    header_audit=header_grade,
                    scope='Published native forecast restart restored into actual native pool and reencoded; no hex forecast claim'))+'\n')
