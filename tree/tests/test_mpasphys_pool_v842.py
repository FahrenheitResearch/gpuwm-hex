"""Registry and real native ABI transport; not whole-forecast receipts."""
import site_support
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.mpasphys import Backend, NativePoolArguments, SlotCall, StepFlags, resolve_namelist
from hexcore.mpasphys.pool_v842 import NativePoolHydration, RegistryLayout

SOURCE = site_support.mpas_source(preferred=('MPAS_SOURCE',))

@pytest.mark.parametrize('scheme', ['off', 'bl_ysu_gwdo', 'bl_ugwp_gwdo'])
@pytest.mark.parametrize('diagnostics', [False, True])
def test_registry_gravity_wave_diagnostics_have_independent_allocation(scheme, diagnostics):
    """Core interface.F:224-253 does not gate diagnostics on a drag scheme."""
    if not SOURCE.exists():
        pytest.skip('pinned native Registry is required')
    text = (f'&physics\nconfig_physics_suite="none"\n'
            f'config_gwdo_scheme="{scheme}"\n'
            f'config_ugwp_diags={str(diagnostics).lower()}\n/')
    layout = RegistryLayout(SOURCE, resolve_namelist(text), text,
                            dict(nCells=2, nEdges=3, nVertices=2, maxEdges=6,
                                 maxEdges2=12, vertexDegree=3, nVertLevels=4))
    assert ('ugwp_diags_stream' in layout.packages) == diagnostics
    assert ('ugwp_orog_stream' in layout.packages) == (scheme == 'bl_ugwp_gwdo')
    fields = {row['name']: row for row in layout.fields}
    assert ('dusfc_ls' in fields) == diagnostics
    assert ('dtaux3d_ls' in fields) == diagnostics
    if diagnostics:
        assert fields['dusfc_ls']['shape'] == [3]
        assert fields['dtaux3d_ls']['shape'] == [4, 3]

@pytest.mark.parametrize('name', ['config_iau_option', 'CONFIG_IAU_OPTION', 'config_IAU_option'])
@pytest.mark.parametrize('value', ['off', 'on', 'constant', 'ramp'])
def test_registry_case_insensitive_iau_package_selection(name, value):
    if not SOURCE.exists():
        pytest.skip('pinned native Registry is required')
    text = f'&physics\nconfig_physics_suite="none"\n/\n&IAU\n{name}="{value}"\n/'
    layout = RegistryLayout(SOURCE, resolve_namelist(text), text,
                            dict(nCells=2,nEdges=3,nVertices=2,maxEdges=6,maxEdges2=12,
                                 vertexDegree=3,nVertLevels=4))
    assert layout.configs['config_IAU_option'] == value
    assert ('iau' in layout.packages) == (value != 'off')
    assert any(r['pool'] == 'tend_iau' for r in layout.fields) == (value != 'off')
    assert 'config_iau_option' not in layout.configs

@pytest.mark.parametrize('suite,scheme,count', [
    ('none','off',1), ('mesoscale_reference','mp_wsm6',6),
    ('convection_permitting','mp_thompson',8),
    ('none','mp_thompson_aerosols',11), ('none','mp_kessler',3),
])
def test_registry_carriers_and_pool_aliases(suite,scheme,count):
    if not SOURCE.exists():
        pytest.skip('pinned native Registry is required')
    text = f'&physics\nconfig_physics_suite="{suite}"\nconfig_microp_scheme="{scheme}"\n/'
    layout = RegistryLayout(SOURCE, resolve_namelist(text), text,
                            dict(nCells=2,nEdges=3,nVertices=2,maxEdges=6,maxEdges2=12,
                                 vertexDegree=3,nVertLevels=4))
    state = next(r for r in layout.fields if r['name']=='scalars')
    assert state['shape'] == [count,4,3]
    assert state['nlevels'] == 2
    assert state['components'][:1] == ['qv']
    assert layout.indices['state','index_qv'] == 1
    assert layout.indices['state','moist_start'] == 1
    assert layout.indices['state','moist_end'] == min(count,6)
    assert layout.indices['state','index_nc'] == (9 if scheme=='mp_thompson_aerosols' else -1)
    water = next(r for r in layout.fields if r['name']=='iswater_lu')
    assert water['pool_name']=='iswater'
    assert water['initial']=='16'
    assert water['shape']==[]
    assert all('CAM' not in r['name'] for r in layout.fields)


def transport(tmp_path):
    data = np.asarray([[1.,-0.,3.],[4.,5.,6.]],dtype='<f4',order='F')
    payload = data.tobytes(order='F')
    (tmp_path/'field.bin').write_bytes(payload)
    row = dict(pool='state',name='rho_zz',pool_name='rho_zz',kind='real',
               dimensions=['nVertLevels','nCells'],shape=[2,3],nlevels=2)
    record = {k:row[k] for k in ('pool','name','kind','dimensions','shape')}
    record.update(payload='field.bin',sha256=hashlib.sha256(payload).hexdigest())
    (tmp_path/'manifest.json').write_text(json.dumps(dict(schema='mpas-native-pool-payload-v1',fields=[record])))
    layout=SimpleNamespace(fields=[row],configs={},dimensions=dict(nVertLevels=2,nCells=2),indices={})
    return layout,data


def test_transport_digest_and_manifest_are_checked_before_mutation(tmp_path):
    layout,_ = transport(tmp_path)
    NativePoolHydration(None,layout,tmp_path)
    (tmp_path/'field.bin').write_bytes(b'\0'*24)
    with pytest.raises(ValueError,match='digest differs'):
        NativePoolHydration(None,layout,tmp_path)


@pytest.mark.parametrize('workers', ['1', '8'])
def test_parallel_payload_validation_preserves_words_and_first_refusal(tmp_path, monkeypatch, workers):
    monkeypatch.setenv('GPUWM_HEX_HOST_COPY_THREADS', workers)
    layout, data = transport(tmp_path)
    manifest = json.loads((tmp_path/'manifest.json').read_text())
    row, record = layout.fields[0], manifest['fields'][0]
    layout.fields = [dict(row, name=f'field_{i}') for i in range(16)]
    manifest['fields'] = [dict(record, name=f'field_{i}') for i in range(16)]
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    hydrated = NativePoolHydration(None, layout, tmp_path)
    for row in layout.fields:
        assert hydrated.field_buffer('state', row['name']).tobytes(order='F') == data.tobytes(order='F')
    # An earlier digest failure must win over a later, faster metadata failure.
    manifest['fields'][0]['sha256'] = '0' * 64
    manifest['fields'][1]['kind'] = 'integer'
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest differs at \\('state', 'field_0'\\)"):
        NativePoolHydration(None, layout, tmp_path)
    # A valid digest with a wrong byte count remains a distinct refusal.
    manifest['fields'][0]['sha256'] = record['sha256']
    manifest['fields'][0]['shape'] = layout.fields[0]['shape'] = [3, 3]
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='byte count differs'):
        NativePoolHydration(None, layout, tmp_path)


@pytest.mark.parametrize('configured,nominal,expected', [
    (0., np.float32(480000.03125), np.float32(480000.03125)),
    (-1., np.float32(240000.015625), np.float32(240000.015625)),
    (12345.03125, np.float32(480000.), np.float32(12345.03125)),
])
def test_core_horizontal_length_copies_native_selected_word(tmp_path, configured, nominal, expected):
    row = dict(pool='mesh', name='nominalMinDc', pool_name='nominalMinDc',
               kind='real', dimensions=[], shape=[], nlevels=1)
    payload = nominal.tobytes()
    (tmp_path/'nominal.bin').write_bytes(payload)
    record = {k:row[k] for k in ('pool','name','kind','dimensions','shape')}
    record.update(payload='nominal.bin', sha256=hashlib.sha256(payload).hexdigest())
    (tmp_path/'manifest.json').write_text(json.dumps(dict(
        schema='mpas-native-pool-payload-v1', fields=[record])))
    layout = SimpleNamespace(fields=[row], configs={'config_len_disp':configured})
    hydration = NativePoolHydration(None, layout, tmp_path)
    assert hydration.horizontal_length().view(np.uint32) == expected.view(np.uint32)
    assert np.float32(layout.configs['config_len_disp']).view(np.uint32) == expected.view(np.uint32)


def test_actual_native_time_levels_and_rk_arguments(tmp_path):
    library=os.environ.get('MPAS_NATIVE_LIBRARY')
    if not library:
        pytest.skip('production native library is required')
    from hexcore.mpasphys.native_backend import NativePhysics
    layout,data=transport(tmp_path)
    with NativePhysics(library,test_only=True) as native:
        hydration=NativePoolHydration(native,layout,tmp_path)
        hydration.populate()
        assert np.array_equal(native.get_field('state','rho_zz',level=1).view(np.uint32),data.view(np.uint32))
        assert np.array_equal(native.get_field('state','rho_zz',level=2).view(np.uint32),data.view(np.uint32))
        for pool in ('mesh','diag_physics','sfc_input','atm_input','configs'):
            native.pool(pool)
        native.block()
        config=resolve_namelist('&physics\nconfig_physics_suite="none"\n/')
        outputs={key:np.zeros((2,3),dtype=np.float32) for key in
                 ('tend_ru_physics','tend_rtheta_physics','tend_rho_physics')}
        provider=NativePoolArguments(native,pools={**outputs,**{p:p for p in
                    ('mesh','state','diag','tend','tend_physics','configs')}},
                    its=1,ite=2,synchronize=lambda entry,call:None)
        call=SlotCall('driver','driver',Backend.NATIVE_CPU,1,0,360,StepFlags(),
                      dict(pools={},rk_step=1,dynamics_substep=1),config.execution_options)
        args=provider('physics_get_tend',call)
        assert args['block']==native.framework_object('block')
        assert (args['rk_step'],args['dynamics_substep'])==(1,1)
        assert args['tend_rtheta_physics'] is outputs['tend_rtheta_physics']


@pytest.mark.parametrize('kind,shape', [('real', [2, 3]), ('real', []),
                                      ('integer', [2, 3]), ('real', [0, 3])])
def test_mapped_hydration_owns_each_native_time_level(tmp_path, kind, shape):
    library = os.environ.get('MPAS_NATIVE_LIBRARY')
    if not library:
        pytest.skip('production native library is required')
    from hexcore.mpasphys.native_backend import NativePhysics
    words = np.resize(np.asarray([0x80000000, 0x7f812345, 0x7fc12345,
                                  1, 0xff800000, 0x3f800000], dtype='<u4'),
                      int(np.prod(shape)))
    data = words.view('<f4' if kind == 'real' else '<i4').reshape(shape, order='F')
    row = dict(pool='state', name='mapped_input', pool_name='mapped_input',
               kind=kind, dimensions=['axis' + str(i) for i in range(len(shape))],
               shape=shape, nlevels=2)
    payload = data.tobytes(order='F')
    path = tmp_path / 'input.bin'
    path.write_bytes(payload)
    record = {k: row[k] for k in ('pool', 'name', 'kind', 'dimensions', 'shape')}
    record.update(payload=path.name, sha256=hashlib.sha256(payload).hexdigest())
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(
        schema='mpas-native-pool-payload-v1', fields=[record])))
    layout = SimpleNamespace(fields=[row], configs={}, dimensions={}, indices={})
    with NativePhysics(library, test_only=True) as native:
        NativePoolHydration(native, layout, tmp_path).populate()
        # In-place overwrite detects accidental aliasing of mapped input;
        # overwriting one native time level must not alter its sibling.
        path.write_bytes(bytes(len(payload)))
        for level in (1, 2):
            assert native.get_field('state', row['name'], level=level).tobytes(order='F') == payload
        native.put_field('state', row['name'], np.zeros_like(data), level=1, nlevels=2)
        assert native.get_field('state', row['name'], level=2).tobytes(order='F') == payload


def test_native_namelist_real_integer_spelling_keeps_registry_kind():
    if not SOURCE.exists():pytest.skip('pinned native Registry is required')
    text='&physics\nconfig_physics_suite="mesoscale_reference"\nconfig_bucket_rainnc=10\n/\n&nhyd_model\nconfig_dt=1200\n/'
    config=resolve_namelist(text)
    assert type(config.execution_options['config_bucket_rainnc']) is float
    layout=RegistryLayout(SOURCE,config,text,dict(nCells=2,nEdges=3,nVertices=2,
                    maxEdges=6,maxEdges2=12,vertexDegree=3,nVertLevels=4))
    assert type(layout.configs['config_dt']) is float
    assert type(layout.configs['config_bucket_rainnc']) is float
    assert type(layout.configs['config_number_of_sub_steps']) is int


def test_hydration_preserves_character_time_levels(tmp_path):
    row=dict(pool='state',name='xtime',pool_name='xtime',kind='character',dimensions=[],shape=[],nlevels=2)
    record={k:row[k] for k in ('pool','name','kind','dimensions','shape')};record['text']='example timestamp'
    (tmp_path/'manifest.json').write_text(json.dumps(dict(schema='mpas-native-pool-payload-v1',fields=[record])))
    layout=SimpleNamespace(fields=[row],configs={},dimensions={},indices={})
    seen=[];native=SimpleNamespace(pool=lambda name:None,put_text_field=lambda *args,**kw:seen.append((args,kw)))
    NativePoolHydration(native,layout,tmp_path).populate()
    assert seen==[(('state','xtime','example timestamp'),dict(level=1,nlevels=2)),
                  (('state','xtime','example timestamp'),dict(level=2,nlevels=2))]


def test_selected_rrfs_layout_retains_empty_native_halo_rings():
    import xml.etree.ElementTree as ET
    from hexcore.mpasphys.rrfs.driver_rrfs import prepare_native_layout
    layout = SimpleNamespace(root=ET.fromstring('<registry/>'),
        dimensions={name: 4 for name in ('nCells', 'nCellsSolve', 'nEdges', 'nEdgesSolve',
                                          'nVertices', 'nVerticesSolve')},
        configs={'config_num_halos': 2}, indices={})
    prepare_native_layout(layout)
    assert layout.halo_ends == {name: (4, 4) for name in ('nCells', 'nEdges', 'nVertices')}
    # An actual distributed provider's offsets are retained, never replaced.
    layout.dimensions['nCellsSolve'] = 2
    layout.halo_ends['nCells'] = (3, 4)
    prepare_native_layout(layout)
    assert layout.halo_ends['nCells'] == (3, 4)
    del layout.halo_ends['nCells']
    with pytest.raises(ValueError, match='distributed halo ends missing'):
        prepare_native_layout(layout)


def test_selected_rrfs_six_field_layer_two_native_registration_is_live():
    import xml.etree.ElementTree as ET
    from hexcore.mpasphys.native_backend import NativePhysics
    from hexcore.mpasphys.rrfs.driver_rrfs import prepare_native_layout
    library = os.environ.get('RRFS_NATIVE_LIBRARY')
    if not library:
        pytest.skip('requires the published strict fork native library')
    names = ('sigma_deep', 'sigma_deep_far', 'sub3d_rucuten', 'sub3d_rvcuten',
             'sub3d_rthcuten', 'sub3d_rqvcuten')
    root = ET.fromstring('<registry><var_struct name="diag_physics">' +
        ''.join('<var name="' + name + '" dimensions="nCells Time"/>' for name in names) +
        '</var_struct></registry>')
    dims = {name: 4 for name in ('nCells', 'nCellsSolve', 'nEdges', 'nEdgesSolve',
                                'nVertices', 'nVerticesSolve')}
    rows = [dict(pool='diag_physics', name=name, pool_name=name,
                 dimensions=['nCells'], components=[]) for name in names]
    layout = SimpleNamespace(root=root, dimensions=dims, fields=rows,
                             configs={'config_num_halos': 2}, indices={}, enabled=lambda row: True)
    prepare_native_layout(layout)
    ids = np.arange(1, 6, dtype=np.int32)
    words = np.array([0x80000000, 1, 0x3f800000, 0xc0000000, 0x42fe0000], dtype=np.uint32)
    with NativePhysics(library, test_only=True, pin_set='rrfs') as native:
        native.pool('mesh'); native.pool('diag_physics')
        for name in ('indexToCellID', 'indexToEdgeID', 'indexToVertexID'):
            native.put_field('mesh', name, ids)
        for name in names:
            native.put_field('diag_physics', name, words.view(np.float32))
        hydration = object.__new__(NativePoolHydration)
        hydration.native, hydration.layout = native, layout
        hydration.register()
        native.exchange_group('physics:sub3d_cuten', names, level=1, halo_layers=(2,))
        native.exchange_halo_group('physics:sub3d_cuten')
        for name in names:
            np.testing.assert_array_equal(native.get_field('diag_physics', name).view(np.uint32), words)
