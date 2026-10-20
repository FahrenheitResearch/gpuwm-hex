"""LES boundary admission and optional real CUDA/native replay tests."""
import os
from pathlib import Path
import pytest
from hexcore.mixing_v842_les import (
    les_model_from_string, les_surface_from_string, _extent, ROUTINES,
)


@pytest.mark.parametrize('name,expected', [
    ('none',0),('3d_smagorinsky',1),('prognostic_1.5_order',2),
    ('3d_smagorinsky   ',1),(' 3d_smagorinsky',-1),('3D_smagorinsky',-1),
    ('none\t',-1),('',-1),('unknown',-1),
])
def test_native_model_selector(name, expected):
    assert les_model_from_string(name)==expected


@pytest.mark.parametrize('name,expected', [
    ('none',0),('specified',1),('varying',2),('varying  ',2),
    (' specified',-1),('Specified',-1),('unknown',-1),
])
def test_native_surface_selector(name, expected):
    assert les_surface_from_string(name)==expected


def test_selectors_match_compiled_native_fortran():
    library=os.environ.get('MPAS_LES_NATIVE_LIBRARY')
    if not library:
        pytest.skip('set MPAS_LES_NATIVE_LIBRARY to the strict standalone build')
    import ctypes
    import hashlib
    import json
    native=ctypes.CDLL(library).replay_les_selectors
    native.restype=None
    native.argtypes=[ctypes.c_char_p]+[ctypes.POINTER(ctypes.c_int)]*3
    names=['none','3d_smagorinsky','prognostic_1.5_order','specified','varying',
           '3d_smagorinsky   ',' specified','Specified','3D_smagorinsky',
           'none\t','','unknown','varying  ']
    rows=[]
    for name in names:
        value=name.encode('ascii');size=ctypes.c_int(len(value))
        model=ctypes.c_int();surface=ctypes.c_int()
        native(value,ctypes.byref(size),ctypes.byref(model),ctypes.byref(surface))
        assert (model.value,surface.value)==(les_model_from_string(name),les_surface_from_string(name))
        rows.append({'input':name,'model':model.value,'surface':surface.value})
    receipt=os.environ.get('MPAS_LES_SELECTOR_RECEIPT')
    if receipt:
        Path(receipt).write_text(json.dumps({'kind':'strict native selector replay','process_count':1,'MPI_used':False,
            'native_library_sha256':hashlib.sha256(Path(library).read_bytes()).hexdigest(),
            'different_words':0,'cases':rows},indent=2))


def test_dimension_grammar():
    assert _extent('nvertlevels+1', {'nvertlevels':55})==56
    with pytest.raises(ValueError):
        _extent('__import__("os").system("false")', {})


def test_every_native_les_routine_has_complete_boundary():
    assert set(ROUTINES)=={'les_models','calculate_n2','u_dissipation_3d',
                           'w_dissipation_3d','scalar_dissipation_3d_les'}
    for spec in ROUTINES.values():
        assert set(spec['args'])==set(spec['fields'])
        assert spec['lines'][0]<spec['lines'][1]
    assert 'clock' not in ROUTINES['scalar_dissipation_3d_les']['args']
    assert 'dt' in ROUTINES['scalar_dissipation_3d_les']['args']


def test_later_rk_stages_do_not_touch_retained_frame():
    from hexcore.mixing_v842_les import LesDissipationV842
    # Stage 2/3 must not even resolve or import CUDA: retained tendencies are
    # added by the enclosing MPAS driver, rather than recomputed by this slice.
    sentinel=object()
    launcher=LesDissipationV842()
    for stage in (2,3):
        assert launcher.apply_rk_mixing(stage,tend_u_euler=sentinel)=={'tend_u_euler':sentinel}


@pytest.mark.parametrize('model', ['3d_smagorinsky','prognostic_1.5_order'])
@pytest.mark.parametrize('surface', ['none','specified','varying'])
def test_config_admits_les_only_with_execution_route(model, surface):
    from hexcore.config_v841 import V841DryDycoreConfig
    from hexcore.mixing_v842_les import LesDissipationV842
    from hexcore.errors import ConfigurationRefusal
    config=V841DryDycoreConfig(config_les_model=model,config_les_surface=surface,
                              config_mix_scalars=True,config_surface_heat_flux=.03,
                              config_surface_moisture_flux=.0002,
                              config_surface_drag_coefficient=.0013)
    with pytest.raises(ConfigurationRefusal,match='forecast driver has not connected'):
        config.validate()
    config.validate(les_dissipation=LesDissipationV842())


def test_config_rejects_unknown_native_selector_with_launcher():
    from hexcore.config_v841 import V841DryDycoreConfig
    from hexcore.mixing_v842_les import LesDissipationV842
    from hexcore.errors import ConfigurationRefusal
    with pytest.raises(ConfigurationRefusal,match='unknown native LES model'):
        V841DryDycoreConfig(config_les_model='invalid').validate(les_dissipation=LesDissipationV842())


@pytest.mark.parametrize('config_name', [
    'V841MpasColumnPhysicsConfig', 'V841UpperDampingDryConfig',
    'V841UpperDampingColumnConfig',
    'V841MpasColumnPhysicsGwdoConfig',
    'V841MpasColumnPhysicsSmagorinskyGwdoConfig',
    'V841UpperDampingSmagorinskyConfig',
])
def test_derived_config_preserves_les_execution_route(config_name):
    from hexcore import config_v841
    from hexcore.mixing_v842_les import LesDissipationV842
    from hexcore.errors import ConfigurationRefusal
    config=getattr(config_v841,config_name)(config_les_model='prognostic_1.5_order',
                                          config_les_surface='specified')
    with pytest.raises(ConfigurationRefusal,match='forecast driver has not connected'):
        config.validate()
    config.validate(les_dissipation=LesDissipationV842())


def test_standalone_oracle_sources_are_pristine():
    import hashlib
    root=Path(__file__).parent/'native_les_v842'
    assert hashlib.sha256((root/'mpas_atm_dissipation_models.F').read_bytes()).hexdigest()=='cc6e6b3b211270b7affc81fa44720e9fb6c8e39f9a42ab32c57e7c4d9bb6bcbf'
    assert hashlib.sha256((root/'mpas_constants.F').read_bytes()).hexdigest()=='fa77a86b6d3e9041677ef1cdaf76fbcc172a50607874c80ace5085829ae7be31'


@pytest.mark.parametrize('surface', ['none','specified','varying'])
def test_inactive_native_les_surface_options_need_no_launcher(surface):
    from hexcore.config_v841 import V841DryDycoreConfig
    V841DryDycoreConfig(config_les_surface=surface,
        config_surface_heat_flux=.03,config_surface_moisture_flux=.0002,
        config_surface_drag_coefficient=.0013).validate()


def test_non_les_scalar_filter_still_needs_connected_route():
    from hexcore.config_v841 import V841DryDycoreConfig
    from hexcore.errors import ConfigurationRefusal
    with pytest.raises(ConfigurationRefusal,match='scalar filtering'):
        V841DryDycoreConfig(config_mix_scalars=True).validate()


@pytest.mark.parametrize('surface', [1,2])
def test_native_none_ignores_les_surface_controls(surface):
    library=os.environ.get('MPAS_LES_NATIVE_LIBRARY')
    if not library:
        pytest.skip('set MPAS_LES_NATIVE_LIBRARY to the strict standalone build')
    import ctypes
    import numpy as np
    from replay_les_boundary_v842 import cases
    from hexcore.mixing_v842_les import les_boundary_arguments
    _,prototype=next(cases());spec=les_boundary_arguments()
    fn=ctypes.CDLL(library).replay_les_boundary
    fn.restype=None;fn.argtypes=[ctypes.c_void_p]*len(spec)
    frames=[]
    for option in [0,surface]:
        frame={n:v.copy(order='F') for n,v in prototype.items()}
        frame['les_model_opt']=np.array(0,dtype='i4')
        frame['les_surface_opt']=np.array(option,dtype='i4')
        for name,value in [('config_surface_heat_flux',.03),
                           ('config_surface_moisture_flux',.0002),
                           ('config_surface_drag_coefficient',.0013)]:
            frame[name]=np.array(value if option else 0,dtype='f4')
        fn(*(ctypes.c_void_p(v.ctypes.data) for v in frame.values()))
        frames.append(frame)
    for name,field in spec.items():
        if field['intent']!='in':
            assert np.array_equal(frames[0][name].view('u4'),frames[1][name].view('u4')),name


def test_real_native_tap_records_the_entire_les_boundary(tmp_path):
    """The real-data proof must retain all five complete native call frames."""
    from native_les_v842.prepare_tap import instrument
    from hexcore.mixing_v842_les import ROUTINES
    from pathlib import Path
    source = Path(__file__).parent / 'native_les_v842/mpas_atm_dissipation_models.F'
    output = tmp_path / source.name
    inventory = instrument(source, output)
    assert {row['routine'] for row in inventory['routines']} == set(ROUTINES)
    for row in inventory['routines']:
        assert set(row['recorded']) == set(ROUTINES[row['routine']]['args'])
        for name in row['recorded']:
            for phase in ('pre', 'post'):
                assert output.read_text().count(
                    f"call oracle_write('{row['routine']}', '{phase}', '{name}',") == 1
    assert len(inventory['routines']) == 5


@pytest.mark.parametrize('model,surface', [(0, 2), (1, 0), (1, 1), (2, 1)])
def test_unused_native_surface_arrays_need_no_cell_sentinel(model, surface):
    from hexcore.mixing_v842_les import _surface_extent
    values = dict(les_model_opt=model, les_surface_opt=surface, ncells=2562, cellend=2562)
    assert all(_surface_extent(name, values) == 0 for name in ('hfx', 'qfx', 'ustm'))


def test_varying_surface_fluxes_read_owned_cells_without_sentinel():
    from hexcore.mixing_v842_les import _surface_extent
    values = dict(les_model_opt=2, les_surface_opt=2, ncells=2562, cellend=2500)
    assert [_surface_extent(name, values) for name in ('hfx', 'qfx', 'ustm')] == [2500, 2500, 2562]
