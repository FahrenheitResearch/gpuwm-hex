import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('audit',Path(__file__).parents[1]/'src/hexcore/s2s_surface_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
cli_spec=importlib.util.spec_from_file_location('audit_cli',Path(__file__).parents[1]/'tools/audit_s2s_surface.py')
cli=importlib.util.module_from_spec(cli_spec);cli_spec.loader.exec_module(cli)


def fixture():
    return dict(area=np.full(4,1.275e14),latitude=np.deg2rad([80.,-80.,30.,-30.]),
        ocean=np.array([True,True,True,False]),sst=np.array([271.,272.,295.,290.],dtype='f4'),
        ice=np.array([1.,.5,0.,0.],dtype='f4'),source_sst=np.array([271.,272.,295.,290.],dtype='f4'),
        source_ice=np.array([1.,.5,.01,0.],dtype='f4'),t2=np.array([250.,260.,290.,280.],dtype='f4'))


def test_fractional_area_poles_and_raw_cutoff_are_distinct():
    names,values=audit.metrics(np,**fixture());r=dict(zip(names,values))
    assert r['applied_global_ice_area_km2']==191250000.
    assert r['source_raw_global_ice_area_km2']>r['source_cutoff_global_ice_area_km2']
    assert r['sst_differing_words']==r['ice_differing_words_after_cutoff']==0
    assert r['arctic_model_t2_over_ice_K']==250.
    assert r['antarctic_model_t2_over_ice_K']==260.
    assert r['arctic_applied_xice']==1.
    assert r['antarctic_applied_xice']==.5


def test_wrong_applied_words_and_conversion_are_detected():
    v=fixture();v['ice'][0]=0.;v['sst'][1]+=1
    names,values=audit.metrics(np,**v,xland=np.full(4,2),ivgtyp=np.full(4,17),isltyp=np.full(4,14),albedo=np.full(4,1.1))
    r=dict(zip(names,values))
    assert r['sst_differing_words']==r['ice_differing_words_after_cutoff']==1
    assert r['water_ice_classification_mismatches']==1
    assert r['ice_albedo_outside_0_1']==1
    assert np.isnan(r['arctic_model_t2_over_ice_K'])


def test_unit_sphere_area_refuses():
    v=fixture();v['area']=np.ones(4)
    with pytest.raises(ValueError,match='dimensional global'):
        audit.metrics(np,**v)


class PromotedVariable:
    dimensions = ('Time', 'nCells')

    def __init__(self, values):
        self.dtype = values.dtype
        self.values = values.astype('f8')[None]
        self.raw = False

    def set_auto_maskandscale(self, on):
        self.raw = not on

    def __getitem__(self, key):
        return self.values[key]


def test_word_audit_restores_stored_float32_after_promoted_transport():
    from types import SimpleNamespace
    words = np.array([0, 0x80000000, 1, 0x3f800001, 0x7f7fffff], dtype='u4')
    variable = PromotedVariable(words.view('f4'))
    actual = cli.native_array(SimpleNamespace(variables={'sst':variable}), 'sst', word_audit=True)
    assert actual.dtype == np.float32 and variable.raw
    np.testing.assert_array_equal(actual.view('u4'), words)


def test_word_audit_refuses_genuinely_float64_history():
    from types import SimpleNamespace
    variable = PromotedVariable(np.array([271.], dtype='f8'))
    with pytest.raises(ValueError, match='stored float32 sst'):
        cli.native_array(SimpleNamespace(variables={'sst':variable}), 'sst', word_audit=True)


def test_audit_restores_integer_identity_after_promoted_transport():
    from types import SimpleNamespace
    variable = PromotedVariable(np.array([2147483647,101,11], dtype='i4'))
    actual = cli.native_array(SimpleNamespace(variables={'indexToCellID':variable}), 'indexToCellID')
    assert actual.dtype == np.int32
    np.testing.assert_array_equal(actual, [2147483647,101,11])
