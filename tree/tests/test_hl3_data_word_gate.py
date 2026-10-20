"""Gate regressions: only volatile global file_id is exempt."""
from copy import deepcopy
from pathlib import Path
import pytest
from hexcore import stream_headers_v842 as headers
from hexcore.verify_mpas import compare_sample_headers, stream_header_failures

@pytest.fixture
def pair(tmp_path, monkeypatch):
    native, candidate = tmp_path/'native.nc', tmp_path/'hex.nc'
    native.write_bytes(b'native container'); candidate.write_bytes(b'other container')
    header = dict(format=5, dimension_order=['Time','nCells'], dimensions={'Time':1,'nCells':2},
                  unlimited=['Time'], variable_order=['theta'], global_attribute_order=['file_id','history'],
                  global_attribute_words={'file_id':{'dtype':'i4','values':[1]},
                                          'history':{'dtype':'S1','values':'run'}},
                  variables={'theta':dict(dtype='f4', dimensions=['Time','nCells'], shape=[1,2],
                    attribute_order=['units','file_id'], attribute_words={
                      'units':{'dtype':'S1','values':'K'}, 'file_id':{'dtype':'i4','values':[7]}})})
    actual = deepcopy(header); actual['global_attribute_words']['file_id']['values'] = [999]
    monkeypatch.setattr(headers, 'inspect_stream', lambda path, **kw: header if path == native else actual)
    return native, candidate, header, actual, tmp_path

def grade(pair):
    native,candidate,_,_,scratch=pair
    return compare_sample_headers([dict(strict=str(native),hex=str(candidate),stream='output',steps=[1])],scratch=scratch)

def test_file_id_only_and_container_hash_difference_pass(pair):
    native,candidate,_,_,scratch=pair
    assert not headers.compare_headers(native,candidate,scratch=scratch)['passed']
    result=grade(pair)
    assert result[0]['passed'] and not result[0]['byte_identical']
    assert result[0]['ignored_global_attributes'] == ['file_id']
    assert stream_header_failures(result) == []

@pytest.mark.parametrize('change', ['attribute','attribute_type','attribute_order','variable_file_id','dtype','shape','dimension','missing'])
def test_other_header_changes_fail(pair,change):
    _,_,_,actual,_=pair
    field=actual['variables']['theta']
    if change == 'attribute': actual['global_attribute_words']['history']['values']='changed'
    elif change == 'attribute_type': actual['global_attribute_words']['history']['dtype']='i4'
    elif change == 'attribute_order': field['attribute_order'].reverse()
    elif change == 'variable_file_id': field['attribute_words']['file_id']['values']=[99]
    elif change == 'dtype': field['dtype']='f8'
    elif change == 'shape': field['shape']=[2,1]
    elif change == 'dimension': actual['dimensions']['nCells']=3
    elif change == 'missing': actual['variable_order']=[]; actual['variables']={}
    result=grade(pair)
    assert not result[0]['passed']
    assert stream_header_failures(result)
