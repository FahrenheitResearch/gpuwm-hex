"""Optional real native forecast artifacts, never substituted with mock fields.

MPAS_NATIVE_STREAM_FORECAST_DIR holds straight/, half/, resumed-native/ and
resumed-bridge/ runs. Each contains the native output/restart files named by
its stream deck. A supplementary idealized native check is useful evidence;
it does not replace the campaign's hex/native physics and G0-G3 decks.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hexcore.output import HistoryField
from hexcore.restart_v842 import read_stream,read_restart,write_stream
from hexcore.streams_v842 import registry_stream


def _root():
    value=os.environ.get('MPAS_NATIVE_STREAM_FORECAST_DIR')
    if not value:pytest.skip('native forecast artifacts not supplied')
    return Path(value)


def _final(root,kind):
    return sorted(root.glob(kind+'.*.nc'))[-1]


def _equal(reference,actual):
    assert reference.dimensions==actual.dimensions
    assert set(reference.variables)==set(actual.variables)
    for name,a in reference.variables.items():
        b=actual.variables[name]
        assert a.dimensions==b.dimensions and a.attrs==b.attrs,name
        assert a.values.dtype==b.values.dtype and a.values.shape==b.values.shape,name
        assert a.values.tobytes()==b.values.tobytes(),name


@pytest.mark.parametrize('kind',['history','restart'])
@pytest.mark.parametrize('candidate',['resumed-native','resumed-bridge'])
def test_actual_native_24h_equals_restarted_forecasts(tmp_path,kind,candidate):
    root=_root()
    with read_stream(_final(root/'straight',kind),scratch=tmp_path) as reference:
        with read_stream(_final(root/candidate,kind),scratch=tmp_path) as actual:
            _equal(reference,actual)


@pytest.mark.parametrize('kind,stream_name',[('history','output'),('restart','restart')])
def test_actual_native_file_reencode_all_fields_and_attributes(tmp_path,kind,stream_name):
    root=_root()
    with read_stream(_final(root/'half',kind),scratch=tmp_path) as native:
        mesh=SimpleNamespace(arrays={},dimensions=native.dimensions,attrs=native.global_attrs)
        fields={n:HistoryField(v.values,v.dimensions,v.attrs) for n,v in native.variables.items()}
        stream=registry_stream(stream_name)
        packages={'no_invariant_stream'}
        assert set(native.variables)==set(stream.fields(packages))
        valid=native.variables['xtime'].text[0]
        initial=native.variables['initial_time'].text[0]
        path=tmp_path/'reencoded.nc'
        write_stream(path,stream,mesh,fields,valid,initial_time=initial,active_packages=packages)
        with read_stream(path,scratch=tmp_path) as reencoded:
            _equal(native,reencoded)
            assert native.global_attrs==reencoded.global_attrs


def test_actual_native_restart_authority_restore_reencode(tmp_path):
    root=_root();path=_final(root/'half','restart')
    packages={'no_invariant_stream'}
    with read_stream(path,scratch=tmp_path,select=('xtime','initial_time')) as stamps:
        valid=stamps.variables['xtime'].text[0];initial=stamps.variables['initial_time'].text[0]
    with read_restart(path,valid_time=valid,scratch=tmp_path,active_packages=packages,authority_layout=True) as authority:
        mesh=SimpleNamespace(arrays={},dimensions=authority.dimensions,attrs=authority.global_attrs)
        fields={n:HistoryField(v.values,v.dimensions,v.attrs) for n,v in authority.variables.items()}
        output=tmp_path/'authority-reencoded.nc'
        write_stream(output,'restart',mesh,fields,valid,initial_time=initial,active_packages=packages)
        with read_stream(path,scratch=tmp_path) as native, read_stream(output,scratch=tmp_path) as restored:
            _equal(native,restored)
