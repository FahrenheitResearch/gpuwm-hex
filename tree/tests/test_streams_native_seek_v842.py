"""Whole native seek replay; MPAS mpas_io_streams.F:153-339.

This compares metadata selection against the actual strict Fortran routine,
then exercises Rust record decoding. It does not replace whole model proof.
"""
from dataclasses import replace
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.output import HistoryField
from hexcore.restart_v842 import (_seek_record,read_stream,read_restart,seek_stream,
                                 write_stream,write_restart)
from hexcore.streams_v842 import parse_streams,registry_stream

MODES=('exact_time','nearest','latest_before','earliest_after',
       'latest_strictly_before','earliest_strictly_after')


def cases():
    """Calendar, ties, native cutoff/optimization; MPAS mpas_io_streams.F:202-328."""
    base=datetime(2000,2,28,23)
    def stamp(delta):
        return (base+timedelta(seconds=delta)).strftime('%Y-%m-%d_%H:%M:%S')
    small=[stamp(0),stamp(3600),stamp(7200)]
    regular=[stamp(i*3600) for i in range(24)]
    irregular=[stamp(i*i*1800) for i in range(24)]
    fixtures=[(small,[stamp(x) for x in (-1,0,1800,3600,5400,7200,7201)]),
        (list(reversed(small)),[stamp(1800),stamp(3600),stamp(5400)]),
        ([stamp(0)]*3,[stamp(0),stamp(1)]),
        (regular,[stamp(x) for x in (0,3600,82800,84600,86400)]),
        (irregular,[stamp(x) for x in (0,1800,7200,8100,952200)]),
        ([stamp(0)]*24,[stamp(0),stamp(1)]),
        ([stamp(0)],[stamp(10000*86400-1),stamp(10000*86400),stamp(10000*86400+1)]),
        ([],[stamp(0)]),
        (['2000-02-29_23:59:59.12345678','2000-03-01_00:00:00.00000001'],
         ['2000-02-29_23:59:59.12345678','2000-02-29_23:59:59.12345679',
          '2000-03-01_00:00:00.00000001']),
        (['1900-02-28_23:59:59','1900-03-01_00:00:01'],['1900-03-01_00:00:00'])]
    return [(stamps,target,mode) for stamps,targets in fixtures
            for target in targets for mode in MODES]


@pytest.mark.parametrize('stamps,target,mode',cases())
def test_every_seek_mode_against_strict_native(stamps,target,mode):
    """Replay full pinned MPAS_seekStream; mpas_io_streams.F:153-339."""
    oracle=os.environ.get('MPAS_NATIVE_STREAM_SEEK_ORACLE')
    if not oracle:
        pytest.skip('strict native stream seek oracle not supplied')
    request='\n'.join([f'{len(stamps)} {100+MODES.index(mode)}',target,*stamps])+'\n'
    native=subprocess.run([oracle],input=request,text=True,capture_output=True,
                          check=True,timeout=30)
    row=native.stdout.strip().splitlines()[-1].split('|')
    index,error,count=map(int,row[:3])
    assert count==len(stamps)
    try:
        selected=_seek_record(stamps,target,mode)
    except ValueError:
        selected=None
    assert selected==(index-1 if index else None),(stamps,target,mode,native.stdout)
    assert error==(0 if index else -5)
    if index:
        assert row[3]==stamps[index-1][:32].strip()
    receipt=os.environ.get('MPAS_NATIVE_STREAM_SEEK_RECEIPT')
    if receipt:
        with Path(receipt).open('a') as output:
            output.write(json.dumps(dict(schema='o1-native-stream-seek-column-v1',
                process_count=1,mode=mode,record_count=len(stamps),target=target,
                native_index=index,port_index=selected,differing_selection_words=0))+'\n')


@pytest.mark.parametrize('mode,offset,index',[
    ('exact_time',3600,1),('nearest',1800,0),('latest_before',3600,1),
    ('earliest_after',3600,1),('latest_strictly_before',3600,0),
    ('earliest_strictly_after',3600,2)])
def test_real_rust_decodes_the_selected_record(tmp_path,mode,offset,index):
    """Native selected record handoff; mpas_io_streams.F:2430-3251."""
    if not os.environ.get('GPUWM_HEX_RW_MPAS_STREAM'):
        pytest.skip('real Rust native stream artifact not supplied')
    deck=tmp_path/'streams.atmosphere'
    deck.write_text('''<streams><stream name="fixture" type="input;output"
      filename_template="records.nc" input_interval="initial_only" output_interval="1:00:00"
      clobber_mode="append"><var name="xtime"/><var name="theta"/></stream></streams>''')
    stream=parse_streams(deck)['fixture']
    mesh=SimpleNamespace(dimensions={'nCells':2,'nVertLevels':2},arrays={},attrs={},converted_connectivity=())
    start=datetime(2000,2,28,23)
    path=tmp_path/'records.nc'
    for record in range(3):
        words=np.full((2,2),record+1,dtype=np.float32)
        words.view(np.uint32)[0,0]=record+1  # Distinct subnormal words survive selection.
        write_stream(path,stream,mesh,{'theta':HistoryField(words,('nVertLevels','nCells'))},
            start+timedelta(hours=record),initial_time=start,record_index=record,
            scratch=tmp_path/'write')
    target=start+timedelta(seconds=offset)
    selected,actual=seek_stream(path,valid_time=target,whence=mode,scratch=tmp_path/'read')
    assert selected==index
    assert actual==(start+timedelta(hours=index)).strftime('%Y-%m-%d_%H:%M:%S')
    with read_stream(path,select=('theta',),valid_time=target,whence=mode,
                     scratch=tmp_path/'read') as result:
        words=result.variables['theta'].values
        assert words.shape==(1,2,2)
        assert words.view(np.uint32)[0,0,0]==index+1
        assert words[0,1,1]==index+1


def test_unknown_mode_and_ambiguous_api_are_rejected(tmp_path):
    """Reject inconsistent caller metadata; MPAS mpas_io_streams.F:317-326."""
    with pytest.raises(ValueError,match='unrecognized'):
        _seek_record(['2000-01-01_00:00:00'],'2000-01-01_00:00:00','unknown')
    with pytest.raises(ValueError,match='mutually exclusive'):
        with read_stream(tmp_path/'not-opened.nc',scratch=tmp_path,record=0,
                         valid_time='2000-01-01_00:00:00'):
            pass


def test_real_published_restart_metadata_selection(tmp_path):
    """Actual published native restart clock; mpas_atm_core.F:129-150."""
    reference=os.environ.get('MPAS_NATIVE_POOL_REAL_PUBLICATION')
    if not reference or not os.environ.get('GPUWM_HEX_RW_MPAS_STREAM'):
        pytest.skip('authenticated real native reference/Rust artifact not supplied')
    paths=sorted(Path(reference).glob('restart.*.nc'))
    assert paths
    selected=paths[len(paths)//2]
    with read_stream(selected,select=('xtime',),scratch=tmp_path) as source:
        stamps=source.variables['xtime'].text
    assert len(stamps)==1
    for mode in MODES[:4]:
        assert seek_stream(selected,valid_time=stamps[0],whence=mode,scratch=tmp_path)==(0,stamps[0])


@pytest.mark.parametrize('records,expected',[(2,1),(21,0)])
def test_restart_duplicate_timestamps_follow_native_selection(tmp_path,records,expected):
    """Retire the duplicate timestamp refusal; mpas_io_streams.F:202-328."""
    if not os.environ.get('GPUWM_HEX_RW_MPAS_STREAM'):
        pytest.skip('real Rust native stream artifact not supplied')
    stream=replace(registry_stream('restart'),
        members=(('var','u',''),('var','xtime',''),('var','initial_time','')),
        options={**registry_stream('restart').options,'clobber_mode':'append'})
    mesh=SimpleNamespace(dimensions={'nEdges':2,'nVertLevels':2},arrays={},attrs={},converted_connectivity=())
    path=tmp_path/'restart.nc'
    stamp='2000-02-29_00:00:00'
    for record in range(records):
        words=np.full((2,2),record+1,dtype=np.float32)
        write_restart(path,mesh,{'u':HistoryField(words,('nVertLevels','nEdges'))},stamp,
            initial_time=stamp,stream=stream,record_index=record,scratch=tmp_path/'write')
    with read_restart(path,valid_time=stamp,stream=stream,scratch=tmp_path/'read') as source:
        assert np.all(source.variables['u'].values==expected+1)
