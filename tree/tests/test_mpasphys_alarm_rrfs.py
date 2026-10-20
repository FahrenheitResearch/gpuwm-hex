"""Pristine fork clock/reset witnesses, deliberately separate from forecast proof."""
import ctypes
from fractions import Fraction
import os
from pathlib import Path
import numpy as np
import pytest
from hexcore.mpasphys.driver_v842 import Alarm


@pytest.fixture(scope='module')
def oracle():
    path=os.environ.get('RRFS_ALARM_REFERENCE_DIR')
    if path is None:
        pytest.skip('set RRFS_ALARM_REFERENCE_DIR to the owned clock/reset oracle')
    return Path(path)


@pytest.mark.parametrize('dt,period',[(360,1800),(700,1800),(1000,360),(360,360),(90,135),(60,900),(150,3600)])
@pytest.mark.parametrize('delayed',[False,True])
@pytest.mark.parametrize('lookahead',[False,True])
def test_alarm_matches_fork_clock(oracle,dt,period,delayed,lookahead):
    times=np.arange(0,10000,dt,dtype=np.int64)
    times[3:]+=period*4
    flags=np.zeros(len(times),dtype=np.int32);nexts=np.zeros(len(times),dtype=np.int64)
    lib=ctypes.CDLL(str(oracle/'libclock.so'))
    lib.schedule.argtypes=[ctypes.c_int64,ctypes.c_int64,ctypes.c_int,ctypes.c_void_p,
        ctypes.c_int64,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    start=period if delayed else 0;ahead=dt if lookahead else 0
    lib.schedule(period,start,len(times),times.ctypes.data,ahead,1,flags.ctypes.data,nexts.ctypes.data)
    alarm=Alarm(Fraction(period),Fraction(start));got_flags=[];got_next=[]
    for now in times:
        got_flags.append(int(alarm.ringing(Fraction(int(now)),Fraction(ahead))))
        alarm.reset(Fraction(int(now)),Fraction(ahead));got_next.append(int(alarm.next_ring))
    assert got_flags==flags.tolist()
    assert got_next==nexts.tolist()


@pytest.mark.parametrize('flags',range(8))
def test_complete_native_reset_zeroes_only_associated_fields(oracle,flags):
    # Include halo/dummy extent and negative-zero/subnormal words: native resets
    # the whole associated array to positive zero, not merely solved columns.
    words=np.array([[0x80000000,1,0x7f800000],[0x3f800000,0xbf800000,0x7fc00000],
                    [0x00000002,0x80000001,0x41200000]],dtype=np.uint32,order='F')
    fields=words.view(np.float32).copy(order='F');expected=words.copy(order='F')
    lib=ctypes.CDLL(str(oracle/'libreset.so'))
    lib.reset_fields.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    lib.reset_fields(len(fields),flags,fields.ctypes.data)
    for i in range(3):
        if flags&(1<<i):expected[:,i]=0
    assert np.array_equal(fields.view(np.uint32),expected)
