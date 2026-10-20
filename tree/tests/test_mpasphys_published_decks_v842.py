"""Published-deck control and schedule witnesses, never forecast gate evidence."""
from dataclasses import asdict
import ctypes
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from hexcore.config import parse_fortran_namelist
from hexcore.mpasphys.driver_v842 import PhysicsManager, interval_seconds, resolve_namelist
from test_mpasphys_driver_v842 import native_control


def test_published_deck_control_and_schedules():
    published = os.environ.get('MPAS_PHYSICS_DECKS')
    authority = os.environ.get('MPAS_PHYSICS_REFERENCE_DIR')
    if not published or not authority:
        pytest.skip('set MPAS_PHYSICS_DECKS and MPAS_PHYSICS_REFERENCE_DIR')
    root, native = Path(published), Path(authority)
    publication = json.loads((root/'PUBLICATION.json').read_text())
    lib = ctypes.CDLL(str(native/'libclock.so'))
    lib.schedule.argtypes = [ctypes.c_int64,ctypes.c_int64,ctypes.c_int,
                            ctypes.c_void_p,ctypes.c_int64,ctypes.c_int,
                            ctypes.c_void_p,ctypes.c_void_p]
    records = []
    for index_path in sorted(root.rglob('INDEX.json')):
        index = json.loads(index_path.read_text())
        runtime = index['runtime']
        for row in index['decks']:
            folder = index_path.parent/row['deck']
            text = (folder/'namelist.atmosphere').read_text()
            for filename in ('namelist.atmosphere','streams.atmosphere'):
                rel = (folder/filename).relative_to(root).as_posix()
                assert hashlib.sha256((folder/filename).read_bytes()).hexdigest() == publication['files'][rel]
            config = resolve_namelist(text)
            supplied = parse_fortran_namelist(text)['physics']
            proc, slots = native_control(native,config.suite,supplied)
            assert proc.returncode == 0, proc.stdout+proc.stderr
            assert [config.selectors[s] for s in slots] == proc.stdout.splitlines()[:9]
            # The f0 index records suite values before MPAS control corrections;
            # pristine control is the authority for corrected selectors.
            streams = {e.attrib['name']:e.attrib.get('output_interval','none')
                       for e in ET.parse(folder/'streams.atmosphere').getroot()
                       if e.tag in ('stream','immutable_stream')}
            dt = runtime['dt']; duration = interval_seconds(runtime['duration'])
            manager = PhysicsManager(config,dt,streams=streams)
            times = np.arange(0,int(duration),dt,dtype=np.int64)
            flags = [manager.advance(int(t)) for t in times]
            schedule = {}
            for name,key in [('radiation_lw','config_radtlw_interval'),
                             ('radiation_sw','config_radtsw_interval'),
                             ('convection','config_conv_interval'),
                             ('rain_bucket','config_bucket_update'),
                             ('radiation_bucket','config_bucket_update')]:
                period = interval_seconds(config.options[key]); bucket = 'bucket' in name
                active = (config.active('convection') if name == 'rain_bucket' else
                          config.active('radiation_lw') or config.active('radiation_sw')
                          if name == 'radiation_bucket' else config.active(name))
                got = [getattr(f,name) for f in flags]
                expected = np.zeros(len(times),dtype=np.int32)
                if active and period is None and not bucket:
                    expected.fill(1)
                elif active and period is not None:
                    assert period.denominator == 1
                    nexts = np.zeros(len(times),dtype=np.int64)
                    lib.schedule(int(period),int(period) if bucket else 0,len(times),times.ctypes.data,
                                 dt if bucket else 0,1,expected.ctypes.data,nexts.ctypes.data)
                assert got == expected.astype(bool).tolist(), (folder,name)
                schedule[name] = [i+1 for i,x in enumerate(got) if x]
            schedule['pbl'] = list(range(1,len(times)+1)) if config.active('pbl') and config.driver_physics else []
            if 'pbl' in manager.alarms:
                assert manager.alarms['pbl'].next_ring == Fraction(0)
            records.append({'deck':folder.relative_to(root).as_posix(),'dt':dt,
                            'steps':len(times),'selectors':dict(config.selectors),
                            'schedule':schedule,'control_mismatches':0,'schedule_mismatches':0})
    assert records
    destination = os.environ.get('MPAS_PHYSICS_DECK_RECEIPT')
    if destination:
        Path(destination).write_text(json.dumps({'schema':'mpas-port.deck-control-schedule/v1',
            'scope':'control + standalone recurring clocks; no forecast or manager-stream I/O proof',
            'publication_sha256':hashlib.sha256((root/'PUBLICATION.json').read_bytes()).hexdigest(),
            'decks':records},indent=2)+'\n')
